#include "../src/llama-ext.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-model.h"
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, pos + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool test_sparse_metadata_seq_id_bounds(llama_model * model) {
    if (!model->graph_supports_recurrent_sparse_snapshots()) {
        fprintf(stderr, "%s : skipping because model graph does not support recurrent sparse snapshots\n", __func__);
        return true;
    }

    constexpr llama_seq_id high_seq_id = 3;
    llama_memory_recurrent memory(*model, GGML_TYPE_F32, GGML_TYPE_F32, false, 4, 1, 3, nullptr);

    memory.rs_idx[0] = 2;
    memory.rs_plane_pos = { 10, 11, 12, 13 };
    memory.rs_plane_pos_sparse[0] = true;
    memory.cells[0].pos = 13;
    memory.cells[0].tail = 0;
    memory.cells[0].seq_id.insert(0);
    memory.used = 1;

    if (memory.seq_rm(high_seq_id, -1, -1)) {
        fprintf(stderr, "%s : high sequence removal was accepted\n", __func__);
        return false;
    }
    memory.seq_cp(0, high_seq_id, -1, -1);
    memory.seq_cp(high_seq_id, 0, -1, -1);
    memory.seq_keep(high_seq_id);
    memory.seq_add(high_seq_id, -1, -1, 1);
    memory.seq_div(high_seq_id, -1, -1, 2);

    if (memory.rs_idx[0] != 2 || memory.rs_plane_pos != std::vector<llama_pos>({ 10, 11, 12, 13 }) ||
        !memory.rs_plane_pos_sparse[0] || memory.cells[0].pos != 13 || memory.cells[0].tail != 0 ||
        !memory.cells[0].has_seq_id(0) || memory.used != 1) {
        fprintf(stderr, "%s : malformed sequence changed valid recurrent metadata\n", __func__);
        return false;
    }

    fprintf(stderr, "%s : malformed sequence was rejected outside n_seq_max\n", __func__);
    return true;
}

// Roll back multiple sequences, then replay them in a single batch whose
// per-seq token count exceeds n_ubatch: each seq's replay spans several
// ubatches while its rollback restore is still pending. Compared against a
// reference context that never advanced past the rollback point and decodes
// the identical replay batch.
static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab) {
    constexpr uint32_t  n_seqs     = 2;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return llama_init_from_model(model, cparams);
    };

    llama_context * ctx_roll = make_ctx_multi();
    llama_context * ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_roll);
        llama_free(ctx_ref);
    };

    if (llama_n_rs_seq(ctx_roll) < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        cleanup();
        return true;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (llama_pos pos = 0; pos < (llama_pos) p0; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        ok = ok && llama_decode(ctx_ref,  batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll, batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        cleanup();
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll, batch) == 0;
    ok = ok && llama_decode(ctx_ref, batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        cleanup();
        return false;
    }

    // identical ubatch shapes from bit-exact states: a correct implementation
    // matches bitwise, so eps only allows backend scheduling noise
    constexpr float eps = 1e-7f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll, i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            cleanup();
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float diff = std::fabs(l_roll[t] - l_ref[t]);
            if (diff > eps && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
        }
    }

    if (diff_max > eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, seq_first, pos_first);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g)\n", __func__, (double) diff_max);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref, batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll, batch_one) == 0;
        ok = ok && llama_decode(ctx_ref, batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            diff_tail = std::max(diff_tail, std::fabs(l_roll[t] - l_ref[t]));
        }
    }

    if (!ok || diff_tail > eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail);
        cleanup();
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g)\n", __func__, (double) diff_tail);
    cleanup();
    return true;
}

static bool test_sparse_selected_then_trailing(const common_params & params, llama_model * model, const int n_vocab) {
    constexpr uint32_t n_seqs = 2;

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seqs;
    cparams.n_rs_seq   = 1;
    cparams.n_ctx      = 128;
    cparams.n_batch    = 64;
    cparams.n_ubatch   = 32;
    cparams.kv_unified = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to init context\n", __func__);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx);
    };
    const auto fail = [&](const char * stage) {
        fprintf(stderr, "%s : failed at %s\n", __func__, stage);
        cleanup();
        return false;
    };

    if (!llama_recurrent_sparse_snapshots_supported(ctx)) {
        fprintf(stderr, "%s : skipping because sparse snapshots are unsupported\n", __func__);
        cleanup();
        return true;
    }

    {
        llama_context * ctx_split = llama_init_from_model(model, cparams);
        if (ctx_split == nullptr) {
            return fail("split validation context init");
        }

        llama_batch split_batch = llama_batch_init(6, 0, 1);
        common_batch_add(split_batch, 1 % n_vocab, 0, { 0 }, true);
        for (llama_pos pos = 0; pos < 5; ++pos) {
            common_batch_add(split_batch, (11*pos + 38) % n_vocab, pos, { 1 }, pos == 4);
        }

        const bool mode_ok = llama_recurrent_set_sparse_snapshot_mode(ctx_split, true, -1);
        const int split_ret = mode_ok ? llama_decode(ctx_split, split_batch) : 0;
        llama_batch_free(split_batch);
        llama_free(ctx_split);
        if (!mode_ok || split_ret == 0) {
            return fail("unequal sparse split rejection");
        }
        fprintf(stderr, "%s : unequal sparse sequence split rejected\n", __func__);
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((11*(uint32_t) pos + 37*seq + 1) % (uint32_t) n_vocab);
    };
    const auto decode_ranges = [&](llama_context * decode_ctx, const std::vector<std::pair<uint32_t, llama_pos>> & ranges, uint32_t count) {
        llama_batch batch = llama_batch_init((int32_t) ranges.size() * count, 0, 1);
        for (const auto & [seq, start] : ranges) {
            for (uint32_t i = 0; i < count; ++i) {
                const llama_pos pos = start + (llama_pos) i;
                common_batch_add(batch, tok(seq, pos), pos, { (llama_seq_id) seq }, i + 1 == count);
            }
        }
        const bool ok = llama_decode(decode_ctx, batch) == 0;
        llama_batch_free(batch);
        return ok;
    };

    {
        llama_context * ctx_normal = llama_init_from_model(model, cparams);
        llama_context * ctx_ref = llama_init_from_model(model, cparams);
        const auto transition_cleanup = [&]() {
            llama_free(ctx_normal);
            llama_free(ctx_ref);
        };
        if (ctx_normal == nullptr || ctx_ref == nullptr) {
            transition_cleanup();
            return fail("normal transition context init");
        }
        if (!llama_recurrent_set_sparse_snapshot_mode(ctx_normal, true, -1) ||
            !decode_ranges(ctx_normal, { { 0, 0 } }, 4) ||
            !llama_recurrent_set_sparse_snapshot_mode(ctx_normal, false, -1)) {
            transition_cleanup();
            return fail("sparse-to-normal setup");
        }

        common_prompt_checkpoint ckpt;
        ckpt.update_tgt(ctx_normal, 0, 0);
        ckpt.load_tgt(ctx_ref, 0, 0);

        if (!decode_ranges(ctx_normal, { { 0, 4 } }, 2) || !decode_ranges(ctx_ref, { { 0, 4 } }, 2)) {
            transition_cleanup();
            return fail("normal transition decode");
        }
        if (!llama_memory_seq_rm(llama_get_memory(ctx_normal), 0, 5, -1) ||
            !llama_memory_seq_rm(llama_get_memory(ctx_ref), 0, 5, -1) ||
            !decode_ranges(ctx_normal, { { 0, 5 } }, 1) || !decode_ranges(ctx_ref, { { 0, 5 } }, 1)) {
            transition_cleanup();
            return fail("normal transition rollback");
        }

        const float * logits_normal = llama_get_logits_ith(ctx_normal, 0);
        const float * logits_ref = llama_get_logits_ith(ctx_ref, 0);
        if (logits_normal == nullptr || logits_ref == nullptr) {
            transition_cleanup();
            return fail("normal transition logits");
        }
        constexpr float eps = 1e-5f;
        for (int token = 0; token < n_vocab; ++token) {
            if (std::fabs(logits_normal[token] - logits_ref[token]) > eps) {
                transition_cleanup();
                return fail("normal transition output comparison");
            }
        }
        transition_cleanup();
        fprintf(stderr, "%s : sparse-to-normal rollback matched reference output\n", __func__);
    }

    if (!llama_recurrent_set_sparse_snapshot_mode(ctx, true, -1)) {
        return fail("enable trailing mode");
    }
    if (!decode_ranges(ctx, { { 0, 0 }, { 1, 0 } }, 4) || !decode_ranges(ctx, { { 0, 4 }, { 1, 4 } }, 5)) {
        return fail("two-sequence trailing decode");
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, 4, -1)) {
        return fail("sequence 0 pre-verification restore");
    }
    if (!llama_recurrent_set_sparse_snapshot_mode(ctx, true, 2) || !decode_ranges(ctx, { { 0, 4 } }, 5)) {
        return fail("sequence 0 selected replay");
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, 7, -1)) {
        return fail("sequence 0 selected boundary");
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx), 1, 4, -1)) {
        return fail("sequence 1 pre-verification restore");
    }
    if (!llama_recurrent_set_sparse_snapshot_mode(ctx, true, 1) || !decode_ranges(ctx, { { 1, 4 } }, 5)) {
        return fail("sequence 1 selected replay");
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 1, 6, -1)) {
        return fail("sequence 1 selected boundary");
    }

    if (!llama_recurrent_set_sparse_snapshot_mode(ctx, true, -1) || !decode_ranges(ctx, { { 0, 7 }, { 1, 6 } }, 3)) {
        return fail("selected-to-trailing co-batch");
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, 7, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx), 1, 6, -1)) {
        return fail("co-batched trailing boundaries");
    }

    fprintf(stderr, "%s : selected and trailing sparse boundaries survived two-sequence replay\n", __func__);
    cleanup();
    return true;
}

static bool test_sparse_short_trailing_gaps(const common_params & params, llama_model * model, const int n_vocab) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 3;
    cparams.n_ctx     = 32;
    cparams.n_batch   = 16;
    cparams.n_ubatch  = 16;

    llama_context * probe = llama_init_from_model(model, cparams);
    if (probe == nullptr) {
        fprintf(stderr, "%s : failed to init capability context\n", __func__);
        return false;
    }
    const bool supported = llama_recurrent_sparse_snapshots_supported(probe);
    llama_free(probe);
    if (!supported) {
        fprintf(stderr, "%s : skipping because sparse snapshots are unsupported\n", __func__);
        return true;
    }

    const auto tok = [&](llama_pos pos) {
        return (llama_token) ((17*(uint32_t) pos + 3) % (uint32_t) n_vocab);
    };
    const auto decode_range = [&](llama_context * ctx, llama_pos first, uint32_t count) {
        llama_batch batch = llama_batch_init(count, 0, 1);
        for (uint32_t i = 0; i < count; ++i) {
            const llama_pos pos = first + (llama_pos) i;
            common_batch_add(batch, tok(pos), pos, { 0 }, i + 1 == count);
        }
        const bool ok = llama_decode(ctx, batch) == 0;
        llama_batch_free(batch);
        return ok;
    };

    constexpr float eps = 1e-5f;
    for (uint32_t trailing : { 1u, 2u }) {
        const llama_pos gap_pos = trailing == 1 ? 1 : 0;
        for (llama_pos rollback_pos : { gap_pos, (llama_pos) 2 }) {
            llama_context * ctx_sparse = llama_init_from_model(model, cparams);
            llama_context * ctx_ref    = llama_init_from_model(model, cparams);
            if (ctx_sparse == nullptr || ctx_ref == nullptr) {
                llama_free(ctx_sparse);
                llama_free(ctx_ref);
                fprintf(stderr, "%s : failed to init T=%u contexts\n", __func__, trailing);
                return false;
            }

            const bool ok = llama_recurrent_set_sparse_snapshot_mode(ctx_sparse, true, -1) &&
                decode_range(ctx_sparse, 0, 3) && decode_range(ctx_sparse, 3, trailing) &&
                decode_range(ctx_ref, 0, (uint32_t) rollback_pos + 1) &&
                llama_memory_seq_rm(llama_get_memory(ctx_sparse), 0, rollback_pos + 1, -1) &&
                decode_range(ctx_sparse, rollback_pos + 1, 1) && decode_range(ctx_ref, rollback_pos + 1, 1);
            if (!ok) {
                llama_free(ctx_sparse);
                llama_free(ctx_ref);
                fprintf(stderr, "%s : T=%u rollback at position %d failed\n", __func__, trailing, rollback_pos);
                return false;
            }

            const float * logits_sparse = llama_get_logits_ith(ctx_sparse, 0);
            const float * logits_ref    = llama_get_logits_ith(ctx_ref, 0);
            if (logits_sparse == nullptr || logits_ref == nullptr) {
                llama_free(ctx_sparse);
                llama_free(ctx_ref);
                fprintf(stderr, "%s : T=%u rollback logits missing\n", __func__, trailing);
                return false;
            }
            for (int token = 0; token < n_vocab; ++token) {
                if (std::fabs(logits_sparse[token] - logits_ref[token]) > eps) {
                    llama_free(ctx_sparse);
                    llama_free(ctx_ref);
                    fprintf(stderr, "%s : T=%u rollback at position %d mismatched token %d\n", __func__, trailing, rollback_pos, token);
                    return false;
                }
            }
            llama_free(ctx_sparse);
            llama_free(ctx_ref);
        }
    }

    fprintf(stderr, "%s : K=4 convolution and recurrent-attention gaps matched rollback references\n", __func__);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    if (!test_sparse_metadata_seq_id_bounds(model)) {
        return 1;
    }

    llama_context * ctx_src = make_ctx(params, model);
    llama_context * ctx_dst = make_ctx(params, model);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src, "The quick brown fox jumps over the lazy dog", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }
    if (tokens.empty()) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    tokens.resize(n_rs_seq + 1, tokens.back());

    const uint32_t  n_tokens     = tokens.size();
    const llama_pos rollback_pos = (llama_pos) n_tokens - n_rollback;

    // Decode the full prompt on the source, then roll back three positions.
    // Replaying them crosses DSV4's ratio-4 compressor boundary.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    constexpr float eps = 1e-5f;
    std::vector<std::vector<float>> logits_src_replay(n_rollback);
    const auto replay_and_compare = [&](const char * mode) {
        for (uint32_t i = 0; i < n_rollback; ++i) {
            const llama_pos pos = rollback_pos + i;
            if (!decode_one(ctx_src, tokens[pos], pos) ||
                !decode_one(ctx_dst, tokens[pos], pos)) {
                fprintf(stderr, "%s : %s replay failed at position %d\n", __func__, mode, pos);
                return false;
            }

            const float * logits_src = llama_get_logits_ith(ctx_src, 0);
            const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
            if (logits_src == nullptr || logits_dst == nullptr) {
                fprintf(stderr, "%s : missing %s logits at position %d\n", __func__, mode, pos);
                return false;
            }

            logits_src_replay[i].assign(logits_src, logits_src + n_vocab);
            for (int token = 0; token < n_vocab; ++token) {
                if (std::fabs(logits_src[token] - logits_dst[token]) > eps) {
                    fprintf(stderr, "%s : %s logits mismatch at position %d, token %d (%g != %g)\n",
                            __func__, mode, pos, token, (double) logits_src[token], (double) logits_dst[token]);
                    return false;
                }
            }
        }
        return true;
    };
    if (!replay_and_compare("full")) {
        return 1;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, rollback_pos, -1) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_dst), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : partial rollback failed\n", __func__);
        return 1;
    }

    constexpr llama_state_seq_flags partial_flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    common_prompt_checkpoint ckpt_partial;
    ckpt_partial.update_tgt(ctx_src, 0, partial_flags);
    ckpt_partial.load_tgt(ctx_dst, 0, partial_flags);

    if (!replay_and_compare("partial")) {
        return 1;
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a different prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, rollback_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    for (uint32_t i = 0; i < n_rollback; ++i) {
        const llama_pos pos = rollback_pos + i;
        if (!decode_one(ctx_dirty, tokens[pos], pos)) {
            fprintf(stderr, "%s : dirty replay failed at position %d\n", __func__, pos);
            return 1;
        }

        const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
        if (logits_dirty == nullptr) {
            fprintf(stderr, "%s : missing dirty logits at position %d\n", __func__, pos);
            return 1;
        }

        for (int token = 0; token < n_vocab; ++token) {
            if (std::fabs(logits_src_replay[i][token] - logits_dirty[token]) > eps) {
                fprintf(stderr, "%s : dirty-ctx logits mismatch at position %d, token %d (%g != %g)\n",
                        __func__, pos, token, (double) logits_src_replay[i][token], (double) logits_dirty[token]);
                return 1;
            }
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);

    if (!test_multi_seq_split_replay(params, model, n_vocab)) {
        return 1;
    }

    if (!test_sparse_selected_then_trailing(params, model, n_vocab)) {
        return 1;
    }

    if (!test_sparse_short_trailing_gaps(params, model, n_vocab)) {
        return 1;
    }

    return 0;
}
