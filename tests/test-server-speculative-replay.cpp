#include "../src/llama-model.h"
#include "server-speculative-replay.h"

#undef NDEBUG
#include <cassert>
#include <array>
#include <vector>

static void test_replay_state() {
    server_speculative_replay_state state;

    assert(!state.replaying());
    assert(!state.mtp_gpu_snapshots_armed());
    assert(!state.mtp_gpu_replay_pending());
    assert(!state.excludes_replayed_token_from_acceptance());

    state.discard_mtp_gpu_snapshot_arm();
    assert(!state.replaying());

    state.arm_mtp_gpu_snapshots(true);
    state.arm_mtp_gpu_snapshots(true);
    assert(state.mtp_gpu_snapshots_armed());
    state.reset();
    assert(!state.mtp_gpu_snapshots_armed());

    state.arm_mtp_gpu_snapshots(false);
    assert(!state.replaying());

    state.arm_mtp_gpu_snapshots(true);
    state.arm_mtp_gpu_snapshots(false);
    assert(!state.mtp_gpu_snapshots_armed());

    state.arm_mtp_gpu_snapshots(true);
    assert(state.mtp_gpu_snapshots_armed());
    assert(!state.replaying());

    llama_model_ptr sampler_model(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    assert(sampler_model != nullptr);

    common_params_sampling sampler_params;
    sampler_params.samplers = { COMMON_SAMPLER_TYPE_TOP_K };

    llama_tokens replay_tokens {11, 12, 13};
    common_sampler_ptr sampler(common_sampler_init(sampler_model.get(), sampler_params));
    assert(sampler != nullptr);
    common_sampler * sampler_ptr = sampler.get();
    state.begin_mtp_gpu_replay(std::move(replay_tokens), std::move(sampler), 2);

    assert(sampler == nullptr);
    assert(state.replaying());
    assert(!state.mtp_gpu_snapshots_armed());
    assert(state.mtp_gpu_replay_pending());
    assert(state.mtp_gpu_replay_selected_token() == 2);
    assert(!state.excludes_replayed_token_from_acceptance());

    llama_tokens accepted;
    common_sampler_ptr accepted_sampler;
    assert(state.consume_mtp_gpu_replay(accepted, accepted_sampler) == 2);
    assert(accepted.size() == 3);
    assert(accepted[0] == 11 && accepted[1] == 12 && accepted[2] == 13);
    assert(accepted_sampler.get() == sampler_ptr);
    assert(state.replaying());
    assert(!state.mtp_gpu_replay_pending());
    assert(!state.excludes_replayed_token_from_acceptance());

    state.finish_verification();
    assert(!state.replaying());

    state.set_checkpoint_replay(true);
    state.set_checkpoint_replay(true);
    state.discard_mtp_gpu_snapshot_arm();
    assert(state.replaying());
    assert(!state.mtp_gpu_replay_pending());
    assert(state.excludes_replayed_token_from_acceptance());
    state.reset();
    assert(!state.replaying());

    state.set_checkpoint_replay(true);
    state.finish_verification();

    state.set_checkpoint_replay(false);
    assert(!state.replaying());

    // Reset and discard must both leave the state reusable and clear owned payloads.
    state.arm_mtp_gpu_snapshots(true);
    state.discard_mtp_gpu_snapshot_arm();
    assert(!state.replaying());
    assert(!state.mtp_gpu_snapshots_armed());

    llama_tokens reset_tokens {21};
    common_sampler_ptr reset_sampler(common_sampler_init(sampler_model.get(), sampler_params));
    assert(reset_sampler != nullptr);
    state.arm_mtp_gpu_snapshots(true);
    state.begin_mtp_gpu_replay(std::move(reset_tokens), std::move(reset_sampler), 0);
    assert(reset_sampler == nullptr);
    assert(state.mtp_gpu_replay_pending());
    assert(state.mtp_gpu_replay_selected_token() == 0);
    state.reset();
    assert(!state.replaying());
    assert(!state.mtp_gpu_replay_pending());
    assert(!state.mtp_gpu_snapshots_armed());
}

static void test_affected_slot_predicate() {
    struct test_token {
        int32_t id_slot;
    };
    struct test_case {
        int32_t replay_slot_id;
        bool sparse_verification;
        std::array<bool, 5> affected;
    };

    const std::vector<test_token> tokens = { { 1 }, { 3 }, { 1 } };
    const std::array<test_case, 4> cases = {
        test_case {  2, false, { false, false, true,  false, false } },
        test_case {  2, true,  { false, false, true,  false, false } },
        test_case { -1, true,  { false, true,  false, true,  false } },
        test_case { -1, false, { false, false, false, false, false } },
    };

    for (const auto & test : cases) {
        for (size_t slot_id = 0; slot_id < test.affected.size(); ++slot_id) {
            assert(server_sparse_batch_slot_is_affected(
                    test.replay_slot_id, test.sparse_verification, tokens, int32_t(slot_id)) ==
                    test.affected[slot_id]);
        }
    }
}

int main() {
    test_replay_state();
    test_affected_slot_predicate();

    return 0;
}
