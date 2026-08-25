#pragma once

#include "common.h"
#include "sampling.h"

#include <cstdint>
#include <utility>

template <class TokenRange>
bool server_sparse_batch_slot_is_affected(
        int32_t replay_slot_id, bool sparse_verification, const TokenRange & tokens, int32_t slot_id) {
    if (replay_slot_id >= 0) {
        return replay_slot_id == slot_id;
    }
    if (!sparse_verification) {
        return false;
    }
    for (const auto & token : tokens) {
        if (token.id_slot == slot_id) {
            return true;
        }
    }
    return false;
}

// Owns replay state for one speculative batch.
// Checkpoint and capped-MTP GPU replay use the same slot lifetime.
// Only GPU replay owns accepted tokens and its sampler snapshot.
struct server_speculative_replay_state {
    bool mtp_gpu_snapshots_armed() const {
        return phase == phase_type::MTP_GPU_SNAPSHOTS_ARMED;
    }

    bool mtp_gpu_replay_pending() const {
        return phase == phase_type::MTP_GPU_REPLAY_PENDING;
    }

    bool excludes_replayed_token_from_acceptance() const {
        return phase == phase_type::CHECKPOINT_REPLAY;
    }

    uint32_t mtp_gpu_replay_selected_token() const {
        GGML_ASSERT(mtp_gpu_replay_pending());
        return n_accepted;
    }

    void arm_mtp_gpu_snapshots() {
        GGML_ASSERT(phase == phase_type::IDLE || phase == phase_type::MTP_GPU_SNAPSHOTS_ARMED);
        phase = phase_type::MTP_GPU_SNAPSHOTS_ARMED;
    }

    void begin_checkpoint_replay() {
        GGML_ASSERT(phase == phase_type::IDLE || phase == phase_type::CHECKPOINT_REPLAY);
        phase = phase_type::CHECKPOINT_REPLAY;
    }

    void begin_mtp_gpu_replay(llama_tokens tokens, common_sampler_ptr sampler, uint32_t accepted) {
        GGML_ASSERT(phase == phase_type::MTP_GPU_SNAPSHOTS_ARMED && sampler != nullptr);
        accepted_tokens = std::move(tokens);
        accepted_sampler = std::move(sampler);
        n_accepted = accepted;
        phase = phase_type::MTP_GPU_REPLAY_PENDING;
    }

    uint32_t consume_mtp_gpu_replay(llama_tokens & tokens, common_sampler_ptr & sampler) {
        GGML_ASSERT(mtp_gpu_replay_pending() && accepted_sampler != nullptr);
        tokens = std::move(accepted_tokens);
        sampler = std::move(accepted_sampler);
        const uint32_t accepted = n_accepted;
        clear_payload();
        phase = phase_type::IDLE;
        return accepted;
    }

    void discard_mtp_gpu_snapshot_arm() {
        GGML_ASSERT(phase == phase_type::IDLE ||
                    phase == phase_type::MTP_GPU_SNAPSHOTS_ARMED ||
                    phase == phase_type::CHECKPOINT_REPLAY);
        if (phase == phase_type::MTP_GPU_SNAPSHOTS_ARMED) {
            phase = phase_type::IDLE;
        }
    }

    void finish_verification() {
        GGML_ASSERT(phase != phase_type::MTP_GPU_SNAPSHOTS_ARMED &&
                    phase != phase_type::MTP_GPU_REPLAY_PENDING);
        reset();
    }

    void reset() {
        phase = phase_type::IDLE;
        clear_payload();
    }

private:
    enum class phase_type {
        IDLE,
        MTP_GPU_SNAPSHOTS_ARMED,
        CHECKPOINT_REPLAY,
        MTP_GPU_REPLAY_PENDING,
    };

    void clear_payload() {
        accepted_tokens.clear();
        accepted_sampler.reset();
        n_accepted = 0;
    }

    phase_type phase = phase_type::IDLE;
    llama_tokens accepted_tokens;
    common_sampler_ptr accepted_sampler;
    uint32_t n_accepted = 0;
};
