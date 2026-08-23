#pragma once

#include "common.h"
#include "sampling.h"

#include <cstdint>
#include <utility>

// Owns the server-side lifecycle of a replayed speculative batch.  Checkpoint
// replay and capped-MTP GPU replay share the same slot-level lifetime, while
// only GPU replay carries accepted tokens and their sampler snapshot.
struct server_speculative_replay_state {
    bool replaying() const {
        return phase == phase_type::CHECKPOINT_REPLAY ||
               phase == phase_type::MTP_GPU_REPLAY_PENDING ||
               phase == phase_type::MTP_GPU_REPLAY_CONSUMED;
    }

    bool mtp_gpu_snapshots_armed() const {
        return phase == phase_type::MTP_GPU_SNAPSHOTS_ARMED;
    }

    bool mtp_gpu_replay_pending() const {
        return phase == phase_type::MTP_GPU_REPLAY_PENDING;
    }

    bool excludes_replayed_token_from_acceptance() const {
        return phase == phase_type::CHECKPOINT_REPLAY;
    }

    void arm_mtp_gpu_snapshots(bool enabled) {
        // A failed checkpoint capture can leave an unused arm behind after
        // its draft is cleared.  Re-arming the next fresh draft replaces it.
        GGML_ASSERT(phase == phase_type::IDLE ||
                    phase == phase_type::MTP_GPU_SNAPSHOTS_ARMED);
        phase = enabled ? phase_type::MTP_GPU_SNAPSHOTS_ARMED : phase_type::IDLE;
    }

    void set_checkpoint_replay(bool enabled) {
        // Replay can revisit the checkpoint-restore path.  Preserve the
        // previous boolean assignment's idempotent replay transition.
        GGML_ASSERT(phase == phase_type::IDLE ||
                    phase == phase_type::CHECKPOINT_REPLAY);
        phase = enabled ? phase_type::CHECKPOINT_REPLAY : phase_type::IDLE;
    }

    void begin_mtp_gpu_replay(
            llama_tokens tokens,
            common_sampler_ptr sampler,
            uint32_t accepted) {
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
        phase = phase_type::MTP_GPU_REPLAY_CONSUMED;
        return accepted;
    }

    void discard_mtp_gpu_snapshot_arm() {
        GGML_ASSERT(phase == phase_type::IDLE ||
                    phase == phase_type::MTP_GPU_SNAPSHOTS_ARMED ||
                    phase == phase_type::CHECKPOINT_REPLAY);
        if (phase == phase_type::MTP_GPU_SNAPSHOTS_ARMED) {
            phase = phase_type::IDLE;
        }
        clear_payload();
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
        MTP_GPU_REPLAY_CONSUMED,
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
