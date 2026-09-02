#pragma once

#include "common.h"
#include "sampling.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

template <class TokenRange>
bool server_batch_range_has_slot(
        const TokenRange & tokens, size_t offset, size_t count, int32_t slot_id) {
    GGML_ASSERT(offset <= tokens.size() && count <= tokens.size() - offset);
    for (size_t index = offset; index < offset + count; ++index) {
        if (tokens[index].id_slot == slot_id) {
            return true;
        }
    }
    return false;
}

template <class TokenRange>
bool server_failed_batch_slot_is_affected(
        int32_t replay_slot_id, bool include_batch_slots, const TokenRange & tokens, int32_t slot_id) {
    if (replay_slot_id >= 0) {
        return replay_slot_id == slot_id;
    }
    if (!include_batch_slots) {
        return false;
    }
    return server_batch_range_has_slot(tokens, 0, tokens.size(), slot_id);
}

inline int32_t server_atomic_batch_retry_size(int32_t n_batch, int32_t span) {
    GGML_ASSERT(n_batch > 0);
    GGML_ASSERT(span > 0);
    return (n_batch / 2 / span) * span;
}

inline bool server_atomic_batch_decode_is_retryable(int32_t decode_result) {
    return decode_result == 1;
}

template <class SetState>
bool server_restore_mtp_slot_state(const std::vector<uint8_t> & data, SetState && set_state) {
    const std::vector<uint8_t> empty;
    if (!set_state(empty)) {
        return false;
    }
    if (data.empty() || set_state(data)) {
        return true;
    }
    set_state(empty);
    return false;
}

template <class SetState>
bool server_clear_mtp_slot_state(SetState && set_state) {
    return server_restore_mtp_slot_state(std::vector<uint8_t>(), std::forward<SetState>(set_state));
}

template <class GetState, class SetState>
bool server_copy_mtp_slot_state(GetState && get_state, SetState && set_state) {
    std::vector<uint8_t> data;
    if (!get_state(data)) {
        server_clear_mtp_slot_state(std::forward<SetState>(set_state));
        return false;
    }
    return server_restore_mtp_slot_state(data, std::forward<SetState>(set_state));
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

template <class PromptClear>
bool server_clear_pending_mtp_gpu_replay_on_release(
        const server_speculative_replay_state & state, PromptClear && prompt_clear) {
    if (!state.mtp_gpu_replay_pending()) {
        return false;
    }
    prompt_clear();
    return true;
}
