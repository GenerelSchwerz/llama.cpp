#include "server-speculative-replay.h"

#undef NDEBUG
#include <cassert>

int main() {
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

    // The replay state owns this opaque pointer but never dereferences it.
    // Release it after the round trip so the test sentinel is not deleted.
    llama_tokens replay_tokens {11, 12, 13};
    int sampler_storage = 0;
    auto * fake_sampler = reinterpret_cast<common_sampler *>(&sampler_storage);
    common_sampler_ptr sampler(fake_sampler);
    state.begin_mtp_gpu_replay(std::move(replay_tokens), std::move(sampler), 2);

    assert(sampler == nullptr);
    assert(state.replaying());
    assert(!state.mtp_gpu_snapshots_armed());
    assert(state.mtp_gpu_replay_pending());
    assert(!state.excludes_replayed_token_from_acceptance());

    llama_tokens accepted;
    common_sampler_ptr accepted_sampler;
    assert(state.consume_mtp_gpu_replay(accepted, accepted_sampler) == 2);
    assert(accepted.size() == 3);
    assert(accepted[0] == 11 && accepted[1] == 12 && accepted[2] == 13);
    assert(accepted_sampler.release() == fake_sampler);
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

    state.arm_mtp_gpu_snapshots(true);
    state.discard_mtp_gpu_snapshot_arm();
    assert(!state.replaying());
    assert(!state.mtp_gpu_snapshots_armed());

    llama_tokens reset_tokens {21};
    int reset_sampler_storage = 0;
    auto * reset_fake_sampler = reinterpret_cast<common_sampler *>(&reset_sampler_storage);
    common_sampler_ptr reset_sampler(reset_fake_sampler);
    state.arm_mtp_gpu_snapshots(true);
    state.begin_mtp_gpu_replay(std::move(reset_tokens), std::move(reset_sampler), 0);
    assert(state.consume_mtp_gpu_replay(accepted, accepted_sampler) == 0);
    assert(accepted_sampler.release() == reset_fake_sampler);
    state.reset();
    assert(!state.replaying());

    return 0;
}
