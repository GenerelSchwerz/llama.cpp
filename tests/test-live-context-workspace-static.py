#!/usr/bin/env python3
"""Guard live-context sizing, fallback ordering, and idle trim boundaries."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    memory = (ROOT / "src/llama-memory.h").read_text(encoding="utf-8")
    kv = (ROOT / "src/llama-kv-cache.cpp").read_text(encoding="utf-8")
    context = (ROOT / "src/llama-context.cpp").read_text(encoding="utf-8")
    server = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")
    cuda = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")

    require("virtual uint32_t get_attn_reserve_capacity() const { return 0; }" in memory,
            "unsupported memory must fail closed to zero bounded capacity")
    require("return init_full();" in memory,
            "unsupported bounded reservation must preserve full reservation")
    require("uint32_t llama_kv_cache::get_reserve_n_kv" in kv and
            "idx + 1" in kv and "used_max_p1()" in kv,
            "standard KV must publish planned and existing physical high rows")

    decode_start = context.find("int llama_context::decode(")
    decode_end = context.find("uint32_t llama_context::output_reserve(", decode_start)
    decode = context[decode_start:decode_end]
    upfront = decode.find("if (!live_exact_batch_plan)")
    update = decode.find("memory_update(false", upfront)
    exact = decode.find("mctx->get_attn_reserve_n_kv()", update)
    require(0 <= upfront < update < exact,
            "default-off must reserve before update and live sizing after exact batch publication")

    require("std::all_of(slots.begin(), slots.end()" in server and
            "llama_trim_transient_memory(ctx_tgt)" in server,
            "server trim must be guarded by the all-slots-idle boundary")
    require("ggml_backend_cuda_trim_transient_pools" in cuda and
            "cudaStreamSynchronize" in cuda and "->trim()" in cuda,
            "CUDA trim capability must synchronize before releasing pool tails")
    require("ggml_backend_cuda_vmm_pool_stats_get" in cuda and
            "ggml_backend_cuda_vmm_pool_stats_reset" in cuda,
            "published W02 telemetry must coexist with idle trimming")
    trim = cuda[cuda.find("size_t trim() override"):
                cuda.find("static bool ggml_backend_cuda_vmm_pool_stats_get")]
    require("track_telemetry();" in trim and
            "remove_mapped" in trim and "released" in trim,
            "idle trim must contract current mapped telemetry without discarding its peak")


if __name__ == "__main__":
    main()
