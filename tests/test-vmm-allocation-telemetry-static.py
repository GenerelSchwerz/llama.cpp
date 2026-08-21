#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(text: str, needle: str, message: str) -> None:
    if needle not in text:
        raise AssertionError(message)


def main() -> None:
    cuda = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
    bench = (ROOT / "tools/llama-bench/llama-bench.cpp").read_text(encoding="utf-8")

    for symbol in (
        "ggml_backend_cuda_vmm_pool_stats_get",
        "ggml_backend_cuda_vmm_pool_stats_reset",
        "live_peak_bytes",
        "mapped_peak_bytes",
        "active_pools",
    ):
        require(cuda, symbol, f"CUDA VMM telemetry is missing {symbol}")

    require(
        cuda,
        "if (telemetry_tracked || !stats.enabled.load(std::memory_order_relaxed))",
        "CUDA VMM allocation accounting must stay dormant until explicitly enabled",
    )
    if "ggml_backend_cuda_trim_transient_pools" in cuda:
        raise AssertionError("W02 must not import transient-pool trimming policy")

    for field in (
        '"cuda_vmm_live_peak_bytes"',
        '"cuda_vmm_mapped_peak_bytes"',
        '"cuda_device_context_buffer_bytes"',
        '"cuda_device_compute_buffer_bytes"',
        '"accelerator_host_context_buffer_bytes"',
        '"accelerator_host_compute_buffer_bytes"',
        '"host_context_buffer_bytes"',
        '"host_compute_buffer_bytes"',
    ):
        require(bench, field, f"llama-bench output is missing {field}")

    require(
        bench,
        "if (ggml_backend_buft_is_host(buft))",
        "allocation classification must use buffer capability rather than backend names",
    )
    require(
        bench,
        "ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU",
        "allocation classification must use the owning device type",
    )
    require(
        bench,
        "const bool kv_in_cuda_owner_total = !inst.no_kv_offload || inst.kv_cpu_pinned;",
        "ordinary-host KV must not be subtracted from the historical CUDA-owner total",
    )

    enable_block = bench.split("if (params.kv_memory) {", 1)[1].split("llama_context * ctx", 1)[0]
    require(
        enable_block,
        "cuda_vmm_pool_stats_reset(inst.main_gpu)",
        "--kv-memory must be the opt-in telemetry client",
    )
    prefix = bench.split("if (params.kv_memory) {", 1)[0]
    if "cuda_vmm_pool_stats_reset(" in prefix:
        raise AssertionError("VMM telemetry must not be enabled on the default benchmark path")

    for excluded in (
        "phase_aware_workspace",
        "live_context_workspace",
        "flash_attn_native_quants",
    ):
        if excluded in bench:
            raise AssertionError(f"W02 must not depend on later feature field {excluded}")


if __name__ == "__main__":
    main()
