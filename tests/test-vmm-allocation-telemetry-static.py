#!/usr/bin/env python3

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(text: str, needle: str, message: str) -> None:
    if needle not in text:
        raise AssertionError(message)


def main() -> None:
    cuda = (ROOT / "ggml/src/ggml-cuda/ggml-cuda.cu").read_text(encoding="utf-8")
    bench = (ROOT / "tools/llama-bench/llama-bench.cpp").read_text(encoding="utf-8")
    public_backend = (ROOT / "ggml/include/ggml-backend.h").read_text(encoding="utf-8")

    for symbol in (
        "ggml_backend_cuda_vmm_pool_stats_get",
        "ggml_backend_cuda_vmm_pool_stats_reset",
        "live_peak_bytes",
        "mapped_peak_bytes",
        "active_pools",
    ):
        require(cuda, symbol, f"CUDA VMM telemetry is missing {symbol}")

    cuda_payload = cuda.split("struct ggml_cuda_vmm_pool_stats {", 1)[1].split("};", 1)[0]
    bench_payload = bench.split("struct bench_cuda_vmm_pool_stats {", 1)[1].split("};", 1)[0]
    cuda_payload_fields = [line.strip() for line in cuda_payload.splitlines() if line.strip()]
    bench_payload_fields = [line.strip() for line in bench_payload.splitlines() if line.strip()]
    if cuda_payload_fields != bench_payload_fields:
        raise AssertionError("private CUDA VMM proc-address payload layouts must remain identical")

    atomic_stats = cuda.split("struct ggml_cuda_vmm_pool_stats_atomic {", 1)[1].split(
        "static ggml_cuda_vmm_pool_stats_atomic", 1)[0]
    require(" ".join(atomic_stats.split()), "std::atomic<bool> enabled { false };",
            "CUDA VMM allocation accounting must stay dormant until explicitly enabled")
    require(
        cuda,
        "ggml_backend_cuda_trim_transient_pools",
        "published W02 telemetry must coexist with live-policy idle trimming",
    )

    for field in (
        '"cuda_device_context_buffer_bytes"',
        '"cuda_device_compute_buffer_bytes"',
        '"accelerator_host_context_buffer_bytes"',
        '"accelerator_host_compute_buffer_bytes"',
        '"host_context_buffer_bytes"',
        '"host_compute_buffer_bytes"',
    ):
        require(bench, field, f"llama-bench output is missing {field}")

    output_fields = bench.split("static const std::vector<std::string> fields = {", 1)[1].split("};", 1)[0]
    output_values = bench.split("std::vector<std::string> values = {", 1)[1].split("};", 1)[0]
    field_cursor = -1
    output_cursor = -1
    for field, value in (
        ("cuda_vmm_live_model_bytes", "cuda_vmm_model.live_bytes"),
        ("cuda_vmm_mapped_model_bytes", "cuda_vmm_model.mapped_bytes"),
        ("cuda_vmm_live_context_bytes", "cuda_vmm_context.live_bytes"),
        ("cuda_vmm_mapped_context_bytes", "cuda_vmm_context.mapped_bytes"),
        ("cuda_vmm_live_after_workload_bytes", "cuda_vmm_after_workload.live_bytes"),
        ("cuda_vmm_mapped_after_workload_bytes", "cuda_vmm_after_workload.mapped_bytes"),
        ("cuda_vmm_live_peak_bytes", "cuda_vmm_live_peak_bytes"),
        ("cuda_vmm_mapped_peak_bytes", "cuda_vmm_mapped_peak_bytes"),
        ("cuda_vmm_live_after_context_bytes", "cuda_vmm_after_context.live_bytes"),
        ("cuda_vmm_mapped_after_context_bytes", "cuda_vmm_after_context.mapped_bytes"),
        ("cuda_vmm_active_pools_model", "cuda_vmm_model.active_pools"),
        ("cuda_vmm_active_pools_context", "cuda_vmm_context.active_pools"),
        ("cuda_vmm_active_pools_after_workload", "cuda_vmm_after_workload.active_pools"),
        ("cuda_vmm_active_pools_after_context", "cuda_vmm_after_context.active_pools"),
    ):
        field_cursor = output_fields.find(f'"{field}"', field_cursor + 1)
        output_cursor = output_values.find(f"std::to_string({value})", output_cursor + 1)
        if field_cursor < 0 or output_cursor < 0:
            raise AssertionError(f"llama-bench VMM output mapping is missing {field}")

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

    bench_main = bench.split("int llama_bench(int argc, char ** argv) {", 1)[1]
    enable_block = bench_main.split("if (params.kv_memory) {", 1)[1].split("llama_context * ctx", 1)[0]
    require(
        enable_block,
        "vmm_telemetry.reset()",
        "--kv-memory must be the opt-in telemetry client",
    )
    prefix = bench_main.split("if (params.kv_memory) {", 1)[0]
    if "vmm_telemetry.reset()" in prefix:
        raise AssertionError("VMM telemetry must not be enabled on the default benchmark path")

    for symbol in (
        "ggml_backend_cuda_vmm_pool_stats_get",
        "ggml_backend_cuda_vmm_pool_stats_reset",
    ):
        if symbol in public_backend:
            raise AssertionError(f"private proc-address extension leaked into public compile surface: {symbol}")

    require(
        bench,
        "live_context_workspace",
        "composed telemetry benchmark must publish the selected live-workspace policy",
    )
    for excluded in ("phase_aware_workspace", "flash_attn_native_quants"):
        if excluded in bench:
            raise AssertionError(f"W02 must not depend on later feature field {excluded}")


if __name__ == "__main__":
    main()
