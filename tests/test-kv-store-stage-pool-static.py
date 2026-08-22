#!/usr/bin/env python3

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(source: str, marker: str, description: str) -> None:
    if marker not in source:
        raise AssertionError(f"KV store-stage pool lacks {description}: {marker}")


def main() -> None:
    cache = (ROOT / "src/llama-kv-cache.cpp").read_text(encoding="utf-8")
    key = (ROOT / "src/llama-kv-cache-store-stage.h").read_text(encoding="utf-8")
    test = (ROOT / "tests/test-kv-store-stage-pool.cpp").read_text(encoding="utf-8")

    for field in ("side", "buft", "type", "n_embd"):
        require(key, field, f"exact compatibility field {field}")
    for side in ("llama_kv_store_stage_side::K", "llama_kv_store_stage_side::V"):
        require(cache, side, f"separate {side[-1]} stage domain")

    selector = cache.split("const auto store_stage_buft", 1)[1].split(
        "const auto pooled_store_stage", 1
    )[0]
    for capability in (
        "!ggml_is_quantized(type)",
        "GGML_BACKEND_DEVICE_TYPE_CPU",
        "ggml_backend_buft_is_host(buft)",
        "direct_store && staged_store",
    ):
        require(selector, capability, "capability/layout eligibility check")

    pool = cache.split("const auto pooled_store_stage", 1)[1].split(
        "for (uint32_t il = 0; il < n_layer", 1
    )[0]
    require(pool, "entry.key == key", "exact-match reuse")
    require(pool, "ctx_for_buft(buft)", "buffer-type-owned allocation")
    if pool.count("ggml_new_tensor_2d") != 1:
        raise AssertionError("store-stage pool must have one allocation site")

    placement = cache.split("ggml_tensor * k_store_stage", 1)[1].split(
        "if (k) {", 1
    )[0]
    for fallback in (
        "ggml_backend_buft_is_host(buft) && !compact_native_exact",
        "has_v && !v_trans",
        "cannot preserve accelerator",
    ):
        require(placement, fallback, "unchanged unsupported fallback")

    forbidden = re.compile(r"LLM_ARCH|QWEN|GEMMA|getenv|GGML_CUDA_KVARN", re.IGNORECASE)
    for path, source in (
        ("src/llama-kv-cache-store-stage.h", key),
        ("store-stage pooling block", pool),
    ):
        match = forbidden.search(source)
        if match:
            raise AssertionError(f"{path} contains forbidden coupling: {match.group(0)}")

    for regression in (
        "test_key_compatibility",
        "test_reused_stage_graph_lifetime",
        "pooled stage producer/consumer order changed",
        "k_stage",
        "v_stage",
    ):
        require(test, regression, "focused lifetime regression")


if __name__ == "__main__":
    main()
