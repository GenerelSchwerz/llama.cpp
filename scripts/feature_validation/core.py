"""Compatibility façade, scheduling, statistics, and runtime validation helpers.

Manifest and command contracts live in :mod:`feature_validation.contracts`;
source and build identity lives in :mod:`feature_validation.provenance`. Both
are re-exported here to preserve the established ``core`` API. Commands remain
argv lists until flock's ``-c`` interface requires shell quoting.
"""

from __future__ import annotations

import math
import os
import pathlib
import re
import shlex
import statistics
from typing import Any, Iterable

from .artifacts import (
    canonical_json,
    canonical_sha256,
    load_json,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from .contracts import (
    FORBIDDEN_TOKENS,
    IMMUTABLE_AB_OPTIONS,
    LLAMA_BENCH_OPTION_ARITY,
    LLAMA_COMMON_OPTION_ARITY,
    MEMORY_EVIDENCE_CATEGORIES,
    OPAQUE_WRAPPERS,
    PROFILE_TOOLS,
    PURPOSE_ORDER,
    SCHEMA_VERSION,
    ManifestError,
    ProvenanceError,
    ValidationError,
    _basename,
    _forbidden_selection_key,
    _stage_screen_ids,
    _validate_profiler_details,
    _validate_repetitions,
    _validate_screening_policy,
    _validate_selection,
    load_and_validate_manifest,
    option_arities,
    option_names,
    reject_forbidden_tokens,
    safe_slug,
    stage_by_id,
    validate_argv,
    validate_artifact_directory_isolation,
    validate_direct_target,
    validate_manifest,
    validate_sha256,
    variant_executables,
)

from .provenance import (
    BUILD_PROVENANCE_KIND,
    BUILD_PROVENANCE_SCHEMA_VERSION,
    _find_cmake_cache,
    _git_text,
    binary_snapshot,
    capture_cmake_build_provenance,
    capture_provenance,
    cmake_snapshot,
    command_capture,
    git_snapshot,
    git_source_file_manifest,
    provenance_identity_spec,
    stat_identity,
    verify_cmake_build_provenance,
    verify_variant,
)


GPU_LOCK = "/tmp/beellama-single-gpu.lock"


def quote_argv(argv: Iterable[str]) -> str:
    values = [str(value) for value in argv]
    if not values:
        raise ValidationError("cannot quote an empty argv")
    if any("\x00" in value for value in values):
        raise ValidationError("argv contains a NUL byte")
    return shlex.join(values)


def flock_argv(inner_argv: list[str]) -> list[str]:
    """Return the required whole-command lock representation."""
    return ["flock", GPU_LOCK, "-c", quote_argv(inner_argv)]


def build_schedule(manifest: dict[str, Any], stage: dict[str, Any]) -> list[dict[str, Any]]:
    policy = manifest["repetition_policy"]
    if stage["purpose"] == "exactness":
        orientations = ["AB"]
    elif stage.get("screening_policy") is not None:
        orientations = [stage["screening_policy"]["order"]]
    else:
        orientations = list(policy["order"])
    screens = stage.get("screens", [{"id": "default", "args": [], "weight": 1.0}])
    schedule: list[dict[str, Any]] = []
    for pair_index, orientation in enumerate(orientations, start=1):
        variant_order = ["baseline", "candidate"] if orientation == "AB" else ["candidate", "baseline"]
        for screen in screens:
            for order_index, variant in enumerate(variant_order, start=1):
                schedule.append(
                    {
                        "run_id": f"pair-{pair_index:02d}-{safe_slug(str(screen['id']))}-{order_index:02d}-{variant}",
                        "pair": pair_index,
                        "orientation": orientation,
                        "order": order_index,
                        "variant": variant,
                        "screen": str(screen["id"]),
                        "weight": float(screen.get("weight", 1.0)),
                    }
                )
    return schedule


def extract_metric(metric: dict[str, Any], stdout: str, stderr: str) -> float:
    stream_name = metric.get("stream", "stdout")
    if stream_name == "stdout":
        text = stdout
    elif stream_name == "stderr":
        text = stderr
    elif stream_name == "combined":
        text = stdout + "\n" + stderr
    else:
        raise ValidationError(f"unsupported metric stream {stream_name!r}")
    pattern = metric.get("regex")
    if not isinstance(pattern, str):
        raise ValidationError("metric regex is required")
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        raise ValidationError(f"metric regex did not match: {pattern!r}")
    policy = metric.get("match", "last")
    if policy == "only" and len(matches) != 1:
        raise ValidationError(f"metric regex matched {len(matches)} times; exactly one was required")
    match = matches[0] if policy == "first" else matches[-1]
    group = metric.get("group", 1)
    try:
        value = float(match.group(group))
    except (IndexError, ValueError) as error:
        raise ValidationError(f"metric match cannot be converted to float: {error}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValidationError(f"metric must be finite and positive, observed {value}")
    return value


def aggregate_screen_values(values: list[tuple[float, float]], mode: str) -> float:
    if not values:
        raise ValidationError("cannot aggregate an empty screen set")
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        raise ValidationError("screen weights must sum to a positive value")
    if mode == "weighted_mean":
        return sum(value * weight for value, weight in values) / total_weight
    if mode == "weighted_harmonic":
        return total_weight / sum(weight / value for value, weight in values)
    raise ValidationError(f"unknown aggregation mode {mode!r}")


T_CRITICAL_95 = {
    1: 12.7062047364,
    2: 4.3026527297,
    3: 3.1824463053,
    4: 2.7764451052,
}


def paired_log_report(
    pairs: list[dict[str, Any]],
    *,
    direction: str,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    if len(pairs) not in (3, 5):
        raise ValidationError("paired statistics require exactly 3 or 5 independent matched pairs")
    if direction not in ("higher", "lower"):
        raise ValidationError("metric direction must be 'higher' or 'lower'")
    logs: list[float] = []
    raw: list[dict[str, Any]] = []
    for pair in pairs:
        baseline = float(pair["baseline"])
        candidate = float(pair["candidate"])
        if baseline <= 0 or candidate <= 0:
            raise ValidationError("paired log ratios require positive observations")
        log_ratio = math.log(candidate / baseline)
        logs.append(log_ratio)
        raw.append({**pair, "log_ratio": log_ratio, "percent_change": 100.0 * math.expm1(log_ratio)})
    mean_log = statistics.fmean(logs)
    standard_error = statistics.stdev(logs) / math.sqrt(len(logs))
    margin = T_CRITICAL_95[len(logs) - 1] * standard_error
    lower_log = mean_log - margin
    upper_log = mean_log + margin
    percent = 100.0 * math.expm1(mean_log)
    interval = [100.0 * math.expm1(lower_log), 100.0 * math.expm1(upper_log)]

    if direction == "higher":
        benefit_interval = interval
        benefit_percent = percent
    else:
        benefit_interval = [-interval[1], -interval[0]]
        benefit_percent = -percent
    improvement = float(thresholds["improvement_percent"])
    regression = float(thresholds["regression_percent"])
    equivalence = float(thresholds["equivalence_percent"])
    if benefit_interval[0] >= improvement:
        decision = "improvement"
    elif benefit_interval[1] <= -regression:
        decision = "regression"
    elif benefit_interval[0] >= -equivalence and benefit_interval[1] <= equivalence:
        decision = "equivalent"
    else:
        decision = "inconclusive"
    return {
        "pair_count": len(pairs),
        "method": "paired log ratio with two-sided Student-t 95% interval",
        "direction": direction,
        "geometric_ratio": math.exp(mean_log),
        "percent_change": percent,
        "percent_interval_95": interval,
        "benefit_percent": benefit_percent,
        "benefit_interval_95": benefit_interval,
        "decision": decision,
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "raw_pairs": raw,
    }


def single_pair_screen_report(
    pair: dict[str, Any], *, direction: str, regression_threshold_percent: float
) -> dict[str, Any]:
    """Report one fail-fast screening pair without claiming statistical confidence."""
    if direction not in ("higher", "lower"):
        raise ValidationError("metric direction must be 'higher' or 'lower'")
    baseline = float(pair["baseline"])
    candidate = float(pair["candidate"])
    if baseline <= 0 or candidate <= 0:
        raise ValidationError("single-pair screening ratios require positive observations")
    log_ratio = math.log(candidate / baseline)
    percent = 100.0 * math.expm1(log_ratio)
    benefit_percent = percent if direction == "higher" else -percent
    signal = (
        "regression_signal"
        if benefit_percent <= -float(regression_threshold_percent)
        else "clear_to_continue"
    )
    return {
        "pair_count": 1,
        "method": "single matched-pair fail-fast screen",
        "direction": direction,
        "geometric_ratio": math.exp(log_ratio),
        "percent_change": percent,
        "benefit_percent": benefit_percent,
        "confidence_interval": None,
        "confidence_claim": "none",
        "signal": signal,
        "regression_threshold_percent": float(regression_threshold_percent),
        "raw_pairs": [
            {
                **pair,
                "log_ratio": log_ratio,
                "percent_change": percent,
            }
        ],
        "evidence_boundary": (
            "A single-pair screen is a preregistered fail-fast signal only; it is not "
            "a confidence interval, equivalence result, or performance acceptance claim."
        ),
    }


def performance_outcome(decision: str, policy: dict[str, Any]) -> str:
    """Map an executed statistical decision to a fail-closed gate outcome."""
    if decision in policy["acceptable_decisions"]:
        return "passed"
    if decision == "inconclusive":
        return "unresolved"
    if decision == "regression":
        return "failed"
    raise ValidationError(f"unknown performance decision {decision!r}")


def sterile_environment(*parts: dict[str, Any]) -> dict[str, str]:
    environment = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LC_ALL": "C",
    }
    for part in parts:
        for key, value in part.items():
            environment[str(key)] = str(value)
    return environment


def command_for_run(
    manifest: dict[str, Any], stage: dict[str, Any], run: dict[str, Any]
) -> tuple[list[str], dict[str, str]]:
    variant_name = run["variant"]
    variant = manifest["variants"][variant_name]
    command = stage["command"]
    executable = variant_executables(variant)[str(command.get("executable_role", "default"))]
    screen = next(item for item in stage.get("screens", [{"id": "default", "args": []}]) if item["id"] == run["screen"])
    argv = [
        str(pathlib.Path(executable["path"]).expanduser().resolve()),
        *[str(value) for value in command.get("common_args", [])],
        *[str(value) for value in screen.get("args", [])],
        *[str(value) for value in command.get(f"{variant_name}_args", [])],
    ]
    reject_forbidden_tokens(argv)
    validate_direct_target(pathlib.Path(argv[0]), direct_harness=executable.get("direct_harness"))
    validate_argv(argv[1:], command.get("cli_schema"), argv[0])
    environment = sterile_environment(
        manifest.get("environment", {}),
        variant.get("environment", {}),
        command.get("environment", {}),
        command.get(f"{variant_name}_environment", {}),
    )
    return argv, environment


def host_load_snapshot() -> dict[str, Any]:
    meminfo: dict[str, int] = {}
    path = pathlib.Path("/proc/meminfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, remainder = line.partition(":")
            if separator and key in {"MemAvailable", "MemFree", "SwapFree", "Mlocked", "PageTables"}:
                match = re.search(r"([0-9]+)", remainder)
                if match:
                    meminfo[f"{key}_kib"] = int(match.group(1))
    return {
        "captured_utc": utc_now(),
        "load_average": list(os.getloadavg()),
        "meminfo": meminfo,
    }
