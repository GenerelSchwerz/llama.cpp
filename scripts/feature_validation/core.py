"""Core schema, provenance, scheduling, and statistics for feature validation.

This module intentionally uses only the Python standard library.  Commands are
represented as argv lists until the one place where flock's ``-c`` interface
requires a safely quoted shell command.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import platform
import re
import shlex
import shutil
import statistics
import subprocess
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


SCHEMA_VERSION = 1
BUILD_PROVENANCE_SCHEMA_VERSION = 1
BUILD_PROVENANCE_KIND = "beellama-feature-validation-cmake-build"
GPU_LOCK = "/tmp/beellama-single-gpu.lock"
PURPOSE_ORDER = {
    "exactness": 0,
    "smoke": 1,
    "kernel_screen": 2,
    "production_confirmation": 3,
    "long_context_acceptance": 4,
}
PROFILE_TOOLS = {"nsys", "ncu"}
MEMORY_EVIDENCE_CATEGORIES = {
    "gpu_memory_events",
    "allocation_lifetimes",
    "captured_device_high_water",
    "copy_activity",
    "cuda_api",
    "vmm",
    "nvtx_ranges",
}
FORBIDDEN_TOKENS = {"taskset"}
OPAQUE_WRAPPERS = {
    "bash",
    "dash",
    "env",
    "flock",
    "nice",
    "nohup",
    "python",
    "python3",
    "sh",
    "sudo",
    "timeout",
    "xargs",
    "zsh",
}
IMMUTABLE_AB_OPTIONS = {
    "-m",
    "--model",
    "-p",
    "--prompt",
    "-n",
    "--n-predict",
    "-d",
    "--depth",
    "-r",
    "--repetitions",
    "-b",
    "--batch-size",
    "-ub",
    "--ubatch-size",
    "-t",
    "--threads",
    "-tb",
    "--threads-batch",
    "-C",
    "--cpu-mask",
    "--cpu-range",
    "--cpu-range-batch",
    "--cpu-strict",
    "-ngl",
    "--n-gpu-layers",
    "-fa",
    "--flash-attn",
    "-ctk",
    "--cache-type-k",
    "-ctv",
    "--cache-type-v",
    "--seed",
}

# This is deliberately a validation schema, not an invocation generator.  A
# manifest may extend it with explicit option arities for a local harness.
LLAMA_COMMON_OPTION_ARITY: dict[str, int] = {
    "-m": 1,
    "--model": 1,
    "-p": 1,
    "--prompt": 1,
    "-n": 1,
    "--n-predict": 1,
    "-d": 1,
    "--depth": 1,
    "-r": 1,
    "--repetitions": 1,
    "-b": 1,
    "--batch-size": 1,
    "-ub": 1,
    "--ubatch-size": 1,
    "-t": 1,
    "--threads": 1,
    "-tb": 1,
    "--threads-batch": 1,
    "-C": 1,
    "--cpu-mask": 1,
    "--cpu-range": 1,
    "--cpu-range-batch": 1,
    "--cpu-strict": 1,
    "--poll": 1,
    "-ngl": 1,
    "--n-gpu-layers": 1,
    "--n-gpu-layers-draft": 1,
    "-sm": 1,
    "--split-mode": 1,
    "-mg": 1,
    "--main-gpu": 1,
    "-fa": 1,
    "--flash-attn": 1,
    "-ctk": 1,
    "--cache-type-k": 1,
    "-ctv": 1,
    "--cache-type-v": 1,
    "-nkvo": 0,
    "--kv-offload": 0,
    "--no-kv-offload": 0,
    "--kv-cpu-pinned": 0,
    "--no-kv-cpu-pinned": 0,
    "--recurrent-state-offload": 0,
    "--no-recurrent-state-offload": 0,
    "--kv-gpu-layers": 1,
    "--spec-draft-kv-gpu-layers": 1,
    "--spec-draft-ubatch-size": 1,
    "--spec-mtp-rs-planes": 1,
    "--phase-aware-workspace": 0,
    "--no-phase-aware-workspace": 0,
    "--live-context-workspace": 0,
    "--no-live-context-workspace": 0,
    "--flash-attn-native-quants": 0,
    "--no-flash-attn-native-quants": 0,
    "--no-warmup": 0,
    "--progress": 0,
    "--kv-memory": 0,
    "-o": 1,
    "--output": 1,
    "--ctx-size": 1,
    "--parallel": 1,
    "--cont-batching": 0,
    "--kv-unified": 0,
    "--fit": 1,
    "--seed": 1,
    "--cache-ram": 1,
    "--verbosity": 1,
    "--spec-type": 1,
    "--spec-draft-model": 1,
    "--spec-draft-n-max": 1,
    "--spec-draft-p-min": 1,
    "--spec-draft-type-k": 1,
    "--spec-draft-type-v": 1,
    "--host": 1,
    "--port": 1,
    "--alias": 1,
}

# llama-bench has its own argument parser.  Unlike common_arg, both of its
# no-KV-offload spellings consume an explicit 0/1 value.
LLAMA_BENCH_OPTION_ARITY = {
    **LLAMA_COMMON_OPTION_ARITY,
    "-nkvo": 1,
    "--no-kv-offload": 1,
}


class ValidationError(ValueError):
    """Base class for fail-closed validation errors."""


class ManifestError(ValidationError):
    """The preregistered manifest is incomplete or inconsistent."""


class ProvenanceError(ValidationError):
    """Observed source/build/input identity does not match the manifest."""


def stat_identity(path: pathlib.Path) -> dict[str, int]:
    observed = path.stat()
    return {
        "size": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
        "device": observed.st_dev,
        "inode": observed.st_ino,
    }


def command_capture(
    argv: list[str], cwd: pathlib.Path | None = None, timeout: float = 30.0
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "error": f"{type(error).__name__}: {error}"}
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


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


def safe_slug(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ManifestError(
            f"{value!r} is not a safe identifier; use letters, digits, dot, underscore, or dash"
        )
    return value


def validate_sha256(value: Any, label: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", text):
        raise ManifestError(f"{label} must be a 64-digit SHA-256")
    return text.lower()


def _basename(token: str) -> str:
    return pathlib.PurePath(token).name


def reject_forbidden_tokens(argv: Iterable[str]) -> None:
    for token in argv:
        if _basename(str(token)) in FORBIDDEN_TOKENS:
            raise ValidationError("taskset is forbidden; use llama-native affinity controls")


def validate_direct_target(
    executable: pathlib.Path,
    *,
    profiler: bool = False,
    direct_harness: dict[str, Any] | None = None,
) -> None:
    name = executable.name
    if name in OPAQUE_WRAPPERS:
        raise ValidationError(f"opaque/wrapper target {name!r} is not allowed")
    if not profiler:
        return
    if re.fullmatch(r"llama-(bench|server|cli|perplexity)", name):
        return
    if direct_harness is None:
        raise ValidationError(
            "profiler target must be a direct llama binary or a declared direct local harness"
        )
    if direct_harness.get("kind") != "native_executable":
        raise ValidationError("a profiler direct harness must be a native executable")
    source_files = direct_harness.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValidationError("a direct harness requires source_files with expected SHA-256 values")
    for item in source_files:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ValidationError("each direct-harness source file needs path and sha256")
        validate_sha256(item["sha256"], "direct-harness source sha256")
        if not pathlib.Path(item["path"]).is_absolute():
            raise ManifestError("direct-harness source paths must be absolute")


def option_arities(
    schema: dict[str, Any] | None,
    target_executable: str | pathlib.Path | None = None,
) -> tuple[dict[str, int], bool, int | None]:
    schema = schema or {"builtin": "llama", "allow_positionals": False}
    builtin = schema.get("builtin", "llama")
    if builtin not in ("llama", "none"):
        raise ManifestError(f"unknown CLI schema builtin {builtin!r}")
    if builtin == "llama":
        if target_executable is None:
            options = dict(LLAMA_COMMON_OPTION_ARITY)
        else:
            target_name = pathlib.Path(str(target_executable)).name
            if target_name == "llama-bench":
                options = dict(LLAMA_BENCH_OPTION_ARITY)
            elif re.fullmatch(r"llama-(server|cli|perplexity)", target_name):
                options = dict(LLAMA_COMMON_OPTION_ARITY)
            else:
                raise ManifestError(
                    f"builtin llama CLI schema cannot identify target {target_name!r}; "
                    "use builtin=none with explicit option arities for a harness"
                )
    else:
        options = {}
    for option, raw_arity in schema.get("options", {}).items():
        arity = int(raw_arity)
        if arity not in (0, 1):
            raise ManifestError(f"option {option!r} arity must be 0 or 1")
        options[str(option)] = arity
    allow_unknown = bool(schema.get("allow_unknown_options", False))
    if schema.get("allow_positionals", False):
        positional_limit = schema.get("max_positionals")
        return options, allow_unknown, None if positional_limit is None else int(positional_limit)
    return options, allow_unknown, 0


def validate_argv(
    argv: list[str],
    schema: dict[str, Any] | None = None,
    target_executable: str | pathlib.Path | None = None,
) -> None:
    reject_forbidden_tokens(argv)
    options, allow_unknown, positional_limit = option_arities(schema, target_executable)
    positionals = 0
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            raise ValidationError("'--' positional passthrough is not allowed in validated commands")
        if token.startswith("-") and token != "-":
            option, separator, inline_value = token.partition("=")
            if option not in options:
                if allow_unknown:
                    index += 1
                    continue
                raise ValidationError(f"unknown option {option!r}; declare its arity in cli_schema")
            arity = options[option]
            if arity == 0 and separator:
                raise ValidationError(f"zero-arity option {option} cannot take a value")
            if arity == 1:
                if separator:
                    if not inline_value:
                        raise ValidationError(f"option {option} requires a non-empty value")
                else:
                    if index + 1 >= len(argv):
                        raise ValidationError(f"option {option} requires one value")
                    index += 1
            index += 1
            continue
        positionals += 1
        if positional_limit is not None and positionals > positional_limit:
            raise ValidationError(
                f"unexpected positional argument {token!r}; this often means a value was given to a zero-arity option"
            )
        index += 1


def option_names(
    argv: list[str],
    schema: dict[str, Any] | None = None,
    target_executable: str | pathlib.Path | None = None,
) -> list[str]:
    """Return option names while respecting arity, so negative values are values."""
    options, allow_unknown, _ = option_arities(schema, target_executable)
    names: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("-") and token != "-":
            option, separator, _ = token.partition("=")
            if option in options:
                names.append(option)
                if options[option] == 1 and not separator:
                    index += 1
            elif allow_unknown:
                names.append(option)
        index += 1
    return names


def _forbidden_selection_key(value: Any, path: str = "selection") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {"architecture", "architecture_name", "model_family", "model_name"}:
                raise ManifestError(
                    f"{path}.{key} is forbidden; select by workload, tensor layout, and backend capability"
                )
            _forbidden_selection_key(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_selection_key(child, f"{path}[{index}]")


def _validate_selection(stage: dict[str, Any]) -> None:
    selection = stage.get("selection")
    if not isinstance(selection, dict):
        raise ManifestError(f"stage {stage.get('id')!r} requires a selection object")
    required = {"workload", "tensor_layout", "backend_capabilities", "execution_mode"}
    missing = sorted(required - selection.keys())
    if missing:
        raise ManifestError(f"stage {stage['id']!r} selection is missing {missing}")
    if selection["workload"] not in {"exactness", "prefill", "decode", "mixed"}:
        raise ManifestError("workload must be exactness, prefill, decode, or mixed")
    if not isinstance(selection["tensor_layout"], dict) or not selection["tensor_layout"]:
        raise ManifestError("tensor_layout must be a non-empty property object")
    if not isinstance(selection["backend_capabilities"], list):
        raise ManifestError("backend_capabilities must be an array")
    if not all(isinstance(value, str) and value for value in selection["backend_capabilities"]):
        raise ManifestError("backend_capabilities entries must be non-empty strings")
    if selection["execution_mode"] not in {
        "direct_process",
        "direct_command",
        "cuda_graph_replay",
        "production_binary",
    }:
        raise ManifestError("unknown execution_mode")
    _forbidden_selection_key(selection)
    if selection["execution_mode"] == "cuda_graph_replay":
        capabilities = {str(value) for value in selection["backend_capabilities"]}
        if "cuda_graphs" not in capabilities:
            raise ManifestError("cuda_graph_replay requires the cuda_graphs capability")


def _validate_repetitions(manifest: dict[str, Any]) -> None:
    policy = manifest.get("repetition_policy")
    if not isinstance(policy, dict):
        raise ManifestError("repetition_policy is required")
    if policy.get("minimum_pairs") != 3 or policy.get("maximum_pairs") != 5:
        raise ManifestError("repetition policy must preregister exactly 3 initial and 5 maximum pairs")
    order = policy.get("order")
    if order != ["AB", "BA", "AB", "BA", "AB"]:
        raise ManifestError("repetition order must be the preregistered AB/BA/AB/BA/AB sequence")
    if policy.get("confidence_level") != 0.95:
        raise ManifestError("the current honest paired interval supports confidence_level 0.95")
    rule = policy.get("extension_rule")
    if not isinstance(rule, dict) or rule.get("kind") != "extend_only_if_inconclusive":
        raise ManifestError("extension_rule.kind must be extend_only_if_inconclusive")
    thresholds = rule.get("thresholds")
    required = {"improvement_percent", "regression_percent", "equivalence_percent"}
    if not isinstance(thresholds, dict) or required - thresholds.keys():
        raise ManifestError(f"extension thresholds must include {sorted(required)}")
    for key in required:
        if float(thresholds[key]) < 0:
            raise ManifestError(f"extension threshold {key} must be non-negative")


def _validate_screening_policy(stage: dict[str, Any]) -> None:
    policy = stage.get("screening_policy")
    if policy is None:
        return
    if stage.get("purpose") not in {"smoke", "kernel_screen"}:
        raise ManifestError("screening_policy is limited to smoke and kernel_screen stages")
    if not isinstance(policy, dict) or policy.get("kind") != "single_pair_fail_fast":
        raise ManifestError("screening_policy.kind must be single_pair_fail_fast")
    required = {
        "kind",
        "order",
        "regression_threshold_percent",
        "confidence_claim",
    }
    if set(policy) != required:
        raise ManifestError(f"screening_policy requires exactly {sorted(required)}")
    if policy.get("order") not in {"AB", "BA"}:
        raise ManifestError("single-pair screening order must be AB or BA")
    threshold = policy.get("regression_threshold_percent")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold <= 0:
        raise ManifestError("single-pair screening requires a positive regression_threshold_percent")
    if policy.get("confidence_claim") != "none":
        raise ManifestError("single-pair screening must declare confidence_claim=none")
    if "decision_policy" in stage:
        raise ManifestError(
            "single-pair screening cannot declare a statistical decision_policy"
        )


def _validate_profiler_details(
    config: dict[str, Any], stage: dict[str, Any], *, label: str
) -> None:
    if stage.get("resource") != "gpu":
        raise ManifestError(f"{label} must target a GPU stage")
    selector = config.get("selector")
    if not isinstance(selector, dict) or not selector.get("kernel_regex"):
        raise ManifestError(f"{label} requires a discovery selector with kernel_regex")
    selector_keys = {"kernel_regex", "graph_only", "grid", "block", "occurrence_policy"}
    if set(selector) - selector_keys:
        unknown = sorted(set(selector) - selector_keys)
        raise ManifestError(f"{label} selector has unknown keys {unknown}")
    try:
        re.compile(str(selector["kernel_regex"]))
    except re.error as error:
        raise ManifestError(f"{label} kernel_regex is invalid: {error}") from error
    if "graph_only" in selector and not isinstance(selector["graph_only"], bool):
        raise ManifestError(f"{label} selector.graph_only must be boolean")
    for shape_name in ("grid", "block"):
        shape = selector.get(shape_name)
        if shape is not None and (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in shape
            )
        ):
            raise ManifestError(f"{label} selector.{shape_name} must have three positive integers")
    occurrence_policy = selector.get("occurrence_policy", {"kind": "all"})
    if (
        not isinstance(occurrence_policy, dict)
        or set(occurrence_policy) != {"kind"}
        or occurrence_policy.get("kind") not in {"all", "first", "middle", "last"}
    ):
        raise ManifestError(
            f"{label} selector occurrence_policy.kind must be all, first, middle, or last"
        )
    graph_trace = config.get("cuda_graph_trace")
    if not isinstance(graph_trace, dict) or graph_trace.get("applicability") not in {
        "required",
        "not_applicable",
    }:
        raise ManifestError(
            f"{label}.cuda_graph_trace must preregister required or not_applicable"
        )
    selection = stage["selection"]
    graph_replay = selection.get("cuda_graph_replay", {})
    graph_applicable = (
        selection.get("execution_mode") == "cuda_graph_replay"
        or (
            isinstance(graph_replay, dict)
            and graph_replay.get("applicability") == "required"
        )
        or selector.get("graph_only")
    )
    if graph_applicable and graph_trace["applicability"] != "required":
        raise ManifestError(
            "CUDA-graph-replay or graph-only profiling requires graph-node tracing"
        )
    if graph_trace["applicability"] == "required":
        if set(graph_trace) != {"applicability", "granularity", "launch_origin"}:
            raise ManifestError("required CUDA graph tracing has unknown or missing fields")
        if graph_trace.get("granularity") != "node":
            raise ManifestError("required CUDA graph tracing must use granularity=node")
        if graph_trace.get("launch_origin") not in {"host-only", "host-and-device"}:
            raise ManifestError(
                "required CUDA graph tracing needs host-only or host-and-device launch_origin"
            )
    else:
        if set(graph_trace) != {"applicability", "reason"} or not graph_trace.get("reason"):
            raise ManifestError("not-applicable CUDA graph tracing requires exactly a reason")
    memory_evidence = config.get("memory_evidence")
    if memory_evidence is not None:
        if not isinstance(memory_evidence, dict) or set(memory_evidence) != {
            "mode",
            "categories",
        }:
            raise ManifestError(
                f"{label}.memory_evidence requires exactly mode and categories"
            )
        if memory_evidence.get("mode") not in {"required", "optional"}:
            raise ManifestError(
                f"{label}.memory_evidence.mode must be required or optional"
            )
        categories = memory_evidence.get("categories")
        if (
            not isinstance(categories, list)
            or not categories
            or any(not isinstance(item, str) for item in categories)
            or len(set(categories)) != len(categories)
            or not set(categories).issubset(MEMORY_EVIDENCE_CATEGORIES)
        ):
            raise ManifestError(
                f"{label}.memory_evidence.categories must be a non-empty unique subset of "
                f"{sorted(MEMORY_EVIDENCE_CATEGORIES)}"
            )
    metrics = config.get("metrics", [])
    if not isinstance(metrics, list) or any(
        not isinstance(value, str) or not value for value in metrics
    ):
        raise ManifestError(f"{label} metrics must be non-empty strings")
    timeout = config.get("timeout_seconds")
    if timeout is not None and (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not (0 < float(timeout) <= 86400)
    ):
        raise ManifestError(f"{label} timeout_seconds must be in (0, 86400]")


def _stage_screen_ids(stage: dict[str, Any]) -> set[str]:
    return {
        str(item["id"])
        for item in stage.get("screens", [{"id": "default"}])
    }


def variant_executables(variant: dict[str, Any]) -> dict[str, dict[str, Any]]:
    executables = variant.get("executables")
    if not isinstance(executables, dict) or not executables:
        raise ManifestError("variant executables must be a non-empty role object")
    return executables


def validate_artifact_directory_isolation(
    manifest: dict[str, Any], directory: pathlib.Path
) -> None:
    """Fail closed unless one exact artifact directory satisfies source isolation.

    The default policy keeps artifacts physically outside every declared source
    worktree.  The opt-in policy permits an in-source directory only when Git
    reports that directory itself ignored from every variant source root that
    contains it.  Checking the exact path is intentional: an ignored synthetic
    child says nothing about whether real files directly below the directory
    would be untracked.
    """
    artifact_root = pathlib.Path(str(manifest["artifact_root"])).expanduser().resolve()
    resolved_directory = directory.expanduser().resolve()
    try:
        resolved_directory.relative_to(artifact_root)
    except ValueError as error:
        raise ManifestError(
            f"artifact directory must remain below artifact_root: {resolved_directory}"
        ) from error

    policy = str(manifest.get("artifact_root_policy", "outside_sources"))
    seen_roots: set[pathlib.Path] = set()
    for variant in manifest["variants"].values():
        source_root = pathlib.Path(str(variant["source_root"])).expanduser().resolve()
        if source_root in seen_roots:
            continue
        seen_roots.add(source_root)
        try:
            relative = resolved_directory.relative_to(source_root)
        except ValueError:
            continue
        if policy != "git_ignored_inside_sources":
            raise ManifestError(
                "artifact_root must be outside every variant source repository unless "
                "artifact_root_policy=git_ignored_inside_sources"
            )
        if not relative.parts or ".git" in relative.parts:
            raise ManifestError("an in-source artifact directory cannot be the source root or .git")
        ignored = subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "check-ignore",
                "--quiet",
                "--no-index",
                "--",
                f"{resolved_directory}{os.sep}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if ignored.returncode == 1:
            raise ManifestError(
                "artifact_root_policy=git_ignored_inside_sources requires the exact "
                f"artifact directory to be Git-ignored by every containing variant "
                f"source repository: {resolved_directory} is not ignored by {source_root}"
            )
        if ignored.returncode != 0:
            detail = ignored.stderr.strip() or f"exit status {ignored.returncode}"
            raise ManifestError(
                f"cannot verify Git-ignore isolation for {resolved_directory} in "
                f"{source_root}: {detail}"
            )


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION}")
    study = manifest.get("study")
    if not isinstance(study, dict):
        raise ManifestError("study object is required")
    safe_slug(str(study.get("id", "")))
    budget = study.get("early_decision_target_seconds")
    if budget != {"minimum": 30, "maximum": 90}:
        raise ManifestError("study must state the 30-90 second early-decision target")
    artifact_root = manifest.get("artifact_root")
    if not isinstance(artifact_root, str) or not pathlib.Path(artifact_root).is_absolute():
        raise ManifestError("artifact_root must be an absolute path")
    artifact_root_policy = manifest.get("artifact_root_policy", "outside_sources")
    if artifact_root_policy not in {"outside_sources", "git_ignored_inside_sources"}:
        raise ManifestError(
            "artifact_root_policy must be outside_sources or git_ignored_inside_sources"
        )

    variants = manifest.get("variants")
    if not isinstance(variants, dict) or set(variants) != {"baseline", "candidate"}:
        raise ManifestError("variants must contain exactly baseline and candidate")
    for name, variant in variants.items():
        if not isinstance(variant, dict):
            raise ManifestError(f"variant {name} must be an object")
        for key in ("source_root", "expected_commit", "tree_policy"):
            if not variant.get(key):
                raise ManifestError(f"variant {name} requires {key}")
        if not pathlib.Path(variant["source_root"]).is_absolute():
            raise ManifestError(f"variant {name} source_root must be absolute")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", str(variant["expected_commit"])):
            raise ManifestError(f"variant {name} expected_commit must be a full 40-digit commit")
        if variant["tree_policy"] not in ("clean", "expected_dirty"):
            raise ManifestError(f"variant {name} tree_policy must be clean or expected_dirty")
        if variant["tree_policy"] == "expected_dirty" and not variant.get("expected_dirty_fingerprint"):
            raise ManifestError(f"variant {name} expected_dirty requires expected_dirty_fingerprint")
        if variant["tree_policy"] == "expected_dirty":
            validate_sha256(
                variant["expected_dirty_fingerprint"],
                f"variant {name} expected_dirty_fingerprint",
            )
        for role, executable in variant_executables(variant).items():
            safe_slug(str(role))
            if not isinstance(executable, dict):
                raise ManifestError(f"variant {name} executable role {role} must be an object")
            if not executable.get("path") or not executable.get("expected_sha256"):
                raise ManifestError(f"variant {name} executable role {role} requires path and expected_sha256")
            if not pathlib.Path(executable["path"]).is_absolute():
                raise ManifestError(f"variant {name} executable role {role} path must be absolute")
            validate_sha256(executable["expected_sha256"], f"variant {name}/{role} expected_sha256")
            build = executable.get("build")
            if not isinstance(build, dict) or build.get("mode") not in ("cmake_required", "not_applicable"):
                raise ManifestError(
                    f"variant {name} executable role {role} build.mode must be cmake_required or not_applicable"
                )
            if build["mode"] == "not_applicable" and not build.get("reason"):
                raise ManifestError(f"variant {name} executable role {role} non-CMake build requires a reason")
            sidecar = executable.get("provenance_sidecar")
            if build["mode"] == "cmake_required" and not isinstance(sidecar, dict):
                raise ManifestError(
                    f"variant {name} executable role {role} CMake build requires provenance_sidecar"
                )
            if sidecar is not None:
                if not isinstance(sidecar, dict) or set(sidecar) != {"path", "sha256"}:
                    raise ManifestError(
                        f"variant {name} executable role {role} provenance_sidecar requires only path and sha256"
                    )
                if not pathlib.Path(sidecar["path"]).is_absolute():
                    raise ManifestError(
                        f"variant {name} executable role {role} provenance_sidecar path must be absolute"
                    )
                validate_sha256(sidecar["sha256"], f"variant {name}/{role} provenance_sidecar sha256")
    validate_artifact_directory_isolation(manifest, pathlib.Path(artifact_root))

    inputs = manifest.get("inputs", [])
    if not isinstance(inputs, list):
        raise ManifestError("inputs must be an array")
    for item in inputs:
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise ManifestError("every input requires path and sha256")
        if not pathlib.Path(item["path"]).is_absolute():
            raise ManifestError("every input path must be absolute")
        validate_sha256(item["sha256"], f"input {item['path']} sha256")

    _validate_repetitions(manifest)
    stages = manifest.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ManifestError("stages must be a non-empty array")
    seen: set[str] = set()
    purpose_counts = {purpose: 0 for purpose in PURPOSE_ORDER}
    smoke_workloads: set[str] = set()
    screening_orders: list[str] = []
    last_order = -1
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ManifestError("every stage must be an object")
        stage_id = safe_slug(str(stage.get("id", "")))
        if stage_id in seen:
            raise ManifestError(f"duplicate stage id {stage_id!r}")
        seen.add(stage_id)
        purpose = stage.get("purpose")
        if purpose not in PURPOSE_ORDER:
            raise ManifestError(f"stage {stage_id!r} has unknown purpose {purpose!r}")
        _validate_screening_policy(stage)
        if stage.get("screening_policy") is not None:
            screening_orders.append(stage["screening_policy"]["order"])
        purpose_counts[purpose] += 1
        order = PURPOSE_ORDER[purpose]
        if order < last_order:
            raise ManifestError("validation stages must follow the declared ladder order")
        last_order = order
        if index == 0 and purpose != "exactness":
            raise ManifestError("the first stage must be exactness")
        if stage.get("resource") not in ("cpu", "gpu"):
            raise ManifestError(f"stage {stage_id!r} resource must be cpu or gpu")
        timeout = stage.get("timeout_seconds")
        if not isinstance(timeout, (int, float)) or not (0 < float(timeout) <= 86400):
            raise ManifestError(f"stage {stage_id!r} requires timeout_seconds in (0, 86400]")
        if not stage.get("progress"):
            raise ManifestError(f"stage {stage_id!r} must declare a visible progress mechanism")
        _validate_selection(stage)
        selection = stage["selection"]
        if purpose == "exactness" and selection["workload"] != "exactness":
            raise ManifestError("the exactness gate selection.workload must be exactness")
        if purpose == "smoke":
            if selection["workload"] not in {"prefill", "decode"}:
                raise ManifestError("smoke stages must select prefill or decode")
            if selection.get("duration_class") != "short":
                raise ManifestError("smoke stages must declare duration_class=short")
            smoke_workloads.add(selection["workload"])
            if selection["workload"] == "decode":
                graph = selection.get("cuda_graph_replay")
                if not isinstance(graph, dict) or graph.get("applicability") not in {
                    "required",
                    "not_applicable",
                }:
                    raise ManifestError(
                        "decode smoke must preregister CUDA-graph applicability as required or not_applicable"
                    )
                if graph["applicability"] == "required" and selection["execution_mode"] != "cuda_graph_replay":
                    raise ManifestError("applicable decode CUDA graphs must be screened by replay")
                if graph["applicability"] == "not_applicable" and not graph.get("reason"):
                    raise ManifestError("not-applicable decode CUDA graphs require a reason")
        if purpose == "production_confirmation":
            if selection["execution_mode"] != "production_binary":
                raise ManifestError("production confirmation must execute a production binary")
            if not isinstance(selection.get("context_depth_tokens"), int) or selection["context_depth_tokens"] <= 0:
                raise ManifestError("production confirmation requires a positive selected context depth")
        if purpose == "long_context_acceptance":
            if selection.get("acceptance_class") != "long_context":
                raise ManifestError("final acceptance must declare acceptance_class=long_context")
            if not isinstance(selection.get("context_depth_tokens"), int) or selection["context_depth_tokens"] <= 0:
                raise ManifestError("long-context acceptance requires a positive context depth")
        command = stage.get("command")
        if not isinstance(command, dict):
            raise ManifestError(f"stage {stage_id!r} requires command")
        if command.get("cli_schema", {}).get("allow_unknown_options", False):
            raise ManifestError("allow_unknown_options is not fail-closed; declare every option arity")
        executable_role = str(command.get("executable_role", "default"))
        target_paths: dict[str, str] = {}
        for variant_name, variant in variants.items():
            if executable_role not in variant_executables(variant):
                raise ManifestError(
                    f"stage {stage_id!r} executable role {executable_role!r} is absent from {variant_name}"
                )
            target_paths[variant_name] = str(
                variant_executables(variant)[executable_role]["path"]
            )
        common_args = [str(value) for value in command.get("common_args", [])]
        baseline_args = [str(value) for value in command.get("baseline_args", [])]
        candidate_args = [str(value) for value in command.get("candidate_args", [])]
        for variant_name in ("baseline", "candidate"):
            delta = baseline_args if variant_name == "baseline" else candidate_args
            validate_argv(
                common_args + delta,
                command.get("cli_schema"),
                target_paths[variant_name],
            )
        baseline_environment = {
            str(key): str(value) for key, value in command.get("baseline_environment", {}).items()
        }
        candidate_environment = {
            str(key): str(value) for key, value in command.get("candidate_environment", {}).items()
        }
        if baseline_args != candidate_args or baseline_environment != candidate_environment:
            controlled = command.get("controlled_delta")
            if not isinstance(controlled, dict) or not controlled.get("reason"):
                raise ManifestError(f"stage {stage_id!r} A/B differences require a controlled_delta reason")
            if controlled.get("baseline_args", []) != baseline_args or controlled.get(
                "candidate_args", []
            ) != candidate_args:
                raise ManifestError("controlled_delta must reproduce the exact baseline/candidate args")
            if controlled.get("baseline_environment", {}) != baseline_environment or controlled.get(
                "candidate_environment", {}
            ) != candidate_environment:
                raise ManifestError("controlled_delta must reproduce exact baseline/candidate environments")
            observed_options = sorted(
                set(
                    option_names(
                        baseline_args,
                        command.get("cli_schema"),
                        target_paths["baseline"],
                    )
                )
                | set(
                    option_names(
                        candidate_args,
                        command.get("cli_schema"),
                        target_paths["candidate"],
                    )
                )
            )
            if sorted(controlled.get("allowed_options", [])) != observed_options:
                raise ManifestError("controlled_delta.allowed_options must exactly name every delta option")
            forbidden = sorted(set(observed_options) & IMMUTABLE_AB_OPTIONS)
            if forbidden:
                raise ManifestError(
                    f"stage {stage_id!r} changes immutable A/B workload settings in variant args: {forbidden}"
                )
            observed_environment = sorted(set(baseline_environment) | set(candidate_environment))
            if sorted(controlled.get("allowed_environment", [])) != observed_environment:
                raise ManifestError(
                    "controlled_delta.allowed_environment must exactly name every delta environment variable"
                )
        screens = stage.get("screens", [{"id": "default", "args": [], "weight": 1.0}])
        if not isinstance(screens, list) or not screens:
            raise ManifestError(f"stage {stage_id!r} screens must be non-empty")
        for screen in screens:
            safe_slug(str(screen.get("id", "")))
            if float(screen.get("weight", 0)) <= 0:
                raise ManifestError("every screen weight must be positive")
            screen_args = [str(value) for value in screen.get("args", [])]
            validate_argv(
                common_args + screen_args + baseline_args,
                command.get("cli_schema"),
                target_paths["baseline"],
            )
            validate_argv(
                common_args + screen_args + candidate_args,
                command.get("cli_schema"),
                target_paths["candidate"],
            )
        if purpose == "kernel_screen":
            if selection["workload"] != "prefill" or selection["execution_mode"] != "direct_command":
                raise ManifestError("kernel screens must use a direct-command prefill workload")
            spans = {screen.get("span_class") for screen in screens}
            if spans != {"low", "mid", "high"}:
                raise ManifestError("kernel_screen must preregister low, mid, and high spans")
            aggregation = stage.get("aggregation")
            if not isinstance(aggregation, dict) or aggregation.get("mode") not in (
                "weighted_mean",
                "weighted_harmonic",
            ):
                raise ManifestError("kernel_screen requires weighted_mean or weighted_harmonic aggregation")
            if aggregation.get("weight_source") != "real_ubatch_geometry":
                raise ManifestError("kernel_screen weights must come from real_ubatch_geometry")
            for screen in screens:
                if not isinstance(screen.get("span_tokens"), int) or screen["span_tokens"] <= 0:
                    raise ManifestError("kernel screens require a positive span_tokens")
                occurrences = screen.get("ubatch_occurrences")
                if not isinstance(occurrences, int) or occurrences <= 0:
                    raise ManifestError("kernel screens require positive ubatch_occurrences")
                if float(screen["weight"]) != float(occurrences):
                    raise ManifestError("kernel screen weight must equal its real ubatch_occurrences")
        if purpose != "exactness":
            if not isinstance(stage.get("metric"), dict):
                raise ManifestError(f"performance stage {stage_id!r} requires a metric extractor")
            if stage.get("screening_policy") is None:
                policy = stage.get("decision_policy")
                if not isinstance(policy, dict):
                    raise ManifestError(
                        f"performance stage {stage_id!r} requires an explicit decision_policy"
                    )
                acceptable = policy.get("acceptable_decisions")
                if (
                    not isinstance(acceptable, list)
                    or not acceptable
                    or not set(acceptable).issubset({"improvement", "equivalent"})
                ):
                    raise ManifestError(
                        "decision_policy.acceptable_decisions must be a non-empty subset of "
                        "improvement/equivalent"
                    )
                if policy.get("regression") != "fail":
                    raise ManifestError("decision_policy.regression must be fail")
                if policy.get("inconclusive_after_maximum") != "unresolved_fail":
                    raise ManifestError(
                        "decision_policy.inconclusive_after_maximum must be unresolved_fail"
                    )
        if purpose == "exactness" and stage.get("comparison", {}).get("mode") not in (
            "stdout_sha256",
            "file_sha256",
        ):
            raise ManifestError("exactness stage requires stdout_sha256 or file_sha256 comparison")
    if stages[-1].get("purpose") != "long_context_acceptance" or not stages[-1].get("mandatory"):
        raise ManifestError("the final stage must be a mandatory long_context_acceptance gate")
    for purpose in ("exactness", "kernel_screen", "production_confirmation", "long_context_acceptance"):
        if purpose_counts[purpose] != 1:
            raise ManifestError(f"the ladder requires exactly one {purpose} stage")
    if smoke_workloads != {"prefill", "decode"}:
        raise ManifestError("the ladder requires short prefill and short decode smoke stages")
    if any(first == second for first, second in zip(screening_orders, screening_orders[1:])):
        raise ManifestError("adjacent single-pair screening stages must alternate AB/BA order")

    profiler = manifest.get("profiler")
    if profiler is not None:
        if not isinstance(profiler, dict):
            raise ManifestError("profiler must be an object")
        profiler_keys = {
            "pattern",
            "profile_repetitions",
            "stage",
            "variant",
            "screen",
            "timeout_seconds",
            "selector",
            "cuda_graph_trace",
            "memory_evidence",
            "metrics",
        }
        profiler_required = {
            "pattern",
            "profile_repetitions",
            "stage",
            "variant",
            "selector",
            "cuda_graph_trace",
        }
        if profiler_required - set(profiler) or set(profiler) - profiler_keys:
            raise ManifestError("profiler has unknown or missing fields")
        if profiler.get("pattern") != "one_nsys_discovery_then_one_filtered_ncu":
            raise ManifestError("profiler pattern must preregister one NSYS discovery then one filtered NCU")
        if profiler.get("profile_repetitions") != 1:
            raise ManifestError("profiler capture must run once per preregistered stage")
        stage_id = profiler.get("stage")
        if stage_id not in seen:
            raise ManifestError(f"profiler references unknown stage {stage_id!r}")
        profiled_stage = stage_by_id(manifest, str(stage_id))
        if profiled_stage["purpose"] != "production_confirmation":
            raise ManifestError("profiler discovery/capture must target production_confirmation")
        target_variant = profiler.get("variant")
        if target_variant not in variants:
            raise ManifestError("profiler variant must be baseline or candidate")
        if profiler.get("screen", "default") not in _stage_screen_ids(profiled_stage):
            raise ManifestError("profiler screen is absent from its target stage")
        _validate_profiler_details(profiler, profiled_stage, label="profiler")

    diagnostics = manifest.get("early_diagnostics", [])
    if not isinstance(diagnostics, list):
        raise ManifestError("early_diagnostics must be an array")
    diagnostic_ids: set[str] = set()
    diagnostic_stages: set[str] = set()
    for config in diagnostics:
        if not isinstance(config, dict):
            raise ManifestError("each early diagnostic must be an object")
        diagnostic_required = {
            "id",
            "trigger",
            "on_clear",
            "evidence_claim",
            "pattern",
            "profile_repetitions",
            "stage",
            "variants",
            "screen_selection",
            "selector",
            "cuda_graph_trace",
        }
        diagnostic_keys = diagnostic_required | {
            "timeout_seconds",
            "memory_evidence",
            "metrics",
        }
        if diagnostic_required - set(config) or set(config) - diagnostic_keys:
            raise ManifestError("early diagnostic has unknown or missing fields")
        diagnostic_id = safe_slug(str(config.get("id", "")))
        if diagnostic_id in diagnostic_ids:
            raise ManifestError(f"duplicate early diagnostic id {diagnostic_id!r}")
        diagnostic_ids.add(diagnostic_id)
        if config.get("trigger") != "regression_signal_only":
            raise ManifestError("early diagnostic trigger must be regression_signal_only")
        if config.get("on_clear") != "skip":
            raise ManifestError("early diagnostic on_clear policy must be skip")
        if config.get("evidence_claim") != "diagnostic_only":
            raise ManifestError("early diagnostic evidence_claim must be diagnostic_only")
        if config.get("pattern") != "independent_nsys_then_filtered_ncu_per_variant":
            raise ManifestError(
                "early diagnostic must independently discover and capture each variant"
            )
        if config.get("profile_repetitions") != 1:
            raise ManifestError("early diagnostic profiling must run once per variant")
        if config.get("variants") != ["baseline", "candidate"]:
            raise ManifestError("early diagnostic variants must be baseline then candidate")
        stage_id = config.get("stage")
        if stage_id not in seen:
            raise ManifestError(f"early diagnostic references unknown stage {stage_id!r}")
        if stage_id in diagnostic_stages:
            raise ManifestError("only one early diagnostic is allowed per screening stage")
        diagnostic_stages.add(str(stage_id))
        profiled_stage = stage_by_id(manifest, str(stage_id))
        if profiled_stage.get("purpose") not in {"smoke", "kernel_screen"}:
            raise ManifestError("early diagnostics are limited to smoke and kernel_screen stages")
        if profiled_stage.get("screening_policy") is None:
            raise ManifestError("early diagnostics require a single-pair screening_policy")
        screen_selection = config.get("screen_selection")
        if not isinstance(screen_selection, dict) or screen_selection.get("kind") not in {
            "fixed",
            "largest_observed_regression",
        }:
            raise ManifestError(
                "early diagnostic screen_selection.kind must be fixed or "
                "largest_observed_regression"
            )
        if screen_selection["kind"] == "fixed":
            if set(screen_selection) != {"kind", "screen"}:
                raise ManifestError(
                    "fixed diagnostic screen selection requires exactly kind and screen"
                )
            if screen_selection.get("screen") not in _stage_screen_ids(profiled_stage):
                raise ManifestError("fixed diagnostic screen is absent from its target stage")
        elif set(screen_selection) != {"kind"}:
            raise ManifestError(
                "largest-observed-regression screen selection requires exactly kind"
            )
        _validate_profiler_details(
            config, profiled_stage, label=f"early_diagnostics[{diagnostic_id}]"
        )


def load_and_validate_manifest(path: pathlib.Path) -> tuple[dict[str, Any], str]:
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise ManifestError("manifest root must be an object")
    validate_manifest(manifest)
    return manifest, sha256_file(path)


def _git_text(root: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProvenanceError(f"git {' '.join(args)} failed for {root}: {result.stderr.strip()}")
    return result.stdout


def git_snapshot(root: pathlib.Path) -> dict[str, Any]:
    requested = root.expanduser().resolve()
    if not requested.is_dir():
        raise ProvenanceError(f"source_root is not a directory: {requested}")
    resolved = pathlib.Path(_git_text(requested, "rev-parse", "--show-toplevel").strip()).resolve()
    if resolved != requested:
        raise ProvenanceError(
            "source_root must be the exact Git worktree root: "
            f"declared {requested}, git reported {resolved}"
        )
    status = _git_text(resolved, "status", "--porcelain=v1", "--untracked-files=all")
    unstaged = subprocess.run(
        ["git", "-C", str(resolved), "diff", "--binary", "HEAD", "--"],
        check=True,
        capture_output=True,
    ).stdout
    staged = subprocess.run(
        ["git", "-C", str(resolved), "diff", "--binary", "--cached", "HEAD", "--"],
        check=True,
        capture_output=True,
    ).stdout
    untracked_raw = subprocess.run(
        ["git", "-C", str(resolved), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    untracked: list[dict[str, Any]] = []
    for raw in untracked_raw.split(b"\0"):
        if not raw:
            continue
        relative = pathlib.Path(os.fsdecode(raw))
        source = resolved / relative
        if source.is_file():
            untracked.append({"path": str(relative), "sha256": sha256_file(source)})
    fingerprint_body = {
        "status": status,
        "unstaged_sha256": sha256_bytes(unstaged),
        "staged_sha256": sha256_bytes(staged),
        "untracked": untracked,
    }
    return {
        "root": str(resolved),
        "head": _git_text(resolved, "rev-parse", "HEAD").strip(),
        "head_tree": _git_text(resolved, "rev-parse", "HEAD^{tree}").strip(),
        "branch": _git_text(resolved, "branch", "--show-current").strip(),
        **fingerprint_body,
        "dirty_fingerprint": canonical_sha256(fingerprint_body),
    }


def git_source_file_manifest(root: pathlib.Path) -> dict[str, Any]:
    """Hash every tracked or untracked, non-ignored path in one exact worktree."""
    resolved = root.expanduser().resolve()
    discovered = pathlib.Path(_git_text(resolved, "rev-parse", "--show-toplevel").strip()).resolve()
    if discovered != resolved:
        raise ProvenanceError(
            "source_root must be the exact Git worktree root before source hashing: "
            f"declared {resolved}, git reported {discovered}"
        )
    listed = subprocess.run(
        [
            "git",
            "-C",
            str(resolved),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=True,
        capture_output=True,
    ).stdout
    records: list[dict[str, Any]] = []
    for raw in sorted({item for item in listed.split(b"\0") if item}):
        relative = pathlib.Path(os.fsdecode(raw))
        source = resolved / relative
        if source.is_symlink():
            target = os.fsencode(os.readlink(source))
            record = {
                "path": str(relative),
                "kind": "symlink",
                "sha256": sha256_bytes(target),
            }
        elif source.is_file():
            record = {
                "path": str(relative),
                "kind": "file",
                "sha256": sha256_file(source),
            }
        elif source.is_dir():
            head = _git_text(source, "rev-parse", "HEAD").strip()
            record = {
                "path": str(relative),
                "kind": "gitlink",
                "head": head,
                "sha256": sha256_bytes(head.encode("ascii")),
            }
        else:
            record = {
                "path": str(relative),
                "kind": "missing",
                "sha256": None,
            }
        records.append(record)
    return {
        "count": len(records),
        "sha256": canonical_sha256(records),
        "files": records,
    }


def _find_cmake_cache(executable: pathlib.Path, build: dict[str, Any]) -> pathlib.Path | None:
    if build.get("cache"):
        candidate = pathlib.Path(build["cache"]).expanduser().resolve()
        return candidate if candidate.is_file() else None
    for parent in executable.parents:
        candidate = parent / "CMakeCache.txt"
        if candidate.is_file():
            return candidate
    return None


def cmake_snapshot(
    executable: pathlib.Path,
    build: dict[str, Any],
    source_root: pathlib.Path | None = None,
) -> dict[str, Any]:
    if build["mode"] == "not_applicable":
        return {"mode": "not_applicable", "reason": build["reason"]}
    cache = _find_cmake_cache(executable, build)
    if cache is None:
        raise ProvenanceError(f"no discoverable CMakeCache.txt for {executable}")
    entries: dict[str, str] = {}
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        entries[key_type.split(":", 1)[0]] = value
    expected = {str(key): str(value) for key, value in build.get("expected_options", {}).items()}
    mismatches = {
        key: {"expected": value, "observed": entries.get(key)}
        for key, value in expected.items()
        if entries.get(key) != value
    }
    if mismatches:
        raise ProvenanceError(f"CMake option mismatch for {executable}: {mismatches}")
    home_directory = entries.get("CMAKE_HOME_DIRECTORY")
    if source_root is not None:
        expected_root = source_root.expanduser().resolve()
        if not home_directory:
            raise ProvenanceError(f"CMake cache has no CMAKE_HOME_DIRECTORY: {cache}")
        observed_home = pathlib.Path(home_directory).expanduser()
        if not observed_home.is_absolute() or observed_home.resolve() != expected_root:
            raise ProvenanceError(
                "CMAKE_HOME_DIRECTORY does not match source_root: "
                f"cache {cache} reports {home_directory!r}, source_root is {expected_root}"
            )
    cache_directory = entries.get("CMAKE_CACHEFILE_DIR")
    if cache_directory:
        observed_cache_directory = pathlib.Path(cache_directory).expanduser()
        if (
            not observed_cache_directory.is_absolute()
            or observed_cache_directory.resolve() != cache.parent.resolve()
        ):
            raise ProvenanceError(
                "CMAKE_CACHEFILE_DIR does not match the selected cache location: "
                f"cache {cache} reports {cache_directory!r}"
            )
    return {
        "mode": "cmake",
        "cache": str(cache),
        "cache_sha256": sha256_file(cache),
        "cache_stat": stat_identity(cache),
        "home_directory": home_directory,
        "cache_directory": cache_directory,
        "entries": entries,
    }


def binary_snapshot(executable: pathlib.Path) -> dict[str, Any]:
    resolved = executable.expanduser().resolve()
    if not resolved.is_file():
        raise ProvenanceError(f"executable is missing: {resolved}")
    if not os.access(resolved, os.X_OK):
        raise ProvenanceError(f"executable is not executable: {resolved}")
    record: dict[str, Any] = {
        "path": str(executable),
        "resolved_path": str(resolved),
        **stat_identity(resolved),
        "sha256": sha256_file(resolved),
    }
    prefix = resolved.read_bytes()[:4]
    if prefix == b"\x7fELF":
        ldd = command_capture(["ldd", str(resolved)])
        if ldd.get("returncode") != 0:
            raise ProvenanceError(f"ldd failed for {resolved}")
        libraries: list[dict[str, Any]] = []
        seen: set[pathlib.Path] = set()
        for line in str(ldd.get("stdout", "")).splitlines():
            match = re.search(r"(?:=>\s+)?(/[^\s]+)\s+\(0x", line)
            if match is None:
                continue
            library = pathlib.Path(match.group(1)).resolve()
            if library in seen or not library.is_file():
                continue
            seen.add(library)
            libraries.append(
                {"path": str(library), **stat_identity(library), "sha256": sha256_file(library)}
            )
        record["format"] = "elf"
        # Raw ldd output contains ASLR load addresses, so retaining it in the
        # provenance identity would make every resume differ despite identical
        # binaries and libraries.  Preserve the stable invocation/result and
        # the fully hashed resolved library inventory instead.
        record["ldd"] = {
            "argv": ldd["argv"],
            "returncode": ldd["returncode"],
            "resolved_library_count": len(libraries),
        }
        record["libraries"] = libraries
    else:
        with resolved.open("rb") as source:
            first_line = source.readline(4096).decode("utf-8", errors="replace").rstrip()
        record["format"] = "script_or_other"
        record["shebang"] = first_line if first_line.startswith("#!") else None
    return record


def capture_cmake_build_provenance(
    source_root: pathlib.Path,
    executable: pathlib.Path,
    cache: pathlib.Path,
) -> dict[str, Any]:
    resolved_root = source_root.expanduser().resolve()
    resolved_executable = executable.expanduser().resolve()
    resolved_cache = cache.expanduser().resolve()
    source_snapshot = git_snapshot(resolved_root)
    source_files = git_source_file_manifest(resolved_root)
    source_after_hashing = git_snapshot(resolved_root)
    if canonical_sha256(source_snapshot) != canonical_sha256(source_after_hashing):
        raise ProvenanceError("source identity changed while registering the CMake build")
    source = {
        **source_after_hashing,
        "source_files": source_files,
    }
    identity = {
        "schema_version": BUILD_PROVENANCE_SCHEMA_VERSION,
        "kind": BUILD_PROVENANCE_KIND,
        "source": source,
        "executable": binary_snapshot(resolved_executable),
        "build": cmake_snapshot(
            resolved_executable,
            {
                "mode": "cmake_required",
                "cache": str(resolved_cache),
                "expected_options": {},
            },
            resolved_root,
        ),
    }
    return {
        **identity,
        "registered_utc": utc_now(),
        "identity_fingerprint": canonical_sha256(identity),
    }


def verify_cmake_build_provenance(
    name: str,
    role: str,
    sidecar_spec: dict[str, Any],
    source: dict[str, Any],
    source_files: dict[str, Any],
    binary: dict[str, Any],
    build: dict[str, Any],
) -> dict[str, Any]:
    sidecar_path = pathlib.Path(sidecar_spec["path"]).expanduser().resolve()
    if not sidecar_path.is_file():
        raise ProvenanceError(f"{name}/{role} build provenance sidecar is missing: {sidecar_path}")
    observed_sha256 = sha256_file(sidecar_path)
    if observed_sha256 != sidecar_spec["sha256"]:
        raise ProvenanceError(
            f"{name}/{role} build provenance sidecar SHA-256 mismatch: "
            f"expected {sidecar_spec['sha256']}, observed {observed_sha256}"
        )
    try:
        sidecar = load_json(sidecar_path)
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError(
            f"{name}/{role} build provenance sidecar is unreadable: {sidecar_path}: {error}"
        ) from error
    if not isinstance(sidecar, dict):
        raise ProvenanceError(f"{name}/{role} build provenance sidecar must be an object")
    expected_fields = {
        "schema_version",
        "kind",
        "source",
        "executable",
        "build",
        "registered_utc",
        "identity_fingerprint",
    }
    if set(sidecar) != expected_fields:
        raise ProvenanceError(
            f"{name}/{role} build provenance sidecar has unknown or missing fields"
        )
    if (
        sidecar.get("schema_version") != BUILD_PROVENANCE_SCHEMA_VERSION
        or sidecar.get("kind") != BUILD_PROVENANCE_KIND
    ):
        raise ProvenanceError(f"{name}/{role} build provenance sidecar has an unsupported schema")
    identity = {
        "schema_version": sidecar.get("schema_version"),
        "kind": sidecar.get("kind"),
        "source": sidecar.get("source"),
        "executable": sidecar.get("executable"),
        "build": sidecar.get("build"),
    }
    if sidecar.get("identity_fingerprint") != canonical_sha256(identity):
        raise ProvenanceError(f"{name}/{role} build provenance sidecar fingerprint is invalid")
    current = {
        "source": {**source, "source_files": source_files},
        "executable": binary,
        "build": build,
    }
    for key in ("source", "executable", "build"):
        if canonical_sha256(sidecar.get(key)) != canonical_sha256(current[key]):
            raise ProvenanceError(
                f"{name}/{role} build provenance {key} mismatch; re-register the executable"
            )
    return {
        "path": str(sidecar_path),
        "sha256": observed_sha256,
        **stat_identity(sidecar_path),
        "registered_utc": sidecar.get("registered_utc"),
        "identity_fingerprint": sidecar["identity_fingerprint"],
    }


def verify_variant(name: str, variant: dict[str, Any]) -> dict[str, Any]:
    source_root = pathlib.Path(variant["source_root"]).expanduser().resolve()
    source = git_snapshot(source_root)
    if source["head"] != variant["expected_commit"]:
        raise ProvenanceError(
            f"{name} source commit mismatch: expected {variant['expected_commit']}, observed {source['head']}"
        )
    if variant["tree_policy"] == "clean" and source["status"]:
        raise ProvenanceError(f"{name} source tree is dirty but the manifest requires clean")
    if variant["tree_policy"] == "expected_dirty":
        expected = variant["expected_dirty_fingerprint"]
        if source["dirty_fingerprint"] != expected:
            raise ProvenanceError(
                f"{name} dirty-tree fingerprint mismatch: expected {expected}, observed {source['dirty_fingerprint']}"
            )
    executables: dict[str, Any] = {}
    source_file_manifest: dict[str, Any] | None = None
    for role, executable in variant_executables(variant).items():
        executable_path = pathlib.Path(executable["path"]).expanduser().resolve()
        binary = binary_snapshot(executable_path)
        if binary["sha256"] != executable["expected_sha256"]:
            raise ProvenanceError(
                f"{name}/{role} binary SHA-256 mismatch: expected {executable['expected_sha256']}, "
                f"observed {binary['sha256']}"
            )
        harness = executable.get("direct_harness")
        if harness:
            validate_direct_target(pathlib.Path(binary["resolved_path"]), direct_harness=harness)
            harness_source_files: list[dict[str, Any]] = []
            for item in harness["source_files"]:
                source_path = pathlib.Path(item["path"]).expanduser().resolve()
                if not source_path.is_file() or sha256_file(source_path) != item["sha256"]:
                    raise ProvenanceError(f"direct-harness source identity mismatch: {source_path}")
                harness_source_files.append(
                    {
                        "path": str(source_path),
                        "sha256": item["sha256"],
                        **stat_identity(source_path),
                    }
                )
            harness = {**harness, "source_files": harness_source_files}
        build = cmake_snapshot(executable_path, executable["build"], source_root)
        build_provenance = None
        if executable["build"]["mode"] == "cmake_required":
            if source_file_manifest is None:
                source_file_manifest = git_source_file_manifest(source_root)
            build_provenance = verify_cmake_build_provenance(
                name,
                role,
                executable["provenance_sidecar"],
                source,
                source_file_manifest,
                binary,
                build,
            )
        executables[role] = {
            "binary": binary,
            "build": build,
            "build_provenance": build_provenance,
            "direct_harness": harness,
        }
    return {"source": source, "executables": executables}


def capture_provenance(manifest: dict[str, Any], manifest_sha256: str) -> dict[str, Any]:
    variants = {
        name: verify_variant(name, variant) for name, variant in manifest["variants"].items()
    }
    inputs: list[dict[str, Any]] = []
    for item in manifest.get("inputs", []):
        path = pathlib.Path(item["path"]).expanduser().resolve()
        if not path.is_file():
            raise ProvenanceError(f"input is missing: {path}")
        observed = sha256_file(path)
        if observed != item["sha256"]:
            raise ProvenanceError(
                f"input SHA-256 mismatch for {path}: expected {item['sha256']}, observed {observed}"
            )
        inputs.append(
            {
                "role": item.get("role", "input"),
                "path": str(path),
                **stat_identity(path),
                "sha256": observed,
            }
        )
    runtime_tools: dict[str, Any] = {}
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path:
        resolved_tool = pathlib.Path(nvidia_smi_path).resolve()
        runtime_tools["nvidia-smi"] = {
            "path": str(resolved_tool),
            "sha256": sha256_file(resolved_tool),
            **stat_identity(resolved_tool),
        }
    host = {
        "captured_utc": utc_now(),
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "python": platform.python_version(),
        "load_average": list(os.getloadavg()),
        "proc_loadavg": pathlib.Path("/proc/loadavg").read_text(encoding="utf-8").strip()
        if pathlib.Path("/proc/loadavg").is_file()
        else None,
        "lscpu": command_capture(["lscpu", "--json"]),
        "nvidia_smi_version": command_capture(["nvidia-smi", "--version"]),
        "nsys_version": command_capture(["nsys", "--version"]),
        "ncu_version": command_capture(["ncu", "--version"]),
        "runtime_tools": runtime_tools,
    }
    identity = {
        "manifest_sha256": manifest_sha256,
        "runner_schema_version": SCHEMA_VERSION,
        "variants": variants,
        "inputs": inputs,
        "runtime_tools": runtime_tools,
    }
    body = {
        **identity,
        "host": host,
    }
    body["identity_fingerprint"] = canonical_sha256(identity)
    body["fingerprint"] = canonical_sha256(body)
    return body


def provenance_identity_spec(
    provenance: dict[str, Any],
    selections: list[tuple[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build per-process identity guards from one fully hashed provenance capture.

    Large binaries, libraries, inputs, and harness sources are hashed once when
    provenance is captured.  Fresh processes recheck their size/mtime/device/
    inode tuple; a new invocation or resume recaptures and rehashes provenance.
    """
    if selections is None:
        selections = [
            (variant_name, role)
            for variant_name, variant in provenance["variants"].items()
            for role in variant["executables"]
        ]
    files: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, str]] = {}
    for variant_name, role in selections:
        variant = provenance["variants"][variant_name]
        executable = variant["executables"][role]
        binary = executable["binary"]
        for item in [
            binary,
            *binary.get("libraries", []),
            *(executable.get("direct_harness") or {}).get("source_files", []),
        ]:
            identity_path = item.get("resolved_path", item["path"])
            files[identity_path] = {
                "path": identity_path,
                **{
                    key: item[key]
                    for key in ("sha256", "size", "mtime_ns", "ctime_ns", "device", "inode")
                },
                "rehash": False,
                "verification": "stat_against_initial_sha256_capture",
            }
        build = executable["build"]
        if build.get("cache"):
            files[build["cache"]] = {
                "path": build["cache"],
                "sha256": build["cache_sha256"],
                **build["cache_stat"],
                "rehash": False,
                "verification": "stat_against_initial_sha256_capture",
            }
        build_provenance = executable.get("build_provenance")
        if build_provenance:
            files[build_provenance["path"]] = {
                "path": build_provenance["path"],
                **{
                    key: build_provenance[key]
                    for key in ("sha256", "size", "mtime_ns", "ctime_ns", "device", "inode")
                },
                "rehash": False,
                "verification": "stat_against_initial_sha256_capture",
            }
        source = variant["source"]
        sources[source["root"]] = {
            "root": source["root"],
            "head": source["head"],
            "dirty_fingerprint": source["dirty_fingerprint"],
        }
    for item in provenance["inputs"]:
        files[item["path"]] = {
            "path": item["path"],
            **{
                key: item[key]
                for key in ("sha256", "size", "mtime_ns", "ctime_ns", "device", "inode")
            },
            "rehash": False,
            "verification": "stat_against_initial_sha256_capture",
        }
    for item in provenance.get("runtime_tools", {}).values():
        files[item["path"]] = {
            "path": item["path"],
            **{
                key: item[key]
                for key in ("sha256", "size", "mtime_ns", "ctime_ns", "device", "inode")
            },
            "rehash": False,
            "verification": "stat_against_initial_sha256_capture",
        }
    return ([files[key] for key in sorted(files)], [sources[key] for key in sorted(sources)])


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


def stage_by_id(manifest: dict[str, Any], stage_id: str) -> dict[str, Any]:
    for stage in manifest["stages"]:
        if stage["id"] == stage_id:
            return stage
    raise ManifestError(f"unknown stage {stage_id!r}")


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
