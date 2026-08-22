"""Direct-target NSYS discovery and discovery-derived NCU planning."""

from __future__ import annotations

import csv
import io
import json
import pathlib
import re
import sqlite3
from contextlib import closing
from typing import Any

from .core import (
    ManifestError,
    ValidationError,
    flock_argv,
    quote_argv,
    reject_forbidden_tokens,
    validate_direct_target,
)
from .nsys_memory import (
    assess_memory_evidence,
    enforce_memory_evidence,
    parse_memory_inventory,
)


def _column(columns: list[str], *candidates: str) -> str | None:
    lowered = {item.lower(): item for item in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for candidate in candidates:
        for item in columns:
            if candidate.lower() in item.lower():
                return item
    return None


def _normalize_shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip().strip("()[]{}")
    if not text:
        return None
    parts = re.split(r"[xX, ]+", text)
    try:
        return [int(part) for part in parts if part]
    except ValueError:
        return None


def _parse_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    launches: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("name"):
            raise ValidationError(f"invalid profiler JSONL record on line {line_number}")
        launches.append(value)
    return launches


def _string_lookup(connection: sqlite3.Connection) -> dict[int, str]:
    names = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    table = next((name for name in names if name.lower() == "stringids"), None)
    if table is None:
        return {}
    columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
    id_column = _column(columns, "id")
    value_column = _column(columns, "value", "string")
    if id_column is None or value_column is None:
        return {}
    return {
        int(row[0]): str(row[1])
        for row in connection.execute(
            f'SELECT "{id_column}", "{value_column}" FROM "{table}"'
        )
    }


def _parse_sqlite(path: pathlib.Path) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        tables = [
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        kernel_tables = [name for name in tables if "kernel" in name.lower()]
        if not kernel_tables:
            raise ValidationError("NSYS SQLite contains no kernel activity table")
        preferred = next(
            (name for name in kernel_tables if "cupti_activity_kind_kernel" in name.lower()),
            kernel_tables[0],
        )
        columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{preferred}")')]
        name_column = _column(columns, "demangledName", "shortName", "name")
        start_column = _column(columns, "start")
        end_column = _column(columns, "end")
        if name_column is None or start_column is None:
            raise ValidationError(f"kernel table {preferred!r} lacks name/start columns")
        graph_column = _column(columns, "graphNodeId", "graphNode", "graphId")
        grid_columns = [_column(columns, f"grid{axis}") for axis in "XYZ"]
        block_columns = [_column(columns, f"block{axis}") for axis in "XYZ"]
        string_ids = _string_lookup(connection)
        select_columns = [name_column, start_column]
        optional = [end_column, graph_column, *grid_columns, *block_columns]
        for item in optional:
            if item is not None and item not in select_columns:
                select_columns.append(item)
        quoted = ", ".join(f'"{item}"' for item in select_columns)
        rows = list(connection.execute(f'SELECT {quoted} FROM "{preferred}" ORDER BY "{start_column}"'))
        index = {name: offset for offset, name in enumerate(select_columns)}
        launches: list[dict[str, Any]] = []
        for row in rows:
            raw_name = row[index[name_column]]
            name = string_ids.get(int(raw_name), str(raw_name)) if isinstance(raw_name, int) else str(raw_name)
            record: dict[str, Any] = {
                "name": name,
                "start_ns": int(row[index[start_column]]),
                "end_ns": int(row[index[end_column]]) if end_column and row[index[end_column]] is not None else None,
                "graph_node_id": row[index[graph_column]] if graph_column else None,
            }
            if all(item is not None for item in grid_columns):
                record["grid"] = [int(row[index[item]]) for item in grid_columns if item is not None]
            if all(item is not None for item in block_columns):
                record["block"] = [int(row[index[item]]) for item in block_columns if item is not None]
            launches.append(record)
        return launches


def parse_discovery(path: pathlib.Path) -> dict[str, Any]:
    """Parse an NSYS SQLite export, or JSONL for hermetic tests/fake logs."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValidationError(f"profiler discovery input is missing: {resolved}")
    launches = _parse_jsonl(resolved) if resolved.suffix == ".jsonl" else _parse_sqlite(resolved)
    launches.sort(key=lambda item: int(item.get("start_ns", item.get("start", 0))))
    occurrences: dict[str, int] = {}
    normalized: list[dict[str, Any]] = []
    for launch_index, raw in enumerate(launches, start=1):
        name = str(raw["name"])
        occurrences[name] = occurrences.get(name, 0) + 1
        normalized.append(
            {
                "launch_index": launch_index,
                "name_occurrence": occurrences[name],
                "name": name,
                "graph_node_id": raw.get("graph_node_id", raw.get("graph_id")),
                "start_ns": raw.get("start_ns", raw.get("start")),
                "end_ns": raw.get("end_ns", raw.get("end")),
                "grid": _normalize_shape(raw.get("grid")),
                "block": _normalize_shape(raw.get("block")),
            }
        )
    groups: dict[str, dict[str, Any]] = {}
    for launch in normalized:
        shape_key = json.dumps(
            {
                "name": launch["name"],
                "grid": launch["grid"],
                "block": launch["block"],
                "graph_node_id": launch["graph_node_id"],
            },
            sort_keys=True,
        )
        group = groups.setdefault(
            shape_key,
            {
                "name": launch["name"],
                "grid": launch["grid"],
                "block": launch["block"],
                "graph_node_id": launch["graph_node_id"],
                "count": 0,
                "launch_indices": [],
                "name_occurrences": [],
            },
        )
        group["count"] += 1
        group["launch_indices"].append(launch["launch_index"])
        group["name_occurrences"].append(launch["name_occurrence"])
    return {
        "source": str(resolved),
        "launch_count": len(normalized),
        "graph_node_trace": {
            "launches_with_graph_node": sum(
                item["graph_node_id"] not in {None, 0, "0"} for item in normalized
            ),
            "distinct_graph_node_ids": sorted(
                {
                    str(item["graph_node_id"])
                    for item in normalized
                    if item["graph_node_id"] not in {None, 0, "0"}
                }
            ),
        },
        "launches": normalized,
        "groups": sorted(groups.values(), key=lambda item: item["launch_indices"][0]),
    }


def validate_native_affinity(target_argv: list[str], direct_harness: dict[str, Any] | None = None) -> None:
    if not target_argv:
        raise ValidationError("profiler target argv is empty")
    executable = pathlib.Path(target_argv[0])
    validate_direct_target(executable, profiler=True, direct_harness=direct_harness)
    reject_forbidden_tokens(target_argv)
    if direct_harness is not None:
        if direct_harness.get("native_llama_affinity") is not True:
            raise ValidationError("direct profiler harness must declare native_llama_affinity=true")
        return
    args = target_argv[1:]
    values: dict[str, str] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"-C", "--cpu-mask", "--cpu-range", "--cpu-range-batch", "--cpu-strict"}:
            if index + 1 >= len(args):
                raise ValidationError(f"native affinity option {token} has no value")
            values[token] = args[index + 1]
            index += 2
        else:
            index += 1
    strict = values.get("--cpu-strict") == "1"
    mask = values.get("-C", values.get("--cpu-mask"))
    ranged = "--cpu-range" in values and "--cpu-range-batch" in values
    bench_mask = executable.name == "llama-bench" and mask == "0x7"
    if not strict or not (bench_mask or ranged):
        raise ValidationError(
            "direct llama profiler target requires --cpu-strict 1 and either -C 0x7 "
            "for the established 0-2 llama-bench shape or both native cpu ranges"
        )


def nsys_discovery_argv(
    target_argv: list[str],
    output_prefix: pathlib.Path,
    direct_harness: dict[str, Any] | None = None,
    *,
    cuda_graph_trace: dict[str, Any] | None = None,
    request_cuda_memory_usage: bool = False,
    nsys_executable: str = "nsys",
) -> list[str]:
    validate_native_affinity(target_argv, direct_harness)
    argv = [
        nsys_executable,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "--output",
        str(output_prefix),
    ]
    if cuda_graph_trace and cuda_graph_trace["applicability"] == "required":
        argv.append(
            "--cuda-graph-trace="
            f"{cuda_graph_trace['granularity']}:{cuda_graph_trace['launch_origin']}"
        )
    if request_cuda_memory_usage:
        argv.append("--cuda-memory-usage=true")
    return [*argv, *target_argv]


def verify_nsys_graph_trace_capability(
    config: dict[str, Any],
    *,
    help_text: str,
    version_text: str,
) -> dict[str, Any]:
    """Fail before capture when required node tracing is unavailable."""
    required = config["applicability"] == "required"
    evidence = {
        "applicability": config["applicability"],
        "requested": (
            f"{config.get('granularity')}:{config.get('launch_origin')}" if required else None
        ),
        "nsys_version": version_text.strip(),
        "help_mentions_cuda_graph_trace": "--cuda-graph-trace" in help_text,
        "help_mentions_node_granularity": bool(re.search(r"\bnode\b", help_text)),
    }
    if required and not (
        evidence["help_mentions_cuda_graph_trace"]
        and evidence["help_mentions_node_granularity"]
    ):
        raise ValidationError(
            "this NSYS version does not advertise required --cuda-graph-trace node support"
        )
    evidence["status"] = "supported" if required else "not_applicable"
    return evidence


def verify_nsys_memory_trace_capability(
    config: dict[str, Any] | None,
    *,
    help_text: str,
    version_text: str,
) -> dict[str, Any]:
    """Verify the exact NSYS binary before requesting CUDA memory events."""
    if config is None:
        return {
            "requested": False,
            "mode": None,
            "categories": [],
            "nsys_version": version_text.strip(),
            "help_mentions_cuda_memory_usage": "--cuda-memory-usage" in help_text,
            "status": "not_configured",
        }
    supported = "--cuda-memory-usage" in help_text
    evidence = {
        "requested": True,
        "mode": config["mode"],
        "categories": list(config["categories"]),
        "nsys_version": version_text.strip(),
        "help_mentions_cuda_memory_usage": supported,
        "requested_option": "--cuda-memory-usage=true" if supported else None,
        "status": "supported" if supported else "unsupported",
    }
    if config["mode"] == "required" and not supported:
        raise ValidationError(
            "this NSYS version does not advertise required --cuda-memory-usage support"
        )
    return evidence


def verify_graph_node_discovery(
    discovery: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    observed = int(discovery["graph_node_trace"]["launches_with_graph_node"])
    result = {
        "applicability": config["applicability"],
        "requested": config.get("granularity"),
        "observed_launches_with_graph_node": observed,
        "status": "not_applicable",
    }
    if config["applicability"] == "required":
        if observed == 0:
            raise ValidationError(
                "NSYS graph-node tracing was required and explicitly requested, but the export "
                "contains no nonzero graph node IDs"
            )
        result["status"] = "verified"
    return result


def select_regression_diagnostic(
    stage: dict[str, Any],
    stage_report: dict[str, Any] | None,
    screen_selection: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a preregistered diagnostic target only after a regression signal."""
    if not stage_report:
        return {"triggered": False, "reason": "screening_stage_not_executed"}
    observation = stage_report.get("screening_observation")
    if not isinstance(observation, dict):
        return {"triggered": False, "reason": "stage_has_no_screening_observation"}
    if observation.get("signal") != "regression_signal":
        return {"triggered": False, "reason": "no_regression_signal"}
    raw_pairs = observation.get("raw_pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != 1:
        raise ValidationError("early diagnostic requires exactly one preserved screening pair")
    pair = raw_pairs[0]
    raw_screens = pair.get("raw_screens")
    if not isinstance(raw_screens, dict):
        raise ValidationError("screening report lacks raw per-screen observations")
    values: dict[str, dict[str, float]] = {}
    for variant in ("baseline", "candidate"):
        rows = raw_screens.get(variant)
        if not isinstance(rows, list):
            raise ValidationError(f"screening report lacks {variant} per-screen observations")
        for row in rows:
            screen = str(row["screen"])
            values.setdefault(screen, {})[variant] = float(row["value"])
    stage_screens = [
        str(item["id"])
        for item in stage.get("screens", [{"id": "default"}])
    ]
    for screen in stage_screens:
        if set(values.get(screen, {})) != {"baseline", "candidate"}:
            raise ValidationError(f"screening report lacks a matched pair for screen {screen!r}")
    direction = str(observation.get("direction", stage["metric"]["direction"]))

    def details(screen: str) -> dict[str, Any]:
        baseline = values[screen]["baseline"]
        candidate = values[screen]["candidate"]
        if baseline <= 0 or candidate <= 0:
            raise ValidationError("diagnostic screen selection requires positive observations")
        percent = 100.0 * (candidate / baseline - 1.0)
        benefit = percent if direction == "higher" else -percent
        return {
            "screen": screen,
            "baseline": baseline,
            "candidate": candidate,
            "percent_change": percent,
            "benefit_percent": benefit,
            "regression_severity_percent": -benefit,
        }

    candidates = [details(screen) for screen in stage_screens]
    if screen_selection["kind"] == "fixed":
        selected = next(
            item for item in candidates if item["screen"] == screen_selection["screen"]
        )
    else:
        selected = max(
            candidates,
            key=lambda item: (
                item["regression_severity_percent"],
                -stage_screens.index(item["screen"]),
            ),
        )
    return {
        "triggered": True,
        "reason": "preregistered_regression_signal",
        "stage": stage["id"],
        "selection_policy": screen_selection,
        "selected_screen": selected,
        "all_screen_observations": candidates,
        "confidence_claim": "none",
    }


def _selector_matches(launch: dict[str, Any], selector: dict[str, Any]) -> bool:
    pattern = selector.get("kernel_regex")
    if not isinstance(pattern, str) or not pattern:
        raise ManifestError("profiler selector requires kernel_regex")
    if re.search(pattern, launch["name"]) is None:
        return False
    if selector.get("graph_only") and launch.get("graph_node_id") in {None, 0, "0"}:
        return False
    for key in ("grid", "block"):
        if key in selector and _normalize_shape(selector[key]) != launch.get(key):
            return False
    return True


def build_ncu_plan(
    discovery: dict[str, Any],
    selector: dict[str, Any],
    target_argv: list[str],
    output_root: pathlib.Path,
    *,
    metrics: list[str] | None = None,
    direct_harness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one filtered NCU capture from the current NSYS discovery.

    The selected occurrences must be contiguous among one exact kernel name;
    otherwise one skip/count range would silently include unselected launches.
    Every value is derived from this discovery, never carried across builds.
    """
    validate_native_affinity(target_argv, direct_harness)
    matches = [launch for launch in discovery["launches"] if _selector_matches(launch, selector)]
    if not matches:
        raise ValidationError("profiler selector matched no discovered kernel launches")
    occurrence_kind = selector.get("occurrence_policy", {"kind": "all"}).get("kind", "all")
    if occurrence_kind == "first":
        selected = matches[:1]
    elif occurrence_kind == "middle":
        selected = matches[(len(matches) - 1) // 2 : (len(matches) - 1) // 2 + 1]
    elif occurrence_kind == "last":
        selected = matches[-1:]
    else:
        selected = matches
    names = {launch["name"] for launch in selected}
    if len(names) != 1:
        raise ValidationError("one filtered NCU capture requires one discovered kernel name")
    occurrences = [int(launch["name_occurrence"]) for launch in selected]
    expected = list(range(min(occurrences), max(occurrences) + 1))
    if occurrences != expected:
        raise ValidationError(
            "selected same-kernel occurrences are not contiguous; narrow the discovery stage or selector"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    kernel_name = selected[0]["name"]
    kernel_regex = f"^{re.escape(kernel_name)}$"
    launch_skip = min(occurrences) - 1
    launch_count = len(occurrences)
    argv = [
        "ncu",
        "--target-processes",
        "all",
        "--graph-profiling",
        "node",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        f"regex:{kernel_regex}",
        "--launch-skip",
        str(launch_skip),
        "--launch-count",
        str(launch_count),
    ]
    if metrics:
        argv.extend(["--metrics", ",".join(metrics)])
    prefix = output_root / f"capture-launch-{selected[0]['launch_index']:06d}"
    argv.extend(["--export", str(prefix), "--force-overwrite", *target_argv])
    report_path = pathlib.Path(f"{prefix}.ncu-rep")
    expected_shapes = sorted(
        {
            json.dumps({"grid": item.get("grid"), "block": item.get("block")}, sort_keys=True)
            for item in selected
        }
    )
    command = {
        "selected_launches": selected,
        "derivation": {
            "kernel_name": kernel_name,
            "matching_launch_count": len(matches),
            "occurrence_policy": occurrence_kind,
            "name_occurrences": occurrences,
            "launch_skip": launch_skip,
            "launch_count": launch_count,
            "discovery_launch_indices": [item["launch_index"] for item in selected],
            "expected_shapes": [json.loads(item) for item in expected_shapes],
        },
        "report_path": str(report_path),
        "direct_argv": argv,
        "direct_command": quote_argv(argv),
        "flock_argv": flock_argv(argv),
        "flock_command": quote_argv(flock_argv(argv)),
    }
    return {
        "pattern": "one_nsys_discovery_then_one_filtered_ncu",
        "scope": "one preregistered investigation stage, not benchmark repetitions",
        "discovery_source": discovery["source"],
        "selector": selector,
        "selected_count": len(selected),
        "command": command,
        "limitations": [
            "Cross-process launch order can drift; the exported report must be verified against kernel, count, and available shapes before it is usable.",
            "Filtered kernel evidence does not prove end-to-end performance, resource use, output exactness, or long-context behavior.",
        ],
    }


def parse_ncu_raw_csv(text: str) -> dict[str, Any]:
    """Extract unique kernel captures from ``ncu --import --csv --page raw``."""
    lines = text.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if "Kernel Name" in line and "ID" in line),
        None,
    )
    if header_index is None:
        return {"captures": [], "columns": [], "parse_status": "unverifiable_no_header"}
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    columns = reader.fieldnames or []
    captures: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(reader, start=1):
        name = (row.get("Kernel Name") or "").strip()
        capture_id = (row.get("ID") or "").strip()
        if not name or not capture_id:
            continue
        capture_key = "|".join(
            (row.get(column) or "").strip()
            for column in ("Process ID", "Context", "Stream", "ID")
        )
        capture = captures.setdefault(
            capture_key,
            {
                "id": capture_id,
                "process_id": (row.get("Process ID") or "").strip() or None,
                "context": (row.get("Context") or "").strip() or None,
                "stream": (row.get("Stream") or "").strip() or None,
                "kernel_name": name,
                "grid": _normalize_shape(row.get("Grid Size")),
                "block": _normalize_shape(row.get("Block Size")),
                "metrics": [],
                "raw_row_numbers": [],
            },
        )
        if capture["kernel_name"] != name:
            raise ValidationError(f"NCU capture ID {capture_key} has conflicting kernel names")
        metric_name = (row.get("Metric Name") or "").strip()
        metric_value = (row.get("Metric Value") or "").strip()
        if metric_name and metric_value:
            metric = {
                "name": metric_name,
                "unit": (row.get("Metric Unit") or "").strip() or None,
                "value": metric_value,
            }
            if metric not in capture["metrics"]:
                capture["metrics"].append(metric)
        capture["raw_row_numbers"].append(row_number)
    return {
        "captures": list(captures.values()),
        "columns": columns,
        "parse_status": "parsed" if captures else "unverifiable_no_captures",
    }


def _canonical_kernel_name(name: str) -> str:
    """Normalize only known NSYS/NCU template-constant demangler differences."""
    return re.sub(r"\((?:int|bool|ggml_type)\)([-+]?[0-9]+)", r"\1", name)


def verify_ncu_capture(plan: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    """Verify a fresh NCU report and expose launch-order drift instead of assuming it away."""
    derivation = plan["command"]["derivation"]
    expected_name = str(derivation["kernel_name"])
    expected_count = int(derivation["launch_count"])
    captures = parsed["captures"]
    issues: list[str] = []
    if parsed["parse_status"] != "parsed":
        return {
            "status": "unverifiable",
            "issues": [parsed["parse_status"]],
            "expected_kernel": expected_name,
            "expected_capture_count": expected_count,
            "observed_captures": captures,
        }
    if len(captures) != expected_count:
        issues.append(f"expected {expected_count} captures, observed {len(captures)}")
    observed_names = sorted({item["kernel_name"] for item in captures})
    canonical_expected_name = _canonical_kernel_name(expected_name)
    canonical_observed_names = sorted({_canonical_kernel_name(item) for item in observed_names})
    if canonical_observed_names != [canonical_expected_name]:
        issues.append(f"expected kernel {expected_name!r}, observed {observed_names!r}")
    expected_shapes = {
        json.dumps(item, sort_keys=True) for item in derivation.get("expected_shapes", [])
    }
    observed_shapes = {
        json.dumps({"grid": item["grid"], "block": item["block"]}, sort_keys=True)
        for item in captures
    }
    if expected_shapes and any(item["grid"] is None or item["block"] is None for item in captures):
        return {
            "status": "unverifiable",
            "issues": [*issues, "NCU raw export lacks grid/block shape data"],
            "expected_kernel": expected_name,
            "expected_capture_count": expected_count,
            "observed_captures": captures,
        }
    if expected_shapes and not observed_shapes.issubset(expected_shapes):
        issues.append("observed NCU launch shape was absent from NSYS discovery selection")
    return {
        "status": "verified" if not issues else "failed",
        "issues": issues,
        "expected_kernel": expected_name,
        "canonical_expected_kernel": canonical_expected_name,
        "observed_kernel_names": observed_names,
        "canonical_observed_kernel_names": canonical_observed_names,
        "kernel_name_normalization": (
            "known NSYS/NCU integral template-cast spelling only; raw names retained"
        ),
        "expected_capture_count": expected_count,
        "observed_capture_count": len(captures),
        "observed_captures": captures,
        "cross_process_launch_order": (
            "not assumed; report kernel/count/shapes were checked after the separate NCU process"
        ),
    }


def agent_profile_summary(profile_root: pathlib.Path) -> dict[str, Any]:
    """Expose a compact, machine-readable view while retaining all raw artifacts."""
    root = profile_root.resolve()

    def read_json(name: str) -> dict[str, Any] | None:
        path = root / name
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValidationError(f"profiler artifact is not an object: {path}")
        return value

    plan = read_json("ncu-plan.json")
    discovery = read_json("discovery-inventory.json")
    verification = read_json("ncu-verification.json")
    memory_inventory = read_json("memory-inventory.json")
    state = read_json("profile-state.json")
    if plan is None or discovery is None or state is None:
        raise ValidationError(f"profiler artifacts are incomplete: {root}")
    derivation = plan["command"]["derivation"]
    selected = plan["command"]["selected_launches"]
    timings: dict[str, Any] = {}
    for kind, filename in (("nsys", "nsys-result.json"), ("ncu", "ncu-result.json")):
        result = read_json(filename)
        if result is not None:
            timing = result.get("timing", {})
            timings[kind] = {
                "target_process_seconds": timing.get("target_process_seconds"),
                "locked_lifecycle_wall_seconds": timing.get("locked_lifecycle_wall_seconds"),
                "toolkit_overhead_seconds": timing.get("toolkit_overhead_seconds"),
            }
    captures = (
        verification.get("parsed_export", {}).get("captures", [])
        if verification is not None
        else []
    )
    memory_summary: dict[str, Any] = {
        "assessment_status": "legacy_unavailable",
        "categories": {},
    }
    if memory_inventory is not None:
        memory_categories = memory_inventory.get("categories", {})
        memory_summary = {
            "assessment_status": memory_inventory.get("assessment", {}).get(
                "status", "not_evaluated"
            ),
            "issues": memory_inventory.get("assessment", {}).get("issues", []),
            "categories": {
                name: {
                    "status": details.get("status", "unavailable"),
                    "reason": details.get("reason"),
                    "event_count": details.get("event_count"),
                }
                for name, details in memory_categories.items()
            },
            "allocation_groups": memory_categories.get("gpu_memory_events", {}).get(
                "groups", []
            ),
            "captured_outstanding_by_memory_kind": memory_categories.get(
                "gpu_memory_events", {}
            ).get("captured_outstanding_by_process_device_kind", []),
            "allocation_lifetime_counts": {
                key: memory_categories.get("allocation_lifetimes", {}).get(key)
                for key in (
                    "paired_lifetime_count",
                    "unmatched_allocation_count",
                    "unmatched_deallocation_count",
                )
            },
            "captured_device_high_water": memory_categories.get(
                "captured_device_high_water", {}
            ).get("by_process_device", []),
            "copy_groups": memory_categories.get("copy_activity", {}).get("groups", []),
            "memory_api_groups": memory_categories.get("cuda_api", {}).get(
                "memory_api_groups", []
            ),
            "vmm_groups": memory_categories.get("vmm", {}).get("groups", []),
            "memory_boundaries": memory_inventory.get("memory_boundaries", {}),
        }
    return {
        "status": (
            "verified"
            if verification is not None and verification.get("status") == "verified"
            else state.get("ncu_verification_status", "not_executed")
        ),
        "stage": plan["stage"],
        "variant": plan["variant"],
        "screen": plan["screen"],
        "nsys": {
            "total_launch_count": discovery["launch_count"],
            "graph_node_trace": discovery["graph_node_trace"],
            "selected_kernel": derivation["kernel_name"],
            "matching_launch_count": derivation["matching_launch_count"],
            "selected_launch_count": derivation["launch_count"],
            "selected_launch_indices": derivation["discovery_launch_indices"],
            "selected_name_occurrences": derivation["name_occurrences"],
            "selected_shapes": derivation.get("expected_shapes", []),
            "selected_graph_node_ids": sorted(
                {
                    str(item["graph_node_id"])
                    for item in selected
                    if item.get("graph_node_id") not in {None, 0, "0"}
                }
            ),
        },
        "ncu": {
            "verification_status": (
                verification.get("status") if verification is not None else "not_executed"
            ),
            "issues": verification.get("issues", []) if verification is not None else [],
            "observed_capture_count": (
                verification.get("observed_capture_count") if verification is not None else None
            ),
            "captures": captures,
        },
        "memory": memory_summary,
        "timing_seconds": timings,
        "artifacts": {
            "root": str(root),
            "discovery_inventory": str(root / "discovery-inventory.json"),
            "memory_inventory": str(root / "memory-inventory.json"),
            "memory_trace_capability": str(
                root / "nsys-memory-trace-capability.json"
            ),
            "nsys_report": str(root / "discovery.nsys-rep"),
            "nsys_stdout": str(root / "nsys-stdout.log"),
            "nsys_stderr": str(root / "nsys-stderr.log"),
            "ncu_plan": str(root / "ncu-plan.json"),
            "ncu_report": plan["command"]["report_path"],
            "ncu_stdout": str(root / "ncu-stdout.log"),
            "ncu_stderr": str(root / "ncu-stderr.log"),
            "ncu_verification": str(root / "ncu-verification.json"),
        },
        "evidence_boundary": (
            "Diagnostic-only NSYS/NCU evidence; it is not a statistical performance, "
            "correctness, resource, production, or long-context acceptance result."
        ),
    }


def compare_agent_profiles(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Place independently discovered evidence side by side without inventing statistics."""
    if baseline.get("variant") != "baseline" or candidate.get("variant") != "candidate":
        raise ValidationError("agent profile comparison requires baseline then candidate")

    def metric_inventory(profile: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "capture_id": capture.get("id"),
                "kernel_name": capture.get("kernel_name"),
                "metrics": capture.get("metrics", []),
            }
            for capture in profile.get("ncu", {}).get("captures", [])
        ]

    return {
        "status": "diagnostic_side_by_side_only",
        "same_selected_kernel": (
            baseline["nsys"]["selected_kernel"] == candidate["nsys"]["selected_kernel"]
        ),
        "same_selected_shapes": (
            baseline["nsys"]["selected_shapes"] == candidate["nsys"]["selected_shapes"]
        ),
        "launch_inventory": {
            variant: {
                "total": profile["nsys"]["total_launch_count"],
                "matching": profile["nsys"]["matching_launch_count"],
                "selected": profile["nsys"]["selected_launch_count"],
            }
            for variant, profile in (("baseline", baseline), ("candidate", candidate))
        },
        "raw_ncu_metrics": {
            "baseline": metric_inventory(baseline),
            "candidate": metric_inventory(candidate),
        },
        "memory_inventory": {
            "baseline": baseline.get("memory", {}),
            "candidate": candidate.get("memory", {}),
        },
        "timing_seconds": {
            "baseline": baseline.get("timing_seconds", {}),
            "candidate": candidate.get("timing_seconds", {}),
        },
        "comparison_boundary": (
            "Raw one-capture diagnostics are shown side by side; no paired interval, "
            "equivalence, acceptance decision, or generic aggregation across metric kinds "
            "is inferred."
        ),
    }
