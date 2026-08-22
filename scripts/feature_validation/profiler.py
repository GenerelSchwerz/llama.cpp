"""Direct-target NSYS discovery and discovery-derived NCU planning."""

from __future__ import annotations

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
) -> list[str]:
    validate_native_affinity(target_argv, direct_harness)
    return [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--force-overwrite=true",
        "--output",
        str(output_prefix),
        *target_argv,
    ]


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
    selected = [launch for launch in discovery["launches"] if _selector_matches(launch, selector)]
    if not selected:
        raise ValidationError("profiler selector matched no discovered kernel launches")
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
    command = {
        "selected_launches": selected,
        "derivation": {
            "kernel_name": kernel_name,
            "name_occurrences": occurrences,
            "launch_skip": launch_skip,
            "launch_count": launch_count,
            "discovery_launch_indices": [item["launch_index"] for item in selected],
        },
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
            "Kernel and occurrence filters are valid only for the recorded discovery identity.",
            "Filtered kernel evidence does not prove end-to-end performance, resource use, output exactness, or long-context behavior.",
        ],
    }
