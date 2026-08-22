"""Schema-tolerant, evidence-bounded NSYS CUDA memory inventory."""

from __future__ import annotations

import pathlib
import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from typing import Any

from .core import ValidationError


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _column(columns: list[str], *aliases: str) -> str | None:
    by_name = {_normalized(item): item for item in columns}
    return next((by_name[_normalized(item)] for item in aliases if _normalized(item) in by_name), None)


def _tables(connection: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ]


def _table(
    tables: list[str], *, exact: tuple[str, ...] = (), contains: tuple[str, ...] = ()
) -> str | None:
    exact_normalized = {_normalized(item) for item in exact}
    for name in tables:
        if _normalized(name) in exact_normalized:
            return name
    for name in tables:
        lowered = name.lower()
        if contains and all(item.lower() in lowered for item in contains):
            return name
    return None


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({_identifier(table)})")
    ]


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _enum_lookup(
    connection: sqlite3.Connection, tables: list[str], *table_names: str
) -> dict[int, str]:
    table = _table(tables, exact=tuple(table_names))
    if table is None:
        return {}
    columns = _columns(connection, table)
    id_column = _column(columns, "id", "value")
    label_column = _column(columns, "label", "name")
    if id_column is None or label_column is None:
        return {}
    result: dict[int, str] = {}
    query = (
        f"SELECT {_identifier(id_column)}, {_identifier(label_column)} "
        f"FROM {_identifier(table)}"
    )
    for key, label in connection.execute(query):
        if key is not None and label is not None:
            result[int(key)] = str(label)
    return result


def _label(value: Any, lookup: dict[int, str]) -> str | None:
    if value is None:
        return None
    if isinstance(value, int) and value in lookup:
        return lookup[value]
    return str(value)


def _operation(value: str | None) -> str | None:
    lowered = (value or "").lower()
    if "dealloc" in lowered or "free" in lowered or "release" in lowered:
        return "deallocation"
    if "alloc" in lowered or "create" in lowered or lowered in {"allocate", "allocation"}:
        return "allocation"
    return None


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason, "event_count": None}


def _memory_categories(
    connection: sqlite3.Connection, tables: list[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    table = _table(
        tables,
        exact=("CUDA_GPU_MEMORY_USAGE_EVENTS", "CUDA_MEMORY_USAGE_EVENTS"),
        contains=("memory", "usage", "event"),
    )
    if table is None:
        reason = "NSYS export contains no CUDA GPU memory-usage event table"
        return _unavailable(reason), _unavailable(reason), _unavailable(reason)
    columns = _columns(connection, table)
    selected = {
        "start": _column(columns, "start", "timestamp", "time"),
        "process": _column(columns, "globalPid", "processId", "pid"),
        "device": _column(columns, "deviceId", "device"),
        "context": _column(columns, "contextId", "context"),
        "address": _column(columns, "address", "virtualAddress", "ptr"),
        "bytes": _column(columns, "bytes", "size", "allocationSize"),
        "kind": _column(columns, "memKind", "memoryKind", "memoryKindId"),
        "operation": _column(
            columns, "memoryOperationType", "operationType", "operation", "op"
        ),
        "name": _column(columns, "name", "variableName"),
        "correlation": _column(columns, "correlationId", "correlation"),
    }
    required = {"start", "bytes", "kind", "operation"}
    missing = sorted(key for key in required if selected[key] is None)
    if missing:
        reason = f"memory-usage table {table!r} lacks required columns {missing}"
        return _unavailable(reason), _unavailable(reason), _unavailable(reason)
    query_columns = list(dict.fromkeys(item for item in selected.values() if item is not None))
    query = (
        "SELECT "
        + ", ".join(_identifier(item) for item in query_columns)
        + f" FROM {_identifier(table)} ORDER BY {_identifier(selected['start'])}"
    )
    indexes = {name: offset for offset, name in enumerate(query_columns)}
    operation_names = _enum_lookup(
        connection, tables, "ENUM_CUDA_DEV_MEM_EVENT_OPER", "ENUM_CUDA_MEMORY_OPERATION"
    )
    kind_names = _enum_lookup(connection, tables, "ENUM_CUDA_MEM_KIND", "ENUM_CUDA_MEMORY_KIND")

    def value(row: tuple[Any, ...], key: str) -> Any:
        column = selected[key]
        return row[indexes[column]] if column is not None else None

    events: list[dict[str, Any]] = []
    unknown_operations = 0
    unknown_kinds = 0
    for row in connection.execute(query):
        raw_operation = _label(value(row, "operation"), operation_names)
        operation = _operation(raw_operation)
        kind = _label(value(row, "kind"), kind_names)
        unknown_operations += operation is None
        unknown_kinds += kind is None or (not kind_names and str(kind).isdigit())
        events.append(
            {
                "start_ns": _as_int(value(row, "start")),
                "process_id": _as_int(value(row, "process")),
                "device_id": _as_int(value(row, "device")),
                "context_id": _as_int(value(row, "context")),
                "address": _as_int(value(row, "address")),
                "bytes": _as_int(value(row, "bytes")),
                "memory_kind": kind,
                "operation": operation,
                "raw_operation": raw_operation,
                "name": value(row, "name"),
                "correlation_id": _as_int(value(row, "correlation")),
            }
        )

    semantic_problem = bool(events and (unknown_operations or unknown_kinds))
    if not events and (not operation_names or not kind_names):
        semantic_problem = True
    memory_status = "partial" if semantic_problem else "available"
    memory_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in events:
        key = (event["memory_kind"], event["operation"])
        group = memory_groups.setdefault(
            key,
            {
                "memory_kind": event["memory_kind"],
                "operation": event["operation"],
                "event_count": 0,
                "total_bytes": 0,
                "maximum_bytes": 0,
            },
        )
        group["event_count"] += 1
        if event["bytes"] is not None:
            group["total_bytes"] += event["bytes"]
            group["maximum_bytes"] = max(group["maximum_bytes"], event["bytes"])
    memory = {
        "status": memory_status,
        "reason": (
            "operation or memory-kind enum semantics are unavailable for some rows"
            if semantic_problem
            else None
        ),
        "table": table,
        "columns": selected,
        "event_count": len(events),
        "unknown_operation_count": unknown_operations,
        "unknown_memory_kind_count": unknown_kinds,
        "groups": sorted(
            memory_groups.values(),
            key=lambda item: (str(item["memory_kind"]), str(item["operation"])),
        ),
        "events": events,
    }

    if selected["address"] is None or semantic_problem:
        lifetime = _unavailable(
            "allocation lifetimes require address plus resolved allocation/deallocation semantics"
        )
    else:
        active: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        lifetimes: list[dict[str, Any]] = []
        unmatched_deallocations: list[dict[str, Any]] = []
        for event in events:
            key = (
                event["process_id"],
                event["device_id"],
                event["memory_kind"],
                event["address"],
            )
            if event["operation"] == "allocation":
                active[key].append(event)
            elif active[key]:
                allocated = active[key].pop()
                lifetimes.append(
                    {
                        "process_id": event["process_id"],
                        "device_id": event["device_id"],
                        "memory_kind": event["memory_kind"],
                        "address": event["address"],
                        "bytes": allocated["bytes"],
                        "allocation_start_ns": allocated["start_ns"],
                        "deallocation_start_ns": event["start_ns"],
                        "lifetime_ns": (
                            event["start_ns"] - allocated["start_ns"]
                            if event["start_ns"] is not None
                            and allocated["start_ns"] is not None
                            else None
                        ),
                    }
                )
            else:
                unmatched_deallocations.append(event)
        unmatched_allocations = [event for pending in active.values() for event in pending]
        lifetime = {
            "status": "available",
            "reason": None,
            "event_count": len(events),
            "paired_lifetime_count": len(lifetimes),
            "lifetimes": lifetimes,
            "unmatched_allocation_count": len(unmatched_allocations),
            "unmatched_allocations": unmatched_allocations,
            "unmatched_deallocation_count": len(unmatched_deallocations),
            "unmatched_deallocations": unmatched_deallocations,
            "interpretation": (
                "Unmatched allocations can be intentionally live at capture end; this inventory "
                "does not infer a leak."
            ),
        }

    if semantic_problem or any(event["bytes"] is None for event in events):
        high_water = _unavailable(
            "captured high-water requires resolved operations, memory kinds, and byte counts"
        )
    else:
        outstanding: dict[tuple[Any, ...], int] = defaultdict(int)
        peaks: dict[tuple[Any, ...], int] = defaultdict(int)
        minima: dict[tuple[Any, ...], int] = defaultdict(int)
        combined_outstanding: dict[tuple[Any, ...], int] = defaultdict(int)
        combined_peaks: dict[tuple[Any, ...], int] = defaultdict(int)
        combined_minima: dict[tuple[Any, ...], int] = defaultdict(int)
        for event in events:
            kind = str(event["memory_kind"] or "unknown")
            key = (event["process_id"], event["device_id"], kind)
            delta = int(event["bytes"] or 0)
            signed = delta if event["operation"] == "allocation" else -delta
            outstanding[key] += signed
            peaks[key] = max(peaks[key], outstanding[key])
            minima[key] = min(minima[key], outstanding[key])
            if "device" not in kind.lower() and "array" not in kind.lower():
                continue
            combined_key = (event["process_id"], event["device_id"])
            combined_outstanding[combined_key] += signed
            combined_peaks[combined_key] = max(
                combined_peaks[combined_key], combined_outstanding[combined_key]
            )
            combined_minima[combined_key] = min(
                combined_minima[combined_key], combined_outstanding[combined_key]
            )
        kind_rows = [
            {
                "process_id": key[0],
                "device_id": key[1],
                "memory_kind": key[2],
                "captured_outstanding_high_water_bytes": peaks[key],
                "captured_final_outstanding_bytes": outstanding[key],
                "minimum_running_balance_bytes": minima[key],
            }
            for key in sorted(peaks, key=lambda item: tuple(str(value) for value in item))
        ]
        memory["captured_outstanding_by_process_device_kind"] = kind_rows
        combined_rows = [
            {
                "process_id": key[0],
                "device_id": key[1],
                "captured_outstanding_high_water_bytes": combined_peaks[key],
                "captured_final_outstanding_bytes": combined_outstanding[key],
                "minimum_running_balance_bytes": combined_minima[key],
            }
            for key in sorted(
                combined_peaks, key=lambda item: tuple(str(value) for value in item)
            )
        ]
        negative = any(
            row["minimum_running_balance_bytes"] < 0 for row in combined_rows
        )
        high_water = {
            "status": "partial" if negative else "available",
            "reason": (
                "a negative running balance proves that capture-start allocations were missing"
                if negative
                else None
            ),
            "event_count": sum(
                1
                for event in events
                if "device" in str(event["memory_kind"]).lower()
                or "array" in str(event["memory_kind"]).lower()
            ),
            "by_process_device": combined_rows,
            "by_process_device_kind": [
                row
                for row in kind_rows
                if "device" in row["memory_kind"].lower()
                or "array" in row["memory_kind"].lower()
            ],
            "interpretation": (
                "Captured allocation-event balance only; not nvidia-smi process VRAM, VMM virtual "
                "reservation, or a whole-system device-memory high-water."
            ),
        }
    return memory, lifetime, high_water


def _copy_category(connection: sqlite3.Connection, tables: list[str]) -> dict[str, Any]:
    def activity_table(kind: str) -> str | None:
        exact = _table(tables, exact=(f"CUPTI_ACTIVITY_KIND_{kind.upper()}",))
        if exact is not None:
            return exact
        return next(
            (
                name
                for name in tables
                if kind in name.lower()
                and "activity" in name.lower()
                and not name.lower().startswith("enum")
            ),
            None,
        )

    definitions = (
        ("memcpy", activity_table("memcpy")),
        ("memset", activity_table("memset")),
    )
    missing_tables = [kind for kind, table in definitions if table is None]
    if missing_tables:
        return _unavailable(f"NSYS export lacks activity tables for {missing_tables}")
    copy_kinds = _enum_lookup(connection, tables, "ENUM_CUDA_MEMCPY_OPER")
    memory_kinds = _enum_lookup(connection, tables, "ENUM_CUDA_MEM_KIND")
    summaries: list[dict[str, Any]] = []
    total_events = 0
    for activity, table in definitions:
        assert table is not None
        columns = _columns(connection, table)
        start = _column(columns, "start", "timestamp")
        end = _column(columns, "end", "stop")
        size = _column(columns, "bytes", "size")
        if start is None or size is None:
            return _unavailable(f"{activity} table {table!r} lacks start/bytes columns")
        kind = _column(columns, "copyKind", "operation", "kind")
        src = _column(columns, "srcKind", "sourceKind")
        dst = _column(columns, "dstKind", "destinationKind")
        mem = _column(columns, "memKind", "memoryKind")
        selected = list(dict.fromkeys(item for item in (start, end, size, kind, src, dst, mem) if item))
        indexes = {name: offset for offset, name in enumerate(selected)}
        query = "SELECT " + ", ".join(_identifier(item) for item in selected) + f" FROM {_identifier(table)}"
        grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in connection.execute(query):
            def raw(column: str | None) -> Any:
                return row[indexes[column]] if column is not None else None

            key = (
                activity,
                _label(raw(kind), copy_kinds),
                _label(raw(src), memory_kinds),
                _label(raw(dst), memory_kinds),
                _label(raw(mem), memory_kinds),
            )
            summary = grouped.setdefault(
                key,
                {
                    "activity": activity,
                    "operation": key[1],
                    "source_memory_kind": key[2],
                    "destination_memory_kind": key[3],
                    "memory_kind": key[4],
                    "event_count": 0,
                    "total_bytes": 0,
                    "total_duration_ns": 0,
                    "maximum_bytes": 0,
                },
            )
            row_bytes = int(raw(size))
            summary["event_count"] += 1
            summary["total_bytes"] += row_bytes
            summary["maximum_bytes"] = max(summary["maximum_bytes"], row_bytes)
            if end is not None and raw(end) is not None:
                summary["total_duration_ns"] += int(raw(end)) - int(raw(start))
            total_events += 1
        summaries.extend(grouped.values())
    return {
        "status": "available",
        "reason": None,
        "event_count": total_events,
        "groups": sorted(summaries, key=lambda item: tuple(str(value) for value in item.values())),
        "raw_rows": "retained in the NSYS SQLite export; this JSON preserves complete counts and byte totals grouped by observed operation and memory kind",
    }


def _api_categories(
    connection: sqlite3.Connection, tables: list[str], strings: dict[int, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    api_tables = [
        name
        for name in tables
        if "cupti_activity_kind_runtime" in name.lower()
        or "cupti_activity_kind_driver" in name.lower()
    ]
    if not api_tables:
        reason = "NSYS export contains no CUDA runtime/driver API activity table"
        return _unavailable(reason), _unavailable(reason)
    summaries: dict[str, dict[str, Any]] = {}
    unresolved_names = 0
    total_events = 0
    for table in api_tables:
        columns = _columns(connection, table)
        start = _column(columns, "start", "timestamp")
        end = _column(columns, "end", "stop")
        name = _column(columns, "name", "functionName", "apiName")
        name_id = _column(columns, "nameId", "stringId")
        if start is None or (name is None and name_id is None):
            reason = f"CUDA API table {table!r} lacks start and name/nameId columns"
            return _unavailable(reason), _unavailable(reason)
        selected = list(dict.fromkeys(item for item in (start, end, name, name_id) if item))
        indexes = {column: offset for offset, column in enumerate(selected)}
        query = "SELECT " + ", ".join(_identifier(item) for item in selected) + f" FROM {_identifier(table)}"
        for row in connection.execute(query):
            if name is not None:
                api_name = str(row[indexes[name]])
            else:
                raw_id = _as_int(row[indexes[name_id]])
                api_name = strings.get(raw_id, "") if raw_id is not None else ""
            if not api_name:
                unresolved_names += 1
                api_name = "<unresolved>"
            summary = summaries.setdefault(
                api_name,
                {"name": api_name, "event_count": 0, "total_duration_ns": 0},
            )
            summary["event_count"] += 1
            if end is not None and row[indexes[end]] is not None:
                summary["total_duration_ns"] += int(row[indexes[end]]) - int(row[indexes[start]])
            total_events += 1
    status = "partial" if unresolved_names else "available"
    all_summaries = sorted(summaries.values(), key=lambda item: item["name"])
    memory_api = [
        item
        for item in all_summaries
        if any(token in item["name"].lower() for token in ("mem", "malloc", "alloc", "free"))
    ]
    vmm_pattern = re.compile(
        r"(?:cu|cuda)mem(?:addressreserve|addressfree|create|release|map|unmap|setaccess)",
        re.IGNORECASE,
    )
    vmm_events = [item for item in all_summaries if vmm_pattern.search(item["name"])]
    common = {
        "status": status,
        "reason": "some CUDA API names could not be resolved" if unresolved_names else None,
        "table_count": len(api_tables),
        "tables": api_tables,
        "event_count": total_events,
        "unresolved_name_count": unresolved_names,
    }
    return (
        {**common, "all_api_groups": all_summaries, "memory_api_groups": memory_api},
        {
            **common,
            "event_count": sum(item["event_count"] for item in vmm_events),
            "groups": vmm_events,
            "interpretation": (
                "A zero count is evidence of no named VMM calls only when CUDA API status is "
                "available; an absent API table is unavailable, not zero."
            ),
        },
    )


def _string_lookup(connection: sqlite3.Connection, tables: list[str]) -> dict[int, str]:
    table = _table(tables, exact=("StringIds", "StringId"))
    if table is None:
        return {}
    columns = _columns(connection, table)
    key = _column(columns, "id")
    value = _column(columns, "value", "string", "text")
    if key is None or value is None:
        return {}
    query = f"SELECT {_identifier(key)}, {_identifier(value)} FROM {_identifier(table)}"
    return {int(row[0]): str(row[1]) for row in connection.execute(query)}


def _nvtx_category(
    connection: sqlite3.Connection, tables: list[str], strings: dict[int, str]
) -> dict[str, Any]:
    table = next(
        (
            name
            for name in tables
            if "nvtx" in name.lower()
            and ("event" in name.lower() or "range" in name.lower())
            and not name.lower().startswith("enum")
        ),
        None,
    )
    if table is None:
        return _unavailable("NSYS export contains no NVTX event/range table")
    columns = _columns(connection, table)
    start = _column(columns, "start", "startTimestamp", "timestamp")
    end = _column(columns, "end", "endTimestamp", "stop")
    text = _column(columns, "text", "message", "name")
    text_id = _column(columns, "textId", "nameId", "stringId")
    if start is None or (text is None and text_id is None):
        return _unavailable(f"NVTX table {table!r} lacks start and text/textId columns")
    selected = list(dict.fromkeys(item for item in (start, end, text, text_id) if item))
    indexes = {column: offset for offset, column in enumerate(selected)}
    query = (
        "SELECT "
        + ", ".join(_identifier(item) for item in selected)
        + f" FROM {_identifier(table)} ORDER BY {_identifier(start)}"
    )
    ranges: list[dict[str, Any]] = []
    unresolved = 0
    for row in connection.execute(query):
        if text is not None and row[indexes[text]] is not None:
            label = str(row[indexes[text]])
        else:
            raw_id = _as_int(row[indexes[text_id]]) if text_id is not None else None
            label = strings.get(raw_id, "") if raw_id is not None else ""
        if not label:
            unresolved += 1
            label = "<unresolved>"
        start_ns = int(row[indexes[start]])
        end_ns = int(row[indexes[end]]) if end is not None and row[indexes[end]] is not None else None
        ranges.append(
            {
                "text": label,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": end_ns - start_ns if end_ns is not None else None,
            }
        )
    return {
        "status": "partial" if unresolved else "available",
        "reason": "some NVTX labels could not be resolved" if unresolved else None,
        "table": table,
        "event_count": len(ranges),
        "unresolved_label_count": unresolved,
        "ranges": ranges,
        "phase_attribution": {
            "status": "not_implemented",
            "reason": (
                "Robust allocation-to-nested-range attribution across NSYS schemas is a bounded "
                "follow-up; ranges and allocation timestamps are preserved separately."
            ),
        },
    }


def parse_memory_inventory(path: pathlib.Path) -> dict[str, Any]:
    """Inventory memory evidence without interpreting missing evidence as zero."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValidationError(f"NSYS memory inventory input is missing: {resolved}")
    with closing(sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)) as connection:
        tables = _tables(connection)
        strings = _string_lookup(connection, tables)
        memory, lifetimes, high_water = _memory_categories(connection, tables)
        copies = _copy_category(connection, tables)
        cuda_api, vmm = _api_categories(connection, tables, strings)
        nvtx = _nvtx_category(connection, tables, strings)
    return {
        "schema_version": 1,
        "source": str(resolved),
        "categories": {
            "gpu_memory_events": memory,
            "allocation_lifetimes": lifetimes,
            "captured_device_high_water": high_water,
            "copy_activity": copies,
            "cuda_api": cuda_api,
            "vmm": vmm,
            "nvtx_ranges": nvtx,
        },
        "memory_boundaries": {
            "process_vram": (
                "Measured separately by persistent nvidia-smi telemetry; never inferred from "
                "allocation events."
            ),
            "pinned_host_memory": (
                "CUDA memory events labelled Pinned are captured allocator events, not a proof "
                "of total process pinned bytes and never inferred from VmLck or process VRAM."
            ),
            "ordinary_host_memory": (
                "Measured separately from /proc/PID/status VmRSS/VmHWM; pageable CUDA events do "
                "not replace process host-memory telemetry."
            ),
            "per_phase_device_high_water": (
                "Unavailable automatically: robust NVTX attribution is a bounded follow-up."
            ),
            "allocation_backtraces": (
                "Unavailable automatically: backtrace collection and symbolization are a bounded "
                "follow-up."
            ),
        },
    }


def assess_memory_evidence(
    inventory: dict[str, Any],
    config: dict[str, Any] | None,
    capability: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate preregistered availability; zero events remain valid available evidence."""
    if config is None:
        return {
            "status": "not_requested",
            "mode": None,
            "required_categories": [],
            "issues": [],
        }
    categories = [str(item) for item in config["categories"]]
    issues: list[str] = []
    if capability.get("status") != "supported":
        issues.append("NSYS did not advertise --cuda-memory-usage support")
    available = inventory.get("categories", {})
    for category in categories:
        status = available.get(category, {}).get("status", "unavailable")
        if status != "available":
            issues.append(f"{category} evidence is {status}, not available")
    required = config["mode"] == "required"
    return {
        "status": (
            "failed"
            if required and issues
            else "satisfied"
            if not issues
            else "optional_incomplete"
        ),
        "mode": config["mode"],
        "required_categories": categories if required else [],
        "requested_categories": categories,
        "issues": issues,
        "zero_is_evidence": (
            "A zero event count satisfies availability only when the category table and required "
            "columns/semantics were present."
        ),
    }


def enforce_memory_evidence(assessment: dict[str, Any]) -> None:
    """Fail closed only after the caller has persisted the assessment."""
    if assessment.get("status") == "failed":
        raise ValidationError(
            f"required NSYS memory evidence is unavailable: {assessment.get('issues', [])}"
        )
