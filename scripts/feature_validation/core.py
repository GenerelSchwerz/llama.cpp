"""Provenance, scheduling, statistics, and runtime helpers for validation.

Manifest and command contracts live in :mod:`feature_validation.contracts`
and are re-exported here to preserve the established ``core`` API. Commands
remain argv lists until flock's ``-c`` interface requires shell quoting.
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


BUILD_PROVENANCE_SCHEMA_VERSION = 1
BUILD_PROVENANCE_KIND = "beellama-feature-validation-cmake-build"
GPU_LOCK = "/tmp/beellama-single-gpu.lock"


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
