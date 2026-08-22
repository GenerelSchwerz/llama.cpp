#!/usr/bin/env python3

"""Manifest-driven feature/performance validation command line interface."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any

from feature_validation import core, profiler
from feature_validation.runner import StudyRunner, execute_run_spec


TOOL_SCRIPT = pathlib.Path(__file__).resolve()


def _load_spec(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise core.ValidationError("run spec must be a schema_version 1 object")
    return value


def _identity_files(provenance: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    files: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for variant in provenance["variants"].values():
        for executable in variant["executables"].values():
            binary = executable["binary"]
            files[binary["resolved_path"]] = {
                "path": binary["resolved_path"],
                **{key: binary[key] for key in ("sha256", "size", "mtime_ns", "device", "inode")},
                "rehash": True,
            }
            for library in binary.get("libraries", []):
                files[library["path"]] = {"path": library["path"], **library}
            if executable["build"].get("cache"):
                build = executable["build"]
                files[build["cache"]] = {
                    "path": build["cache"],
                    "sha256": build["cache_sha256"],
                    **build["cache_stat"],
                }
        source = variant["source"]
        sources.append(
            {
                "root": source["root"],
                "head": source["head"],
                "dirty_fingerprint": source["dirty_fingerprint"],
            }
        )
    for item in provenance["inputs"]:
        files[item["path"]] = {
            "path": item["path"],
            **{key: item[key] for key in ("sha256", "size", "mtime_ns", "device", "inode")},
        }
    return ([files[key] for key in sorted(files)], sources)


def _tool_identity(name: str) -> dict[str, Any]:
    found = shutil.which(name)
    if found is None:
        raise core.ValidationError(f"required optional profiler tool is unavailable: {name}")
    path = pathlib.Path(found).resolve()
    return {
        "path": str(path),
        "sha256": core.sha256_file(path),
        **core.stat_identity(path),
        "rehash": True,
    }


def _launch_locked(spec: dict[str, Any], spec_path: pathlib.Path) -> dict[str, Any]:
    result_path = pathlib.Path(spec["result_path"])
    if spec_path.exists() or result_path.exists():
        raise core.ValidationError(
            f"refusing to overwrite an existing profiler attempt: {spec_path} / {result_path}"
        )
    inner = [sys.executable, str(TOOL_SCRIPT), "_execute-run", str(spec_path)]
    wrapper = core.flock_argv(inner)
    spec["gpu_lock"] = {
        "path": core.GPU_LOCK,
        "whole_command_argv": wrapper,
        "whole_command": core.quote_argv(wrapper),
        "inner_lifecycle": "telemetry preflight, direct profiler lifecycle, telemetry cleanup",
    }
    core.write_json_atomic(spec_path, spec)
    print(f"[feature-validation] launch {core.quote_argv(wrapper)}", flush=True)
    completed = subprocess.run(wrapper, check=False)
    if not result_path.is_file():
        raise core.ValidationError(
            f"locked profiler command returned {completed.returncode} without result artifact"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "success":
        raise core.ValidationError(f"profiler command failed: {result.get('error')}")
    return result


def _profile(manifest_path: pathlib.Path, *, resume: bool, execute_ncu: bool) -> dict[str, Any]:
    manifest, manifest_sha = core.load_and_validate_manifest(manifest_path)
    config = manifest.get("profiler")
    if not isinstance(config, dict):
        raise core.ManifestError("manifest has no profiler section")
    provenance = core.capture_provenance(manifest, manifest_sha)
    stage = core.stage_by_id(manifest, config["stage"])
    if stage["resource"] != "gpu":
        raise core.ManifestError("profiler stage must use resource=gpu")
    variant_name = config["variant"]
    screens = stage.get("screens", [{"id": "default", "args": [], "weight": 1.0}])
    screen_id = config.get("screen", screens[0]["id"])
    if screen_id not in {item["id"] for item in screens}:
        raise core.ManifestError(f"profiler screen {screen_id!r} is not in stage {stage['id']!r}")
    study_root = pathlib.Path(manifest["artifact_root"]) / (
        f"{core.safe_slug(manifest['study']['id'])}-{manifest_sha[:12]}"
    )
    profile_root = study_root / "profiles" / f"{stage['id']}-{variant_name}-{screen_id}"
    state_path = profile_root / "profile-state.json"
    plan_path = profile_root / "ncu-plan.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not resume:
            raise core.ValidationError(f"profile artifacts already exist; use --resume: {profile_root}")
        if state.get("manifest_sha256") != manifest_sha:
            raise core.ProvenanceError("profile resume manifest identity mismatch")
        if state.get("provenance_identity_fingerprint") != provenance["identity_fingerprint"]:
            raise core.ProvenanceError("profile resume provenance identity mismatch")
        if state.get("ncu_executed") or not execute_ncu:
            return json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        if resume:
            raise core.ValidationError(f"no resumable profile exists at {profile_root}")
        profile_root.mkdir(parents=True, exist_ok=False)
        state = {
            "schema_version": 1,
            "manifest_sha256": manifest_sha,
            "provenance_identity_fingerprint": provenance["identity_fingerprint"],
            "stage": stage["id"],
            "variant": variant_name,
            "screen": screen_id,
            "nsys_discovery_completed": False,
            "ncu_plan_completed": False,
            "ncu_executed": False,
        }
        core.write_json_atomic(profile_root / "provenance.json", provenance)
        core.write_json_atomic(state_path, state)

    scheduled = next(
        item
        for item in core.build_schedule(manifest, stage)
        if item["variant"] == variant_name and item["screen"] == screen_id
    )
    target_argv, environment = core.command_for_run(manifest, stage, scheduled)
    context = {
        "artifact_dir": profile_root,
        "variant": variant_name,
        "pair": 1,
        "screen": screen_id,
        "run_id": "profile-discovery",
    }
    target_argv = [value.format_map(context) for value in target_argv]
    executable_role = str(stage["command"].get("executable_role", "default"))
    direct_harness = core.variant_executables(manifest["variants"][variant_name])[
        executable_role
    ].get("direct_harness")
    profiler.validate_native_affinity(target_argv, direct_harness)
    identity_files, source_identities = _identity_files(provenance)

    sqlite_path = profile_root / "discovery.sqlite"
    if not state["nsys_discovery_completed"]:
        nsys_identity = _tool_identity("nsys")
        identity_files_with_nsys = [*identity_files, nsys_identity]
        prefix = profile_root / "discovery"
        nsys_argv = profiler.nsys_discovery_argv(target_argv, prefix, direct_harness)
        spec_path = profile_root / "nsys-run-spec.json"
        spec = {
            "schema_version": 1,
            "run_id": f"profile--{stage['id']}--nsys-discovery",
            "attempt": 1,
            "resource": "gpu",
            "argv": nsys_argv,
            "environment": environment,
            "cwd": str(profile_root),
            "timeout_seconds": float(config.get("timeout_seconds", stage["timeout_seconds"])),
            "progress": "NSYS target output plus five-second runner heartbeat",
            "result_path": str(profile_root / "nsys-result.json"),
            "stdout_path": str(profile_root / "nsys-stdout.log"),
            "stderr_path": str(profile_root / "nsys-stderr.log"),
            "identity_files": identity_files_with_nsys,
            "source_identities": source_identities,
            "host_load_before": core.host_load_snapshot(),
        }
        _launch_locked(spec, spec_path)
        report_path = prefix.with_suffix(".nsys-rep")
        if not report_path.is_file():
            raise core.ValidationError(f"NSYS report is missing: {report_path}")
        export_argv = [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite",
            "true",
            "--output",
            str(sqlite_path),
            str(report_path),
        ]
        print(f"[feature-validation] NSYS export {core.quote_argv(export_argv)}", flush=True)
        exported = subprocess.run(export_argv, check=False, capture_output=True, text=True)
        core.write_json_atomic(
            profile_root / "nsys-export.json",
            {
                "argv": export_argv,
                "command": core.quote_argv(export_argv),
                "returncode": exported.returncode,
                "stdout": exported.stdout,
                "stderr": exported.stderr,
            },
        )
        if exported.returncode != 0 or not sqlite_path.is_file():
            raise core.ValidationError(f"NSYS SQLite export failed: {exported.stderr.strip()}")
        state["nsys_discovery_completed"] = True
        core.write_json_atomic(state_path, state)

    discovery = profiler.parse_discovery(sqlite_path)
    core.write_json_atomic(profile_root / "discovery-inventory.json", discovery)
    plan = profiler.build_ncu_plan(
        discovery,
        config["selector"],
        target_argv,
        profile_root / "ncu",
        metrics=[str(value) for value in config.get("metrics", [])],
        direct_harness=direct_harness,
    )
    plan["stage"] = stage["id"]
    plan["variant"] = variant_name
    plan["screen"] = screen_id
    plan["profile_repetitions"] = 1
    core.write_json_atomic(plan_path, plan)
    state["ncu_plan_completed"] = True
    core.write_json_atomic(state_path, state)

    if execute_ncu and not state["ncu_executed"]:
        ncu_identity = _tool_identity("ncu")
        command = plan["command"]["direct_argv"]
        spec_path = profile_root / "ncu-run-spec.json"
        spec = {
            "schema_version": 1,
            "run_id": f"profile--{stage['id']}--filtered-ncu",
            "attempt": 1,
            "resource": "gpu",
            "argv": command,
            "environment": environment,
            "cwd": str(profile_root),
            "timeout_seconds": float(config.get("timeout_seconds", stage["timeout_seconds"])),
            "progress": "filtered NCU target output plus five-second runner heartbeat",
            "result_path": str(profile_root / "ncu-result.json"),
            "stdout_path": str(profile_root / "ncu-stdout.log"),
            "stderr_path": str(profile_root / "ncu-stderr.log"),
            "identity_files": [*identity_files, ncu_identity],
            "source_identities": source_identities,
            "host_load_before": core.host_load_snapshot(),
        }
        _launch_locked(spec, spec_path)
        state["ncu_executed"] = True
        core.write_json_atomic(state_path, state)
    return plan


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate schema and fail-closed provenance")
    validate.add_argument("manifest", type=pathlib.Path)
    run = subparsers.add_parser("run", help="run validation stages")
    run.add_argument("manifest", type=pathlib.Path)
    run.add_argument("--through", choices=("early", "production", "acceptance"), default="early")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    profile = subparsers.add_parser("profile", help="one NSYS discovery and one filtered NCU plan/capture")
    profile.add_argument("manifest", type=pathlib.Path)
    profile.add_argument("--resume", action="store_true")
    profile.add_argument("--execute-ncu", action="store_true")
    internal = subparsers.add_parser("_execute-run", help=argparse.SUPPRESS)
    internal.add_argument("spec", type=pathlib.Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "validate":
            manifest, manifest_sha = core.load_and_validate_manifest(args.manifest)
            provenance = core.capture_provenance(manifest, manifest_sha)
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "manifest_sha256": manifest_sha,
                        "provenance_identity_fingerprint": provenance["identity_fingerprint"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "run":
            runner = StudyRunner(args.manifest, TOOL_SCRIPT)
            summary = runner.run(
                through=args.through,
                resume=args.resume,
                retry_failed=args.retry_failed,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif args.command == "profile":
            plan = _profile(args.manifest, resume=args.resume, execute_ncu=args.execute_ncu)
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            result = execute_run_spec(_load_spec(args.spec))
            return 0 if result["status"] == "success" else 1
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        print(f"feature-validation: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
