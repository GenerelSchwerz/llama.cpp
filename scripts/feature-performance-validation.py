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
from feature_validation.runner import StudyRunner, execute_run_spec, launch_locked_spec


TOOL_SCRIPT = pathlib.Path(__file__).resolve()


def _load_spec(path: pathlib.Path) -> dict[str, Any]:
    value = core.load_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise core.ValidationError("run spec must be a schema_version 1 object")
    return value


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


def _profile_one(
    manifest: dict[str, Any],
    manifest_sha: str,
    provenance: dict[str, Any],
    config: dict[str, Any],
    *,
    variant_name: str,
    screen_id: str,
    profile_base: pathlib.Path,
    resume: bool,
    execute_ncu: bool,
) -> dict[str, Any]:
    stage = core.stage_by_id(manifest, config["stage"])
    if stage["resource"] != "gpu":
        raise core.ManifestError("profiler stage must use resource=gpu")
    screens = stage.get("screens", [{"id": "default", "args": [], "weight": 1.0}])
    if screen_id not in {item["id"] for item in screens}:
        raise core.ManifestError(f"profiler screen {screen_id!r} is not in stage {stage['id']!r}")
    profile_root = profile_base / f"{stage['id']}-{variant_name}-{screen_id}"
    state_path = profile_root / "profile-state.json"
    plan_path = profile_root / "ncu-plan.json"
    if state_path.exists():
        core.validate_artifact_directory_isolation(manifest, profile_root)
        state = core.load_json(state_path)
        if not resume:
            raise core.ValidationError(f"profile artifacts already exist; use --resume: {profile_root}")
        if state.get("manifest_sha256") != manifest_sha:
            raise core.ProvenanceError("profile resume manifest identity mismatch")
        if state.get("provenance_identity_fingerprint") != provenance["identity_fingerprint"]:
            raise core.ProvenanceError("profile resume provenance identity mismatch")
        if state.get("ncu_executed"):
            if state.get("ncu_verification_status") != "verified":
                raise core.ValidationError(
                    "the preserved NCU capture is failed or unverifiable; start a new profile identity"
                )
    else:
        if resume:
            raise core.ValidationError(f"no resumable profile exists at {profile_root}")
        profile_root.mkdir(parents=True, exist_ok=False)
        core.validate_artifact_directory_isolation(manifest, profile_root)
        state = {
            "schema_version": 1,
            "manifest_sha256": manifest_sha,
            "provenance_identity_fingerprint": provenance["identity_fingerprint"],
            "stage": stage["id"],
            "variant": variant_name,
            "screen": screen_id,
            "evidence_claim": config.get("evidence_claim", "kernel_investigation_only"),
            "nsys_discovery_completed": False,
            "memory_evidence_status": "not_evaluated",
            "ncu_plan_completed": False,
            "ncu_executed": False,
            "ncu_verification_status": "not_executed",
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
    identity_files, source_identities = core.provenance_identity_spec(
        provenance, [(variant_name, executable_role)]
    )
    nvidia_smi_identity = provenance.get("runtime_tools", {}).get("nvidia-smi")
    nvidia_smi = nvidia_smi_identity["path"] if nvidia_smi_identity else "nvidia-smi"

    sqlite_path = profile_root / "discovery.sqlite"
    memory_capability_path = profile_root / "nsys-memory-trace-capability.json"
    memory_config = config.get("memory_evidence")
    memory_capability: dict[str, Any]
    if not state["nsys_discovery_completed"]:
        nsys_identity = _tool_identity("nsys")
        nsys_help = core.command_capture(
            [nsys_identity["path"], "profile", "--help=cuda"]
        )
        nsys_version = core.command_capture([nsys_identity["path"], "--version"])
        if (
            nsys_help.get("returncode") != 0
            and (
                config["cuda_graph_trace"]["applicability"] == "required"
                or (memory_config is not None and memory_config["mode"] == "required")
            )
        ):
            raise core.ValidationError(
                f"cannot inspect NSYS CUDA tracing support: {nsys_help.get('stderr', nsys_help.get('error'))}"
            )
        graph_capability = profiler.verify_nsys_graph_trace_capability(
            config["cuda_graph_trace"],
            help_text=str(nsys_help.get("stdout", "")) + str(nsys_help.get("stderr", "")),
            version_text=str(nsys_version.get("stdout", "")) + str(nsys_version.get("stderr", "")),
        )
        core.write_json_atomic(
            profile_root / "nsys-graph-trace-capability.json",
            {
                **graph_capability,
                "help_command": nsys_help,
                "version_command": nsys_version,
            },
        )
        memory_capability = profiler.verify_nsys_memory_trace_capability(
            memory_config,
            help_text=str(nsys_help.get("stdout", ""))
            + str(nsys_help.get("stderr", "")),
            version_text=str(nsys_version.get("stdout", ""))
            + str(nsys_version.get("stderr", "")),
        )
        core.write_json_atomic(
            memory_capability_path,
            {
                **memory_capability,
                "help_command": nsys_help,
                "version_command": nsys_version,
            },
        )
        identity_files_with_nsys = [*identity_files, nsys_identity]
        prefix = profile_root / "discovery"
        nsys_argv = profiler.nsys_discovery_argv(
            target_argv,
            prefix,
            direct_harness,
            cuda_graph_trace=config["cuda_graph_trace"],
            request_cuda_memory_usage=memory_capability["status"] == "supported",
            nsys_executable=nsys_identity["path"],
        )
        spec_path = profile_root / "nsys-run-spec.json"
        spec = {
            "schema_version": 1,
            "run_id": f"profile--{stage['id']}--nsys-discovery",
            "attempt": 1,
            "study_identity": {
                "manifest_sha256": manifest_sha,
                "provenance_identity_fingerprint": provenance["identity_fingerprint"],
            },
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
            "nvidia_smi_identity": nvidia_smi_identity,
            "nvidia_smi": nvidia_smi,
            "host_load_before": core.host_load_snapshot(),
        }
        core.validate_artifact_directory_isolation(manifest, profile_root)
        result = launch_locked_spec(spec, spec_path, TOOL_SCRIPT)
        if result.get("status") != "success":
            raise core.ValidationError(f"NSYS command failed: {result.get('error')}")
        report_path = prefix.with_suffix(".nsys-rep")
        if not report_path.is_file():
            raise core.ValidationError(f"NSYS report is missing: {report_path}")
        export_argv = [
            nsys_identity["path"],
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
    elif memory_capability_path.is_file():
        memory_capability = core.load_json(memory_capability_path)
    elif memory_config is not None:
        raise core.ValidationError(
            "resumed NSYS discovery predates configured memory capability evidence; "
            "start a new profile identity"
        )
    else:
        memory_capability = {
            "requested": False,
            "mode": None,
            "categories": [],
            "status": "not_configured",
        }

    discovery = profiler.parse_discovery(sqlite_path)
    discovery["graph_node_verification"] = profiler.verify_graph_node_discovery(
        discovery, config["cuda_graph_trace"]
    )
    core.write_json_atomic(profile_root / "discovery-inventory.json", discovery)
    memory_inventory = profiler.parse_memory_inventory(sqlite_path)
    memory_assessment = profiler.assess_memory_evidence(
        memory_inventory, memory_config, memory_capability
    )
    memory_inventory["request"] = memory_capability
    memory_inventory["assessment"] = memory_assessment
    core.write_json_atomic(profile_root / "memory-inventory.json", memory_inventory)
    state["memory_evidence_status"] = memory_assessment["status"]
    core.write_json_atomic(state_path, state)
    profiler.enforce_memory_evidence(memory_assessment)
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
    plan["evidence_claim"] = config.get("evidence_claim", "kernel_investigation_only")
    core.write_json_atomic(plan_path, plan)
    state["ncu_plan_completed"] = True
    core.write_json_atomic(state_path, state)

    if execute_ncu and not state["ncu_executed"]:
        ncu_identity = _tool_identity("ncu")
        command = [ncu_identity["path"], *plan["command"]["direct_argv"][1:]]
        spec_path = profile_root / "ncu-run-spec.json"
        spec = {
            "schema_version": 1,
            "run_id": f"profile--{stage['id']}--filtered-ncu",
            "attempt": 1,
            "study_identity": {
                "manifest_sha256": manifest_sha,
                "provenance_identity_fingerprint": provenance["identity_fingerprint"],
            },
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
            "nvidia_smi_identity": nvidia_smi_identity,
            "nvidia_smi": nvidia_smi,
            "host_load_before": core.host_load_snapshot(),
        }
        core.validate_artifact_directory_isolation(manifest, profile_root)
        result = launch_locked_spec(spec, spec_path, TOOL_SCRIPT)
        if result.get("status") != "success":
            raise core.ValidationError(f"NCU command failed: {result.get('error')}")
        state["ncu_executed"] = True
        report_path = pathlib.Path(plan["command"]["report_path"])
        verification: dict[str, Any]
        if not report_path.is_file() or report_path.stat().st_size == 0:
            verification = {
                "status": "unverifiable",
                "issues": [f"NCU report is missing or empty: {report_path}"],
                "report_path": str(report_path),
            }
        else:
            import_argv = [
                ncu_identity["path"],
                "--import",
                str(report_path),
                "--csv",
                "--page",
                "raw",
                "--print-kernel-base",
                "demangled",
            ]
            imported = core.command_capture(import_argv, cwd=profile_root, timeout=120.0)
            if imported.get("returncode") != 0:
                verification = {
                    "status": "unverifiable",
                    "issues": ["NCU report import failed"],
                }
            else:
                parsed = profiler.parse_ncu_raw_csv(
                    str(imported.get("stdout", "")) + "\n" + str(imported.get("stderr", ""))
                )
                verification = profiler.verify_ncu_capture(plan, parsed)
                verification["parsed_export"] = parsed
            verification.update(
                {
                    "report_path": str(report_path),
                    "report_sha256": core.sha256_file(report_path),
                    "report_size": report_path.stat().st_size,
                    "import_command": imported,
                }
            )
        core.write_json_atomic(profile_root / "ncu-verification.json", verification)
        state["ncu_verification_status"] = verification["status"]
        core.write_json_atomic(state_path, state)
        if verification["status"] != "verified":
            raise core.ValidationError(
                f"NCU capture is {verification['status']}: {verification.get('issues')}"
            )
    return plan


def _study_root(manifest: dict[str, Any], manifest_sha: str) -> pathlib.Path:
    return pathlib.Path(manifest["artifact_root"]) / (
        f"{core.safe_slug(manifest['study']['id'])}-{manifest_sha[:12]}"
    )


def _register_build(
    source_root: pathlib.Path,
    executable: pathlib.Path,
    cache: pathlib.Path,
    output: pathlib.Path | None,
    *,
    force: bool,
) -> dict[str, Any]:
    resolved_root = source_root.expanduser().resolve()
    resolved_executable = executable.expanduser().resolve()
    resolved_cache = cache.expanduser().resolve()
    resolved_output = (
        output.expanduser().resolve()
        if output is not None
        else pathlib.Path(f"{resolved_executable}.build-provenance.json")
    )
    if resolved_output in {resolved_executable, resolved_cache}:
        raise core.ValidationError("build provenance sidecar cannot overwrite the executable or cache")
    if resolved_output.exists() and not force:
        raise core.ValidationError(
            f"build provenance sidecar already exists; pass --force to replace it: {resolved_output}"
        )
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError:
        pass
    else:
        ignored = subprocess.run(
            [
                "git",
                "-C",
                str(resolved_root),
                "check-ignore",
                "-q",
                "--no-index",
                str(resolved_output),
            ],
            check=False,
        )
        if ignored.returncode != 0:
            raise core.ValidationError(
                "a sidecar inside source_root must be Git-ignored so registration does not alter source identity"
            )
    sidecar = core.capture_cmake_build_provenance(
        resolved_root,
        resolved_executable,
        resolved_cache,
    )
    core.write_json_atomic(resolved_output, sidecar)
    return {
        "status": "registered",
        "sidecar": {
            "path": str(resolved_output),
            "sha256": core.sha256_file(resolved_output),
        },
        "identity_fingerprint": sidecar["identity_fingerprint"],
        "source_root": sidecar["source"]["root"],
        "source_commit": sidecar["source"]["head"],
        "source_dirty_fingerprint": sidecar["source"]["dirty_fingerprint"],
        "source_file_count": sidecar["source"]["source_files"]["count"],
        "executable_sha256": sidecar["executable"]["sha256"],
        "cache_sha256": sidecar["build"]["cache_sha256"],
    }


def _profile(manifest_path: pathlib.Path, *, resume: bool, execute_ncu: bool) -> dict[str, Any]:
    manifest, manifest_sha = core.load_and_validate_manifest(manifest_path)
    config = manifest.get("profiler")
    if not isinstance(config, dict):
        raise core.ManifestError("manifest has no profiler section")
    provenance = core.capture_provenance(manifest, manifest_sha)
    stage = core.stage_by_id(manifest, config["stage"])
    screens = stage.get("screens", [{"id": "default"}])
    return _profile_one(
        manifest,
        manifest_sha,
        provenance,
        config,
        variant_name=str(config["variant"]),
        screen_id=str(config.get("screen", screens[0]["id"])),
        profile_base=_study_root(manifest, manifest_sha) / "profiles",
        resume=resume,
        execute_ncu=execute_ncu,
    )


def _diagnose(manifest_path: pathlib.Path, *, resume: bool) -> dict[str, Any]:
    manifest, manifest_sha = core.load_and_validate_manifest(manifest_path)
    configs = manifest.get("early_diagnostics")
    if not isinstance(configs, list) or not configs:
        raise core.ManifestError("manifest has no early_diagnostics section")
    provenance = core.capture_provenance(manifest, manifest_sha)
    study_root = _study_root(manifest, manifest_sha)
    state_path = study_root / "state.json"
    if not state_path.is_file():
        raise core.ValidationError("early diagnostics require a preserved screening study state")
    state = core.load_json(state_path)
    if state.get("manifest_sha256") != manifest_sha:
        raise core.ProvenanceError("diagnostic study manifest identity mismatch")
    if state.get("provenance_identity_fingerprint") != provenance["identity_fingerprint"]:
        raise core.ProvenanceError("diagnostic study provenance identity mismatch")

    diagnostic_root = study_root / "early-diagnostics"
    report_path = diagnostic_root / "agent-diagnostic-report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "not_triggered",
        "generated_utc": core.utc_now(),
        "manifest_sha256": manifest_sha,
        "provenance_identity_fingerprint": provenance["identity_fingerprint"],
        "trigger_policy": "regression_signal_only",
        "profiles": [],
        "investigations": [],
        "normal_early_screen_profiler_cost": "zero when no regression signal is emitted",
        "evidence_boundary": (
            "These profiles are diagnostic-only and cannot pass correctness, statistical "
            "performance, resource, production, or long-context acceptance gates."
        ),
    }
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    core.validate_artifact_directory_isolation(manifest, diagnostic_root)
    core.write_json_atomic(report_path, report)
    for config in configs:
        stage = core.stage_by_id(manifest, str(config["stage"]))
        trigger = profiler.select_regression_diagnostic(
            stage,
            state.get("stages", {}).get(stage["id"]),
            config["screen_selection"],
        )
        investigation = {
            "id": config["id"],
            "stage": stage["id"],
            "trigger": trigger,
            "status": "not_triggered",
            "variant_profiles": {},
        }
        report["investigations"].append(investigation)
        core.write_json_atomic(report_path, report)
        if not trigger["triggered"]:
            continue
        report["status"] = "running"
        investigation["status"] = "running"
        screen_id = str(trigger["selected_screen"]["screen"])
        profile_base = diagnostic_root / str(config["id"])
        for variant_name in config["variants"]:
            profile_root = profile_base / f"{stage['id']}-{variant_name}-{screen_id}"
            try:
                _profile_one(
                    manifest,
                    manifest_sha,
                    provenance,
                    config,
                    variant_name=str(variant_name),
                    screen_id=screen_id,
                    profile_base=profile_base,
                    resume=resume and profile_root.exists(),
                    execute_ncu=True,
                )
                compact = profiler.agent_profile_summary(profile_root)
                if compact["status"] != "verified":
                    raise core.ValidationError(
                        f"{variant_name} diagnostic profile is {compact['status']}"
                    )
                investigation["variant_profiles"][variant_name] = compact
                report["profiles"].append(compact)
                core.write_json_atomic(report_path, report)
            except Exception as error:
                investigation["status"] = "failed"
                investigation["error"] = f"{type(error).__name__}: {error}"
                report["status"] = "failed"
                core.write_json_atomic(report_path, report)
                raise
        investigation["status"] = "verified"
        investigation["cross_variant_discovery"] = (
            "independent; no launch index, kernel spelling, or shape was reused across variants"
        )
        investigation["agent_comparison"] = profiler.compare_agent_profiles(
            investigation["variant_profiles"]["baseline"],
            investigation["variant_profiles"]["candidate"],
        )
        report["status"] = "verified"
        core.write_json_atomic(report_path, report)
    report["generated_utc"] = core.utc_now()
    core.write_json_atomic(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser(
        "register-build",
        help="bind one CMake executable to its exact source, cache, libraries, and final path",
    )
    register.add_argument("--source-root", required=True, type=pathlib.Path)
    register.add_argument("--executable", required=True, type=pathlib.Path)
    register.add_argument("--cache", required=True, type=pathlib.Path)
    register.add_argument("--output", type=pathlib.Path)
    register.add_argument("--force", action="store_true")
    validate = subparsers.add_parser("validate", help="validate schema and fail-closed provenance")
    validate.add_argument("manifest", type=pathlib.Path)
    run = subparsers.add_parser("run", help="run validation stages")
    run.add_argument("manifest", type=pathlib.Path)
    run.add_argument("--through", choices=("early", "production", "acceptance"), default="early")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--retry-failed", action="store_true")
    run.add_argument(
        "--diagnose-regressions",
        action="store_true",
        help=(
            "after a preregistered early regression, run independent "
            "baseline/candidate NSYS+NCU diagnostics"
        ),
    )
    profile = subparsers.add_parser(
        "profile", help="one NSYS discovery and one filtered NCU plan/capture"
    )
    profile.add_argument("manifest", type=pathlib.Path)
    profile.add_argument("--resume", action="store_true")
    profile.add_argument("--execute-ncu", action="store_true")
    diagnose = subparsers.add_parser(
        "diagnose", help="run preregistered NSYS+NCU diagnostics for a preserved early regression"
    )
    diagnose.add_argument("manifest", type=pathlib.Path)
    diagnose.add_argument("--resume", action="store_true")
    internal = subparsers.add_parser("_execute-run", help=argparse.SUPPRESS)
    internal.add_argument("spec", type=pathlib.Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "register-build":
            registration = _register_build(
                args.source_root,
                args.executable,
                args.cache,
                args.output,
                force=args.force,
            )
            print(json.dumps(registration, indent=2, sort_keys=True))
        elif args.command == "validate":
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
            try:
                summary = runner.run(
                    through=args.through,
                    resume=args.resume,
                    retry_failed=args.retry_failed,
                )
            except core.ValidationError as run_error:
                if args.diagnose_regressions:
                    try:
                        diagnostic = _diagnose(args.manifest, resume=args.resume)
                        print(json.dumps(diagnostic, indent=2, sort_keys=True))
                    except Exception as diagnostic_error:
                        raise core.ValidationError(
                            f"{run_error}; regression diagnostic failed: "
                            f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                        ) from diagnostic_error
                raise
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif args.command == "profile":
            plan = _profile(args.manifest, resume=args.resume, execute_ncu=args.execute_ncu)
            print(json.dumps(plan, indent=2, sort_keys=True))
        elif args.command == "diagnose":
            report = _diagnose(args.manifest, resume=args.resume)
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            result = execute_run_spec(_load_spec(args.spec))
            return 0 if result["status"] == "success" else 1
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        print(f"feature-validation: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
