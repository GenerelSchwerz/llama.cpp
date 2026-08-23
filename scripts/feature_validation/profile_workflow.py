"""Profiler capture orchestration for the feature-validation CLI."""

from __future__ import annotations

import pathlib
import subprocess
from typing import Any, Callable

from . import core, profiler


def profile_one(
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
    tool_script: pathlib.Path,
    tool_identity: Callable[[str], dict[str, Any]],
    launch_run: Callable[[dict[str, Any], pathlib.Path, pathlib.Path], dict[str, Any]],
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
        nsys_identity = tool_identity("nsys")
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
        result = launch_run(spec, spec_path, tool_script)
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
        ncu_identity = tool_identity("ncu")
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
        result = launch_run(spec, spec_path, tool_script)
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
