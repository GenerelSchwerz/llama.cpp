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

from feature_validation import core, profiler, profile_workflow
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
    return profile_workflow.profile_one(
        manifest,
        manifest_sha,
        provenance,
        config,
        variant_name=variant_name,
        screen_id=screen_id,
        profile_base=profile_base,
        resume=resume,
        execute_ncu=execute_ncu,
        tool_script=TOOL_SCRIPT,
        tool_identity=_tool_identity,
        launch_run=launch_locked_spec,
    )


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
