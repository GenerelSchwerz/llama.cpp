"""Study execution, clean-process isolation, adaptive pairs, and safe resume."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from . import core
from .telemetry import TelemetrySession


def _render(value: str, context: dict[str, Any]) -> str:
    try:
        return value.format_map({key: str(item) for key, item in context.items()})
    except KeyError as error:
        raise core.ValidationError(f"unknown command template field {error.args[0]!r}") from error


def _environment_command(environment: dict[str, str], argv: list[str]) -> list[str]:
    return ["env", "-i", *[f"{key}={environment[key]}" for key in sorted(environment)], *argv]


def _kill_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=3.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.wait(timeout=3.0)


def _drain(stream: Any, chunks: list[bytes], label: str, output_lock: threading.Lock) -> None:
    while True:
        chunk = stream.readline()
        if not chunk:
            return
        chunks.append(chunk)
        with output_lock:
            sys.stdout.buffer.write(f"[{label}] ".encode("utf-8") + chunk)
            if not chunk.endswith(b"\n"):
                sys.stdout.buffer.write(b"\n")
            sys.stdout.buffer.flush()


def _verify_spec_identity(spec: dict[str, Any]) -> None:
    for item in spec.get("identity_files", []):
        path = pathlib.Path(item["path"]).resolve()
        if not path.is_file():
            raise core.ProvenanceError(f"run identity file disappeared: {path}")
        stat = core.stat_identity(path)
        for key in ("size", "mtime_ns", "device", "inode"):
            if key in item and stat[key] != item[key]:
                raise core.ProvenanceError(
                    f"run identity metadata changed for {path}: {key} expected {item[key]}, observed {stat[key]}"
                )
        if item.get("rehash", False) or "mtime_ns" not in item:
            observed = core.sha256_file(path)
            if observed != item["sha256"]:
                raise core.ProvenanceError(
                    f"run identity changed for {path}: expected {item['sha256']}, observed {observed}"
                )
    for item in spec.get("source_identities", []):
        observed = core.git_snapshot(pathlib.Path(item["root"]))
        for key in ("head", "dirty_fingerprint"):
            if observed[key] != item[key]:
                raise core.ProvenanceError(
                    f"source identity changed for {item['root']}: {key} expected {item[key]}, observed {observed[key]}"
                )


def execute_run_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one fresh direct process.  GPU callers invoke this under flock."""
    output_path = pathlib.Path(spec["result_path"])
    stdout_path = pathlib.Path(spec["stdout_path"])
    stderr_path = pathlib.Path(spec["stderr_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = core.utc_now()
    start_monotonic = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    telemetry = TelemetrySession(
        spec["resource"],
        nvidia_smi=spec.get("nvidia_smi", "nvidia-smi"),
        first_sample_timeout=float(spec.get("telemetry_first_sample_timeout", 12.0)),
    )
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    threads: list[threading.Thread] = []
    error: str | None = None
    timed_out = False
    returncode: int | None = None
    telemetry_report: dict[str, Any] | None = None
    try:
        _verify_spec_identity(spec)
        telemetry.start()
        argv = [str(value) for value in spec["argv"]]
        environment = {str(key): str(value) for key, value in spec["environment"].items()}
        process = subprocess.Popen(
            argv,
            cwd=spec.get("cwd"),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        telemetry.attach(process.pid)
        print(
            f"[feature-validation] started run={spec['run_id']} pid={process.pid} "
            f"timeout={spec['timeout_seconds']}s progress={spec['progress']}",
            flush=True,
        )
        output_lock = threading.Lock()
        assert process.stdout is not None and process.stderr is not None
        threads = [
            threading.Thread(
                target=_drain,
                args=(process.stdout, stdout_chunks, "target stdout", output_lock),
                daemon=True,
            ),
            threading.Thread(
                target=_drain,
                args=(process.stderr, stderr_chunks, "target stderr", output_lock),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        next_heartbeat = time.monotonic() + 5.0
        timeout = float(spec["timeout_seconds"])
        while process.poll() is None:
            elapsed = time.monotonic() - start_monotonic
            if elapsed > timeout:
                timed_out = True
                _kill_group(process)
                break
            if time.monotonic() >= next_heartbeat:
                print(
                    f"[feature-validation] progress run={spec['run_id']} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                next_heartbeat += 5.0
            time.sleep(0.05)
        returncode = process.wait()
    except Exception as caught:  # the result must preserve every failed attempt
        error = f"{type(caught).__name__}: {caught}"
        if process is not None:
            _kill_group(process)
            returncode = process.poll()
    finally:
        for thread in threads:
            thread.join(timeout=3.0)
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        try:
            telemetry_report = telemetry.stop()
        except Exception as caught:
            telemetry_error = f"{type(caught).__name__}: {caught}"
            error = f"{error}; telemetry cleanup: {telemetry_error}" if error else telemetry_error
    stdout = b"".join(stdout_chunks)
    stderr = b"".join(stderr_chunks)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    contamination = bool(
        spec["resource"] == "gpu"
        and telemetry_report
        and not telemetry_report["clean_process_evidence"]["clean_throughout"]
    )
    if contamination and error is None:
        error = "GPU clean-process evidence was contaminated"
    status = "success" if returncode == 0 and not timed_out and error is None else "failed"
    result: dict[str, Any] = {
        "run_id": spec["run_id"],
        "status": status,
        "attempt": spec["attempt"],
        "schedule": spec.get("schedule"),
        "started_utc": started,
        "finished_utc": core.utc_now(),
        "elapsed_seconds": time.monotonic() - start_monotonic,
        "pid": process.pid if process is not None else None,
        "returncode": returncode,
        "timed_out": timed_out,
        "error": error,
        "argv": spec["argv"],
        "direct_command": core.quote_argv(spec["argv"]),
        "environment": spec["environment"],
        "environment_command": core.quote_argv(_environment_command(spec["environment"], spec["argv"])),
        "gpu_lock": spec.get("gpu_lock"),
        "progress": spec["progress"],
        "host_load_before": spec["host_load_before"],
        "host_load_after": core.host_load_snapshot(),
        "stdout": {
            "path": str(stdout_path),
            "size": len(stdout),
            "sha256": hashlib.sha256(stdout).hexdigest(),
        },
        "stderr": {
            "path": str(stderr_path),
            "size": len(stderr),
            "sha256": hashlib.sha256(stderr).hexdigest(),
        },
        "telemetry": telemetry_report,
    }
    if status == "success" and spec.get("metric"):
        try:
            result["metric_value"] = core.extract_metric(
                spec["metric"],
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )
        except core.ValidationError as caught:
            result["status"] = "failed"
            result["error"] = f"ValidationError: {caught}"
    comparison_path = spec.get("comparison_path")
    if result["status"] == "success" and comparison_path:
        path = pathlib.Path(comparison_path)
        if not path.is_file():
            result["status"] = "failed"
            result["error"] = f"comparison artifact is missing: {path}"
        else:
            result["comparison_artifact"] = {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": core.sha256_file(path),
            }
    core.write_json_atomic(output_path, result)
    print(
        f"[feature-validation] finished run={spec['run_id']} status={result['status']} "
        f"elapsed={result['elapsed_seconds']:.3f}s",
        flush=True,
    )
    return result


class StudyRunner:
    def __init__(self, manifest_path: pathlib.Path, tool_script: pathlib.Path) -> None:
        self.manifest_path = manifest_path.expanduser().resolve()
        self.manifest, self.manifest_sha256 = core.load_and_validate_manifest(self.manifest_path)
        self.tool_script = tool_script.expanduser().resolve()
        study_id = core.safe_slug(str(self.manifest["study"]["id"]))
        self.study_root = pathlib.Path(self.manifest["artifact_root"]) / (
            f"{study_id}-{self.manifest_sha256[:12]}"
        )
        self.state_path = self.study_root / "state.json"
        self.provenance: dict[str, Any] | None = None
        self.state: dict[str, Any] | None = None

    def prepare(self, *, resume: bool) -> None:
        provenance = core.capture_provenance(self.manifest, self.manifest_sha256)
        if self.state_path.exists():
            if not resume:
                raise core.ValidationError(
                    f"deterministic study directory already exists; use --resume: {self.study_root}"
                )
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("manifest_sha256") != self.manifest_sha256:
                raise core.ProvenanceError("resume manifest identity mismatch")
            if state.get("provenance_identity_fingerprint") != provenance["identity_fingerprint"]:
                raise core.ProvenanceError("resume provenance identity mismatch")
            self.state = state
        else:
            if resume:
                raise core.ValidationError(f"no resumable study exists at {self.study_root}")
            self.study_root.mkdir(parents=True, exist_ok=False)
            core.write_json_atomic(self.study_root / "manifest.snapshot.json", self.manifest)
            core.write_json_atomic(self.study_root / "provenance.json", provenance)
            self.state = {
                "schema_version": 1,
                "study_id": self.manifest["study"]["id"],
                "manifest_sha256": self.manifest_sha256,
                "provenance_identity_fingerprint": provenance["identity_fingerprint"],
                "created_utc": core.utc_now(),
                "updated_utc": core.utc_now(),
                "runs": {},
                "stages": {},
            }
            self._save_state()
        self.provenance = provenance

    def _save_state(self) -> None:
        assert self.state is not None
        self.state["updated_utc"] = core.utc_now()
        core.write_json_atomic(self.state_path, self.state)

    def _identity_spec(
        self, stage: dict[str, Any], scheduled: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        assert self.provenance is not None
        files: dict[str, dict[str, Any]] = {}
        sources: list[dict[str, str]] = []
        variant_name = scheduled["variant"]
        variant = self.provenance["variants"][variant_name]
        role = str(stage["command"].get("executable_role", "default"))
        executable = variant["executables"][role]
        binary = executable["binary"]
        files[binary["resolved_path"]] = {
            "path": binary["resolved_path"],
            **{key: binary[key] for key in ("sha256", "size", "mtime_ns", "device", "inode")},
            "rehash": True,
        }
        for library in binary.get("libraries", []):
            files[library["path"]] = {"path": library["path"], **library}
        build = executable["build"]
        if build.get("cache"):
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
        for item in self.provenance["inputs"]:
            files[item["path"]] = {
                "path": item["path"],
                **{key: item[key] for key in ("sha256", "size", "mtime_ns", "device", "inode")},
            }
        return (
            [files[path] for path in sorted(files)],
            sources,
        )

    def _execute(
        self,
        stage: dict[str, Any],
        scheduled: dict[str, Any],
        *,
        retry_failed: bool,
    ) -> dict[str, Any]:
        assert self.state is not None
        run_id = f"{stage['id']}--{scheduled['run_id']}"
        prior = self.state["runs"].get(run_id, [])
        if prior and prior[-1]["status"] == "success":
            print(f"[feature-validation] resume skip successful run={run_id}", flush=True)
            return prior[-1]
        if prior and not retry_failed:
            raise core.ValidationError(
                f"run {run_id} has a preserved failed attempt; use --retry-failed explicitly"
            )
        attempt = len(prior) + 1
        run_root = self.study_root / "runs" / stage["id"] / scheduled["run_id"] / f"attempt-{attempt:02d}"
        run_root.mkdir(parents=True, exist_ok=False)
        argv, environment = core.command_for_run(self.manifest, stage, scheduled)
        context = {
            "artifact_dir": run_root,
            "variant": scheduled["variant"],
            "pair": scheduled["pair"],
            "screen": scheduled["screen"],
            "run_id": scheduled["run_id"],
        }
        argv = [_render(value, context) for value in argv]
        environment = {key: _render(value, context) for key, value in environment.items()}
        identity_files, source_identities = self._identity_spec(stage, scheduled)
        result_path = run_root / "result.json"
        spec_path = run_root / "run-spec.json"
        comparison = stage.get("comparison", {})
        comparison_path = None
        if comparison.get("mode") == "file_sha256":
            if not comparison.get("path"):
                raise core.ManifestError("file_sha256 comparison requires path")
            comparison_path = _render(str(comparison["path"]), context)
        spec: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "attempt": attempt,
            "schedule": scheduled,
            "resource": stage["resource"],
            "argv": argv,
            "environment": environment,
            "cwd": str(run_root),
            "timeout_seconds": float(stage["timeout_seconds"]),
            "progress": stage["progress"],
            "result_path": str(result_path),
            "stdout_path": str(run_root / "stdout.log"),
            "stderr_path": str(run_root / "stderr.log"),
            "metric": stage.get("metric"),
            "comparison_path": comparison_path,
            "identity_files": identity_files,
            "source_identities": source_identities,
            "host_load_before": core.host_load_snapshot(),
        }
        if stage["resource"] == "gpu":
            inner = [sys.executable, str(self.tool_script), "_execute-run", str(spec_path)]
            wrapper = core.flock_argv(inner)
            spec["gpu_lock"] = {
                "path": core.GPU_LOCK,
                "whole_command_argv": wrapper,
                "whole_command": core.quote_argv(wrapper),
                "inner_lifecycle": "telemetry preflight, direct target, telemetry cleanup",
            }
            core.write_json_atomic(spec_path, spec)
            print(f"[feature-validation] launch {core.quote_argv(wrapper)}", flush=True)
            launch = subprocess.run(wrapper, check=False)
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                result = {
                    "run_id": run_id,
                    "attempt": attempt,
                    "status": "failed",
                    "error": f"flock/internal executor returned {launch.returncode} without result",
                    "gpu_lock": spec["gpu_lock"],
                }
                core.write_json_atomic(result_path, result)
        else:
            spec["gpu_lock"] = None
            core.write_json_atomic(spec_path, spec)
            result = execute_run_spec(spec)
        prior.append(result)
        self.state["runs"][run_id] = prior
        self._save_state()
        if result["status"] != "success":
            raise core.ValidationError(
                f"run {run_id} attempt {attempt} failed; preserved at {result_path}: {result.get('error')}"
            )
        return result

    def _pair_values(
        self,
        stage: dict[str, Any],
        schedule: list[dict[str, Any]],
        results: dict[str, dict[str, Any]],
        pair: int,
    ) -> dict[str, Any]:
        aggregation = stage.get("aggregation", {"mode": "weighted_mean"})["mode"]
        values: dict[str, float] = {}
        raw_screens: dict[str, list[dict[str, Any]]] = {}
        for variant in ("baseline", "candidate"):
            screen_values: list[tuple[float, float]] = []
            raw_screens[variant] = []
            for scheduled in [item for item in schedule if item["pair"] == pair and item["variant"] == variant]:
                result = results[scheduled["run_id"]]
                value = float(result["metric_value"])
                screen_values.append((value, float(scheduled["weight"])))
                raw_screens[variant].append(
                    {"screen": scheduled["screen"], "value": value, "weight": scheduled["weight"]}
                )
            values[variant] = core.aggregate_screen_values(screen_values, aggregation)
        orientation = next(item["orientation"] for item in schedule if item["pair"] == pair)
        return {
            "pair": pair,
            "orientation": orientation,
            "baseline": values["baseline"],
            "candidate": values["candidate"],
            "aggregation": aggregation,
            "raw_screens": raw_screens,
        }

    def _run_exactness(
        self, stage: dict[str, Any], *, retry_failed: bool
    ) -> dict[str, Any]:
        schedule = core.build_schedule(self.manifest, stage)
        results = [self._execute(stage, item, retry_failed=retry_failed) for item in schedule]
        mode = stage["comparison"]["mode"]
        hashes: dict[str, dict[str, str]] = {}
        for scheduled, result in zip(schedule, results, strict=True):
            source = result["stdout"] if mode == "stdout_sha256" else result["comparison_artifact"]
            hashes.setdefault(scheduled["screen"], {})[scheduled["variant"]] = source["sha256"]
        mismatches = {
            screen: values
            for screen, values in hashes.items()
            if values.get("baseline") != values.get("candidate")
        }
        report = {
            "purpose": stage["purpose"],
            "status": "passed" if not mismatches else "failed",
            "comparison_mode": mode,
            "hashes": hashes,
            "mismatches": mismatches,
            "raw_run_ids": [result["run_id"] for result in results],
        }
        if mismatches:
            assert self.state is not None
            self.state["stages"][stage["id"]] = report
            self._save_state()
            raise core.ValidationError(f"exactness gate {stage['id']} failed: {mismatches}")
        return report

    def _run_performance(
        self, stage: dict[str, Any], *, retry_failed: bool
    ) -> dict[str, Any]:
        schedule = core.build_schedule(self.manifest, stage)
        results: dict[str, dict[str, Any]] = {}
        pairs: list[dict[str, Any]] = []
        thresholds = self.manifest["repetition_policy"]["extension_rule"]["thresholds"]
        for pair in range(1, 4):
            for item in [entry for entry in schedule if entry["pair"] == pair]:
                results[item["run_id"]] = self._execute(stage, item, retry_failed=retry_failed)
            pairs.append(self._pair_values(stage, schedule, results, pair))
        report = core.paired_log_report(
            pairs,
            direction=stage["metric"]["direction"],
            thresholds=thresholds,
        )
        initial_report = report
        extension = {
            "rule": "extend_only_if_inconclusive",
            "initial_decision": report["decision"],
            "initial_statistics": initial_report,
            "extended": report["decision"] == "inconclusive",
        }
        if report["decision"] == "inconclusive":
            for pair in range(4, 6):
                for item in [entry for entry in schedule if entry["pair"] == pair]:
                    results[item["run_id"]] = self._execute(stage, item, retry_failed=retry_failed)
                pairs.append(self._pair_values(stage, schedule, results, pair))
            report = core.paired_log_report(
                pairs,
                direction=stage["metric"]["direction"],
                thresholds=thresholds,
            )
            extension["final_decision"] = report["decision"]
            extension["not_run"] = []
        else:
            extension["final_decision"] = report["decision"]
            extension["not_run"] = [
                {
                    "pair": pair,
                    "status": "not_run_by_preregistered_conclusive_rule",
                    "orientation": self.manifest["repetition_policy"]["order"][pair - 1],
                }
                for pair in (4, 5)
            ]
        return {
            "purpose": stage["purpose"],
            "status": "completed",
            "statistics": report,
            "adaptive_repetition": extension,
            "raw_run_ids": [result["run_id"] for result in results.values()],
        }

    def run(self, *, through: str, resume: bool, retry_failed: bool) -> dict[str, Any]:
        self.prepare(resume=resume)
        assert self.state is not None
        limits = {"early": 2, "production": 3, "acceptance": 4}
        if through not in limits:
            raise core.ValidationError(f"unknown validation phase {through!r}")
        start = time.monotonic()
        early_elapsed: float | None = None
        for stage in self.manifest["stages"]:
            if core.PURPOSE_ORDER[stage["purpose"]] > limits[through]:
                continue
            print(
                f"[feature-validation] stage={stage['id']} purpose={stage['purpose']} "
                f"progress={stage['progress']}",
                flush=True,
            )
            if stage["purpose"] == "exactness":
                report = self._run_exactness(stage, retry_failed=retry_failed)
            else:
                report = self._run_performance(stage, retry_failed=retry_failed)
            self.state["stages"][stage["id"]] = report
            self._save_state()
            if stage["purpose"] == "kernel_screen":
                early_elapsed = time.monotonic() - start
        final_gate = next(item for item in self.manifest["stages"] if item["purpose"] == "long_context_acceptance")
        final_passed = self.state["stages"].get(final_gate["id"], {}).get("status") == "completed"
        summary = {
            "study_id": self.manifest["study"]["id"],
            "artifact_root": str(self.study_root),
            "through": through,
            "early_elapsed_seconds": early_elapsed,
            "early_target_seconds": self.manifest["study"]["early_decision_target_seconds"],
            "final_long_context_acceptance_completed": final_passed,
            "status": "acceptance_complete" if final_passed else (
                "production_confirmation_only_not_final_acceptance"
                if through == "production"
                else "early_screen_only_not_production_or_final_acceptance"
            ),
            "evidence_boundary": (
                "Early/direct-kernel/Nsight evidence does not prove end-to-end performance, resource use, "
                "output exactness, or long-context behavior; each has its own gate."
            ),
            "stages": self.state["stages"],
        }
        core.write_json_atomic(self.study_root / "summary.json", summary)
        return summary
