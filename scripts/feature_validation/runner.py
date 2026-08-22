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


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_group_disappearance(
    process: subprocess.Popen[Any], pgid: int, timeout: float
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        # poll() also reaps an exited direct child so its zombie cannot keep the
        # process group observable while descendants are being checked.
        process.poll()
        if not _process_group_exists(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def _cleanup_process_group(
    process: subprocess.Popen[Any],
    pgid: int,
    *,
    term_timeout: float = 3.0,
    kill_timeout: float = 3.0,
) -> dict[str, Any]:
    """Terminate an entire tracked group and always reap its direct leader.

    A leader may exit while descendants keep its process group alive.  Group
    existence, rather than leader status, therefore controls TERM/KILL
    escalation.  Cleanup failures are returned as evidence instead of raised so
    callers can still stop telemetry and serialize the attempt result.
    """
    report: dict[str, Any] = {
        "pgid": pgid,
        "term_sent": False,
        "kill_sent": False,
        "group_gone": False,
        "leader_reaped": False,
        "errors": [],
    }
    if pgid == os.getpgrp():
        report["errors"].append("refusing to signal the executor's own process group")
    else:
        try:
            group_exists = _process_group_exists(pgid)
        except (Exception, KeyboardInterrupt) as caught:
            group_exists = True
            report["errors"].append(
                f"process-group existence check: {type(caught).__name__}: {caught}"
            )
        if group_exists:
            try:
                os.killpg(pgid, signal.SIGTERM)
                report["term_sent"] = True
            except ProcessLookupError:
                pass
            except (Exception, KeyboardInterrupt) as caught:
                report["errors"].append(
                    f"process-group SIGTERM: {type(caught).__name__}: {caught}"
                )
            try:
                report["group_gone"] = _wait_for_group_disappearance(
                    process, pgid, term_timeout
                )
            except (Exception, KeyboardInterrupt) as caught:
                report["errors"].append(
                    f"process-group TERM wait: {type(caught).__name__}: {caught}"
                )
        else:
            report["group_gone"] = True
        if not report["group_gone"]:
            try:
                os.killpg(pgid, signal.SIGKILL)
                report["kill_sent"] = True
            except ProcessLookupError:
                pass
            except (Exception, KeyboardInterrupt) as caught:
                report["errors"].append(
                    f"process-group SIGKILL: {type(caught).__name__}: {caught}"
                )
            try:
                report["group_gone"] = _wait_for_group_disappearance(
                    process, pgid, kill_timeout
                )
            except (Exception, KeyboardInterrupt) as caught:
                report["errors"].append(
                    f"process-group KILL wait: {type(caught).__name__}: {caught}"
                )
        if not report["group_gone"]:
            report["errors"].append(f"process group {pgid} remained after SIGKILL")

    try:
        if process.poll() is None:
            process.wait(timeout=max(0.1, kill_timeout))
        else:
            process.wait(timeout=0)
        report["leader_reaped"] = True
    except (Exception, KeyboardInterrupt) as caught:
        report["errors"].append(f"leader reap: {type(caught).__name__}: {caught}")
    return report


def _append_error(error: str | None, label: str, caught: BaseException | str) -> str:
    detail = caught if isinstance(caught, str) else f"{type(caught).__name__}: {caught}"
    addition = f"{label}: {detail}"
    return f"{error}; {addition}" if error else addition


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
        for key in ("size", "mtime_ns", "ctime_ns", "device", "inode"):
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
        sampler_identity=spec.get("nvidia_smi_identity"),
    )
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    threads: list[threading.Thread] = []
    error: str | None = None
    timed_out = False
    returncode: int | None = None
    telemetry_report: dict[str, Any] | None = None
    identity_seconds: float | None = None
    telemetry_startup_seconds: float | None = None
    target_started: float | None = None
    target_finished: float | None = None
    cleanup_seconds: float | None = None
    process_group_id: int | None = None
    process_group_cleanup: dict[str, Any] | None = None
    try:
        phase_start = time.monotonic()
        _verify_spec_identity(spec)
        identity_seconds = time.monotonic() - phase_start
        phase_start = time.monotonic()
        telemetry.start()
        telemetry_startup_seconds = time.monotonic() - phase_start
        argv = [str(value) for value in spec["argv"]]
        environment = {str(key): str(value) for key, value in spec["environment"].items()}
        target_started = time.monotonic()
        process = subprocess.Popen(
            argv,
            cwd=spec.get("cwd"),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        # start_new_session makes the direct target the leader of a new process
        # group.  Keep this identity even if that leader exits before cleanup.
        process_group_id = process.pid
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
                break
            if time.monotonic() >= next_heartbeat:
                print(
                    f"[feature-validation] progress run={spec['run_id']} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                next_heartbeat += 5.0
            time.sleep(0.05)
        returncode = process.poll()
        target_finished = time.monotonic()
    except (Exception, KeyboardInterrupt) as caught:  # preserve failures and reap on user interrupt
        if target_started is not None and target_finished is None:
            target_finished = time.monotonic()
        error = f"{type(caught).__name__}: {caught}"
    finally:
        cleanup_started = time.monotonic()
        if process is not None and process_group_id is not None:
            try:
                process_group_cleanup = _cleanup_process_group(
                    process,
                    process_group_id,
                    term_timeout=float(spec.get("cleanup_term_timeout_seconds", 3.0)),
                    kill_timeout=float(spec.get("cleanup_kill_timeout_seconds", 3.0)),
                )
                returncode = process.returncode
                for cleanup_error in process_group_cleanup["errors"]:
                    error = _append_error(error, "process cleanup", cleanup_error)
            except (Exception, KeyboardInterrupt) as caught:
                # The cleanup primitive is intended to report rather than raise,
                # but preserve the result even if an unforeseen cleanup failure
                # escapes it.
                error = _append_error(error, "process cleanup", caught)
                try:
                    returncode = process.poll()
                except (Exception, KeyboardInterrupt) as poll_error:
                    error = _append_error(error, "process status", poll_error)
        for thread in threads:
            try:
                thread.join(timeout=3.0)
                if thread.is_alive():
                    error = _append_error(error, "output cleanup", "drain thread did not stop")
            except (Exception, KeyboardInterrupt) as caught:
                error = _append_error(error, "output cleanup", caught)
        if process is not None:
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except (Exception, KeyboardInterrupt) as caught:
                    error = _append_error(error, "stdout cleanup", caught)
            if process.stderr is not None:
                try:
                    process.stderr.close()
                except (Exception, KeyboardInterrupt) as caught:
                    error = _append_error(error, "stderr cleanup", caught)
        try:
            telemetry_report = telemetry.stop()
        except (Exception, KeyboardInterrupt) as caught:
            error = _append_error(error, "telemetry cleanup", caught)
        cleanup_seconds = time.monotonic() - cleanup_started
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
    total_seconds = time.monotonic() - start_monotonic
    target_seconds = (
        target_finished - target_started
        if target_started is not None and target_finished is not None
        else 0.0
    )
    result: dict[str, Any] = {
        "run_id": spec["run_id"],
        "status": status,
        "attempt": spec["attempt"],
        "schedule": spec.get("schedule"),
        "started_utc": started,
        "finished_utc": core.utc_now(),
        "elapsed_seconds": total_seconds,
        "timing": {
            "identity_verification_seconds": identity_seconds,
            "telemetry_startup_seconds": telemetry_startup_seconds,
            "target_process_seconds": target_seconds,
            "cleanup_seconds": cleanup_seconds,
            "toolkit_overhead_seconds": max(0.0, total_seconds - target_seconds),
            "locked_wrapper_to_executor_seconds": (
                max(0.0, start_monotonic - float(spec["launcher_requested_monotonic"]))
                if spec.get("launcher_requested_monotonic") is not None
                else None
            ),
        },
        "pid": process.pid if process is not None else None,
        "process_group_id": process_group_id,
        "process_group_cleanup": process_group_cleanup,
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


def _run_launcher_with_interrupt_forwarding(
    wrapper: list[str], spec: dict[str, Any]
) -> tuple[int | None, bool, dict[str, Any] | None, list[str]]:
    """Run an isolated wrapper and forward outer SIGINT to its whole group."""
    process = subprocess.Popen(wrapper, start_new_session=True)
    pgid = process.pid
    interrupted = False
    errors: list[str] = []
    while process.poll() is None:
        try:
            process.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            continue
        except KeyboardInterrupt:
            interrupted = True
            try:
                os.killpg(pgid, signal.SIGINT)
            except ProcessLookupError:
                pass
            except Exception as caught:
                errors.append(f"SIGINT forward: {type(caught).__name__}: {caught}")

    cleanup: dict[str, Any] | None = None
    if interrupted:
        # flock may exit before the internal executor finishes recording its
        # interrupted target.  Give that executor its own TERM/KILL cleanup
        # window, then force down any remaining launcher descendants.
        grace = (
            float(spec.get("cleanup_term_timeout_seconds", 3.0))
            + float(spec.get("cleanup_kill_timeout_seconds", 3.0))
            + 2.0
        )
        try:
            group_gone = _wait_for_group_disappearance(process, pgid, grace)
        except KeyboardInterrupt:
            try:
                os.killpg(pgid, signal.SIGINT)
            except ProcessLookupError:
                pass
            except Exception as caught:
                errors.append(f"repeated SIGINT forward: {type(caught).__name__}: {caught}")
            group_gone = False
        except Exception as caught:
            errors.append(f"launcher group wait: {type(caught).__name__}: {caught}")
            group_gone = False
        if not group_gone:
            cleanup = _cleanup_process_group(
                process,
                pgid,
                term_timeout=float(spec.get("cleanup_term_timeout_seconds", 3.0)),
                kill_timeout=float(spec.get("cleanup_kill_timeout_seconds", 3.0)),
            )
            errors.extend(str(item) for item in cleanup["errors"])
    try:
        process.wait(timeout=0)
    except Exception as caught:
        errors.append(f"launcher reap: {type(caught).__name__}: {caught}")
    return process.returncode, interrupted, cleanup, errors


def launch_locked_spec(
    spec: dict[str, Any], spec_path: pathlib.Path, tool_script: pathlib.Path
) -> dict[str, Any]:
    """Launch one internal lifecycle under the required whole-command GPU lock."""
    result_path = pathlib.Path(spec["result_path"])
    if spec_path.exists() or result_path.exists():
        raise core.ValidationError(
            f"refusing to overwrite an existing attempt: {spec_path} / {result_path}"
        )
    inner = [sys.executable, str(tool_script), "_execute-run", str(spec_path)]
    wrapper = core.flock_argv(inner)
    spec["gpu_lock"] = {
        "path": core.GPU_LOCK,
        "whole_command_argv": wrapper,
        "whole_command": core.quote_argv(wrapper),
        "inner_lifecycle": "telemetry preflight, direct target, telemetry cleanup",
    }
    spec["launcher_requested_monotonic"] = time.monotonic()
    core.write_json_atomic(spec_path, spec)
    print(f"[feature-validation] launch {core.quote_argv(wrapper)}", flush=True)
    returncode, interrupted, launcher_cleanup, launcher_errors = (
        _run_launcher_with_interrupt_forwarding(wrapper, spec)
    )
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        launcher_seconds = time.monotonic() - spec["launcher_requested_monotonic"]
        result.setdefault("timing", {})["locked_lifecycle_wall_seconds"] = launcher_seconds
        result["timing"]["locked_lifecycle_overhead_seconds"] = max(
            0.0,
            launcher_seconds - float(result["timing"].get("target_process_seconds", 0.0)),
        )
        result["launcher"] = {
            "returncode": returncode,
            "interrupted": interrupted,
            "process_group_cleanup": launcher_cleanup,
            "errors": launcher_errors,
        }
        if interrupted:
            result["status"] = "failed"
            result["error"] = _append_error(
                result.get("error"), "outer launcher", "KeyboardInterrupt"
            )
        for launcher_error in launcher_errors:
            result["status"] = "failed"
            result["error"] = _append_error(
                result.get("error"), "launcher cleanup", launcher_error
            )
        core.write_json_atomic(result_path, result)
        return result
    result = {
        "run_id": spec["run_id"],
        "attempt": spec["attempt"],
        "status": "failed",
        "error": (
            "outer launcher KeyboardInterrupt produced no internal result"
            if interrupted
            else f"flock/internal executor returned {returncode} without result"
        ),
        "gpu_lock": spec["gpu_lock"],
        "launcher": {
            "returncode": returncode,
            "interrupted": interrupted,
            "process_group_cleanup": launcher_cleanup,
            "errors": launcher_errors,
        },
    }
    core.write_json_atomic(result_path, result)
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
        self.preparation_elapsed_seconds = 0.0
        self.resume_mode = False

    def prepare(self, *, resume: bool) -> None:
        if self.study_root.exists():
            core.validate_artifact_directory_isolation(self.manifest, self.study_root)
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
            core.validate_artifact_directory_isolation(self.manifest, self.study_root)
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
        core.validate_artifact_directory_isolation(self.manifest, run_root)
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
        assert self.provenance is not None
        role = str(stage["command"].get("executable_role", "default"))
        identity_files, source_identities = core.provenance_identity_spec(
            self.provenance, [(scheduled["variant"], role)]
        )
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
            "nvidia_smi_identity": self.provenance.get("runtime_tools", {}).get("nvidia-smi"),
            "nvidia_smi": (
                self.provenance.get("runtime_tools", {})
                .get("nvidia-smi", {})
                .get("path", "nvidia-smi")
            ),
            "host_load_before": core.host_load_snapshot(),
        }
        if stage["resource"] == "gpu":
            core.validate_artifact_directory_isolation(self.manifest, run_root)
            result = launch_locked_spec(spec, spec_path, self.tool_script)
        else:
            spec["gpu_lock"] = None
            core.write_json_atomic(spec_path, spec)
            core.validate_artifact_directory_isolation(self.manifest, run_root)
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
            "execution_status": "completed",
            "status": "passed" if not mismatches else "failed",
            "comparison_mode": mode,
            "hashes": hashes,
            "mismatches": mismatches,
            "raw_run_ids": [result["run_id"] for result in results],
        }
        return report

    def _run_performance(
        self, stage: dict[str, Any], *, retry_failed: bool
    ) -> dict[str, Any]:
        schedule = core.build_schedule(self.manifest, stage)
        results: dict[str, dict[str, Any]] = {}
        screening_policy = stage.get("screening_policy")
        if screening_policy is not None:
            for item in schedule:
                results[item["run_id"]] = self._execute(
                    stage, item, retry_failed=retry_failed
                )
            pair = self._pair_values(stage, schedule, results, 1)
            observation = core.single_pair_screen_report(
                pair,
                direction=stage["metric"]["direction"],
                regression_threshold_percent=screening_policy[
                    "regression_threshold_percent"
                ],
            )
            passed = observation["signal"] == "clear_to_continue"
            return {
                "purpose": stage["purpose"],
                "execution_status": "completed",
                "status": "passed" if passed else "failed",
                "screening_policy": screening_policy,
                "screening_observation": observation,
                "adaptive_repetition": {
                    "applicable": False,
                    "reason": "single_pair_fail_fast_screen",
                    "statistical_pairs": "not_applicable_by_preregistered_screening_policy",
                },
                "raw_run_ids": [result["run_id"] for result in results.values()],
            }
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
            "execution_status": "completed",
            "status": core.performance_outcome(report["decision"], stage["decision_policy"]),
            "decision_policy": stage["decision_policy"],
            "statistics": report,
            "adaptive_repetition": extension,
            "raw_run_ids": [result["run_id"] for result in results.values()],
        }

    def _summary(self, through: str, early_elapsed: float | None) -> dict[str, Any]:
        assert self.state is not None
        final_gate = next(
            item
            for item in self.manifest["stages"]
            if item["purpose"] == "long_context_acceptance"
        )
        final_report = self.state["stages"].get(final_gate["id"])
        final_executed = bool(
            final_report and final_report.get("execution_status") == "completed"
        )
        final_passed = bool(final_report and final_report.get("status") == "passed")
        blocking = next(
            (
                (stage, self.state["stages"][stage["id"]])
                for stage in self.manifest["stages"]
                if self.state["stages"].get(stage["id"], {}).get("status")
                in {"failed", "unresolved"}
            ),
            None,
        )
        if blocking:
            blocking_stage, blocking_report = blocking
            prefix = (
                "acceptance"
                if blocking_stage["purpose"] == "long_context_acceptance"
                else blocking_stage["purpose"]
            )
            status = f"{prefix}_{blocking_report['status']}"
        elif through == "acceptance" and final_report:
            status = {
                "passed": "acceptance_complete",
                "failed": "acceptance_failed",
                "unresolved": "acceptance_unresolved",
            }.get(final_report.get("status"), "acceptance_not_completed")
        elif through == "production":
            status = "production_confirmation_only_not_final_acceptance"
        else:
            status = "early_screen_only_not_production_or_final_acceptance"
        early_stage_ids = {
            stage["id"]
            for stage in self.manifest["stages"]
            if core.PURPOSE_ORDER[stage["purpose"]] <= core.PURPOSE_ORDER["kernel_screen"]
        }
        early_attempts = [
            attempts[-1]
            for run_id, attempts in self.state["runs"].items()
            if run_id.split("--", 1)[0] in early_stage_ids and attempts
        ]
        target_seconds = sum(
            float(item.get("timing", {}).get("target_process_seconds", 0.0))
            for item in early_attempts
        )
        per_run_overhead = sum(
            float(item.get("timing", {}).get("toolkit_overhead_seconds", 0.0))
            for item in early_attempts
        )
        locked_lifecycle = sum(
            float(
                item.get("timing", {}).get(
                    "locked_lifecycle_wall_seconds", item.get("elapsed_seconds", 0.0)
                )
            )
            for item in early_attempts
        )
        return {
            "study_id": self.manifest["study"]["id"],
            "artifact_root": str(self.study_root),
            "through": through,
            "early_elapsed_seconds": early_elapsed,
            "early_target_seconds": self.manifest["study"]["early_decision_target_seconds"],
            "early_timing": {
                "mode": "resume" if self.resume_mode else "fresh",
                "run_count": len(early_attempts),
                "preparation_seconds": self.preparation_elapsed_seconds,
                "target_process_seconds": target_seconds,
                "recorded_per_run_toolkit_overhead_seconds": per_run_overhead,
                "locked_lifecycle_wall_seconds": locked_lifecycle,
                "locked_wrapper_to_executor_seconds": sum(
                    float(
                        item.get("timing", {}).get("locked_wrapper_to_executor_seconds")
                        or 0.0
                    )
                    for item in early_attempts
                ),
                "identity_verification_seconds": sum(
                    float(item.get("timing", {}).get("identity_verification_seconds") or 0.0)
                    for item in early_attempts
                ),
                "telemetry_startup_seconds": sum(
                    float(item.get("timing", {}).get("telemetry_startup_seconds") or 0.0)
                    for item in early_attempts
                ),
                "wall_minus_target_seconds": (
                    max(0.0, early_elapsed - target_seconds)
                    if early_elapsed is not None and not self.resume_mode
                    else None
                ),
            },
            "final_long_context_acceptance_executed": final_executed,
            "final_long_context_acceptance_passed": final_passed,
            "status": status,
            "evidence_boundary": (
                "Early/direct-kernel/Nsight evidence does not prove end-to-end performance, resource use, "
                "output exactness, or long-context behavior; each has its own gate."
            ),
            "stages": self.state["stages"],
        }

    def run(self, *, through: str, resume: bool, retry_failed: bool) -> dict[str, Any]:
        start = time.monotonic()
        self.resume_mode = resume
        self.prepare(resume=resume)
        self.preparation_elapsed_seconds = time.monotonic() - start
        assert self.state is not None
        limits = {"early": 2, "production": 3, "acceptance": 4}
        if through not in limits:
            raise core.ValidationError(f"unknown validation phase {through!r}")
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
            if report["status"] != "passed":
                summary = self._summary(through, early_elapsed)
                core.write_json_atomic(self.study_root / "summary.json", summary)
                raise core.ValidationError(
                    f"stage {stage['id']} executed but is not acceptable: {report['status']}"
                )
        summary = self._summary(through, early_elapsed)
        core.write_json_atomic(self.study_root / "summary.json", summary)
        return summary
