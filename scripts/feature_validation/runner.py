"""Study execution, clean-process isolation, adaptive pairs, and safe resume."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from . import core
from .lifecycle import (
    cleanup_process_group as _cleanup_process_group,
    process_group_exists as _process_group_exists,
    terminate_process_group as _terminate_process_group,
)
from .telemetry import TelemetrySession


def _render(value: str, context: dict[str, Any]) -> str:
    try:
        return value.format_map({key: str(item) for key, item in context.items()})
    except KeyError as error:
        raise core.ValidationError(f"unknown command template field {error.args[0]!r}") from error


def _environment_command(environment: dict[str, str], argv: list[str]) -> list[str]:
    return ["env", "-i", *[f"{key}={environment[key]}" for key in sorted(environment)], *argv]


def _process_identity(pid: int) -> dict[str, int | None]:
    """Return a PID plus Linux start ticks so stale ownership cannot hit PID reuse."""
    start_ticks: int | None = None
    stat_path = pathlib.Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="utf-8")
        closing = raw.rfind(")")
        fields = raw[closing + 2 :].split()
        start_ticks = int(fields[19])
    except (OSError, ValueError, IndexError):
        pass
    return {"pid": pid, "start_time_ticks": start_ticks}


def _process_identity_is_live(identity: dict[str, Any] | None) -> bool:
    if not identity:
        return False
    pid = int(identity["pid"])
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    expected = identity.get("start_time_ticks")
    return expected is None or _process_identity(pid).get("start_time_ticks") == expected


def _lifecycle_path(spec: dict[str, Any]) -> pathlib.Path | None:
    value = spec.get("lifecycle_path")
    if value is None:
        return None
    path = pathlib.Path(str(value)).resolve()
    result_parent = pathlib.Path(spec["result_path"]).resolve().parent
    if path.parent != result_parent:
        raise core.ValidationError("lifecycle ownership record must stay in the attempt directory")
    return path


def _write_lifecycle(path: pathlib.Path | None, lifecycle: dict[str, Any]) -> None:
    if path is None:
        return
    lifecycle["updated_utc"] = core.utc_now()
    core.write_json_atomic(path, lifecycle)


def _read_lifecycle(spec: dict[str, Any]) -> dict[str, Any] | None:
    path = _lifecycle_path(spec)
    if path is None or not path.is_file():
        return None
    lifecycle = core.load_json(path)
    if lifecycle.get("run_id") != spec["run_id"] or lifecycle.get("attempt") != spec["attempt"]:
        raise core.ValidationError("lifecycle ownership record does not match this attempt")
    expected_identity = spec.get("study_identity")
    if expected_identity is not None and lifecycle.get("study_identity") != expected_identity:
        raise core.ProvenanceError("lifecycle ownership provenance does not match this study")
    return lifecycle


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
    lifecycle_path: pathlib.Path | None = None
    lifecycle: dict[str, Any] = {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "attempt": spec["attempt"],
        "study_identity": spec.get("study_identity"),
        "executor": _process_identity(os.getpid()),
        "target": None,
        "state": "executor_started",
        "created_utc": core.utc_now(),
    }
    try:
        lifecycle_path = _lifecycle_path(spec)
        _write_lifecycle(lifecycle_path, lifecycle)
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
        lifecycle["target"] = {
            **_process_identity(process.pid),
            "process_group_id": process_group_id,
        }
        lifecycle["state"] = "target_running"
        _write_lifecycle(lifecycle_path, lifecycle)
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
        lifecycle["state"] = "cleanup_complete"
        lifecycle["process_group_cleanup"] = process_group_cleanup
        lifecycle["telemetry_stopped"] = telemetry_report is not None
        try:
            _write_lifecycle(lifecycle_path, lifecycle)
        except (Exception, KeyboardInterrupt) as caught:
            error = _append_error(error, "lifecycle cleanup record", caught)
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
        "study_identity": spec.get("study_identity"),
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
) -> tuple[
    int | None,
    bool,
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[str],
    list[str],
]:
    """Keep flock alive while an identified executor cleans its target group."""
    interrupted = False
    forwarded = False
    errors: list[str] = []
    interrupt_signals: list[str] = []
    pending_signals: list[int] = []
    old_handlers: dict[int, Any] = {}
    can_install_handlers = threading.current_thread() is threading.main_thread()

    def capture_interrupt(signum: int, _frame: Any) -> None:
        pending_signals.append(signum)

    if can_install_handlers:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, capture_interrupt)

    try:
        process = subprocess.Popen(wrapper, start_new_session=True)
    except BaseException:
        if can_install_handlers:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
        raise
    pgid = process.pid

    cleanup: dict[str, Any] | None = None
    owned_target_cleanup: dict[str, Any] | None = None
    interruption_deadline: float | None = None
    try:
        while process.poll() is None:
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
            except KeyboardInterrupt:
                pending_signals.append(signal.SIGINT)

            if pending_signals:
                interrupted = True
                interrupt_signals.extend(signal.Signals(item).name for item in pending_signals)
                pending_signals.clear()
            if interrupted and not forwarded:
                # The lifecycle record is written before telemetry or target
                # creation.  Wait briefly for an executor that has just crossed
                # the flock boundary; if no record appears, the wrapper is only
                # waiting for the lock and is safe to cancel as a group.
                owner_deadline = time.monotonic() + 0.5
                lifecycle = _read_lifecycle(spec)
                while lifecycle is None and process.poll() is None and time.monotonic() < owner_deadline:
                    time.sleep(0.02)
                    lifecycle = _read_lifecycle(spec)
                if lifecycle is None:
                    cleanup = _cleanup_process_group(
                        process,
                        pgid,
                        term_timeout=float(spec.get("cleanup_term_timeout_seconds", 3.0)),
                        kill_timeout=float(spec.get("cleanup_kill_timeout_seconds", 3.0)),
                    )
                    errors.extend(str(item) for item in cleanup["errors"])
                else:
                    executor = lifecycle.get("executor")
                    if not _process_identity_is_live(executor):
                        errors.append("owned executor disappeared before interrupt forwarding")
                    else:
                        try:
                            os.kill(int(executor["pid"]), signal.SIGINT)
                        except ProcessLookupError:
                            errors.append("owned executor disappeared during interrupt forwarding")
                        except Exception as caught:
                            errors.append(
                                f"executor SIGINT forward: {type(caught).__name__}: {caught}"
                            )
                forwarded = True
                interruption_deadline = time.monotonic() + (
                    float(spec.get("cleanup_term_timeout_seconds", 3.0))
                    + float(spec.get("cleanup_kill_timeout_seconds", 3.0))
                    + 5.0
                )
            if (
                interrupted
                and process.poll() is None
                and interruption_deadline is not None
                and time.monotonic() >= interruption_deadline
            ):
                # The normal path leaves the lock-owning wrapper untouched until
                # the executor has reaped its target.  If that executor stalls,
                # use its provenance-bound target PGID as the final fallback,
                # then tear down the wrapper only after the target group is gone.
                lifecycle = _read_lifecycle(spec)
                target = lifecycle.get("target") if lifecycle else None
                if target and target.get("process_group_id") is not None:
                    owned_target_cleanup = _terminate_process_group(
                        int(target["process_group_id"]),
                        term_timeout=float(spec.get("cleanup_term_timeout_seconds", 3.0)),
                        kill_timeout=float(spec.get("cleanup_kill_timeout_seconds", 3.0)),
                    )
                    errors.extend(
                        f"owned target cleanup: {item}"
                        for item in owned_target_cleanup["errors"]
                    )
                cleanup = _cleanup_process_group(
                    process,
                    pgid,
                    term_timeout=float(spec.get("cleanup_term_timeout_seconds", 3.0)),
                    kill_timeout=float(spec.get("cleanup_kill_timeout_seconds", 3.0)),
                )
                errors.extend(str(item) for item in cleanup["errors"])
    finally:
        if pending_signals:
            interrupted = True
            interrupt_signals.extend(signal.Signals(item).name for item in pending_signals)
            pending_signals.clear()
        if can_install_handlers:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
    try:
        process.wait(timeout=0)
    except Exception as caught:
        errors.append(f"launcher reap: {type(caught).__name__}: {caught}")
    return (
        process.returncode,
        interrupted,
        cleanup,
        owned_target_cleanup,
        errors,
        interrupt_signals,
    )


def launch_locked_spec(
    spec: dict[str, Any], spec_path: pathlib.Path, tool_script: pathlib.Path
) -> dict[str, Any]:
    """Launch one internal lifecycle under the required whole-command GPU lock."""
    result_path = pathlib.Path(spec["result_path"])
    spec.setdefault(
        "lifecycle_path",
        str(result_path.with_name(f"{result_path.stem}-lifecycle-owner.json")),
    )
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
    (
        returncode,
        interrupted,
        launcher_cleanup,
        owned_target_cleanup,
        launcher_errors,
        interrupt_signals,
    ) = (
        _run_launcher_with_interrupt_forwarding(wrapper, spec)
    )
    if result_path.is_file():
        result = core.load_json(result_path)
        launcher_seconds = time.monotonic() - spec["launcher_requested_monotonic"]
        result.setdefault("timing", {})["locked_lifecycle_wall_seconds"] = launcher_seconds
        result["timing"]["locked_lifecycle_overhead_seconds"] = max(
            0.0,
            launcher_seconds - float(result["timing"].get("target_process_seconds", 0.0)),
        )
        result["launcher"] = {
            "returncode": returncode,
            "interrupted": interrupted,
            "interrupt_signals": interrupt_signals,
            "process_group_cleanup": launcher_cleanup,
            "owned_target_fallback_cleanup": owned_target_cleanup,
            "errors": launcher_errors,
        }
        if interrupted:
            result["status"] = "failed"
            result["error"] = _append_error(
                result.get("error"),
                "outer launcher",
                "parent interruption (" + ", ".join(interrupt_signals or ["SIGINT"]) + ")",
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
            "outer launcher parent interruption produced no internal result"
            if interrupted
            else f"flock/internal executor returned {returncode} without result"
        ),
        "gpu_lock": spec["gpu_lock"],
        "launcher": {
            "returncode": returncode,
            "interrupted": interrupted,
            "interrupt_signals": interrupt_signals,
            "process_group_cleanup": launcher_cleanup,
            "owned_target_fallback_cleanup": owned_target_cleanup,
            "errors": launcher_errors,
        },
    }
    core.write_json_atomic(result_path, result)
    return result


_ATTEMPT_DIRECTORY = re.compile(r"attempt-([0-9]+)")
_ATTEMPT_EVIDENCE_NAME = "attempt-evidence.json"
_RECOVERED_RESULT_NAME = "interrupted-result.json"


def _attempt_file_inventory(attempt_root: pathlib.Path) -> list[dict[str, Any]]:
    """Hash every preserved attempt artifact except the self-referential seal."""
    inventory: list[dict[str, Any]] = []
    for path in sorted(
        attempt_root.rglob("*"), key=lambda item: str(item.relative_to(attempt_root))
    ):
        relative = str(path.relative_to(attempt_root))
        if relative == _ATTEMPT_EVIDENCE_NAME or path.is_dir():
            continue
        if path.is_symlink():
            target = os.fsencode(os.readlink(path))
            inventory.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "sha256": hashlib.sha256(target).hexdigest(),
                }
            )
            continue
        if not path.is_file():
            raise core.ValidationError(
                f"attempt evidence contains unsupported non-file artifact: {path}"
            )
        before = core.stat_identity(path)
        digest = core.sha256_file(path)
        after = core.stat_identity(path)
        if before != after:
            raise core.ValidationError(f"attempt artifact changed while being sealed: {path}")
        inventory.append(
            {
                "path": relative,
                "kind": "file",
                "sha256": digest,
                **after,
            }
        )
    return inventory


def _attempt_evidence_reference(path: pathlib.Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": core.sha256_file(path),
        "kind": "failed_or_interrupted_attempt_seal",
    }


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
            state = core.load_json(self.state_path)
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

    def _study_identity(self) -> dict[str, str]:
        assert self.provenance is not None
        return {
            "manifest_sha256": self.manifest_sha256,
            "provenance_identity_fingerprint": self.provenance["identity_fingerprint"],
        }

    def _seal_failed_attempt(
        self,
        run_root: pathlib.Path,
        run_id: str,
        attempt: int,
    ) -> dict[str, Any]:
        evidence_path = run_root / _ATTEMPT_EVIDENCE_NAME
        if evidence_path.exists():
            evidence = core.load_json(evidence_path)
            if (
                evidence.get("run_id") != run_id
                or evidence.get("attempt") != attempt
                or evidence.get("study_identity") != self._study_identity()
                or pathlib.Path(str(evidence.get("artifact_directory", ""))).resolve()
                != run_root.resolve()
            ):
                raise core.ProvenanceError(
                    f"failed-attempt evidence identity mismatch: {evidence_path}"
                )
            observed_inventory = _attempt_file_inventory(run_root)
            if (
                evidence.get("inventory_sha256")
                != core.canonical_sha256(observed_inventory)
                or evidence.get("files") != observed_inventory
            ):
                raise core.ProvenanceError(
                    f"preserved failed-attempt evidence changed: {run_root}"
                )
            return _attempt_evidence_reference(evidence_path)

        inventory = _attempt_file_inventory(run_root)
        evidence = {
            "schema_version": 1,
            "kind": "failed_or_interrupted_attempt_evidence",
            "run_id": run_id,
            "attempt": attempt,
            "artifact_directory": str(run_root.resolve()),
            "study_identity": self._study_identity(),
            "sealed_utc": core.utc_now(),
            "files": inventory,
            "inventory_sha256": core.canonical_sha256(inventory),
        }
        core.write_json_atomic(evidence_path, evidence)
        return _attempt_evidence_reference(evidence_path)

    def _assert_incomplete_attempt_inactive(self, run_root: pathlib.Path) -> None:
        lifecycle_path = run_root / "lifecycle-owner.json"
        if not lifecycle_path.is_file():
            return
        try:
            lifecycle = core.load_json(lifecycle_path)
        except json.JSONDecodeError as error:
            raise core.ProvenanceError(
                f"cannot establish ownership of incomplete attempt {run_root}: {error}"
            ) from error
        identity = lifecycle.get("study_identity")
        if identity is not None and identity != self._study_identity():
            raise core.ProvenanceError(
                f"incomplete attempt lifecycle provenance mismatch: {lifecycle_path}"
            )
        if _process_identity_is_live(lifecycle.get("executor")):
            raise core.ValidationError(
                f"incomplete attempt still has a live owned executor; retry later: {run_root}"
            )
        target = lifecycle.get("target")
        if target and _process_group_exists(int(target["process_group_id"])):
            raise core.ValidationError(
                f"incomplete attempt still has a live owned target group; retry later: {run_root}"
            )

    def _recover_incomplete_attempt(
        self,
        run_root: pathlib.Path,
        run_id: str,
        attempt: int,
        scheduled: dict[str, Any],
    ) -> dict[str, Any]:
        self._assert_incomplete_attempt_inactive(run_root)
        preexisting_evidence: dict[str, Any] | None = None
        if (run_root / _ATTEMPT_EVIDENCE_NAME).is_file():
            preexisting_evidence = self._seal_failed_attempt(
                run_root, run_id, attempt
            )
        spec_path = run_root / "run-spec.json"
        preserved_spec: dict[str, Any] | None = None
        if spec_path.is_file():
            spec = core.load_json(spec_path)
            if spec.get("run_id") != run_id or spec.get("attempt") != attempt:
                raise core.ProvenanceError(
                    f"incomplete run spec does not match its deterministic directory: {spec_path}"
                )
            spec_identity = spec.get("study_identity")
            if spec_identity is not None and spec_identity != self._study_identity():
                raise core.ProvenanceError(
                    f"incomplete run spec provenance mismatch: {spec_path}"
                )
            preserved_spec = {
                "path": str(spec_path),
                "sha256": core.sha256_file(spec_path),
                "study_identity_present": spec_identity is not None,
            }

        child_result_path = run_root / "result.json"
        preserved_child_result: dict[str, Any] | None = None
        if child_result_path.is_file():
            child_result = core.load_json(child_result_path)
            if (
                child_result.get("run_id") != run_id
                or child_result.get("attempt") != attempt
            ):
                raise core.ProvenanceError(
                    "unindexed child result does not match its deterministic directory: "
                    f"{child_result_path}"
                )
            preserved_child_result = {
                "path": str(child_result_path),
                "sha256": core.sha256_file(child_result_path),
                "reported_status": child_result.get("status"),
                "metric_disposition": "preserved_not_counted",
            }

        recovered_path = run_root / _RECOVERED_RESULT_NAME
        if recovered_path.exists():
            recovered = core.load_json(recovered_path)
            if (
                recovered.get("run_id") != run_id
                or recovered.get("attempt") != attempt
                or recovered.get("study_identity") != self._study_identity()
                or recovered.get("status") != "failed"
            ):
                raise core.ProvenanceError(
                    f"recovered interruption record identity mismatch: {recovered_path}"
                )
        else:
            recovered = {
                "schema_version": 1,
                "run_id": run_id,
                "attempt": attempt,
                "schedule": scheduled,
                "status": "failed",
                "error": (
                    "recovered an unindexed attempt directory after parent interruption; "
                    "all artifacts are preserved and no metric is counted"
                ),
                "recovered_utc": core.utc_now(),
                "study_identity": self._study_identity(),
                "artifact_directory": str(run_root.resolve()),
                "preserved_run_spec": preserved_spec,
                "preserved_child_result": preserved_child_result,
                "metric_disposition": "not_counted_interrupted_attempt",
            }
            if preexisting_evidence is None:
                core.write_json_atomic(recovered_path, recovered)
            else:
                recovered["recovery_record"] = (
                    "state_only_to_preserve_preexisting_immutable_attempt_seal"
                )
        recovered["attempt_evidence"] = (
            preexisting_evidence
            or self._seal_failed_attempt(run_root, run_id, attempt)
        )
        return recovered

    def _reconcile_attempts(
        self,
        stage: dict[str, Any],
        scheduled: dict[str, Any],
        run_id: str,
    ) -> tuple[list[dict[str, Any]], int]:
        assert self.state is not None
        run_base = self.study_root / "runs" / stage["id"] / scheduled["run_id"]
        prior = self.state["runs"].get(run_id, [])
        state_changed = False
        by_attempt: dict[int, dict[str, Any]] = {}
        for record in prior:
            attempt = int(record.get("attempt", 0))
            if attempt <= 0 or attempt in by_attempt:
                raise core.ProvenanceError(f"invalid or duplicate attempt number for {run_id}")
            by_attempt[attempt] = record

        disk_attempts: dict[int, pathlib.Path] = {}
        if run_base.is_dir():
            for child in run_base.iterdir():
                if not child.is_dir() or not child.name.startswith("attempt-"):
                    continue
                match = _ATTEMPT_DIRECTORY.fullmatch(child.name)
                if match is None:
                    raise core.ProvenanceError(
                        f"unrecognized attempt directory cannot be reused: {child}"
                    )
                number = int(match.group(1))
                if child.name != f"attempt-{number:02d}":
                    raise core.ProvenanceError(
                        f"noncanonical attempt directory cannot be reused: {child}"
                    )
                if number <= 0 or number in disk_attempts:
                    raise core.ProvenanceError(f"invalid duplicate attempt directory: {child}")
                disk_attempts[number] = child

        for attempt, record in sorted(by_attempt.items()):
            run_root = run_base / f"attempt-{attempt:02d}"
            if disk_attempts.get(attempt) != run_root:
                raise core.ProvenanceError(
                    f"state references a missing deterministic attempt directory: {run_root}"
                )
            if record.get("status") != "success":
                reference = self._seal_failed_attempt(run_root, run_id, attempt)
                existing = record.get("attempt_evidence")
                if existing is not None and existing != reference:
                    raise core.ProvenanceError(
                        f"state failed-attempt evidence reference mismatch: {run_root}"
                    )
                if existing is None:
                    state_changed = True
                record["attempt_evidence"] = reference

        recovered_any = False
        for attempt, run_root in sorted(disk_attempts.items()):
            if attempt in by_attempt:
                continue
            recovered = self._recover_incomplete_attempt(
                run_root, run_id, attempt, scheduled
            )
            by_attempt[attempt] = recovered
            recovered_any = True

        ordered = [by_attempt[number] for number in sorted(by_attempt)]
        if recovered_any or state_changed or ordered != prior:
            self.state["runs"][run_id] = ordered
            self._save_state()
        highest = max({0, *by_attempt, *disk_attempts})
        return ordered, highest + 1

    def _execute(
        self,
        stage: dict[str, Any],
        scheduled: dict[str, Any],
        *,
        retry_failed: bool,
    ) -> dict[str, Any]:
        assert self.state is not None
        run_id = f"{stage['id']}--{scheduled['run_id']}"
        prior, attempt = self._reconcile_attempts(stage, scheduled, run_id)
        if prior and prior[-1]["status"] == "success":
            print(f"[feature-validation] resume skip successful run={run_id}", flush=True)
            return prior[-1]
        if prior and not retry_failed:
            raise core.ValidationError(
                f"run {run_id} has a preserved failed attempt; use --retry-failed explicitly"
            )
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
            "study_identity": self._study_identity(),
            "schedule": scheduled,
            "resource": stage["resource"],
            "argv": argv,
            "environment": environment,
            "cwd": str(run_root),
            "timeout_seconds": float(stage["timeout_seconds"]),
            "progress": stage["progress"],
            "result_path": str(result_path),
            "lifecycle_path": str(run_root / "lifecycle-owner.json"),
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
            result["attempt_evidence"] = self._seal_failed_attempt(
                run_root, run_id, attempt
            )
            self._save_state()
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
