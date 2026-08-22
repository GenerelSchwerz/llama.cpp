"""Low-overhead, process-owned telemetry for validation runs."""

from __future__ import annotations

import os
import pathlib
import pty
import re
import shutil
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from typing import Any

from .core import ValidationError, sha256_file, utc_now


PROC_STATUS_FIELDS = {
    "VmRSS",
    "VmHWM",
    "VmSize",
    "RssAnon",
    "RssFile",
    "RssShmem",
    "VmPin",
    "VmLck",
}
MEMINFO_FIELDS = {
    "MemTotal",
    "MemFree",
    "MemAvailable",
    "Buffers",
    "Cached",
    "SwapTotal",
    "SwapFree",
    "Mlocked",
    "AnonPages",
    "PageTables",
}


def _kib_value(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"([0-9]+)", value)
    return int(match.group(1)) if match else None


def _number(value: str | None) -> float | None:
    if value is None or value.strip().upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    match = re.search(r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", value)
    return float(match.group(0)) if match else None


def _text(node: ET.Element, path: str) -> str | None:
    child = node.find(path)
    return child.text.strip() if child is not None and child.text else None


def parse_nvidia_smi_xml(document: str, captured_utc: str | None = None) -> dict[str, Any]:
    """Parse the useful subset of one ``nvidia-smi -q -x`` document."""
    try:
        root = ET.fromstring(document.lstrip("\ufeff \t\r\n"))
    except ET.ParseError as error:
        raise ValidationError(f"invalid nvidia-smi XML: {error}") from error
    gpus: list[dict[str, Any]] = []
    for index, gpu in enumerate(root.findall("gpu")):
        processes: list[dict[str, Any]] = []
        for process in gpu.findall("./processes/process_info"):
            pid_text = _text(process, "pid")
            if not pid_text or not pid_text.isdigit():
                continue
            processes.append(
                {
                    "pid": int(pid_text),
                    "name": _text(process, "process_name"),
                    "type": _text(process, "type"),
                    "used_memory_mib": _number(_text(process, "used_memory")),
                }
            )
        gpus.append(
            {
                "index": index,
                "uuid": _text(gpu, "uuid"),
                "name": _text(gpu, "product_name"),
                "temperature_c": _number(_text(gpu, "./temperature/gpu_temp")),
                "power_draw_w": _number(_text(gpu, "./gpu_power_readings/instant_power_draw"))
                or _number(_text(gpu, "./gpu_power_readings/average_power_draw"))
                or _number(_text(gpu, "./gpu_power_readings/power_draw"))
                or _number(_text(gpu, "./power_readings/power_draw")),
                "power_limit_w": _number(_text(gpu, "./gpu_power_readings/current_power_limit"))
                or _number(_text(gpu, "./power_readings/power_limit")),
                "graphics_clock_mhz": _number(_text(gpu, "./clocks/graphics_clock")),
                "sm_clock_mhz": _number(_text(gpu, "./clocks/sm_clock")),
                "memory_clock_mhz": _number(_text(gpu, "./clocks/mem_clock")),
                "memory_used_mib": _number(_text(gpu, "./fb_memory_usage/used")),
                "memory_total_mib": _number(_text(gpu, "./fb_memory_usage/total")),
                "processes": processes,
            }
        )
    return {
        "captured_utc": captured_utc or utc_now(),
        "driver_version": _text(root, "driver_version"),
        "cuda_version": _text(root, "cuda_version"),
        "gpus": gpus,
    }


def _read_status(pid: int) -> dict[str, int] | None:
    path = pathlib.Path("/proc") / str(pid) / "status"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    values: dict[str, int] = {}
    for line in lines:
        key, separator, remainder = line.partition(":")
        if separator and key in PROC_STATUS_FIELDS:
            parsed = _kib_value(remainder)
            if parsed is not None:
                values[f"{key}_kib"] = parsed
    return values


def _read_meminfo() -> dict[str, int]:
    path = pathlib.Path("/proc/meminfo")
    if not path.is_file():
        return {}
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, remainder = line.partition(":")
        if separator and key in MEMINFO_FIELDS:
            parsed = _kib_value(remainder)
            if parsed is not None:
                values[f"{key}_kib"] = parsed
    return values


def _process_tree(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    proc = pathlib.Path("/proc")
    try:
        entries = list(proc.iterdir())
    except OSError:
        return {root_pid}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8", errors="replace").split()
            if len(fields) >= 4:
                parents[int(entry.name)] = int(fields[3])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _terminate_owned(process: subprocess.Popen[Any], timeout: float = 3.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.wait(timeout=timeout)


def _is_compute_process(process: dict[str, Any]) -> bool:
    process_type = process.get("type")
    if not process_type:
        return True
    return "C" in str(process_type).upper().split("+")


class TelemetrySession:
    """Own one persistent GPU sampler and one lightweight proc reader.

    A session is constructed before the target starts, performs the GPU-clean
    preflight, then attaches to the direct target PID.  ``stop`` is idempotent
    and always reaps the sampler it started.
    """

    def __init__(
        self,
        resource: str,
        *,
        nvidia_smi: str = "nvidia-smi",
        gpu_period_seconds: float = 1.0,
        proc_period_seconds: float = 0.25,
        first_sample_timeout: float = 12.0,
    ) -> None:
        if resource not in {"cpu", "gpu"}:
            raise ValidationError(f"unknown telemetry resource {resource!r}")
        self.resource = resource
        self.nvidia_smi = nvidia_smi
        self.gpu_period_seconds = gpu_period_seconds
        self.proc_period_seconds = proc_period_seconds
        self.first_sample_timeout = first_sample_timeout
        self.gpu_samples: list[dict[str, Any]] = []
        self.proc_samples: list[dict[str, Any]] = []
        self.parse_errors: list[str] = []
        self.sampler_stderr: list[str] = []
        self.target_pid: int | None = None
        self.sampler_pid: int | None = None
        self._sampler: subprocess.Popen[str] | None = None
        self._gpu_stream: Any | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._proc_reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._first_sample = threading.Event()
        self._lock = threading.Lock()
        self._started_utc: str | None = None
        self._stopped_utc: str | None = None
        self.clean_preflight: dict[str, Any] | None = None

    def start(self) -> None:
        if self._started_utc is not None:
            raise ValidationError("telemetry session was already started")
        self._started_utc = utc_now()
        if self.resource == "cpu":
            return
        interval = max(1, int(round(self.gpu_period_seconds)))
        master_fd, slave_fd = pty.openpty()
        try:
            self._sampler = subprocess.Popen(
                [self.nvidia_smi, "-q", "-x", "-l", str(interval)],
                stdout=slave_fd,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as error:
            os.close(master_fd)
            os.close(slave_fd)
            raise ValidationError(f"cannot start persistent nvidia-smi sampler: {error}") from error
        os.close(slave_fd)
        self._gpu_stream = os.fdopen(
            master_fd,
            "r",
            encoding="utf-8",
            errors="replace",
            newline=None,
        )
        self.sampler_pid = self._sampler.pid
        self._reader = threading.Thread(target=self._read_gpu_stream, name="fperf-nvidia-smi", daemon=True)
        self._stderr_reader = threading.Thread(
            target=self._read_sampler_stderr,
            name="fperf-nvidia-smi-stderr",
            daemon=True,
        )
        self._reader.start()
        self._stderr_reader.start()
        if not self._first_sample.wait(self.first_sample_timeout):
            self.stop()
            stderr = "; ".join(self.sampler_stderr)
            raise ValidationError(
                "persistent nvidia-smi sampler produced no complete XML sample"
                + (f": {stderr}" if stderr else "")
            )
        try:
            self.assert_gpu_clean()
        except Exception:
            self.stop()
            raise

    def _read_gpu_stream(self) -> None:
        assert self._sampler is not None and self._gpu_stream is not None
        buffer = ""
        closing = "</nvidia_smi_log>"
        try:
            while not self._stop.is_set():
                chunk = self._gpu_stream.readline()
                if chunk == "":
                    break
                buffer += chunk
                while closing in buffer:
                    document, buffer = buffer.split(closing, 1)
                    document += closing
                    try:
                        sample = parse_nvidia_smi_xml(document)
                    except ValidationError as error:
                        self.parse_errors.append(str(error))
                    else:
                        with self._lock:
                            self.gpu_samples.append(sample)
                        self._first_sample.set()
        except (OSError, ValueError) as error:
            if not self._stop.is_set():
                self.parse_errors.append(f"sampler reader: {type(error).__name__}: {error}")

    def _read_sampler_stderr(self) -> None:
        assert self._sampler is not None and self._sampler.stderr is not None
        try:
            for line in self._sampler.stderr:
                self.sampler_stderr.append(line.rstrip())
        except (OSError, ValueError) as error:
            self.parse_errors.append(f"sampler stderr reader: {type(error).__name__}: {error}")

    def assert_gpu_clean(self) -> None:
        if self.resource != "gpu":
            return
        with self._lock:
            sample = self.gpu_samples[-1] if self.gpu_samples else None
        if sample is None:
            raise ValidationError("no GPU sample exists for clean-process preflight")
        all_processes = [
            process
            for gpu in sample["gpus"]
            for process in gpu.get("processes", [])
            if process.get("pid") not in {None, self.sampler_pid}
        ]
        processes = [process for process in all_processes if _is_compute_process(process)]
        self.clean_preflight = {
            "captured_utc": sample["captured_utc"],
            "observed_compute_processes": processes,
            "ambient_noncompute_processes": [
                process for process in all_processes if not _is_compute_process(process)
            ],
            "clean": not processes,
        }
        if processes:
            raise ValidationError(f"GPU clean-process preflight failed: {processes}")

    def attach(self, pid: int) -> None:
        if self._started_utc is None:
            raise ValidationError("telemetry must start before attaching a target")
        if self.target_pid is not None:
            raise ValidationError("telemetry is already attached")
        self.target_pid = int(pid)
        self._sample_proc()
        self._proc_reader = threading.Thread(target=self._read_proc_loop, name="fperf-proc", daemon=True)
        self._proc_reader.start()

    def _sample_proc(self) -> None:
        if self.target_pid is None:
            return
        pids = _process_tree(self.target_pid)
        processes: dict[str, Any] = {}
        for pid in sorted(pids):
            status = _read_status(pid)
            if status is not None:
                processes[str(pid)] = status
        self.proc_samples.append(
            {
                "captured_utc": utc_now(),
                "root_pid": self.target_pid,
                "processes": processes,
                "host_meminfo": _read_meminfo(),
            }
        )

    def _read_proc_loop(self) -> None:
        while not self._stop.wait(self.proc_period_seconds):
            self._sample_proc()

    def stop(self) -> dict[str, Any]:
        if self._stopped_utc is None:
            self._stopped_utc = utc_now()
        self._stop.set()
        if self._sampler is not None:
            _terminate_owned(self._sampler)
        if self._reader is not None:
            self._reader.join(timeout=3.0)
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=3.0)
        if self._proc_reader is not None:
            self._proc_reader.join(timeout=3.0)
        if self._gpu_stream is not None:
            self._gpu_stream.close()
        if self._sampler is not None:
            if self._sampler.stderr is not None:
                self._sampler.stderr.close()
        if self.target_pid is not None:
            self._sample_proc()
        return self.report()

    def report(self) -> dict[str, Any]:
        allowed = _process_tree(self.target_pid) if self.target_pid else set()
        for sample in self.proc_samples:
            allowed.update(int(pid) for pid in sample.get("processes", {}))
        gpu_vram: list[float] = []
        foreign: list[dict[str, Any]] = []
        ambient: list[dict[str, Any]] = []
        for sample in self.gpu_samples:
            total = 0.0
            observed_target = False
            for gpu in sample.get("gpus", []):
                for process in gpu.get("processes", []):
                    pid = process.get("pid")
                    used = process.get("used_memory_mib")
                    if pid in allowed and used is not None:
                        total += float(used)
                        observed_target = True
                    elif pid not in {None, self.sampler_pid}:
                        observation = {
                            "captured_utc": sample["captured_utc"],
                            "gpu_uuid": gpu.get("uuid"),
                            **process,
                        }
                        if _is_compute_process(process):
                            foreign.append(observation)
                        else:
                            ambient.append(observation)
            if observed_target:
                gpu_vram.append(total)
        rss_values: list[int] = []
        hwm_values: list[int] = []
        vm_lck_values: list[int] = []
        vm_pin_values: list[int] = []
        for sample in self.proc_samples:
            statuses = sample.get("processes", {}).values()
            rss_values.append(sum(int(item.get("VmRSS_kib", 0)) for item in statuses))
            hwm_values.append(sum(int(item.get("VmHWM_kib", 0)) for item in statuses))
            vm_lck_values.append(sum(int(item.get("VmLck_kib", 0)) for item in statuses))
            vm_pin_values.append(sum(int(item.get("VmPin_kib", 0)) for item in statuses))
        sampler_reaped = self._sampler is None or self._sampler.poll() is not None
        sampler_path_text = shutil.which(self.nvidia_smi) or self.nvidia_smi
        sampler_path = pathlib.Path(sampler_path_text).expanduser().resolve()
        sampler_identity = {
            "path": str(sampler_path),
            "sha256": sha256_file(sampler_path) if sampler_path.is_file() else None,
            "argv": [self.nvidia_smi, "-q", "-x", "-l", str(max(1, int(round(self.gpu_period_seconds))))],
        }
        return {
            "started_utc": self._started_utc,
            "stopped_utc": self._stopped_utc,
            "resource": self.resource,
            "sampler": {
                "kind": "one persistent nvidia-smi -q -x -l process" if self.resource == "gpu" else None,
                "pid": self.sampler_pid,
                "period_seconds": self.gpu_period_seconds if self.resource == "gpu" else None,
                "reaped": sampler_reaped,
                "invocation_count": 1 if self.resource == "gpu" else 0,
                "identity": sampler_identity if self.resource == "gpu" else None,
                "stderr": self.sampler_stderr,
            },
            "clean_process_evidence": {
                "preflight": self.clean_preflight,
                "foreign_compute_process_observations": foreign,
                "ambient_noncompute_process_observations": ambient,
                "clean_throughout": bool(self.clean_preflight and self.clean_preflight["clean"] and not foreign)
                if self.resource == "gpu"
                else None,
            },
            "process_vram": {
                "unit": "MiB",
                "peak": max(gpu_vram) if gpu_vram else None,
                "source": "nvidia-smi process used_memory",
            },
            "ordinary_host_memory": {
                "unit": "KiB",
                "peak_process_tree_vmrss": max(rss_values) if rss_values else None,
                "peak_process_tree_vmhwm": max(hwm_values) if hwm_values else None,
                "source": "/proc/PID/status plus /proc/meminfo",
            },
            "pinned_host_memory": {
                "directly_measured": False,
                "bytes": None,
                "limitation": (
                    "No direct per-process page-locked allocation counter was available. "
                    "VmLck/VmPin observations are retained only as ordinary proc fields; pinned "
                    "bytes are never inferred from VmLck, VmPin, or process VRAM."
                ),
                "observed_vm_lck_kib_peak_not_pinned_inference": max(vm_lck_values)
                if vm_lck_values
                else None,
                "observed_vm_pin_kib_peak_not_pinned_inference": max(vm_pin_values)
                if vm_pin_values
                else None,
            },
            "gpu_samples": self.gpu_samples,
            "proc_samples": self.proc_samples,
            "parse_errors": self.parse_errors,
        }

    def __enter__(self) -> "TelemetrySession":
        self.start()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.stop()
