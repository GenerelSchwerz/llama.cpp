"""Process-group ownership and fail-closed cleanup for validation runs."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_owned_process(
    process: subprocess.Popen[Any], timeout: float = 3.0
) -> None:
    """Stop and reap a private helper process using its isolated process group."""
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


def wait_for_group_disappearance(
    pgid: int,
    timeout: float,
    process: subprocess.Popen[Any] | None = None,
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        # poll() also reaps an exited direct child so its zombie cannot keep the
        # process group observable while descendants are being checked.
        if process is not None:
            process.poll()
        if not process_group_exists(pgid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def terminate_process_group(
    pgid: int,
    *,
    process: subprocess.Popen[Any] | None = None,
    term_timeout: float = 3.0,
    kill_timeout: float = 3.0,
) -> dict[str, Any]:
    """Terminate an entire tracked group, including descendants after leader exit."""
    report: dict[str, Any] = {
        "pgid": pgid,
        "term_sent": False,
        "kill_sent": False,
        "group_gone": False,
        "errors": [],
    }
    if pgid == os.getpgrp():
        report["errors"].append("refusing to signal the executor's own process group")
        return report
    try:
        group_exists = process_group_exists(pgid)
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
            report["group_gone"] = wait_for_group_disappearance(
                pgid, term_timeout, process
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
            report["group_gone"] = wait_for_group_disappearance(
                pgid, kill_timeout, process
            )
        except (Exception, KeyboardInterrupt) as caught:
            report["errors"].append(
                f"process-group KILL wait: {type(caught).__name__}: {caught}"
            )
    if not report["group_gone"]:
        report["errors"].append(f"process group {pgid} remained after SIGKILL")
    return report


def cleanup_process_group(
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
    report = terminate_process_group(
        pgid,
        process=process,
        term_timeout=term_timeout,
        kill_timeout=kill_timeout,
    )
    report["leader_reaped"] = False

    try:
        if process.poll() is None:
            process.wait(timeout=max(0.1, kill_timeout))
        else:
            process.wait(timeout=0)
        report["leader_reaped"] = True
    except (Exception, KeyboardInterrupt) as caught:
        report["errors"].append(f"leader reap: {type(caught).__name__}: {caught}")
    return report
