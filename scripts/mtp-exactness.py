#!/usr/bin/env python3

"""Run reproducible llama-server token-exactness matrices.

The input is a JSON manifest.  Each case gets a fresh server process, a sterile
environment, a captured request/response/log/progress directory, and an exact
comparison with the named reference case.  The runner deliberately uses only
the Python standard library so it can also be used from clean benchmark hosts.

Example manifest:

{
  "request_file": "/tmp/request.json",
  "output_dir": "/tmp/mtp-exact",
  "common_args": ["--model", "/models/target.gguf", "--ctx-size", "4096"],
  "comparison_contract": {
    "require_explicit_references": true,
    "required_identity_keys": ["model", "draft_depth", "sampler"],
    "common_identity": {"model": "target.gguf", "sampler": "greedy-seed-1234"}
  },
  "cases": [
    {"name": "golden-mtp6", "role": "golden",
     "server": "/clean-bee/bin/llama-server",
     "identity": {"draft_depth": 6},
     "args": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "6"]},
    {"name": "candidate-mtp6", "compare_to": "golden-mtp6",
     "server": "build/bin/llama-server",
     "identity": {"draft_depth": 6},
     "args": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "6"]}
  ]
}
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def json_dump(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def lookup_dimension(identity: dict[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = identity
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            return False, None
        current = current[component]
    return True, current


def merged_case_args(manifest: dict[str, Any], case: dict[str, Any]) -> list[str]:
    return [
        *[str(value) for value in manifest.get("common_args", [])],
        *[str(value) for value in case.get("common_args", [])],
        *[str(value) for value in case.get("args", [])],
    ]


def last_option_value(args: list[str], aliases: tuple[str, ...]) -> str | None:
    value: str | None = None
    index = 0
    while index < len(args):
        argument = args[index]
        if argument in aliases:
            if index + 1 >= len(args):
                raise ValueError(f"option {argument!r} is missing its value")
            value = args[index + 1]
            index += 2
            continue
        for alias in aliases:
            prefix = alias + "="
            if argument.startswith(prefix):
                value = argument[len(prefix):]
                break
        index += 1
    return value


def resolved_int_setting(
    manifest: dict[str, Any],
    case: dict[str, Any],
    aliases: tuple[str, ...],
    environment_name: str,
) -> int | None:
    environment = {
        **manifest.get("environment", {}),
        **case.get("environment", {}),
    }
    value = environment.get(environment_name)
    cli_value = last_option_value(merged_case_args(manifest, case), aliases)
    if cli_value is not None:
        value = cli_value
    if value is None:
        return None
    try:
        return int(str(value), 10)
    except ValueError as error:
        raise ValueError(
            f"setting {aliases[0]!r}/{environment_name} is not an integer: {value!r}"
        ) from error


def validate_declared_ubatch_geometry(
    manifest: dict[str, Any],
    case: dict[str, Any],
    identity: dict[str, Any],
) -> None:
    has_target, declared_target = lookup_dimension(identity, "ubatch")
    has_draft, declared_draft = lookup_dimension(identity, "effective_draft_ubatch")
    if not has_draft:
        return
    if not has_target:
        raise ValueError(
            f"case {case['name']!r} declares effective_draft_ubatch without ubatch"
        )
    if not isinstance(declared_target, int) or not isinstance(declared_draft, int):
        raise ValueError(
            f"case {case['name']!r} ubatch identities must be integers"
        )

    configured_target = resolved_int_setting(
        manifest,
        case,
        ("--ubatch-size", "--ubatch", "-ub"),
        "LLAMA_ARG_UBATCH",
    )
    if configured_target is not None and configured_target != declared_target:
        raise ValueError(
            f"case {case['name']!r} declares ubatch={declared_target}, "
            f"but its command/environment configures {configured_target}"
        )

    configured_draft = resolved_int_setting(
        manifest,
        case,
        ("--spec-draft-ubatch-size", "--ubatch-size-draft", "-ubd"),
        "LLAMA_ARG_SPEC_DRAFT_UBATCH",
    )
    effective_draft = (
        declared_target if configured_draft is None or configured_draft == 0 else configured_draft
    )
    if effective_draft != declared_draft:
        raise ValueError(
            f"case {case['name']!r} declares effective_draft_ubatch={declared_draft}, "
            f"but its command/environment resolves to {effective_draft}"
        )


def case_identity(manifest: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    contract = manifest.get("comparison_contract", {})
    common = contract.get("common_identity", {})
    identity = case.get("identity", {})
    if not isinstance(common, dict) or not isinstance(identity, dict):
        raise TypeError("comparison identities must be JSON objects")
    return deep_merge(common, identity)


def validate_request_identity(
    identity: dict[str, Any],
    request: dict[str, Any],
    case_name: str,
    step_name: str,
) -> None:
    """Fail closed when the declared sampler identity is not the sent request.

    Exact token comparisons are meaningful only when both sides receive the
    same sampler inputs.  The comparison contract already checks declared
    identities against one another; this check also binds those declarations
    to the actual request body recorded in each completion artifact.
    """

    bindings = (
        ("sampler.seed", "seed"),
        ("sampler.temperature", "temperature"),
    )
    for identity_path, request_key in bindings:
        declared, expected = lookup_dimension(identity, identity_path)
        if not declared:
            continue
        if request_key not in request:
            raise ValueError(
                f"case {case_name!r} step {step_name!r} declares {identity_path}="
                f"{expected!r}, but request field {request_key!r} is absent"
            )
        actual = request[request_key]
        if isinstance(actual, bool) or isinstance(expected, bool) or actual != expected:
            raise ValueError(
                f"case {case_name!r} step {step_name!r} declares {identity_path}="
                f"{expected!r}, but request field {request_key!r} is {actual!r}"
            )


def case_role(manifest: dict[str, Any], case: dict[str, Any]) -> str:
    if "role" in case:
        return str(case["role"])
    if manifest.get("reference") == str(case["name"]):
        return "golden"
    return "candidate"


def validate_case_graph(manifest: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    names = [str(case["name"]) for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("case names must be unique")

    names_set = set(names)
    roles = {str(case["name"]): case_role(manifest, case) for case in cases}
    contract = manifest.get("comparison_contract", {})
    explicit = bool(contract.get("require_explicit_references", False))
    legacy_reference = manifest.get("reference")
    required_keys = [str(value) for value in contract.get("required_identity_keys", [])]

    for case in cases:
        name = str(case["name"])
        role = case_role(manifest, case)
        reference = case.get("compare_to")
        if role not in {"golden", "candidate"}:
            raise ValueError(f"case {name!r} has invalid role {role!r}")
        if "artifact_dir" in case and role != "golden":
            raise ValueError(f"artifact-backed case {name!r} must have role 'golden'")
        if role == "golden":
            if reference is not None:
                raise ValueError(f"golden case {name!r} must not set compare_to")
        else:
            if reference is None:
                if explicit:
                    raise ValueError(f"candidate case {name!r} must set compare_to")
                reference = legacy_reference
            if reference is None:
                raise ValueError(f"candidate case {name!r} has no comparison reference")
            if str(reference) not in names_set:
                raise ValueError(f"case {name!r} references unknown case {reference!r}")
            if str(reference) == name:
                raise ValueError(f"case {name!r} cannot compare to itself")
            if roles[str(reference)] != "golden":
                raise ValueError(
                    f"case {name!r} must compare directly to a golden case, not {reference!r}"
                )

        identity = case_identity(manifest, case)
        for key in required_keys:
            present, _ = lookup_dimension(identity, key)
            if not present:
                raise ValueError(f"case {name!r} is missing required identity key {key!r}")
        validate_declared_ubatch_geometry(manifest, case, identity)


def capture_command(command: list[str], cwd: pathlib.Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": f"{type(error).__name__}: {error}"}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def hashed_file_record(path: pathlib.Path, hash_cache: dict[pathlib.Path, str]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved not in hash_cache:
        print(f"[provenance] hashing {resolved} ({resolved.stat().st_size} bytes)", flush=True)
        hash_cache[resolved] = sha256_file(resolved)
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": hash_cache[resolved],
    }


def linked_library_provenance(
    executable: pathlib.Path, hash_cache: dict[pathlib.Path, str]
) -> dict[str, Any]:
    result = capture_command(["ldd", str(executable)])
    libraries: list[dict[str, Any]] = []
    seen: set[pathlib.Path] = set()
    for line in str(result.get("stdout", "")).splitlines():
        match = re.search(r"(?:=>\s+)?(/[^\s]+)\s+\(0x", line)
        if match is None:
            continue
        path = pathlib.Path(match.group(1))
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        record = hashed_file_record(path, hash_cache)
        record["ldd_line"] = line.strip()
        libraries.append(record)
    result["libraries"] = libraries
    return result


def git_provenance(path: pathlib.Path, hash_cache: dict[pathlib.Path, str]) -> dict[str, Any]:
    root_result = capture_command(["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"])
    if root_result.get("returncode") != 0:
        return {"root_lookup": root_result}
    root = pathlib.Path(str(root_result["stdout"]).strip()).resolve()
    head = capture_command(["git", "rev-parse", "HEAD"], root)
    branch = capture_command(["git", "branch", "--show-current"], root)
    status = capture_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], root
    )
    diff = capture_command(["git", "diff", "--binary", "HEAD", "--"], root)
    staged = capture_command(["git", "diff", "--binary", "--cached", "HEAD", "--"], root)
    untracked_result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    untracked: list[dict[str, Any]] = []
    if untracked_result.returncode == 0:
        for raw_path in untracked_result.stdout.split(b"\0"):
            if not raw_path:
                continue
            relative = pathlib.Path(os.fsdecode(raw_path))
            source = root / relative
            if source.is_file():
                record = hashed_file_record(source, hash_cache)
                record["relative_path"] = str(relative)
                untracked.append(record)

    fingerprint = {
        "status": status.get("stdout", ""),
        "unstaged_diff_sha256": sha256_bytes(str(diff.get("stdout", "")).encode("utf-8")),
        "staged_diff_sha256": sha256_bytes(str(staged.get("stdout", "")).encode("utf-8")),
        "untracked": [
            {"relative_path": item["relative_path"], "sha256": item["sha256"]}
            for item in untracked
        ],
    }
    return {
        "root": str(root),
        "head": str(head.get("stdout", "")).strip(),
        "branch": str(branch.get("stdout", "")).strip(),
        "status": str(status.get("stdout", "")),
        "unstaged_diff_sha256": fingerprint["unstaged_diff_sha256"],
        "staged_diff_sha256": fingerprint["staged_diff_sha256"],
        "untracked": untracked,
        "dirty_fingerprint_sha256": canonical_sha256(fingerprint),
    }


def cmake_provenance(executable: pathlib.Path, hash_cache: dict[pathlib.Path, str]) -> dict[str, Any]:
    cache_path = next(
        (parent / "CMakeCache.txt" for parent in executable.parents if (parent / "CMakeCache.txt").is_file()),
        None,
    )
    if cache_path is None:
        return {}
    record = hashed_file_record(cache_path, hash_cache)
    entries: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        entries[key] = value
    record["entries"] = entries
    return record


def host_provenance() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "lscpu": capture_command(["lscpu", "--json"]),
        "nvidia_smi": capture_command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,pci.bus_id,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ]
        ),
        "nvidia_smi_full": capture_command(["nvidia-smi", "-q"]),
        "nvcc": capture_command(["nvcc", "--version"]),
        "ccache": capture_command(["ccache", "--show-stats"]),
    }


def request_json(
    method: str,
    url: str,
    body: Any | None = None,
    timeout: float = 5.0,
    headers: dict[str, str] | None = None,
) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def clean_environment(extra: dict[str, str]) -> dict[str, str]:
    # Keep the executable search path and CUDA toolkit location, but do not let
    # LLAMA_ARG_* or earlier experiment variables leak into comparison cases.
    environment = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LC_ALL": "C",
    }
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        environment["CUDA_PATH"] = cuda_path
    environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def process_tree_pids(root_pid: int) -> set[int]:
    parent_by_pid: dict[int, int] = {}
    for stat_path in pathlib.Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = stat_path.read_text(encoding="utf-8", errors="replace")
            closing = stat.rfind(")")
            fields = stat[closing + 2 :].split()
            parent_by_pid[int(stat_path.parent.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue

    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for child, parent in parent_by_pid.items():
            if parent in result and child not in result:
                result.add(child)
                changed = True
    return result


def gpu_sample(pid: int) -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {}
    command = [
        nvidia_smi,
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    watched_pids = process_tree_pids(pid)
    processes: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 2:
            with contextlib.suppress(ValueError):
                process_pid = int(fields[0])
                if process_pid in watched_pids:
                    processes.append(
                        {"pid": process_pid, "vram_mib": int(fields[1])}
                    )
    if not processes:
        return {}
    return {
        "vram_mib": sum(item["vram_mib"] for item in processes),
        "gpu_processes": processes,
    }


def process_sample(pid: int) -> dict[str, Any]:
    status_path = pathlib.Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return {}
    wanted = {"VmRSS", "VmHWM", "VmSize", "RssAnon", "RssFile", "RssShmem"}
    result: dict[str, Any] = {}
    for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in wanted:
            result[key] = value.strip()
    return result


def stop_process(process: subprocess.Popen[bytes], grace_seconds: float = 20.0) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def load_artifact_case(
    manifest: dict[str, Any],
    case: dict[str, Any],
    output_root: pathlib.Path,
) -> dict[str, Any]:
    name = str(case["name"])
    source_dir = pathlib.Path(case["artifact_dir"]).expanduser().resolve()
    required_files = [
        source_dir / "response.json",
        source_dir / "summary.json",
        source_dir / "identity.json",
        source_dir / "request.json",
        source_dir / "prompt-tokens.json",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"artifact case {name!r} is missing files: {missing}")

    source_response = json.loads((source_dir / "response.json").read_text(encoding="utf-8"))
    source_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    source_identity = json.loads((source_dir / "identity.json").read_text(encoding="utf-8"))
    source_prompt = json.loads((source_dir / "prompt-tokens.json").read_text(encoding="utf-8"))
    declared_identity = case_identity(manifest, case)
    if source_identity != declared_identity:
        raise ValueError(
            f"artifact case {name!r} identity does not match its manifest declaration"
        )

    tokens = [int(token) for token in source_response.get("tokens", [])]
    content = str(source_response.get("content", "")).encode("utf-8")
    tokens_sha256 = sha256_bytes(json.dumps(tokens, separators=(",", ":")).encode("ascii"))
    content_sha256 = sha256_bytes(content)
    if source_summary.get("tokens_sha256") != tokens_sha256:
        raise ValueError(f"artifact case {name!r} token hash does not match response.json")
    if source_summary.get("content_sha256") != content_sha256:
        raise ValueError(f"artifact case {name!r} content hash does not match response.json")

    case_dir = output_root / name
    case_dir.mkdir(parents=True, exist_ok=False)
    copied_files: list[dict[str, Any]] = []
    for source in required_files:
        destination = case_dir / f"source-{source.name}"
        shutil.copy2(source, destination)
        copied_files.append(
            {
                "source": str(source),
                "copy": str(destination),
                "sha256": sha256_file(destination),
            }
        )
    json_dump(
        case_dir / "artifact-reference.json",
        {"source_dir": str(source_dir), "files": copied_files},
    )

    summary = {
        **source_summary,
        "name": name,
        "role": "golden",
        "compare_to": None,
        "identity": declared_identity,
        "artifact_source": str(source_dir),
    }
    json_dump(case_dir / "summary.json", summary)
    print(
        f"[{name}] loaded golden artifact tokens={len(tokens)} token_sha256={tokens_sha256}",
        flush=True,
    )
    step_summary = {
        "name": "completion",
        "type": "completion",
        "request_semantics_sha256": source_summary["request_semantics_sha256"],
        "prompt_tokens_sha256": source_summary["prompt_tokens_sha256"],
        "token_count": len(tokens),
        "tokens_sha256": tokens_sha256,
        "content_sha256": content_sha256,
    }
    step_result = {
        "name": "completion",
        "summary": step_summary,
        "prompt_tokens": [int(token) for token in source_prompt.get("tokens", [])],
        "tokens": tokens,
        "content": content,
    }
    return {
        "summary": summary,
        "steps": [step_result],
        "tokens": tokens,
        "content": content,
    }


def wait_for_server(base_url: str, process: subprocess.Popen[bytes], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        try:
            health = request_json("GET", f"{base_url}/health", timeout=2)
            if health.get("status") == "ok":
                return health
            last_error = f"health status is {health!r}"
        except (OSError, ValueError, urllib.error.HTTPError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise TimeoutError(f"server startup timed out: {last_error}")


def resolved_sequence(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Return the validated same-process action sequence.

    Older manifests describe one completion with the top-level request.  Keep
    their artifact layout stable by representing that request as an implicit
    completion while reporting that the legacy layout is in use.
    """

    raw_sequence = manifest.get("sequence")
    if raw_sequence is None:
        return [{"name": "completion", "type": "completion"}], True
    if not isinstance(raw_sequence, list) or not raw_sequence:
        raise ValueError("manifest sequence must be a non-empty JSON array")

    sequence: list[dict[str, Any]] = []
    names: set[str] = set()
    completion_names: set[str] = set()
    for index, raw_step in enumerate(raw_sequence):
        if not isinstance(raw_step, dict):
            raise TypeError(f"sequence step {index} must be a JSON object")
        step = dict(raw_step)
        name = str(step.get("name", ""))
        if not name:
            raise ValueError(f"sequence step {index} must have a non-empty name")
        if name in names:
            raise ValueError(f"sequence step name {name!r} is duplicated")
        names.add(name)

        step_type = str(step.get("type", "completion"))
        step["type"] = step_type
        if step_type == "completion":
            prompt_from = step.get("prompt_from")
            if prompt_from is not None:
                if not isinstance(prompt_from, dict):
                    raise TypeError(f"completion step {name!r} prompt_from must be an object")
                source = str(prompt_from.get("step", ""))
                if source not in completion_names:
                    raise ValueError(
                        f"completion step {name!r} prompt_from must name an earlier "
                        f"completion, got {source!r}"
                    )
                for count_key in ("prompt_token_count", "response_token_count"):
                    if count_key in prompt_from:
                        count = prompt_from[count_key]
                        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                            raise ValueError(
                                f"completion step {name!r} {count_key} must be a "
                                "non-negative integer"
                            )
                if "suffix" in prompt_from and not isinstance(prompt_from["suffix"], str):
                    raise TypeError(f"completion step {name!r} prompt_from suffix must be a string")
            require_equal_to = step.get("require_equal_to")
            if require_equal_to is not None and str(require_equal_to) not in completion_names:
                raise ValueError(
                    f"completion step {name!r} require_equal_to must name an earlier "
                    f"completion, got {require_equal_to!r}"
                )
            request_override = step.get("request", {})
            if not isinstance(request_override, dict):
                raise TypeError(f"completion step {name!r} request must be an object")
            completion_names.add(name)
        elif step_type == "wait_server_state":
            state = str(step.get("state", ""))
            if state not in {"sleeping", "awake"}:
                raise ValueError(
                    f"wait_server_state step {name!r} state must be 'sleeping' or 'awake'"
                )
            for key, default in (("timeout_seconds", 120.0), ("poll_interval_seconds", 0.5)):
                value = float(step.get(key, default))
                if value <= 0:
                    raise ValueError(f"wait_server_state step {name!r} {key} must be positive")
        elif step_type == "reload_models":
            model = step.get("model")
            if model is not None and not isinstance(model, str):
                raise TypeError(f"reload_models step {name!r} model must be a string")
            timeout = float(step.get("timeout_seconds", 300))
            if timeout <= 0:
                raise ValueError(f"reload_models step {name!r} timeout_seconds must be positive")
        else:
            raise ValueError(f"sequence step {name!r} has unsupported type {step_type!r}")
        sequence.append(step)

    if not completion_names:
        raise ValueError("manifest sequence must contain at least one completion step")
    return sequence, False


def truncate_tokens(tokens: list[int], count: Any, label: str) -> list[int]:
    if count is None:
        return list(tokens)
    count_int = int(count)
    if count_int > len(tokens):
        raise ValueError(f"{label} count {count_int} exceeds available token count {len(tokens)}")
    return list(tokens[:count_int])


def build_continuation_prompt(
    base_url: str,
    prompt_from: dict[str, Any],
    completion_results: dict[str, dict[str, Any]],
    model: str | None = None,
) -> tuple[list[int], dict[str, Any]]:
    source_name = str(prompt_from["step"])
    source = completion_results[source_name]
    source_prompt = truncate_tokens(
        source["prompt_tokens"], prompt_from.get("prompt_token_count"), "prompt_from prompt"
    )
    source_response = truncate_tokens(
        source["tokens"], prompt_from.get("response_token_count"), "prompt_from response"
    )
    suffix = str(prompt_from.get("suffix", ""))
    suffix_tokens: list[int] = []
    if suffix:
        tokenize_request: dict[str, Any] = {
            "content": suffix,
            "add_special": False,
            "parse_special": True,
        }
        if model is not None:
            tokenize_request["model"] = model
        tokenized = request_json(
            "POST",
            f"{base_url}/tokenize",
            tokenize_request,
            timeout=30,
        )
        suffix_tokens = [int(token) for token in tokenized["tokens"]]

    prompt_tokens = [*source_prompt, *source_response, *suffix_tokens]
    construction = {
        "source_step": source_name,
        "source_prompt_count": len(source_prompt),
        "source_response_count": len(source_response),
        "suffix": suffix,
        "suffix_tokens": suffix_tokens,
        "prompt_tokens_sha256": canonical_sha256(prompt_tokens),
    }
    return prompt_tokens, construction


def wait_for_server_state(
    base_url: str,
    process: subprocess.Popen[bytes],
    case_name: str,
    step: dict[str, Any],
    step_dir: pathlib.Path,
) -> dict[str, Any]:
    state = str(step["state"])
    expected_sleeping = state == "sleeping"
    timeout = float(step.get("timeout_seconds", 120))
    poll_interval = float(step.get("poll_interval_seconds", 0.5))
    started = time.monotonic()
    deadline = started + timeout
    samples: list[dict[str, Any]] = []
    next_report = 0.0
    last_error = "no /props response"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited while waiting for state {state!r} with code {process.returncode}"
            )
        elapsed = time.monotonic() - started
        sample: dict[str, Any] = {
            "elapsed_seconds": round(elapsed, 3),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        try:
            props = request_json("GET", f"{base_url}/props", timeout=5)
            sample["props"] = props
            observed = props.get("is_sleeping")
            if observed is expected_sleeping:
                samples.append(sample)
                with (step_dir / "progress.jsonl").open("a", encoding="utf-8") as progress:
                    progress.write(json.dumps(sample, sort_keys=True) + "\n")
                summary = {
                    "name": str(step["name"]),
                    "type": "wait_server_state",
                    "expected_state": state,
                    "observed_is_sleeping": observed,
                    "elapsed_seconds": elapsed,
                    "sample_count": len(samples),
                }
                json_dump(step_dir / "summary.json", summary)
                print(
                    f"[{case_name}/{step['name']}] observed server state={state} "
                    f"after {elapsed:.1f}s",
                    flush=True,
                )
                return summary
            last_error = f"is_sleeping={observed!r}"
        except (OSError, ValueError, urllib.error.HTTPError) as caught:
            sample["error"] = str(caught)
            last_error = str(caught)
        samples.append(sample)
        with (step_dir / "progress.jsonl").open("a", encoding="utf-8") as progress:
            progress.write(json.dumps(sample, sort_keys=True) + "\n")
        if elapsed >= next_report:
            print(
                f"[{case_name}/{step['name']}] waiting for state={state} "
                f"elapsed={elapsed:.1f}s last={last_error}",
                flush=True,
            )
            next_report = elapsed + 10.0
        time.sleep(poll_interval)
    raise TimeoutError(f"timed out waiting for server state {state!r}: {last_error}")


def find_model_status(models_response: Any, model: str) -> str | None:
    if not isinstance(models_response, dict):
        return None
    for item in models_response.get("data", []):
        if not isinstance(item, dict):
            continue
        if item.get("id") != model and item.get("model") != model:
            continue
        status = item.get("status")
        if isinstance(status, dict) and isinstance(status.get("value"), str):
            return str(status["value"])
    return None


def reload_models(
    base_url: str,
    process: subprocess.Popen[bytes],
    case: dict[str, Any],
    case_name: str,
    step: dict[str, Any],
    step_dir: pathlib.Path,
    models_preset_path: pathlib.Path | None,
) -> dict[str, Any]:
    preset = case.get("router_preset")
    if not isinstance(preset, dict) or models_preset_path is None:
        raise ValueError(
            f"reload_models step {step['name']!r} requires case.router_preset"
        )
    replacement = pathlib.Path(preset["reload_file"]).expanduser().resolve()
    if not replacement.is_file():
        raise FileNotFoundError(replacement)

    model = step.get("model", case.get("request", {}).get("model"))
    if not isinstance(model, str) or not model:
        raise ValueError(
            f"reload_models step {step['name']!r} requires a model name in the step "
            "or case request"
        )
    timeout = float(step.get("timeout_seconds", 300))
    started = time.monotonic()
    models_before = request_json("GET", f"{base_url}/models", timeout=30)
    status_before = find_model_status(models_before, model)
    json_dump(step_dir / "models-before.json", models_before)
    if status_before != "loaded":
        raise RuntimeError(
            f"model {model!r} must be loaded before explicit reload, got {status_before!r}"
        )

    temporary = models_preset_path.with_name(models_preset_path.name + ".next")
    shutil.copyfile(replacement, temporary)
    os.replace(temporary, models_preset_path)
    replacement_record = {
        "source": str(replacement),
        "source_sha256": sha256_file(replacement),
        "active": str(models_preset_path),
        "active_sha256": sha256_file(models_preset_path),
    }
    json_dump(step_dir / "preset-replacement.json", replacement_record)

    reload_response = request_json("POST", f"{base_url}/models/reload", {}, timeout=timeout)
    json_dump(step_dir / "reload-response.json", reload_response)
    if not isinstance(reload_response, dict) or reload_response.get("success") is not True:
        raise RuntimeError(f"model reload failed: {reload_response!r}")

    deadline = time.monotonic() + timeout
    next_report = 0.0
    models_after: Any = None
    status_after: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"router exited during model reload with code {process.returncode}"
            )
        models_after = request_json("GET", f"{base_url}/models", timeout=30)
        status_after = find_model_status(models_after, model)
        elapsed = time.monotonic() - started
        sample = {
            "elapsed_seconds": round(elapsed, 3),
            "status": status_after,
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        with (step_dir / "progress.jsonl").open("a", encoding="utf-8") as progress:
            progress.write(json.dumps(sample, sort_keys=True) + "\n")
        if status_after == "unloaded":
            break
        if elapsed >= next_report:
            print(
                f"[{case_name}/{step['name']}] waiting for model={model} to unload "
                f"after reload; status={status_after!r} elapsed={elapsed:.1f}s",
                flush=True,
            )
            next_report = elapsed + 10.0
        time.sleep(0.25)
    if status_after != "unloaded":
        raise TimeoutError(
            f"model {model!r} did not become unloaded after preset reload; "
            f"last status={status_after!r}"
        )
    json_dump(step_dir / "models-after.json", models_after)
    elapsed = time.monotonic() - started
    summary = {
        "name": str(step["name"]),
        "type": "reload_models",
        "model": model,
        "status_before": status_before,
        "status_after": status_after,
        "elapsed_seconds": elapsed,
        "replacement_sha256": replacement_record["source_sha256"],
    }
    json_dump(step_dir / "summary.json", summary)
    print(
        f"[{case_name}/{step['name']}] explicit router reload changed model={model} "
        f"from {status_before} to {status_after} in {elapsed:.1f}s",
        flush=True,
    )
    return summary


def compare_tokens(reference: list[int], candidate: list[int]) -> dict[str, Any]:
    shared = min(len(reference), len(candidate))
    mismatch = next((index for index in range(shared) if reference[index] != candidate[index]), None)
    exact = mismatch is None and len(reference) == len(candidate)
    if mismatch is None and not exact:
        mismatch = shared
    context_start = max(0, (mismatch or 0) - 8)
    context_end = min(max(len(reference), len(candidate)), (mismatch or 0) + 9)
    return {
        "exact": exact,
        "first_mismatch": mismatch,
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "reference_context": reference[context_start:context_end],
        "candidate_context": candidate[context_start:context_end],
    }


def compare_completion_sequences(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    reference_steps = reference.get("steps", [])
    candidate_steps = candidate.get("steps", [])
    reference_names = [str(step["name"]) for step in reference_steps]
    candidate_names = [str(step["name"]) for step in candidate_steps]
    order_exact = reference_names == candidate_names
    step_comparisons: list[dict[str, Any]] = []

    for index in range(max(len(reference_steps), len(candidate_steps))):
        if index >= len(reference_steps) or index >= len(candidate_steps):
            step_comparisons.append(
                {
                    "index": index,
                    "reference_name": (
                        str(reference_steps[index]["name"])
                        if index < len(reference_steps)
                        else None
                    ),
                    "candidate_name": (
                        str(candidate_steps[index]["name"])
                        if index < len(candidate_steps)
                        else None
                    ),
                    "exact": False,
                    "content_exact": False,
                    "request_semantics_exact": False,
                    "prompt_tokens_exact": False,
                }
            )
            continue
        reference_step = reference_steps[index]
        candidate_step = candidate_steps[index]
        comparison = compare_tokens(reference_step["tokens"], candidate_step["tokens"])
        comparison.update(
            {
                "index": index,
                "reference_name": str(reference_step["name"]),
                "candidate_name": str(candidate_step["name"]),
                "content_exact": reference_step["content"] == candidate_step["content"],
                "request_semantics_exact": (
                    reference_step["summary"]["request_semantics_sha256"]
                    == candidate_step["summary"]["request_semantics_sha256"]
                ),
                "prompt_tokens_exact": (
                    reference_step["summary"]["prompt_tokens_sha256"]
                    == candidate_step["summary"]["prompt_tokens_sha256"]
                ),
            }
        )
        step_comparisons.append(comparison)

    tokens_exact = order_exact and all(step.get("exact", False) for step in step_comparisons)
    content_exact = order_exact and all(
        step.get("content_exact", False) for step in step_comparisons
    )
    request_semantics_exact = order_exact and all(
        step.get("request_semantics_exact", False) for step in step_comparisons
    )
    prompt_tokens_exact = order_exact and all(
        step.get("prompt_tokens_exact", False) for step in step_comparisons
    )
    first_failed_step = next(
        (
            step
            for step in step_comparisons
            if not step.get("exact", False)
            or not step.get("content_exact", False)
            or not step.get("request_semantics_exact", False)
            or not step.get("prompt_tokens_exact", False)
        ),
        None,
    )
    return {
        "exact": tokens_exact,
        "content_exact": content_exact,
        "request_semantics_exact": request_semantics_exact,
        "prompt_tokens_exact": prompt_tokens_exact,
        "step_order_exact": order_exact,
        "reference_steps": reference_names,
        "candidate_steps": candidate_names,
        "first_failed_step": None if first_failed_step is None else first_failed_step["index"],
        "first_mismatch": (
            None if first_failed_step is None else first_failed_step.get("first_mismatch")
        ),
        "steps": step_comparisons,
    }


def run_completion_step(
    manifest: dict[str, Any],
    case: dict[str, Any],
    step: dict[str, Any],
    step_dir: pathlib.Path,
    base_url: str,
    process: subprocess.Popen[bytes],
    environment: dict[str, str],
    request_body: dict[str, Any],
    completion_results: dict[str, dict[str, Any]],
    prompt_prefix: dict[str, Any] | None,
) -> dict[str, Any]:
    case_name = str(case["name"])
    step_name = str(step["name"])
    request_for_step = deep_merge(request_body, case.get("request", {}))
    request_for_step = deep_merge(request_for_step, step.get("request", {}))
    validate_request_identity(
        case_identity(manifest, case), request_for_step, case_name, step_name
    )
    request_for_step["return_tokens"] = True
    request_for_step["stream"] = False
    prompt_tokens: list[int] | None = None

    prompt_from = step.get("prompt_from")
    if prompt_from is not None and prompt_prefix is not None:
        raise ValueError(
            f"completion step {step_name!r} cannot combine prompt_from and prompt_prefix"
        )
    if prompt_from is not None:
        prompt_tokens, construction = build_continuation_prompt(
            base_url,
            prompt_from,
            completion_results,
            str(request_for_step["model"]) if "model" in request_for_step else None,
        )
        request_for_step["prompt"] = prompt_tokens
        json_dump(step_dir / "prompt-construction.json", construction)
    elif prompt_prefix is not None:
        if not isinstance(request_for_step.get("prompt"), str):
            raise ValueError("prompt_prefix requires the base request prompt to be a string")
        source_path = pathlib.Path(prompt_prefix["response_file"]).expanduser().resolve()
        source_response = json.loads(source_path.read_text(encoding="utf-8"))
        source_tokens = [int(token) for token in source_response.get("tokens", [])]
        prefix_count = int(prompt_prefix["token_count"])
        if prefix_count < 0 or prefix_count > len(source_tokens):
            raise ValueError(
                f"prompt_prefix token_count {prefix_count} is outside source response "
                f"range 0..{len(source_tokens)}"
            )
        tokenize_request = {
            "content": request_for_step["prompt"],
            "add_special": True,
            "parse_special": True,
        }
        if "model" in request_for_step:
            tokenize_request["model"] = request_for_step["model"]
        tokenized = request_json(
            "POST",
            f"{base_url}/tokenize",
            tokenize_request,
            timeout=30,
        )
        base_prompt_tokens = [int(token) for token in tokenized["tokens"]]
        prompt_tokens = base_prompt_tokens + source_tokens[:prefix_count]
        request_for_step["prompt"] = prompt_tokens
        json_dump(
            step_dir / "prompt-prefix.json",
            {
                "base_prompt_tokens": len(base_prompt_tokens),
                "prefix_tokens": prefix_count,
                "response_file": str(source_path),
                "source_tokens_sha256": canonical_sha256(source_tokens),
            },
        )

    if prompt_tokens is None:
        prompt = request_for_step.get("prompt")
        if isinstance(prompt, str):
            tokenize_request = {
                "content": prompt,
                "add_special": True,
                "parse_special": True,
            }
            if "model" in request_for_step:
                tokenize_request["model"] = request_for_step["model"]
            tokenized = request_json(
                "POST",
                f"{base_url}/tokenize",
                tokenize_request,
                timeout=30,
            )
            prompt_tokens = [int(token) for token in tokenized["tokens"]]
        elif isinstance(prompt, list):
            prompt_tokens = [int(token) for token in prompt]
        else:
            raise ValueError("exactness runner requires prompt to be a string or token-id list")

    prompt_tokens_sha256 = canonical_sha256(prompt_tokens)
    request_semantics_sha256 = canonical_sha256(request_for_step)
    json_dump(
        step_dir / "prompt-tokens.json",
        {
            "count": len(prompt_tokens),
            "sha256": prompt_tokens_sha256,
            "tokens": prompt_tokens,
        },
    )
    json_dump(step_dir / "request.json", request_for_step)

    completion_timeout = float(
        step.get(
            "timeout_seconds",
            case.get("timeout_seconds", manifest.get("timeout_seconds", 3600)),
        )
    )
    completion_started = time.monotonic()
    stderr_file = (step_dir / "request.stderr").open("wb")
    completion = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys,urllib.request; body=open(sys.argv[2],'rb').read(); "
                "r=urllib.request.Request(sys.argv[1],data=body,method='POST',"
                "headers={'Content-Type':'application/json'}); "
                "open(sys.argv[3],'wb').write(urllib.request.urlopen("
                "r,timeout=float(sys.argv[4])).read())"
            ),
            f"{base_url}/completion",
            str(step_dir / "request.json"),
            str(step_dir / "response.raw.json"),
            str(completion_timeout),
        ],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=stderr_file,
    )
    samples: list[dict[str, Any]] = []
    peak_vram_mib = 0
    next_report = 0.0
    try:
        while completion.poll() is None:
            elapsed = time.monotonic() - completion_started
            if elapsed > completion_timeout + 5:
                completion.kill()
                completion.wait(timeout=10)
                raise TimeoutError(f"completion timed out after {completion_timeout}s")
            sample: dict[str, Any] = {
                "elapsed_seconds": round(elapsed, 3),
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "process": process_sample(process.pid),
                **gpu_sample(process.pid),
            }
            peak_vram_mib = max(peak_vram_mib, int(sample.get("vram_mib", 0)))
            try:
                slots_url = f"{base_url}/slots"
                if "model" in request_for_step:
                    slots_url += "?model=" + urllib.parse.quote(
                        str(request_for_step["model"]), safe=""
                    )
                sample["slots"] = request_json("GET", slots_url, timeout=2)
            except (OSError, ValueError, urllib.error.HTTPError) as slot_error:
                sample["slots_error"] = str(slot_error)
            samples.append(sample)
            with (step_dir / "progress.jsonl").open("a", encoding="utf-8") as progress:
                progress.write(json.dumps(sample, sort_keys=True) + "\n")
            if elapsed >= next_report:
                predicted = "?"
                slots = sample.get("slots")
                if isinstance(slots, list) and slots:
                    predicted = str(slots[0].get("next_token", slots[0].get("n_decoded", "?")))
                print(
                    f"[{case_name}/{step_name}] elapsed={elapsed:.1f}s "
                    f"progress={predicted} vram={sample.get('vram_mib', '?')} MiB",
                    flush=True,
                )
                next_report = elapsed + 10.0
            time.sleep(float(manifest.get("poll_interval_seconds", 1)))
    finally:
        stderr_file.close()
    if completion.returncode != 0:
        raise RuntimeError(f"completion client exited with code {completion.returncode}")

    response_body = json.loads((step_dir / "response.raw.json").read_bytes())
    json_dump(step_dir / "response.json", response_body)
    (step_dir / "output.txt").write_text(
        str(response_body.get("content", "")), encoding="utf-8"
    )
    tokens = [int(token) for token in response_body.get("tokens", [])]
    content = str(response_body.get("content", "")).encode("utf-8")
    elapsed = time.monotonic() - completion_started
    summary = {
        "name": step_name,
        "type": "completion",
        "request_semantics_sha256": request_semantics_sha256,
        "prompt_tokens_sha256": prompt_tokens_sha256,
        "prompt_token_count": len(prompt_tokens),
        "elapsed_seconds": elapsed,
        "token_count": len(tokens),
        "tokens_sha256": canonical_sha256(tokens),
        "content_sha256": sha256_bytes(content),
        "peak_vram_mib": peak_vram_mib,
        "timings": response_body.get("timings", {}),
    }
    json_dump(step_dir / "summary.json", summary)
    print(
        f"[{case_name}/{step_name}] complete tokens={len(tokens)} "
        f"token_sha256={summary['tokens_sha256']} peak_vram={peak_vram_mib} MiB",
        flush=True,
    )
    return {
        "name": step_name,
        "summary": summary,
        "prompt_tokens": prompt_tokens,
        "tokens": tokens,
        "content": content,
    }


def run_case(
    manifest: dict[str, Any],
    case: dict[str, Any],
    server_path: pathlib.Path,
    request_body: dict[str, Any],
    output_root: pathlib.Path,
) -> dict[str, Any]:
    name = str(case["name"])
    case_dir = output_root / name
    case_dir.mkdir(parents=True, exist_ok=False)
    host = str(manifest.get("host", "127.0.0.1"))
    port = free_port(host)
    common_args = [str(value) for value in manifest.get("common_args", [])]
    common_args.extend(str(value) for value in case.get("common_args", []))
    case_args = [str(value) for value in case.get("args", [])]
    models_preset_path: pathlib.Path | None = None
    router_preset = case.get("router_preset")
    if router_preset is not None:
        if not isinstance(router_preset, dict):
            raise TypeError(f"case {name!r} router_preset must be an object")
        if last_option_value(
            [*common_args, *case_args], ("--models-preset",)
        ) is not None:
            raise ValueError(
                f"case {name!r} cannot combine router_preset with --models-preset"
            )
        initial_preset = pathlib.Path(router_preset["initial_file"]).expanduser().resolve()
        reload_preset = pathlib.Path(router_preset["reload_file"]).expanduser().resolve()
        if not initial_preset.is_file():
            raise FileNotFoundError(initial_preset)
        if not reload_preset.is_file():
            raise FileNotFoundError(reload_preset)
        models_preset_path = case_dir / "router-models.ini"
        shutil.copyfile(initial_preset, models_preset_path)
        json_dump(
            case_dir / "router-preset-sources.json",
            {
                "initial_file": str(initial_preset),
                "initial_sha256": sha256_file(initial_preset),
                "reload_file": str(reload_preset),
                "reload_sha256": sha256_file(reload_preset),
                "active_file": str(models_preset_path),
            },
        )
        case_args.extend(["--models-preset", str(models_preset_path)])
    command = [str(server_path), *common_args, *case_args, "--host", host, "--port", str(port)]
    environment = clean_environment({
        **manifest.get("environment", {}),
        **case.get("environment", {}),
    })
    identity = case_identity(manifest, case)
    sequence, legacy_layout = resolved_sequence(manifest)
    global_prompt_prefix = case.get("prompt_prefix", manifest.get("prompt_prefix"))
    if not legacy_layout and global_prompt_prefix is not None:
        raise ValueError(
            "explicit sequences must attach prompt_prefix to a completion step, "
            "not the manifest or case"
        )

    (case_dir / "server-command.txt").write_text(
        shlex.join(["env", "-i", *[f"{key}={value}" for key, value in environment.items()], *command]) + "\n",
        encoding="utf-8",
    )
    json_dump(case_dir / "environment.json", environment)
    json_dump(case_dir / "identity.json", identity)

    log_file = (case_dir / "server.log").open("wb")
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (case_dir / "server.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    base_url = f"http://{host}:{port}"
    started = time.monotonic()
    error: BaseException | None = None
    peak_vram_mib = 0
    completion_results: dict[str, dict[str, Any]] = {}
    ordered_results: list[dict[str, Any]] = []
    step_summaries: list[dict[str, Any]] = []

    try:
        health = wait_for_server(base_url, process, float(manifest.get("startup_timeout_seconds", 600)))
        json_dump(case_dir / "health.json", health)
        print(f"[{name}] server ready after {time.monotonic() - started:.1f}s", flush=True)

        for index, step in enumerate(sequence):
            if legacy_layout:
                step_dir = case_dir
            else:
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(step["name"])).strip("-")
                step_dir = case_dir / "steps" / f"{index:02d}-{safe_name or 'step'}"
                step_dir.mkdir(parents=True, exist_ok=False)

            if step["type"] == "completion":
                prompt_prefix = step.get("prompt_prefix", global_prompt_prefix)
                result = run_completion_step(
                    manifest,
                    case,
                    step,
                    step_dir,
                    base_url,
                    process,
                    environment,
                    request_body,
                    completion_results,
                    prompt_prefix,
                )
                require_equal_to = step.get("require_equal_to")
                if require_equal_to is not None:
                    earlier = completion_results[str(require_equal_to)]
                    within_case = compare_tokens(earlier["tokens"], result["tokens"])
                    within_case["content_exact"] = earlier["content"] == result["content"]
                    within_case["request_semantics_exact"] = (
                        earlier["summary"]["request_semantics_sha256"]
                        == result["summary"]["request_semantics_sha256"]
                    )
                    within_case["prompt_tokens_exact"] = (
                        earlier["summary"]["prompt_tokens_sha256"]
                        == result["summary"]["prompt_tokens_sha256"]
                    )
                    within_case["reference"] = str(require_equal_to)
                    result["summary"]["within_case_comparison"] = within_case
                    json_dump(step_dir / "summary.json", result["summary"])
                    json_dump(step_dir / "within-case-comparison.json", within_case)
                    if not all(
                        within_case[key]
                        for key in (
                            "exact",
                            "content_exact",
                            "request_semantics_exact",
                            "prompt_tokens_exact",
                        )
                    ):
                        raise RuntimeError(
                            f"completion step {step['name']!r} did not match required "
                            f"earlier step {require_equal_to!r}"
                        )
                completion_results[result["name"]] = result
                ordered_results.append(result)
                step_summaries.append(result["summary"])
                peak_vram_mib = max(
                    peak_vram_mib, int(result["summary"].get("peak_vram_mib", 0))
                )
            elif step["type"] == "wait_server_state":
                step_summaries.append(
                    wait_for_server_state(base_url, process, name, step, step_dir)
                )
            else:
                step_summaries.append(
                    reload_models(
                        base_url,
                        process,
                        case,
                        name,
                        step,
                        step_dir,
                        models_preset_path,
                    )
                )
    except BaseException as caught:
        error = caught
    finally:
        stop_process(process)
        log_file.close()

    elapsed = time.monotonic() - started
    if error is not None:
        json_dump(case_dir / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise error
    if not ordered_results:
        raise RuntimeError("case completed without a completion result")
    final_result = ordered_results[-1]
    summary = {
        "name": name,
        "role": case_role(manifest, case),
        "compare_to": case.get("compare_to", manifest.get("reference")),
        "identity": identity,
        "server": str(server_path),
        "server_sha256": sha256_file(server_path),
        "elapsed_seconds": elapsed,
        "peak_vram_mib": peak_vram_mib,
        "completion_count": len(ordered_results),
        "steps": step_summaries,
        "request_semantics_sha256": canonical_sha256(
            [result["summary"]["request_semantics_sha256"] for result in ordered_results]
        ),
        "prompt_tokens_sha256": canonical_sha256(
            [result["summary"]["prompt_tokens_sha256"] for result in ordered_results]
        ),
        "token_count": sum(len(result["tokens"]) for result in ordered_results),
        "tokens_sha256": canonical_sha256([result["tokens"] for result in ordered_results]),
        "content_sha256": canonical_sha256(
            [result["summary"]["content_sha256"] for result in ordered_results]
        ),
        "timings": final_result["summary"].get("timings", {}),
    }
    if legacy_layout:
        summary.update(final_result["summary"])
        summary.update(
            {
                "name": name,
                "role": case_role(manifest, case),
                "compare_to": case.get("compare_to", manifest.get("reference")),
                "identity": identity,
                "server": str(server_path),
                "server_sha256": sha256_file(server_path),
                "elapsed_seconds": elapsed,
                "steps": step_summaries,
            }
        )
    json_dump(case_dir / "summary.json", summary)
    print(
        f"[{name}] complete completion_steps={len(ordered_results)} "
        f"token_sha256={summary['tokens_sha256']} peak_vram={peak_vram_mib} MiB",
        flush=True,
    )
    return {
        "summary": summary,
        "steps": ordered_results,
        "tokens": final_result["tokens"],
        "content": final_result["content"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=pathlib.Path, help="JSON matrix manifest")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="record exact comparisons but return success when cases differ",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if "request_file" in manifest:
        request_path = pathlib.Path(manifest["request_file"]).expanduser().resolve()
        request_body = json.loads(request_path.read_text(encoding="utf-8"))
    else:
        request_body = dict(manifest["request"])
    _, legacy_layout = resolved_sequence(manifest)
    output_root = pathlib.Path(manifest["output_dir"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "manifest.json").write_bytes(args.manifest.read_bytes())

    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError("manifest must contain at least one case")
    if not all(isinstance(case, dict) for case in cases):
        raise TypeError("every manifest case must be a JSON object")
    validate_case_graph(manifest, cases)
    if not legacy_layout and any("artifact_dir" in case for case in cases):
        raise ValueError("artifact-backed golden cases currently require a single completion")

    default_server = manifest.get("server")
    server_paths: dict[str, pathlib.Path] = {}
    for case in cases:
        name = str(case["name"])
        if "artifact_dir" in case:
            continue
        server_value = case.get("server", default_server)
        if server_value is None:
            raise ValueError(f"case {name!r} does not specify a server")
        server_path = pathlib.Path(server_value).expanduser().resolve()
        if not server_path.is_file():
            raise FileNotFoundError(server_path)
        server_paths[name] = server_path

    unique_servers = sorted({path for path in server_paths.values()}, key=str)
    hash_cache: dict[pathlib.Path, str] = {}
    server_provenance: list[dict[str, Any]] = []
    for server_path in unique_servers:
        server_provenance.append(
            {
                "executable": hashed_file_record(server_path, hash_cache),
                "linked_libraries": linked_library_provenance(server_path, hash_cache),
                "git": git_provenance(server_path, hash_cache),
                "cmake": cmake_provenance(server_path, hash_cache),
            }
        )

    input_file_provenance: list[dict[str, Any]] = []
    for item in manifest.get("input_files", []):
        if isinstance(item, str):
            role = "input"
            input_path = pathlib.Path(item).expanduser().resolve()
        elif isinstance(item, dict):
            role = str(item.get("role", "input"))
            input_path = pathlib.Path(item["path"]).expanduser().resolve()
        else:
            raise TypeError("input_files entries must be paths or objects with role/path")
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        record = hashed_file_record(input_path, hash_cache)
        record["role"] = role
        input_file_provenance.append(record)

    artifact_provenance: list[dict[str, Any]] = []
    for case in cases:
        if "artifact_dir" not in case:
            continue
        artifact_dir = pathlib.Path(case["artifact_dir"]).expanduser().resolve()
        files: list[dict[str, Any]] = []
        for filename in (
            "response.json",
            "summary.json",
            "identity.json",
            "request.json",
            "prompt-tokens.json",
        ):
            path = artifact_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(hashed_file_record(path, hash_cache))
        for parent_filename in ("manifest.json", "provenance.json"):
            path = artifact_dir.parent / parent_filename
            if path.is_file():
                files.append(hashed_file_record(path, hash_cache))
        artifact_provenance.append(
            {"case": str(case["name"]), "source_dir": str(artifact_dir), "files": files}
        )

    provenance = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "servers": server_provenance,
        "input_files": input_file_provenance,
        "artifact_references": artifact_provenance,
        "host": host_provenance(),
        "request_sha256": canonical_sha256(request_body),
        "runner": str(pathlib.Path(__file__).resolve()),
        "runner_sha256": sha256_file(pathlib.Path(__file__).resolve()),
    }
    json_dump(output_root / "provenance.json", provenance)

    results: dict[str, dict[str, Any]] = {}
    try:
        for case in cases:
            name = str(case["name"])
            if "artifact_dir" in case:
                results[name] = load_artifact_case(manifest, case, output_root)
            else:
                results[name] = run_case(
                    manifest, case, server_paths[name], request_body, output_root
                )
    except BaseException as error:
        json_dump(output_root / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise

    comparisons: dict[str, Any] = {}
    all_exact = True
    required_identity_keys = [
        str(value)
        for value in manifest.get("comparison_contract", {}).get("required_identity_keys", [])
    ]
    legacy_reference = manifest.get("reference")
    references: dict[str, str] = {}
    for case in cases:
        name = str(case["name"])
        result = results[name]
        role = case_role(manifest, case)
        if role == "golden":
            comparisons[name] = {"role": "golden"}
            print(f"[{name}] golden case; no comparison required", flush=True)
            continue

        reference_name = str(case.get("compare_to", legacy_reference))
        reference = results[reference_name]
        references[name] = reference_name
        comparison = compare_completion_sequences(reference, result)
        comparison["reference"] = reference_name
        comparison["role"] = role

        dimension_mismatches: dict[str, Any] = {}
        candidate_identity = result["summary"]["identity"]
        reference_identity = reference["summary"]["identity"]
        for key in required_identity_keys:
            candidate_present, candidate_value = lookup_dimension(candidate_identity, key)
            reference_present, reference_value = lookup_dimension(reference_identity, key)
            if not candidate_present or not reference_present or candidate_value != reference_value:
                dimension_mismatches[key] = {
                    "candidate_present": candidate_present,
                    "candidate": candidate_value,
                    "reference_present": reference_present,
                    "reference": reference_value,
                }

        comparison["identity_exact"] = not dimension_mismatches
        comparison["identity_mismatches"] = dimension_mismatches
        comparison["contract_exact"] = (
            comparison["identity_exact"]
            and comparison["request_semantics_exact"]
            and comparison["prompt_tokens_exact"]
        )
        comparisons[name] = comparison
        required = bool(case.get("require_equal", True))
        if required and (
            not comparison["contract_exact"]
            or not comparison["exact"]
            or not comparison["content_exact"]
        ):
            all_exact = False
        print(
            f"[{name}] vs [{reference_name}] contract_exact={comparison['contract_exact']} "
            f"tokens_exact={comparison['exact']} content_exact={comparison['content_exact']} "
            f"first_mismatch={comparison['first_mismatch']}",
            flush=True,
        )
    json_dump(output_root / "comparisons.json", comparisons)
    json_dump(
        output_root / "summary.json",
        {
            "all_required_exact": all_exact,
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "references": references,
            "results": {name: result["summary"] for name, result in results.items()},
        },
    )
    return 0 if all_exact or args.allow_mismatch else 1


if __name__ == "__main__":
    raise SystemExit(main())
