"""Shared fixtures for the feature-validation executable test suite."""

from __future__ import annotations

import copy
import hashlib
import pathlib
import subprocess
import tempfile

from feature_validation import core


def file_sha256(path: pathlib.Path) -> str:
    """Independent digest oracle used to test production provenance helpers."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FixtureMixin:
    temporary: tempfile.TemporaryDirectory[str]
    root: pathlib.Path
    repo: pathlib.Path
    executable: pathlib.Path

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.repo = self.root / "source tree"
        self.repo.mkdir()
        self.executable = self.repo / "fake bench"
        self.executable.write_text(
            "#!/usr/bin/env python3\nprint('metric=100.0')\n",
            encoding="utf-8",
        )
        self.executable.chmod(0o755)
        (self.repo / ".gitignore").write_text("build/\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "tests@example.invalid"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Feature Tests"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "fixture"], cwd=self.repo, check=True
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifest(self) -> dict:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        variant = {
            "source_root": str(self.repo),
            "expected_commit": head,
            "tree_policy": "clean",
            "executables": {
                "default": {
                    "path": str(self.executable),
                    "expected_sha256": file_sha256(self.executable),
                    "build": {"mode": "not_applicable", "reason": "test script"},
                }
            },
        }
        selection = {
            "workload": "decode",
            "tensor_layout": {"k": "q8_0", "v": "q8_0", "query_width": 1},
            "backend_capabilities": ["direct_standard_quant_attention"],
            "execution_mode": "direct_process",
        }
        command = {
            "common_args": [],
            "baseline_args": [],
            "candidate_args": [],
            "cli_schema": {"builtin": "none", "allow_positionals": False},
        }
        manifest = {
            "schema_version": 1,
            "study": {
                "id": "unit-fixture",
                "early_decision_target_seconds": {"minimum": 30, "maximum": 90},
            },
            "artifact_root": str(self.root / "artifacts"),
            "variants": {
                "baseline": copy.deepcopy(variant),
                "candidate": copy.deepcopy(variant),
            },
            "inputs": [],
            "repetition_policy": {
                "minimum_pairs": 3,
                "maximum_pairs": 5,
                "order": ["AB", "BA", "AB", "BA", "AB"],
                "confidence_level": 0.95,
                "extension_rule": {
                    "kind": "extend_only_if_inconclusive",
                    "thresholds": {
                        "improvement_percent": 1.0,
                        "regression_percent": 1.0,
                        "equivalence_percent": 0.5,
                    },
                },
            },
            "stages": [
                {
                    "id": "exactness",
                    "purpose": "exactness",
                    "resource": "cpu",
                    "timeout_seconds": 5,
                    "mandatory": True,
                    "progress": "fake process exits immediately",
                    "selection": {**copy.deepcopy(selection), "workload": "exactness"},
                    "command": copy.deepcopy(command),
                    "comparison": {"mode": "stdout_sha256"},
                },
                {
                    "id": "prefill-smoke",
                    "purpose": "smoke",
                    "resource": "cpu",
                    "timeout_seconds": 5,
                    "mandatory": True,
                    "progress": "one line per fake process",
                    "selection": {
                        **copy.deepcopy(selection),
                        "workload": "prefill",
                        "duration_class": "short",
                    },
                    "command": copy.deepcopy(command),
                    "metric": {"regex": r"metric=([0-9.]+)", "direction": "higher"},
                },
                {
                    "id": "decode-smoke",
                    "purpose": "smoke",
                    "resource": "cpu",
                    "timeout_seconds": 5,
                    "mandatory": True,
                    "progress": "one line per fake process",
                    "selection": {
                        **copy.deepcopy(selection),
                        "duration_class": "short",
                        "cuda_graph_replay": {
                            "applicability": "not_applicable",
                            "reason": "CPU-only fake fixture",
                        },
                    },
                    "command": copy.deepcopy(command),
                    "metric": {"regex": r"metric=([0-9.]+)", "direction": "higher"},
                },
                {
                    "id": "kernel",
                    "purpose": "kernel_screen",
                    "resource": "cpu",
                    "timeout_seconds": 5,
                    "mandatory": True,
                    "progress": "one line per fake process",
                    "selection": {
                        **copy.deepcopy(selection),
                        "workload": "prefill",
                        "execution_mode": "direct_command",
                    },
                    "command": copy.deepcopy(command),
                    "screens": [
                        {
                            "id": "low",
                            "span_class": "low",
                            "span_tokens": 64,
                            "ubatch_occurrences": 1,
                            "args": [],
                            "weight": 1,
                        },
                        {
                            "id": "mid",
                            "span_class": "mid",
                            "span_tokens": 256,
                            "ubatch_occurrences": 2,
                            "args": [],
                            "weight": 2,
                        },
                        {
                            "id": "high",
                            "span_class": "high",
                            "span_tokens": 512,
                            "ubatch_occurrences": 1,
                            "args": [],
                            "weight": 1,
                        },
                    ],
                    "aggregation": {
                        "mode": "weighted_harmonic",
                        "weight_source": "real_ubatch_geometry",
                    },
                    "metric": {"regex": r"metric=([0-9.]+)", "direction": "higher"},
                },
                {
                    "id": "production",
                    "purpose": "production_confirmation",
                    "resource": "cpu",
                    "timeout_seconds": 5,
                    "mandatory": True,
                    "progress": "one line per fake process",
                    "selection": {
                        **copy.deepcopy(selection),
                        "execution_mode": "production_binary",
                        "context_depth_tokens": 4096,
                    },
                    "command": copy.deepcopy(command),
                    "metric": {"regex": r"metric=([0-9.]+)", "direction": "higher"},
                },
                {
                    "id": "long",
                    "purpose": "long_context_acceptance",
                    "resource": "cpu",
                    "timeout_seconds": 5,
                    "mandatory": True,
                    "progress": "one line per fake process",
                    "selection": {
                        **copy.deepcopy(selection),
                        "execution_mode": "production_binary",
                        "context_depth_tokens": 32768,
                        "acceptance_class": "long_context",
                    },
                    "command": copy.deepcopy(command),
                    "metric": {"regex": r"metric=([0-9.]+)", "direction": "higher"},
                },
            ],
        }
        decision_policy = {
            "acceptable_decisions": ["improvement", "equivalent"],
            "regression": "fail",
            "inconclusive_after_maximum": "unresolved_fail",
        }
        for stage in manifest["stages"]:
            if stage["purpose"] != "exactness":
                stage["decision_policy"] = copy.deepcopy(decision_policy)
        return manifest

    def cmake_manifest(
        self,
    ) -> tuple[dict, pathlib.Path, pathlib.Path, pathlib.Path]:
        build = self.repo / "build"
        built_executable = build / "bin" / "fake bench"
        built_executable.parent.mkdir(parents=True)
        built_executable.write_text(
            "#!/usr/bin/env python3\nprint('metric=100.0')\n",
            encoding="utf-8",
        )
        built_executable.chmod(0o755)
        cache = build / "CMakeCache.txt"
        cache.write_text(
            "\n".join(
                [
                    f"CMAKE_HOME_DIRECTORY:INTERNAL={self.repo}",
                    f"CMAKE_CACHEFILE_DIR:INTERNAL={build}",
                    "CMAKE_BUILD_TYPE:STRING=Release",
                    "GGML_CUDA:BOOL=OFF",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        sidecar = pathlib.Path(f"{built_executable}.build-provenance.json")
        core.write_json_atomic(
            sidecar,
            core.capture_cmake_build_provenance(self.repo, built_executable, cache),
        )
        manifest = self.manifest()
        executable = {
            "path": str(built_executable),
            "expected_sha256": file_sha256(built_executable),
            "provenance_sidecar": {
                "path": str(sidecar),
                "sha256": file_sha256(sidecar),
            },
            "build": {
                "mode": "cmake_required",
                "cache": str(cache),
                "expected_options": {
                    "CMAKE_BUILD_TYPE": "Release",
                    "GGML_CUDA": "OFF",
                },
            },
        }
        for variant in manifest["variants"].values():
            variant["executables"] = {"default": copy.deepcopy(executable)}
        return manifest, built_executable, cache, sidecar
