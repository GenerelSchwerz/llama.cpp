#!/usr/bin/env python3

"""CPU-only tests for the feature/performance validation toolkit."""

from __future__ import annotations

import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from feature_validation import core, profiler, telemetry  # noqa: E402
from feature_validation.runner import StudyRunner, _verify_spec_identity  # noqa: E402


CLI_SPEC = importlib.util.spec_from_file_location(
    "feature_performance_validation_cli", SCRIPTS / "feature-performance-validation.py"
)
assert CLI_SPEC is not None and CLI_SPEC.loader is not None
CLI = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(CLI)


def file_sha256(path: pathlib.Path) -> str:
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
        subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=self.repo, check=True)
        subprocess.run(["git", "config", "user.name", "Feature Tests"], cwd=self.repo, check=True)
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=self.repo, check=True)

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
            "variants": {"baseline": copy.deepcopy(variant), "candidate": copy.deepcopy(variant)},
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


class QuotingAndArityTest(unittest.TestCase):
    def test_safe_quoting_round_trips_spaces_quotes_and_metacharacters(self) -> None:
        argv = ["/tmp/a b", "single'quote", 'double"quote', "$(do-not-run)", "semi;colon"]
        rendered = core.quote_argv(argv)
        self.assertEqual(shlex.split(rendered), argv)

    def test_elf_snapshot_is_stable_across_ldd_aslr_addresses(self) -> None:
        first = core.binary_snapshot(pathlib.Path("/bin/true"))
        second = core.binary_snapshot(pathlib.Path("/bin/true"))

        self.assertEqual(core.canonical_sha256(first), core.canonical_sha256(second))
        self.assertNotIn("stdout", first["ldd"])

    def test_zero_arity_option_rejects_separate_value(self) -> None:
        with self.assertRaisesRegex(core.ValidationError, "unexpected positional"):
            core.validate_argv(["--no-kv-offload", "1"])

    def test_zero_arity_option_rejects_equals_value(self) -> None:
        with self.assertRaisesRegex(core.ValidationError, "zero-arity"):
            core.validate_argv(["--no-kv-offload=1"])

    def test_common_no_kv_offload_is_bare(self) -> None:
        core.validate_argv(
            ["--no-kv-offload"],
            {"builtin": "llama", "allow_positionals": False},
            "/tmp/llama-cli",
        )
        with self.assertRaisesRegex(core.ValidationError, "unexpected positional"):
            core.validate_argv(
                ["--no-kv-offload", "1"],
                {"builtin": "llama", "allow_positionals": False},
                "/tmp/llama-cli",
            )

    def test_bench_no_kv_offload_short_and_long_aliases_take_values(self) -> None:
        schema = {"builtin": "llama", "allow_positionals": False}
        core.validate_argv(["-nkvo", "1"], schema, "/tmp/llama-bench")
        core.validate_argv(["--no-kv-offload", "0"], schema, "/tmp/llama-bench")
        with self.assertRaisesRegex(core.ValidationError, "requires one value"):
            core.validate_argv(["--no-kv-offload"], schema, "/tmp/llama-bench")

    def test_llama_schema_rejects_unidentified_harness_without_override(self) -> None:
        with self.assertRaisesRegex(core.ManifestError, "cannot identify target"):
            core.validate_argv([], {"builtin": "llama"}, "/tmp/custom-harness")

    def test_value_option_accepts_negative_numeric_value(self) -> None:
        core.validate_argv(["--kv-gpu-layers", "-1"])

    def test_taskset_is_rejected_anywhere(self) -> None:
        with self.assertRaisesRegex(core.ValidationError, "taskset"):
            core.reject_forbidden_tokens(["ncu", "/usr/bin/taskset", "llama-bench"])

    def test_profiler_wrapper_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(core.ValidationError, "wrapper"):
            core.validate_direct_target(pathlib.Path("/usr/bin/bash"), profiler=True)


class ManifestAndScheduleTest(FixtureMixin, unittest.TestCase):
    def test_checked_in_example_is_structurally_valid_and_schema_is_json(self) -> None:
        schema_path = SCRIPTS / "feature_validation" / "manifest.schema.json"
        example_path = SCRIPTS.parent / "examples" / "feature-performance-validation" / "manifest.example.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertIn(
            "provenance_sidecar",
            schema["$defs"]["executable"]["allOf"][0]["then"]["required"],
        )
        core.validate_manifest(json.loads(example_path.read_text(encoding="utf-8")))

    def test_valid_manifest_and_balanced_schedule(self) -> None:
        manifest = self.manifest()
        core.validate_manifest(manifest)
        stage = next(item for item in manifest["stages"] if item["purpose"] == "kernel_screen")
        schedule = core.build_schedule(manifest, stage)
        orientations = []
        for pair in range(1, 6):
            pair_runs = [item for item in schedule if item["pair"] == pair]
            orientations.append(pair_runs[0]["orientation"])
            self.assertEqual(len({item["variant"] for item in pair_runs}), 2)
        self.assertEqual(orientations, ["AB", "BA", "AB", "BA", "AB"])

    def test_single_pair_screening_is_explicit_and_limited_to_early_stages(self) -> None:
        manifest = self.manifest()
        stage = next(item for item in manifest["stages"] if item["purpose"] == "kernel_screen")
        del stage["decision_policy"]
        stage["screening_policy"] = {
            "kind": "single_pair_fail_fast",
            "order": "BA",
            "regression_threshold_percent": 2.0,
            "confidence_claim": "none",
        }

        core.validate_manifest(manifest)
        schedule = core.build_schedule(manifest, stage)

        self.assertEqual(len(schedule), 6)
        self.assertEqual({item["pair"] for item in schedule}, {1})
        self.assertEqual({item["orientation"] for item in schedule}, {"BA"})
        production = next(
            item for item in manifest["stages"] if item["purpose"] == "production_confirmation"
        )
        del production["decision_policy"]
        production["screening_policy"] = copy.deepcopy(stage["screening_policy"])
        with self.assertRaisesRegex(core.ManifestError, "limited to smoke and kernel_screen"):
            core.validate_manifest(manifest)

    def test_single_pair_screening_rejects_statistical_claims(self) -> None:
        manifest = self.manifest()
        stage = manifest["stages"][1]
        stage["screening_policy"] = {
            "kind": "single_pair_fail_fast",
            "order": "AB",
            "regression_threshold_percent": 2.0,
            "confidence_claim": "none",
        }
        with self.assertRaisesRegex(core.ManifestError, "statistical decision_policy"):
            core.validate_manifest(manifest)

    def test_adjacent_single_pair_screens_must_alternate_order(self) -> None:
        manifest = self.manifest()
        for stage in manifest["stages"][1:3]:
            del stage["decision_policy"]
            stage["screening_policy"] = {
                "kind": "single_pair_fail_fast",
                "order": "AB",
                "regression_threshold_percent": 2.0,
                "confidence_claim": "none",
            }
        with self.assertRaisesRegex(core.ManifestError, "must alternate AB/BA"):
            core.validate_manifest(manifest)

    def test_command_for_run_preserves_bench_specific_arity(self) -> None:
        manifest = self.manifest()
        bench = self.repo / "llama-bench"
        self.executable.rename(bench)
        for variant in manifest["variants"].values():
            variant["executables"]["default"]["path"] = str(bench)
        stage = manifest["stages"][1]
        stage["command"]["common_args"] = ["--no-kv-offload", "0"]
        stage["command"]["cli_schema"] = {"builtin": "llama", "allow_positionals": False}
        scheduled = core.build_schedule(manifest, stage)[0]

        argv, _ = core.command_for_run(manifest, stage, scheduled)

        self.assertEqual(argv[-2:], ["--no-kv-offload", "0"])

    def test_architecture_selector_is_rejected(self) -> None:
        manifest = self.manifest()
        manifest["stages"][1]["selection"]["architecture"] = "forbidden"
        with self.assertRaisesRegex(core.ManifestError, "forbidden"):
            core.validate_manifest(manifest)

    def test_kernel_screen_requires_low_mid_high(self) -> None:
        manifest = self.manifest()
        next(item for item in manifest["stages"] if item["purpose"] == "kernel_screen")["screens"].pop()
        with self.assertRaisesRegex(core.ManifestError, "low, mid, and high"):
            core.validate_manifest(manifest)

    def test_long_context_gate_must_be_final_and_mandatory(self) -> None:
        manifest = self.manifest()
        manifest["stages"][-1]["mandatory"] = False
        with self.assertRaisesRegex(core.ManifestError, "mandatory"):
            core.validate_manifest(manifest)

    def test_short_prefill_and_decode_smokes_are_both_required(self) -> None:
        manifest = self.manifest()
        manifest["stages"] = [stage for stage in manifest["stages"] if stage["id"] != "decode-smoke"]
        with self.assertRaisesRegex(core.ManifestError, "prefill and short decode"):
            core.validate_manifest(manifest)

    def test_performance_stage_requires_fail_closed_decision_policy(self) -> None:
        manifest = self.manifest()
        del manifest["stages"][1]["decision_policy"]
        with self.assertRaisesRegex(core.ManifestError, "decision_policy"):
            core.validate_manifest(manifest)

    def test_graph_replay_profiler_requires_node_trace(self) -> None:
        example_path = (
            SCRIPTS.parent
            / "examples"
            / "feature-performance-validation"
            / "manifest.example.json"
        )
        manifest = json.loads(example_path.read_text(encoding="utf-8"))
        manifest["profiler"]["cuda_graph_trace"] = {
            "applicability": "not_applicable",
            "reason": "invalid for this graph-replay stage",
        }
        stage = core.stage_by_id(manifest, manifest["profiler"]["stage"])
        stage["selection"]["cuda_graph_replay"] = {"applicability": "required"}
        with self.assertRaisesRegex(core.ManifestError, "requires graph-node tracing"):
            core.validate_manifest(manifest)

    def test_cuda_graph_capability_alone_does_not_claim_graph_replay(self) -> None:
        example_path = (
            SCRIPTS.parent
            / "examples"
            / "feature-performance-validation"
            / "manifest.example.json"
        )
        manifest = json.loads(example_path.read_text(encoding="utf-8"))
        manifest["profiler"]["cuda_graph_trace"] = {
            "applicability": "not_applicable",
            "reason": "production prefill launches kernels directly despite backend support",
        }
        manifest["profiler"]["selector"]["graph_only"] = False
        core.validate_manifest(manifest)

    def test_early_diagnostic_is_explicitly_regression_gated_and_paired(self) -> None:
        example_path = (
            SCRIPTS.parent
            / "examples"
            / "feature-performance-validation"
            / "manifest.example.json"
        )
        manifest = json.loads(example_path.read_text(encoding="utf-8"))
        diagnostic = manifest["early_diagnostics"][0]
        diagnostic["trigger"] = "always"
        with self.assertRaisesRegex(core.ManifestError, "regression_signal_only"):
            core.validate_manifest(manifest)
        diagnostic["trigger"] = "regression_signal_only"
        diagnostic["variants"] = ["candidate"]
        with self.assertRaisesRegex(core.ManifestError, "baseline then candidate"):
            core.validate_manifest(manifest)

    def test_graph_replay_early_diagnostic_requires_node_trace(self) -> None:
        example_path = (
            SCRIPTS.parent
            / "examples"
            / "feature-performance-validation"
            / "manifest.example.json"
        )
        manifest = json.loads(example_path.read_text(encoding="utf-8"))
        diagnostic = manifest["early_diagnostics"][0]
        diagnostic["stage"] = "short-graph-decode-smoke"
        diagnostic["screen_selection"] = {"kind": "fixed", "screen": "default"}
        with self.assertRaisesRegex(core.ManifestError, "requires graph-node tracing"):
            core.validate_manifest(manifest)

    def test_workload_setting_cannot_hide_in_candidate_delta(self) -> None:
        manifest = self.manifest()
        stage = next(item for item in manifest["stages"] if item["purpose"] == "kernel_screen")
        stage["command"]["candidate_args"] = ["-ub", "128"]
        stage["command"]["cli_schema"] = {
            "builtin": "none",
            "allow_positionals": False,
            "options": {"-ub": 1},
        }
        stage["command"]["controlled_delta"] = {
            "reason": "invalid test delta",
            "baseline_args": [],
            "candidate_args": ["-ub", "128"],
            "baseline_environment": {},
            "candidate_environment": {},
            "allowed_options": ["-ub"],
            "allowed_environment": [],
        }
        with self.assertRaisesRegex(core.ManifestError, "immutable"):
            core.validate_manifest(manifest)


class ProvenanceTest(FixtureMixin, unittest.TestCase):
    def test_clean_provenance_succeeds(self) -> None:
        manifest = self.manifest()
        provenance = core.capture_provenance(manifest, "manifest-hash")
        self.assertEqual(
            provenance["variants"]["baseline"]["executables"]["default"]["binary"]["sha256"],
            file_sha256(self.executable),
        )
        self.assertFalse(provenance["variants"]["baseline"]["source"]["status"])

    def test_wrong_binary_hash_fails_closed(self) -> None:
        manifest = self.manifest()
        manifest["variants"]["candidate"]["executables"]["default"][
            "expected_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(core.ProvenanceError, "binary SHA-256 mismatch"):
            core.capture_provenance(manifest, "manifest-hash")

    def test_registered_cmake_build_succeeds(self) -> None:
        manifest, _, _, sidecar = self.cmake_manifest()
        core.validate_manifest(manifest)
        provenance = core.capture_provenance(manifest, "manifest-hash")
        registered = provenance["variants"]["baseline"]["executables"]["default"][
            "build_provenance"
        ]
        self.assertEqual(registered["path"], str(sidecar.resolve()))
        sidecar_body = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertEqual(sidecar_body["source"]["root"], str(self.repo.resolve()))
        self.assertEqual(
            sidecar_body["build"]["home_directory"],
            str(self.repo.resolve()),
        )
        self.assertGreater(sidecar_body["source"]["source_files"]["count"], 0)
        files, _ = core.provenance_identity_spec(
            provenance, [("baseline", "default")]
        )
        self.assertIn(str(sidecar.resolve()), {item["path"] for item in files})

    def test_registered_cmake_build_with_direct_harness_succeeds(self) -> None:
        manifest, _, _, _ = self.cmake_manifest()
        harness = {
            "kind": "native_executable",
            "native_llama_affinity": True,
            "source_files": [
                {
                    "path": str(self.executable.resolve()),
                    "sha256": file_sha256(self.executable),
                }
            ],
        }
        for variant in manifest["variants"].values():
            variant["executables"]["default"]["direct_harness"] = copy.deepcopy(
                harness
            )
        provenance = core.capture_provenance(manifest, "manifest-hash")
        observed = provenance["variants"]["baseline"]["executables"]["default"]
        self.assertEqual(
            observed["direct_harness"]["source_files"][0]["sha256"],
            file_sha256(self.executable),
        )

    def test_register_build_command_writes_manifest_ready_sidecar(self) -> None:
        _, built_executable, cache, sidecar = self.cmake_manifest()
        sidecar.unlink()
        result = CLI._register_build(
            self.repo,
            built_executable,
            cache,
            None,
            force=False,
        )
        self.assertEqual(result["status"], "registered")
        self.assertEqual(result["sidecar"]["path"], str(sidecar.resolve()))
        self.assertEqual(result["sidecar"]["sha256"], file_sha256(sidecar))
        with self.assertRaisesRegex(core.ValidationError, "already exists"):
            CLI._register_build(
                self.repo,
                built_executable,
                cache,
                None,
                force=False,
            )

    def test_register_build_rejects_nonignored_sidecar_inside_source(self) -> None:
        _, built_executable, cache, _ = self.cmake_manifest()
        with self.assertRaisesRegex(core.ValidationError, "must be Git-ignored"):
            CLI._register_build(
                self.repo,
                built_executable,
                cache,
                self.repo / "tracked-sidecar.json",
                force=False,
            )

    def test_cmake_manifest_without_sidecar_fails_closed(self) -> None:
        manifest, _, _, _ = self.cmake_manifest()
        del manifest["variants"]["candidate"]["executables"]["default"][
            "provenance_sidecar"
        ]
        with self.assertRaisesRegex(core.ManifestError, "requires provenance_sidecar"):
            core.validate_manifest(manifest)

    def test_missing_sidecar_file_fails_closed(self) -> None:
        manifest, _, _, sidecar = self.cmake_manifest()
        sidecar.unlink()
        with self.assertRaisesRegex(core.ProvenanceError, "sidecar is missing"):
            core.capture_provenance(manifest, "manifest-hash")

    def test_nested_source_root_fails_closed(self) -> None:
        nested = self.repo / "ignored archive"
        nested.mkdir()
        with self.assertRaisesRegex(core.ProvenanceError, "exact Git worktree root"):
            core.git_snapshot(nested)

    def test_cmake_home_directory_mismatch_fails_closed(self) -> None:
        _, built_executable, cache, _ = self.cmake_manifest()
        cache.write_text(
            cache.read_text(encoding="utf-8").replace(
                str(self.repo), str(self.root / "different source")
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(core.ProvenanceError, "CMAKE_HOME_DIRECTORY"):
            core.capture_cmake_build_provenance(self.repo, built_executable, cache)

    def test_source_change_after_registration_fails_sidecar(self) -> None:
        manifest, _, _, _ = self.cmake_manifest()
        self.executable.write_text(
            self.executable.read_text(encoding="utf-8").replace("100.0", "101.0"),
            encoding="utf-8",
        )
        dirty = core.git_snapshot(self.repo)["dirty_fingerprint"]
        for variant in manifest["variants"].values():
            variant["tree_policy"] = "expected_dirty"
            variant["expected_dirty_fingerprint"] = dirty
        with self.assertRaisesRegex(core.ProvenanceError, "provenance source mismatch"):
            core.capture_provenance(manifest, "manifest-hash")

    def test_copied_binary_and_sidecar_require_reregistration(self) -> None:
        manifest, built_executable, _, sidecar = self.cmake_manifest()
        copied = self.root / "copied" / "fake bench"
        copied.parent.mkdir()
        shutil.copy2(built_executable, copied)
        copied_sidecar = pathlib.Path(f"{copied}.build-provenance.json")
        shutil.copy2(sidecar, copied_sidecar)
        for variant in manifest["variants"].values():
            executable = variant["executables"]["default"]
            executable["path"] = str(copied)
            executable["expected_sha256"] = file_sha256(copied)
            executable["provenance_sidecar"] = {
                "path": str(copied_sidecar),
                "sha256": file_sha256(copied_sidecar),
            }
        with self.assertRaisesRegex(core.ProvenanceError, "provenance executable mismatch"):
            core.capture_provenance(manifest, "manifest-hash")

    def test_wrong_sidecar_hash_fails_closed(self) -> None:
        manifest, _, _, _ = self.cmake_manifest()
        manifest["variants"]["candidate"]["executables"]["default"][
            "provenance_sidecar"
        ]["sha256"] = "0" * 64
        with self.assertRaisesRegex(core.ProvenanceError, "sidecar SHA-256 mismatch"):
            core.capture_provenance(manifest, "manifest-hash")

    def test_sidecar_internal_fingerprint_fails_closed(self) -> None:
        manifest, _, _, sidecar = self.cmake_manifest()
        body = json.loads(sidecar.read_text(encoding="utf-8"))
        body["source"]["head"] = "0" * 40
        core.write_json_atomic(sidecar, body)
        for variant in manifest["variants"].values():
            variant["executables"]["default"]["provenance_sidecar"][
                "sha256"
            ] = file_sha256(sidecar)
        with self.assertRaisesRegex(core.ProvenanceError, "fingerprint is invalid"):
            core.capture_provenance(manifest, "manifest-hash")

    def test_dirty_tree_fails_closed(self) -> None:
        manifest = self.manifest()
        (self.repo / "untracked").write_text("dirt", encoding="utf-8")
        with self.assertRaisesRegex(core.ProvenanceError, "dirty"):
            core.capture_provenance(manifest, "manifest-hash")

    def test_fresh_runs_use_stat_guards_without_rehashing_large_identity_files(self) -> None:
        provenance = core.capture_provenance(self.manifest(), "manifest-hash")
        files, _ = core.provenance_identity_spec(
            provenance, [("baseline", "default")]
        )
        self.assertTrue(files)
        self.assertTrue(all(item["rehash"] is False for item in files))
        with mock.patch.object(
            core,
            "sha256_file",
            side_effect=AssertionError("per-run rehash should not occur"),
        ):
            _verify_spec_identity({"identity_files": files, "source_identities": []})

    def test_stat_cache_rejects_same_size_change_with_restored_mtime(self) -> None:
        provenance = core.capture_provenance(self.manifest(), "manifest-hash")
        files, _ = core.provenance_identity_spec(
            provenance, [("baseline", "default")]
        )
        original = self.executable.stat()
        changed = self.executable.read_text(encoding="utf-8").replace("100.0", "999.9")
        self.executable.write_text(changed, encoding="utf-8")
        self.executable.chmod(0o755)
        __import__("os").utime(
            self.executable,
            ns=(original.st_atime_ns, original.st_mtime_ns),
        )
        with self.assertRaisesRegex(core.ProvenanceError, "ctime_ns"):
            _verify_spec_identity({"identity_files": files, "source_identities": []})


class StatisticsTest(unittest.TestCase):
    def test_paired_log_report_keeps_all_raw_samples(self) -> None:
        pairs = [
            {"pair": 1, "orientation": "AB", "baseline": 100.0, "candidate": 102.0},
            {"pair": 2, "orientation": "BA", "baseline": 101.0, "candidate": 103.0},
            {"pair": 3, "orientation": "AB", "baseline": 99.0, "candidate": 101.0},
        ]
        report = core.paired_log_report(
            pairs,
            direction="higher",
            thresholds={
                "improvement_percent": 1.0,
                "regression_percent": 1.0,
                "equivalence_percent": 0.5,
            },
        )
        self.assertEqual(report["pair_count"], 3)
        self.assertEqual(len(report["raw_pairs"]), 3)
        self.assertGreater(report["percent_change"], 1.9)
        self.assertEqual(report["decision"], "improvement")

    def test_inconclusive_result_is_explicit(self) -> None:
        pairs = [
            {"pair": 1, "orientation": "AB", "baseline": 100.0, "candidate": 98.0},
            {"pair": 2, "orientation": "BA", "baseline": 100.0, "candidate": 102.0},
            {"pair": 3, "orientation": "AB", "baseline": 100.0, "candidate": 100.0},
        ]
        report = core.paired_log_report(
            pairs,
            direction="higher",
            thresholds={
                "improvement_percent": 1.0,
                "regression_percent": 1.0,
                "equivalence_percent": 0.5,
            },
        )
        self.assertEqual(report["decision"], "inconclusive")

    def test_single_pair_screen_has_no_confidence_claim(self) -> None:
        report = core.single_pair_screen_report(
            {"pair": 1, "orientation": "BA", "baseline": 100.0, "candidate": 97.0},
            direction="higher",
            regression_threshold_percent=2.0,
        )

        self.assertEqual(report["signal"], "regression_signal")
        self.assertIsNone(report["confidence_interval"])
        self.assertEqual(report["confidence_claim"], "none")
        self.assertEqual(len(report["raw_pairs"]), 1)
        clear = core.single_pair_screen_report(
            {"pair": 1, "orientation": "AB", "baseline": 100.0, "candidate": 99.0},
            direction="higher",
            regression_threshold_percent=2.0,
        )
        self.assertEqual(clear["signal"], "clear_to_continue")

    def test_weighted_harmonic_matches_prefill_throughput_geometry(self) -> None:
        value = core.aggregate_screen_values([(100.0, 1.0), (200.0, 3.0)], "weighted_harmonic")
        self.assertAlmostEqual(value, 160.0)

    def test_metric_parser_fails_when_no_sample_exists(self) -> None:
        with self.assertRaisesRegex(core.ValidationError, "did not match"):
            core.extract_metric({"regex": r"metric=([0-9.]+)"}, "nothing", "")

    def test_gate_outcome_distinguishes_completed_from_acceptable(self) -> None:
        policy = {
            "acceptable_decisions": ["improvement", "equivalent"],
            "regression": "fail",
            "inconclusive_after_maximum": "unresolved_fail",
        }
        self.assertEqual(core.performance_outcome("improvement", policy), "passed")
        self.assertEqual(core.performance_outcome("regression", policy), "failed")
        self.assertEqual(core.performance_outcome("inconclusive", policy), "unresolved")


class ProfilerTest(unittest.TestCase):
    def test_regression_diagnostic_selects_preregistered_largest_screen(self) -> None:
        stage = {
            "id": "spans",
            "metric": {"direction": "higher"},
            "screens": [{"id": "low"}, {"id": "mid"}, {"id": "high"}],
        }
        report = {
            "screening_observation": {
                "signal": "regression_signal",
                "direction": "higher",
                "raw_pairs": [
                    {
                        "raw_screens": {
                            "baseline": [
                                {"screen": "low", "value": 100},
                                {"screen": "mid", "value": 100},
                                {"screen": "high", "value": 100},
                            ],
                            "candidate": [
                                {"screen": "low", "value": 99},
                                {"screen": "mid", "value": 91},
                                {"screen": "high", "value": 96},
                            ],
                        }
                    }
                ],
            }
        }

        selected = profiler.select_regression_diagnostic(
            stage, report, {"kind": "largest_observed_regression"}
        )

        self.assertTrue(selected["triggered"])
        self.assertEqual(selected["selected_screen"]["screen"], "mid")
        self.assertEqual(len(selected["all_screen_observations"]), 3)
        report["screening_observation"]["signal"] = "clear_to_continue"
        skipped = profiler.select_regression_diagnostic(
            stage, report, {"kind": "largest_observed_regression"}
        )
        self.assertEqual(skipped, {"triggered": False, "reason": "no_regression_signal"})

    def test_nsys_explicitly_requests_and_verifies_graph_node_trace(self) -> None:
        config = {
            "applicability": "required",
            "granularity": "node",
            "launch_origin": "host-only",
        }
        evidence = profiler.verify_nsys_graph_trace_capability(
            config,
            help_text="--cuda-graph-trace graph|node",
            version_text="Nsight Systems 2026.1",
        )
        self.assertEqual(evidence["status"], "supported")
        argv = profiler.nsys_discovery_argv(
            ["/tmp/llama-bench", "-C", "0x7", "--cpu-strict", "1"],
            pathlib.Path("/tmp/discovery"),
            cuda_graph_trace=config,
        )
        self.assertIn("--cuda-graph-trace=node:host-only", argv)

    def test_nsys_unsupported_or_missing_graph_nodes_fails_closed(self) -> None:
        config = {
            "applicability": "required",
            "granularity": "node",
            "launch_origin": "host-only",
        }
        with self.assertRaisesRegex(core.ValidationError, "does not advertise"):
            profiler.verify_nsys_graph_trace_capability(
                config,
                help_text="old help without graph node tracing",
                version_text="Nsight Systems old",
            )
        with self.assertRaisesRegex(core.ValidationError, "contains no nonzero"):
            profiler.verify_graph_node_discovery(
                {"graph_node_trace": {"launches_with_graph_node": 0}}, config
            )

    def test_jsonl_discovery_and_one_filtered_ncu_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            discovery_path = root / "fake.jsonl"
            discovery_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "name": "kernel<q8>",
                                "start_ns": 10,
                                "end_ns": 20,
                                "grid": [1, 2, 3],
                                "block": [32, 1, 1],
                                "graph_node_id": 7,
                            }
                        ),
                        json.dumps({"name": "other", "start_ns": 30, "end_ns": 40}),
                        json.dumps(
                            {
                                "name": "kernel<q8>",
                                "start_ns": 50,
                                "end_ns": 60,
                                "grid": [1, 2, 3],
                                "block": [32, 1, 1],
                                "graph_node_id": 8,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            discovery = profiler.parse_discovery(discovery_path)
            plan = profiler.build_ncu_plan(
                discovery,
                {"kernel_regex": r"kernel<q8>", "graph_only": True},
                ["/tmp/llama-bench", "-C", "0x7", "--cpu-strict", "1"],
                root / "ncu",
            )
            self.assertEqual(discovery["launch_count"], 3)
            self.assertEqual(plan["selected_count"], 2)
            self.assertEqual(plan["command"]["derivation"]["launch_skip"], 0)
            self.assertEqual(plan["command"]["derivation"]["launch_count"], 2)
            self.assertIn("--graph-profiling", plan["command"]["direct_argv"])
            self.assertIn("flock /tmp/beellama-single-gpu.lock -c", plan["command"]["flock_command"])

            csv_text = (
                '==PROF== fake\n'
                '"ID","Kernel Name","Grid Size","Block Size","Metric Name","Metric Value"\n'
                '"1","kernel<q8>","1 2 3","32 1 1","gpu__time_duration.sum","10"\n'
                '"2","kernel<q8>","1 2 3","32 1 1","gpu__time_duration.sum","11"\n'
            )
            parsed = profiler.parse_ncu_raw_csv(csv_text)
            verified = profiler.verify_ncu_capture(plan, parsed)
            self.assertEqual(verified["status"], "verified")
            self.assertEqual(verified["observed_capture_count"], 2)
            self.assertEqual(
                parsed["captures"][0]["metrics"],
                [{"name": "gpu__time_duration.sum", "unit": None, "value": "10"}],
            )

    def test_ncu_report_count_mismatch_and_missing_columns_do_not_verify(self) -> None:
        plan = {
            "command": {
                "derivation": {
                    "kernel_name": "kernel",
                    "launch_count": 2,
                    "expected_shapes": [{"grid": [1, 1, 1], "block": [32, 1, 1]}],
                }
            }
        }
        one = profiler.parse_ncu_raw_csv(
            '"ID","Kernel Name","Grid Size","Block Size"\n'
            '"1","kernel","1 1 1","32 1 1"\n'
        )
        self.assertEqual(profiler.verify_ncu_capture(plan, one)["status"], "failed")
        missing_shapes = profiler.parse_ncu_raw_csv(
            '"ID","Kernel Name"\n"1","kernel"\n"2","kernel"\n'
        )
        self.assertEqual(
            profiler.verify_ncu_capture(plan, missing_shapes)["status"], "unverifiable"
        )

    def test_agent_summary_surfaces_discovery_ncu_metrics_timing_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            core.write_json_atomic(
                root / "ncu-plan.json",
                {
                    "stage": "screen",
                    "variant": "candidate",
                    "screen": "mid",
                    "command": {
                        "selected_launches": [
                            {
                                "launch_index": 12,
                                "name_occurrence": 3,
                                "graph_node_id": 9,
                            }
                        ],
                        "derivation": {
                            "kernel_name": "kernel",
                            "matching_launch_count": 4,
                            "launch_count": 1,
                            "discovery_launch_indices": [12],
                            "name_occurrences": [3],
                            "expected_shapes": [
                                {"grid": [2, 1, 1], "block": [32, 1, 1]}
                            ],
                        },
                        "report_path": str(root / "capture.ncu-rep"),
                    },
                },
            )
            core.write_json_atomic(
                root / "discovery-inventory.json",
                {
                    "launch_count": 20,
                    "graph_node_trace": {
                        "launches_with_graph_node": 5,
                        "distinct_graph_node_ids": ["9"],
                    },
                },
            )
            core.write_json_atomic(
                root / "profile-state.json", {"ncu_verification_status": "verified"}
            )
            core.write_json_atomic(
                root / "ncu-verification.json",
                {
                    "status": "verified",
                    "issues": [],
                    "observed_capture_count": 1,
                    "parsed_export": {
                        "captures": [
                            {
                                "kernel_name": "kernel",
                                "metrics": [
                                    {
                                        "name": "gpu__time_duration.sum",
                                        "unit": "nsecond",
                                        "value": "1234",
                                    }
                                ],
                            }
                        ]
                    },
                },
            )
            core.write_json_atomic(
                root / "nsys-result.json",
                {"timing": {"target_process_seconds": 2.5}},
            )
            core.write_json_atomic(
                root / "ncu-result.json",
                {"timing": {"target_process_seconds": 3.5}},
            )

            summary = profiler.agent_profile_summary(root)

        self.assertEqual(summary["status"], "verified")
        self.assertEqual(summary["nsys"]["selected_kernel"], "kernel")
        self.assertEqual(summary["nsys"]["selected_graph_node_ids"], ["9"])
        self.assertEqual(
            summary["ncu"]["captures"][0]["metrics"][0]["value"], "1234"
        )
        self.assertEqual(summary["timing_seconds"]["nsys"]["target_process_seconds"], 2.5)
        self.assertIn("diagnostic-only", summary["evidence_boundary"].lower())

    def test_agent_comparison_keeps_raw_metrics_without_acceptance_claim(self) -> None:
        def profile(variant: str, value: str) -> dict[str, object]:
            return {
                "variant": variant,
                "nsys": {
                    "selected_kernel": "kernel",
                    "selected_shapes": [{"grid": [1, 1, 1], "block": [32, 1, 1]}],
                    "total_launch_count": 10,
                    "matching_launch_count": 2,
                    "selected_launch_count": 1,
                },
                "ncu": {
                    "captures": [
                        {
                            "id": "1",
                            "kernel_name": "kernel",
                            "metrics": [{"name": "duration", "value": value}],
                        }
                    ]
                },
                "timing_seconds": {},
            }

        comparison = profiler.compare_agent_profiles(
            profile("baseline", "10"), profile("candidate", "12")
        )

        self.assertEqual(comparison["status"], "diagnostic_side_by_side_only")
        self.assertEqual(
            comparison["raw_ncu_metrics"]["candidate"][0]["metrics"][0]["value"],
            "12",
        )
        self.assertIn("no paired interval", comparison["comparison_boundary"].lower())

    def test_nsys_ncu_template_cast_spelling_normalizes_for_verification(self) -> None:
        plan = {
            "command": {
                "derivation": {
                    "kernel_name": "kernel<(int)256, (bool)0, (ggml_type)1>",
                    "launch_count": 1,
                    "expected_shapes": [{"grid": [1, 1, 1], "block": [32, 1, 1]}],
                }
            }
        }
        parsed = profiler.parse_ncu_raw_csv(
            '"ID","Kernel Name","Grid Size","Block Size"\n'
            '"1","kernel<256, 0, 1>","1 1 1","32 1 1"\n'
        )

        verified = profiler.verify_ncu_capture(plan, parsed)

        self.assertEqual(verified["status"], "verified")
        self.assertNotEqual(verified["expected_kernel"], verified["observed_kernel_names"][0])
        self.assertEqual(
            verified["canonical_expected_kernel"], verified["canonical_observed_kernel_names"][0]
        )

    def test_fake_nsys_sqlite_parses_names_shapes_counts_and_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "fake.sqlite"
            import sqlite3

            with contextlib.closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
                connection.execute("INSERT INTO StringIds VALUES (1, 'graph_kernel')")
                connection.execute(
                    "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
                    "start INTEGER, end INTEGER, demangledName INTEGER, graphNodeId INTEGER, "
                    "gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, blockY INTEGER, blockZ INTEGER)"
                )
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10, 20, 1, 9, 3, 2, 1, 32, 1, 1)"
                )
                connection.commit()
            parsed = profiler.parse_discovery(path)
            self.assertEqual(parsed["launches"][0]["name"], "graph_kernel")
            self.assertEqual(parsed["launches"][0]["grid"], [3, 2, 1])
            self.assertEqual(parsed["groups"][0]["count"], 1)
            self.assertEqual(parsed["groups"][0]["launch_indices"], [1])

    def test_profiler_requires_native_affinity(self) -> None:
        with self.assertRaisesRegex(core.ValidationError, "native"):
            profiler.validate_native_affinity(["/tmp/llama-bench", "--progress"])

    def test_non_bench_profiler_target_requires_native_ranges(self) -> None:
        with self.assertRaisesRegex(core.ValidationError, "native"):
            profiler.validate_native_affinity(
                ["/tmp/llama-cli", "-C", "0x7", "--cpu-strict", "1"]
            )

    def test_noncontiguous_selected_occurrences_fail_closed(self) -> None:
        discovery = {
            "source": "fake",
            "launches": [
                {"name": "kernel", "name_occurrence": 1, "launch_index": 1, "graph_node_id": 1},
                {"name": "kernel", "name_occurrence": 2, "launch_index": 2, "graph_node_id": None},
                {"name": "kernel", "name_occurrence": 3, "launch_index": 3, "graph_node_id": 3},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            core.ValidationError, "not contiguous"
        ):
            profiler.build_ncu_plan(
                discovery,
                {"kernel_regex": "kernel", "graph_only": True},
                ["/tmp/llama-bench", "-C", "0x7", "--cpu-strict", "1"],
                pathlib.Path(temporary),
            )

    def test_semantic_last_occurrence_is_derived_from_discovery(self) -> None:
        discovery = {
            "source": "fake",
            "launches": [
                {
                    "name": "kernel",
                    "name_occurrence": occurrence,
                    "launch_index": occurrence * 10,
                    "graph_node_id": None,
                    "grid": [1, 1, 1],
                    "block": [32, 1, 1],
                }
                for occurrence in range(1, 4)
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            plan = profiler.build_ncu_plan(
                discovery,
                {
                    "kernel_regex": "kernel",
                    "graph_only": False,
                    "occurrence_policy": {"kind": "last"},
                },
                ["/tmp/llama-bench", "-C", "0x7", "--cpu-strict", "1"],
                pathlib.Path(temporary),
            )
        derivation = plan["command"]["derivation"]
        self.assertEqual(derivation["matching_launch_count"], 3)
        self.assertEqual(derivation["occurrence_policy"], "last")
        self.assertEqual(derivation["launch_skip"], 2)
        self.assertEqual(derivation["launch_count"], 1)
        self.assertEqual(derivation["discovery_launch_indices"], [30])


class DiagnosticCliTest(FixtureMixin, unittest.TestCase):
    def _manifest_with_diagnostic(self, signal: str) -> pathlib.Path:
        manifest = self.manifest()
        stage = next(item for item in manifest["stages"] if item["purpose"] == "kernel_screen")
        stage["resource"] = "gpu"
        del stage["decision_policy"]
        stage["screening_policy"] = {
            "kind": "single_pair_fail_fast",
            "order": "BA",
            "regression_threshold_percent": 2.0,
            "confidence_claim": "none",
        }
        manifest["early_diagnostics"] = [
            {
                "id": "kernel-regression",
                "trigger": "regression_signal_only",
                "on_clear": "skip",
                "evidence_claim": "diagnostic_only",
                "pattern": "independent_nsys_then_filtered_ncu_per_variant",
                "profile_repetitions": 1,
                "stage": stage["id"],
                "variants": ["baseline", "candidate"],
                "screen_selection": {"kind": "largest_observed_regression"},
                "selector": {
                    "kernel_regex": "kernel",
                    "graph_only": False,
                    "occurrence_policy": {"kind": "last"},
                },
                "cuda_graph_trace": {
                    "applicability": "not_applicable",
                    "reason": "direct fake prefill",
                },
                "metrics": ["gpu__time_duration.sum"],
            }
        ]
        manifest_path = self.root / "diagnostic.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_sha = file_sha256(manifest_path)
        provenance = core.capture_provenance(manifest, manifest_sha)
        study_root = pathlib.Path(manifest["artifact_root"]) / (
            f"{manifest['study']['id']}-{manifest_sha[:12]}"
        )
        study_root.mkdir(parents=True)
        candidate = {"low": 99.0, "mid": 91.0, "high": 96.0}
        stage_report = {
            "purpose": "kernel_screen",
            "execution_status": "completed",
            "status": "failed" if signal == "regression_signal" else "passed",
            "screening_observation": {
                "signal": signal,
                "direction": "higher",
                "raw_pairs": [
                    {
                        "raw_screens": {
                            "baseline": [
                                {"screen": name, "value": 100.0, "weight": 1}
                                for name in ("low", "mid", "high")
                            ],
                            "candidate": [
                                {"screen": name, "value": candidate[name], "weight": 1}
                                for name in ("low", "mid", "high")
                            ],
                        }
                    }
                ],
            },
        }
        core.write_json_atomic(
            study_root / "state.json",
            {
                "schema_version": 1,
                "manifest_sha256": manifest_sha,
                "provenance_identity_fingerprint": provenance["identity_fingerprint"],
                "runs": {},
                "stages": {stage["id"]: stage_report},
            },
        )
        return manifest_path

    def test_diagnostic_runs_independent_profiles_for_both_variants(self) -> None:
        manifest_path = self._manifest_with_diagnostic("regression_signal")

        def compact(path: pathlib.Path) -> dict[str, object]:
            variant = "baseline" if "-baseline-" in path.name else "candidate"
            return {
                "status": "verified",
                "variant": variant,
                "nsys": {
                    "selected_kernel": "kernel",
                    "selected_shapes": [],
                    "total_launch_count": 1,
                    "matching_launch_count": 1,
                    "selected_launch_count": 1,
                },
                "ncu": {"captures": []},
                "timing_seconds": {},
                "artifacts": {"root": str(path)},
                "evidence_boundary": "diagnostic only",
            }

        with mock.patch.object(CLI, "_profile_one") as profile_one, mock.patch.object(
            CLI.profiler, "agent_profile_summary", side_effect=compact
        ):
            report = CLI._diagnose(manifest_path, resume=False)

        self.assertEqual(report["status"], "verified")
        self.assertEqual(profile_one.call_count, 2)
        self.assertEqual(
            [call.kwargs["variant_name"] for call in profile_one.call_args_list],
            ["baseline", "candidate"],
        )
        self.assertTrue(all(call.kwargs["execute_ncu"] for call in profile_one.call_args_list))
        trigger = report["investigations"][0]["trigger"]
        self.assertEqual(trigger["selected_screen"]["screen"], "mid")
        roots = [item["artifacts"]["root"] for item in report["profiles"]]
        self.assertNotEqual(roots[0], roots[1])
        self.assertEqual(
            report["investigations"][0]["agent_comparison"]["status"],
            "diagnostic_side_by_side_only",
        )

    def test_clear_screen_skips_all_profiler_work(self) -> None:
        manifest_path = self._manifest_with_diagnostic("clear_to_continue")
        with mock.patch.object(CLI, "_profile_one") as profile_one:
            report = CLI._diagnose(manifest_path, resume=False)
        self.assertEqual(report["status"], "not_triggered")
        profile_one.assert_not_called()

    def test_unverifiable_profile_fails_closed_and_is_preserved(self) -> None:
        manifest_path = self._manifest_with_diagnostic("regression_signal")
        with mock.patch.object(CLI, "_profile_one"), mock.patch.object(
            CLI.profiler,
            "agent_profile_summary",
            return_value={"status": "unverifiable", "variant": "baseline"},
        ), self.assertRaisesRegex(core.ValidationError, "unverifiable"):
            CLI._diagnose(manifest_path, resume=False)
        manifest_sha = file_sha256(manifest_path)
        report_path = (
            self.root
            / "artifacts"
            / f"unit-fixture-{manifest_sha[:12]}"
            / "early-diagnostics"
            / "agent-diagnostic-report.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["investigations"][0]["status"], "failed")

    def test_run_flag_does_not_invoke_diagnostics_after_a_pass(self) -> None:
        manifest_path = self.root / "passing.json"
        manifest_path.write_text(json.dumps(self.manifest()), encoding="utf-8")
        summary = {"status": "early_screen_only_not_production_or_final_acceptance"}
        with mock.patch.object(StudyRunner, "run", return_value=summary), mock.patch.object(
            CLI, "_diagnose"
        ) as diagnose, mock.patch.object(
            sys,
            "argv",
            [
                "feature-performance-validation.py",
                "run",
                str(manifest_path),
                "--diagnose-regressions",
            ],
        ), contextlib.redirect_stdout(io.StringIO()):
            returncode = CLI.main()
        self.assertEqual(returncode, 0)
        diagnose.assert_not_called()

    def test_run_flag_profiles_a_regression_but_preserves_nonzero_status(self) -> None:
        manifest_path = self.root / "regression.json"
        manifest_path.write_text(json.dumps(self.manifest()), encoding="utf-8")
        diagnostic = {"status": "verified", "evidence_boundary": "diagnostic only"}
        with mock.patch.object(
            StudyRunner, "run", side_effect=core.ValidationError("screen failed")
        ), mock.patch.object(
            CLI, "_diagnose", return_value=diagnostic
        ) as diagnose, mock.patch.object(
            sys,
            "argv",
            [
                "feature-performance-validation.py",
                "run",
                str(manifest_path),
                "--diagnose-regressions",
            ],
        ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            returncode = CLI.main()
        self.assertEqual(returncode, 2)
        diagnose.assert_called_once_with(manifest_path, resume=False)


class TelemetryTest(unittest.TestCase):
    def test_persistent_sampler_parses_gpu_state_and_is_reaped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fake = root / "fake-nvidia-smi"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import sys, time\n"
                "if '--version' in sys.argv:\n"
                "    print('fake nvidia-smi 1.0')\n"
                "    raise SystemExit\n"
                "doc = '''<nvidia_smi_log><driver_version>test-driver</driver_version>"
                "<cuda_version>test-cuda</cuda_version><gpu><product_name>fake-gpu</product_name>"
                "<uuid>GPU-FAKE</uuid><fb_memory_usage><total>100 MiB</total><used>0 MiB</used>"
                "</fb_memory_usage><temperature><gpu_temp>42 C</gpu_temp></temperature>"
                "<gpu_power_readings><power_draw>50 W</power_draw><current_power_limit>100 W</current_power_limit>"
                "</gpu_power_readings><clocks><graphics_clock>1000 MHz</graphics_clock>"
                "<sm_clock>900 MHz</sm_clock><mem_clock>800 MHz</mem_clock></clocks>"
                "<processes><process_info><pid>42</pid><type>G</type>"
                "<process_name>display</process_name><used_memory>4 MiB</used_memory>"
                "</process_info></processes></gpu></nvidia_smi_log>'''\n"
                "while True:\n"
                "    print(doc, flush=True)\n"
                "    time.sleep(0.1)\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            target = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(.35)"])
            session = telemetry.TelemetrySession(
                "gpu",
                nvidia_smi=str(fake),
                proc_period_seconds=0.05,
                first_sample_timeout=2.0,
            )
            try:
                session.start()
                sampler_pid = session.sampler_pid
                session.attach(target.pid)
                target.wait(timeout=2.0)
                report = session.stop()
            finally:
                if target.poll() is None:
                    target.terminate()
                    target.wait()
            self.assertTrue(report["sampler"]["reaped"])
            self.assertEqual(report["sampler"]["invocation_count"], 1)
            self.assertEqual(report["gpu_samples"][0]["gpus"][0]["temperature_c"], 42.0)
            self.assertTrue(report["clean_process_evidence"]["clean_throughout"])
            self.assertTrue(report["clean_process_evidence"]["ambient_noncompute_process_observations"])
            self.assertFalse(report["pinned_host_memory"]["directly_measured"])
            self.assertIn("never inferred", report["pinned_host_memory"]["limitation"])
            assert sampler_pid is not None
            with self.assertRaises(ProcessLookupError):
                __import__("os").kill(sampler_pid, 0)

    def test_contaminated_preflight_fails_and_reaps_sampler(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fake = pathlib.Path(temporary) / "fake-nvidia-smi"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import time\n"
                "doc = '''<nvidia_smi_log><gpu><uuid>GPU-FAKE</uuid><processes><process_info>"
                "<pid>123</pid><process_name>foreign</process_name><used_memory>5 MiB</used_memory>"
                "</process_info></processes></gpu></nvidia_smi_log>'''\n"
                "while True:\n"
                "    print(doc, flush=True)\n"
                "    time.sleep(.1)\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            session = telemetry.TelemetrySession("gpu", nvidia_smi=str(fake), first_sample_timeout=2.0)
            with self.assertRaisesRegex(core.ValidationError, "preflight"):
                session.start()
            self.assertTrue(session.report()["sampler"]["reaped"])
            assert session.sampler_pid is not None
            with self.assertRaises(ProcessLookupError):
                __import__("os").kill(session.sampler_pid, 0)


class RunnerTest(FixtureMixin, unittest.TestCase):
    def test_single_pair_screen_runs_once_and_fails_on_preregistered_signal(self) -> None:
        manifest = self.manifest()
        manifest_path = self.root / "single-pair-screen.json"
        stage = next(item for item in manifest["stages"] if item["purpose"] == "kernel_screen")
        del stage["decision_policy"]
        stage["screening_policy"] = {
            "kind": "single_pair_fail_fast",
            "order": "BA",
            "regression_threshold_percent": 2.0,
            "confidence_claim": "none",
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        runner = StudyRunner(manifest_path, SCRIPTS / "feature-performance-validation.py")

        def fake_execute(_stage: dict, scheduled: dict, **_kwargs: object) -> dict:
            return {
                "run_id": scheduled["run_id"],
                "metric_value": 97.0 if scheduled["variant"] == "candidate" else 100.0,
            }

        with mock.patch.object(runner, "_execute", side_effect=fake_execute) as execute:
            report = runner._run_performance(stage, retry_failed=False)

        self.assertEqual(execute.call_count, 6)
        self.assertEqual(report["status"], "failed")
        observation = report["screening_observation"]
        self.assertEqual(observation["signal"], "regression_signal")
        self.assertIsNone(observation["confidence_interval"])
        self.assertFalse(report["adaptive_repetition"]["applicable"])

    def test_final_gate_execution_is_not_acceptance_without_a_pass(self) -> None:
        manifest_path = self.root / "gate-status.json"
        manifest_path.write_text(json.dumps(self.manifest()), encoding="utf-8")
        for gate_status, expected in (
            ("failed", "acceptance_failed"),
            ("unresolved", "acceptance_unresolved"),
        ):
            runner = StudyRunner(manifest_path, SCRIPTS / "feature-performance-validation.py")
            runner.state = {
                "runs": {},
                "stages": {
                    "long": {
                        "purpose": "long_context_acceptance",
                        "execution_status": "completed",
                        "status": gate_status,
                    }
                }
            }
            summary = runner._summary("acceptance", None)
            self.assertEqual(summary["status"], expected)
            self.assertTrue(summary["final_long_context_acceptance_executed"])
            self.assertFalse(summary["final_long_context_acceptance_passed"])

    def test_final_gate_regression_and_inconclusive_statistics_fail_closed(self) -> None:
        manifest = self.manifest()
        manifest_path = self.root / "statistical-gate.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        stage = manifest["stages"][-1]
        scenarios = {
            "failed": [90, 90, 90, 90, 90],
            "unresolved": [98, 102, 100, 99, 101],
        }
        for expected_status, candidates in scenarios.items():
            runner = StudyRunner(manifest_path, SCRIPTS / "feature-performance-validation.py")

            def fake_execute(_stage: dict, scheduled: dict, **_kwargs: object) -> dict:
                value = (
                    candidates[scheduled["pair"] - 1]
                    if scheduled["variant"] == "candidate"
                    else 100
                )
                return {"run_id": scheduled["run_id"], "metric_value": value}

            with mock.patch.object(runner, "_execute", side_effect=fake_execute):
                report = runner._run_performance(stage, retry_failed=False)
            self.assertEqual(report["execution_status"], "completed")
            self.assertEqual(report["status"], expected_status)
            runner.state = {"stages": {"long": report}, "runs": {}}
            summary = runner._summary("acceptance", None)
            self.assertNotEqual(summary["status"], "acceptance_complete")
            self.assertFalse(summary["final_long_context_acceptance_passed"])

    def test_resume_rejects_changed_binary_provenance(self) -> None:
        manifest = self.manifest()
        manifest_path = self.root / "resume-provenance.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        first = StudyRunner(manifest_path, SCRIPTS / "feature-performance-validation.py")
        first.prepare(resume=False)
        self.executable.write_text("#!/usr/bin/env python3\nprint('changed')\n", encoding="utf-8")
        second = StudyRunner(manifest_path, SCRIPTS / "feature-performance-validation.py")
        with self.assertRaisesRegex(core.ProvenanceError, "dirty|binary SHA-256 mismatch"):
            second.prepare(resume=True)

    def test_clean_balanced_processes_and_resume(self) -> None:
        manifest = self.manifest()
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        tool_script = SCRIPTS / "feature-performance-validation.py"
        first = StudyRunner(manifest_path, tool_script)
        summary = first.run(through="early", resume=False, retry_failed=False)
        self.assertEqual(summary["status"], "early_screen_only_not_production_or_final_acceptance")
        self.assertEqual(summary["early_timing"]["mode"], "fresh")
        self.assertEqual(summary["early_timing"]["run_count"], 32)
        self.assertGreater(summary["early_timing"]["target_process_seconds"], 0)
        self.assertGreater(summary["early_timing"]["identity_verification_seconds"], 0)
        state = json.loads(first.state_path.read_text(encoding="utf-8"))
        attempts = [attempt for values in state["runs"].values() for attempt in values]
        self.assertEqual(len(attempts), 32)
        self.assertEqual(len({attempt["pid"] for attempt in attempts}), 32)
        orientations = {
            attempt["schedule"]["pair"]: attempt["schedule"]["orientation"]
            for attempt in attempts
            if attempt["schedule"]["pair"] <= 3 and attempt["schedule"]["screen"] == "low"
        }
        self.assertEqual(orientations, {1: "AB", 2: "BA", 3: "AB"})
        second = StudyRunner(manifest_path, tool_script)
        second.run(through="early", resume=True, retry_failed=False)
        resumed = json.loads(second.state_path.read_text(encoding="utf-8"))
        self.assertTrue(all(len(values) == 1 for values in resumed["runs"].values()))

    def test_inconclusive_three_pairs_extends_to_five_only(self) -> None:
        self.executable.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
            "pair = int(args.get('--pair', 1))\n"
            "variant = args.get('--variant', 'same')\n"
            "candidate = [0, 98, 102, 100, 99, 101][pair]\n"
            "print(f\"metric={candidate if variant == 'candidate' else 100}\")\n",
            encoding="utf-8",
        )
        self.executable.chmod(0o755)
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "adaptive fixture"], cwd=self.repo, check=True)
        manifest = self.manifest()
        stage = next(item for item in manifest["stages"] if item["purpose"] == "kernel_screen")
        stage["command"] = {
            "common_args": ["--pair", "{pair}"],
            "baseline_args": ["--variant", "baseline"],
            "candidate_args": ["--variant", "candidate"],
            "cli_schema": {
                "builtin": "none",
                "allow_positionals": False,
                "options": {"--pair": 1, "--variant": 1},
            },
            "controlled_delta": {
                "reason": "fake candidate values exercise the preregistered extension rule",
                "baseline_args": ["--variant", "baseline"],
                "candidate_args": ["--variant", "candidate"],
                "baseline_environment": {},
                "candidate_environment": {},
                "allowed_options": ["--variant"],
                "allowed_environment": [],
            },
        }
        manifest_path = self.root / "adaptive.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        runner = StudyRunner(manifest_path, SCRIPTS / "feature-performance-validation.py")
        with self.assertRaisesRegex(core.ValidationError, "not acceptable: unresolved"):
            runner.run(through="early", resume=False, retry_failed=False)
        state = json.loads(runner.state_path.read_text(encoding="utf-8"))
        report = state["stages"]["kernel"]
        self.assertTrue(report["adaptive_repetition"]["extended"])
        self.assertEqual(report["statistics"]["pair_count"], 5)
        self.assertEqual(len(report["statistics"]["raw_pairs"]), 5)
        self.assertEqual(report["execution_status"], "completed")
        self.assertEqual(report["status"], "unresolved")

    def test_failed_attempt_is_preserved_and_retry_requires_opt_in(self) -> None:
        flag = self.root / "allow-run"
        self.executable.write_text(
            "#!/usr/bin/env python3\n"
            "import os, sys\n"
            "flag = os.environ['TEST_ALLOW_FLAG']\n"
            "if not os.path.exists(flag):\n"
            "    print('not yet', file=sys.stderr)\n"
            "    raise SystemExit(7)\n"
            "print('metric=100.0')\n",
            encoding="utf-8",
        )
        self.executable.chmod(0o755)
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "retry fixture"], cwd=self.repo, check=True)
        manifest = self.manifest()
        manifest["environment"] = {"TEST_ALLOW_FLAG": str(flag)}
        manifest_path = self.root / "retry.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        first = StudyRunner(manifest_path, SCRIPTS / "feature-performance-validation.py")
        with self.assertRaisesRegex(core.ValidationError, "preserved"):
            first.run(through="early", resume=False, retry_failed=False)
        second = StudyRunner(manifest_path, SCRIPTS / "feature-performance-validation.py")
        with self.assertRaisesRegex(core.ValidationError, "--retry-failed"):
            second.run(through="early", resume=True, retry_failed=False)
        flag.write_text("go", encoding="utf-8")
        third = StudyRunner(manifest_path, SCRIPTS / "feature-performance-validation.py")
        third.run(through="early", resume=True, retry_failed=True)
        state = json.loads(third.state_path.read_text(encoding="utf-8"))
        first_run = state["runs"]["exactness--pair-01-default-01-baseline"]
        self.assertEqual([item["status"] for item in first_run], ["failed", "success"])


if __name__ == "__main__":
    unittest.main()
