#!/usr/bin/env python3

"""CPU-only tests for the feature/performance validation toolkit."""

from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from feature_validation import core, profiler, telemetry  # noqa: E402
from feature_validation.runner import StudyRunner  # noqa: E402


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
            "executable": str(self.executable),
            "expected_sha256": file_sha256(self.executable),
            "source_root": str(self.repo),
            "expected_commit": head,
            "tree_policy": "clean",
            "build": {"mode": "not_applicable", "reason": "test script"},
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
        return {
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


class QuotingAndArityTest(unittest.TestCase):
    def test_safe_quoting_round_trips_spaces_quotes_and_metacharacters(self) -> None:
        argv = ["/tmp/a b", "single'quote", 'double"quote', "$(do-not-run)", "semi;colon"]
        rendered = core.quote_argv(argv)
        self.assertEqual(shlex.split(rendered), argv)

    def test_zero_arity_option_rejects_separate_value(self) -> None:
        with self.assertRaisesRegex(core.ValidationError, "unexpected positional"):
            core.validate_argv(["--no-kv-offload", "1"])

    def test_zero_arity_option_rejects_equals_value(self) -> None:
        with self.assertRaisesRegex(core.ValidationError, "zero-arity"):
            core.validate_argv(["--no-kv-offload=1"])

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
        self.assertEqual(json.loads(schema_path.read_text(encoding="utf-8"))["$schema"], "https://json-schema.org/draft/2020-12/schema")
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

    def test_workload_setting_cannot_hide_in_candidate_delta(self) -> None:
        manifest = self.manifest()
        stage = next(item for item in manifest["stages"] if item["purpose"] == "kernel_screen")
        stage["command"]["candidate_args"] = ["-ub", "128"]
        stage["command"]["cli_schema"] = {"builtin": "llama", "allow_positionals": False}
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
        manifest["variants"]["candidate"]["expected_sha256"] = "0" * 64
        with self.assertRaisesRegex(core.ProvenanceError, "binary SHA-256 mismatch"):
            core.capture_provenance(manifest, "manifest-hash")

    def test_dirty_tree_fails_closed(self) -> None:
        manifest = self.manifest()
        (self.repo / "untracked").write_text("dirt", encoding="utf-8")
        with self.assertRaisesRegex(core.ProvenanceError, "dirty"):
            core.capture_provenance(manifest, "manifest-hash")


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

    def test_weighted_harmonic_matches_prefill_throughput_geometry(self) -> None:
        value = core.aggregate_screen_values([(100.0, 1.0), (200.0, 3.0)], "weighted_harmonic")
        self.assertAlmostEqual(value, 160.0)

    def test_metric_parser_fails_when_no_sample_exists(self) -> None:
        with self.assertRaisesRegex(core.ValidationError, "did not match"):
            core.extract_metric({"regex": r"metric=([0-9.]+)"}, "nothing", "")


class ProfilerTest(unittest.TestCase):
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
            self.assertIn("flock /tmp/beellama-single-gpu.lock -c", plan["command"]["flock_command"])

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
        runner.run(through="early", resume=False, retry_failed=False)
        state = json.loads(runner.state_path.read_text(encoding="utf-8"))
        report = state["stages"]["kernel"]
        self.assertTrue(report["adaptive_repetition"]["extended"])
        self.assertEqual(report["statistics"]["pair_count"], 5)
        self.assertEqual(len(report["statistics"]["raw_pairs"]), 5)

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
