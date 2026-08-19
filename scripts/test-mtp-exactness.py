#!/usr/bin/env python3

"""Unit tests for scripts/mtp-exactness.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


RUNNER_PATH = pathlib.Path(__file__).with_name("mtp-exactness.py")
SPEC = importlib.util.spec_from_file_location("mtp_exactness", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {RUNNER_PATH}")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class JsonHelpersTest(unittest.TestCase):
    def test_canonical_hash_ignores_object_key_order(self) -> None:
        self.assertEqual(
            RUNNER.canonical_sha256({"a": 1, "b": [2, 3]}),
            RUNNER.canonical_sha256({"b": [2, 3], "a": 1}),
        )

    def test_deep_merge_preserves_unoverridden_nested_values(self) -> None:
        self.assertEqual(
            RUNNER.deep_merge(
                {"model": "m", "sampler": {"seed": 1, "temperature": 0}},
                {"sampler": {"seed": 2}, "draft_depth": 6},
            ),
            {
                "model": "m",
                "sampler": {"seed": 2, "temperature": 0},
                "draft_depth": 6,
            },
        )

    def test_lookup_dimension_supports_dotted_paths(self) -> None:
        identity = {"sampler": {"seed": 1234}}
        self.assertEqual(RUNNER.lookup_dimension(identity, "sampler.seed"), (True, 1234))
        self.assertEqual(RUNNER.lookup_dimension(identity, "sampler.temperature"), (False, None))

    def test_request_sampler_identity_matches_sent_fields(self) -> None:
        RUNNER.validate_request_identity(
            {"sampler": {"seed": 1234, "temperature": 0.8}},
            {"seed": 1234, "temperature": 0.8},
            "candidate",
            "completion",
        )

    def test_request_sampler_identity_rejects_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "sampler.seed=1234"):
            RUNNER.validate_request_identity(
                {"sampler": {"seed": 1234, "temperature": 0.8}},
                {"seed": 7, "temperature": 0.8},
                "candidate",
                "completion",
            )

    def test_request_sampler_identity_rejects_implicit_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "field 'temperature' is absent"):
            RUNNER.validate_request_identity(
                {"sampler": {"temperature": 0}},
                {"seed": 1234},
                "candidate",
                "completion",
            )


class TokenComparisonTest(unittest.TestCase):
    def test_exact_tokens(self) -> None:
        result = RUNNER.compare_tokens([1, 2, 3], [1, 2, 3])
        self.assertTrue(result["exact"])
        self.assertIsNone(result["first_mismatch"])

    def test_first_token_mismatch(self) -> None:
        result = RUNNER.compare_tokens([1, 2, 3], [1, 9, 3])
        self.assertFalse(result["exact"])
        self.assertEqual(result["first_mismatch"], 1)
        self.assertEqual(result["reference_context"], [1, 2, 3])
        self.assertEqual(result["candidate_context"], [1, 9, 3])

    def test_length_mismatch_starts_after_shared_prefix(self) -> None:
        result = RUNNER.compare_tokens([1, 2], [1, 2, 3])
        self.assertFalse(result["exact"])
        self.assertEqual(result["first_mismatch"], 2)


class SequenceTest(unittest.TestCase):
    def test_legacy_manifest_resolves_to_one_completion(self) -> None:
        sequence, legacy = RUNNER.resolved_sequence({"request": {"prompt": "test"}})
        self.assertTrue(legacy)
        self.assertEqual(sequence, [{"name": "completion", "type": "completion"}])

    def test_sequence_rejects_forward_prompt_reference(self) -> None:
        manifest = {
            "sequence": [
                {
                    "name": "second",
                    "type": "completion",
                    "prompt_from": {"step": "first"},
                },
                {"name": "first", "type": "completion"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "earlier completion"):
            RUNNER.resolved_sequence(manifest)

    def test_sequence_rejects_unknown_action(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported type"):
            RUNNER.resolved_sequence(
                {"sequence": [{"name": "shell", "type": "run_command"}]}
            )

    def test_sequence_validates_sleep_state(self) -> None:
        sequence, legacy = RUNNER.resolved_sequence(
            {
                "sequence": [
                    {"name": "first", "type": "completion"},
                    {
                        "name": "asleep",
                        "type": "wait_server_state",
                        "state": "sleeping",
                    },
                    {
                        "name": "second",
                        "type": "completion",
                        "prompt_from": {"step": "first"},
                    },
                ]
            }
        )
        self.assertFalse(legacy)
        self.assertEqual([step["name"] for step in sequence], ["first", "asleep", "second"])

    def test_sequence_accepts_typed_reload_and_within_case_reference(self) -> None:
        sequence, legacy = RUNNER.resolved_sequence(
            {
                "sequence": [
                    {"name": "before", "type": "completion"},
                    {"name": "reload", "type": "reload_models", "model": "m"},
                    {
                        "name": "after",
                        "type": "completion",
                        "require_equal_to": "before",
                    },
                ]
            }
        )
        self.assertFalse(legacy)
        self.assertEqual(sequence[1]["type"], "reload_models")

    def test_sequence_rejects_forward_within_case_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "require_equal_to"):
            RUNNER.resolved_sequence(
                {
                    "sequence": [
                        {
                            "name": "before",
                            "type": "completion",
                            "require_equal_to": "after",
                        },
                        {"name": "after", "type": "completion"},
                    ]
                }
            )

    def test_continuation_prompt_uses_exact_prior_tokens_and_unspecial_suffix(self) -> None:
        prior = {
            "first": {
                "prompt_tokens": [10, 11],
                "tokens": [20, 21, 22],
            }
        }
        with mock.patch.object(RUNNER, "request_json", return_value={"tokens": [30, 31]}) as call:
            tokens, construction = RUNNER.build_continuation_prompt(
                "http://127.0.0.1:1234",
                {
                    "step": "first",
                    "prompt_token_count": 1,
                    "response_token_count": 2,
                    "suffix": "next",
                },
                prior,
            )
        self.assertEqual(tokens, [10, 20, 21, 30, 31])
        self.assertEqual(construction["source_prompt_count"], 1)
        self.assertEqual(construction["source_response_count"], 2)
        self.assertFalse(call.call_args.args[2]["add_special"])

    def test_continuation_prompt_rejects_out_of_range_count(self) -> None:
        prior = {"first": {"prompt_tokens": [10], "tokens": [20]}}
        with self.assertRaisesRegex(ValueError, "exceeds available"):
            RUNNER.build_continuation_prompt(
                "http://127.0.0.1:1234",
                {"step": "first", "response_token_count": 2},
                prior,
            )


class SequenceComparisonTest(unittest.TestCase):
    @staticmethod
    def result(second_token: int = 4, prompt_hash: str = "p2") -> dict:
        def step(name: str, tokens: list[int], prompt: str) -> dict:
            return {
                "name": name,
                "tokens": tokens,
                "content": bytes(tokens),
                "summary": {
                    "request_semantics_sha256": f"r-{name}",
                    "prompt_tokens_sha256": prompt,
                },
            }

        return {
            "steps": [
                step("first", [1, 2], "p1"),
                step("second", [3, second_token], prompt_hash),
            ]
        }

    def test_all_completion_steps_are_compared(self) -> None:
        comparison = RUNNER.compare_completion_sequences(self.result(), self.result())
        self.assertTrue(comparison["exact"])
        self.assertTrue(comparison["content_exact"])
        self.assertTrue(comparison["prompt_tokens_exact"])
        self.assertIsNone(comparison["first_failed_step"])

    def test_later_step_mismatch_is_localized(self) -> None:
        comparison = RUNNER.compare_completion_sequences(self.result(), self.result(9))
        self.assertFalse(comparison["exact"])
        self.assertEqual(comparison["first_failed_step"], 1)
        self.assertEqual(comparison["first_mismatch"], 1)

    def test_model_status_lookup_is_strict(self) -> None:
        response = {
            "data": [
                {"id": "a", "status": {"value": "loaded"}},
                {"id": "b", "status": {"value": "unloaded"}},
            ]
        }
        self.assertEqual(RUNNER.find_model_status(response, "a"), "loaded")
        self.assertEqual(RUNNER.find_model_status(response, "b"), "unloaded")
        self.assertIsNone(RUNNER.find_model_status(response, "missing"))

    def test_later_prompt_mismatch_breaks_contract(self) -> None:
        comparison = RUNNER.compare_completion_sequences(
            self.result(), self.result(prompt_hash="different")
        )
        self.assertTrue(comparison["exact"])
        self.assertFalse(comparison["prompt_tokens_exact"])
        self.assertEqual(comparison["first_failed_step"], 1)


class ComparisonContractTest(unittest.TestCase):
    def manifest(self) -> dict:
        return {
            "comparison_contract": {
                "require_explicit_references": True,
                "required_identity_keys": [
                    "model",
                    "draft_depth",
                    "sampler.seed",
                ],
                "common_identity": {
                    "model": "model.gguf",
                    "sampler": {"seed": 1234},
                },
            }
        }

    def cases(self) -> list[dict]:
        return [
            {
                "name": "golden-mtp6",
                "role": "golden",
                "identity": {"draft_depth": 6},
            },
            {
                "name": "candidate-mtp6",
                "compare_to": "golden-mtp6",
                "identity": {"draft_depth": 6},
            },
        ]

    def test_explicit_golden_graph_is_valid(self) -> None:
        RUNNER.validate_case_graph(self.manifest(), self.cases())

    def test_duplicate_case_names_are_rejected(self) -> None:
        cases = self.cases()
        cases[1]["name"] = cases[0]["name"]
        with self.assertRaisesRegex(ValueError, "unique"):
            RUNNER.validate_case_graph(self.manifest(), cases)

    def test_candidate_without_reference_is_rejected(self) -> None:
        cases = self.cases()
        del cases[1]["compare_to"]
        with self.assertRaisesRegex(ValueError, "must set compare_to"):
            RUNNER.validate_case_graph(self.manifest(), cases)

    def test_unknown_reference_is_rejected(self) -> None:
        cases = self.cases()
        cases[1]["compare_to"] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown"):
            RUNNER.validate_case_graph(self.manifest(), cases)

    def test_candidate_reference_is_rejected(self) -> None:
        cases = self.cases()
        cases.append(
            {
                "name": "candidate-two",
                "compare_to": "candidate-mtp6",
                "identity": {"draft_depth": 6},
            }
        )
        with self.assertRaisesRegex(ValueError, "directly to a golden"):
            RUNNER.validate_case_graph(self.manifest(), cases)

    def test_missing_required_nested_identity_is_rejected(self) -> None:
        manifest = self.manifest()
        manifest["comparison_contract"]["common_identity"]["sampler"] = {}
        with self.assertRaisesRegex(ValueError, "sampler.seed"):
            RUNNER.validate_case_graph(manifest, self.cases())

    def test_legacy_global_reference_remains_supported(self) -> None:
        manifest = {"reference": "target"}
        cases = [{"name": "target"}, {"name": "candidate"}]
        RUNNER.validate_case_graph(manifest, cases)
        self.assertEqual(RUNNER.case_role(manifest, cases[0]), "golden")
        self.assertEqual(RUNNER.case_role(manifest, cases[1]), "candidate")

    def test_artifact_backed_candidate_is_rejected(self) -> None:
        cases = self.cases()
        cases[1]["artifact_dir"] = "/tmp/not-used"
        with self.assertRaisesRegex(ValueError, "must have role 'golden'"):
            RUNNER.validate_case_graph(self.manifest(), cases)

    def test_declared_ubatch_geometry_matches_inherited_draft(self) -> None:
        manifest = self.manifest()
        manifest["common_args"] = ["--ubatch-size", "512"]
        manifest["comparison_contract"]["required_identity_keys"].extend(
            ["ubatch", "effective_draft_ubatch"]
        )
        manifest["comparison_contract"]["common_identity"].update(
            {"ubatch": 512, "effective_draft_ubatch": 512}
        )
        RUNNER.validate_case_graph(manifest, self.cases())

    def test_cli_draft_ubatch_cannot_disagree_with_identity(self) -> None:
        manifest = self.manifest()
        manifest["common_args"] = [
            "--ubatch-size", "512", "--spec-draft-ubatch-size", "128"
        ]
        manifest["comparison_contract"]["common_identity"].update(
            {"ubatch": 512, "effective_draft_ubatch": 512}
        )
        with self.assertRaisesRegex(ValueError, "effective_draft_ubatch=512"):
            RUNNER.validate_case_graph(manifest, self.cases())

    def test_environment_draft_ubatch_cannot_disagree_with_identity(self) -> None:
        manifest = self.manifest()
        manifest["common_args"] = ["--ubatch-size", "512"]
        manifest["environment"] = {"LLAMA_ARG_SPEC_DRAFT_UBATCH": "128"}
        manifest["comparison_contract"]["common_identity"].update(
            {"ubatch": 512, "effective_draft_ubatch": 512}
        )
        with self.assertRaisesRegex(ValueError, "resolves to 128"):
            RUNNER.validate_case_graph(manifest, self.cases())

    def test_target_ubatch_cannot_disagree_with_identity(self) -> None:
        manifest = self.manifest()
        manifest["common_args"] = ["--ubatch-size=128"]
        manifest["comparison_contract"]["common_identity"].update(
            {"ubatch": 512, "effective_draft_ubatch": 512}
        )
        with self.assertRaisesRegex(ValueError, "declares ubatch=512"):
            RUNNER.validate_case_graph(manifest, self.cases())


class ArtifactGoldenTest(unittest.TestCase):
    def test_load_artifact_case_verifies_and_copies_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            identity = {"model": "m", "draft_depth": 2}
            response = {"tokens": [1, 2, 3], "content": "abc"}
            summary = {
                "tokens_sha256": RUNNER.sha256_bytes(b"[1,2,3]"),
                "content_sha256": RUNNER.sha256_bytes(b"abc"),
                "request_semantics_sha256": "request-hash",
                "prompt_tokens_sha256": "prompt-hash",
            }
            fixtures = {
                "identity.json": identity,
                "response.json": response,
                "summary.json": summary,
                "request.json": {"prompt": "test"},
                "prompt-tokens.json": {"tokens": [4, 5]},
            }
            for filename, value in fixtures.items():
                (source / filename).write_text(json.dumps(value), encoding="utf-8")

            manifest = {"comparison_contract": {"common_identity": identity}}
            case = {"name": "golden", "role": "golden", "artifact_dir": str(source)}
            result = RUNNER.load_artifact_case(manifest, case, output)

            self.assertEqual(result["tokens"], [1, 2, 3])
            self.assertEqual(result["content"], b"abc")
            self.assertTrue((output / "golden" / "artifact-reference.json").is_file())

    def test_load_artifact_case_rejects_tampered_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            identity = {"model": "m"}
            fixtures = {
                "identity.json": identity,
                "response.json": {"tokens": [9], "content": "changed"},
                "summary.json": {"tokens_sha256": "wrong", "content_sha256": "wrong"},
                "request.json": {"prompt": "test"},
                "prompt-tokens.json": {"tokens": [4, 5]},
            }
            for filename, value in fixtures.items():
                (source / filename).write_text(json.dumps(value), encoding="utf-8")

            manifest = {"comparison_contract": {"common_identity": identity}}
            case = {"name": "golden", "role": "golden", "artifact_dir": str(source)}
            with self.assertRaisesRegex(ValueError, "token hash"):
                RUNNER.load_artifact_case(manifest, case, output)


if __name__ == "__main__":
    unittest.main()
