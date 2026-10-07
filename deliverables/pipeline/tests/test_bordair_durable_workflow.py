from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PIPELINE_ROOT.parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from aegis_research import bordair_evaluation as evaluation  # noqa: E402
from aegis_research import bordair_evaluation_validation as validation  # noqa: E402
from aegis_research import bordair_promotion as promotion  # noqa: E402
from aegis_research.bordair_paths import RunPaths  # noqa: E402
from aegis_research.bordair_provenance import (  # noqa: E402
    DETECTOR_CORPUS_VERSION,
    detector_source_label,
    training_identity_sha256,
    validate_summary_training_identity,
)


def workflow_args(run_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_root=run_root,
        evaluation_dir=None,
        training_dir=None,
        manifest=None,
        final_metadata=None,
        text_led_metadata=None,
        external_benign_metadata=None,
        targets=[],
    )


class RunPathsTests(unittest.TestCase):
    def test_every_default_is_derived_from_one_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "isolated-run"
            paths = RunPaths(root)
            expected = {
                paths.manifest: "corpus_manifest_v7.json",
                paths.development_metadata: "development_metadata_v7.csv",
                paths.regression_metadata: "regression_metadata_v7.csv",
                paths.final_metadata: "final_metadata_v7.csv",
                paths.text_led_metadata: "text_led_final_metadata_v7.csv",
                paths.external_benign_metadata: "external_benign_metadata_v7.csv",
                paths.image_root: "images_v7",
                paths.training_directory: "training_v7",
                paths.evaluation_directory: "evaluation_v7",
                paths.feature_cache_directory: "features_v7",
            }
            for path, name in expected.items():
                with self.subTest(path=path):
                    self.assertEqual(path.parent, paths.root)
                    self.assertEqual(path.name, name)

    def test_corpus_overrides_cannot_mix_two_run_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = RunPaths(Path(temporary) / "run-a")
            other_manifest = Path(temporary) / "run-b" / "corpus_manifest_v7.json"
            with self.assertRaisesRegex(ValueError, "Set --run-root"):
                paths.corpus_input(other_manifest, paths.manifest, "--manifest")

    def test_validator_and_promotion_resolve_identical_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = workflow_args(Path(temporary) / "run")
            validator_paths = validation.resolve_cli_paths(args)
            promotion_paths = promotion.resolve_cli_paths(args)
            self.assertEqual(validator_paths, promotion_paths)
            paths = validator_paths[0]
            self.assertEqual(validator_paths[1], paths.evaluation_directory)
            self.assertEqual(validator_paths[2], paths.training_directory)
            self.assertEqual(validator_paths[3], paths.manifest)


class StrictValidationTests(unittest.TestCase):
    def test_evaluator_and_strict_corpus_validator_target_schema_four(self) -> None:
        self.assertEqual(evaluation.EXPECTED_V7_MANIFEST_SCHEMA, 4)

    def test_shared_validator_requires_fused_pooling_and_doubled_dimension(self) -> None:
        valid = SimpleNamespace(pooling="text_image_tokens", feature_dim=1792)
        validation.validate_fused_representation("llava05b", valid)

        with self.assertRaisesRegex(ValueError, "text_image_tokens"):
            validation.validate_fused_representation(
                "llava05b",
                SimpleNamespace(pooling="image_tokens", feature_dim=896),
            )
        with self.assertRaisesRegex(ValueError, "feature_dim=1792"):
            validation.validate_fused_representation(
                "llava05b",
                SimpleNamespace(pooling="text_image_tokens", feature_dim=896),
            )
        validation.validate_fused_representation(
            "qwen25vl3b",
            SimpleNamespace(pooling="text_image_tokens", feature_dim=4096),
        )

    def test_evaluator_runs_strict_corpus_validation_before_loading_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                **vars(workflow_args(Path(temporary) / "run")),
                target="llava05b",
                summary=None,
                artifact=None,
                output_dir=None,
            )
            with patch.object(evaluation, "parse_args", return_value=args), patch.object(
                evaluation,
                "validate_tracked_v7_corpus",
                side_effect=RuntimeError("strict-validation-sentinel"),
            ) as strict:
                with self.assertRaisesRegex(RuntimeError, "strict-validation-sentinel"):
                    evaluation.main()
            strict.assert_called_once()

    def test_evidence_validator_and_promotion_validate_corpus_first(self) -> None:
        for module in (validation, promotion):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                args = workflow_args(Path(temporary) / "run")
                with patch.object(module, "parse_args", return_value=args), patch.object(
                    evaluation,
                    "validate_tracked_v7_corpus",
                    side_effect=RuntimeError("strict-validation-sentinel"),
                ) as strict:
                    with self.assertRaisesRegex(RuntimeError, "strict-validation-sentinel"):
                        module.main()
                strict.assert_called_once()


class DetectorIdentityTests(unittest.TestCase):
    def test_training_identity_is_deterministic_and_configuration_sensitive(self) -> None:
        configuration = {
            "weights": {"benign": 1.0, "malicious": 1.1},
            "poolings": ["text_image_tokens"],
            "random_seed": 42,
        }
        candidate = {
            "pooling": "text_image_tokens",
            "learning_rate": 0.02,
            "l2": 0.001,
            "block_threshold": 0.4,
        }
        first = training_identity_sha256(configuration, candidate)
        reordered = training_identity_sha256(
            {
                "random_seed": 42,
                "poolings": ["text_image_tokens"],
                "weights": {"malicious": 1.1, "benign": 1.0},
            },
            dict(reversed(list(candidate.items()))),
        )
        self.assertEqual(first, reordered)

        changed_weight = {
            **configuration,
            "weights": {"benign": 1.0, "malicious": 1.2},
        }
        self.assertNotEqual(
            first,
            training_identity_sha256(changed_weight, candidate),
        )
        changed_candidate = {**candidate, "l2": 0.01}
        self.assertNotEqual(
            first,
            training_identity_sha256(configuration, changed_candidate),
        )

    def test_source_label_contains_full_training_identity_digest(self) -> None:
        digest = training_identity_sha256(
            {"epochs": 600},
            {"pooling": "text_image_tokens", "candidate_id": 0},
        )
        label = detector_source_label(
            corpus_version="bordair_ocr_multichannel_counterfactual_v7",
            corpus_manifest_sha256="a" * 64,
            target="llava05b",
            pooling="text_image_tokens",
            training_identity_sha256_value=digest,
        )
        self.assertTrue(label.endswith(f"cfgcand-{digest}"))

    def test_summary_identity_validation_rejects_tampering(self) -> None:
        configuration = {"epochs": 600, "weights": {"benign": 1.0}}
        candidate = {"pooling": "text_image_tokens", "candidate_id": 0}
        digest = training_identity_sha256(configuration, candidate)
        source = detector_source_label(
            corpus_version=DETECTOR_CORPUS_VERSION,
            corpus_manifest_sha256="b" * 64,
            target="llava05b",
            pooling="text_image_tokens",
            training_identity_sha256_value=digest,
        )
        summary = {
            "training_configuration": configuration,
            "selected_candidate": candidate,
            "training_identity": {"schema_version": 1, "sha256": digest},
        }
        self.assertEqual(
            validate_summary_training_identity(
                summary,
                target="llava05b",
                pooling="text_image_tokens",
                corpus_manifest_sha256="b" * 64,
                artifact_source=source,
            ),
            digest,
        )
        tampered = {
            **summary,
            "training_configuration": {"epochs": 601, "weights": {"benign": 1.0}},
        }
        with self.assertRaisesRegex(ValueError, "identity digest"):
            validate_summary_training_identity(
                tampered,
                target="llava05b",
                pooling="text_image_tokens",
                corpus_manifest_sha256="b" * 64,
                artifact_source=source,
            )


class WrapperTests(unittest.TestCase):
    def test_training_cli_and_readme_use_canonical_hard_wired_run_root(self) -> None:
        script = PIPELINE_ROOT / "scripts" / "train_bordair_detector.py"
        help_result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=REPOSITORY,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertNotIn("--run-root", help_result.stdout)
        rejected = subprocess.run(
            [
                sys.executable,
                str(script),
                "qwen25vl3b",
                "--run-root",
                "substitution-run",
                "--extract-only",
            ],
            cwd=REPOSITORY,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("unrecognized arguments: --run-root", rejected.stderr)
        readme = (PIPELINE_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("canonical hard-wired", readme)
        script_marker = (
            "  deliverables/pipeline/scripts/"
            "train_bordair_detector.py qwen25vl3b `"
        )
        script_offset = readme.index(script_marker)
        block_start = readme.rfind("```powershell", 0, script_offset)
        block_end = readme.find("```", script_offset)
        self.assertGreaterEqual(block_start, 0)
        self.assertGreater(block_end, script_offset)
        binding = readme[block_start:block_end]
        self.assertIn("docker run --rm --gpus all `", binding)
        self.assertIn("  --entrypoint python `", binding)
        self.assertIn("  $imageId `\n" + script_marker, binding)
        self.assertIn(
            "$imageId = (docker image inspect --format '{{.Id}}' $image).Trim()",
            readme[:script_offset],
        )
        self.assertNotIn("python " + script_marker.lstrip(), binding)
        self.assertNotIn("--run-root", binding)

    def test_tracked_workflow_wrappers_expose_run_root(self) -> None:
        wrappers = (
            "validate_bordair_corpus.py",
            "evaluate_bordair_detector.py",
            "validate_bordair_evaluation.py",
            "validate_bordair_promotion.py",
        )
        for name in wrappers:
            with self.subTest(wrapper=name):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(PIPELINE_ROOT / "scripts" / name),
                        "--help",
                    ],
                    cwd=REPOSITORY,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--run-root", result.stdout)


if __name__ == "__main__":
    unittest.main()
