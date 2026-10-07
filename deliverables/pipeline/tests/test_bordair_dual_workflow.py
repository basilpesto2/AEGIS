from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PIPELINE_ROOT.parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from AEGIS.detector_artifact import DetectorArtifact, save_detector_artifact  # noqa: E402
from AEGIS.detector_set import detector_set_identity_sha256  # noqa: E402
from AEGIS.logistic import LogisticRegressionNumpy  # noqa: E402
from aegis_research.bordair_dual import (  # noqa: E402
    ChannelCandidate,
    build_pair_manifest,
    build_one_shot_qualification_binding,
    combine_actions,
    head_action,
    load_detector_pair,
    pair_identity_sha256,
    qualification_cache_locator,
    runtime_detector_identity_sha256,
    score_pair,
    select_validation_pair,
    validate_pair_manifest_payload,
)
from aegis_research.bordair_dual_evaluation_validation import (  # noqa: E402
    validate_recomputed_row,
)
from aegis_research.bordair_dual_evaluation import (  # noqa: E402
    acceptance_report,
    score_dual_panels,
    validate_development_qualification_evidence,
    validate_exact_frozen_cache_report,
)
from aegis_research.bordair_dual_promotion import (  # noqa: E402
    validate_dual_promotion_candidate,
    validate_runtime_detector_config,
)
from aegis_research.bordair_dual_training import train_dual_pair  # noqa: E402
from aegis_research.bordair_evaluation import display_path  # noqa: E402
from aegis_research.bordair_paths import RunPaths  # noqa: E402
from aegis_research import bordair_training as training_workflow  # noqa: E402


DIGEST = "a" * 64


def fitted_classifier(weights: tuple[float, ...], bias: float = 0.0) -> LogisticRegressionNumpy:
    classifier = LogisticRegressionNumpy(
        learning_rate=0.05,
        epochs=10,
        l2=0.1,
        standardize=True,
        random_seed=42,
    )
    classifier.weights_ = np.asarray(weights, dtype=np.float64)
    classifier.bias_ = bias
    classifier.mean_ = np.zeros(len(weights), dtype=np.float64)
    classifier.scale_ = np.ones(len(weights), dtype=np.float64)
    return classifier


def detector(pooling: str, *, model_id: str = "synthetic/model") -> DetectorArtifact:
    return DetectorArtifact(
        classifier=fitted_classifier((2.0, -0.5)),
        threshold=0.6,
        uncertainty_margin=0.1,
        model_family="generic",
        model_id=model_id,
        model_revision="revision",
        tokenizer_revision="tokenizer",
        preprocessing_sha256=DIGEST,
        layer=-1,
        pooling=pooling,
        source=f"synthetic-{pooling}",
    )


def create_pair(root: Path) -> tuple[dict[str, object], Path]:
    image_path = root / "image.npz"
    text_path = root / "text.npz"
    image = detector("image_tokens")
    text = detector("text_tokens")
    save_detector_artifact(image_path, image)
    save_detector_artifact(text_path, text)
    manifest = build_pair_manifest(
        target="synthetic",
        image_path=image_path,
        image_artifact=image,
        image_review_threshold=0.5,
        text_path=text_path,
        text_artifact=text,
        text_review_threshold=0.5,
        output_directory=root,
        training_identity_sha256_value="b" * 64,
        corpus={
            "version": "schema4-test",
            "manifest_sha256": "c" * 64,
            "development_metadata_sha256": "d" * 64,
            "regression_metadata_sha256": "e" * 64,
        },
    )
    path = root / "detector_pair_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest, path


def create_qualification_evidence(
    root: Path,
    *,
    image_hash: str,
    text_hash: str,
    corpus_hash: str,
) -> Path:
    evidence = root / "frozen" / "evidence"
    evidence.mkdir(parents=True)
    protocol_path = evidence.parent / "protocol.json"
    evaluator_path = evidence.parent / "evaluate_once.py"
    validator_path = evidence.parent / "validate_once.py"
    attempt_path = evidence / "attempt_started.json"
    development_manifest_path = root / "output_manifest.json"
    results_json_path = evidence / "dual_head_frozen_results.json"
    results_csv_path = evidence / "dual_head_frozen_results.csv"
    evaluation_manifest_path = evidence / "evaluation_manifest.json"
    validation_report_path = evidence / "validation_report.json"
    validation_manifest_path = evidence / "validation_manifest.json"
    metadata_directory = evidence.parent / "metadata"
    metadata_directory.mkdir()
    metadata_paths = {
        "regression": metadata_directory / "final_metadata_v7.csv",
        "text_led": metadata_directory / "text_led_final_metadata_v7.csv",
        "external_benign": metadata_directory
        / "external_benign_metadata_v7.csv",
    }
    panel_labels = (
        ("regression", [1] * 10 + [0] * 10),
        ("text_led", [1] * 10),
        ("external_benign", [0] * 20),
    )
    for panel, labels in panel_labels:
        with metadata_paths[panel].open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "sample_id",
                    "label_id",
                    "prompt_text",
                    "image_sha256",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            for index, label in enumerate(labels):
                sample_id = f"{panel}-{index:02d}"
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "label_id": label,
                        "prompt_text": f"caller text for {sample_id}",
                        "image_sha256": hashlib.sha256(
                            f"image for {sample_id}".encode()
                        ).hexdigest(),
                    }
                )
    evaluator_path.write_text("# synthetic evaluator\n", encoding="utf-8")
    validator_path.write_text("# synthetic validator\n", encoding="utf-8")
    development_manifest_path.write_text('{"schema_version":1}\n', encoding="utf-8")

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    protocol_id = "synthetic-schema4-dual-head-frozen-once-v1"
    metadata_hashes = {
        panel: digest(path) for panel, path in metadata_paths.items()
    }
    cache_statistics = {
        "hits": 50,
        "misses": 0,
        "invalid_entries": 0,
        "rebuilds": 0,
        "writes": 0,
        "write_errors": 0,
        "delegate_calls": 0,
    }
    protocol = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "status": "declared_before_frozen_execution",
        "target": "synthetic",
        "execution_limit": 1,
        "selection_or_tuning_permitted": False,
        "development_binding": {
            "manifest_sha256": digest(development_manifest_path),
            "image_artifact_sha256": image_hash,
            "text_artifact_sha256": text_hash,
        },
        "corpus_binding": {
            "manifest_schema": 4,
            "manifest_sha256": corpus_hash,
            "fixed_metadata_sha256": metadata_hashes["regression"],
            "text_led_metadata_sha256": metadata_hashes["text_led"],
            "external_benign_metadata_sha256": metadata_hashes["external_benign"],
        },
        "artifact_contract": {
            "format": "AEGIS DetectorArtifact NPZ version 1",
            "image_pooling": "image_tokens",
            "text_pooling": "text_tokens",
            "per_head_dimension": 2,
        },
        "cache_contract": {
            "pooling": "text_image_tokens",
            "dimension": 4,
            "split_order": ["text_tokens[0:2]", "image_tokens[2:4]"],
            "expected_rows": 50,
            "required_statistics": cache_statistics,
        },
        "action_contract": {
            "per_head": "artifact threshold",
            "composition": "or",
        },
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    attempt = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "protocol_sha256": digest(protocol_path),
        "execution_ordinal": 1,
        "retries_permitted": False,
        "evaluator_sha256": digest(evaluator_path),
        "started_utc": "2026-01-01T00:00:00+00:00",
    }
    attempt_path.write_text(json.dumps(attempt), encoding="utf-8")

    shared = {
        "model_family": "generic",
        "model_id": "synthetic/model",
        "model_revision": "revision",
        "tokenizer_revision": "tokenizer",
        "preprocessing_sha256": DIGEST,
        "layer": -1,
    }
    artifact_entry = lambda name: {
        "artifact_version": 1,
        "sha256": image_hash if name == "image" else text_hash,
        "pooling": f"{name}_tokens",
        "feature_dim": 2,
        "block_threshold": 0.6,
        "review_threshold": 0.5,
        "uncertainty_margin": 0.1,
        **shared,
    }
    rows: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    sequence = 0
    for panel, labels in panel_labels:
        for index, label in enumerate(labels):
            sequence += 1
            sample_id = f"{panel}-{index:02d}"
            image_score = 0.9 if panel == "regression" and label == 1 else 0.1
            text_score = 0.9 if panel == "text_led" and label == 1 else 0.1
            image_action = "block" if image_score >= 0.6 else "allow"
            text_action = "block" if text_score >= 0.6 else "allow"
            combined = "block" if "block" in (image_action, text_action) else "allow"
            expected_action = "block" if label == 1 else "allow"
            cache_hash = hashlib.sha256(sample_id.encode()).hexdigest()
            cache_path = f"cache/{panel}/{sample_id}.npz"
            prompt_text = f"caller text for {sample_id}"
            image_sha256 = hashlib.sha256(
                f"image for {sample_id}".encode()
            ).hexdigest()
            rows.append(
                {
                    "sequence": sequence,
                    "panel": panel,
                    "sample_id": sample_id,
                    "label_id": label,
                    "expected_action": expected_action,
                    "image_head_score": image_score,
                    "image_head_block_threshold": 0.6,
                    "image_head_review_threshold": 0.5,
                    "image_head_action": image_action,
                    "text_head_score": text_score,
                    "text_head_block_threshold": 0.6,
                    "text_head_review_threshold": 0.5,
                    "text_head_action": text_action,
                    "combined_action": combined,
                    "accepted": combined == expected_action,
                    "cache_path": cache_path,
                    "cache_sha256": cache_hash,
                    "caller_text_sha256": hashlib.sha256(
                        prompt_text.encode()
                    ).hexdigest(),
                    "image_sha256": image_sha256,
                }
            )
            entries.append(
                {
                    "panel": panel,
                    "sample_id": sample_id,
                    "path": cache_path,
                    "sha256": cache_hash,
                    "bytes": 10,
                }
            )
    cache_tree = hashlib.sha256(
        ("\n".join(f"{entry['path']}|10|{entry['sha256']}" for entry in entries) + "\n").encode()
    ).hexdigest()

    def panel_summary(panel: str) -> dict[str, object]:
        selected = [row for row in rows if row["panel"] == panel]
        malicious = [row for row in selected if row["label_id"] == 1]
        benign = [row for row in selected if row["label_id"] == 0]
        count = lambda field: {
            action: sum(row[field] == action for row in selected)
            for action in ("allow", "review", "block")
        }
        return {
            "rows": len(selected),
            "malicious_rows": len(malicious),
            "benign_rows": len(benign),
            "malicious_blocked": sum(row["combined_action"] == "block" for row in malicious),
            "benign_blocked": sum(row["combined_action"] == "block" for row in benign),
            "combined_action_counts": count("combined_action"),
            "image_head_action_counts": count("image_head_action"),
            "text_head_action_counts": count("text_head_action"),
        }

    panels = {name: panel_summary(name) for name in metadata_hashes}
    gate_checks = {
        "external_benign_exact_counts": True,
        "external_benign_fewer_than_3_blocked": True,
        "fixed_benign_fewer_than_3_blocked": True,
        "fixed_exact_counts": True,
        "fixed_malicious_10_of_10_blocked": True,
        "text_led_exact_counts": True,
        "text_led_malicious_10_of_10_blocked": True,
    }
    results = {
        "schema_version": 1,
        "evaluation_id": protocol_id,
        "promotion_authorized": False,
        "one_shot_attempt": attempt,
        "protocol": {"sha256": digest(protocol_path), "payload": protocol},
        "artifacts": {
            "development_manifest": {"sha256": digest(development_manifest_path)},
            "shared_provenance": shared,
            "image_head": artifact_entry("image"),
            "text_head": artifact_entry("text"),
        },
        "split_contract": {
            "fused_dimension": 4,
            "text_slice": [0, 2],
            "image_slice": [2, 4],
        },
        "action_contract": protocol["action_contract"],
        "cache": {
            "delegate_object_calls": 0,
            "entry_tree_sha256": cache_tree,
            "entries": entries,
            "report": {
                "enabled": True,
                "schema_version": 1,
                "expected_rows": 50,
                "rebuild_requested": False,
                "write_error_details": [],
                **cache_statistics,
            },
        },
        "panels": panels,
        "gates": {"checks": gate_checks, "passed": True},
        "results": rows,
    }
    results_json_path.write_text(json.dumps(results), encoding="utf-8")
    results_csv_path.write_text("row\n" + "\n".join(str(i) for i in range(50)) + "\n", encoding="utf-8")
    evaluation_manifest = {
        "schema_version": 1,
        "evaluation_id": protocol_id,
        "row_count": 50,
        "corpus_manifest_sha256": corpus_hash,
        "cache_entry_tree_sha256": cache_tree,
        "artifacts": {
            "image_head_sha256": image_hash,
            "text_head_sha256": text_hash,
        },
        "development_manifest": {"sha256": digest(development_manifest_path)},
        "protocol": {"sha256": digest(protocol_path)},
        "attempt": {"sha256": digest(attempt_path)},
        "evaluator": {"sha256": digest(evaluator_path)},
        "validator": {"sha256": digest(validator_path)},
        "panel_metadata_sha256": metadata_hashes,
        "outputs": {
            "dual_head_frozen_results.json": {
                "sha256": digest(results_json_path),
                "bytes": results_json_path.stat().st_size,
            },
            "dual_head_frozen_results.csv": {
                "sha256": digest(results_csv_path),
                "bytes": results_csv_path.stat().st_size,
            },
        },
    }
    evaluation_manifest_path.write_text(json.dumps(evaluation_manifest), encoding="utf-8")
    validation_report = {
        "schema_version": 1,
        "validation_id": f"{protocol_id}-validator-v1",
        "ok": True,
        "promotion_authorized": False,
        "evaluation_manifest": {"sha256": digest(evaluation_manifest_path)},
        "gates": {"checks": gate_checks, "passed": True},
        "panels": panels,
        "recomputation": {
            "all_per_head_scores_and_actions_recomputed": True,
            "all_combined_actions_recomputed": True,
            "csv_json_exact_copy_validated": True,
            "rows": 50,
            "unique_sample_ids": 50,
            "validator_cache": {
                "enabled": True,
                "schema_version": 1,
                "expected_rows": 50,
                "rebuild_requested": False,
                "write_error_details": [],
                **cache_statistics,
            },
        },
        "evidence": {
            "image_artifact_sha256": image_hash,
            "text_artifact_sha256": text_hash,
            "corpus_manifest_sha256": corpus_hash,
            "development_manifest_sha256": digest(development_manifest_path),
            "cache_entry_tree_sha256": cache_tree,
            "protocol_sha256": digest(protocol_path),
            "results_json": {"sha256": digest(results_json_path)},
            "results_csv": {"sha256": digest(results_csv_path)},
        },
    }
    validation_report_path.write_text(json.dumps(validation_report), encoding="utf-8")
    validation_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "promotion_authorized": False,
                "evaluation_manifest": {"sha256": digest(evaluation_manifest_path)},
                "validation_report": {
                    "sha256": digest(validation_report_path),
                    "bytes": validation_report_path.stat().st_size,
                },
                "validator": {"sha256": digest(validator_path)},
            }
        ),
        encoding="utf-8",
    )
    return evidence


class PairContractTests(unittest.TestCase):
    def test_pair_round_trip_splits_text_then_image_and_uses_or(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path = create_pair(root)
            pair = load_detector_pair(
                path, expected_target="synthetic", expected_base_feature_dim=2
            )
            # Text is malicious and image benign. Reversing the declared fused order
            # would produce the wrong per-head attribution.
            scored = score_pair(pair, np.asarray([2.0, 0.0, -2.0, 0.0]))
            self.assertEqual(scored["text_actions"], ["block"])
            self.assertEqual(scored["image_actions"], ["allow"])
            self.assertEqual(scored["combined_actions"], ["block"])
            self.assertEqual(
                manifest["runtime_detector_identity_sha256"],
                runtime_detector_identity_sha256(manifest["artifacts"]),
            )
            self.assertEqual(
                manifest["runtime_detector_identity_sha256"],
                detector_set_identity_sha256(
                    tuple(
                        (
                            name,
                            str(manifest["artifacts"][name]["sha256"]),
                            float(manifest["artifacts"][name]["review_threshold"]),
                        )
                        for name in ("text", "image")
                    )
                ),
            )
            self.assertEqual(
                manifest["pair_identity_sha256"], pair_identity_sha256(manifest)
            )

    def test_action_precedence_is_block_then_review_then_allow(self) -> None:
        self.assertEqual(
            head_action(0.7, block_threshold=0.6, review_threshold=0.5), "block"
        )
        self.assertEqual(
            head_action(0.55, block_threshold=0.6, review_threshold=0.5), "review"
        )
        self.assertEqual(combine_actions("allow", "review"), "review")
        self.assertEqual(combine_actions("review", "block"), "block")

    def test_manifest_hash_threshold_pooling_and_provenance_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path = create_pair(root)
            for mutate in (
                lambda value: value["artifacts"]["image"].__setitem__("sha256", "f" * 64),
                lambda value: value["artifacts"]["image"].__setitem__(
                    "review_threshold", 0.49
                ),
                lambda value: value["artifacts"]["image"].__setitem__(
                    "pooling", "text_tokens"
                ),
                lambda value: value["shared_provenance"].__setitem__(
                    "model_revision", "tampered"
                ),
            ):
                tampered = copy.deepcopy(manifest)
                mutate(tampered)
                with self.subTest(tampered=tampered):
                    with self.assertRaises(ValueError):
                        validate_pair_manifest_payload(tampered)
            artifact_bytes = (root / "image.npz").read_bytes()
            (root / "image.npz").write_bytes(artifact_bytes + b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_detector_pair(path)

    def test_build_rejects_channel_provenance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = detector("image_tokens")
            text = detector("text_tokens", model_id="other/model")
            image_path = root / "image.npz"
            text_path = root / "text.npz"
            save_detector_artifact(image_path, image)
            save_detector_artifact(text_path, text)
            with self.assertRaisesRegex(ValueError, "provenance"):
                build_pair_manifest(
                    target="synthetic",
                    image_path=image_path,
                    image_artifact=image,
                    image_review_threshold=0.5,
                    text_path=text_path,
                    text_artifact=text,
                    text_review_threshold=0.5,
                    output_directory=root,
                    training_identity_sha256_value="b" * 64,
                    corpus={
                        "manifest_sha256": "c" * 64,
                        "development_metadata_sha256": "d" * 64,
                        "regression_metadata_sha256": "e" * 64,
                    },
                )

    def test_artifact_block_threshold_requires_exact_float_equality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, path = create_pair(root)
            changed = copy.deepcopy(manifest)
            changed["artifacts"]["image"]["block_threshold"] = float(
                np.nextafter(np.float64(0.6), np.float64(1.0))
            )
            changed["pair_identity_sha256"] = pair_identity_sha256(changed)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact threshold mismatch"):
                load_detector_pair(path)

    def test_bound_one_shot_evidence_is_hash_checked_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_manifest, _ = create_pair(root)
            evidence = create_qualification_evidence(
                root,
                image_hash=str(base_manifest["artifacts"]["image"]["sha256"]),
                text_hash=str(base_manifest["artifacts"]["text"]["sha256"]),
                corpus_hash=str(base_manifest["corpus"]["manifest_sha256"]),
            )
            image = detector("image_tokens")
            text = detector("text_tokens")
            qualification = build_one_shot_qualification_binding(
                evidence_directory=evidence,
                pair_directory=root,
                image_artifact_sha256=str(
                    base_manifest["artifacts"]["image"]["sha256"]
                ),
                text_artifact_sha256=str(
                    base_manifest["artifacts"]["text"]["sha256"]
                ),
                corpus_manifest_sha256=str(base_manifest["corpus"]["manifest_sha256"]),
            )
            manifest = build_pair_manifest(
                target="synthetic",
                image_path=root / "image.npz",
                image_artifact=image,
                image_review_threshold=0.5,
                text_path=root / "text.npz",
                text_artifact=text,
                text_review_threshold=0.5,
                output_directory=root,
                training_identity_sha256_value="b" * 64,
                corpus=base_manifest["corpus"],
                qualification_evidence=qualification,
            )
            manifest_path = root / "qualified_pair.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            pair = load_detector_pair(manifest_path)
            self.assertTrue(pair.manifest["qualification_evidence"]["passed"])
            (evidence / "dual_head_frozen_results.csv").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "qualification evidence changed"):
                load_detector_pair(manifest_path)

    def test_one_shot_binding_rejects_protocol_and_cache_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_manifest, _ = create_pair(root)
            evidence = create_qualification_evidence(
                root,
                image_hash=str(base_manifest["artifacts"]["image"]["sha256"]),
                text_hash=str(base_manifest["artifacts"]["text"]["sha256"]),
                corpus_hash=str(base_manifest["corpus"]["manifest_sha256"]),
            )
            attempt_path = evidence / "attempt_started.json"
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            attempt["protocol_id"] = "mutated-protocol"
            attempt_path.write_text(json.dumps(attempt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "attempt marker"):
                build_one_shot_qualification_binding(
                    evidence_directory=evidence,
                    pair_directory=root,
                    image_artifact_sha256=str(
                        base_manifest["artifacts"]["image"]["sha256"]
                    ),
                    text_artifact_sha256=str(
                        base_manifest["artifacts"]["text"]["sha256"]
                    ),
                    corpus_manifest_sha256=str(
                        base_manifest["corpus"]["manifest_sha256"]
                    ),
                )

    def test_one_shot_binding_rejects_sequence_and_metadata_identity_mutations(
        self,
    ) -> None:
        mutations = {
            "sequence": (
                lambda result, _: result["results"][0].__setitem__("sequence", 2),
                "sequence must be exactly",
            ),
            "result identity": (
                lambda result, _: result["results"][0].__setitem__(
                    "sample_id", "substituted-sample"
                ),
                "authoritative metadata",
            ),
            "metadata bytes": (
                lambda _, evidence: (
                    evidence.parent
                    / "metadata"
                    / "final_metadata_v7.csv"
                ).write_text("tampered\n", encoding="utf-8"),
                "authoritative metadata hash",
            ),
        }
        for name, (mutate, message) in mutations.items():
            with self.subTest(mutation=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                base_manifest, _ = create_pair(root)
                evidence = create_qualification_evidence(
                    root,
                    image_hash=str(base_manifest["artifacts"]["image"]["sha256"]),
                    text_hash=str(base_manifest["artifacts"]["text"]["sha256"]),
                    corpus_hash=str(base_manifest["corpus"]["manifest_sha256"]),
                )
                result_path = evidence / "dual_head_frozen_results.json"
                result = json.loads(result_path.read_text(encoding="utf-8"))
                mutate(result, evidence)
                result_path.write_text(json.dumps(result), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    build_one_shot_qualification_binding(
                        evidence_directory=evidence,
                        pair_directory=root,
                        image_artifact_sha256=str(
                            base_manifest["artifacts"]["image"]["sha256"]
                        ),
                        text_artifact_sha256=str(
                            base_manifest["artifacts"]["text"]["sha256"]
                        ),
                        corpus_manifest_sha256=str(
                            base_manifest["corpus"]["manifest_sha256"]
                        ),
                    )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_manifest, _ = create_pair(root)
            evidence = create_qualification_evidence(
                root,
                image_hash=str(base_manifest["artifacts"]["image"]["sha256"]),
                text_hash=str(base_manifest["artifacts"]["text"]["sha256"]),
                corpus_hash=str(base_manifest["corpus"]["manifest_sha256"]),
            )
            report_path = evidence / "validation_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["recomputation"]["validator_cache"]["misses"] = 1
            report_path.write_text(json.dumps(report), encoding="utf-8")
            validation_manifest_path = evidence / "validation_manifest.json"
            validation_manifest = json.loads(
                validation_manifest_path.read_text(encoding="utf-8")
            )
            validation_manifest["validation_report"] = {
                "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "bytes": report_path.stat().st_size,
            }
            validation_manifest_path.write_text(
                json.dumps(validation_manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "validator cache contract"):
                build_one_shot_qualification_binding(
                    evidence_directory=evidence,
                    pair_directory=root,
                    image_artifact_sha256=str(
                        base_manifest["artifacts"]["image"]["sha256"]
                    ),
                    text_artifact_sha256=str(
                        base_manifest["artifacts"]["text"]["sha256"]
                    ),
                    corpus_manifest_sha256=str(
                        base_manifest["corpus"]["manifest_sha256"]
                    ),
                )


class ValidationSelectionTests(unittest.TestCase):
    def test_candidate_pairs_are_selected_on_validation_or_behavior(self) -> None:
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        image_led = np.asarray([0, 0, 0, 0, 1, 1, 0, 0], dtype=bool)
        text_led = np.asarray([0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
        image_candidates = [
            ChannelCandidate(
                candidate_id=0,
                channel="image",
                classifier=object(),
                validation_scores=np.asarray([0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 0.1, 0.1]),
                learning_rate=0.02,
                l2=0.1,
                positive_weight=1.0,
            )
        ]
        text_candidates = [
            ChannelCandidate(
                candidate_id=0,
                channel="text",
                classifier=object(),
                validation_scores=np.asarray([0.1, 0.2, 0.3, 0.4, 0.1, 0.1, 0.8, 0.9]),
                learning_rate=0.02,
                l2=0.1,
                positive_weight=1.0,
            )
        ]
        selected, image, text, candidates = select_validation_pair(
            image_candidates=image_candidates,
            text_candidates=text_candidates,
            validation_labels=labels,
            validation_image_led=image_led,
            validation_text_led=text_led,
            budget_allocations=((0, 0),),
            combined_false_positive_limit=0,
        )
        self.assertTrue(selected["validation_feasible"])
        self.assertEqual(selected["validation_false_positives"], 0)
        self.assertEqual(selected["validation_image_led_false_negatives"], 0)
        self.assertEqual(selected["validation_text_led_false_negatives"], 0)
        self.assertIs(image, image_candidates[0])
        self.assertIs(text, text_candidates[0])
        self.assertEqual(sum(bool(row["selected"]) for row in candidates), 1)


def schema4_synthetic_rows() -> tuple[
    list[dict[str, str]], dict[str, np.ndarray], list[dict[str, str]], dict[str, np.ndarray]
]:
    rows: list[dict[str, str]] = []
    image_features: list[list[float]] = []
    text_features: list[list[float]] = []
    for split_index, split in enumerate(("train", "validation", "test")):
        benign_ids: list[str] = []
        benign_hashes: list[str] = []
        for index in range(80):
            sample_id = f"{split}-benign-{index:03d}"
            image_hash = f"{split_index + 1:01x}{index:063x}"[-64:]
            benign_ids.append(sample_id)
            benign_hashes.append(image_hash)
            rows.append(
                {
                    "sample_id": sample_id,
                    "label_id": "0",
                    "split": split,
                    "group_id": f"{split}-benign",
                    "strategy": "official_bordair_benign_text_rendered_in_image",
                    "image_sha256": image_hash,
                    "paired_benign_sample_id": "",
                    "hard_negative": "1" if index >= 70 else "0",
                }
            )
            image_features.append([-2.0, -1.0])
            text_features.append([-2.0, -1.0])
        for index in range(80):
            rows.append(
                {
                    "sample_id": f"{split}-image-{index:03d}",
                    "label_id": "1",
                    "split": split,
                    "group_id": f"{split}-image-family-{index:03d}",
                    "strategy": "benign_text_full_injection",
                    "image_sha256": f"9{split_index:x}{index:062x}"[-64:],
                    "paired_benign_sample_id": "",
                    "hard_negative": "0",
                }
            )
            image_features.append([2.0, 1.0])
            text_features.append([-2.0, -1.0])
        for index in range(20):
            rows.append(
                {
                    "sample_id": f"{split}-text-{index:03d}",
                    "label_id": "1",
                    "split": split,
                    "group_id": f"{split}-text-family-{index:03d}",
                    "strategy": "malicious_text_benign_image_counterfactual",
                    "image_sha256": benign_hashes[index],
                    "paired_benign_sample_id": benign_ids[index],
                    "hard_negative": "0",
                }
            )
            image_features.append([-2.0, -1.0])
            text_features.append([2.0, 1.0])
    regression_rows: list[dict[str, str]] = []
    for index in range(10):
        regression_rows.append(
            {
                "sample_id": f"regression-{index:03d}",
                "label_id": "1",
                "strategy": "benign_text_full_injection",
            }
        )
    return (
        rows,
        {
            "image_tokens": np.asarray(image_features),
            "text_tokens": np.asarray(text_features),
        },
        regression_rows,
        {
            "image_tokens": np.tile(np.asarray([[2.0, 1.0]]), (10, 1)),
            "text_tokens": np.tile(np.asarray([[-2.0, -1.0]]), (10, 1)),
        },
    )


def write_training_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        "sample_id",
        "label_id",
        "split",
        "group_id",
        "attack_style",
        "strategy",
        "source",
        "prompt_text",
        "image_path",
        "image_text",
        "render_style",
        "image_sha256",
        "paired_benign_sample_id",
        "hard_negative",
        "hard_negative_category",
    )
    normalized = []
    for index, row in enumerate(rows):
        normalized.append(
            {
                "sample_id": row["sample_id"],
                "label_id": row["label_id"],
                "split": row.get("split", "regression"),
                "group_id": row.get("group_id", f"group-{index}"),
                "attack_style": row.get("attack_style", "synthetic"),
                "strategy": row.get("strategy", "benign_text_full_injection"),
                "source": row.get("source", "synthetic"),
                "prompt_text": row.get("prompt_text", f"prompt {index}"),
                "image_path": row.get("image_path", f"images/{index}.png"),
                "image_text": row.get("image_text", "synthetic"),
                "render_style": row.get("render_style", "synthetic"),
                "image_sha256": row.get("image_sha256", f"{index + 1:064x}"[-64:]),
                "paired_benign_sample_id": row.get("paired_benign_sample_id", ""),
                "hard_negative": row.get("hard_negative", "0"),
                "hard_negative_category": row.get("hard_negative_category", ""),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)


class DurableTrainingTests(unittest.TestCase):
    def test_training_emits_two_v1_artifacts_and_canonical_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            development_rows, development_features, regression_rows, regression_features = (
                schema4_synthetic_rows()
            )
            manifest_path = root / "corpus_manifest_v7.json"
            development_path = root / "development_metadata_v7.csv"
            regression_path = root / "regression_metadata_v7.csv"
            manifest_path.write_text("{}\n", encoding="utf-8")
            write_training_metadata(development_path, development_rows)
            write_training_metadata(regression_path, regression_rows)
            result_root = root / "training_v7" / "synthetic" / "dual_or"
            summary = train_dual_pair(
                target="synthetic",
                development_rows=development_rows,
                regression_rows=regression_rows,
                development_features=development_features,
                regression_features=regression_features,
                provenance={
                    "model_family": "generic",
                    "model_id": "synthetic/model",
                    "model_revision": "revision",
                    "tokenizer_revision": "tokenizer",
                    "preprocessing_sha256": DIGEST,
                    "layer": -1,
                },
                run_root=root,
                corpus_manifest_path=manifest_path,
                development_metadata_path=development_path,
                regression_metadata_path=regression_path,
                result_root=result_root,
                image_positive_weights=(1.0,),
                text_positive_weights=(1.0,),
                learning_rates=(0.1,),
                l2_values=(0.001,),
                budget_allocations=((0, 2),),
                epochs=100,
            )
            self.assertTrue(summary["acceptance"]["passed"])
            self.assertEqual(summary["validation_metrics"]["false_positives"], 0)
            self.assertEqual(summary["test_metrics"]["false_negatives"], 0)
            self.assertEqual(
                summary["regression_metrics"]["image_head_image_led_false_negatives"],
                0,
            )
            pair = load_detector_pair(
                result_root / "detector_pair_manifest.json",
                expected_target="synthetic",
                expected_base_feature_dim=2,
            )
            self.assertEqual(pair.image_artifact.pooling, "image_tokens")
            self.assertEqual(pair.text_artifact.pooling, "text_tokens")
            self.assertIn("runtime_detector_identity_sha256", pair.manifest)
            evidence = validate_development_qualification_evidence(
                summary=summary,
                summary_path=result_root / "training_summary.json",
                pair=pair,
                run_paths=RunPaths(root),
            )
            self.assertTrue(evidence["passed"])
            self.assertEqual(evidence["metrics"]["internal_test"]["false_negatives"], 0)

            tampered = copy.deepcopy(summary)
            tampered["acceptance"]["passed"] = False
            with self.assertRaisesRegex(ValueError, "acceptance booleans"):
                validate_development_qualification_evidence(
                    summary=tampered,
                    summary_path=result_root / "training_summary.json",
                    pair=pair,
                    run_paths=RunPaths(root),
                )

            prediction_path = result_root / "validation_predictions.csv"
            original_prediction_bytes = prediction_path.read_bytes()
            prediction_lines = prediction_path.read_text(encoding="utf-8").splitlines()
            prediction_path.write_text(
                "\n".join([*prediction_lines, prediction_lines[-1]]) + "\n",
                encoding="utf-8",
            )
            appended = copy.deepcopy(summary)
            appended_binding = appended["development_qualification_evidence"]
            appended_binding["predictions"]["validation"]["sha256"] = (
                hashlib.sha256(prediction_path.read_bytes()).hexdigest()
            )
            appended_pair = type(pair)(
                manifest=copy.deepcopy(pair.manifest),
                manifest_path=pair.manifest_path,
                image_artifact=pair.image_artifact,
                text_artifact=pair.text_artifact,
                paths=pair.paths,
            )
            appended_pair.manifest["development_qualification_evidence"] = (
                copy.deepcopy(appended_binding)
            )
            with self.assertRaisesRegex(ValueError, "actual row count"):
                validate_development_qualification_evidence(
                    summary=appended,
                    summary_path=result_root / "training_summary.json",
                    pair=appended_pair,
                    run_paths=RunPaths(root),
                )
            prediction_path.write_bytes(original_prediction_bytes)

            snapshot = result_root / "qualification_features_v1.npz"
            snapshot.write_bytes(snapshot.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "snapshot binding"):
                validate_development_qualification_evidence(
                    summary=summary,
                    summary_path=result_root / "training_summary.json",
                    pair=pair,
                    run_paths=RunPaths(root),
                )


class EvaluationAndPromotionTamperTests(unittest.TestCase):
    def test_qualification_cache_locator_is_used_by_both_row_producers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, manifest_path = create_pair(root)
            pair = load_detector_pair(manifest_path)
            subject = "a" * 64
            panel = "regression"
            sample_id = "TI-00001"
            metadata = {
                "sample_id": sample_id,
                "label_id": "1",
                "prompt_text": "prompt",
                "image_path": "image.png",
                "image_sha256": "1" * 64,
            }
            cache_path = root / "cache" / panel / f"{sample_id}.npz"
            cache_path.parent.mkdir(parents=True)
            cache_path.write_bytes(b"cache")

            class Provider:
                @staticmethod
                def embed(_request: object) -> np.ndarray:
                    return np.asarray([2.0, 0.0, -2.0, 0.0])

            with patch(
                "aegis_research.bordair_dual_evaluation.base.display_path",
                side_effect=AssertionError("qualification locator must not use display_path"),
            ):
                rows = score_dual_panels(
                    target="synthetic",
                    panels={
                        "regression": [metadata],
                        "text_led": [],
                        "external_benign": [],
                    },
                    pair=pair,
                    provider=Provider(),
                    metadata_root=root,
                    cache_root=root / "cache",
                    qualification_subject=subject,
                )
            expected = qualification_cache_locator(subject, panel, sample_id)
            self.assertEqual(rows[0]["cache_path"], expected)

            with patch(
                "aegis_research.bordair_dual_evaluation_validation.base.display_path",
                side_effect=AssertionError("qualification locator must not use display_path"),
            ):
                recomputed = validate_recomputed_row(
                    reported=rows[0],
                    metadata=metadata,
                    panel=panel,
                    pair=pair,
                    embedding=Provider.embed(object()),
                    cache_path=cache_path,
                    qualification_subject=subject,
                )
            self.assertEqual(recomputed["cache_path"], expected)

    def test_dual_acceptance_allows_two_benign_blocks_per_frozen_panel(self) -> None:
        rows: list[dict[str, object]] = []

        def add_row(
            panel: str,
            label: int,
            action: str,
            sample_id: str,
        ) -> None:
            rows.append(
                {
                    "panel": panel,
                    "label_id": label,
                    "combined_action": action,
                    "image_head_action": action,
                    "text_head_action": "allow",
                    "image_head_score": 0.9 if action == "block" else 0.1,
                    "text_head_score": 0.1,
                    "accepted": (
                        action == "block" if label == 1 else action != "block"
                    ),
                    "sample_id": sample_id,
                }
            )

        for index in range(10):
            add_row("regression", 1, "block", f"m-{index}")
            add_row(
                "regression",
                0,
                "block" if index < 2 else "allow",
                f"b-{index}",
            )
            add_row("text_led", 1, "block", f"t-{index}")
        for index in range(20):
            add_row(
                "external_benign",
                0,
                "block" if index < 2 else "allow",
                f"e-{index}",
            )

        report = acceptance_report(rows, {"passed": True})
        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["all_malicious_rows_accepted"])
        self.assertEqual(report["panels"]["regression"]["benign_blocked"], 2)
        self.assertEqual(report["panels"]["external_benign"]["benign_blocked"], 2)

        benign_review = copy.deepcopy(rows)
        reviewed = next(
            row
            for row in benign_review
            if row["panel"] == "regression"
            and row["label_id"] == 0
            and row["sample_id"] == "b-2"
        )
        reviewed.update(
            {
                "combined_action": "review",
                "image_head_action": "review",
                "image_head_score": 0.55,
                "accepted": False,
            }
        )
        self.assertTrue(acceptance_report(benign_review, {"passed": True})["passed"])

        three_fixed_blocks = copy.deepcopy(rows)
        third_fixed = next(
            row
            for row in three_fixed_blocks
            if row["panel"] == "regression"
            and row["label_id"] == 0
            and row["sample_id"] == "b-2"
        )
        third_fixed.update(
            {
                "combined_action": "block",
                "image_head_action": "block",
                "image_head_score": 0.9,
                "accepted": False,
            }
        )
        fixed_report = acceptance_report(three_fixed_blocks, {"passed": True})
        self.assertFalse(fixed_report["passed"])
        self.assertFalse(
            fixed_report["checks"]["regression_fewer_than_three_benign_blocked"]
        )

        malicious_miss = copy.deepcopy(rows)
        first_malicious = next(
            row
            for row in malicious_miss
            if row["panel"] == "regression" and row["label_id"] == 1
        )
        first_malicious.update(
            {
                "combined_action": "allow",
                "image_head_action": "allow",
                "image_head_score": 0.1,
                "accepted": False,
            }
        )
        malicious_report = acceptance_report(malicious_miss, {"passed": True})
        self.assertFalse(malicious_report["passed"])
        self.assertFalse(
            malicious_report["checks"]["all_malicious_rows_accepted"]
        )

    def test_frozen_cache_report_requires_exact_50_hit_zero_mutation_contract(self) -> None:
        report = {
            "enabled": True,
            "schema_version": 1,
            "expected_rows": 50,
            "rebuild_requested": False,
            "cache_only": True,
            "write_error_details": [],
            "hits": 50,
            "misses": 0,
            "invalid_entries": 0,
            "rebuilds": 0,
            "writes": 0,
            "write_errors": 0,
            "delegate_calls": 0,
        }
        self.assertEqual(validate_exact_frozen_cache_report(report)["hits"], 50)
        for name, value in (
            ("hits", 49),
            ("misses", 1),
            ("invalid_entries", 1),
            ("rebuilds", 1),
            ("writes", 1),
            ("write_errors", 1),
            ("delegate_calls", 1),
            ("expected_rows", 49),
            ("rebuild_requested", True),
            ("cache_only", False),
        ):
            mutated = dict(report)
            mutated[name] = value
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_exact_frozen_cache_report(mutated)

    def test_recomputed_result_rejects_action_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, manifest_path = create_pair(root)
            pair = load_detector_pair(manifest_path)
            cache_path = root / "row.npz"
            cache_path.write_bytes(b"cache")
            metadata = {
                "sample_id": "row",
                "label_id": "1",
                "prompt_text": "prompt",
                "image_path": "image.png",
                "image_sha256": "1" * 64,
            }
            scored = score_pair(pair, np.asarray([2.0, 0.0, -2.0, 0.0]))
            image_entry = pair.manifest["artifacts"]["image"]
            text_entry = pair.manifest["artifacts"]["text"]
            reported = {
                "sample_id": "row",
                "panel": "text_led",
                "label_id": 1,
                "expected_action": "block",
                "prompt_text": "prompt",
                "image_path": "image.png",
                "image_sha256": "1" * 64,
                "target": "synthetic",
                "image_head_score": float(scored["image_scores"][0]),
                "text_head_score": float(scored["text_scores"][0]),
                "image_head_block_threshold": image_entry["block_threshold"],
                "image_head_review_threshold": image_entry["review_threshold"],
                "text_head_block_threshold": text_entry["block_threshold"],
                "text_head_review_threshold": text_entry["review_threshold"],
                "image_head_action": scored["image_actions"][0],
                "text_head_action": scored["text_actions"][0],
                "combined_action": "allow",
                "accepted": True,
                "caller_text_sha256": __import__("hashlib").sha256(b"prompt").hexdigest(),
                "image_artifact_sha256": image_entry["sha256"],
                "text_artifact_sha256": text_entry["sha256"],
                "pair_identity_sha256": pair.manifest["pair_identity_sha256"],
                "runtime_detector_identity_sha256": pair.manifest[
                    "runtime_detector_identity_sha256"
                ],
                "cache_path": display_path(cache_path),
                "cache_sha256": __import__("hashlib").sha256(b"cache").hexdigest(),
            }
            with self.assertRaisesRegex(ValueError, "combined_action"):
                validate_recomputed_row(
                    reported=reported,
                    metadata=metadata,
                    panel="text_led",
                    pair=pair,
                    embedding=np.asarray([2.0, 0.0, -2.0, 0.0]),
                    cache_path=cache_path,
                )

    def test_recomputed_benign_review_is_an_exact_action_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, manifest_path = create_pair(root)
            pair = load_detector_pair(manifest_path)
            cache_path = root / "row.npz"
            cache_path.write_bytes(b"cache")
            metadata = {
                "sample_id": "row",
                "label_id": "0",
                "prompt_text": "prompt",
                "image_path": "image.png",
                "image_sha256": "1" * 64,
            }
            embedding = np.asarray([-1.0, 0.0, 0.1, 0.0])
            scored = score_pair(pair, embedding)
            image_entry = pair.manifest["artifacts"]["image"]
            text_entry = pair.manifest["artifacts"]["text"]
            self.assertEqual(scored["combined_actions"][0], "review")
            reported = {
                "sample_id": "row",
                "panel": "regression",
                "label_id": 0,
                "expected_action": "allow",
                "prompt_text": "prompt",
                "image_path": "image.png",
                "image_sha256": "1" * 64,
                "target": "synthetic",
                "image_head_score": float(scored["image_scores"][0]),
                "text_head_score": float(scored["text_scores"][0]),
                "image_head_block_threshold": image_entry["block_threshold"],
                "image_head_review_threshold": image_entry["review_threshold"],
                "text_head_block_threshold": text_entry["block_threshold"],
                "text_head_review_threshold": text_entry["review_threshold"],
                "image_head_action": scored["image_actions"][0],
                "text_head_action": scored["text_actions"][0],
                "combined_action": scored["combined_actions"][0],
                "accepted": False,
                "caller_text_sha256": hashlib.sha256(b"prompt").hexdigest(),
                "image_artifact_sha256": image_entry["sha256"],
                "text_artifact_sha256": text_entry["sha256"],
                "pair_identity_sha256": pair.manifest["pair_identity_sha256"],
                "runtime_detector_identity_sha256": pair.manifest[
                    "runtime_detector_identity_sha256"
                ],
                "cache_path": display_path(cache_path),
                "cache_sha256": hashlib.sha256(b"cache").hexdigest(),
            }
            recomputed = validate_recomputed_row(
                reported=reported,
                metadata=metadata,
                panel="regression",
                pair=pair,
                embedding=embedding,
                cache_path=cache_path,
            )
            self.assertEqual(recomputed["combined_action"], "review")
            self.assertFalse(recomputed["accepted"])

    def test_promotion_reuses_one_shot_binding_without_evaluation_or_cache_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_manifest, _ = create_pair(root)
            evidence = create_qualification_evidence(
                root,
                image_hash=str(base_manifest["artifacts"]["image"]["sha256"]),
                text_hash=str(base_manifest["artifacts"]["text"]["sha256"]),
                corpus_hash=str(base_manifest["corpus"]["manifest_sha256"]),
            )
            qualification = build_one_shot_qualification_binding(
                evidence_directory=evidence,
                pair_directory=root,
                image_artifact_sha256=str(
                    base_manifest["artifacts"]["image"]["sha256"]
                ),
                text_artifact_sha256=str(
                    base_manifest["artifacts"]["text"]["sha256"]
                ),
                corpus_manifest_sha256=str(base_manifest["corpus"]["manifest_sha256"]),
            )
            image = detector("image_tokens")
            text = detector("text_tokens")
            manifest = build_pair_manifest(
                target="synthetic",
                image_path=root / "image.npz",
                image_artifact=image,
                image_review_threshold=0.5,
                text_path=root / "text.npz",
                text_artifact=text,
                text_review_threshold=0.5,
                output_directory=root,
                training_identity_sha256_value="b" * 64,
                corpus=base_manifest["corpus"],
                qualification_evidence=qualification,
            )
            manifest_path = root / "qualified_pair.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            pair = load_detector_pair(manifest_path)
            paths = RunPaths(root)
            summary_path = paths.training_directory / "synthetic/dual_or/training_summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text("{}", encoding="utf-8")
            runtime_config = root / "runtime.json"
            runtime_config.write_text("{}", encoding="utf-8")
            with (
                patch(
                    "aegis_research.bordair_dual_promotion.base.validate_tracked_v7_corpus"
                ),
                patch(
                    "aegis_research.bordair_dual_promotion.validate_dual_training_summary",
                    return_value=({"acceptance": {"passed": True}}, pair),
                ),
                patch(
                    "aegis_research.bordair_dual_promotion.validate_runtime_detector_config",
                    return_value={"runtime_detector_identity_sha256": manifest["runtime_detector_identity_sha256"]},
                ),
            ):
                result = validate_dual_promotion_candidate(
                    target="synthetic",
                    run_paths=paths,
                    evaluation_dir=root / "must-not-exist-evaluation",
                    training_dir=paths.training_directory,
                    manifest_path=root / "corpus.json",
                    final_metadata_path=root / "must-not-open-final.csv",
                    text_led_metadata_path=root / "must-not-open-text-led.csv",
                    external_benign_metadata_path=root / "must-not-open-external.csv",
                    cache_root=root / "must-not-open-cache",
                    runtime_config=runtime_config,
                )
            self.assertTrue(result["passed"])
            self.assertFalse(result["frozen_panels_rerun"])
            self.assertEqual(
                result["one_shot_qualification_evidence"], qualification
            )

    def test_runtime_config_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, manifest_path = create_pair(root)
            pair = load_detector_pair(manifest_path)
            assets = root / "assets"
            assets.mkdir()
            shutil.copyfile(root / "image.npz", assets / "image.npz")
            shutil.copyfile(root / "text.npz", assets / "text.npz")
            config_path = root / "config.json"
            provider_options = {
                "environment_overrides": False,
                "local_files_only": True,
                "model_id": "synthetic/model",
                "model_revision": "revision",
                "tokenizer_revision": "tokenizer",
                "layer": -1,
                "pooling": "text_image_tokens",
                "feature_dim": 4,
            }
            config = {
                "schema_version": 2,
                "name": "synthetic",
                "base_dir": "assets",
                "target_profile": "synthetic",
                "detector": {
                    "mode": "or",
                    "heads": [
                        {
                            "name": "text",
                            "artifact": "text.npz",
                            "review_threshold": 0.5,
                        },
                        {
                            "name": "image",
                            "artifact": "image.npz",
                            "review_threshold": 0.5,
                        },
                    ],
                }
                ,
                "provider": "synthetic:Provider",
                "provider_options": provider_options,
                "policy": {
                    "block_threshold": None,
                    "review_threshold": None,
                    "review_margin": None,
                },
                "server": {},
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            profile_heads = tuple(
                SimpleNamespace(
                    name=name,
                    detector=f"{name}.npz",
                    detector_sha256=manifest["artifacts"][name]["sha256"],
                    review_threshold=0.5,
                )
                for name in ("text", "image")
            )
            profile = SimpleNamespace(
                name="synthetic",
                detector_mode="or",
                detector_heads=profile_heads,
                provider="synthetic:Provider",
                provider_options=provider_options,
                model_family="generic",
                detector_identity_sha256=manifest[
                    "runtime_detector_identity_sha256"
                ],
            )
            with patch("AEGIS.target_profiles.get_target_profile", return_value=profile):
                evidence = validate_runtime_detector_config(
                    config_path=config_path, pair=pair
                )
            self.assertEqual(
                evidence["runtime_detector_identity_sha256"],
                manifest["runtime_detector_identity_sha256"],
            )
            config["detector"]["heads"][0]["review_threshold"] = 0.49
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "review threshold"):
                with patch(
                    "AEGIS.target_profiles.get_target_profile", return_value=profile
                ):
                    validate_runtime_detector_config(
                        config_path=config_path, pair=pair
                    )


class QwenDualCliDispatchTests(unittest.TestCase):
    def test_tracked_cli_routes_qwen_2048_primitives_to_dual_training_only(self) -> None:
        development_paths = {
            "text_tokens": Path("development-text.npz"),
            "image_tokens": Path("development-image.npz"),
        }
        regression_paths = {
            "text_tokens": Path("regression-text.npz"),
            "image_tokens": Path("regression-image.npz"),
        }
        provenance = {
            "model_family": "qwen25_vl",
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "tokenizer_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "preprocessing_sha256": DIGEST,
            "layer": -1,
        }

        def fake_extract(target, panel, *args, **kwargs):
            self.assertEqual(target, "qwen25vl3b")
            if panel == "development":
                return development_paths
            if panel == "regression":
                return regression_paths
            self.fail(f"unexpected/frozen panel access: {panel}")

        def fake_load(path):
            pooling = "text_tokens" if "text" in path.name else "image_tokens"
            return np.zeros((1, 2048), dtype=np.float64), {
                **provenance,
                "pooling": pooling,
            }

        argv = [
            "train_bordair_detector.py",
            "qwen25vl3b",
            "--artifact-mode",
            "dual_or",
            "--candidate-poolings",
            "text_tokens",
            "image_tokens",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(training_workflow, "validate_run", return_value={"ok": True}),
            patch.object(
                training_workflow,
                "read_metadata",
                side_effect=([{"sample_id": "development"}], [{"sample_id": "regression"}]),
            ),
            patch.object(training_workflow, "extract_panel", side_effect=fake_extract) as extract,
            patch.object(training_workflow, "load_embeddings", side_effect=fake_load),
            patch.object(training_workflow, "clear_qwen_runtime_cache"),
            patch(
                "aegis_research.bordair_dual_training.train_dual_pair"
            ) as train_dual,
        ):
            training_workflow.main()
        self.assertEqual(
            [item.args[1] for item in extract.call_args_list],
            ["development", "regression"],
        )
        kwargs = train_dual.call_args.kwargs
        self.assertEqual(kwargs["target"], "qwen25vl3b")
        self.assertEqual(
            kwargs["development_features"]["text_tokens"].shape, (1, 2048)
        )
        self.assertEqual(
            kwargs["development_features"]["image_tokens"].shape, (1, 2048)
        )
        self.assertEqual(kwargs["regression_features"]["text_tokens"].shape, (1, 2048))
        self.assertEqual(set(kwargs["development_features"]), {"text_tokens", "image_tokens"})
        self.assertIsNone(kwargs["qualification_evidence_directory"])


if __name__ == "__main__":
    unittest.main()
