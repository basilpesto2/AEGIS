from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
PIPELINE = REPOSITORY / "deliverables" / "pipeline"
SCRIPTS = PIPELINE / "scripts"
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(SCRIPTS))

from aegis_research.signals import build_feature_views  # noqa: E402
import score_feature_bundle  # noqa: E402
from deliverables.ablation_report import run_ablation as ablation  # noqa: E402
from deliverables.red_teaming import evaluate_adaptive as adaptive  # noqa: E402


class StrictLegacyAblationImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.detector = self.root / "legacy_detector.npz"
        self.features = self.root / "features.npz"
        self.scores = self.root / "scores.csv"
        self.summary = self.root / "summary.json"
        self._build_evidence()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build_evidence(self) -> None:
        revision = "r" * 40
        preprocessing = "a" * 64
        np.savez(
            self.detector,
            weights=np.asarray([0.7, -0.3], dtype=np.float64),
            mean=np.zeros(2, dtype=np.float64),
            scale=np.ones(2, dtype=np.float64),
            bias=np.asarray([0.1], dtype=np.float64),
            threshold=np.asarray([0.6], dtype=np.float64),
            source=np.asarray(["legacy-fixture"]),
            model_family=np.asarray(["generic"]),
            model_id=np.asarray(["fixture/model"]),
            model_revision=np.asarray([revision]),
            tokenizer_revision=np.asarray([revision]),
            preprocessing_sha256=np.asarray([preprocessing]),
            pooling=np.asarray(["image_tokens"]),
            layer=np.asarray([-1], dtype=np.int64),
        )
        ids = ["a-1", "a-2", "b-1", "b-2"]
        text = np.asarray(
            [[0.0, 0.0], [0.2, 0.1], [0.4, -0.2], [0.1, 0.3]],
            dtype=np.float64,
        )
        image = np.asarray(
            [[0.3, 0.1], [0.1, 0.5], [0.8, -0.2], [0.2, 0.6]],
            dtype=np.float64,
        )
        attribution = np.zeros((4, 1), dtype=np.float64)
        provenance = {
            "model_family": "generic",
            "model_id": "fixture/model",
            "model_revision": revision,
            "tokenizer_revision": revision,
            "preprocessing_sha256": preprocessing,
            "layer": -1,
            "pooling": "image_tokens",
        }
        np.savez(
            self.features,
            sample_ids=np.asarray(ids),
            text_embeddings=text,
            image_embeddings=image,
            attribution_features=attribution,
            feature_source=np.asarray(["legacy-fixture-features"]),
            id_column=np.asarray(["variant_id"]),
            **{
                key: np.asarray([value], dtype=np.int64)
                if key == "layer"
                else np.asarray([value])
                for key, value in provenance.items()
            },
        )
        metadata = [
            {
                "base_sample_id": base,
                "query_index": str(query),
                "variant_type": "original" if query == 1 else "variant",
                "variant_id": sample_id,
                "label_id": "1",
            }
            for base, query, sample_id in (
                ("a", 1, "a-1"),
                ("a", 2, "a-2"),
                ("b", 1, "b-1"),
                ("b", 2, "b-2"),
            )
        ]
        score_rows, _ = score_feature_bundle._score_legacy_single(
            rows=metadata,
            views=build_feature_views(text, image, attribution),
            feature_source="legacy-fixture-features",
            bundle_provenance=provenance,
            detector_path=self.detector,
        )
        with self.scores.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(score_rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(score_rows)
        payload = adaptive._evaluate_legacy(
            argparse.Namespace(
                detector=self.detector,
                features=self.features,
                scores=self.scores,
                threshold=None,
                threshold_kind="detector_artifact_block_threshold",
                traffic_mode="shadow",
            ),
            (1, 2),
        )
        self.summary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _args(self, summary: Path | None = None) -> argparse.Namespace:
        return argparse.Namespace(
            adaptive_summary=summary or self.summary,
            adaptive_scores=self.scores,
            adaptive_detector=self.detector,
            adaptive_pair_manifest=None,
            adaptive_features=self.features,
        )

    def _mutated_summary(self, mutate) -> Path:
        payload = json.loads(self.summary.read_text(encoding="utf-8"))
        mutate(payload)
        output = self.root / "mutated_summary.json"
        output.write_text(json.dumps(payload), encoding="utf-8")
        return output

    def test_current_schema_three_evidence_recomputes(self) -> None:
        rows, evidence = ablation._load_legacy_adaptive_evidence(self._args())
        self.assertEqual([row["query_budget"] for row in rows], [1, 2])
        self.assertEqual(evidence["detector_mode"], "legacy_single")

    def test_rejects_schema_downgrade(self) -> None:
        mutated = self._mutated_summary(
            lambda payload: payload.__setitem__("schema_version", 2)
        )
        with self.assertRaisesRegex(ValueError, "schema 3"):
            ablation._load_legacy_adaptive_evidence(self._args(mutated))

    def test_rejects_wrong_oracle_method(self) -> None:
        mutated = self._mutated_summary(
            lambda payload: payload.__setitem__("method", "lowest_score_without_order")
        )
        with self.assertRaisesRegex(ValueError, "method"):
            ablation._load_legacy_adaptive_evidence(self._args(mutated))

    def test_rejects_mutated_summary_metric(self) -> None:
        def mutate(payload: dict[str, object]) -> None:
            payload["summary"][0]["malicious_recall"] = 0.123

        mutated = self._mutated_summary(mutate)
        with self.assertRaisesRegex(ValueError, "summary row 0 differs"):
            ablation._load_legacy_adaptive_evidence(self._args(mutated))

    def test_rejects_mutated_oracle_selection(self) -> None:
        def mutate(payload: dict[str, object]) -> None:
            payload["selections"][0]["selected_variant_id"] = "fabricated"

        mutated = self._mutated_summary(mutate)
        with self.assertRaisesRegex(ValueError, "selections row 0 differs"):
            ablation._load_legacy_adaptive_evidence(self._args(mutated))

    def test_rejects_mutated_score_even_when_hash_is_redeclared(self) -> None:
        with self.scores.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["risk_score"] = "0.0"
        with self.scores.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

        def mutate(payload: dict[str, object]) -> None:
            payload["provenance"]["scores_sha256"] = ablation._sha256(self.scores)

        mutated = self._mutated_summary(mutate)
        with self.assertRaisesRegex(ValueError, "differs from recomputation"):
            ablation._load_legacy_adaptive_evidence(self._args(mutated))

    def test_rejects_feature_id_misalignment_with_redeclared_hash(self) -> None:
        with self.scores.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["variant_id"] = "not-in-feature-bundle"
        with self.scores.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

        def mutate(payload: dict[str, object]) -> None:
            payload["provenance"]["scores_sha256"] = ablation._sha256(self.scores)

        mutated = self._mutated_summary(mutate)
        with self.assertRaisesRegex(ValueError, "variant IDs"):
            ablation._load_legacy_adaptive_evidence(self._args(mutated))


if __name__ == "__main__":
    unittest.main()
