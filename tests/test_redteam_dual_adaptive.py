from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from deliverables.red_teaming import evaluate_adaptive as adaptive


class _PairFixture:
    base_feature_dim = 2
    manifest = {
        "shared_provenance": {
            "model_family": "generic",
            "model_id": "fixture/model",
            "model_revision": "revision",
            "tokenizer_revision": "tokenizer",
            "preprocessing_sha256": "a" * 64,
            "layer": -1,
        }
    }


class DualAdaptiveEvaluatorTests(unittest.TestCase):
    def test_or_precedence_and_runtime_decisive_tie_break(self) -> None:
        decision = adaptive._dual_decision(
            text_score=0.75,
            image_score=0.90,
            text_block=0.70,
            text_review=0.50,
            image_block=0.80,
            image_review=0.60,
            text_pooling="text_tokens",
            image_pooling="image_tokens",
        )
        self.assertEqual(decision["recommended_action"], "block")
        self.assertEqual(decision["decisive_head"], "image")
        self.assertEqual(
            decision["head_decisions"]["text"]["recommended_action"], "block"
        )
        self.assertEqual(
            decision["head_decisions"]["image"]["recommended_action"], "block"
        )

        lexical_tie = adaptive._dual_decision(
            text_score=0.75,
            image_score=0.75,
            text_block=0.50,
            text_review=0.25,
            image_block=0.50,
            image_review=0.25,
            text_pooling="text_tokens",
            image_pooling="image_tokens",
        )
        self.assertEqual(lexical_tie["decisive_head"], "image")

    def test_score_row_is_bound_to_recomputed_heads_and_identities(self) -> None:
        decision = adaptive._dual_decision(
            text_score=0.75,
            image_score=0.20,
            text_block=0.70,
            text_review=0.50,
            image_block=0.80,
            image_review=0.60,
            text_pooling="text_tokens",
            image_pooling="image_tokens",
        )
        manifest = {
            "runtime_detector_identity_sha256": "b" * 64,
            "pair_identity_sha256": "c" * 64,
        }
        row = {
            "detector_mode": "dual_or",
            "risk_score": str(decision["risk_score"]),
            "detector_threshold": str(decision["threshold"]),
            "review_threshold": str(decision["review_threshold"]),
            "verdict": str(decision["verdict"]),
            "uncertain": str(decision["uncertain"]),
            "recommended_action": str(decision["recommended_action"]),
            "decisive_head": str(decision["decisive_head"]),
            "runtime_detector_identity_sha256": "b" * 64,
            "pair_identity_sha256": "c" * 64,
        }
        for name in ("text", "image"):
            head = decision["head_decisions"][name]
            row[f"{name}_risk_score"] = str(head["risk_score"])
            row[f"{name}_block_threshold"] = str(head["threshold"])
            row[f"{name}_review_threshold"] = str(head["review_threshold"])
            row[f"{name}_action"] = str(head["recommended_action"])
            row[f"{name}_pooling"] = str(head["pooling"])

        adaptive._validate_dual_score_row(row, decision, manifest)
        row["image_action"] = "block"
        with self.assertRaisesRegex(ValueError, "image_action"):
            adaptive._validate_dual_score_row(row, decision, manifest)

    def test_fused_bundle_validates_both_primitive_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez(
                path,
                sample_ids=np.asarray(["v1", "v2"]),
                feature_source=np.asarray(["fixture"]),
                text_embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                image_embeddings=np.asarray([[5.0, 6.0], [7.0, 8.0]]),
                attribution_features=np.zeros((2, 1)),
                model_family=np.asarray(["generic"]),
                model_id=np.asarray(["fixture/model"]),
                model_revision=np.asarray(["revision"]),
                tokenizer_revision=np.asarray(["tokenizer"]),
                preprocessing_sha256=np.asarray(["a" * 64]),
                layer=np.asarray([-1]),
                pooling=np.asarray(["text_image_tokens"]),
                id_column=np.asarray(["variant_id"]),
            )
            provenance, ids, fused = adaptive._dual_feature_snapshot(
                path, _PairFixture()
            )
        self.assertEqual(ids, ["v1", "v2"])
        self.assertEqual(fused.shape, (2, 4))
        self.assertEqual(provenance["text_pooling"], "text_tokens")
        self.assertEqual(provenance["image_pooling"], "image_tokens")


if __name__ == "__main__":
    unittest.main()
