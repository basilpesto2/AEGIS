from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.evaluation import predict_from_threshold, run_logistic_validation_grid
from AEGIS.io import align_embeddings_with_metadata, load_embeddings, load_metadata
from AEGIS.labels import coerce_binary_labels
from AEGIS.logistic import LogisticRegressionNumpy
from AEGIS.metrics import average_precision, detection_report
from AEGIS.signals import binary_entropy, cross_modal_consistency_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-modal consistency, uncertainty triage, and modality attribution."
    )
    parser.add_argument("--text-embeddings", required=True)
    parser.add_argument("--image-embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="experiment_split")
    parser.add_argument("--fit-split", default="fit")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    parser.add_argument("--l2-values", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--epochs-values", type=int, nargs="+", default=[300, 600])
    parser.add_argument("--malicious-prior", type=float, default=0.5)
    parser.add_argument("--abstain-fractions", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--permutation-seeds", type=int, nargs="+", default=[201, 202, 203, 204, 205])
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    if metadata is None:
        raise ValueError("metadata is required.")
    text_features, aligned_metadata, sample_ids = _load_aligned(args.text_embeddings, metadata)
    image_features, image_metadata, image_ids = _load_aligned(args.image_embeddings, metadata)
    if aligned_metadata is None or image_metadata is None:
        raise ValueError("metadata alignment failed.")
    if not np.array_equal(sample_ids, image_ids):
        raise ValueError("text and image embeddings do not align to the same sample IDs.")

    labels = coerce_binary_labels(aligned_metadata[args.label_column])
    splits = aligned_metadata[args.split_column].astype(str).to_numpy()
    fit_mask = splits == args.fit_split
    validation_mask = splits == args.validation_split
    test_mask = splits == args.test_split

    consistency, consistency_names = cross_modal_consistency_features(
        text_features, image_features
    )
    feature_views = {
        "text_tokens": text_features,
        "image_tokens": image_features,
        "concat_all": np.concatenate([text_features, image_features], axis=1),
        "cross_modal_consistency": consistency,
    }

    comparison_rows = []
    selected_by_feature: dict[str, dict] = {}
    for feature_name, features in feature_views.items():
        grid = run_logistic_validation_grid(
            features,
            labels=labels,
            fit_mask=fit_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            learning_rates=args.learning_rates,
            l2_values=args.l2_values,
            epochs_values=args.epochs_values,
            malicious_prior=args.malicious_prior,
        )
        selected = max((result.as_row() for result in grid), key=lambda row: row["validation_auprc"])
        selected_by_feature[feature_name] = selected
        comparison_rows.append({"feature": feature_name, **selected})

    concat = feature_views["concat_all"]
    selected_concat = selected_by_feature["concat_all"]
    classifier = LogisticRegressionNumpy(
        learning_rate=float(selected_concat["learning_rate"]),
        l2=float(selected_concat["l2"]),
        epochs=int(selected_concat["epochs"]),
    ).fit(concat[fit_mask], labels[fit_mask])
    threshold = float(selected_concat["threshold"])
    test_x = concat[test_mask]
    test_labels = labels[test_mask]
    test_scores = classifier.predict_proba(test_x)
    uncertainty = binary_entropy(test_scores)

    triage_rows = []
    for abstain_fraction in args.abstain_fractions:
        if not 0.0 <= abstain_fraction < 1.0:
            raise ValueError("abstain fractions must be in [0, 1).")
        keep_count = max(1, int(round(len(test_labels) * (1.0 - abstain_fraction))))
        keep = np.argsort(uncertainty)[:keep_count]
        kept_labels = test_labels[keep]
        kept_scores = test_scores[keep]
        kept_predictions = predict_from_threshold(kept_scores, threshold)
        report = detection_report(kept_labels, kept_scores, kept_predictions)
        accuracy = float(np.mean(kept_predictions == kept_labels))
        triage_rows.append(
            {
                "abstain_fraction": abstain_fraction,
                "coverage": keep_count / len(test_labels),
                "n_kept": keep_count,
                "mean_kept_uncertainty": float(np.mean(uncertainty[keep])),
                "accuracy": accuracy,
                "selective_error": 1.0 - accuracy,
                **report,
            }
        )

    if classifier.weights_ is None:
        raise RuntimeError("classifier weights are unavailable.")
    dimension = text_features.shape[1]
    text_weight_norm = float(np.linalg.norm(classifier.weights_[:dimension]))
    image_weight_norm = float(np.linalg.norm(classifier.weights_[dimension:]))
    baseline_auprc = average_precision(test_labels, test_scores)
    attribution_rows = []
    for block_name, start, end, weight_norm in [
        ("text_tokens", 0, dimension, text_weight_norm),
        ("image_tokens", dimension, 2 * dimension, image_weight_norm),
    ]:
        drops = []
        for seed in args.permutation_seeds:
            permuted = test_x.copy()
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(permuted))
            permuted[:, start:end] = permuted[order, start:end]
            drops.append(baseline_auprc - average_precision(test_labels, classifier.predict_proba(permuted)))
        attribution_rows.append(
            {
                "feature_block": block_name,
                "weight_l2_norm": weight_norm,
                "weight_norm_fraction": weight_norm / (text_weight_norm + image_weight_norm),
                "baseline_test_auprc": baseline_auprc,
                "permutation_auprc_drop_mean": float(np.mean(drops)),
                "permutation_auprc_drop_std": float(np.std(drops)),
                "permutation_seeds": len(drops),
            }
        )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    comparison_path = prefix.with_name(f"{prefix.name}_feature_comparison.csv")
    triage_path = prefix.with_name(f"{prefix.name}_uncertainty_triage.csv")
    attribution_path = prefix.with_name(f"{prefix.name}_modality_attribution.csv")
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    pd.DataFrame(triage_rows).to_csv(triage_path, index=False)
    pd.DataFrame(attribution_rows).to_csv(attribution_path, index=False)
    print(
        json.dumps(
            {
                "comparison": str(comparison_path),
                "triage": str(triage_path),
                "attribution": str(attribution_path),
                "consistency_features": consistency_names,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _load_aligned(
    path: str,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame | None, np.ndarray]:
    embeddings, sample_ids = load_embeddings(path)
    return align_embeddings_with_metadata(embeddings, sample_ids, metadata)


if __name__ == "__main__":
    main()
