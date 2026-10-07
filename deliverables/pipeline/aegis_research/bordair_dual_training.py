from __future__ import annotations

"""Training workflow for a validation-selected image/text detector pair."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

import numpy as np

from .bordair_dual import (
    ChannelCandidate,
    build_pair_manifest,
    build_one_shot_qualification_binding,
    combine_actions,
    head_action,
    pair_metrics,
    select_validation_pair,
    sha256_file,
)
from .bordair_provenance import (
    DETECTOR_CORPUS_VERSION,
    detector_source_label,
    training_identity_sha256,
)


IMAGE_LED_STRATEGY = "benign_text_full_injection"
TEXT_LED_STRATEGY = "malicious_text_benign_image_counterfactual"
BENIGN_STRATEGY = "official_bordair_benign_text_rendered_in_image"
DEFAULT_IMAGE_POSITIVE_WEIGHTS = (0.75, 1.0, 1.1, 1.25)
DEFAULT_TEXT_POSITIVE_WEIGHTS = (1.0, 2.0, 4.0)
DEFAULT_LEARNING_RATES = (0.02, 0.05)
DEFAULT_L2_VALUES = (0.001, 0.01, 0.1)
DEFAULT_BUDGET_ALLOCATIONS = ((0, 2), (1, 1), (2, 0))
DEFAULT_EPOCHS = 600
DEFAULT_RANDOM_SEED = 42
VALIDATION_BENIGN_FP_LIMIT = 2
INTERNAL_BENIGN_FP_LIMIT = 8
TRAINING_EVIDENCE_SCHEMA_VERSION = 1
TRAINING_EVIDENCE_FILENAME = "qualification_features_v1.npz"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(_json_safe(row) for row in rows)


def _write_training_feature_evidence(
    path: Path,
    *,
    validation_rows: list[dict[str, str]],
    validation_labels: np.ndarray,
    validation_image_led: np.ndarray,
    validation_text_led: np.ndarray,
    validation_image_features: np.ndarray,
    validation_text_features: np.ndarray,
    internal_rows: list[dict[str, str]],
    internal_labels: np.ndarray,
    internal_image_led: np.ndarray,
    internal_text_led: np.ndarray,
    internal_image_features: np.ndarray,
    internal_text_features: np.ndarray,
    regression_rows: list[dict[str, str]],
    regression_labels: np.ndarray,
    regression_image_features: np.ndarray,
    regression_text_features: np.ndarray,
) -> None:
    """Write the exact post-selection panels needed for independent rescoring."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": np.asarray([TRAINING_EVIDENCE_SCHEMA_VERSION], dtype=np.int64),
        "validation_sample_ids": np.asarray(
            [row["sample_id"] for row in validation_rows]
        ),
        "validation_labels": np.asarray(validation_labels, dtype=np.int64),
        "validation_image_led": np.asarray(validation_image_led, dtype=np.bool_),
        "validation_text_led": np.asarray(validation_text_led, dtype=np.bool_),
        "validation_image_features": np.asarray(
            validation_image_features, dtype=np.float64
        ),
        "validation_text_features": np.asarray(
            validation_text_features, dtype=np.float64
        ),
        "internal_sample_ids": np.asarray([row["sample_id"] for row in internal_rows]),
        "internal_labels": np.asarray(internal_labels, dtype=np.int64),
        "internal_image_led": np.asarray(internal_image_led, dtype=np.bool_),
        "internal_text_led": np.asarray(internal_text_led, dtype=np.bool_),
        "internal_image_features": np.asarray(internal_image_features, dtype=np.float64),
        "internal_text_features": np.asarray(internal_text_features, dtype=np.float64),
        "regression_sample_ids": np.asarray(
            [row["sample_id"] for row in regression_rows]
        ),
        "regression_labels": np.asarray(regression_labels, dtype=np.int64),
        "regression_image_features": np.asarray(
            regression_image_features, dtype=np.float64
        ),
        "regression_text_features": np.asarray(
            regression_text_features, dtype=np.float64
        ),
    }
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)


def _positive_grid(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and duplicate-free")
    if any(not np.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def validate_channel_rows(rows: list[dict[str, str]]) -> set[str]:
    """Recheck the schema-4 channel semantics before fitting either head."""

    by_id: dict[str, dict[str, str]] = {}
    family_by_split: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in by_id:
            raise ValueError(f"missing or duplicate development sample ID: {sample_id!r}")
        by_id[sample_id] = row
        try:
            label = int(row["label_id"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid label for {sample_id}") from exc
        split = row.get("split")
        strategy = row.get("strategy")
        if split not in family_by_split:
            raise ValueError(f"unsupported split for {sample_id}: {split!r}")
        valid = (
            (label == 0 and strategy == BENIGN_STRATEGY)
            or (label == 1 and strategy == IMAGE_LED_STRATEGY)
            or (label == 1 and strategy == TEXT_LED_STRATEGY)
        )
        if not valid:
            raise ValueError(
                f"unexpected label/strategy combination for {sample_id}: "
                f"{label}/{strategy}"
            )
        if label == 1:
            family_by_split[split].add(row.get("group_id", ""))
    if (
        family_by_split["train"] & family_by_split["validation"]
        or family_by_split["train"] & family_by_split["test"]
        or family_by_split["validation"] & family_by_split["test"]
    ):
        raise ValueError("malicious family leakage exists across development splits")

    paired_benign_ids: set[str] = set()
    for row in rows:
        if int(row["label_id"]) != 1 or row["strategy"] != TEXT_LED_STRATEGY:
            continue
        paired_id = row.get("paired_benign_sample_id", "").strip()
        if not paired_id or paired_id in paired_benign_ids:
            raise ValueError(f"missing or reused paired benign ID: {paired_id!r}")
        paired = by_id.get(paired_id)
        if paired is None or int(paired["label_id"]) != 0:
            raise ValueError(f"paired benign row is missing or malicious: {paired_id}")
        if paired["split"] != row["split"]:
            raise ValueError(f"counterfactual pair crosses splits: {row['sample_id']}")
        if paired["image_sha256"] != row["image_sha256"]:
            raise ValueError(f"counterfactual pair image differs: {row['sample_id']}")
        if row["split"] == "train" and paired.get("hard_negative") != "0":
            raise ValueError(f"training paired benign row is a hard negative: {paired_id}")
        paired_benign_ids.add(paired_id)
    return paired_benign_ids


def _channel_weights(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    *,
    paired_benign_ids: set[str],
    positive_weight: float,
    benign_weight: float,
    paired_benign_weight: float,
    hard_negative_weight: float,
) -> np.ndarray:
    weights = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        if int(labels[index]) == 1:
            weights[index] = positive_weight
        elif row.get("hard_negative") == "1":
            weights[index] = hard_negative_weight
        elif row["sample_id"] in paired_benign_ids:
            weights[index] = paired_benign_weight
        else:
            weights[index] = benign_weight
    return weights


def fit_channel_grid(
    *,
    channel: str,
    features: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, str]],
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    paired_benign_ids: set[str],
    positive_weights: Iterable[float],
    learning_rates: Iterable[float],
    l2_values: Iterable[float],
    benign_weight: float,
    paired_benign_weight: float,
    hard_negative_weight: float,
    epochs: int,
    random_seed: int,
) -> tuple[list[ChannelCandidate], list[dict[str, Any]]]:
    from AEGIS.logistic import LogisticRegressionNumpy

    if channel not in {"image", "text"}:
        raise ValueError(f"unsupported channel: {channel!r}")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[0] != len(rows) or y.shape != (len(rows),):
        raise ValueError("features, labels, and development rows are not aligned")
    selected_rows = [row for row, include in zip(rows, train_mask) if include]
    selected_labels = y[train_mask]
    if set(np.unique(selected_labels)) != {0, 1}:
        raise ValueError(f"{channel} training subset must contain both classes")
    candidates: list[ChannelCandidate] = []
    diagnostics: list[dict[str, Any]] = []
    candidate_id = 0
    for learning_rate in _positive_grid(learning_rates, "learning-rate grid"):
        for l2 in _positive_grid(l2_values, "L2 grid"):
            for positive_weight in _positive_grid(
                positive_weights, f"{channel} positive-weight grid"
            ):
                weights = _channel_weights(
                    selected_rows,
                    selected_labels,
                    paired_benign_ids=paired_benign_ids,
                    positive_weight=positive_weight,
                    benign_weight=benign_weight,
                    paired_benign_weight=paired_benign_weight,
                    hard_negative_weight=hard_negative_weight,
                )
                classifier = LogisticRegressionNumpy(
                    learning_rate=learning_rate,
                    epochs=epochs,
                    l2=l2,
                    standardize=True,
                    random_seed=random_seed,
                ).fit(x[train_mask], selected_labels, sample_weight=weights)
                validation_scores = classifier.predict_proba(x[validation_mask])
                candidates.append(
                    ChannelCandidate(
                        candidate_id=candidate_id,
                        channel=channel,
                        classifier=classifier,
                        validation_scores=validation_scores,
                        learning_rate=learning_rate,
                        l2=l2,
                        positive_weight=positive_weight,
                    )
                )
                diagnostics.append(
                    {
                        "channel": channel,
                        "candidate_id": candidate_id,
                        "learning_rate": learning_rate,
                        "l2": l2,
                        "positive_weight": positive_weight,
                        "epochs": epochs,
                        "standardize": True,
                        "random_seed": random_seed,
                        "training_rows": int(np.sum(train_mask)),
                        "training_benign_rows": int(np.sum(selected_labels == 0)),
                        "training_malicious_rows": int(np.sum(selected_labels == 1)),
                        "training_weighted_mass": float(np.sum(weights)),
                    }
                )
                candidate_id += 1
    return candidates, diagnostics


def _panel_result(
    *,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    image_led: np.ndarray,
    text_led: np.ndarray,
    image_features: np.ndarray,
    text_features: np.ndarray,
    image_candidate: ChannelCandidate,
    text_candidate: ChannelCandidate,
    image_threshold: float,
    text_threshold: float,
    image_review_threshold: float,
    text_review_threshold: float,
    panel: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_scores = image_candidate.classifier.predict_proba(image_features)
    text_scores = text_candidate.classifier.predict_proba(text_features)
    metrics = pair_metrics(
        labels=labels,
        image_scores=image_scores,
        text_scores=text_scores,
        image_threshold=image_threshold,
        text_threshold=text_threshold,
        image_led=image_led,
        text_led=text_led,
    )
    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        image_action = head_action(
            float(image_scores[index]),
            block_threshold=image_threshold,
            review_threshold=image_review_threshold,
        )
        text_action = head_action(
            float(text_scores[index]),
            block_threshold=text_threshold,
            review_threshold=text_review_threshold,
        )
        predictions.append(
            {
                "panel": panel,
                "sample_id": row["sample_id"],
                "label_id": int(row["label_id"]),
                "strategy": row.get("strategy", ""),
                "image_head_score": float(image_scores[index]),
                "image_head_action": image_action,
                "text_head_score": float(text_scores[index]),
                "text_head_action": text_action,
                "combined_action": combine_actions(image_action, text_action),
            }
        )
    return metrics, predictions


def train_dual_pair(
    *,
    target: str,
    development_rows: list[dict[str, str]],
    regression_rows: list[dict[str, str]],
    development_features: dict[str, np.ndarray],
    regression_features: dict[str, np.ndarray],
    provenance: dict[str, Any],
    run_root: Path,
    corpus_manifest_path: Path,
    development_metadata_path: Path,
    regression_metadata_path: Path,
    result_root: Path | None = None,
    image_positive_weights: Iterable[float] = DEFAULT_IMAGE_POSITIVE_WEIGHTS,
    text_positive_weights: Iterable[float] = DEFAULT_TEXT_POSITIVE_WEIGHTS,
    learning_rates: Iterable[float] = DEFAULT_LEARNING_RATES,
    l2_values: Iterable[float] = DEFAULT_L2_VALUES,
    budget_allocations: Iterable[tuple[int, int]] = DEFAULT_BUDGET_ALLOCATIONS,
    benign_weight: float = 1.0,
    paired_benign_weight: float = 2.0,
    hard_negative_weight: float = 3.5,
    epochs: int = DEFAULT_EPOCHS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    artifact_source_overrides: Mapping[str, str] | None = None,
    qualification_evidence_directory: Path | None = None,
) -> dict[str, Any]:
    """Fit, select, qualify, and emit a dual pair without frozen-panel access."""

    from AEGIS.detector_artifact import (
        DetectorArtifact,
        load_detector_artifact,
        save_detector_artifact,
    )

    paired_ids = validate_channel_rows(development_rows)
    image_positive_grid = _positive_grid(
        image_positive_weights, "image positive-weight grid"
    )
    text_positive_grid = _positive_grid(
        text_positive_weights, "text positive-weight grid"
    )
    learning_rate_grid = _positive_grid(learning_rates, "learning-rate grid")
    l2_grid = _positive_grid(l2_values, "L2 grid")
    threshold_budget_allocations = tuple(
        (int(image_budget), int(text_budget))
        for image_budget, text_budget in budget_allocations
    )
    if not threshold_budget_allocations:
        raise ValueError("threshold budget allocations must not be empty")
    for name, value in (
        ("benign weight", benign_weight),
        ("paired benign weight", paired_benign_weight),
        ("hard-negative weight", hard_negative_weight),
    ):
        _positive_grid((value,), name)
    if epochs <= 0 or random_seed < 0:
        raise ValueError("epochs must be positive and random seed non-negative")
    explicit_sources: dict[str, str] | None = None
    if artifact_source_overrides is not None:
        if set(artifact_source_overrides) != {"image", "text"}:
            raise ValueError("artifact source overrides require image and text entries")
        explicit_sources = {
            name: str(artifact_source_overrides[name]) for name in ("image", "text")
        }
        if any(not value for value in explicit_sources.values()):
            raise ValueError("artifact source overrides must be non-empty")
    required_poolings = {"image_tokens", "text_tokens"}
    if set(development_features) != required_poolings or set(regression_features) != required_poolings:
        raise ValueError("dual training requires exactly image_tokens and text_tokens features")
    development_count = len(development_rows)
    for panel_name, matrices, expected_rows in (
        ("development", development_features, development_count),
        ("regression", regression_features, len(regression_rows)),
    ):
        shapes = {name: np.asarray(value).shape for name, value in matrices.items()}
        if any(len(shape) != 2 or shape[0] != expected_rows for shape in shapes.values()):
            raise ValueError(f"{panel_name} feature matrices are not row-aligned: {shapes}")
        if shapes["image_tokens"] != shapes["text_tokens"]:
            raise ValueError(f"{panel_name} primitive dimensions do not match")
    base_dim = int(np.asarray(development_features["image_tokens"]).shape[1])
    if np.asarray(regression_features["image_tokens"]).shape[1] != base_dim:
        raise ValueError("development/regression primitive dimensions differ")

    labels = np.asarray([int(row["label_id"]) for row in development_rows], dtype=np.int64)
    splits = np.asarray([row["split"] for row in development_rows])
    strategies = np.asarray([row["strategy"] for row in development_rows])
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    internal_mask = splits == "test"
    benign = labels == 0
    image_led = (labels == 1) & (strategies == IMAGE_LED_STRATEGY)
    text_led = (labels == 1) & (strategies == TEXT_LED_STRATEGY)
    if not np.all(train_mask | validation_mask | internal_mask):
        raise ValueError("development rows contain an unsupported split")
    image_train = train_mask & (benign | image_led)
    text_train = train_mask & (benign | text_led)
    # The exclusions are safety-critical: neutral text and benign carrier images
    # must never inherit the malicious label from the other modality.
    if np.any(image_train & text_led) or np.any(text_train & image_led):
        raise RuntimeError("channel-specific training exclusion failed")

    image_candidates, image_diagnostics = fit_channel_grid(
        channel="image",
        features=development_features["image_tokens"],
        labels=labels,
        rows=development_rows,
        train_mask=image_train,
        validation_mask=validation_mask,
        paired_benign_ids=paired_ids,
        positive_weights=image_positive_grid,
        learning_rates=learning_rate_grid,
        l2_values=l2_grid,
        benign_weight=benign_weight,
        paired_benign_weight=paired_benign_weight,
        hard_negative_weight=hard_negative_weight,
        epochs=epochs,
        random_seed=random_seed,
    )
    text_candidates, text_diagnostics = fit_channel_grid(
        channel="text",
        features=development_features["text_tokens"],
        labels=labels,
        rows=development_rows,
        train_mask=text_train,
        validation_mask=validation_mask,
        paired_benign_ids=paired_ids,
        positive_weights=text_positive_grid,
        learning_rates=learning_rate_grid,
        l2_values=l2_grid,
        benign_weight=benign_weight,
        paired_benign_weight=paired_benign_weight,
        hard_negative_weight=hard_negative_weight,
        epochs=epochs,
        random_seed=random_seed,
    )
    selected, image_candidate, text_candidate, pair_candidates = select_validation_pair(
        image_candidates=image_candidates,
        text_candidates=text_candidates,
        validation_labels=labels[validation_mask],
        validation_image_led=image_led[validation_mask],
        validation_text_led=text_led[validation_mask],
        budget_allocations=threshold_budget_allocations,
        combined_false_positive_limit=VALIDATION_BENIGN_FP_LIMIT,
    )
    image_threshold = float(selected["image_threshold"])
    text_threshold = float(selected["text_threshold"])
    image_review = max(0.0, image_threshold - 0.02)
    text_review = max(0.0, text_threshold - 0.02)

    validation_rows = [row for row, include in zip(development_rows, validation_mask) if include]
    internal_rows = [row for row, include in zip(development_rows, internal_mask) if include]
    validation_metrics, validation_predictions = _panel_result(
        rows=validation_rows,
        labels=labels[validation_mask],
        image_led=image_led[validation_mask],
        text_led=text_led[validation_mask],
        image_features=development_features["image_tokens"][validation_mask],
        text_features=development_features["text_tokens"][validation_mask],
        image_candidate=image_candidate,
        text_candidate=text_candidate,
        image_threshold=image_threshold,
        text_threshold=text_threshold,
        image_review_threshold=image_review,
        text_review_threshold=text_review,
        panel="validation",
    )
    # Internal qualification begins only after the validation winner is fixed.
    internal_metrics, internal_predictions = _panel_result(
        rows=internal_rows,
        labels=labels[internal_mask],
        image_led=image_led[internal_mask],
        text_led=text_led[internal_mask],
        image_features=development_features["image_tokens"][internal_mask],
        text_features=development_features["text_tokens"][internal_mask],
        image_candidate=image_candidate,
        text_candidate=text_candidate,
        image_threshold=image_threshold,
        text_threshold=text_threshold,
        image_review_threshold=image_review,
        text_review_threshold=text_review,
        panel="internal_test",
    )
    regression_labels = np.asarray(
        [int(row["label_id"]) for row in regression_rows], dtype=np.int64
    )
    regression_metrics, regression_predictions = _panel_result(
        rows=regression_rows,
        labels=regression_labels,
        image_led=regression_labels == 1,
        text_led=np.zeros(len(regression_rows), dtype=bool),
        image_features=regression_features["image_tokens"],
        text_features=regression_features["text_tokens"],
        image_candidate=image_candidate,
        text_candidate=text_candidate,
        image_threshold=image_threshold,
        text_threshold=text_threshold,
        image_review_threshold=image_review,
        text_review_threshold=text_review,
        panel="prior_regression",
    )

    validation_checks = {
        "exact_80_benign_80_image_led_20_text_led": (
            validation_metrics["benign_rows"] == 80
            and validation_metrics["image_led_rows"] == 80
            and validation_metrics["text_led_rows"] == 20
        ),
        "image_head_blocks_image_led_80_of_80": (
            validation_metrics["image_head_image_led_false_negatives"] == 0
        ),
        "text_head_blocks_text_led_20_of_20": (
            validation_metrics["text_head_text_led_false_negatives"] == 0
        ),
        "combined_benign_false_positives_at_most_2_of_80": (
            validation_metrics["false_positives"] <= VALIDATION_BENIGN_FP_LIMIT
        ),
    }
    internal_checks = {
        "exact_80_benign_80_image_led_20_text_led": (
            internal_metrics["benign_rows"] == 80
            and internal_metrics["image_led_rows"] == 80
            and internal_metrics["text_led_rows"] == 20
        ),
        "image_head_blocks_image_led_80_of_80": (
            internal_metrics["image_head_image_led_false_negatives"] == 0
        ),
        "text_head_blocks_text_led_20_of_20": (
            internal_metrics["text_head_text_led_false_negatives"] == 0
        ),
        "combined_benign_false_positives_at_most_8_of_80": (
            internal_metrics["false_positives"] <= INTERNAL_BENIGN_FP_LIMIT
        ),
    }
    regression_checks = {
        "exact_10_malicious_rows": (
            regression_metrics["malicious_rows"] == 10
            and regression_metrics["benign_rows"] == 0
        ),
        "image_head_blocks_prior_image_attacks_10_of_10": (
            regression_metrics["image_head_image_led_false_negatives"] == 0
        ),
    }
    acceptance_passed = bool(
        all(validation_checks.values())
        and all(internal_checks.values())
        and all(regression_checks.values())
    )

    grid_configuration = {
        "schema_version": 1,
        "mode": "dual_or",
        "representations": {"image": "image_tokens", "text": "text_tokens"},
        "classifier_grid": {
            "learning_rates": list(learning_rate_grid),
            "l2_values": list(l2_grid),
            "image_positive_weights": list(image_positive_grid),
            "text_positive_weights": list(text_positive_grid),
            "epochs": epochs,
            "standardize": True,
            "random_seed": random_seed,
        },
        "sample_weights": {
            "ordinary_benign": benign_weight,
            "counterfactual_paired_benign": paired_benign_weight,
            "hard_negative_benign": hard_negative_weight,
        },
        "threshold_budget_allocations": [
            list(value) for value in threshold_budget_allocations
        ],
        "selection_uses_validation_only": True,
        "internal_test_is_post_selection_qualification_only": True,
        "prior_regression_is_post_selection_qualification_only": True,
        "frozen_panels_are_excluded": True,
        "artifact_source_policy": (
            "explicit_byte_reproduction"
            if explicit_sources is not None
            else "canonical_training_identity"
        ),
        "artifact_source_overrides": explicit_sources,
    }
    selected_identity_value = _json_safe(dict(selected))
    training_digest = training_identity_sha256(
        grid_configuration,
        selected_identity_value,
    )
    manifest_hash = sha256_file(corpus_manifest_path)
    image_source = detector_source_label(
        corpus_version=DETECTOR_CORPUS_VERSION,
        corpus_manifest_sha256=manifest_hash,
        target=target,
        pooling="image_tokens",
        training_identity_sha256_value=training_digest,
    )
    text_source = detector_source_label(
        corpus_version=DETECTOR_CORPUS_VERSION,
        corpus_manifest_sha256=manifest_hash,
        target=target,
        pooling="text_tokens",
        training_identity_sha256_value=training_digest,
    )
    if explicit_sources is not None:
        image_source = explicit_sources["image"]
        text_source = explicit_sources["text"]
    result_directory = (
        (run_root / "training_v7" / target / "dual_or")
        if result_root is None
        else result_root
    ).resolve()
    result_directory.mkdir(parents=True, exist_ok=True)
    image_path = result_directory / f"{target}_image_head_v1.npz"
    text_path = result_directory / f"{target}_text_head_v1.npz"
    pair_manifest_path = result_directory / "detector_pair_manifest.json"
    summary_path = result_directory / "training_summary.json"
    generated_paths = {
        "image_fit_diagnostics.csv": image_diagnostics,
        "text_fit_diagnostics.csv": text_diagnostics,
        "candidate_pair_metrics.csv": pair_candidates,
        "validation_predictions.csv": validation_predictions,
        "internal_test_predictions.csv": internal_predictions,
        "prior_regression_predictions.csv": regression_predictions,
    }
    for filename, rows in generated_paths.items():
        _write_csv(result_directory / filename, rows)
    training_feature_evidence_path = result_directory / TRAINING_EVIDENCE_FILENAME
    _write_training_feature_evidence(
        training_feature_evidence_path,
        validation_rows=validation_rows,
        validation_labels=labels[validation_mask],
        validation_image_led=image_led[validation_mask],
        validation_text_led=text_led[validation_mask],
        validation_image_features=development_features["image_tokens"][validation_mask],
        validation_text_features=development_features["text_tokens"][validation_mask],
        internal_rows=internal_rows,
        internal_labels=labels[internal_mask],
        internal_image_led=image_led[internal_mask],
        internal_text_led=text_led[internal_mask],
        internal_image_features=development_features["image_tokens"][internal_mask],
        internal_text_features=development_features["text_tokens"][internal_mask],
        regression_rows=regression_rows,
        regression_labels=regression_labels,
        regression_image_features=regression_features["image_tokens"],
        regression_text_features=regression_features["text_tokens"],
    )

    prediction_bindings = {
        panel: {
            "path": filename,
            "sha256": sha256_file(result_directory / filename),
            "rows": len(rows),
        }
        for panel, filename, rows in (
            ("validation", "validation_predictions.csv", validation_predictions),
            ("internal_test", "internal_test_predictions.csv", internal_predictions),
            ("prior_regression", "prior_regression_predictions.csv", regression_predictions),
        )
    }

    summary: dict[str, Any] = {
        "schema_version": 2,
        "artifact_mode": "dual_or",
        "target": target,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "Fit image heads on all train benign plus image-led malicious rows and "
            "text heads on all train benign plus text-led malicious rows. Exclude "
            "the irrelevant malicious channel. Select complete image/text pairs by "
            "validation OR behavior only: require zero relevant-channel misses and "
            "at most 2/80 combined benign false positives, then rank by fewer false "
            "positives, larger worst relevant-channel logit margin, AUPRC, AUROC, "
            "rank separation, and deterministic hyperparameter tie-breakers. Internal "
            "test and prior regression are pass/fail qualification only."
        ),
        "training_configuration": grid_configuration,
        "selected_candidate": selected_identity_value,
        "training_identity": {"schema_version": 1, "sha256": training_digest},
        "provenance": dict(provenance),
        "base_feature_dimension": base_dim,
        "channel_training": {
            "image": {
                "included": "all benign + image-led malicious",
                "excluded": "text-led malicious",
            },
            "text": {
                "included": "all benign + text-led malicious",
                "excluded": "image-led malicious",
            },
        },
        "validation_metrics": validation_metrics,
        "test_metrics": internal_metrics,
        "regression_metrics": regression_metrics,
        "acceptance": {
            "passed": acceptance_passed,
            "validation_checks": validation_checks,
            "internal_checks": internal_checks,
            "regression_checks": regression_checks,
        },
        "corpus_provenance": {
            "manifest_sha256": manifest_hash,
            "development_metadata_sha256": sha256_file(development_metadata_path),
            "regression_metadata_sha256": sha256_file(regression_metadata_path),
        },
        "development_qualification_evidence": {
            "schema_version": TRAINING_EVIDENCE_SCHEMA_VERSION,
            "feature_snapshot": {
                "path": TRAINING_EVIDENCE_FILENAME,
                "sha256": sha256_file(training_feature_evidence_path),
                "bytes": training_feature_evidence_path.stat().st_size,
            },
            "predictions": prediction_bindings,
            "corpus": {
                "manifest_sha256": manifest_hash,
                "development_metadata_sha256": sha256_file(development_metadata_path),
                "regression_metadata_sha256": sha256_file(regression_metadata_path),
            },
            "artifact_sha256": None,
        },
        "artifact_pair": None,
    }
    if acceptance_passed:
        shared_kwargs = {
            "model_family": str(provenance["model_family"]),
            "model_id": str(provenance["model_id"]),
            "model_revision": str(provenance["model_revision"]),
            "tokenizer_revision": str(provenance["tokenizer_revision"]),
            "preprocessing_sha256": str(provenance["preprocessing_sha256"]),
            "layer": int(provenance["layer"]),
        }
        image_artifact = DetectorArtifact(
            classifier=image_candidate.classifier,
            threshold=image_threshold,
            uncertainty_margin=0.02,
            pooling="image_tokens",
            source=image_source,
            **shared_kwargs,
        )
        text_artifact = DetectorArtifact(
            classifier=text_candidate.classifier,
            threshold=text_threshold,
            uncertainty_margin=0.02,
            pooling="text_tokens",
            source=text_source,
            **shared_kwargs,
        )
        save_detector_artifact(image_path, image_artifact)
        save_detector_artifact(text_path, text_artifact)
        # Require exact version-1 round trips before declaring the pair.
        image_artifact = load_detector_artifact(image_path)
        text_artifact = load_detector_artifact(text_path)
        summary["development_qualification_evidence"]["artifact_sha256"] = {
            "image": sha256_file(image_path),
            "text": sha256_file(text_path),
        }
        qualification_evidence: dict[str, Any] | None = None
        qualification_output_files: list[Path] = []
        if qualification_evidence_directory is not None:
            source_evidence = qualification_evidence_directory.expanduser().resolve()
            source_protocol_path = source_evidence.parent / "protocol.json"
            source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
            generic_package = (
                isinstance(source_protocol, dict)
                and source_protocol.get("package_format")
                == "generic_dual_evaluator_compat_v1"
            )
            source_development_manifest = source_evidence.parent / "development_manifest.json"
            if not source_development_manifest.is_file():
                source_development_manifest = source_evidence.parent.parent / "output_manifest.json"
            metadata_source_files = {
                "regression_metadata": run_root.expanduser().resolve()
                / "final_metadata_v7.csv",
                "text_led_metadata": run_root.expanduser().resolve()
                / "text_led_final_metadata_v7.csv",
                "external_benign_metadata": run_root.expanduser().resolve()
                / "external_benign_metadata_v7.csv",
            }
            metadata_file_keys = set(metadata_source_files)
            source_files = {
                "attempt_started": source_evidence / "attempt_started.json",
                "development_manifest": source_development_manifest,
                "evaluator": source_evidence.parent / "evaluate_once.py",
                "protocol": source_evidence.parent / "protocol.json",
                "results_json": source_evidence / "dual_head_frozen_results.json",
                "results_csv": source_evidence / "dual_head_frozen_results.csv",
                "evaluation_manifest": source_evidence / "evaluation_manifest.json",
                "validator": source_evidence.parent / "validate_once.py",
                "validation_report": source_evidence / "validation_report.json",
                "validation_manifest": source_evidence / "validation_manifest.json",
                **metadata_source_files,
            }
            destination_root = result_directory / "qualification_evidence"
            if generic_package:
                if destination_root.exists():
                    raise FileExistsError(
                        f"generic qualification destination already exists: {destination_root}"
                    )
                shutil.copytree(source_evidence.parent, destination_root)
            destination_evidence = destination_root / "evidence"
            destination_evidence.mkdir(parents=True, exist_ok=True)
            destination_files = {
                "attempt_started": destination_evidence / "attempt_started.json",
                "development_manifest": destination_root / "development_manifest.json",
                "evaluator": destination_root / "evaluate_once.py",
                "protocol": destination_root / "protocol.json",
                **{
                    name: destination_evidence / source.name
                    for name, source in source_files.items()
                    if name
                    not in {
                        "attempt_started",
                        "development_manifest",
                        "evaluator",
                        "protocol",
                        "validator",
                    }
                    and name not in metadata_file_keys
                },
                "validator": destination_root / "validate_once.py",
                **{
                    name: destination_root / "metadata" / source.name
                    for name, source in metadata_source_files.items()
                },
            }
            for name, source in source_files.items():
                if not source.is_file():
                    raise FileNotFoundError(
                        f"qualification evidence source is missing: {source}"
                    )
                destination_files[name].parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination_files[name])
                if sha256_file(source) != sha256_file(destination_files[name]):
                    raise RuntimeError(f"qualification evidence copy differs: {name}")
            qualification_output_files = list(destination_files.values())
            if generic_package:
                qualification_output_files = sorted(
                    path for path in destination_root.rglob("*") if path.is_file()
                )
            qualification_evidence = build_one_shot_qualification_binding(
                evidence_directory=destination_evidence,
                pair_directory=result_directory,
                image_artifact_sha256=sha256_file(image_path),
                text_artifact_sha256=sha256_file(text_path),
                corpus_manifest_sha256=manifest_hash,
            )
        pair_manifest = build_pair_manifest(
            target=target,
            image_path=image_path,
            image_artifact=image_artifact,
            image_review_threshold=image_review,
            text_path=text_path,
            text_artifact=text_artifact,
            text_review_threshold=text_review,
            output_directory=result_directory,
            training_identity_sha256_value=training_digest,
            corpus={
                "version": DETECTOR_CORPUS_VERSION,
                **summary["corpus_provenance"],
            },
            development_qualification_evidence=summary[
                "development_qualification_evidence"
            ],
            qualification_evidence=qualification_evidence,
        )
        _write_json(pair_manifest_path, pair_manifest)
        summary["artifact_pair"] = {
            "manifest": pair_manifest_path.name,
            "manifest_sha256": sha256_file(pair_manifest_path),
            "pair_identity_sha256": pair_manifest["pair_identity_sha256"],
            "runtime_detector_identity_sha256": pair_manifest[
                "runtime_detector_identity_sha256"
            ],
            "artifacts": pair_manifest["artifacts"],
            "qualification_evidence": pair_manifest["qualification_evidence"],
        }
    else:
        for path in (image_path, text_path, pair_manifest_path):
            path.unlink(missing_ok=True)
    _write_json(summary_path, summary)
    output_files = [result_directory / name for name in generated_paths]
    output_files.append(training_feature_evidence_path)
    output_files.append(summary_path)
    if acceptance_passed:
        output_files.extend((image_path, text_path, pair_manifest_path))
        output_files.extend(qualification_output_files)
    output_manifest = {
        "schema_version": 2,
        "artifact_mode": "dual_or",
        "target": target,
        "training_identity_sha256": training_digest,
        "pair_identity_sha256": (
            summary["artifact_pair"]["pair_identity_sha256"]
            if isinstance(summary["artifact_pair"], dict)
            else None
        ),
        "outputs": {
            path.relative_to(result_directory).as_posix(): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in output_files
        },
    }
    _write_json(result_directory / "output_manifest.json", output_manifest)
    if not acceptance_passed:
        raise RuntimeError(f"{target} dual detector pair failed qualification gates")
    return _json_safe(summary)
