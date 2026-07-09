from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from AEGIS.metrics import detection_report


@dataclass(frozen=True)
class CalibrationCriteria:
    max_false_positive_rate: float = 0.01
    min_recall: float = 0.90
    review_margin: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_false_positive_rate < 1.0:
            raise ValueError("max_false_positive_rate must be in [0, 1).")
        if not 0.0 <= self.min_recall <= 1.0:
            raise ValueError("min_recall must be in [0, 1].")
        if not 0.0 <= self.review_margin < 0.5:
            raise ValueError("review_margin must be in [0, 0.5).")


def calibrate_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    criteria: CalibrationCriteria,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be 1D arrays with the same shape.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1.")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("calibration requires benign and malicious labels.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite.")

    candidates = np.unique(scores)
    feasible = []
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        report = detection_report(labels, scores, predictions)
        fpr = report["fp"] / (report["fp"] + report["tn"]) if report["fp"] + report["tn"] else 1.0
        if fpr <= criteria.max_false_positive_rate and report["recall"] >= criteria.min_recall:
            feasible.append((float(threshold), report, float(fpr)))

    if feasible:
        # Prefer the highest recall, then lowest FPR, then lower threshold for less brittle blocking.
        threshold, report, fpr = max(
            feasible,
            key=lambda item: (item[1]["recall"], -item[2], -item[0]),
        )
        feasible_status = True
    else:
        # Fall back to the best recall under the FPR cap so operators see the actual shortfall.
        under_fpr = []
        for threshold in candidates:
            predictions = (scores >= threshold).astype(np.int64)
            report = detection_report(labels, scores, predictions)
            fpr = (
                report["fp"] / (report["fp"] + report["tn"])
                if report["fp"] + report["tn"]
                else 1.0
            )
            if fpr <= criteria.max_false_positive_rate:
                under_fpr.append((float(threshold), report, float(fpr)))
        if not under_fpr:
            threshold = float(np.nextafter(np.max(scores), 1.0))
            predictions = (scores >= threshold).astype(np.int64)
            report = detection_report(labels, scores, predictions)
            fpr = (
                report["fp"] / (report["fp"] + report["tn"])
                if report["fp"] + report["tn"]
                else 1.0
            )
        else:
            threshold, report, fpr = max(
                under_fpr,
                key=lambda item: (item[1]["recall"], item[1]["precision"], -item[0]),
            )
        feasible_status = False

    return {
        "criteria": asdict(criteria),
        "threshold": float(threshold),
        "review_margin": float(criteria.review_margin),
        "feasible": feasible_status,
        "metrics": report,
        "false_positive_rate": float(fpr),
        "n_samples": int(len(labels)),
        "n_benign": int(np.sum(labels == 0)),
        "n_malicious": int(np.sum(labels == 1)),
    }


def policy_payload(
    calibration: dict[str, object],
    *,
    detector_path: str,
    name: str,
    require_matching_provenance: bool = True,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": name,
        "detector": detector_path,
        "policy": {
            "block_threshold": calibration["threshold"],
            "review_margin": calibration["review_margin"],
            "action_on_error": "review",
            "require_matching_provenance": require_matching_provenance,
            "hash_images": False,
        },
        "calibration": calibration,
        "logging": {
            "store_raw_prompts": False,
            "store_prompt_sha256": True,
            "store_image_sha256": False,
            "store_scores": True,
            "store_actions": True,
        },
    }
