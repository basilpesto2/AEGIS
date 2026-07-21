from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from aegis_research.signals import build_feature_views  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an aligned feature bundle with a saved research detector.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("metadata is empty")
    id_column = "sample_id" if "sample_id" in rows[0] else "variant_id"
    with np.load(args.features, allow_pickle=False) as bundle:
        sample_ids = np.asarray(bundle["sample_ids"]).astype(str)
        expected = np.asarray([row[id_column] for row in rows])
        if not np.array_equal(sample_ids, expected):
            raise ValueError("feature IDs are not aligned to metadata")
        views = build_feature_views(
            np.asarray(bundle["text_embeddings"], dtype=np.float64),
            np.asarray(bundle["image_embeddings"], dtype=np.float64),
            np.asarray(bundle["attribution_features"], dtype=np.float64),
        )
    with np.load(args.detector, allow_pickle=False) as detector:
        metadata = json.loads(str(np.asarray(detector["metadata_json"]).reshape(-1)[0]))
        view_name = str(metadata["feature_view"])
        features = views[view_name]
        weights = np.asarray(detector["weights"], dtype=np.float64)
        mean = np.asarray(detector["mean"], dtype=np.float64)
        scale = np.asarray(detector["scale"], dtype=np.float64)
        bias = float(np.asarray(detector["bias"]).reshape(-1)[0])
        threshold = float(np.asarray(detector["threshold"]).reshape(-1)[0])
    if features.shape[1] != len(weights):
        raise ValueError(f"detector expects {len(weights)} features, view {view_name} has {features.shape[1]}")
    logits = ((features - mean) / scale) @ weights + bias
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    output_rows = []
    for row, score in zip(rows, scores):
        output_rows.append({**row, "risk_score": float(score), "detector_threshold": threshold, "feature_view": view_name})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({"rows": len(rows), "feature_view": view_name, "threshold": threshold, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
