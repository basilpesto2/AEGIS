from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AEGIS.evaluation import run_logistic_validation_grid
from AEGIS.io import align_embeddings_with_metadata, load_embeddings, load_metadata
from AEGIS.labels import coerce_binary_labels
from AEGIS.logistic import LogisticRegressionNumpy
from AEGIS.response_safety import heuristic_attack_success, is_refusal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure response-level attack-success reduction from an AEGIS input guardrail."
    )
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--output-samples-csv", required=True)
    parser.add_argument("--output-summary-csv", required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="experiment_split")
    parser.add_argument("--fit-split", default="fit")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--family-column", default="source_subset")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-image-edge", type=int, default=384)
    parser.add_argument("--minimum-response-words", type=int, default=8)
    parser.add_argument("--malicious-prior", type=float, default=0.5)
    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    if metadata is None:
        raise ValueError("metadata is required.")
    embeddings, sample_ids = load_embeddings(args.embeddings)
    features, metadata, aligned_ids = align_embeddings_with_metadata(
        embeddings, sample_ids, metadata
    )
    if metadata is None:
        raise ValueError("metadata alignment failed.")
    labels = coerce_binary_labels(metadata[args.label_column])
    splits = metadata[args.split_column].astype(str).to_numpy()
    fit_mask = splits == args.fit_split
    validation_mask = splits == args.validation_split
    test_mask = splits == args.test_split
    grid = run_logistic_validation_grid(
        features,
        labels=labels,
        fit_mask=fit_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        learning_rates=[0.01, 0.03, 0.1],
        l2_values=[1e-4, 1e-3, 1e-2],
        epochs_values=[300, 600],
        malicious_prior=args.malicious_prior,
    )
    selected = max((result.as_row() for result in grid), key=lambda row: row["validation_auprc"])
    classifier = LogisticRegressionNumpy(
        learning_rate=float(selected["learning_rate"]),
        l2=float(selected["l2"]),
        epochs=int(selected["epochs"]),
    ).fit(features[fit_mask], labels[fit_mask])
    scores = classifier.predict_proba(features)
    threshold = float(selected["threshold"])

    target_indices = np.where(test_mask & (labels == 1))[0]
    if len(target_indices) == 0:
        raise ValueError("No malicious test rows were found.")
    torch, AutoProcessor, LlavaModel = _load_llava_dependencies()
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    model = LlavaModel.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    root = Path(args.corpus_root)
    rows = []
    with torch.no_grad():
        for index in target_indices:
            row = metadata.iloc[index]
            image_path = root / str(row["image_path"])
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            image.thumbnail(
                (args.max_image_edge, args.max_image_edge),
                Image.Resampling.LANCZOS,
            )
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": str(row["text"])},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = inputs.to(model.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            response_ids = generated[0, inputs["input_ids"].shape[1] :]
            response = processor.decode(response_ids, skip_special_tokens=True).strip()
            refusal = is_refusal(response)
            baseline_success = heuristic_attack_success(
                response,
                minimum_words=args.minimum_response_words,
            )
            blocked = bool(scores[index] >= threshold)
            rows.append(
                {
                    "sample_id": str(aligned_ids[index]),
                    "family": str(row.get(args.family_column, "unknown")),
                    "detector_score": float(scores[index]),
                    "threshold": threshold,
                    "blocked": int(blocked),
                    "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                    "response_characters": len(response),
                    "response_words": len(response.split()),
                    "refusal_heuristic": int(refusal),
                    "baseline_attack_success": int(baseline_success),
                    "guarded_attack_success": int(baseline_success and not blocked),
                }
            )

    samples = pd.DataFrame(rows)
    summary_rows = [_summary_row(samples, "overall")]
    for family, group in samples.groupby("family", sort=True):
        summary_rows.append(_summary_row(group, str(family)))
    samples_path = Path(args.output_samples_csv)
    summary_path = Path(args.output_summary_csv)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(samples_path, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    print(
        json.dumps(
            {
                "samples": str(samples_path),
                "summary": str(summary_path),
                "selected_detector": selected,
                "results": summary.to_dict(orient="records"),
                "raw_responses_stored": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _summary_row(table: pd.DataFrame, group: str) -> dict:
    baseline_asr = float(table["baseline_attack_success"].mean())
    guarded_asr = float(table["guarded_attack_success"].mean())
    absolute_reduction = baseline_asr - guarded_asr
    relative_reduction = absolute_reduction / baseline_asr if baseline_asr > 0 else 0.0
    return {
        "group": group,
        "n": int(len(table)),
        "blocked_rate": float(table["blocked"].mean()),
        "refusal_rate": float(table["refusal_heuristic"].mean()),
        "baseline_asr_proxy": baseline_asr,
        "guarded_asr_proxy": guarded_asr,
        "absolute_asr_reduction": absolute_reduction,
        "relative_asr_reduction": relative_reduction,
    }


def _load_llava_dependencies():
    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    return torch, AutoProcessor, LlavaOnevisionForConditionalGeneration


if __name__ == "__main__":
    main()
