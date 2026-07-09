from __future__ import annotations

import argparse
import json
from pathlib import Path

from AEGIS.cli import DEFAULT_DETECTOR_PATH
from AEGIS.detector_artifact import load_detector_artifact
from AEGIS.io import align_embeddings_with_metadata, load_embeddings
from AEGIS.litmus import LITMUS_SOURCE, evaluate_litmus_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the default AEGIS classify detector on the held-out litmus panel."
    )
    parser.add_argument(
        "--metadata",
        default=f"data/processed/{LITMUS_SOURCE}/metadata.csv",
    )
    parser.add_argument(
        "--corpus-root",
        default=f"data/processed/{LITMUS_SOURCE}",
    )
    parser.add_argument("--detector", default=DEFAULT_DETECTOR_PATH)
    parser.add_argument("--cache-dir", default="models/huggingface")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--max-image-edge", type=int, default=384)
    parser.add_argument(
        "--embeddings",
        default=f"outputs/{LITMUS_SOURCE}_qwen25vl3b_text_tokens.npz",
    )
    parser.add_argument(
        "--scores-output",
        default=f"outputs/{LITMUS_SOURCE}_scores.csv",
    )
    parser.add_argument(
        "--summary-output",
        default=f"outputs/{LITMUS_SOURCE}_summary.json",
    )
    parser.add_argument(
        "--refresh-embeddings",
        action="store_true",
        help="Re-extract features even when the embedding file already exists.",
    )
    parser.add_argument(
        "--require-perfect",
        action="store_true",
        help="Exit non-zero if any false positive or false negative is observed.",
    )
    args = parser.parse_args()

    artifact = load_detector_artifact(args.detector)
    metadata_path = Path(args.metadata)
    embeddings_path = Path(args.embeddings)
    if args.refresh_embeddings or not embeddings_path.exists():
        _extract_embeddings(
            artifact=artifact,
            metadata_path=metadata_path,
            corpus_root=Path(args.corpus_root),
            output_path=embeddings_path,
            args=args,
        )

    features, sample_ids = load_embeddings(embeddings_path)
    import pandas as pd

    metadata = pd.read_csv(metadata_path)
    features, metadata, _ = align_embeddings_with_metadata(
        features,
        sample_ids,
        metadata,
        id_column="sample_id",
    )
    if metadata is None:
        raise ValueError("Litmus metadata is required.")

    evaluation = evaluate_litmus_predictions(artifact, features, metadata)
    scores_output = Path(args.scores_output)
    summary_output = Path(args.summary_output)
    scores_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    evaluation.scores.to_csv(scores_output, index=False)
    summary_output.write_text(
        json.dumps(evaluation.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(evaluation.summary, indent=2, sort_keys=True))

    if args.require_perfect and not bool(evaluation.summary["perfect"]):
        raise SystemExit(
            "Litmus classification was not perfect; inspect "
            f"{scores_output} for failing sample ids."
        )


def _extract_embeddings(*, artifact, metadata_path: Path, corpus_root: Path, output_path: Path, args) -> None:
    if artifact.model_family == "qwen25_vl":
        from AEGIS.adapters.qwen25_vl import (
            Qwen25VLExtractionConfig,
            extract_qwen25_vl_embeddings,
        )

        config = Qwen25VLExtractionConfig(
            model_id=artifact.model_id,
            cache_dir=args.cache_dir,
            layer=artifact.layer,
            pooling=artifact.pooling,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        extract_qwen25_vl_embeddings(
            metadata_path=metadata_path,
            corpus_root=corpus_root,
            output_path=output_path,
            config=config,
        )
        return

    if artifact.model_family == "llava_onevision":
        from AEGIS.adapters.llava_onevision import (
            LlavaOnevisionExtractionConfig,
            extract_llava_onevision_pooling_embeddings,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        config = LlavaOnevisionExtractionConfig(
            model_id=artifact.model_id,
            cache_dir=args.cache_dir,
            layer=artifact.layer,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            max_image_edge=args.max_image_edge,
        )
        paths = extract_llava_onevision_pooling_embeddings(
            metadata_path=metadata_path,
            corpus_root=corpus_root,
            output_dir=output_path.parent,
            poolings=[artifact.pooling],
            config=config,
            output_prefix=output_path.stem,
        )
        extracted = paths[artifact.pooling]
        if extracted != output_path:
            output_path.write_bytes(extracted.read_bytes())
        return

    raise ValueError(f"Unsupported model family: {artifact.model_family!r}.")


if __name__ == "__main__":
    main()
