from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from AEGIS.io import (
    align_embeddings_with_metadata,
    load_embeddings,
    load_metadata,
    save_score_table,
)
from AEGIS.labels import coerce_binary_labels
from AEGIS.logistic import LogisticRegressionNumpy
from AEGIS.metrics import detection_report
from AEGIS.svd_detector import SVDMaliciousnessDetector, pseudo_labels_from_scores
from AEGIS.evaluation import threshold_from_prior


DEFAULT_DETECTOR_PATH = "models/aegis/aegis_qwen25vl3b_text_detector.npz"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="AEGIS",
        description="Malicious prompt detection utilities for MLLM guardrail experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Fit an SVD detector and score embeddings.")
    _add_common_io_args(score_parser)
    score_parser.add_argument("--output", required=True, help="CSV path for scored samples.")
    score_parser.add_argument("--components", type=int, default=8, help="Number of SVD components.")
    score_parser.add_argument(
        "--malicious-prior",
        type=float,
        default=None,
        help="If set, threshold top prior fraction as malicious.",
    )
    score_parser.add_argument(
        "--fit-split",
        default=None,
        help="Optional split value used to fit the SVD subspace, e.g. train.",
    )
    score_parser.set_defaults(func=command_score)

    classifier_parser = subparsers.add_parser(
        "pseudo-train",
        help="Fit SVD pseudo-labels and train a lightweight logistic classifier.",
    )
    _add_common_io_args(classifier_parser)
    classifier_parser.add_argument("--output", required=True, help="CSV path for classifier scores.")
    classifier_parser.add_argument("--components", type=int, default=8)
    classifier_parser.add_argument("--malicious-prior", type=float, required=True)
    classifier_parser.add_argument("--train-split", default="train")
    classifier_parser.add_argument("--test-split", default="test")
    classifier_parser.add_argument("--epochs", type=int, default=500)
    classifier_parser.add_argument("--learning-rate", type=float, default=0.1)
    classifier_parser.set_defaults(func=command_pseudo_train)

    qwen_parser = subparsers.add_parser(
        "extract-qwen25-vl",
        help="Extract Qwen2.5-VL prompt embeddings into the AEGIS NPZ format.",
    )
    qwen_parser.add_argument("--metadata", required=True, help="Corpus metadata CSV.")
    qwen_parser.add_argument("--corpus-root", required=True, help="Root directory for image paths.")
    qwen_parser.add_argument("--output", required=True, help="Output embeddings NPZ path.")
    qwen_parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    qwen_parser.add_argument("--cache-dir", default="models/huggingface")
    qwen_parser.add_argument("--layer", type=int, default=-1)
    qwen_parser.add_argument(
        "--pooling",
        choices=["last_token", "mean_tokens", "text_tokens", "image_tokens"],
        default="last_token",
    )
    qwen_parser.add_argument("--torch-dtype", default="auto")
    qwen_parser.add_argument("--device-map", default="auto")
    qwen_parser.add_argument(
        "--add-generation-prompt",
        action="store_true",
        help="Append the assistant generation prompt before extracting hidden states.",
    )
    qwen_parser.add_argument("--min-pixels", type=int, default=None)
    qwen_parser.add_argument("--max-pixels", type=int, default=None)
    qwen_parser.add_argument("--max-samples", type=int, default=None)
    qwen_parser.set_defaults(func=command_extract_qwen25_vl)

    qwen_layers_parser = subparsers.add_parser(
        "extract-qwen25-vl-layers",
        help="Extract several Qwen2.5-VL layers into separate AEGIS NPZ files.",
    )
    qwen_layers_parser.add_argument("--metadata", required=True, help="Corpus metadata CSV.")
    qwen_layers_parser.add_argument("--corpus-root", required=True, help="Root directory for image paths.")
    qwen_layers_parser.add_argument("--output-dir", required=True, help="Directory for layer NPZ files.")
    qwen_layers_parser.add_argument("--output-prefix", default=None, help="Prefix for generated NPZ files.")
    qwen_layers_parser.add_argument("--layers", type=int, nargs="+", required=True)
    qwen_layers_parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    qwen_layers_parser.add_argument("--cache-dir", default="models/huggingface")
    qwen_layers_parser.add_argument(
        "--pooling",
        choices=["last_token", "mean_tokens", "text_tokens", "image_tokens"],
        default="last_token",
    )
    qwen_layers_parser.add_argument("--torch-dtype", default="auto")
    qwen_layers_parser.add_argument("--device-map", default="auto")
    qwen_layers_parser.add_argument(
        "--add-generation-prompt",
        action="store_true",
        help="Append the assistant generation prompt before extracting hidden states.",
    )
    qwen_layers_parser.add_argument("--min-pixels", type=int, default=None)
    qwen_layers_parser.add_argument("--max-pixels", type=int, default=None)
    qwen_layers_parser.add_argument("--max-samples", type=int, default=None)
    qwen_layers_parser.set_defaults(func=command_extract_qwen25_vl_layers)

    qwen_poolings_parser = subparsers.add_parser(
        "extract-qwen25-vl-poolings",
        help="Extract several Qwen2.5-VL pooling views in one model pass.",
    )
    qwen_poolings_parser.add_argument("--metadata", required=True, help="Corpus metadata CSV.")
    qwen_poolings_parser.add_argument("--corpus-root", required=True, help="Root directory for image paths.")
    qwen_poolings_parser.add_argument("--output-dir", required=True, help="Directory for pooling NPZ files.")
    qwen_poolings_parser.add_argument("--output-prefix", default=None, help="Prefix for generated NPZ files.")
    qwen_poolings_parser.add_argument("--poolings", nargs="+", required=True, choices=["last_token", "mean_tokens", "text_tokens", "image_tokens"])
    qwen_poolings_parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    qwen_poolings_parser.add_argument("--cache-dir", default="models/huggingface")
    qwen_poolings_parser.add_argument("--layer", type=int, default=-1)
    qwen_poolings_parser.add_argument("--torch-dtype", default="auto")
    qwen_poolings_parser.add_argument("--device-map", default="auto")
    qwen_poolings_parser.add_argument("--add-generation-prompt", action="store_true")
    qwen_poolings_parser.add_argument("--min-pixels", type=int, default=None)
    qwen_poolings_parser.add_argument("--max-pixels", type=int, default=None)
    qwen_poolings_parser.add_argument("--max-samples", type=int, default=None)
    qwen_poolings_parser.set_defaults(func=command_extract_qwen25_vl_poolings)

    llava_parser = subparsers.add_parser(
        "extract-llava-onevision-poolings",
        help="Extract several LLaVA-OneVision pooling views in one model pass.",
    )
    llava_parser.add_argument("--metadata", required=True)
    llava_parser.add_argument("--corpus-root", required=True)
    llava_parser.add_argument("--output-dir", required=True)
    llava_parser.add_argument("--output-prefix", default=None)
    llava_parser.add_argument("--poolings", nargs="+", required=True, choices=["last_token", "mean_tokens", "text_tokens", "image_tokens"])
    llava_parser.add_argument("--model-id", default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    llava_parser.add_argument("--cache-dir", default="models/huggingface")
    llava_parser.add_argument("--layer", type=int, default=-1)
    llava_parser.add_argument("--torch-dtype", default="auto")
    llava_parser.add_argument("--device-map", default="auto")
    llava_parser.add_argument("--add-generation-prompt", action="store_true")
    llava_parser.add_argument("--max-image-edge", type=int, default=384)
    llava_parser.add_argument("--batch-size", type=int, default=1)
    llava_parser.add_argument("--max-batch-characters", type=int, default=None)
    llava_parser.add_argument("--max-samples", type=int, default=None)
    llava_parser.set_defaults(func=command_extract_llava_onevision_poolings)

    train_detector_parser = subparsers.add_parser(
        "train-detector",
        help="Train and save a calibrated logistic malicious-prompt detector.",
    )
    train_detector_parser.add_argument("--embeddings", required=True, nargs="+")
    train_detector_parser.add_argument("--metadata", required=True, nargs="+")
    train_detector_parser.add_argument("--output", required=True)
    train_detector_parser.add_argument("--id-column", default="sample_id")
    train_detector_parser.add_argument("--label-column", default="label")
    train_detector_parser.add_argument("--split-column", default="experiment_split")
    train_detector_parser.add_argument("--fit-split", default="fit")
    train_detector_parser.add_argument("--validation-split", default="val")
    train_detector_parser.add_argument("--test-split", default="test")
    train_detector_parser.add_argument("--learning-rate", type=float, default=0.01)
    train_detector_parser.add_argument("--l2", type=float, default=1e-4)
    train_detector_parser.add_argument("--epochs", type=int, default=300)
    train_detector_parser.add_argument("--malicious-prior", type=float, default=0.5)
    train_detector_parser.add_argument(
        "--threshold-strategy", choices=["f1", "prior", "max-fpr"], default="f1"
    )
    train_detector_parser.add_argument(
        "--max-fpr",
        type=float,
        default=0.01,
        help="Validation false-positive-rate cap used with --threshold-strategy max-fpr.",
    )
    train_detector_parser.add_argument("--uncertainty-margin", type=float, default=0.05)
    train_detector_parser.add_argument("--source", default="")
    train_detector_parser.set_defaults(func=command_train_detector)

    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify one text or image-text prompt with a saved AEGIS detector.",
    )
    text_group = classify_parser.add_mutually_exclusive_group()
    text_group.add_argument("--text", help="Prompt text to classify.")
    text_group.add_argument("--text-file", help="UTF-8 file containing prompt text.")
    classify_parser.add_argument("--image", help="Optional local image path.")
    classify_parser.add_argument("--detector", default=DEFAULT_DETECTOR_PATH)
    classify_parser.add_argument("--cache-dir", default="models/huggingface")
    classify_parser.add_argument("--torch-dtype", default="auto")
    classify_parser.add_argument("--device-map", default="auto")
    classify_parser.add_argument("--min-pixels", type=int, default=None)
    classify_parser.add_argument("--max-pixels", type=int, default=200704)
    classify_parser.add_argument("--max-image-edge", type=int, default=384)
    classify_parser.add_argument("--output", help="Optional JSON result path.")
    classify_parser.set_defaults(func=command_classify)

    guard_features_parser = subparsers.add_parser(
        "guard-features",
        help=(
            "Apply a saved detector to precomputed feature vectors from any LLM/MLLM "
            "embedding provider."
        ),
    )
    guard_features_parser.add_argument("--features", required=True, help="AEGIS NPZ features.")
    guard_features_parser.add_argument("--metadata", default=None, help="Optional metadata CSV.")
    guard_features_parser.add_argument("--detector", default=DEFAULT_DETECTOR_PATH)
    guard_features_parser.add_argument("--id-column", default="sample_id")
    guard_features_parser.add_argument("--output", help="Optional CSV decision table.")
    guard_features_parser.add_argument("--summary-output", help="Optional JSON summary path.")
    guard_features_parser.add_argument(
        "--include-metadata-column",
        action="append",
        default=[],
        help="Copy one aligned metadata column into the output decision table; repeatable.",
    )
    guard_features_parser.add_argument("--block-threshold", type=float, default=None)
    guard_features_parser.add_argument("--review-margin", type=float, default=None)
    guard_features_parser.add_argument(
        "--require-matching-provenance",
        action="store_true",
        help="Require feature NPZ model_family/model_id/pooling to match the detector.",
    )
    guard_features_parser.set_defaults(func=command_guard_features)

    guard_request_parser = subparsers.add_parser(
        "guard-request",
        help=(
            "Apply a saved detector to a live prompt request through a provider "
            "implementing the AEGIS EmbeddingProvider contract."
        ),
    )
    guard_request_parser.add_argument("--provider", required=True, help="Provider spec: module:object.")
    guard_request_parser.add_argument("--detector", default=DEFAULT_DETECTOR_PATH)
    text_group = guard_request_parser.add_mutually_exclusive_group()
    text_group.add_argument("--text", help="Prompt text to guard.")
    text_group.add_argument("--text-file", help="UTF-8 file containing prompt text.")
    guard_request_parser.add_argument("--image", action="append", default=[], help="Optional image path; repeatable.")
    guard_request_parser.add_argument("--request-id", default=None)
    guard_request_parser.add_argument(
        "--payload-json",
        default=None,
        help="Optional JSON payload file. Supports a single request or {'requests': [...]} batch.",
    )
    guard_request_parser.add_argument("--output", help="Optional JSON response path.")
    guard_request_parser.add_argument("--block-threshold", type=float, default=None)
    guard_request_parser.add_argument("--review-margin", type=float, default=None)
    guard_request_parser.add_argument(
        "--action-on-error",
        choices=["allow", "review", "block"],
        default="review",
    )
    guard_request_parser.add_argument("--require-matching-provenance", action="store_true")
    guard_request_parser.add_argument("--hash-images", action="store_true")
    guard_request_parser.set_defaults(func=command_guard_request)

    inspect_parser = subparsers.add_parser(
        "inspect-artifact",
        help="Inspect a saved AEGIS detector artifact without loading any MLLM weights.",
    )
    inspect_parser.add_argument("--detector", default=DEFAULT_DETECTOR_PATH)
    inspect_parser.set_defaults(func=command_inspect_artifact)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run lightweight readiness checks for a detector artifact and local cache.",
    )
    doctor_parser.add_argument("--detector", default=DEFAULT_DETECTOR_PATH)
    doctor_parser.add_argument("--cache-dir", default="models/huggingface")
    doctor_parser.add_argument(
        "--require-cache-dir",
        action="store_true",
        help="Fail if --cache-dir is absent. Model files are not loaded by this check.",
    )
    doctor_parser.set_defaults(func=command_doctor)

    args = parser.parse_args()
    args.func(args)


def _add_common_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embeddings", required=True, help="NPZ file containing embeddings.")
    parser.add_argument("--metadata", default=None, help="CSV metadata file.")
    parser.add_argument("--id-column", default="sample_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--split-column", default="split")


def command_score(args: argparse.Namespace) -> None:
    embeddings, sample_ids = load_embeddings(args.embeddings)
    metadata = load_metadata(args.metadata)
    embeddings, metadata, aligned_ids = align_embeddings_with_metadata(
        embeddings, sample_ids, metadata, id_column=args.id_column
    )

    fit_embeddings = embeddings
    if args.fit_split is not None:
        _require_metadata_column(metadata, args.split_column)
        fit_mask = metadata[args.split_column].astype(str).to_numpy() == args.fit_split
        if not np.any(fit_mask):
            raise ValueError(f"No rows found for fit split {args.fit_split!r}.")
        fit_embeddings = embeddings[fit_mask]

    detector = SVDMaliciousnessDetector(n_components=args.components).fit(fit_embeddings)
    scores = detector.score_samples(embeddings)

    predictions = None
    if args.malicious_prior is not None:
        predictions = detector.predict_from_prior(scores, args.malicious_prior)

    save_score_table(args.output, aligned_ids, scores, metadata, predictions=predictions)

    report = _maybe_report(metadata, args.label_column, scores, predictions)
    _print_json(
        {
            "output": str(Path(args.output)),
            "n_samples": int(len(scores)),
            "n_fit_samples": int(len(fit_embeddings)),
            "n_components": int(detector.components_.shape[0]),
            "metrics": report,
        }
    )


def command_pseudo_train(args: argparse.Namespace) -> None:
    embeddings, sample_ids = load_embeddings(args.embeddings)
    metadata = load_metadata(args.metadata)
    embeddings, metadata, aligned_ids = align_embeddings_with_metadata(
        embeddings, sample_ids, metadata, id_column=args.id_column
    )
    _require_metadata_column(metadata, args.split_column)

    split_values = metadata[args.split_column].astype(str).to_numpy()
    train_mask = split_values == args.train_split
    test_mask = split_values == args.test_split
    if not np.any(train_mask):
        raise ValueError(f"No rows found for train split {args.train_split!r}.")
    if not np.any(test_mask):
        raise ValueError(f"No rows found for test split {args.test_split!r}.")

    detector = SVDMaliciousnessDetector(n_components=args.components).fit(embeddings[train_mask])
    train_scores = detector.score_samples(embeddings[train_mask])
    pseudo_indices, pseudo_labels = pseudo_labels_from_scores(
        train_scores,
        malicious_prior=args.malicious_prior,
    )

    classifier = LogisticRegressionNumpy(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    ).fit(embeddings[train_mask][pseudo_indices], pseudo_labels)

    probabilities = classifier.predict_proba(embeddings)
    predictions = (probabilities >= 0.5).astype(np.int64)
    save_score_table(args.output, aligned_ids, probabilities, metadata, predictions=predictions)

    test_report = None
    if args.label_column in metadata:
        labels = coerce_binary_labels(metadata[args.label_column])
        test_report = detection_report(
            labels[test_mask],
            probabilities[test_mask],
            predictions[test_mask],
        )

    _print_json(
        {
            "output": str(Path(args.output)),
            "n_train_samples": int(np.sum(train_mask)),
            "n_test_samples": int(np.sum(test_mask)),
            "n_pseudo_samples": int(len(pseudo_indices)),
            "n_pseudo_malicious": int(np.sum(pseudo_labels == 1)),
            "n_components": int(detector.components_.shape[0]),
            "test_metrics": test_report,
        }
    )


def command_extract_qwen25_vl(args: argparse.Namespace) -> None:
    from AEGIS.adapters.qwen25_vl import (
        Qwen25VLExtractionConfig,
        extract_qwen25_vl_embeddings,
    )

    config = Qwen25VLExtractionConfig(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        layer=args.layer,
        pooling=args.pooling,
        add_generation_prompt=args.add_generation_prompt,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_samples=args.max_samples,
    )
    extract_qwen25_vl_embeddings(
        metadata_path=args.metadata,
        corpus_root=args.corpus_root,
        output_path=args.output,
        config=config,
    )
    _print_json(
        {
            "output": str(Path(args.output)),
            "model_id": args.model_id,
            "layer": args.layer,
            "pooling": args.pooling,
        }
    )


def command_extract_qwen25_vl_layers(args: argparse.Namespace) -> None:
    from AEGIS.adapters.qwen25_vl import (
        Qwen25VLExtractionConfig,
        extract_qwen25_vl_layer_embeddings,
    )

    config = Qwen25VLExtractionConfig(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        pooling=args.pooling,
        add_generation_prompt=args.add_generation_prompt,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_samples=args.max_samples,
    )
    paths = extract_qwen25_vl_layer_embeddings(
        metadata_path=args.metadata,
        corpus_root=args.corpus_root,
        output_dir=args.output_dir,
        layers=args.layers,
        config=config,
        output_prefix=args.output_prefix,
    )
    _print_json(
        {
            "output_dir": str(Path(args.output_dir)),
            "model_id": args.model_id,
            "layers": {str(layer): str(path) for layer, path in paths.items()},
            "pooling": args.pooling,
        }
    )


def command_extract_qwen25_vl_poolings(args: argparse.Namespace) -> None:
    from AEGIS.adapters.qwen25_vl import (
        Qwen25VLExtractionConfig,
        extract_qwen25_vl_pooling_embeddings,
    )

    config = Qwen25VLExtractionConfig(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        layer=args.layer,
        add_generation_prompt=args.add_generation_prompt,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_samples=args.max_samples,
    )
    paths = extract_qwen25_vl_pooling_embeddings(
        metadata_path=args.metadata,
        corpus_root=args.corpus_root,
        output_dir=args.output_dir,
        poolings=args.poolings,
        config=config,
        output_prefix=args.output_prefix,
    )
    _print_json(
        {
            "output_dir": str(Path(args.output_dir)),
            "model_id": args.model_id,
            "layer": args.layer,
            "poolings": {str(pooling): str(path) for pooling, path in paths.items()},
        }
    )


def command_extract_llava_onevision_poolings(args: argparse.Namespace) -> None:
    from AEGIS.adapters.llava_onevision import (
        LlavaOnevisionExtractionConfig,
        extract_llava_onevision_pooling_embeddings,
    )

    config = LlavaOnevisionExtractionConfig(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        layer=args.layer,
        add_generation_prompt=args.add_generation_prompt,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        max_image_edge=args.max_image_edge,
        batch_size=args.batch_size,
        max_batch_characters=args.max_batch_characters,
        max_samples=args.max_samples,
    )
    paths = extract_llava_onevision_pooling_embeddings(
        metadata_path=args.metadata,
        corpus_root=args.corpus_root,
        output_dir=args.output_dir,
        poolings=args.poolings,
        config=config,
        output_prefix=args.output_prefix,
    )
    _print_json(
        {
            "output_dir": str(Path(args.output_dir)),
            "model_id": args.model_id,
            "layer": args.layer,
            "poolings": {str(pooling): str(path) for pooling, path in paths.items()},
        }
    )


def command_train_detector(args: argparse.Namespace) -> None:
    from AEGIS.detector_artifact import (
        DetectorArtifact,
        infer_embedding_provenance,
        save_detector_artifact,
    )

    if len(args.embeddings) != len(args.metadata):
        raise ValueError("--embeddings and --metadata must contain the same number of paths.")
    feature_parts = []
    label_parts = []
    split_parts = []
    provenances = []
    for embedding_path, metadata_path in zip(args.embeddings, args.metadata, strict=True):
        source_features, sample_ids = load_embeddings(embedding_path)
        source_metadata = load_metadata(metadata_path)
        source_features, source_metadata, _ = align_embeddings_with_metadata(
            source_features, sample_ids, source_metadata, id_column=args.id_column
        )
        _require_metadata_column(source_metadata, args.label_column)
        _require_metadata_column(source_metadata, args.split_column)
        feature_parts.append(source_features)
        label_parts.append(coerce_binary_labels(source_metadata[args.label_column]))
        split_parts.append(source_metadata[args.split_column].astype(str).to_numpy())
        provenances.append(infer_embedding_provenance(embedding_path))
    _require_matching_provenance(provenances)
    embeddings = np.concatenate(feature_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    splits = np.concatenate(split_parts, axis=0)
    fit_mask = splits == args.fit_split
    validation_mask = splits == args.validation_split
    test_mask = splits == args.test_split
    if not np.any(fit_mask):
        raise ValueError(f"No rows found for fit split {args.fit_split!r}.")
    if not np.any(validation_mask):
        raise ValueError(
            f"No rows found for validation split {args.validation_split!r}."
        )

    classifier = LogisticRegressionNumpy(
        learning_rate=args.learning_rate,
        l2=args.l2,
        epochs=args.epochs,
    ).fit(embeddings[fit_mask], labels[fit_mask])
    validation_scores = classifier.predict_proba(embeddings[validation_mask])
    if args.threshold_strategy == "f1":
        threshold = _threshold_for_best_f1(labels[validation_mask], validation_scores)
    elif args.threshold_strategy == "prior":
        threshold = threshold_from_prior(validation_scores, args.malicious_prior)
    else:
        threshold = _threshold_for_max_fpr(
            labels[validation_mask], validation_scores, args.max_fpr
        )
    validation_predictions = (validation_scores >= threshold).astype(np.int64)
    provenance = provenances[0]
    artifact = DetectorArtifact(
        classifier=classifier,
        threshold=threshold,
        uncertainty_margin=args.uncertainty_margin,
        model_family=str(provenance["model_family"]),
        model_id=str(provenance["model_id"]),
        layer=int(provenance["layer"]),
        pooling=str(provenance["pooling"]),
        source=args.source,
    )
    output = save_detector_artifact(args.output, artifact)

    test_metrics = None
    if np.any(test_mask):
        test_scores = classifier.predict_proba(embeddings[test_mask])
        test_predictions = (test_scores >= threshold).astype(np.int64)
        test_metrics = detection_report(
            labels[test_mask], test_scores, test_predictions
        )
    _print_json(
        {
            "output": str(output),
            "feature_dim": artifact.feature_dim,
            "model_family": artifact.model_family,
            "model_id": artifact.model_id,
            "layer": artifact.layer,
            "pooling": artifact.pooling,
            "threshold": artifact.threshold,
            "uncertainty_margin": artifact.uncertainty_margin,
            "threshold_strategy": args.threshold_strategy,
            "n_sources": len(feature_parts),
            "n_fit_samples": int(np.sum(fit_mask)),
            "n_validation_samples": int(np.sum(validation_mask)),
            "n_test_samples": int(np.sum(test_mask)),
            "validation_metrics": detection_report(
                labels[validation_mask], validation_scores, validation_predictions
            ),
            "test_metrics": test_metrics,
        }
    )


def command_classify(args: argparse.Namespace) -> None:
    from AEGIS.detector_artifact import load_detector_artifact

    artifact = load_detector_artifact(args.detector)
    image_path = None if args.image is None else Path(args.image)
    if image_path is not None and not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    if image_path is not None and not image_path.is_file():
        raise FileNotFoundError(f"Image path is not a file: {image_path}")
    prompt_text = (
        str(args.text)
        if args.text is not None
        else Path(args.text_file).read_text(encoding="utf-8")
        if args.text_file is not None
        else ""
    )
    if not prompt_text.strip() and image_path is None:
        raise ValueError("Provide prompt text, an image, or both.")
    if artifact.model_family == "llava_onevision" and image_path is None:
        raise ValueError("This LLaVA detector requires --image for every prompt.")

    with tempfile.TemporaryDirectory(prefix="aegis-classify-") as directory:
        scratch = Path(directory)
        staged_image_path = _stage_inference_image(image_path, scratch)
        metadata_path = scratch / "prompt.csv"
        with metadata_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["sample_id", "text", "image_path"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "sample_id": "prompt",
                    "text": prompt_text,
                    "image_path": staged_image_path,
                }
            )
        embedding_path = _extract_inference_embedding(
            artifact=artifact,
            metadata_path=metadata_path,
            scratch=scratch,
            args=args,
        )
        features, _ = load_embeddings(embedding_path)

    result = artifact.classify(features)[0]
    result.update(
        {
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "image_supplied": image_path is not None,
            "model_family": artifact.model_family,
            "model_id": artifact.model_id,
            "layer": artifact.layer,
            "pooling": artifact.pooling,
            "detector_source": artifact.source,
            "warning": (
                "This is a research risk estimate, not proof of intent or universal "
                "policy compliance. Review uncertain, image-only, and high-impact decisions."
            ),
        }
    )
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    _print_json(result)


def command_guard_features(args: argparse.Namespace) -> None:
    from AEGIS.detector_artifact import infer_embedding_provenance, load_detector_artifact
    from AEGIS.guardrail import (
        GuardrailPolicy,
        GuardrailRuntime,
        decision_summary,
        decisions_to_frame,
    )

    artifact = load_detector_artifact(args.detector)
    features, sample_ids = load_embeddings(args.features)
    metadata = load_metadata(args.metadata)
    features, metadata, aligned_ids = align_embeddings_with_metadata(
        features,
        sample_ids,
        metadata,
        id_column=args.id_column,
    )
    if args.require_matching_provenance:
        provenance = infer_embedding_provenance(args.features)
        for key in ("model_family", "model_id", "pooling"):
            expected = getattr(artifact, key)
            actual = provenance[key]
            if actual != expected:
                raise ValueError(
                    f"Feature provenance mismatch for {key}: {actual!r} != {expected!r}."
                )

    runtime = GuardrailRuntime(
        artifact,
        policy=GuardrailPolicy(
            block_threshold=args.block_threshold,
            review_margin=args.review_margin,
            require_matching_provenance=args.require_matching_provenance,
        ),
    )
    decisions = runtime.evaluate_metadata_features(
        features,
        metadata=metadata,
        sample_ids=aligned_ids,
    )
    table = decisions_to_frame(decisions)
    if args.include_metadata_column:
        if metadata is None:
            raise ValueError("--include-metadata-column requires --metadata.")
        for column in args.include_metadata_column:
            if column not in metadata:
                raise ValueError(f"Metadata is missing requested column {column!r}.")
            table[column] = metadata[column].reset_index(drop=True).to_numpy()
    summary = decision_summary(decisions)
    summary.update(
        {
            "detector": {
                "model_family": artifact.model_family,
                "model_id": artifact.model_id,
                "pooling": artifact.pooling,
                "threshold": runtime.threshold,
                "review_margin": runtime.review_margin,
                "source": artifact.source,
            },
            "features": str(Path(args.features)),
        }
    )

    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
        summary["output"] = str(output)
    if args.summary_output is not None:
        summary_output = Path(args.summary_output)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summary["summary_output"] = str(summary_output)
    _print_json(summary)


def command_guard_request(args: argparse.Namespace) -> None:
    from AEGIS.detector_artifact import load_detector_artifact
    from AEGIS.guardrail import GuardrailPolicy
    from AEGIS.provider_contract import load_provider
    from AEGIS.service import evaluate_request_payload

    artifact = load_detector_artifact(args.detector)
    provider = load_provider(args.provider)
    policy = GuardrailPolicy(
        block_threshold=args.block_threshold,
        review_margin=args.review_margin,
        action_on_error=args.action_on_error,
        require_matching_provenance=args.require_matching_provenance,
        hash_images=args.hash_images,
    )
    payload = _guard_request_payload(args)
    response = evaluate_request_payload(artifact, provider, payload, policy=policy)

    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(response, indent=2, sort_keys=True), encoding="utf-8")
    _print_json(response)


def command_inspect_artifact(args: argparse.Namespace) -> None:
    from AEGIS.readiness import inspect_detector_artifact

    _print_json(inspect_detector_artifact(args.detector))


def command_doctor(args: argparse.Namespace) -> None:
    from AEGIS.readiness import doctor

    report = doctor(
        detector_path=args.detector,
        cache_dir=args.cache_dir,
        require_cache_dir=args.require_cache_dir,
    )
    _print_json(report)
    if not bool(report["ok"]):
        raise SystemExit(1)


def _stage_inference_image(image_path: Path | None, scratch: Path) -> str:
    if image_path is None:
        return ""
    target = scratch / f"prompt_image{image_path.suffix}"
    shutil.copy2(image_path, target)
    return target.name


def _extract_inference_embedding(artifact, metadata_path: Path, scratch: Path, args) -> Path:
    _quiet_transformers_progress()
    if artifact.model_family == "qwen25_vl":
        from AEGIS.adapters.qwen25_vl import (
            Qwen25VLExtractionConfig,
            extract_qwen25_vl_embeddings,
        )

        output_path = scratch / "embedding.npz"
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
            corpus_root=scratch,
            output_path=output_path,
            config=config,
        )
        return output_path
    if artifact.model_family == "llava_onevision":
        from AEGIS.adapters.llava_onevision import (
            LlavaOnevisionExtractionConfig,
            extract_llava_onevision_pooling_embeddings,
        )

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
            corpus_root=scratch,
            output_dir=scratch,
            poolings=[artifact.pooling],
            config=config,
            output_prefix="prompt",
        )
        return paths[artifact.pooling]
    raise ValueError(f"Unsupported model family: {artifact.model_family!r}.")


def _quiet_transformers_progress() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except ImportError:
        pass


def _guard_request_payload(args: argparse.Namespace) -> dict:
    if args.payload_json is not None:
        if args.text is not None or args.text_file is not None or args.image or args.request_id is not None:
            raise ValueError("--payload-json cannot be combined with --text, --text-file, --image, or --request-id.")
        return json.loads(Path(args.payload_json).read_text(encoding="utf-8"))

    prompt_text = (
        str(args.text)
        if args.text is not None
        else Path(args.text_file).read_text(encoding="utf-8")
        if args.text_file is not None
        else ""
    )
    if not prompt_text.strip() and not args.image:
        raise ValueError("Provide --text, --text-file, --image, or --payload-json.")
    return {
        "text": prompt_text,
        "image_paths": args.image,
        "request_id": args.request_id,
    }


def _require_matching_provenance(provenances: list[dict[str, object]]) -> None:
    if not provenances:
        raise ValueError("At least one training source is required.")
    expected = provenances[0]
    for provenance in provenances[1:]:
        for key in ("model_family", "model_id", "layer", "pooling"):
            if provenance[key] != expected[key]:
                raise ValueError(
                    f"Training embeddings disagree on {key}: "
                    f"{expected[key]!r} != {provenance[key]!r}."
                )


def _threshold_for_best_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(np.asarray(scores, dtype=np.float64))
    if len(candidates) == 0:
        raise ValueError("Cannot select a threshold from empty validation scores.")
    best_threshold = float(candidates[0])
    best_key = (-1.0, -1.0, -1.0)
    for candidate in candidates:
        predictions = (scores >= candidate).astype(np.int64)
        report = detection_report(labels, scores, predictions)
        key = (report["f1"], report["recall"], report["precision"])
        if key > best_key:
            best_key = key
            best_threshold = float(candidate)
    return best_threshold


def _threshold_for_max_fpr(
    labels: np.ndarray, scores: np.ndarray, max_fpr: float
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be 1D arrays with the same shape.")
    if not 0.0 <= max_fpr < 1.0:
        raise ValueError("max_fpr must be in [0, 1).")
    if len(scores) == 0:
        raise ValueError("Cannot select a threshold from empty validation scores.")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite.")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("max-fpr thresholding requires benign and malicious labels.")

    best_threshold = None
    best_key = (-1.0, -1.0, 0.0)
    for candidate in np.unique(scores):
        predictions = scores >= candidate
        false_positives = np.sum((labels == 0) & predictions)
        true_negatives = np.sum((labels == 0) & ~predictions)
        fpr = false_positives / (false_positives + true_negatives)
        if fpr > max_fpr:
            continue
        true_positives = np.sum((labels == 1) & predictions)
        false_negatives = np.sum((labels == 1) & ~predictions)
        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives > 0
            else 0.0
        )
        recall = true_positives / (true_positives + false_negatives)
        # Prefer recall under the FPR cap, then precision, then the lowest
        # threshold so the selected operating point is not stricter than needed.
        key = (float(recall), float(precision), -float(candidate))
        if key > best_key:
            best_key = key
            best_threshold = float(candidate)

    if best_threshold is None:
        return float(np.nextafter(np.max(scores), 1.0))
    return best_threshold


def _maybe_report(metadata, label_column: str, scores: np.ndarray, predictions):
    if metadata is None or label_column not in metadata:
        return None
    labels = coerce_binary_labels(metadata[label_column])
    return detection_report(labels, scores, predictions)


def _require_metadata_column(metadata, column: str) -> None:
    if metadata is None:
        raise ValueError("Metadata is required for this command.")
    if column not in metadata:
        raise ValueError(f"Metadata must contain column {column!r}.")


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
