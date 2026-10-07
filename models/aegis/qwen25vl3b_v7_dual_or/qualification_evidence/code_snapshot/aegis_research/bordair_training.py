from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable
import zipfile

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
RUN = REPOSITORY / "outputs" / "bordair_retraining_v1"
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
PIPELINE = REPOSITORY / "deliverables" / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from AEGIS.adapters.llava_onevision import (  # noqa: E402
    LlavaOnevisionExtractionConfig,
    clear_llava_runtime_cache,
    extract_llava_onevision_pooling_embeddings,
    llava_preprocessing_sha256,
)
from AEGIS.adapters.qwen25_vl import (  # noqa: E402
    Qwen25VLExtractionConfig,
    clear_qwen_runtime_cache,
    extract_qwen25_vl_pooling_embeddings,
    qwen_preprocessing_sha256,
)
from AEGIS.detector_artifact import (  # noqa: E402
    DetectorArtifact,
    save_detector_artifact,
)
from AEGIS.logistic import LogisticRegressionNumpy  # noqa: E402
from aegis_research.metrics import classification_metrics  # noqa: E402
from .bordair_corpus import validate_run  # noqa: E402
from .bordair_provenance import (  # noqa: E402
    DETECTOR_CORPUS_VERSION,
    detector_source_label,
    training_identity_sha256,
)


DEVELOPMENT_METADATA = RUN / "development_metadata_v7.csv"
REGRESSION_METADATA = RUN / "regression_metadata_v7.csv"
CORPUS_MANIFEST = RUN / "corpus_manifest_v7.json"
V6_METADATA = {
    "development": RUN / "development_metadata.csv",
    "regression": RUN / "regression_metadata.csv",
}
V6_FEATURE_DIRECTORY = "features_v6"
V6_FEATURE_MANIFEST_SHA256 = {
    "llava05b": {
        "development": "bbc0471013f32e36670ebb82c7b28f464466847264f3f8599b487fff278e49e4",
        "regression": "e6bc6447d5c7976018d472818512b6daae00a69a8e6a3885153ec8c2f6c66a00",
    },
    "qwen25vl3b": {
        "development": "bb1ac387a10af2502b588ab02c0c793d3c9444baafa2b6c7c1dafeced683c793",
        "regression": "4965fddae7cd93590f220617ebb8455995f862893ab2057e287641ca23d8b67c",
    },
}
PRIOR_V7_FEATURE_ROOT = RUN / "features_v7_trials" / "pre_hard_negative"
SCHEMA3_FEATURE_ROOT = (
    RUN / "features_v7_trials" / "schema3_pre_additional_hard_negatives"
)
SCHEMA3_ARCHIVE_MANIFEST_SHA256 = "3267339b2a97a14b8568974026836fd4b823daa16f84652de019aacbb5e170eb"
LLAVA_CACHE = REPOSITORY / "outputs" / "bordair_eval" / "llava_model_cache"
QWEN_CACHE = REPOSITORY / "outputs" / "bordair_eval" / "model_cache"
EXTRACTION_POOLINGS = ("text_tokens", "image_tokens")
CANDIDATE_POOLINGS = (*EXTRACTION_POOLINGS, "text_image_tokens")
FEATURE_DIRECTORY = "features_v7"
TRAINING_DIRECTORY = "training_v7"
CORPUS_VERSION = DETECTOR_CORPUS_VERSION
CANDIDATE_LEARNING_RATES = (0.02, 0.05)
CANDIDATE_L2_VALUES = (1e-3, 1e-2, 1e-1, 3e-1, 1.0)
CANDIDATE_EPOCHS = 600
CANDIDATE_STANDARDIZE = True
CANDIDATE_RANDOM_SEED = 42
IMAGE_ONLY_STRATEGY = "benign_text_full_injection"
TEXT_LED_STRATEGY = "malicious_text_benign_image_counterfactual"
HARD_NEGATIVE_CATEGORIES = {
    "binary_structured",
    "code_errors",
    "credentials_auth",
    "defensive_security",
    "foreign_terse",
    "ignore_rule",
    "lexical_false_friends",
    "policy_legal",
    "prior_v7_test_feedback",
    "prior_v7_validation_feedback",
    "prompt_meta",
    "sysadmin_tools",
}
MAX_VALIDATION_FALSE_POSITIVE_RATE = 2.0 / 80.0
MAX_INTERNAL_TEST_FALSE_POSITIVE_RATE = 0.10

TARGETS = {
    "llava05b": {
        "model_family": "llava_onevision",
        "model_id": "models\\huggingface\\llava-onevision-qwen2-0.5b-ov-hf",
        "model_revision": (
            "c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c"
        ),
        "tokenizer_revision": (
            "c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c"
        ),
        "feature_dim": 896,
    },
    "qwen25vl3b": {
        "model_family": "qwen25_vl",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "tokenizer_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "feature_dim": 2048,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=tuple(TARGETS))
    parser.add_argument(
        "--artifact-mode",
        choices=("fused", "dual_or"),
        default="fused",
        help=(
            "Emit the historical fused detector or a validation-selected image/text "
            "artifact pair composed with OR (default: fused)."
        ),
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract development and regression features without fitting a head.",
    )
    parser.add_argument(
        "--model-id",
        help=(
            "Override the configured model identifier. For Qwen this is also the "
            "identifier or local path loaded by Transformers."
        ),
    )
    parser.add_argument(
        "--runtime-model-id",
        help=(
            "Override the local LLaVA model path without changing the canonical "
            "model identifier recorded in feature provenance."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        help="Override the Hugging Face cache directory used during extraction.",
    )
    parser.add_argument(
        "--model-revision",
        help="Override the pinned model revision used for extraction and provenance.",
    )
    parser.add_argument(
        "--tokenizer-revision",
        help="Override the pinned tokenizer revision used for extraction and provenance.",
    )
    parser.add_argument(
        "--qwen-chunk-size",
        type=int,
        default=32,
        help="Rows per resumable Qwen extraction chunk (default: 32).",
    )
    parser.add_argument(
        "--candidate-poolings",
        nargs="+",
        choices=CANDIDATE_POOLINGS,
        default=CANDIDATE_POOLINGS,
        help=(
            "Representations to fit during a tuning run. Fused runs must include "
            "text_image_tokens; dual_or uses the two primitive channels "
            "(default: all three)."
        ),
    )
    parser.add_argument(
        "--image-only-malicious-weight",
        type=float,
        default=1.0,
        help="Training weight for an image-led malicious example (default: 1.0).",
    )
    parser.add_argument(
        "--text-led-malicious-weight",
        type=float,
        default=2.0,
        help="Training weight for a text-led malicious example (default: 2.0).",
    )
    parser.add_argument(
        "--counterfactual-paired-benign-weight",
        type=float,
        default=2.0,
        help=(
            "Training weight for the benign control paired one-to-one with a "
            "text-led malicious example (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--benign-weight",
        type=float,
        default=1.0,
        help="Training weight for a benign example (default: 1.0).",
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=2.5,
        help="Training weight for a flagged benign hard negative (default: 2.5).",
    )
    parser.add_argument(
        "--candidate-image-only-malicious-weights",
        nargs="+",
        type=float,
        help=(
            "Optional tuning grid for image-only malicious weights. When omitted, "
            "use only --image-only-malicious-weight."
        ),
    )
    parser.add_argument(
        "--candidate-text-led-malicious-weights",
        nargs="+",
        type=float,
        help=(
            "Optional dual-head tuning grid for text-led malicious weights. "
            "Ignored by the historical fused workflow."
        ),
    )
    parser.add_argument(
        "--dual-hard-negative-weight",
        type=float,
        default=3.5,
        help=(
            "Hard-negative benign weight used only by dual-head training "
            "(default: 3.5)."
        ),
    )
    parser.add_argument(
        "--dual-artifact-source",
        help=(
            "Optional exact source string for both dual artifacts. This is intended "
            "only for byte-for-byte reproduction of a previously qualified pair."
        ),
    )
    parser.add_argument(
        "--dual-qualification-evidence-dir",
        type=Path,
        help=(
            "Optional evidence directory from an already-completed one-shot dual "
            "evaluation. Its immutable files are copied and bound into the pair."
        ),
    )
    parser.add_argument(
        "--candidate-hard-negative-weights",
        nargs="+",
        type=float,
        help=(
            "Optional tuning grid for hard-negative benign weights. When omitted, "
            "use only --hard-negative-weight."
        ),
    )
    parser.add_argument(
        "--candidate-learning-rates",
        nargs="+",
        type=float,
        help="Optional learning-rate tuning grid (default: the pinned grid).",
    )
    parser.add_argument(
        "--candidate-l2-values",
        nargs="+",
        type=float,
        help="Optional L2 tuning grid (default: the pinned grid).",
    )
    return parser.parse_args()


def resolved_target_info(args: argparse.Namespace) -> dict[str, object]:
    info = dict(TARGETS[args.target])
    if args.model_id:
        info["model_id"] = str(args.model_id)
    if args.model_revision:
        info["model_revision"] = str(args.model_revision)
    if args.tokenizer_revision:
        info["tokenizer_revision"] = str(args.tokenizer_revision)

    if args.target == "llava05b":
        info["runtime_model_id"] = str(
            args.runtime_model_id
            or (info["model_id"] if args.model_id else LLAVA_CACHE.resolve())
        )
        info["cache_dir"] = str(
            Path(args.cache_dir).resolve()
            if args.cache_dir
            else LLAVA_CACHE.parent.resolve()
        )
    else:
        if args.runtime_model_id:
            raise ValueError(
                "--runtime-model-id is only supported for LLaVA; use --model-id "
                "for a Qwen local model path."
            )
        info["cache_dir"] = str(
            Path(args.cache_dir).resolve()
            if args.cache_dir
            else QWEN_CACHE.resolve()
        )
    if args.qwen_chunk_size <= 0:
        raise ValueError("--qwen-chunk-size must be positive")
    validate_candidate_poolings(
        args.candidate_poolings,
        require_fused=(
            not args.extract_only and getattr(args, "artifact_mode", "fused") == "fused"
        ),
    )
    for name in (
        "image_only_malicious_weight",
        "text_led_malicious_weight",
        "counterfactual_paired_benign_weight",
        "benign_weight",
        "hard_negative_weight",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if hasattr(args, "dual_hard_negative_weight"):
        validate_positive_candidate_grid(
            (args.dual_hard_negative_weight,), name="dual hard-negative weight"
        )
    resolved_candidate_grids(args)
    if getattr(args, "candidate_text_led_malicious_weights", None) is not None:
        validate_positive_candidate_grid(
            args.candidate_text_led_malicious_weights,
            name="candidate text-led malicious weights",
        )
    return info


def validate_candidate_poolings(
    poolings: Iterable[str],
    *,
    require_fused: bool,
) -> tuple[str, ...]:
    if isinstance(poolings, str):
        raise ValueError("candidate poolings must be provided as a sequence")
    values = tuple(poolings)
    if not values:
        raise ValueError("at least one candidate pooling is required")
    unsupported = sorted(set(values) - set(CANDIDATE_POOLINGS))
    if unsupported:
        raise ValueError(f"unsupported candidate poolings: {unsupported}")
    if len(set(values)) != len(values):
        raise ValueError("candidate poolings must not contain duplicates")
    if require_fused and "text_image_tokens" not in values:
        raise ValueError(
            "promotion-producing runs must include text_image_tokens"
        )
    return values


def validate_positive_candidate_grid(
    values: Iterable[float],
    *,
    name: str,
) -> tuple[float, ...]:
    """Return a deterministic, finite, positive, duplicate-free tuning grid."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be provided as a sequence")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be provided as a sequence") from exc
    if not raw_values:
        raise ValueError(f"{name} must contain at least one value")
    normalized: list[float] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError(f"{name} values must be numeric")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{name} values must be finite and positive")
        normalized.append(numeric)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(normalized)


def resolved_candidate_grids(
    args: argparse.Namespace,
) -> dict[str, tuple[float, ...]]:
    """Resolve optional CLI grids while preserving the historical defaults."""
    return {
        "image_only_malicious_weights": validate_positive_candidate_grid(
            args.candidate_image_only_malicious_weights
            if args.candidate_image_only_malicious_weights is not None
            else (args.image_only_malicious_weight,),
            name="candidate image-only malicious weights",
        ),
        "hard_negative_weights": validate_positive_candidate_grid(
            args.candidate_hard_negative_weights
            if args.candidate_hard_negative_weights is not None
            else (args.hard_negative_weight,),
            name="candidate hard-negative weights",
        ),
        "learning_rates": validate_positive_candidate_grid(
            args.candidate_learning_rates
            if args.candidate_learning_rates is not None
            else CANDIDATE_LEARNING_RATES,
            name="candidate learning rates",
        ),
        "l2_values": validate_positive_candidate_grid(
            args.candidate_l2_values
            if args.candidate_l2_values is not None
            else CANDIDATE_L2_VALUES,
            name="candidate L2 values",
        ),
    }


def validation_candidate_rank(row: dict[str, object]) -> tuple[object, ...]:
    """Rank a candidate using validation evidence only."""
    return (
        float(row["validation_all_malicious_recall"]) == 1.0,
        float(row["validation_image_only_recall"]) == 1.0,
        float(row["validation_text_led_recall"]) == 1.0,
        bool(row["validation_fpr_target_met"]),
        -float(row["validation_false_positive_rate"]),
        float(row["validation_f1"]),
        float(row["validation_auprc"]),
        float(row["validation_auroc"]),
        float(row["validation_separation"]),
        -float(row["l2"]),
    )


def select_validation_candidate(
    candidates: Iterable[dict[str, object]],
) -> dict[str, object]:
    """Select a fused candidate without consulting qualification or frozen panels."""
    fused_candidates = [
        row for row in candidates if row.get("pooling") == "text_image_tokens"
    ]
    if not fused_candidates:
        raise RuntimeError("no fused text-image detector candidate was fitted")
    return max(fused_candidates, key=validation_candidate_rank)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_metadata(
    path: Path,
    *,
    require_hard_negative: bool = False,
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"metadata is empty: {path}")
    required = {
        "sample_id",
        "label_id",
        "split",
        "group_id",
        "attack_style",
        "prompt_text",
        "image_path",
        "image_text",
        "image_sha256",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"metadata lacks columns {sorted(missing)}: {path}")
    if require_hard_negative:
        hard_negative_columns = {"hard_negative", "hard_negative_category"}
        missing_hard_negative = hard_negative_columns - set(rows[0])
        if missing_hard_negative:
            raise ValueError(
                f"metadata lacks columns {sorted(missing_hard_negative)}: {path}"
            )
        validate_hard_negative_metadata(rows)
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate sample IDs in {path}")
    return rows


def validate_hard_negative_metadata(rows: list[dict[str, str]]) -> None:
    for row in rows:
        sample_id = row.get("sample_id", "<unknown>")
        flag = row.get("hard_negative")
        category = row.get("hard_negative_category")
        if flag not in {"0", "1"}:
            raise ValueError(
                f"hard_negative must be '0' or '1': {sample_id}"
            )
        if category is None:
            raise ValueError(f"hard_negative_category is missing: {sample_id}")
        try:
            label = int(row["label_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid label_id: {sample_id}") from exc
        if label not in {0, 1}:
            raise ValueError(f"label_id must be 0 or 1: {sample_id}")
        if label == 1:
            if flag != "0" or category != "":
                raise ValueError(
                    f"malicious row cannot carry hard-negative markers: {sample_id}"
                )
        elif flag == "0" and category != "":
            raise ValueError(
                f"ordinary benign row must have an empty hard-negative category: "
                f"{sample_id}"
            )
        elif flag == "1" and category not in HARD_NEGATIVE_CATEGORIES:
            raise ValueError(
                f"unsupported hard-negative category {category!r}: {sample_id}"
            )


def write_adapter_metadata(path: Path, rows: Iterable[dict[str, str]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "text", "image_path", "image_sha256"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in values:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "text": row["prompt_text"],
                    "image_path": row["image_path"],
                    "image_sha256": row["image_sha256"],
                }
            )


def scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    return str(np.asarray(data[key]).reshape(-1)[0])


def extraction_config(
    target: str,
    target_info: dict[str, object],
) -> LlavaOnevisionExtractionConfig | Qwen25VLExtractionConfig:
    if target == "llava05b":
        return LlavaOnevisionExtractionConfig(
            model_id=str(target_info["model_id"]),
            runtime_model_id=str(
                target_info.get("runtime_model_id", target_info["model_id"])
            ),
            model_revision=str(target_info["model_revision"]),
            tokenizer_revision=str(target_info["tokenizer_revision"]),
            cache_dir=(
                str(target_info["cache_dir"])
                if target_info.get("cache_dir") is not None
                else None
            ),
            layer=-1,
            torch_dtype="auto",
            device_map="auto",
            max_image_edge=384,
            batch_size=4,
            max_batch_characters=1_024,
            local_files_only=True,
        )
    if target == "qwen25vl3b":
        return Qwen25VLExtractionConfig(
            model_id=str(target_info["model_id"]),
            model_revision=str(target_info["model_revision"]),
            tokenizer_revision=str(target_info["tokenizer_revision"]),
            cache_dir=(
                str(target_info["cache_dir"])
                if target_info.get("cache_dir") is not None
                else None
            ),
            layer=-1,
            torch_dtype="auto",
            device_map="auto",
            min_pixels=None,
            max_pixels=200_704,
            local_files_only=True,
        )
    raise ValueError(f"unsupported extraction target: {target}")


def expected_preprocessing_sha256(
    target: str,
    target_info: dict[str, object],
) -> str:
    config = extraction_config(target, target_info)
    if target == "llava05b":
        if not isinstance(config, LlavaOnevisionExtractionConfig):
            raise TypeError("LLaVA target resolved to the wrong extraction config")
        fingerprint = llava_preprocessing_sha256(config)
    elif target == "qwen25vl3b":
        if not isinstance(config, Qwen25VLExtractionConfig):
            raise TypeError("Qwen target resolved to the wrong extraction config")
        fingerprint = qwen_preprocessing_sha256(config)
    else:
        raise ValueError(f"unsupported extraction target: {target}")
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise RuntimeError("adapter returned an invalid preprocessing fingerprint")
    return fingerprint


def validate_embedding_file(
    path: Path,
    rows: list[dict[str, str]],
    target: str,
    pooling: str,
    target_info: dict[str, object] | None = None,
) -> None:
    expected = target_info or TARGETS[target]
    with np.load(path, allow_pickle=False) as data:
        required = {
            "embeddings",
            "sample_id",
            "model_family",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "preprocessing_sha256",
            "layer",
            "pooling",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"embedding file lacks {sorted(missing)}: {path}")
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        sample_ids = [str(value) for value in np.asarray(data["sample_id"]).reshape(-1)]
        if sample_ids != [row["sample_id"] for row in rows]:
            raise ValueError(f"sample alignment mismatch: {path}")
        if embeddings.shape != (len(rows), int(expected["feature_dim"])):
            raise ValueError(f"unexpected embedding shape {embeddings.shape}: {path}")
        if not np.all(np.isfinite(embeddings)):
            raise ValueError(f"non-finite embeddings: {path}")
        checks = {
            "model_family": expected["model_family"],
            "model_id": expected["model_id"],
            "model_revision": expected["model_revision"],
            "tokenizer_revision": expected["tokenizer_revision"],
            "pooling": pooling,
        }
        for key, value in checks.items():
            if scalar(data, key).replace("/", "\\") != str(value).replace("/", "\\"):
                raise ValueError(
                    f"embedding provenance mismatch for {key}: "
                    f"{scalar(data, key)!r} != {value!r}"
                )
        if int(np.asarray(data["layer"]).reshape(-1)[0]) != -1:
            raise ValueError(f"unexpected extraction layer: {path}")
        fingerprint = scalar(data, "preprocessing_sha256")
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError(f"invalid preprocessing fingerprint: {path}")
        expected_fingerprint = expected_preprocessing_sha256(target, expected)
        if fingerprint != expected_fingerprint:
            raise ValueError(
                "embedding provenance mismatch for preprocessing_sha256: "
                f"{fingerprint!r} != {expected_fingerprint!r}: {path}"
            )


def feature_cache_is_valid(
    *,
    target: str,
    panel: str,
    feature_root: Path,
    metadata_path: Path,
    adapter_metadata: Path,
    expected_paths: dict[str, Path],
    rows: list[dict[str, str]],
    target_info: dict[str, object],
) -> bool:
    manifest_path = feature_root / "feature_manifest.json"
    if not manifest_path.is_file() or not all(
        path.is_file() for path in expected_paths.values()
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks = {
            "schema_version": 1,
            "target": target,
            "panel": panel,
            "rows": len(rows),
            "source_metadata": str(metadata_path.relative_to(RUN)).replace("\\", "/"),
            "source_metadata_sha256": sha256(metadata_path),
            "adapter_metadata_sha256": sha256(adapter_metadata),
        }
        for key, expected in checks.items():
            if manifest.get(key) != expected:
                raise ValueError(f"feature manifest mismatch for {key}")
        entries = manifest.get("embeddings")
        if not isinstance(entries, dict) or set(entries) != set(expected_paths):
            raise ValueError("feature manifest pooling set changed")
        for pooling, path in expected_paths.items():
            entry = entries[pooling]
            expected_relative = str(path.relative_to(RUN)).replace("\\", "/")
            if entry.get("path") != expected_relative:
                raise ValueError(f"feature manifest path changed for {pooling}")
            if entry.get("sha256") != sha256(path):
                raise ValueError(f"feature hash changed for {pooling}")
            validate_embedding_file(path, rows, target, pooling, target_info)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"discarding invalid {target} {panel} feature cache: {exc}", flush=True)
        return False
    return True


def combine_embedding_chunks(
    *,
    target: str,
    pooling: str,
    rows: list[dict[str, str]],
    chunks: list[tuple[list[dict[str, str]], Path]],
    output_path: Path,
    target_info: dict[str, object],
) -> None:
    embeddings: list[np.ndarray] = []
    sample_ids: list[str] = []
    provenance: dict[str, object] | None = None
    for chunk_rows, chunk_path in chunks:
        validate_embedding_file(
            chunk_path, chunk_rows, target, pooling, target_info
        )
        with np.load(chunk_path, allow_pickle=False) as data:
            embeddings.append(np.asarray(data["embeddings"], dtype=np.float32))
            sample_ids.extend(
                str(value) for value in np.asarray(data["sample_id"]).reshape(-1)
            )
            current: dict[str, object] = {
                "model_family": scalar(data, "model_family"),
                "model_id": scalar(data, "model_id"),
                "model_revision": scalar(data, "model_revision"),
                "tokenizer_revision": scalar(data, "tokenizer_revision"),
                "preprocessing_sha256": scalar(data, "preprocessing_sha256"),
                "layer": int(np.asarray(data["layer"]).reshape(-1)[0]),
                "pooling": scalar(data, "pooling"),
            }
        if provenance is None:
            provenance = current
        elif current != provenance:
            raise ValueError(f"chunk provenance mismatch: {chunk_path}")

    if provenance is None:
        raise ValueError("cannot combine an empty chunk list")
    if sample_ids != [row["sample_id"] for row in rows]:
        raise ValueError("combined Qwen chunks are not aligned with source metadata")
    merged = np.vstack(embeddings).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.npz")
    np.savez(
        temporary,
        embeddings=merged,
        sample_id=np.asarray(sample_ids),
        model_id=np.asarray([str(provenance["model_id"])]),
        model_family=np.asarray([str(provenance["model_family"])]),
        model_revision=np.asarray([str(provenance["model_revision"])]),
        tokenizer_revision=np.asarray([str(provenance["tokenizer_revision"])]),
        preprocessing_sha256=np.asarray([str(provenance["preprocessing_sha256"])]),
        layer=np.asarray([int(provenance["layer"])]),
        pooling=np.asarray([str(provenance["pooling"])]),
    )
    temporary.replace(output_path)
    validate_embedding_file(output_path, rows, target, pooling, target_info)


def extract_qwen_panel_resumably(
    *,
    target: str,
    panel: str,
    feature_root: Path,
    raw: Path,
    rows: list[dict[str, str]],
    config: Qwen25VLExtractionConfig,
    target_info: dict[str, object],
    chunk_size: int = 32,
) -> dict[str, Path]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks_by_pooling: dict[str, list[tuple[list[dict[str, str]], Path]]] = {
        pooling: [] for pooling in EXTRACTION_POOLINGS
    }
    chunk_count = (len(rows) + chunk_size - 1) // chunk_size
    metadata_root = feature_root / "chunk_metadata"
    chunk_root = feature_root / "chunks"
    for chunk_index, start in enumerate(range(0, len(rows), chunk_size), start=1):
        chunk_rows = rows[start : start + chunk_size]
        chunk_metadata = metadata_root / f"chunk_{chunk_index:04d}.csv"
        write_adapter_metadata(chunk_metadata, chunk_rows)
        metadata_hash = sha256(chunk_metadata)
        output_dir = chunk_root / f"{chunk_index:04d}-{metadata_hash[:16]}"
        expected = {
            pooling: output_dir
            / f"chunk_{chunk_index:04d}_layer_m1_{pooling}.npz"
            for pooling in EXTRACTION_POOLINGS
        }
        reuse_existing = all(path.is_file() for path in expected.values())
        if reuse_existing:
            try:
                for pooling, path in expected.items():
                    validate_embedding_file(
                        path, chunk_rows, target, pooling, target_info
                    )
            except (
                EOFError,
                KeyError,
                OSError,
                ValueError,
                zipfile.BadZipFile,
            ) as exc:
                print(
                    f"discarding invalid {target} {panel} chunk "
                    f"{chunk_index}/{chunk_count}: {exc}",
                    flush=True,
                )
                reuse_existing = False
        if reuse_existing:
            normalized = expected
            state = "reused"
        else:
            paths = extract_qwen25_vl_pooling_embeddings(
                metadata_path=chunk_metadata,
                corpus_root=RUN,
                output_dir=output_dir,
                poolings=EXTRACTION_POOLINGS,
                config=config,
                output_prefix=f"chunk_{chunk_index:04d}",
            )
            normalized = {str(pooling): Path(path) for pooling, path in paths.items()}
            for pooling, path in normalized.items():
                validate_embedding_file(
                    path, chunk_rows, target, pooling, target_info
                )
            state = "completed"
        for pooling, path in normalized.items():
            chunks_by_pooling[pooling].append((chunk_rows, path))
        print(
            f"{state} {target} {panel} chunk {chunk_index}/{chunk_count} "
            f"({len(chunk_rows)} rows)",
            flush=True,
        )

    combined = {
        pooling: raw / f"{panel}_layer_m1_{pooling}.npz"
        for pooling in EXTRACTION_POOLINGS
    }
    for pooling, output_path in combined.items():
        combine_embedding_chunks(
            target=target,
            pooling=pooling,
            rows=rows,
            chunks=chunks_by_pooling[pooling],
            output_path=output_path,
            target_info=target_info,
        )
    return combined


def _provenance_without_pooling(provenance: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in provenance.items() if key != "pooling"}


def _save_embedding_matrix(
    path: Path,
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    provenance: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez(
        temporary,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        sample_id=np.asarray([row["sample_id"] for row in rows]),
        model_id=np.asarray([str(provenance["model_id"])]),
        model_family=np.asarray([str(provenance["model_family"])]),
        model_revision=np.asarray([str(provenance["model_revision"])]),
        tokenizer_revision=np.asarray([str(provenance["tokenizer_revision"])]),
        preprocessing_sha256=np.asarray(
            [str(provenance["preprocessing_sha256"])]
        ),
        layer=np.asarray([int(provenance["layer"])]),
        pooling=np.asarray([str(provenance["pooling"])]),
    )
    temporary.replace(path)


def _validated_v6_reuse(
    *,
    target: str,
    panel: str,
    rows: list[dict[str, str]],
    target_info: dict[str, object],
) -> tuple[dict[int, int], dict[str, Path], list[dict[str, str]]]:
    """Return v7-index -> v6-index mappings after exact content validation."""
    metadata_path = V6_METADATA.get(panel)
    if metadata_path is None or not metadata_path.is_file():
        return {}, {}, []
    feature_root = RUN / V6_FEATURE_DIRECTORY / target / panel
    feature_manifest_path = feature_root / "feature_manifest.json"
    adapter_metadata_path = feature_root / "adapter_metadata.csv"
    old_paths = {
        pooling: RUN
        / V6_FEATURE_DIRECTORY
        / target
        / panel
        / "raw"
        / f"{panel}_layer_m1_{pooling}.npz"
        for pooling in EXTRACTION_POOLINGS
    }
    required_paths = (
        feature_manifest_path,
        adapter_metadata_path,
        *old_paths.values(),
    )
    if not all(path.is_file() for path in required_paths):
        return {}, {}, []
    try:
        expected_manifest_hash = V6_FEATURE_MANIFEST_SHA256[target][panel]
        if sha256(feature_manifest_path) != expected_manifest_hash:
            raise ValueError("pinned v6 feature manifest changed")
        manifest = json.loads(
            feature_manifest_path.read_text(encoding="utf-8")
        )
        old_rows = read_metadata(metadata_path)
        manifest_checks = {
            "schema_version": 1,
            "target": target,
            "panel": panel,
            "rows": len(old_rows),
            "source_metadata": str(metadata_path.relative_to(RUN)).replace(
                "\\", "/"
            ),
            "source_metadata_sha256": sha256(metadata_path),
            "adapter_metadata_sha256": sha256(adapter_metadata_path),
        }
        for key, expected in manifest_checks.items():
            if manifest.get(key) != expected:
                raise ValueError(f"v6 feature manifest mismatch for {key}")

        with adapter_metadata_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            adapter_rows = list(csv.DictReader(handle))
        expected_adapter_rows = [
            {
                "sample_id": row["sample_id"],
                "text": row["prompt_text"],
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
            }
            for row in old_rows
        ]
        if adapter_rows != expected_adapter_rows:
            raise ValueError("v6 adapter metadata differs from source metadata")

        entries = manifest.get("embeddings")
        if not isinstance(entries, dict) or set(entries) != set(old_paths):
            raise ValueError("v6 primitive pooling set changed")
        for pooling, path in old_paths.items():
            entry = entries[pooling]
            if not isinstance(entry, dict):
                raise ValueError(f"v6 manifest entry is invalid for {pooling}")
            expected_relative = str(path.relative_to(RUN)).replace("\\", "/")
            if entry.get("path") != expected_relative:
                raise ValueError(f"v6 feature path changed for {pooling}")
            if entry.get("sha256") != sha256(path):
                raise ValueError(f"v6 feature hash changed for {pooling}")
            validate_embedding_file(
                path, old_rows, target, pooling, target_info
            )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        print(
            f"v6 feature reuse unavailable for {target} {panel}: {exc}",
            flush=True,
        )
        return {}, {}, []

    old_by_id = {row["sample_id"]: index for index, row in enumerate(old_rows)}
    verified_hashes: dict[Path, str] = {}

    def actual_hash(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in verified_hashes:
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            verified_hashes[resolved] = sha256(resolved)
        return verified_hashes[resolved]

    reused: dict[int, int] = {}
    for new_index, row in enumerate(rows):
        old_index = old_by_id.get(row["sample_id"])
        if old_index is None:
            continue
        old = old_rows[old_index]
        if row["prompt_text"] != old["prompt_text"]:
            continue
        if row["image_sha256"] != old["image_sha256"]:
            continue
        try:
            if actual_hash(RUN / row["image_path"]) != row["image_sha256"]:
                continue
            if actual_hash(RUN / old["image_path"]) != old["image_sha256"]:
                continue
        except OSError:
            continue
        reused[new_index] = old_index
    return reused, old_paths, old_rows


def _validated_schema3_reuse(
    *,
    target: str,
    panel: str,
    rows: list[dict[str, str]],
    target_info: dict[str, object],
) -> tuple[
    dict[int, int],
    dict[str, Path],
    list[dict[str, str]],
    dict[str, object],
]:
    """Return exact row matches from the immutable schema-3 feature archive."""
    empty: tuple[
        dict[int, int],
        dict[str, Path],
        list[dict[str, str]],
        dict[str, object],
    ] = ({}, {}, [], {})
    if panel not in {"development", "regression"}:
        return empty

    archive_manifest_path = SCHEMA3_FEATURE_ROOT / "archive_manifest.json"
    source_metadata = SCHEMA3_FEATURE_ROOT / f"{panel}_metadata_v7.csv"
    source_corpus_manifest = SCHEMA3_FEATURE_ROOT / "corpus_manifest_v7.json"
    feature_root = SCHEMA3_FEATURE_ROOT / target / panel
    feature_manifest_path = feature_root / "feature_manifest.json"
    adapter_metadata_path = feature_root / "adapter_metadata.csv"
    source_paths = {
        pooling: feature_root / "raw" / f"{panel}_layer_m1_{pooling}.npz"
        for pooling in EXTRACTION_POOLINGS
    }
    required_paths = (
        archive_manifest_path,
        source_metadata,
        source_corpus_manifest,
        feature_manifest_path,
        adapter_metadata_path,
        *source_paths.values(),
    )
    if not all(path.is_file() for path in required_paths):
        return empty

    try:
        if sha256(archive_manifest_path) != SCHEMA3_ARCHIVE_MANIFEST_SHA256:
            raise ValueError("schema-3 archive manifest changed")
        archive_manifest = json.loads(
            archive_manifest_path.read_text(encoding="utf-8")
        )
        archived_files = archive_manifest.get("files")
        if not isinstance(archived_files, dict):
            raise ValueError("schema-3 archive lacks file hashes")
        for name, path in (
            (source_metadata.name, source_metadata),
            (source_corpus_manifest.name, source_corpus_manifest),
        ):
            expected = archived_files.get(name)
            if not isinstance(expected, dict):
                raise ValueError(f"schema-3 archive lacks {name} lineage")
            if expected.get("sha256") != sha256(path):
                raise ValueError(f"schema-3 archive hash changed: {name}")

        source_rows = read_metadata(
            source_metadata,
            require_hard_negative=True,
        )
        feature_manifest = json.loads(
            feature_manifest_path.read_text(encoding="utf-8")
        )
        corpus_manifest = json.loads(
            source_corpus_manifest.read_text(encoding="utf-8")
        )
        source_metadata_hash = sha256(source_metadata)
        if corpus_manifest.get("schema_version") != 3:
            raise ValueError("archived corpus is not schema 3")
        if feature_manifest.get("schema_version") != 1:
            raise ValueError("unexpected archived feature-manifest schema")
        feature_checks = {
            "target": target,
            "panel": panel,
            "rows": len(source_rows),
            "source_metadata": source_metadata.name,
            "source_metadata_sha256": source_metadata_hash,
            "adapter_metadata_sha256": sha256(adapter_metadata_path),
        }
        for key, expected in feature_checks.items():
            if feature_manifest.get(key) != expected:
                raise ValueError(f"archived feature manifest mismatch for {key}")
        corpus_panel_key = "development" if panel == "development" else "regression"
        panel_manifest = corpus_manifest.get(corpus_panel_key)
        if not isinstance(panel_manifest, dict):
            raise ValueError(f"archived corpus lacks {panel} metadata")
        if panel_manifest.get("metadata") != source_metadata.name:
            raise ValueError(f"archived corpus {panel} path changed")
        if panel_manifest.get("metadata_sha256") != source_metadata_hash:
            raise ValueError(f"archived corpus {panel} hash changed")
        if panel_manifest.get("rows") != len(source_rows):
            raise ValueError(f"archived corpus {panel} count changed")

        with adapter_metadata_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            adapter_rows = list(csv.DictReader(handle))
        expected_adapter_rows = [
            {
                "sample_id": row["sample_id"],
                "text": row["prompt_text"],
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
            }
            for row in source_rows
        ]
        if adapter_rows != expected_adapter_rows:
            raise ValueError("schema-3 adapter metadata differs from source metadata")

        entries = feature_manifest.get("embeddings")
        if not isinstance(entries, dict) or set(entries) != set(source_paths):
            raise ValueError("schema-3 primitive pooling set changed")
        for pooling, path in source_paths.items():
            if entries[pooling].get("sha256") != sha256(path):
                raise ValueError(f"schema-3 {pooling} feature hash changed")
            validate_embedding_file(path, source_rows, target, pooling, target_info)
        reuse_lineage = feature_manifest.get("feature_reuse")
        if not isinstance(reuse_lineage, dict):
            raise ValueError("schema-3 feature manifest lacks reuse lineage")
        if (
            int(reuse_lineage.get("rows_reused_total", -1))
            + int(reuse_lineage.get("rows_extracted", -1))
            != len(source_rows)
        ):
            raise ValueError("schema-3 feature lineage row count is inconsistent")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        print(
            f"schema-3 feature reuse unavailable for {target} {panel}: {exc}",
            flush=True,
        )
        return empty

    source_by_id = {
        row["sample_id"]: index for index, row in enumerate(source_rows)
    }
    verified_hashes: dict[Path, str] = {}

    def actual_hash(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in verified_hashes:
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            verified_hashes[resolved] = sha256(resolved)
        return verified_hashes[resolved]

    reused: dict[int, int] = {}
    for new_index, row in enumerate(rows):
        source_index = source_by_id.get(row["sample_id"])
        if source_index is None:
            continue
        source = source_rows[source_index]
        if row != source:
            continue
        try:
            if actual_hash(RUN / row["image_path"]) != row["image_sha256"]:
                continue
            archived_image = SCHEMA3_FEATURE_ROOT / source["image_path"]
            if actual_hash(archived_image) != source["image_sha256"]:
                continue
        except OSError:
            continue
        reused[new_index] = source_index

    details = {
        "source_archive_manifest": str(
            archive_manifest_path.relative_to(RUN)
        ).replace("\\", "/"),
        "source_archive_manifest_sha256": sha256(archive_manifest_path),
        "source_metadata": str(source_metadata.relative_to(RUN)).replace("\\", "/"),
        "source_metadata_sha256": sha256(source_metadata),
        "source_corpus_manifest": str(
            source_corpus_manifest.relative_to(RUN)
        ).replace("\\", "/"),
        "source_corpus_manifest_sha256": sha256(source_corpus_manifest),
        "source_feature_manifest": str(
            feature_manifest_path.relative_to(RUN)
        ).replace("\\", "/"),
        "source_feature_manifest_sha256": sha256(feature_manifest_path),
        "source_adapter_metadata_sha256": sha256(adapter_metadata_path),
        "source_feature_sha256": {
            pooling: sha256(path) for pooling, path in source_paths.items()
        },
        "source_rows": len(source_rows),
        "eligible_source_rows": len(source_rows),
        "validation": (
            "immutable archive manifest, schema-3 corpus and feature manifests, "
            "adapter metadata, primitive feature provenance, exact full row, declared "
            "image SHA-256, and archived/current image bytes must all match"
        ),
    }
    return reused, source_paths, source_rows, details


def _validated_prior_v7_reuse(
    *,
    target: str,
    panel: str,
    rows: list[dict[str, str]],
    target_info: dict[str, object],
) -> tuple[
    dict[int, int],
    dict[str, Path],
    list[dict[str, str]],
    dict[str, object],
]:
    """Validate a development-only archived v7 source and return exact train matches."""
    empty: tuple[
        dict[int, int],
        dict[str, Path],
        list[dict[str, str]],
        dict[str, object],
    ] = ({}, {}, [], {})
    if panel != "development":
        return empty

    source_metadata = PRIOR_V7_FEATURE_ROOT / "development_metadata_v7.csv"
    source_corpus_manifest = PRIOR_V7_FEATURE_ROOT / "corpus_manifest_v7.json"
    feature_root = PRIOR_V7_FEATURE_ROOT / target / "development"
    feature_manifest_path = feature_root / "feature_manifest.json"
    adapter_metadata_path = feature_root / "adapter_metadata.csv"
    source_paths = {
        pooling: feature_root
        / "raw"
        / f"development_layer_m1_{pooling}.npz"
        for pooling in EXTRACTION_POOLINGS
    }
    required_paths = (
        source_metadata,
        source_corpus_manifest,
        feature_manifest_path,
        adapter_metadata_path,
        *source_paths.values(),
    )
    if not all(path.is_file() for path in required_paths):
        return empty

    try:
        source_rows = read_metadata(source_metadata)
        feature_manifest = json.loads(
            feature_manifest_path.read_text(encoding="utf-8")
        )
        corpus_manifest = json.loads(
            source_corpus_manifest.read_text(encoding="utf-8")
        )
        source_metadata_hash = sha256(source_metadata)
        if feature_manifest.get("schema_version") != 1:
            raise ValueError("unexpected archived feature-manifest schema")
        feature_checks = {
            "target": target,
            "panel": "development",
            "rows": len(source_rows),
            "source_metadata": "development_metadata_v7.csv",
            "source_metadata_sha256": source_metadata_hash,
            "adapter_metadata_sha256": sha256(adapter_metadata_path),
        }
        for key, expected in feature_checks.items():
            if feature_manifest.get(key) != expected:
                raise ValueError(f"archived feature manifest mismatch for {key}")
        development_manifest = corpus_manifest.get("development")
        if not isinstance(development_manifest, dict):
            raise ValueError("archived corpus manifest lacks development metadata")
        if development_manifest.get("metadata") != "development_metadata_v7.csv":
            raise ValueError("archived corpus metadata path changed")
        if development_manifest.get("metadata_sha256") != source_metadata_hash:
            raise ValueError("archived corpus metadata hash changed")
        if development_manifest.get("rows") != len(source_rows):
            raise ValueError("archived corpus row count changed")

        with adapter_metadata_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            adapter_rows = list(csv.DictReader(handle))
        expected_adapter_rows = [
            {
                "sample_id": row["sample_id"],
                "text": row["prompt_text"],
                "image_path": row["image_path"],
                "image_sha256": row["image_sha256"],
            }
            for row in source_rows
        ]
        if adapter_rows != expected_adapter_rows:
            raise ValueError("archived adapter metadata differs from source metadata")

        entries = feature_manifest.get("embeddings")
        if not isinstance(entries, dict) or set(entries) != set(source_paths):
            raise ValueError("archived primitive pooling set changed")
        for pooling, path in source_paths.items():
            if entries[pooling].get("sha256") != sha256(path):
                raise ValueError(f"archived {pooling} feature hash changed")
            validate_embedding_file(
                path, source_rows, target, pooling, target_info
            )

        v6_lineage = feature_manifest.get("v6_feature_reuse")
        text_led_rows = [
            row for row in source_rows if row["strategy"] == TEXT_LED_STRATEGY
        ]
        if not isinstance(v6_lineage, dict):
            raise ValueError("archived feature manifest lacks v6 lineage")
        if v6_lineage.get("rows_extracted") != len(text_led_rows):
            raise ValueError("archived extracted-row lineage is inconsistent")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(
            f"prior-v7 feature reuse unavailable for {target} {panel}: {exc}",
            flush=True,
        )
        return empty

    source_by_id = {
        row["sample_id"]: index for index, row in enumerate(source_rows)
    }
    verified_hashes: dict[Path, str] = {}

    def actual_hash(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in verified_hashes:
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            verified_hashes[resolved] = sha256(resolved)
        return verified_hashes[resolved]

    reused: dict[int, int] = {}
    ignored_path_field = {"image_path"}
    for new_index, row in enumerate(rows):
        source_index = source_by_id.get(row["sample_id"])
        if source_index is None:
            continue
        source = source_rows[source_index]
        # Only genuinely extracted old training rows are eligible. This excludes
        # archived validation/test vectors and preserves transitive v6 provenance.
        if source["split"] != "train" or row["split"] != "train":
            continue
        if source["strategy"] != TEXT_LED_STRATEGY:
            continue
        if any(
            row.get(key) != value
            for key, value in source.items()
            if key not in ignored_path_field
        ):
            continue
        if row["prompt_text"] != source["prompt_text"]:
            continue
        if row["image_sha256"] != source["image_sha256"]:
            continue
        try:
            if actual_hash(RUN / row["image_path"]) != row["image_sha256"]:
                continue
            if actual_hash(RUN / source["image_path"]) != source["image_sha256"]:
                continue
        except OSError:
            continue
        reused[new_index] = source_index

    details = {
        "source_metadata": str(source_metadata.relative_to(RUN)).replace("\\", "/"),
        "source_metadata_sha256": sha256(source_metadata),
        "source_corpus_manifest": str(
            source_corpus_manifest.relative_to(RUN)
        ).replace("\\", "/"),
        "source_corpus_manifest_sha256": sha256(source_corpus_manifest),
        "source_feature_manifest": str(
            feature_manifest_path.relative_to(RUN)
        ).replace("\\", "/"),
        "source_feature_manifest_sha256": sha256(feature_manifest_path),
        "source_adapter_metadata_sha256": sha256(adapter_metadata_path),
        "source_feature_sha256": {
            pooling: sha256(path) for pooling, path in source_paths.items()
        },
        "source_rows": len(source_rows),
        "eligible_source_rows": sum(
            row["split"] == "train" and row["strategy"] == TEXT_LED_STRATEGY
            for row in source_rows
        ),
        "validation": (
            "archived corpus, full source metadata, adapter metadata, primitive "
            "feature provenance, sample ID, all non-path row fields, declared image "
            "SHA-256, and source/destination image bytes must match; source and "
            "destination splits must both be train"
        ),
    }
    return reused, source_paths, source_rows, details


def _extract_rows(
    *,
    target: str,
    panel: str,
    feature_root: Path,
    rows: list[dict[str, str]],
    target_info: dict[str, object],
    qwen_chunk_size: int,
) -> dict[str, Path]:
    adapter_metadata = feature_root / "adapter_metadata.csv"
    raw = feature_root / "raw"
    write_adapter_metadata(adapter_metadata, rows)
    expected = {
        pooling: raw / f"{panel}_layer_m1_{pooling}.npz"
        for pooling in EXTRACTION_POOLINGS
    }
    if all(path.is_file() for path in expected.values()):
        try:
            for pooling, path in expected.items():
                validate_embedding_file(
                    path, rows, target, pooling, target_info
                )
            print(
                f"reusing validated {target} {panel} incremental embeddings",
                flush=True,
            )
            return expected
        except (OSError, ValueError, KeyError):
            pass

    config = extraction_config(target, target_info)
    if target == "llava05b":
        if not isinstance(config, LlavaOnevisionExtractionConfig):
            raise TypeError("LLaVA target resolved to the wrong extraction config")
        paths = extract_llava_onevision_pooling_embeddings(
            metadata_path=adapter_metadata,
            corpus_root=RUN,
            output_dir=raw,
            poolings=EXTRACTION_POOLINGS,
            config=config,
            output_prefix=panel,
        )
    else:
        if not isinstance(config, Qwen25VLExtractionConfig):
            raise TypeError("Qwen target resolved to the wrong extraction config")
        paths = extract_qwen_panel_resumably(
            target=target,
            panel=panel,
            feature_root=feature_root,
            raw=raw,
            rows=rows,
            config=config,
            target_info=target_info,
            chunk_size=qwen_chunk_size,
        )
    normalized = {str(pooling): Path(path) for pooling, path in paths.items()}
    for pooling, path in normalized.items():
        validate_embedding_file(path, rows, target, pooling, target_info)
    return normalized


def extract_panel(
    target: str,
    panel: str,
    metadata_path: Path,
    rows: list[dict[str, str]],
    target_info: dict[str, object],
    qwen_chunk_size: int,
) -> dict[str, Path]:
    feature_root = RUN / FEATURE_DIRECTORY / target / panel
    raw = feature_root / "raw"
    adapter_metadata = feature_root / "adapter_metadata.csv"
    write_adapter_metadata(adapter_metadata, rows)
    expected_paths = {
        pooling: raw / f"{panel}_layer_m1_{pooling}.npz"
        for pooling in EXTRACTION_POOLINGS
    }
    if feature_cache_is_valid(
        target=target,
        panel=panel,
        feature_root=feature_root,
        metadata_path=metadata_path,
        adapter_metadata=adapter_metadata,
        expected_paths=expected_paths,
        rows=rows,
        target_info=target_info,
    ):
        print(f"reusing validated {target} {panel} embeddings", flush=True)
        return expected_paths

    schema3_mapping, schema3_paths, schema3_rows, schema3_details = (
        _validated_schema3_reuse(
            target=target,
            panel=panel,
            rows=rows,
            target_info=target_info,
        )
    )
    prior_mapping, prior_paths, prior_rows, prior_details = (
        _validated_prior_v7_reuse(
            target=target,
            panel=panel,
            rows=rows,
            target_info=target_info,
        )
    )
    v6_mapping, v6_paths, v6_rows = _validated_v6_reuse(
        target=target,
        panel=panel,
        rows=rows,
        target_info=target_info,
    )
    # Exact schema-3 rows have first refusal. Older sources fill only still-
    # unmatched rows and never override the immutable immediate predecessor.
    prior_mapping = {
        new_index: old_index
        for new_index, old_index in prior_mapping.items()
        if new_index not in schema3_mapping
    }
    v6_mapping = {
        new_index: old_index
        for new_index, old_index in v6_mapping.items()
        if new_index not in schema3_mapping and new_index not in prior_mapping
    }
    reused_indices = set(schema3_mapping) | set(prior_mapping) | set(v6_mapping)
    new_indices = [
        index for index in range(len(rows)) if index not in reused_indices
    ]
    new_rows = [rows[index] for index in new_indices]
    print(
        f"building {target} {panel} embeddings: "
        f"{len(schema3_mapping)} verified schema-3 rows, "
        f"{len(prior_mapping)} verified prior-v7 rows, "
        f"{len(v6_mapping)} verified v6 rows, "
        f"{len(new_rows)} rows to extract",
        flush=True,
    )
    new_paths: dict[str, Path] = {}
    if new_rows:
        new_paths = _extract_rows(
            target=target,
            panel=f"{panel}_new",
            feature_root=feature_root / "incremental",
            rows=new_rows,
            target_info=target_info,
            qwen_chunk_size=qwen_chunk_size,
        )

    for pooling, output_path in expected_paths.items():
        reused_sources: list[
            tuple[str, dict[int, int], np.ndarray, dict[str, object]]
        ] = []
        if schema3_mapping:
            schema3_features, schema3_provenance = load_embeddings(
                schema3_paths[pooling]
            )
            reused_sources.append(
                (
                    "schema3_pre_additional_hard_negatives",
                    schema3_mapping,
                    schema3_features,
                    schema3_provenance,
                )
            )
        if prior_mapping:
            prior_features, prior_provenance = load_embeddings(
                prior_paths[pooling]
            )
            reused_sources.append(
                (
                    "prior_v7_pre_hard_negative",
                    prior_mapping,
                    prior_features,
                    prior_provenance,
                )
            )
        if v6_mapping:
            v6_features, v6_provenance = load_embeddings(v6_paths[pooling])
            reused_sources.append(
                ("v6", v6_mapping, v6_features, v6_provenance)
            )
        new_features: np.ndarray | None = None
        new_provenance: dict[str, object] | None = None
        if new_rows:
            new_features, new_provenance = load_embeddings(new_paths[pooling])
        provenance = (
            reused_sources[0][3] if reused_sources else new_provenance
        )
        if provenance is None:
            raise RuntimeError(f"no embeddings available for {target} {panel}")
        for source_name, _, _, source_provenance in reused_sources:
            if source_provenance != provenance:
                raise RuntimeError(
                    f"{source_name} {pooling} provenance does not match"
                )
        if new_provenance is not None and new_provenance != provenance:
            raise RuntimeError(
                f"newly extracted {pooling} provenance does not match reused source"
            )
        feature_dim = int(target_info["feature_dim"])
        merged = np.empty((len(rows), feature_dim), dtype=np.float32)
        for _, mapping, source_features, _ in reused_sources:
            for new_index, source_index in mapping.items():
                merged[new_index] = source_features[source_index]
        if new_rows:
            assert new_features is not None
            for extracted_index, new_index in enumerate(new_indices):
                merged[new_index] = new_features[extracted_index]
        _save_embedding_matrix(output_path, merged, rows, provenance)
        validate_embedding_file(
            output_path, rows, target, pooling, target_info
        )

    schema3_manifest_entry = dict(schema3_details)
    schema3_manifest_entry["rows_reused"] = len(schema3_mapping)
    prior_manifest_entry = dict(prior_details)
    prior_manifest_entry["rows_reused"] = len(prior_mapping)
    v6_manifest_entry = {
        "rows_reused": len(v6_mapping),
        "source_metadata": (
            str(V6_METADATA[panel].relative_to(RUN)).replace("\\", "/")
            if v6_paths
            else None
        ),
        "source_metadata_sha256": (
            sha256(V6_METADATA[panel]) if v6_paths else None
        ),
        "source_feature_sha256": (
            {pooling: sha256(path) for pooling, path in v6_paths.items()}
            if v6_paths
            else {}
        ),
        "source_rows": len(v6_rows),
        "validation": (
            "sample ID, exact caller text, declared image SHA-256, and source and "
            "destination image-byte SHA-256 must match"
        ),
    }
    reuse_manifest = {
        "priority": [
            "schema3_pre_additional_hard_negatives",
            "prior_v7_pre_hard_negative",
            "v6",
        ],
        "rows_reused_total": len(reused_indices),
        "rows_extracted": len(new_rows),
        "evaluation_panel_features_reused_for_training": False,
        "sources": {
            "schema3_pre_additional_hard_negatives": schema3_manifest_entry,
            "prior_v7_pre_hard_negative": prior_manifest_entry,
            "v6": v6_manifest_entry,
        },
    }
    manifest = {
        "schema_version": 1,
        "target": target,
        "panel": panel,
        "rows": len(rows),
        "source_metadata": str(metadata_path.relative_to(RUN)).replace("\\", "/"),
        "source_metadata_sha256": sha256(metadata_path),
        "adapter_metadata_sha256": sha256(adapter_metadata),
        "feature_reuse": reuse_manifest,
        "embeddings": {
            pooling: {
                "path": str(path.relative_to(RUN)).replace("\\", "/"),
                "sha256": sha256(path),
            }
            for pooling, path in expected_paths.items()
        },
    }
    feature_root.mkdir(parents=True, exist_ok=True)
    (feature_root / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"completed {target} {panel} extraction", flush=True)
    return expected_paths


def load_embeddings(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["embeddings"], dtype=np.float64)
        provenance: dict[str, object] = {
            "model_family": scalar(data, "model_family"),
            "model_id": scalar(data, "model_id"),
            "model_revision": scalar(data, "model_revision"),
            "tokenizer_revision": scalar(data, "tokenizer_revision"),
            "preprocessing_sha256": scalar(data, "preprocessing_sha256"),
            "layer": int(np.asarray(data["layer"]).reshape(-1)[0]),
            "pooling": scalar(data, "pooling"),
        }
    return features, provenance


def load_candidate_embeddings(
    paths: dict[str, Path],
    pooling: str,
) -> tuple[np.ndarray, dict[str, object]]:
    if pooling in EXTRACTION_POOLINGS:
        return load_embeddings(paths[pooling])
    if pooling != "text_image_tokens":
        raise ValueError(f"unsupported candidate pooling: {pooling}")
    text_features, text_provenance = load_embeddings(paths["text_tokens"])
    image_features, image_provenance = load_embeddings(paths["image_tokens"])
    if text_features.shape[0] != image_features.shape[0]:
        raise ValueError("text and image embeddings have different row counts")
    if _provenance_without_pooling(text_provenance) != _provenance_without_pooling(
        image_provenance
    ):
        raise ValueError("text and image embedding provenance does not match")
    provenance = dict(_provenance_without_pooling(text_provenance))
    provenance["pooling"] = "text_image_tokens"
    return np.concatenate((text_features, image_features), axis=1), provenance


def select_block_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    malicious_subgroups: dict[str, np.ndarray],
    *,
    max_false_positive_rate: float = MAX_VALIDATION_FALSE_POSITIVE_RATE,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("labels and scores must be aligned 1D arrays")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain non-finite values")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("scores must be probabilities in [0, 1]")
    if not malicious_subgroups:
        raise ValueError("threshold selection requires malicious subgroup gates")
    gated = np.zeros(labels.shape, dtype=bool)
    for name, values in malicious_subgroups.items():
        mask = np.asarray(values, dtype=bool)
        if mask.shape != labels.shape:
            raise ValueError(f"{name} validation mask is not aligned with labels")
        if not np.any(mask):
            raise ValueError(f"{name} malicious subgroup is empty")
        if np.any(labels[mask] != 1):
            raise ValueError(f"{name} malicious subgroup includes a benign example")
        gated |= mask
    benign = scores[labels == 0]
    malicious = scores[labels == 1]
    if len(benign) == 0 or len(malicious) == 0:
        raise ValueError("threshold selection requires both classes")
    if np.any((labels == 1) & ~gated):
        raise ValueError("malicious subgroup gates do not cover every malicious row")

    # Validate the configured budget here, but never spend it to lower the
    # threshold. Candidate acceptance applies the budget as a separate gate.
    false_positive_budget(len(benign), max_false_positive_rate)
    threshold = threshold_immediately_below(
        float(np.min(scores[gated])),
        boundary_name="weakest gated malicious score",
    )
    if not np.all(scores[gated] >= threshold):
        raise RuntimeError("selected threshold violates a malicious recall gate")
    return threshold


def threshold_immediately_below(
    score_boundary: float,
    *,
    boundary_name: str = "score boundary",
) -> float:
    boundary = np.float64(score_boundary)
    if not np.isfinite(boundary) or not 0.0 <= boundary <= 1.0:
        raise ValueError(f"{boundary_name} must be a finite probability")
    threshold = float(np.nextafter(boundary, np.float64(-np.inf)))
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"{boundary_name} does not admit a positive detector threshold "
            "immediately below it"
        )
    if not threshold < boundary:
        raise RuntimeError(f"selected threshold is not below {boundary_name}")
    return threshold


def select_operating_block_threshold(
    validation_malicious_scores: np.ndarray,
    prior_regression_malicious_scores: np.ndarray,
) -> float:
    validation = np.asarray(validation_malicious_scores, dtype=np.float64)
    regression = np.asarray(prior_regression_malicious_scores, dtype=np.float64)
    for name, values in (
        ("validation malicious scores", validation),
        ("prior-regression malicious scores", regression),
    ):
        if values.ndim != 1 or len(values) == 0:
            raise ValueError(f"{name} must be a non-empty 1D array")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contain non-finite values")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} must be probabilities in [0, 1]")
    cap_score = min(float(np.min(validation)), float(np.min(regression)))
    return threshold_immediately_below(
        cap_score,
        boundary_name="validation/prior-regression operating-point cap",
    )


def false_positive_budget(benign_count: int, false_positive_rate: float) -> int:
    if isinstance(benign_count, bool) or not isinstance(benign_count, int):
        raise ValueError("benign_count must be an integer")
    if benign_count <= 0:
        raise ValueError("benign_count must be positive")
    if not np.isfinite(false_positive_rate):
        raise ValueError("false_positive_rate must be finite")
    if not 0.0 <= false_positive_rate < 1.0:
        raise ValueError("false_positive_rate must be in [0, 1)")
    allowed = math.floor(float(false_positive_rate) * benign_count)
    if allowed >= benign_count:
        raise ValueError("false-positive budget must leave a benign anchor")
    return allowed


def malicious_subgroup_masks(
    rows: list[dict[str, str]],
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if len(rows) != len(labels):
        raise ValueError("rows and labels are not aligned")
    strategies = np.asarray([row.get("strategy", "") for row in rows])
    malicious = labels == 1
    groups: dict[str, np.ndarray] = {
        "all_malicious": malicious,
        "image_only": malicious & (strategies == IMAGE_ONLY_STRATEGY),
        "text_led": malicious & (strategies == TEXT_LED_STRATEGY),
    }
    for strategy in sorted(set(strategies[malicious])):
        if "split" in str(strategy).lower():
            groups[f"split_{strategy}"] = malicious & (strategies == strategy)
    empty_required = [name for name in ("image_only", "text_led") if not groups[name].any()]
    if empty_required:
        raise ValueError(
            "required malicious subgroups are empty: " + ", ".join(empty_required)
        )
    return groups


def training_sample_weights(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    *,
    image_only_malicious_weight: float = 1.0,
    text_led_malicious_weight: float = 2.0,
    counterfactual_paired_benign_weight: float = 2.0,
    benign_weight: float = 1.0,
    hard_negative_weight: float = 2.5,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if len(rows) != len(labels):
        raise ValueError("rows and labels are not aligned")
    validate_hard_negative_metadata(rows)
    configured_weights = {
        "image-only malicious": image_only_malicious_weight,
        "text-led malicious": text_led_malicious_weight,
        "counterfactual paired benign": counterfactual_paired_benign_weight,
        "ordinary benign": benign_weight,
        "hard-negative benign": hard_negative_weight,
    }
    for name, value in configured_weights.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} weight must be finite and positive")
    paired_benign_ids = validate_counterfactual_training_pairs(rows, labels)
    values = np.empty(len(rows), dtype=np.float64)
    for index, (row, label) in enumerate(zip(rows, labels)):
        if label != int(row["label_id"]):
            raise ValueError(f"label array differs from metadata: {row['sample_id']}")
        if label == 0 and row["hard_negative"] == "1":
            values[index] = hard_negative_weight
        elif label == 0 and row["sample_id"] in paired_benign_ids:
            values[index] = counterfactual_paired_benign_weight
        elif label == 0:
            values[index] = benign_weight
        elif row.get("strategy") == TEXT_LED_STRATEGY:
            values[index] = text_led_malicious_weight
        else:
            values[index] = image_only_malicious_weight
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("training sample weights must be finite and positive")
    return values


def validate_counterfactual_training_pairs(
    rows: list[dict[str, str]],
    labels: np.ndarray,
) -> set[str]:
    """Validate and return benign controls paired with text-led training rows."""
    labels = np.asarray(labels, dtype=np.int64)
    if len(rows) != len(labels):
        raise ValueError("rows and labels are not aligned")
    by_id: dict[str, tuple[dict[str, str], int]] = {}
    for row, label in zip(rows, labels):
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            raise ValueError("training row has an empty sample_id")
        if sample_id in by_id:
            raise ValueError(f"duplicate training sample ID: {sample_id}")
        try:
            metadata_label = int(row["label_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid label_id: {sample_id}") from exc
        if int(label) != metadata_label:
            raise ValueError(f"label array differs from metadata: {sample_id}")
        by_id[sample_id] = (row, metadata_label)

    paired_benign_ids: set[str] = set()
    for row, label in zip(rows, labels):
        if int(label) != 1 or row.get("strategy") != TEXT_LED_STRATEGY:
            continue
        sample_id = row["sample_id"]
        paired_id = row.get("paired_benign_sample_id", "").strip()
        if not paired_id:
            raise ValueError(
                f"text-led malicious row lacks paired_benign_sample_id: {sample_id}"
            )
        if paired_id in paired_benign_ids:
            raise ValueError(
                f"counterfactual benign row is referenced more than once: {paired_id}"
            )
        paired = by_id.get(paired_id)
        if paired is None:
            raise ValueError(
                f"counterfactual benign row does not exist in training split: "
                f"{sample_id} -> {paired_id}"
            )
        paired_row, paired_label = paired
        if paired_label != 0:
            raise ValueError(
                f"counterfactual pair target is not benign: {sample_id} -> {paired_id}"
            )
        source_split = row.get("split", "").strip()
        paired_split = paired_row.get("split", "").strip()
        if not source_split or not paired_split:
            raise ValueError(
                f"counterfactual pair is missing split metadata: "
                f"{sample_id} -> {paired_id}"
            )
        if source_split != paired_split:
            raise ValueError(
                f"counterfactual pair crosses splits: {sample_id} -> {paired_id}"
            )
        if paired_row.get("hard_negative") != "0":
            raise ValueError(
                f"counterfactual paired benign row cannot be a hard negative: "
                f"{paired_id}"
            )
        paired_benign_ids.add(paired_id)
    return paired_benign_ids


def sample_weighting_summary(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(rows) != len(labels) or weights.shape != labels.shape:
        raise ValueError("rows, labels, and weights are not aligned")
    validate_hard_negative_metadata(rows)
    paired_benign_ids = validate_counterfactual_training_pairs(rows, labels)
    strategies = np.asarray([row["strategy"] for row in rows])
    hard_negative = np.asarray(
        [row["hard_negative"] == "1" for row in rows], dtype=bool
    )
    paired_benign = np.asarray(
        [row["sample_id"] in paired_benign_ids for row in rows], dtype=bool
    )
    groups = {
        "ordinary_benign": (labels == 0) & ~hard_negative & ~paired_benign,
        "counterfactual_paired_benign": (labels == 0) & paired_benign,
        "hard_negative_benign": (labels == 0) & hard_negative,
        "image_only_malicious": (labels == 1)
        & (strategies != TEXT_LED_STRATEGY),
        "text_led_malicious": (labels == 1)
        & (strategies == TEXT_LED_STRATEGY),
    }
    if np.any(np.sum(np.vstack(list(groups.values())), axis=0) != 1):
        raise ValueError("sample-weight groups do not partition training rows")
    group_summary = {
        name: {
            "rows": int(np.sum(mask)),
            "weighted_mass": float(np.sum(weights[mask])),
        }
        for name, mask in groups.items()
    }
    categories: dict[str, dict[str, object]] = {}
    for category in sorted(HARD_NEGATIVE_CATEGORIES):
        mask = np.asarray(
            [row["hard_negative_category"] == category for row in rows],
            dtype=bool,
        )
        if np.any(mask):
            categories[category] = {
                "rows": int(np.sum(mask)),
                "weighted_mass": float(np.sum(weights[mask])),
            }
    return {
        "groups": group_summary,
        "counterfactual_pairing": {
            "pairs": len(paired_benign_ids),
            "paired_benign_id_field": "paired_benign_sample_id",
            "one_to_one_required": True,
            "same_split_required": True,
            "paired_benign_must_be_non_hard_negative": True,
        },
        "hard_negative_categories": categories,
        "benign_weighted_mass": float(
            group_summary["ordinary_benign"]["weighted_mass"]
            + group_summary["counterfactual_paired_benign"]["weighted_mass"]
            + group_summary["hard_negative_benign"]["weighted_mass"]
        ),
        "malicious_weighted_mass": float(
            group_summary["image_only_malicious"]["weighted_mass"]
            + group_summary["text_led_malicious"]["weighted_mass"]
        ),
        "total_weighted_mass": float(np.sum(weights)),
    }


def malicious_subgroup_summary(
    scores: np.ndarray,
    mask: np.ndarray,
    block_threshold: float,
) -> dict[str, object]:
    selected = np.asarray(scores, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if len(selected) == 0:
        raise ValueError("malicious subgroup is empty")
    blocked = int(np.sum(selected >= block_threshold))
    return {
        "n": len(selected),
        "blocked": blocked,
        "false_negatives": len(selected) - blocked,
        "recall": blocked / len(selected),
        "minimum_score": float(np.min(selected)),
        "maximum_score": float(np.max(selected)),
    }


def review_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    block_threshold: float,
) -> float:
    benign = np.asarray(scores)[np.asarray(labels) == 0]
    benign_max = float(np.max(benign))
    if benign_max < block_threshold:
        value = (benign_max + block_threshold) / 2.0
    else:
        value = block_threshold * 0.95
    return min(max(value, 0.0), block_threshold - 1e-6)


def prediction_rows(
    rows: list[dict[str, str]],
    scores: np.ndarray,
    block: float,
    review: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row, score in zip(rows, scores):
        value = float(score)
        action = "block" if value >= block else "review" if value >= review else "allow"
        output.append(
            {
                "sample_id": row["sample_id"],
                "label_id": int(row["label_id"]),
                "split": row["split"],
                "group_id": row["group_id"],
                "attack_style": row["attack_style"],
                "risk_score": value,
                "recommended_action": action,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def train(
    target: str,
    development_rows: list[dict[str, str]],
    regression_rows: list[dict[str, str]],
    development_paths: dict[str, Path],
    regression_paths: dict[str, Path],
    *,
    image_only_malicious_weight: float = 1.0,
    text_led_malicious_weight: float = 2.0,
    counterfactual_paired_benign_weight: float = 2.0,
    benign_weight: float = 1.0,
    hard_negative_weight: float = 2.5,
    candidate_poolings: Iterable[str] = CANDIDATE_POOLINGS,
    candidate_image_only_malicious_weights: Iterable[float] | None = None,
    candidate_hard_negative_weights: Iterable[float] | None = None,
    candidate_learning_rates: Iterable[float] = CANDIDATE_LEARNING_RATES,
    candidate_l2_values: Iterable[float] = CANDIDATE_L2_VALUES,
) -> dict[str, object]:
    evaluated_poolings = validate_candidate_poolings(
        candidate_poolings,
        require_fused=True,
    )
    image_only_weight_grid = validate_positive_candidate_grid(
        candidate_image_only_malicious_weights
        if candidate_image_only_malicious_weights is not None
        else (image_only_malicious_weight,),
        name="candidate image-only malicious weights",
    )
    hard_negative_weight_grid = validate_positive_candidate_grid(
        candidate_hard_negative_weights
        if candidate_hard_negative_weights is not None
        else (hard_negative_weight,),
        name="candidate hard-negative weights",
    )
    learning_rate_grid = validate_positive_candidate_grid(
        candidate_learning_rates,
        name="candidate learning rates",
    )
    l2_grid = validate_positive_candidate_grid(
        candidate_l2_values,
        name="candidate L2 values",
    )
    for fixed_name, fixed_weight in (
        ("text-led malicious weight", text_led_malicious_weight),
        (
            "counterfactual paired benign weight",
            counterfactual_paired_benign_weight,
        ),
        ("ordinary benign weight", benign_weight),
    ):
        validate_positive_candidate_grid((fixed_weight,), name=fixed_name)
    labels = np.asarray([int(row["label_id"]) for row in development_rows])
    splits = np.asarray([row["split"] for row in development_rows])
    train_index = np.flatnonzero(splits == "train")
    validation_index = np.flatnonzero(splits == "validation")
    test_index = np.flatnonzero(splits == "test")
    if any(len(index) == 0 for index in (train_index, validation_index, test_index)):
        raise RuntimeError("train, validation, and test partitions must all be non-empty")
    if len(train_index) + len(validation_index) + len(test_index) != len(labels):
        raise RuntimeError("development metadata contains an unsupported split")

    train_rows = [development_rows[int(index)] for index in train_index]
    validation_rows = [development_rows[int(index)] for index in validation_index]
    test_rows = [development_rows[int(index)] for index in test_index]
    train_labels = labels[train_index]
    validation_labels = labels[validation_index]
    test_labels = labels[test_index]
    validation_groups = malicious_subgroup_masks(validation_rows, validation_labels)
    test_groups = malicious_subgroup_masks(test_rows, test_labels)

    candidates: list[dict[str, object]] = []
    fitted: dict[
        int,
        tuple[
            LogisticRegressionNumpy,
            np.ndarray,
            dict[str, object],
            np.ndarray,
        ],
    ] = {}
    candidate_id = 0
    for pooling in evaluated_poolings:
        features, provenance = load_candidate_embeddings(development_paths, pooling)
        for (
            candidate_image_weight,
            candidate_hard_weight,
            learning_rate,
            l2,
        ) in (
            (image_weight, hard_weight, rate, penalty)
            for image_weight in image_only_weight_grid
            for hard_weight in hard_negative_weight_grid
            for rate in learning_rate_grid
            for penalty in l2_grid
        ):
                candidate_weights = training_sample_weights(
                    train_rows,
                    train_labels,
                    image_only_malicious_weight=candidate_image_weight,
                    text_led_malicious_weight=text_led_malicious_weight,
                    counterfactual_paired_benign_weight=(
                        counterfactual_paired_benign_weight
                    ),
                    benign_weight=benign_weight,
                    hard_negative_weight=candidate_hard_weight,
                )
                classifier = LogisticRegressionNumpy(
                    learning_rate=learning_rate,
                    epochs=CANDIDATE_EPOCHS,
                    l2=l2,
                    standardize=CANDIDATE_STANDARDIZE,
                    random_seed=CANDIDATE_RANDOM_SEED,
                ).fit(
                    features[train_index],
                    train_labels,
                    sample_weight=candidate_weights,
                )
                validation_scores = classifier.predict_proba(features[validation_index])
                validation_benign = validation_scores[validation_labels == 0]
                validation_fp_budget = false_positive_budget(
                    len(validation_benign),
                    MAX_VALIDATION_FALSE_POSITIVE_RATE,
                )
                validation_malicious = validation_scores[validation_labels == 1]
                validation_weakest_malicious = float(
                    np.min(validation_malicious)
                )
                block = select_block_threshold(
                    validation_labels,
                    validation_scores,
                    validation_groups,
                )
                validation_unavoidable_false_positives = int(
                    np.sum(validation_benign >= block)
                )
                validation_benign_below_threshold = validation_benign[
                    validation_benign < block
                ]
                validation_highest_benign_below_threshold = (
                    float(np.max(validation_benign_below_threshold))
                    if len(validation_benign_below_threshold)
                    else None
                )
                review = review_threshold(
                    validation_labels, validation_scores, block
                )
                validation_metrics = classification_metrics(
                    validation_labels, validation_scores, block
                )
                separation = float(
                    np.min(validation_malicious) - np.max(validation_benign)
                )
                subgroup_summaries = {
                    name: malicious_subgroup_summary(validation_scores, mask, block)
                    for name, mask in validation_groups.items()
                }
                record: dict[str, object] = {
                    "candidate_id": candidate_id,
                    "pooling": pooling,
                    "learning_rate": learning_rate,
                    "epochs": CANDIDATE_EPOCHS,
                    "l2": l2,
                    "image_only_malicious_weight": candidate_image_weight,
                    "text_led_malicious_weight": text_led_malicious_weight,
                    "counterfactual_paired_benign_weight": (
                        counterfactual_paired_benign_weight
                    ),
                    "benign_weight": benign_weight,
                    "hard_negative_weight": candidate_hard_weight,
                    "block_threshold": block,
                    "review_threshold": review,
                    "validation_block_threshold": block,
                    "validation_review_threshold": review,
                    "validation_separation": separation,
                    "validation_false_positive_budget": validation_fp_budget,
                    "validation_unavoidable_false_positives": (
                        validation_unavoidable_false_positives
                    ),
                    "validation_threshold_rule": (
                        "maximum_specificity_weakest_malicious_predecessor"
                    ),
                    "validation_threshold_is_weakest_malicious_predecessor": (
                        block
                        == float(
                            np.nextafter(
                                np.float64(validation_weakest_malicious),
                                np.float64(-np.inf),
                            )
                        )
                    ),
                    "validation_threshold_ulp_gap": float(
                        validation_weakest_malicious - block
                    ),
                    "validation_highest_benign_below_threshold_score": (
                        validation_highest_benign_below_threshold
                    ),
                    "validation_weakest_malicious_score": float(
                        validation_weakest_malicious
                    ),
                    "validation_fpr_target_met": (
                        validation_metrics["false_positive_rate"]
                        <= MAX_VALIDATION_FALSE_POSITIVE_RATE
                    ),
                    **{
                        f"validation_{name}_{key}": value
                        for name, summary in subgroup_summaries.items()
                        for key, value in summary.items()
                    },
                    **{
                        f"validation_{key}": value
                        for key, value in validation_metrics.items()
                    },
                }
                candidates.append(record)
                fitted[candidate_id] = (
                    classifier,
                    features,
                    provenance,
                    candidate_weights,
                )
                candidate_id += 1

    selected = select_validation_candidate(candidates)
    selected_id = int(selected["candidate_id"])
    classifier, features, provenance, selected_weights = fitted[selected_id]
    weighting_evidence = sample_weighting_summary(
        train_rows,
        train_labels,
        selected_weights,
    )
    validation_selection_block = float(selected["block_threshold"])
    validation_selection_review = float(selected["review_threshold"])
    validation_scores = classifier.predict_proba(features[validation_index])
    test_scores = classifier.predict_proba(features[test_index])

    pooling = str(selected["pooling"])
    regression_features, regression_provenance = load_candidate_embeddings(
        regression_paths, pooling
    )
    if regression_provenance != provenance:
        raise RuntimeError("regression feature provenance does not match development")
    regression_scores = classifier.predict_proba(regression_features)
    regression_labels = np.ones(len(regression_rows), dtype=np.int64)

    validation_malicious_scores = validation_scores[validation_labels == 1]
    validation_operating_cap_score = float(np.min(validation_malicious_scores))
    prior_regression_operating_cap_score = float(np.min(regression_scores))
    operating_cap_score = min(
        validation_operating_cap_score,
        prior_regression_operating_cap_score,
    )
    block = select_operating_block_threshold(
        validation_malicious_scores,
        regression_scores,
    )
    if block > validation_selection_block:
        raise RuntimeError(
            "post-selection operating threshold exceeds validation-only threshold"
        )
    review = review_threshold(validation_labels, validation_scores, block)
    selected_for_summary = dict(selected)
    selected_for_summary.update(
        {
            "block_threshold": block,
            "review_threshold": review,
            "validation_selection_block_threshold": validation_selection_block,
            "validation_selection_review_threshold": validation_selection_review,
            "operating_point_rule": (
                "predecessor_of_lower_validation_and_prior_regression_malicious"
            ),
            "operating_point_validation_malicious_minimum_score": (
                validation_operating_cap_score
            ),
            "operating_point_prior_regression_malicious_minimum_score": (
                prior_regression_operating_cap_score
            ),
            "operating_point_cap_score": operating_cap_score,
            "operating_point_threshold_is_cap_predecessor": (
                block
                == float(
                    np.nextafter(
                        np.float64(operating_cap_score),
                        np.float64(-np.inf),
                    )
                )
            ),
            "operating_point_threshold_ulp_gap": float(
                operating_cap_score - block
            ),
        }
    )

    test_metrics = classification_metrics(test_labels, test_scores, block)
    validation_subgroups = {
        name: malicious_subgroup_summary(validation_scores, mask, block)
        for name, mask in validation_groups.items()
    }
    test_subgroups = {
        name: malicious_subgroup_summary(test_scores, mask, block)
        for name, mask in test_groups.items()
    }
    test_benign_scores = test_scores[test_labels == 0]
    test_benign_summary = {
        "n": len(test_benign_scores),
        "allowed": int(np.sum(test_benign_scores < review)),
        "reviewed": int(
            np.sum((test_benign_scores >= review) & (test_benign_scores < block))
        ),
        "blocked": int(np.sum(test_benign_scores >= block)),
        "maximum_score": float(np.max(test_benign_scores)),
    }

    regression_metrics = classification_metrics(
        regression_labels, regression_scores, block
    )

    result_root = RUN / TRAINING_DIRECTORY / target
    result_root.mkdir(parents=True, exist_ok=True)
    artifact_path = result_root / f"{target}_bordair_ocr_v7.npz"
    corpus_manifest_hash = sha256(CORPUS_MANIFEST)
    training_configuration_value = json_safe(
        {
            "schema_version": 2,
            "candidate_poolings": list(evaluated_poolings),
            "classifier_grid": {
                "learning_rates": list(learning_rate_grid),
                "l2_values": list(l2_grid),
                "epochs": CANDIDATE_EPOCHS,
                "standardize": CANDIDATE_STANDARDIZE,
                "random_seed": CANDIDATE_RANDOM_SEED,
            },
            "sample_weight_grid": {
                "image_only_malicious": list(image_only_weight_grid),
                "hard_negative_benign": list(hard_negative_weight_grid),
                "fixed": {
                    "ordinary_benign": benign_weight,
                    "counterfactual_paired_benign": (
                        counterfactual_paired_benign_weight
                    ),
                    "text_led_malicious": text_led_malicious_weight,
                },
            },
            "sample_weights": {
                "ordinary_benign": benign_weight,
                "counterfactual_paired_benign": (
                    counterfactual_paired_benign_weight
                ),
                "hard_negative_benign": float(
                    selected["hard_negative_weight"]
                ),
                "image_only_malicious": float(
                    selected["image_only_malicious_weight"]
                ),
                "text_led_malicious": text_led_malicious_weight,
            },
            "validation_selection": {
                "maximum_false_positive_rate": (
                    MAX_VALIDATION_FALSE_POSITIVE_RATE
                ),
                "threshold_rule": (
                    "maximum_specificity_weakest_malicious_predecessor"
                ),
                "requires_fused_text_image_representation": True,
            },
            "internal_test_qualification": {
                "maximum_false_positive_rate": (
                    MAX_INTERNAL_TEST_FALSE_POSITIVE_RATE
                ),
                "requires_zero_false_negatives": True,
            },
            "operating_point_rule": (
                "predecessor_of_lower_validation_and_prior_regression_malicious"
            ),
        }
    )
    selected_candidate_value = json_safe(selected_for_summary)
    if not isinstance(training_configuration_value, dict) or not isinstance(
        selected_candidate_value, dict
    ):
        raise RuntimeError("training identity normalization produced a non-object")
    training_identity_digest = training_identity_sha256(
        training_configuration_value,
        selected_candidate_value,
    )
    source = detector_source_label(
        corpus_version=CORPUS_VERSION,
        corpus_manifest_sha256=corpus_manifest_hash,
        target=target,
        pooling=pooling,
        training_identity_sha256_value=training_identity_digest,
    )

    write_csv(result_root / "candidate_metrics.csv", candidates)
    write_csv(
        result_root / "validation_predictions.csv",
        prediction_rows(
            [development_rows[int(index)] for index in validation_index],
            validation_scores,
            block,
            review,
        ),
    )
    write_csv(
        result_root / "test_predictions.csv",
        prediction_rows(
            [development_rows[int(index)] for index in test_index],
            test_scores,
            block,
            review,
        ),
    )
    write_csv(
        result_root / "regression_predictions.csv",
        prediction_rows(regression_rows, regression_scores, block, review),
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "target": target,
        "evaluated_poolings": list(evaluated_poolings),
        "selection_rule": (
            "Validation only: evaluate the configured representations ("
            + ", ".join(evaluated_poolings)
            + "). Require the fused text-then-image representation for promotion, "
            "and require 100% recall "
            "for all malicious, image-only, text-led, and any split-strategy subgroup. "
            "Among fused candidates, prefer meeting a 2/80 benign false-positive-rate "
            "target, lower FPR, F1, AUPRC, AUROC, and class-score separation. Set "
            "each candidate's threshold to the greatest representable float strictly "
            "below its weakest gated validation malicious score. This validation-only "
            "maximum-specificity recall boundary is independent of the false-positive "
            "budget. The budget is floor(rate times benign count) and acts only as an "
            "acceptance and selection-qualification gate; a candidate fails that gate "
            "when its unavoidable validation false positives exceed the budget. After "
            "validation-only candidate selection, set the deployed threshold to the "
            "greatest representable float below the lower of the weakest gated "
            "validation malicious score and weakest designated prior-regression "
            "malicious score. Prior regression therefore caps only the post-selection "
            "operating point and remains a pass/fail qualification panel; it cannot "
            "change candidate ranking. Internal test is also excluded from selection "
            "and used only for qualification. Frozen panels are excluded entirely."
        ),
        "selected_candidate": selected_candidate_value,
        "training_configuration": training_configuration_value,
        "training_identity": {
            "schema_version": 1,
            "sha256": training_identity_digest,
        },
        "operating_point": {
            "rule": (
                "predecessor_of_lower_validation_and_prior_regression_malicious"
            ),
            "validation_malicious_minimum_score": validation_operating_cap_score,
            "prior_regression_malicious_minimum_score": (
                prior_regression_operating_cap_score
            ),
            "cap_score": operating_cap_score,
            "block_threshold": block,
            "threshold_is_cap_predecessor": selected_for_summary[
                "operating_point_threshold_is_cap_predecessor"
            ],
            "threshold_ulp_gap": selected_for_summary[
                "operating_point_threshold_ulp_gap"
            ],
            "candidate_ranking_uses_prior_regression": False,
            "candidate_ranking_uses_internal_test": False,
            "candidate_ranking_uses_frozen_panels": False,
        },
        "validation_metrics": classification_metrics(
            validation_labels, validation_scores, block
        ),
        "validation_subgroups": validation_subgroups,
        "test_metrics": test_metrics,
        "test_subgroups": test_subgroups,
        "test_benign_actions": test_benign_summary,
        "regression_metrics": regression_metrics,
        "regression_blocked": int(np.sum(regression_scores >= block)),
        "regression_rows": len(regression_rows),
        "artifact": None,
        "artifact_sha256": None,
        "artifact_source": source,
        "provenance": provenance,
        "feature_dimension": int(features.shape[1]),
        "sample_weighting": {
            "configured_weights": {
                "ordinary_benign": benign_weight,
                "counterfactual_paired_benign": (
                    counterfactual_paired_benign_weight
                ),
                "hard_negative_benign": float(
                    selected["hard_negative_weight"]
                ),
                "image_only_malicious": float(
                    selected["image_only_malicious_weight"]
                ),
                "text_led_malicious": text_led_malicious_weight,
            },
            **weighting_evidence,
        },
        "maximum_validation_false_positive_rate_target": (
            MAX_VALIDATION_FALSE_POSITIVE_RATE
        ),
        "maximum_internal_test_false_positive_rate": (
            MAX_INTERNAL_TEST_FALSE_POSITIVE_RATE
        ),
        "corpus_provenance": {
            "manifest_sha256": corpus_manifest_hash,
            "development_metadata_sha256": sha256(DEVELOPMENT_METADATA),
            "regression_metadata_sha256": sha256(REGRESSION_METADATA),
        },
        "review_threshold": review,
        "limitations": (
            "Applies to the evaluated Bordair-derived OCR attacks, malicious caller "
            "text paired with benign carrier images, and official safely renderable "
            "benign controls. It does not validate unavailable native images for the "
            "other image-encoding labels or attack families outside this corpus."
        ),
    }
    acceptance_checks = {
        "fused_text_image_representation": pooling == "text_image_tokens",
        "validation_all_malicious_blocked": (
            validation_subgroups["all_malicious"]["false_negatives"] == 0
        ),
        "validation_image_only_all_blocked": (
            validation_subgroups["image_only"]["false_negatives"] == 0
        ),
        "validation_text_led_all_blocked": (
            validation_subgroups["text_led"]["false_negatives"] == 0
        ),
        "validation_false_positive_rate_at_most_0_025": (
            summary["validation_metrics"]["false_positive_rate"]
            <= MAX_VALIDATION_FALSE_POSITIVE_RATE
        ),
        "internal_test_no_false_negatives": test_metrics["false_negatives"] == 0.0,
        "internal_test_false_positive_rate_at_most_0_10": (
            test_metrics["false_positive_rate"]
            <= MAX_INTERNAL_TEST_FALSE_POSITIVE_RATE
        ),
        "internal_test_image_only_all_blocked": (
            test_subgroups["image_only"]["false_negatives"] == 0
        ),
        "internal_test_text_led_all_blocked": (
            test_subgroups["text_led"]["false_negatives"] == 0
        ),
        "prior_regression_all_blocked": int(np.sum(regression_scores >= block))
        == len(regression_rows),
    }
    summary["acceptance"] = {
        "passed": all(acceptance_checks.values()),
        "checks": acceptance_checks,
    }
    if bool(summary["acceptance"]["passed"]):
        artifact = DetectorArtifact(
            classifier=classifier,
            threshold=block,
            uncertainty_margin=0.02,
            model_family=str(provenance["model_family"]),
            model_id=str(provenance["model_id"]),
            model_revision=str(provenance["model_revision"]),
            tokenizer_revision=str(provenance["tokenizer_revision"]),
            preprocessing_sha256=str(provenance["preprocessing_sha256"]),
            layer=int(provenance["layer"]),
            pooling=pooling,
            source=source,
        )
        save_detector_artifact(artifact_path, artifact)
        summary["artifact"] = str(artifact_path.relative_to(RUN)).replace("\\", "/")
        summary["artifact_sha256"] = sha256(artifact_path)
    elif artifact_path.exists():
        artifact_path.unlink()
    normalized_summary = json_safe(summary)
    if not isinstance(normalized_summary, dict):
        raise RuntimeError("training summary normalization produced a non-object")
    summary = normalized_summary
    summary_path = result_root / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    output_files = [
        result_root / "candidate_metrics.csv",
        result_root / "validation_predictions.csv",
        result_root / "test_predictions.csv",
        result_root / "regression_predictions.csv",
        summary_path,
    ]
    if artifact_path.is_file():
        output_files.append(artifact_path)
    output_manifest = {
        "schema_version": 1,
        "target": target,
        "source": source,
        "training_identity_sha256": training_identity_digest,
        "outputs": {
            str(path.relative_to(result_root)).replace("\\", "/"): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_files
        },
    }
    (result_root / "output_manifest.json").write_text(
        json.dumps(
            output_manifest,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        flush=True,
    )
    if not bool(summary["acceptance"]["passed"]):
        raise RuntimeError(f"{target} failed detector-promotion acceptance criteria")
    return summary


def main() -> None:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    corpus_validation = validate_run(RUN)
    if corpus_validation.get("ok") is not True:
        raise RuntimeError("strict Bordair corpus validation did not pass")
    target_info = resolved_target_info(args)
    candidate_grids = resolved_candidate_grids(args)

    development_rows = read_metadata(
        DEVELOPMENT_METADATA, require_hard_negative=True
    )
    regression_rows = read_metadata(
        REGRESSION_METADATA, require_hard_negative=True
    )
    development_paths = extract_panel(
        args.target,
        "development",
        DEVELOPMENT_METADATA,
        development_rows,
        target_info,
        args.qwen_chunk_size,
    )
    regression_paths = extract_panel(
        args.target,
        "regression",
        REGRESSION_METADATA,
        regression_rows,
        target_info,
        args.qwen_chunk_size,
    )
    if not args.extract_only and args.artifact_mode == "fused":
        train(
            args.target,
            development_rows,
            regression_rows,
            development_paths,
            regression_paths,
            image_only_malicious_weight=args.image_only_malicious_weight,
            text_led_malicious_weight=args.text_led_malicious_weight,
            counterfactual_paired_benign_weight=(
                args.counterfactual_paired_benign_weight
            ),
            benign_weight=args.benign_weight,
            hard_negative_weight=args.hard_negative_weight,
            candidate_poolings=args.candidate_poolings,
            candidate_image_only_malicious_weights=(
                candidate_grids["image_only_malicious_weights"]
            ),
            candidate_hard_negative_weights=(
                candidate_grids["hard_negative_weights"]
            ),
            candidate_learning_rates=candidate_grids["learning_rates"],
            candidate_l2_values=candidate_grids["l2_values"],
        )
    elif not args.extract_only:
        from .bordair_dual_training import (
            DEFAULT_IMAGE_POSITIVE_WEIGHTS,
            DEFAULT_L2_VALUES,
            DEFAULT_TEXT_POSITIVE_WEIGHTS,
            train_dual_pair,
        )

        development_primitives: dict[str, np.ndarray] = {}
        regression_primitives: dict[str, np.ndarray] = {}
        development_provenance: dict[str, object] | None = None
        regression_provenance: dict[str, object] | None = None
        for pooling in EXTRACTION_POOLINGS:
            development_matrix, development_item_provenance = load_embeddings(
                development_paths[pooling]
            )
            regression_matrix, regression_item_provenance = load_embeddings(
                regression_paths[pooling]
            )
            development_primitives[pooling] = development_matrix
            regression_primitives[pooling] = regression_matrix
            development_shared = _provenance_without_pooling(
                development_item_provenance
            )
            regression_shared = _provenance_without_pooling(regression_item_provenance)
            if development_provenance is None:
                development_provenance = development_shared
            elif development_provenance != development_shared:
                raise RuntimeError("development primitive provenance differs by channel")
            if regression_provenance is None:
                regression_provenance = regression_shared
            elif regression_provenance != regression_shared:
                raise RuntimeError("regression primitive provenance differs by channel")
        if development_provenance is None or development_provenance != regression_provenance:
            raise RuntimeError("development/regression primitive provenance does not match")
        train_dual_pair(
            target=args.target,
            development_rows=development_rows,
            regression_rows=regression_rows,
            development_features=development_primitives,
            regression_features=regression_primitives,
            provenance=development_provenance,
            run_root=RUN,
            corpus_manifest_path=CORPUS_MANIFEST,
            development_metadata_path=DEVELOPMENT_METADATA,
            regression_metadata_path=REGRESSION_METADATA,
            image_positive_weights=(
                DEFAULT_IMAGE_POSITIVE_WEIGHTS
                if args.candidate_image_only_malicious_weights is None
                else candidate_grids["image_only_malicious_weights"]
            ),
            text_positive_weights=(
                DEFAULT_TEXT_POSITIVE_WEIGHTS
                if args.candidate_text_led_malicious_weights is None
                else validate_positive_candidate_grid(
                    args.candidate_text_led_malicious_weights,
                    name="candidate text-led malicious weights",
                )
            ),
            learning_rates=candidate_grids["learning_rates"],
            l2_values=(
                DEFAULT_L2_VALUES
                if args.candidate_l2_values is None
                else candidate_grids["l2_values"]
            ),
            benign_weight=args.benign_weight,
            paired_benign_weight=args.counterfactual_paired_benign_weight,
            hard_negative_weight=args.dual_hard_negative_weight,
            artifact_source_overrides=(
                None
                if args.dual_artifact_source is None
                else {
                    "image": args.dual_artifact_source,
                    "text": args.dual_artifact_source,
                }
            ),
            qualification_evidence_directory=args.dual_qualification_evidence_dir,
        )

    if args.target == "llava05b":
        clear_llava_runtime_cache()
    else:
        clear_qwen_runtime_cache()
    gc.collect()
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
