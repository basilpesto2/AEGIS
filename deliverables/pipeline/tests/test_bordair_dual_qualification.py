from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from aegis_research import bordair_dual_qualification as qualification  # noqa: E402
from aegis_research.bordair_dual import (  # noqa: E402
    build_one_shot_qualification_binding,
    pair_identity_sha256,
    runtime_detector_identity_sha256,
)
from aegis_research.bordair_paths import RunPaths  # noqa: E402


TARGET = "qwen25vl3b"
QWEN_SHARED = {
    "model_family": "qwen25_vl",
    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "tokenizer_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "preprocessing_sha256": "a" * 64,
    "layer": -1,
    "base_feature_dim": 2048,
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_npz(path: Path, **changes: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as data:
        values = {name: np.asarray(data[name]).copy() for name in data.files}
    values.update(changes)
    np.savez(path, **values)


def _refresh_compatibility_source_bindings(
    workspace: Path, changed: dict[str, Path]
) -> None:
    """Rebind changed generic sources through the compatibility hash envelope."""
    evidence = workspace / "evidence"
    results_path = evidence / "dual_head_frozen_results.json"
    evaluation_path = evidence / "evaluation_manifest.json"
    report_path = evidence / "validation_report.json"
    validation_path = evidence / "validation_manifest.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    bindings = {
        name: qualification._hash_binding(path) for name, path in changed.items()
    }
    for payload in (results, evaluation, report, validation):
        payload["generic_sources"].update(bindings)

    _write_json(results_path, results)
    evaluation["outputs"]["dual_head_frozen_results.json"] = (
        qualification._hash_binding(results_path)
    )
    _write_json(evaluation_path, evaluation)
    report["evaluation_manifest"] = qualification._hash_binding(evaluation_path)
    report["evidence"]["results_json"] = qualification._hash_binding(results_path)
    _write_json(report_path, report)
    validation["evaluation_manifest"] = qualification._hash_binding(
        evaluation_path
    )
    validation["validation_report"] = qualification._hash_binding(report_path)
    _write_json(validation_path, validation)


def _write_minimal_model_snapshot(root: Path, *, weight_bytes: bytes) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "config.json", {"model_type": "synthetic"})
    _write_json(root / "tokenizer.json", {"version": "1.0"})
    _write_json(root / "preprocessor_config.json", {"image_size": 224})
    (root / "model.safetensors").write_bytes(weight_bytes)


class SyntheticQualificationFixture:
    def __init__(self, root: Path) -> None:
        self.run_root = root / "run"
        self.paths = RunPaths(self.run_root)
        self.training_root = (
            self.paths.training_directory / TARGET / "dual_or"
        )
        self.summary_path = self.training_root / "training_summary.json"
        self.output_manifest_path = self.training_root / "output_manifest.json"
        self.pair_manifest_path = self.training_root / "detector_pair_manifest.json"
        self.image_artifact_path = self.training_root / f"{TARGET}_image_head_v1.npz"
        self.text_artifact_path = self.training_root / f"{TARGET}_text_head_v1.npz"
        self.model_cache = root / "huggingface"
        self.model_snapshot = (
            self.model_cache
            / "models--Qwen--Qwen2.5-VL-3B-Instruct"
            / "snapshots"
            / QWEN_SHARED["model_revision"]
        )
        self.workspace = root / "qualification"

        _write_json(self.paths.manifest, {"schema_version": 4})
        development_evidence_paths = {
            "feature_snapshot": self.training_root / "development_features.npz",
            "validation": self.training_root / "validation_predictions.npz",
            "internal_test": self.training_root / "internal_test_predictions.npz",
            "prior_regression": self.training_root / "prior_regression_predictions.npz",
        }
        for name, path in development_evidence_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic {name} evidence".encode("utf-8"))
        _write_json(
            self.summary_path,
            {
                "schema_version": 2,
                "target": TARGET,
                "development_qualification_evidence": {
                    "feature_snapshot": {
                        "path": development_evidence_paths[
                            "feature_snapshot"
                        ].name
                    },
                    "predictions": {
                        name: {"path": development_evidence_paths[name].name}
                        for name in (
                            "validation",
                            "internal_test",
                            "prior_regression",
                        )
                    },
                },
            },
        )
        _write_json(self.output_manifest_path, {"schema_version": 2, "target": TARGET})
        _write_json(self.pair_manifest_path, {"schema_version": 1, "target": TARGET})
        self.image_artifact_path.write_bytes(b"synthetic-qwen-image-head")
        self.text_artifact_path.write_bytes(b"synthetic-qwen-text-head")
        _write_minimal_model_snapshot(
            self.model_snapshot, weight_bytes=b"synthetic qwen weights"
        )

        self.panels = self._panels()
        for panel_rows in self.panels.values():
            for row in panel_rows:
                image_path = self.run_root / row["image_path"]
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(
                    f"image bytes for {row['sample_id']}".encode("utf-8")
                )
        panel_paths = {
            "regression": self.paths.final_metadata,
            "text_led": self.paths.text_led_metadata,
            "external_benign": self.paths.external_benign_metadata,
        }
        self.metadata: dict[str, dict[str, object]] = {}
        for panel, panel_path in panel_paths.items():
            _write_csv(panel_path, self.panels[panel])
            self.metadata[panel] = {
                "path": str(panel_path),
                "sha256": _digest(panel_path),
                "rows": len(self.panels[panel]),
            }
        _write_csv(self.paths.development_metadata, self.panels["regression"])
        _write_csv(self.paths.regression_metadata, self.panels["regression"])

        head = lambda pooling: SimpleNamespace(  # noqa: E731
            pooling=pooling,
            feature_dim=2048,
            threshold=0.6,
            uncertainty_margin=0.1,
        )
        self.pair = SimpleNamespace(
            manifest={
                "target": TARGET,
                "composition": "or",
                "pair_identity_sha256": "b" * 64,
                "runtime_detector_identity_sha256": "c" * 64,
                "shared_provenance": dict(QWEN_SHARED),
                "artifacts": {
                    "image": {"review_threshold": 0.5},
                    "text": {"review_threshold": 0.5},
                },
                "cache_representation": {
                    "pooling": "text_image_tokens",
                    "feature_dim": 4096,
                    "ordering": ["text_tokens", "image_tokens"],
                },
            },
            manifest_path=self.pair_manifest_path,
            image_artifact=head("image_tokens"),
            text_artifact=head("text_tokens"),
            base_feature_dim=2048,
            paths=SimpleNamespace(
                image=self.image_artifact_path,
                text=self.text_artifact_path,
            ),
        )
        artifact_entries = {
            "image": {
                "name": "image",
                "path": self.image_artifact_path.name,
                "sha256": _digest(self.image_artifact_path),
                "format": "AEGIS DetectorArtifact NPZ version 1",
                "pooling": "image_tokens",
                "feature_dim": 2048,
                "block_threshold": 0.6,
                "review_threshold": 0.5,
                "source": "synthetic image head",
            },
            "text": {
                "name": "text",
                "path": self.text_artifact_path.name,
                "sha256": _digest(self.text_artifact_path),
                "format": "AEGIS DetectorArtifact NPZ version 1",
                "pooling": "text_tokens",
                "feature_dim": 2048,
                "block_threshold": 0.6,
                "review_threshold": 0.5,
                "source": "synthetic text head",
            },
        }
        pair_manifest = {
            "schema_version": 1,
            "kind": "aegis_detector_pair",
            "composition": "or",
            "target": TARGET,
            "training_identity_sha256": "e" * 64,
            "corpus": {
                "manifest_sha256": _digest(self.paths.manifest),
                "development_metadata_sha256": _digest(
                    self.paths.development_metadata
                ),
                "regression_metadata_sha256": _digest(
                    self.paths.regression_metadata
                ),
            },
            "shared_provenance": dict(QWEN_SHARED),
            "cache_representation": {
                "pooling": "text_image_tokens",
                "feature_dim": 4096,
                "ordering": ["text_tokens", "image_tokens"],
            },
            "development_qualification_evidence": None,
            "qualification_evidence": None,
            "artifacts": artifact_entries,
            "runtime_detector_identity_sha256": runtime_detector_identity_sha256(
                artifact_entries
            ),
        }
        pair_manifest["pair_identity_sha256"] = pair_identity_sha256(pair_manifest)
        self.pair.manifest = pair_manifest
        _write_json(self.pair_manifest_path, pair_manifest)
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        summary["artifact_pair"] = {
            "manifest_sha256": _digest(self.pair_manifest_path),
            "pair_identity_sha256": pair_manifest["pair_identity_sha256"],
            "runtime_detector_identity_sha256": pair_manifest[
                "runtime_detector_identity_sha256"
            ],
        }
        _write_json(self.summary_path, summary)

    def write_feature_bundle(self) -> Path:
        feature_root = self.workspace / "extracted_features"
        feature_root.mkdir(parents=True, exist_ok=True)
        ordered_metadata = self.workspace / "metadata" / "ordered_frozen_metadata.csv"
        protocol = json.loads(
            (self.workspace / "protocol.json").read_text(encoding="utf-8")
        )
        extraction = protocol["feature_extraction_contract"]
        adapter_options = extraction["adapter_options"]
        aligned_metadata = feature_root / "aligned_source_metadata.csv"
        aligned_metadata.write_bytes(ordered_metadata.read_bytes())
        ordered_rows = qualification._csv(ordered_metadata)[1]
        sample_ids = [row["sample_id"] for row in ordered_rows]
        component_dim = int(self.pair.base_feature_dim)
        text = np.zeros((50, component_dim), dtype=np.float64)
        image = np.zeros((50, component_dim), dtype=np.float64)
        for index in range(50):
            text[index, index % component_dim] = 0.1 + index / 1000.0
            image[index, (index + 1) % component_dim] = 0.2 + index / 1000.0
        bundle_path = feature_root / "feature_bundle.npz"
        np.savez(
            bundle_path,
            sample_ids=np.asarray(sample_ids),
            text_embeddings=text,
            image_embeddings=image,
            attribution_features=np.zeros((50, 1), dtype=np.float64),
            feature_source=np.asarray(["synthetic frozen extraction"]),
            model_family=np.asarray([QWEN_SHARED["model_family"]]),
            model_id=np.asarray([QWEN_SHARED["model_id"]]),
            model_revision=np.asarray([QWEN_SHARED["model_revision"]]),
            tokenizer_revision=np.asarray([QWEN_SHARED["tokenizer_revision"]]),
            preprocessing_sha256=np.asarray([QWEN_SHARED["preprocessing_sha256"]]),
            layer=np.asarray([QWEN_SHARED["layer"]], dtype=np.int64),
            pooling=np.asarray(["text_image_tokens"]),
            id_column=np.asarray(["sample_id"]),
        )
        _write_json(
            feature_root / "feature_manifest.json",
            {
                "schema_version": 2,
                "rows": 50,
                "pooling": "text_image_tokens",
                "component_embedding_dimension": component_dim,
                "embedding_dimension": component_dim * 2,
                "metadata_sha256": _digest(ordered_metadata),
                "aligned_source_metadata_sha256": _digest(aligned_metadata),
                "feature_bundle_sha256": _digest(bundle_path),
                "config": {
                    "model_id": QWEN_SHARED["model_id"],
                    "model_revision": QWEN_SHARED["model_revision"],
                    "tokenizer_revision": QWEN_SHARED["tokenizer_revision"],
                    "layer": QWEN_SHARED["layer"],
                    "local_files_only": True,
                    "min_pixels": adapter_options["min_pixels"],
                    "max_pixels": adapter_options["max_pixels"],
                },
                **{
                    name: QWEN_SHARED[name]
                    for name in (
                        "model_family",
                        "model_id",
                        "model_revision",
                        "tokenizer_revision",
                        "preprocessing_sha256",
                        "layer",
                    )
                },
            },
        )
        return feature_root

    def fake_evaluator(self, args: SimpleNamespace) -> int:
        rows: list[dict[str, object]] = []
        sequence = 0
        image_hash = _digest(self.image_artifact_path)
        text_hash = _digest(self.text_artifact_path)
        for panel in qualification.PANEL_ORDER:
            for metadata in self.panels[panel]:
                sequence += 1
                label = int(metadata["label_id"])
                image_action = (
                    "block" if panel == "regression" and label == 1 else "allow"
                )
                text_action = "block" if panel == "text_led" else "allow"
                combined = (
                    "block" if "block" in (image_action, text_action) else "allow"
                )
                cache_path = (
                    Path(args.cache_root)
                    / panel
                    / f"{metadata['sample_id']}.npz"
                )
                rows.append(
                    {
                        "sequence": sequence,
                        "panel": panel,
                        "sample_id": metadata["sample_id"],
                        "label_id": label,
                        "expected_action": "block" if label == 1 else "allow",
                        "prompt_text": metadata["prompt_text"],
                        "image_path": metadata["image_path"],
                        "image_sha256": metadata["image_sha256"],
                        "target": TARGET,
                        "pair_identity_sha256": self.pair.manifest[
                            "pair_identity_sha256"
                        ],
                        "runtime_detector_identity_sha256": self.pair.manifest[
                            "runtime_detector_identity_sha256"
                        ],
                        "image_artifact_sha256": image_hash,
                        "image_head_score": 0.9 if image_action == "block" else 0.1,
                        "image_head_block_threshold": 0.6,
                        "image_head_review_threshold": 0.5,
                        "image_head_action": image_action,
                        "text_artifact_sha256": text_hash,
                        "text_head_score": 0.9 if text_action == "block" else 0.1,
                        "text_head_block_threshold": 0.6,
                        "text_head_review_threshold": 0.5,
                        "text_head_action": text_action,
                        "combined_action": combined,
                        "accepted": combined
                        == ("block" if label == 1 else "allow"),
                        "caller_text_sha256": hashlib.sha256(
                            metadata["prompt_text"].encode("utf-8")
                        ).hexdigest(),
                        "cache_path": (
                            f"{self.subject}/cache/{panel}/{metadata['sample_id']}.npz"
                        ),
                        "cache_sha256": _digest(cache_path),
                    }
                )
        panels = {
            panel: qualification.dual_eval.panel_summary(
                [row for row in rows if row["panel"] == panel]
            )
            for panel in qualification.PANEL_ORDER
        }
        cache_report = {
            "enabled": True,
            "schema_version": 1,
            "directory": qualification._portable(Path(args.cache_root)),
            "expected_rows": 50,
            "rebuild_requested": False,
            "cache_only": True,
            "write_error_details": [],
            **qualification.ONE_SHOT_CACHE_STATISTICS,
        }
        output = {
            "schema_version": 3,
            "evaluation_version": "v7",
            "artifact_mode": "dual_or",
            "generated_utc": "2026-08-30T00:00:00+00:00",
            "target": TARGET,
            "detector_pair": {
                "manifest_sha256": _digest(self.pair_manifest_path),
                "pair_identity_sha256": self.pair.manifest[
                    "pair_identity_sha256"
                ],
                "runtime_detector_identity_sha256": self.pair.manifest[
                    "runtime_detector_identity_sha256"
                ],
                "composition": "or",
                "cache_ordering": ["text_tokens", "image_tokens"],
                "artifacts": {
                    name: {
                        "sha256": self.pair.manifest["artifacts"][name][
                            "sha256"
                        ]
                    }
                    for name in ("image", "text")
                },
            },
            "training_summary": {"sha256": _digest(self.summary_path)},
            "metadata": self.metadata,
            "runtime_validation": {
                "feature_cache": cache_report,
                "qualification_snapshot": args.qualification_snapshot,
            },
            "panels": panels,
            "acceptance": {"passed": True},
            "results": rows,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / f"{TARGET}_dual_or_results.json"
        csv_path = args.output_dir / f"{TARGET}_dual_or_results.csv"
        _write_json(json_path, output)
        qualification._write_csv(
            csv_path,
            rows,
            qualification.dual_eval.DUAL_CSV_FIELDS,
        )
        return 0

    def fake_validator(self, **kwargs):
        generic = json.loads(
            kwargs["evaluation_json_path"].read_text(encoding="utf-8")
        )
        cache_hashes = {
            f"{row['panel']}/{row['sample_id']}.npz": row["cache_sha256"]
            for row in generic["results"]
        }
        cache_tree = qualification.base.sha256_text(
            json.dumps(cache_hashes, sort_keys=True, separators=(",", ":"))
        )
        return {
            "schema_version": 1,
            "validation_kind": "independent_dual_or_evaluation",
            "generated_utc": "2026-08-30T00:00:01+00:00",
            "target": TARGET,
            "passed": True,
            "pair_identity_sha256": self.pair.manifest["pair_identity_sha256"],
            "runtime_detector_identity_sha256": self.pair.manifest[
                "runtime_detector_identity_sha256"
            ],
            "evidence": {
                "evaluation_json_sha256": _digest(kwargs["evaluation_json_path"]),
                "evaluation_csv_sha256": _digest(kwargs["evaluation_csv_path"]),
                "training_summary_sha256": _digest(
                    kwargs["training_summary_path"]
                ),
                "pair_manifest_sha256": _digest(kwargs["pair_manifest_path"]),
                "image_artifact_sha256": _digest(
                    kwargs["image_artifact_path"]
                ),
                "text_artifact_sha256": _digest(
                    kwargs["text_artifact_path"]
                ),
                "corpus_manifest_sha256": _digest(
                    kwargs["corpus_manifest_path"]
                ),
                "cache_entries": 50,
                "cache_statistics": dict(
                    qualification.ONE_SHOT_CACHE_STATISTICS
                ),
                "cache_tree_sha256": cache_tree,
            },
        }

    @staticmethod
    def _option(arguments: tuple[str, ...] | list[str], name: str) -> str:
        index = list(arguments).index(name)
        return str(arguments[index + 1])

    def fake_snapshot(
        self, root: Path, script_name: str, arguments: tuple[str, ...]
    ) -> None:
        del root
        if script_name == "evaluate_bordair_dual_detector.py":
            run_root = Path(self._option(arguments, "--run-root"))
            workspace = Path(
                self._option(arguments, "--qualification-workspace")
            )
            self.fake_evaluator(
                SimpleNamespace(
                    output_dir=Path(self._option(arguments, "--output-dir")),
                    cache_root=Path(self._option(arguments, "--cache-root")),
                    qualification_snapshot=(
                        qualification.dual_eval.validate_declared_snapshot_receipt(
                            workspace=workspace,
                            target=TARGET,
                            run_paths=RunPaths(run_root),
                        )
                    ),
                )
            )
            return
        if script_name == "validate_bordair_dual_evaluation.py":
            evaluation_dir = Path(
                self._option(arguments, "--evaluation-dir")
            )
            run_root = Path(self._option(arguments, "--run-root"))
            training_dir = Path(self._option(arguments, "--training-dir"))
            frozen_training = training_dir / TARGET / "dual_or"
            report = self.fake_validator(
                evaluation_json_path=(
                    evaluation_dir / f"{TARGET}_dual_or_results.json"
                ),
                evaluation_csv_path=(
                    evaluation_dir / f"{TARGET}_dual_or_results.csv"
                ),
                training_summary_path=frozen_training / "training_summary.json",
                pair_manifest_path=(
                    frozen_training / self.pair_manifest_path.name
                ),
                image_artifact_path=(
                    frozen_training / self.image_artifact_path.name
                ),
                text_artifact_path=(
                    frozen_training / self.text_artifact_path.name
                ),
                corpus_manifest_path=run_root / "corpus_manifest_v7.json",
            )
            _write_json(Path(self._option(arguments, "--output")), report)
            return
        raise AssertionError(f"unexpected frozen script: {script_name}")

    def declared_validation(self):
        protocol = json.loads(
            (self.workspace / "protocol.json").read_text(encoding="utf-8")
        )
        frozen_paths = RunPaths(self.workspace / "input_snapshot" / "run")
        frozen_training = frozen_paths.training_directory / TARGET / "dual_or"
        frozen_pair = SimpleNamespace(
            manifest=self.pair.manifest,
            manifest_path=frozen_training / self.pair_manifest_path.name,
            image_artifact=self.pair.image_artifact,
            text_artifact=self.pair.text_artifact,
            base_feature_dim=self.pair.base_feature_dim,
            paths=SimpleNamespace(
                image=frozen_training / self.image_artifact_path.name,
                text=frozen_training / self.text_artifact_path.name,
            ),
        )
        input_manifest = json.loads(
            (self.workspace / "input_snapshot_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        model_before = qualification._verify_model_snapshot_binding(
            protocol["model_snapshot_binding"]
        )
        return (
            protocol,
            frozen_paths,
            self.panels,
            frozen_pair,
            {"ok": True},
            input_manifest,
            model_before,
        )

    @staticmethod
    def _panels() -> dict[str, list[dict[str, str]]]:
        specifications = (
            ("regression", [1] * 10 + [0] * 10),
            ("text_led", [1] * 10),
            ("external_benign", [0] * 20),
        )
        panels: dict[str, list[dict[str, str]]] = {}
        for panel, labels in specifications:
            rows: list[dict[str, str]] = []
            for index, label in enumerate(labels):
                sample_id = f"{panel}-{index:02d}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "label_id": str(label),
                        "prompt_text": f"fixed caller text for {sample_id}",
                        "image_path": f"images_v7/{sample_id}.png",
                        "image_sha256": hashlib.sha256(
                            f"image bytes for {sample_id}".encode("utf-8")
                        ).hexdigest(),
                    }
                )
            panels[panel] = rows
        return panels

    def validated_inputs(self):
        return (
            {"schema_version": 4},
            self.panels,
            self.metadata,
            {"schema_version": 2},
            self.pair,
            {"ok": True},
        )


class QualificationDeclarationTests(unittest.TestCase):
    def test_cache_lineage_is_protocol_bound_and_canonical(self) -> None:
        subject = "a" * 64
        protocol = {
            "qualification_subject": {
                "protocol_version": qualification.QUALIFICATION_PROTOCOL_VERSION,
            }
        }
        locator = qualification._cache_lineage_path(
            protocol=protocol,
            subject=subject,
            panel="regression",
            sample_id="TI-00001",
        )
        self.assertEqual(
            locator,
            f"{subject}/cache/regression/TI-00001.npz",
        )
        for invalid_subject in ("A" * 64, "a" * 63, "g" * 64):
            with self.subTest(invalid_subject=invalid_subject):
                with self.assertRaisesRegex(ValueError, "subject"):
                    qualification._cache_lineage_path(
                        protocol=protocol,
                        subject=invalid_subject,
                        panel="regression",
                        sample_id="TI-00001",
                    )
        with self.assertRaisesRegex(ValueError, "panel"):
            qualification._cache_lineage_path(
                protocol=protocol,
                subject=subject,
                panel="unknown",
                sample_id="TI-00001",
            )
        for invalid_sample_id in (
            "",
            ".",
            "..",
            "../TI-00001",
            "folder/TI-00001",
            "folder\\TI-00001",
            "C:TI-00001",
            "TI-00001\x00tail",
        ):
            with self.subTest(invalid_sample_id=invalid_sample_id):
                with self.assertRaisesRegex(ValueError, "sample ID"):
                    qualification._cache_lineage_path(
                        protocol=protocol,
                        subject=subject,
                        panel="regression",
                        sample_id=invalid_sample_id,
                    )
        with self.assertRaisesRegex(ValueError, "protocol version"):
            qualification._cache_lineage_path(
                protocol={
                    "qualification_subject": {
                        "protocol_version": "schema4-unknown",
                    }
                },
                subject=subject,
                panel="regression",
                sample_id="TI-00001",
            )

    @staticmethod
    def declare(
        fixture: SyntheticQualificationFixture,
        *,
        development_workspace: Path | None = None,
    ) -> dict[str, object]:
        with patch.object(
            qualification,
            "_validated_inputs",
            return_value=fixture.validated_inputs(),
        ):
            result = qualification.declare_workspace(
                target=TARGET,
                run_root=fixture.run_root,
                model_cache_dir=fixture.model_cache,
                development_workspace=development_workspace,
            )
        fixture.workspace = Path(str(result["workspace"])).resolve()
        return result

    def test_qualification_subject_and_canonical_claim_paths_are_deterministic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            corpus_sha256 = _digest(fixture.paths.manifest)
            ordered_identity_sha256 = "d" * 64
            first, payload = qualification._qualification_subject(
                target=TARGET,
                pair=fixture.pair,
                corpus_sha256=corpus_sha256,
                ordered_identity_sha256=ordered_identity_sha256,
            )
            second, duplicate = qualification._qualification_subject(
                target=TARGET,
                pair=fixture.pair,
                corpus_sha256=corpus_sha256,
                ordered_identity_sha256=ordered_identity_sha256,
            )
            self.assertEqual(first, second)
            self.assertEqual(payload, duplicate)
            self.assertEqual(len(first), 64)
            self.assertNotIn("workspace", payload)
            self.assertNotIn("scores", payload)
            self.assertEqual(
                qualification._canonical_workspace(fixture.paths, TARGET, first),
                fixture.paths.training_directory
                / TARGET
                / "dual_or"
                / qualification.WORKSPACE_PARENT
                / first,
            )
            self.assertEqual(
                qualification._subject_claim_path(fixture.paths, TARGET, first),
                fixture.paths.training_directory
                / TARGET
                / "dual_or"
                / f".{qualification.WORKSPACE_PARENT}-claims"
                / f"{first}.json",
            )

            mutations = (
                ("corpus", corpus_sha256, "e" * 64),
                ("ordered identity", ordered_identity_sha256, "f" * 64),
            )
            for name, original, changed in mutations:
                with self.subTest(mutation=name):
                    mutated, _ = qualification._qualification_subject(
                        target=TARGET,
                        pair=fixture.pair,
                        corpus_sha256=(
                            changed if original == corpus_sha256 else corpus_sha256
                        ),
                        ordered_identity_sha256=(
                            changed
                            if original == ordered_identity_sha256
                            else ordered_identity_sha256
                        ),
                    )
                    self.assertNotEqual(first, mutated)

            fixture.image_artifact_path.write_bytes(b"changed image head")
            changed_artifact, _ = qualification._qualification_subject(
                target=TARGET,
                pair=fixture.pair,
                corpus_sha256=corpus_sha256,
                ordered_identity_sha256=ordered_identity_sha256,
            )
            self.assertNotEqual(first, changed_artifact)

    def test_valid_declaration_is_atomic_and_precedes_all_frozen_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            result = self.declare(fixture)

            self.assertEqual(result["state"], "declared")
            self.assertIs(result["promotable"], True)
            self.assertEqual(
                result["protocol_id"],
                f"qwen25vl3b-{qualification.QUALIFICATION_PROTOCOL_VERSION}",
            )
            self.assertEqual(
                fixture.workspace,
                qualification._canonical_workspace(
                    fixture.paths,
                    TARGET,
                    str(result["qualification_subject_sha256"]),
                ),
            )
            for relative in (
                "protocol.json",
                "declaration_manifest.json",
                "development_manifest.json",
                "subject_claim.json",
                "input_snapshot_manifest.json",
                "model_snapshot_manifest.json",
                "strict_corpus_validation.json",
                "evaluate_once.py",
                "validate_once.py",
                "extract_once.py",
                "metadata/final_metadata_v7.csv",
                "metadata/text_led_final_metadata_v7.csv",
                "metadata/external_benign_metadata_v7.csv",
                "metadata/ordered_frozen_metadata.csv",
            ):
                self.assertTrue((fixture.workspace / relative).is_file(), relative)
            for forbidden in (
                "consume_started.json",
                "evidence/attempt_started.json",
                "evidence/dual_head_frozen_results.json",
                "evidence/validation_report.json",
                "cache_materialization.json",
            ):
                self.assertFalse((fixture.workspace / forbidden).exists(), forbidden)

            protocol = json.loads(
                (fixture.workspace / "protocol.json").read_text(encoding="utf-8")
            )
            self.assertEqual(protocol["target"], TARGET)
            self.assertIs(protocol["promotable"], True)
            self.assertEqual(
                protocol["qualification_execution"],
                qualification.QUALIFICATION_EXECUTION,
            )
            self.assertEqual(protocol["artifact_contract"]["per_head_dimension"], 2048)
            self.assertEqual(protocol["cache_contract"]["dimension"], 4096)
            self.assertEqual(
                protocol["cache_contract"]["split_order"],
                ["text_tokens[0:2048]", "image_tokens[2048:4096]"],
            )
            self.assertEqual(len(protocol["ordered_cases"]), 50)
            self.assertEqual(
                [case["panel"] for case in protocol["ordered_cases"]],
                ["regression"] * 20
                + ["text_led"] * 10
                + ["external_benign"] * 20,
            )
            self.assertEqual(
                [case["sequence"] for case in protocol["ordered_cases"]],
                list(range(1, 51)),
            )
            claim_path = qualification._subject_claim_path(
                fixture.paths,
                TARGET,
                str(result["qualification_subject_sha256"]),
            )
            self.assertTrue(claim_path.is_file())
            self.assertEqual(
                claim_path.read_bytes(),
                (fixture.workspace / "subject_claim.json").read_bytes(),
            )

    def test_declaration_is_single_use_and_preserves_existing_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            development_workspace = fixture.workspace
            development_workspace.mkdir()
            sentinel = development_workspace / "partial.txt"
            sentinel.write_text("do not replace", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to reuse"):
                with patch.object(
                    qualification,
                    "_validated_inputs",
                    return_value=fixture.validated_inputs(),
                ):
                    qualification.declare_workspace(
                        target=TARGET,
                        run_root=fixture.run_root,
                        model_cache_dir=fixture.model_cache,
                        development_workspace=development_workspace,
                    )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not replace")

    def test_canonical_subject_claim_prevents_redeclaration_if_workspace_moves(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            result = self.declare(fixture)
            moved = Path(temporary) / "moved-declaration"
            fixture.workspace.rename(moved)
            with (
                patch.object(
                    qualification,
                    "_validated_inputs",
                    return_value=fixture.validated_inputs(),
                ),
                self.assertRaises(FileExistsError),
            ):
                qualification.declare_workspace(
                    target=TARGET,
                    run_root=fixture.run_root,
                    model_cache_dir=fixture.model_cache,
                )
            claim = qualification._subject_claim_path(
                fixture.paths,
                TARGET,
                str(result["qualification_subject_sha256"]),
            )
            self.assertTrue(claim.is_file())

    def test_explicit_development_workspace_is_non_promotable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            development_workspace = fixture.workspace
            result = self.declare(
                fixture, development_workspace=development_workspace
            )
            self.assertIs(result["promotable"], False)
            self.assertEqual(fixture.workspace, development_workspace.resolve())
            protocol = json.loads(
                (fixture.workspace / "protocol.json").read_text(encoding="utf-8")
            )
            self.assertIs(protocol["promotable"], False)
            self.assertEqual(
                protocol["qualification_execution"],
                qualification.NON_PROMOTABLE_EXECUTION,
            )
            self.assertFalse(
                qualification._subject_claim_path(
                    fixture.paths,
                    TARGET,
                    str(result["qualification_subject_sha256"]),
                ).exists()
            )
            self.assertTrue(
                (
                    development_workspace.parent
                    / f".{development_workspace.name}.development-claim.json"
                ).is_file()
            )

    def test_declaration_crash_removes_stage_and_publishes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            real_copy = qualification._copy
            calls = 0

            def fail_during_copy(source: Path, destination: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("injected declaration crash")
                real_copy(source, destination)

            with (
                patch.object(
                    qualification,
                    "_validated_inputs",
                    return_value=fixture.validated_inputs(),
                ),
                patch.object(qualification, "_copy", side_effect=fail_during_copy),
                self.assertRaisesRegex(OSError, "injected declaration crash"),
            ):
                qualification.declare_workspace(
                    target=TARGET,
                    run_root=fixture.run_root,
                    model_cache_dir=fixture.model_cache,
                )

            claim_files = list(
                (
                    fixture.paths.training_directory
                    / TARGET
                    / "dual_or"
                    / f".{qualification.WORKSPACE_PARENT}-claims"
                ).glob("*.json")
            )
            self.assertEqual(len(claim_files), 1)
            claim = json.loads(claim_files[0].read_text(encoding="utf-8"))
            destination = Path(claim["workspace"])
            self.assertFalse(destination.exists())
            leftovers = list(destination.parent.glob(f".{destination.name}.declare-*"))
            self.assertEqual(leftovers, [])
            with (
                patch.object(
                    qualification,
                    "_validated_inputs",
                    return_value=fixture.validated_inputs(),
                ),
                self.assertRaises(FileExistsError),
            ):
                qualification.declare_workspace(
                    target=TARGET,
                    run_root=fixture.run_root,
                    model_cache_dir=fixture.model_cache,
                )

    def test_pair_pooling_and_dimension_mutations_fail_before_declaration(self) -> None:
        mutations = (
            ("image pooling", "image_artifact", "pooling", "text_tokens"),
            ("text pooling", "text_artifact", "pooling", "image_tokens"),
            ("head dimension", "text_artifact", "feature_dim", 1024),
        )
        for name, object_name, field, value in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticQualificationFixture(Path(temporary))
                setattr(getattr(fixture.pair, object_name), field, value)
                with (
                    patch.object(
                        qualification.base,
                        "validate_tracked_v7_corpus",
                        return_value={"ok": True},
                    ),
                    patch.object(
                        qualification.base,
                        "validate_v7_manifest_and_panels",
                        return_value=(
                            {"schema_version": 4},
                            fixture.panels,
                            fixture.metadata,
                        ),
                    ),
                    patch.object(
                        qualification.dual_eval,
                        "validate_dual_training_summary",
                        return_value=({"schema_version": 2}, fixture.pair),
                    ),
                    self.assertRaisesRegex(ValueError, "pooling/dimension"),
                ):
                    qualification.declare_workspace(
                        target=TARGET,
                        run_root=fixture.run_root,
                        model_cache_dir=fixture.model_cache,
                        development_workspace=fixture.workspace,
                    )
                self.assertFalse(fixture.workspace.exists())


class QualificationModelSnapshotTests(unittest.TestCase):
    def test_qwen_cache_snapshot_binding_rejects_tree_mutations(self) -> None:
        mutations = (
            (
                "changed bytes",
                lambda snapshot: (snapshot / "model.safetensors").write_bytes(
                    b"changed qwen weights"
                ),
            ),
            (
                "added runtime file",
                lambda snapshot: _write_json(
                    snapshot / "generation_config.json", {"temperature": 0.0}
                ),
            ),
            (
                "removed tokenizer",
                lambda snapshot: (snapshot / "tokenizer.json").unlink(),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(mutation=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cache = root / "huggingface"
                snapshot = (
                    cache
                    / "models--Qwen--Qwen2.5-VL-3B-Instruct"
                    / "snapshots"
                    / QWEN_SHARED["model_revision"]
                )
                _write_minimal_model_snapshot(
                    snapshot, weight_bytes=b"synthetic qwen weights"
                )
                binding = qualification._model_snapshot_binding(
                    target=TARGET,
                    shared=QWEN_SHARED,
                    model_cache_dir=cache,
                    llava_runtime_model=None,
                )
                self.assertEqual(binding["loader"]["kind"], "huggingface_cache")
                self.assertEqual(
                    binding["snapshots"]["model"]["revision"],
                    QWEN_SHARED["model_revision"],
                )
                self.assertEqual(
                    binding["snapshots"]["model"]["fingerprint"],
                    binding["snapshots"]["tokenizer"]["fingerprint"],
                )
                verified = qualification._verify_model_snapshot_binding(binding)
                self.assertEqual(
                    verified["snapshots"]["model"]["n_files"], 4
                )
                mutate(snapshot)
                with self.assertRaisesRegex(ValueError, "snapshot changed"):
                    qualification._verify_model_snapshot_binding(binding)

    def test_llava_rejects_an_alternate_checkpoint_for_pinned_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pinned = root / "pinned-llava"
            alternate = root / "alternate-llava"
            _write_minimal_model_snapshot(
                pinned, weight_bytes=b"pinned llava weights"
            )
            _write_minimal_model_snapshot(
                alternate, weight_bytes=b"different llava weights"
            )
            revision = qualification.model_directory_fingerprint(pinned)[
                "content_sha256"
            ]
            shared = {
                "model_family": "llava_onevision",
                "model_id": "models/huggingface/llava-onevision-qwen2-0.5b-ov-hf",
                "model_revision": revision,
                "tokenizer_revision": revision,
                "preprocessing_sha256": "a" * 64,
                "layer": -1,
                "base_feature_dim": 896,
            }
            binding = qualification._model_snapshot_binding(
                target="llava05b",
                shared=shared,
                model_cache_dir=root / "unused-cache",
                llava_runtime_model=pinned,
            )
            self.assertEqual(binding["loader"]["kind"], "local_directory")
            self.assertEqual(
                binding["snapshots"]["model"]["fingerprint"]["content_sha256"],
                revision,
            )
            with self.assertRaisesRegex(ValueError, "snapshot is incomplete"):
                qualification._model_snapshot_binding(
                    target="llava05b",
                    shared=shared,
                    model_cache_dir=root / "unused-cache",
                    llava_runtime_model=alternate,
                )


class QualificationInputSnapshotTests(unittest.TestCase):
    @staticmethod
    def create_snapshot(
        fixture: SyntheticQualificationFixture, stage: Path
    ) -> tuple[RunPaths, dict[str, object]]:
        ordered_rows, _ = qualification._ordered(fixture.panels)
        snapshot_paths, _, manifest = qualification._copy_declared_inputs(
            stage=stage,
            target=TARGET,
            paths=fixture.paths,
            selected_summary=fixture.summary_path,
            ordered_rows=ordered_rows,
            pair=fixture.pair,
            corpus_report={"ok": True},
        )
        return snapshot_paths, manifest

    def test_input_snapshot_freezes_all_cases_and_development_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = SyntheticQualificationFixture(root)
            stage = root / "declared"
            stage.mkdir()
            snapshot_paths, manifest = self.create_snapshot(fixture, stage)
            self.assertEqual(manifest["case_rows"], 50)
            self.assertEqual(manifest["unique_sample_ids"], 50)
            self.assertEqual(manifest["unique_image_files"], 50)
            for relative in (
                "run/corpus_manifest_v7.json",
                "run/final_metadata_v7.csv",
                "run/text_led_final_metadata_v7.csv",
                "run/external_benign_metadata_v7.csv",
                "run/images_v7/regression-00.png",
                "run/training_v7/qwen25vl3b/dual_or/training_summary.json",
                "run/training_v7/qwen25vl3b/dual_or/detector_pair_manifest.json",
                "run/training_v7/qwen25vl3b/dual_or/qwen25vl3b_image_head_v1.npz",
                "run/training_v7/qwen25vl3b/dual_or/qwen25vl3b_text_head_v1.npz",
                "run/training_v7/qwen25vl3b/dual_or/development_features.npz",
            ):
                self.assertIn(relative, manifest["files"], relative)
            validated_paths, validated = qualification._validate_input_snapshot(
                stage, target=TARGET
            )
            self.assertEqual(validated, manifest)
            self.assertEqual(validated_paths.root, snapshot_paths.root)

            source_image = fixture.run_root / "images_v7" / "regression-00.png"
            frozen_image = snapshot_paths.root / "images_v7" / "regression-00.png"
            frozen_bytes = frozen_image.read_bytes()
            source_image.write_bytes(b"source changed after declaration")
            qualification._validate_input_snapshot(stage, target=TARGET)
            self.assertEqual(frozen_image.read_bytes(), frozen_bytes)

    def test_input_snapshot_rejects_added_removed_and_changed_files(self) -> None:
        mutations = (
            (
                "changed bytes",
                lambda snapshot: (
                    snapshot.root / "images_v7" / "regression-00.png"
                ).write_bytes(b"tampered copied image"),
            ),
            (
                "added file",
                lambda snapshot: (
                    snapshot.root / "images_v7" / "undeclared.bin"
                ).write_bytes(b"undeclared"),
            ),
            (
                "removed file",
                lambda snapshot: (
                    snapshot.root / "images_v7" / "regression-00.png"
                ).unlink(),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(mutation=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = SyntheticQualificationFixture(root)
                stage = root / "declared"
                stage.mkdir()
                snapshot_paths, _ = self.create_snapshot(fixture, stage)
                mutate(snapshot_paths)
                with self.assertRaisesRegex(ValueError, "input snapshot"):
                    qualification._validate_input_snapshot(stage, target=TARGET)


class QualificationConsumptionTests(unittest.TestCase):
    def declare(
        self,
        fixture: SyntheticQualificationFixture,
        *,
        development_workspace: Path | None = None,
    ) -> str:
        with patch.object(
            qualification,
            "_validated_inputs",
            return_value=fixture.validated_inputs(),
        ):
            result = qualification.declare_workspace(
                target=TARGET,
                run_root=fixture.run_root,
                model_cache_dir=fixture.model_cache,
                development_workspace=development_workspace,
            )
        fixture.workspace = Path(str(result["workspace"])).resolve()
        fixture.subject = str(result["qualification_subject_sha256"])
        fixture.development_workspace = development_workspace
        return fixture.subject

    def consume(
        self,
        fixture: SyntheticQualificationFixture,
        *,
        extractor=None,
    ) -> dict[str, object]:
        if extractor is None:
            extractor = lambda **_kwargs: fixture.write_feature_bundle()
        with (
            patch.object(
                qualification,
                "_validate_declaration",
                return_value=fixture.declared_validation(),
            ),
            patch.object(
                qualification,
                "_run_declared_extractor",
                side_effect=extractor,
            ),
            patch.object(
                qualification,
                "_run_snapshot",
                side_effect=fixture.fake_snapshot,
            ),
        ):
            return qualification.consume_workspace(
                target=TARGET,
                run_root=fixture.run_root,
                qualification_subject_sha256=fixture.subject,
                development_workspace=getattr(
                    fixture, "development_workspace", None
                ),
            )

    @staticmethod
    def promotion_binding_arguments(
        fixture: SyntheticQualificationFixture,
    ) -> dict[str, object]:
        frozen_training = (
            fixture.workspace
            / "input_snapshot"
            / "run"
            / "training_v7"
            / TARGET
            / "dual_or"
        )
        return {
            "evidence_directory": fixture.workspace / "evidence",
            "pair_directory": fixture.workspace,
            "image_artifact_sha256": _digest(
                frozen_training / fixture.image_artifact_path.name
            ),
            "text_artifact_sha256": _digest(
                frozen_training / fixture.text_artifact_path.name
            ),
            "corpus_manifest_sha256": _digest(
                fixture.workspace
                / "input_snapshot"
                / "run"
                / "corpus_manifest_v7.json"
            ),
        }

    def bind(self, fixture: SyntheticQualificationFixture) -> dict[str, object]:
        return build_one_shot_qualification_binding(
            **self.promotion_binding_arguments(fixture)
        )

    def test_valid_declare_consume_is_complete_and_second_consume_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)
            result = self.consume(fixture)

            self.assertEqual(result["state"], "consumed_and_validated")
            self.assertEqual(result["cache_entries"], 50)
            self.assertTrue((fixture.workspace / "consume_started.json").is_file())
            self.assertEqual(len(list((fixture.workspace / "cache").rglob("*.npz"))), 50)
            for relative in (
                "evidence/attempt_started.json",
                "evidence/dual_head_frozen_results.json",
                "evidence/dual_head_frozen_results.csv",
                "evidence/evaluation_manifest.json",
                "evidence/validation_report.json",
                "evidence/validation_manifest.json",
                "evidence/cache_materialization.json",
                "evidence/generic/dual_or_results.json",
                "evidence/generic/dual_or_results.csv",
                "evidence/generic/dual_or_validation.json",
                "evidence/source_inputs/feature_bundle.npz",
                "evidence/source_inputs/feature_manifest.json",
                "evidence/source_inputs/aligned_source_metadata.csv",
                "evidence/source_inputs/training_summary.json",
                "evidence/source_inputs/detector_pair_manifest.json",
                "runtime_environment.json",
            ):
                self.assertTrue((fixture.workspace / relative).is_file(), relative)
            self.assertEqual(
                (fixture.workspace / "consume_started.json").read_bytes(),
                (fixture.workspace / "evidence" / "attempt_started.json").read_bytes(),
            )
            self.assertEqual(
                list(fixture.workspace.glob(".*-staging-*")),
                [],
            )

            with self.assertRaisesRegex(FileExistsError, "refusing to reuse"):
                self.consume(fixture)

    def test_strict_generic_package_builds_promotion_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)
            self.consume(fixture)
            binding = self.bind(fixture)
            self.assertTrue(binding["passed"])
            self.assertEqual(
                binding["source_format"], qualification.GENERIC_DERIVED_PACKAGE
            )
            cache_files = {
                name: entry
                for name, entry in binding["files"].items()
                if name.startswith("cache::")
            }
            self.assertEqual(len(cache_files), 50)
            self.assertEqual(
                set(cache_files),
                {
                    f"cache::{panel}/{row['sample_id']}"
                    for panel, rows in fixture.panels.items()
                    for row in rows
                },
            )
            self.assertEqual(
                binding["protocol_qualification_subject_sha256"],
                fixture.subject,
            )

    def test_strict_generic_binding_allows_checkout_line_ending_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)
            self.consume(fixture)
            expected = self.bind(fixture)
            approved_scripts = {
                (PIPELINE_ROOT / "scripts" / name).resolve()
                for name in (
                    "evaluate_bordair_dual_detector.py",
                    "validate_bordair_dual_evaluation.py",
                    "extract_mllm_features.py",
                )
            }
            original_read_bytes = Path.read_bytes

            for line_ending in (b"\n", b"\r\n"):
                with self.subTest(line_ending=line_ending):
                    def checkout_read_bytes(path: Path) -> bytes:
                        content = original_read_bytes(path)
                        if path.resolve() in approved_scripts:
                            return content.replace(b"\r\n", b"\n").replace(
                                b"\n", line_ending
                            )
                        return content

                    with patch.object(Path, "read_bytes", checkout_read_bytes):
                        self.assertEqual(self.bind(fixture), expected)

    def test_strict_generic_binding_rejects_changed_approved_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)
            self.consume(fixture)
            original_read_bytes = Path.read_bytes

            for name in (
                "evaluate_bordair_dual_detector.py",
                "validate_bordair_dual_evaluation.py",
                "extract_mllm_features.py",
            ):
                with self.subTest(script=name):
                    changed_script = (PIPELINE_ROOT / "scripts" / name).resolve()

                    def checkout_read_bytes(path: Path) -> bytes:
                        content = original_read_bytes(path)
                        if path.resolve() == changed_script:
                            return content + b"\n# changed approved source\n"
                        return content

                    with patch.object(Path, "read_bytes", checkout_read_bytes):
                        with self.assertRaisesRegex(ValueError, "not approved"):
                            self.bind(fixture)

    def test_strict_generic_binding_allows_different_verifier_runtime_versions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)
            self.consume(fixture)

            with (
                patch.object(
                    qualification.sys,
                    "version",
                    "3.99.0 synthetic promotion verifier",
                ),
                patch.object(
                    qualification.np,
                    "__version__",
                    "99.0.0-synthetic-verifier",
                ),
            ):
                binding = self.bind(fixture)

            self.assertTrue(binding["passed"])
            self.assertEqual(
                binding["source_format"], qualification.GENERIC_DERIVED_PACKAGE
            )

    def test_post_execution_binding_uses_immutable_snapshot_after_source_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)
            self.consume(fixture)

            current = qualification._implementation_sources()
            relative = "AEGIS/target_profiles.py"
            drifted = Path(temporary) / "advanced-target-profiles.py"
            drifted.write_bytes(current[relative].read_bytes() + b"\n# later revision\n")
            advanced = {**current, relative: drifted}

            with (
                patch.object(
                    qualification, "_implementation_sources", return_value=advanced
                ),
                patch.object(qualification.subprocess, "run") as launched,
            ):
                with self.assertRaisesRegex(
                    ValueError, "tracked implementation changed after declaration"
                ):
                    qualification._run_snapshot(
                        fixture.workspace, "evaluate_bordair_dual_detector.py", ()
                    )
                launched.assert_not_called()
                binding = self.bind(fixture)

            self.assertTrue(binding["passed"])
            self.assertEqual(
                binding["source_format"], qualification.GENERIC_DERIVED_PACKAGE
            )

    def test_strict_generic_binding_rejects_tampered_recorded_runtime_versions(
        self,
    ) -> None:
        mutations = (
            ("python", ("python", "version"), "3.98.0 tampered execution"),
            ("numpy", ("packages", "numpy"), "98.0.0-tampered-execution"),
        )
        for name, keys, tampered_value in mutations:
            with self.subTest(version=name), tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticQualificationFixture(Path(temporary))
                self.declare(fixture)
                self.consume(fixture)

                path = fixture.workspace / "runtime_environment.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload[keys[0]][keys[1]] = tampered_value
                _write_json(path, payload)
                _refresh_compatibility_source_bindings(
                    fixture.workspace, {"runtime_environment": path}
                )

                with self.assertRaisesRegex(
                    ValueError, "claim/model/runtime closure"
                ):
                    self.bind(fixture)

    def test_strict_generic_binding_rejects_semantic_source_mutations(self) -> None:
        mutations = (
            "runtime environment",
            "subject claim",
            "model snapshot manifest",
            "implementation snapshot",
            "input snapshot",
            "feature bundle",
            "generic result",
            "generic validation",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticQualificationFixture(Path(temporary))
                self.declare(fixture)
                self.consume(fixture)
                evidence = fixture.workspace / "evidence"
                changed: dict[str, Path] = {}

                if mutation == "runtime environment":
                    path = fixture.workspace / "runtime_environment.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["model_snapshot_pre_post_equal"] = False
                    _write_json(path, payload)
                    changed["runtime_environment"] = path
                elif mutation in {"subject claim", "model snapshot manifest"}:
                    source_name = (
                        "subject_claim"
                        if mutation == "subject claim"
                        else "model_snapshot_manifest"
                    )
                    path = fixture.workspace / f"{source_name}.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if mutation == "subject claim":
                        payload["promotable"] = False
                    else:
                        payload["snapshots"]["model"]["fingerprint"][
                            "total_bytes"
                        ] += 1
                    _write_json(path, payload)
                    declaration_path = fixture.workspace / "declaration_manifest.json"
                    declaration = json.loads(
                        declaration_path.read_text(encoding="utf-8")
                    )
                    declaration["files"][source_name] = qualification._hash_binding(
                        path
                    )
                    _write_json(declaration_path, declaration)
                    changed[source_name] = path
                    changed["declaration_manifest"] = declaration_path
                elif mutation == "implementation snapshot":
                    manifest = json.loads(
                        (
                            fixture.workspace / "implementation_manifest.json"
                        ).read_text(encoding="utf-8")
                    )
                    relative = sorted(manifest["files"])[0]
                    path = fixture.workspace / "code_snapshot" / Path(relative)
                    path.write_bytes(path.read_bytes() + b"\n# semantic tamper\n")
                    changed[f"implementation_snapshot::{relative}"] = path
                elif mutation == "input snapshot":
                    relative = "run/images_v7/regression-00.png"
                    path = fixture.workspace / "input_snapshot" / Path(relative)
                    path.write_bytes(path.read_bytes() + b"tamper")
                    changed[f"input_snapshot::{relative}"] = path
                elif mutation == "feature bundle":
                    path = evidence / "source_inputs" / "feature_bundle.npz"
                    path.write_bytes(path.read_bytes() + b"tamper")
                    changed["feature_bundle"] = path
                elif mutation == "generic result":
                    path = evidence / "generic" / "dual_or_results.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["results"][0]["combined_action"] = "allow"
                    _write_json(path, payload)
                    changed["generic_results_json"] = path
                else:
                    path = evidence / "generic" / "dual_or_validation.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["evidence"]["cache_statistics"]["misses"] = 1
                    _write_json(path, payload)
                    changed["generic_validation_report"] = path

                _refresh_compatibility_source_bindings(
                    fixture.workspace, changed
                )
                with self.assertRaises(ValueError):
                    self.bind(fixture)

    def test_strict_generic_binding_rejects_dynamic_tree_omission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)
            self.consume(fixture)
            implementation = json.loads(
                (fixture.workspace / "implementation_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            relative = sorted(implementation["files"])[0]
            source = fixture.workspace / "code_snapshot" / Path(relative)
            source.rename(source.with_name(source.name + ".omitted"))
            with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                self.bind(fixture)

    def test_strict_generic_binding_rejects_non_atomic_or_non_promotable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            development_workspace = fixture.workspace
            self.declare(
                fixture, development_workspace=development_workspace
            )
            self.consume(fixture)
            with self.assertRaisesRegex(ValueError, "atomic/promotable"):
                self.bind(fixture)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)
            self.consume(fixture)
            evidence = fixture.workspace / "evidence"
            protocol = json.loads(
                (fixture.workspace / "protocol.json").read_text(encoding="utf-8")
            )
            protocol["qualification_execution"] = "non_atomic"
            results = json.loads(
                (evidence / "dual_head_frozen_results.json").read_text(
                    encoding="utf-8"
                )
            )
            report = json.loads(
                (evidence / "validation_report.json").read_text(encoding="utf-8")
            )
            evaluation = json.loads(
                (evidence / "evaluation_manifest.json").read_text(encoding="utf-8")
            )
            validation = json.loads(
                (evidence / "validation_manifest.json").read_text(encoding="utf-8")
            )
            with self.assertRaisesRegex(ValueError, "atomic/promotable"):
                qualification.validate_generic_derived_evidence(
                    evidence_root=evidence,
                    protocol=protocol,
                    results=results,
                    report=report,
                    evaluation_manifest=evaluation,
                    validation_manifest=validation,
                    compatibility_rows=results["results"],
                    compatibility_panels=results["panels"],
                    compatibility_gates=results["gates"],
                    expected_image_sha256=results["artifacts"]["image_head"][
                        "sha256"
                    ],
                    expected_text_sha256=results["artifacts"]["text_head"][
                        "sha256"
                    ],
                    expected_corpus_sha256=results["corpus_manifest"]["sha256"],
                    expected_runtime_identity=fixture.pair.manifest[
                        "runtime_detector_identity_sha256"
                    ],
                )

    def test_failure_after_attempt_is_consumed_and_cleans_partial_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)

            def crashing_extractor(**_kwargs):
                fixture.write_feature_bundle()
                raise RuntimeError("injected post-attempt crash")

            with self.assertRaisesRegex(RuntimeError, "post-attempt crash"):
                self.consume(fixture, extractor=crashing_extractor)

            self.assertTrue((fixture.workspace / "consume_started.json").is_file())
            self.assertTrue((fixture.workspace / "extracted_features").is_dir())
            self.assertTrue((fixture.workspace / "runtime_environment.json").is_file())
            self.assertFalse((fixture.workspace / "cache").exists())
            self.assertFalse((fixture.workspace / "evidence").exists())
            self.assertEqual(list(fixture.workspace.glob(".*-staging-*")), [])
            with self.assertRaises(FileExistsError):
                qualification.consume_workspace(
                    target=TARGET,
                    run_root=fixture.run_root,
                    qualification_subject_sha256=fixture.subject,
                )

    def test_model_snapshot_change_during_extraction_consumes_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)

            def mutating_extractor(**_kwargs):
                feature_root = fixture.write_feature_bundle()
                weight_path = fixture.model_snapshot / "model.safetensors"
                weight_path.write_bytes(weight_path.read_bytes() + b"changed")
                return feature_root

            with self.assertRaisesRegex(ValueError, "model snapshot changed"):
                self.consume(fixture, extractor=mutating_extractor)
            self.assertTrue((fixture.workspace / "consume_started.json").is_file())
            self.assertTrue((fixture.workspace / "extracted_features").is_dir())
            self.assertTrue((fixture.workspace / "runtime_environment.json").is_file())
            self.assertFalse((fixture.workspace / "cache").exists())
            self.assertFalse((fixture.workspace / "evidence").exists())
            self.assertEqual(list(fixture.workspace.glob(".*-staging-*")), [])

    def test_preexisting_partial_output_refuses_without_consuming_attempt(self) -> None:
        partial_paths = (
            "extracted_features",
            "cache",
            "evidence",
            ".cache-staging-test",
            ".evidence-staging-test",
            ".generic-staging-test",
        )
        for relative in partial_paths:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticQualificationFixture(Path(temporary))
                self.declare(fixture)
                partial = fixture.workspace / relative
                partial.mkdir()
                sentinel = partial / "partial.bin"
                sentinel.write_bytes(b"partial")
                with self.assertRaises(FileExistsError):
                    qualification.consume_workspace(
                        target=TARGET,
                        run_root=fixture.run_root,
                        qualification_subject_sha256=fixture.subject,
                    )
                self.assertFalse(
                    (fixture.workspace / "consume_started.json").exists()
                )
                self.assertEqual(sentinel.read_bytes(), b"partial")

    def test_protocol_and_pair_mutations_fail_before_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)
            protocol_path = fixture.workspace / "protocol.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["cache_contract"]["pooling"] = "image_tokens"
            _write_json(protocol_path, protocol)
            with self.assertRaisesRegex(ValueError, "declaration identity"):
                qualification.consume_workspace(
                    target=TARGET,
                    run_root=fixture.run_root,
                    qualification_subject_sha256=fixture.subject,
                )
            self.assertFalse((fixture.workspace / "consume_started.json").exists())

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)
            frozen_pair = (
                fixture.workspace
                / "input_snapshot"
                / "run"
                / "training_v7"
                / TARGET
                / "dual_or"
                / fixture.pair_manifest_path.name
            )
            frozen_pair.write_bytes(frozen_pair.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "input snapshot changed"):
                qualification.consume_workspace(
                    target=TARGET,
                    run_root=fixture.run_root,
                    qualification_subject_sha256=fixture.subject,
                )
            self.assertFalse((fixture.workspace / "consume_started.json").exists())

    def test_feature_order_pooling_and_dimension_mutations_are_rejected(self) -> None:
        mutations = ("order", "pooling", "dimension")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticQualificationFixture(Path(temporary))
                self.declare(fixture)

                def mutated_extractor(**_kwargs):
                    feature_root = fixture.write_feature_bundle()
                    bundle_path = feature_root / "feature_bundle.npz"
                    if mutation == "order":
                        with np.load(bundle_path, allow_pickle=False) as data:
                            reversed_ids = np.asarray(data["sample_ids"])[::-1].copy()
                        _rewrite_npz(bundle_path, sample_ids=reversed_ids)
                    elif mutation == "pooling":
                        _rewrite_npz(
                            bundle_path,
                            pooling=np.asarray(["image_tokens"]),
                        )
                    else:
                        _rewrite_npz(
                            bundle_path,
                            text_embeddings=np.zeros((50, 2047), dtype=np.float64),
                        )
                    manifest_path = feature_root / "feature_manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest["feature_bundle_sha256"] = _digest(bundle_path)
                    _write_json(manifest_path, manifest)
                    return feature_root

                with self.assertRaisesRegex(
                    ValueError,
                    "IDs/order|layer/pooling/ID|dimensions/finiteness",
                ):
                    self.consume(fixture, extractor=mutated_extractor)
                self.assertTrue((fixture.workspace / "consume_started.json").is_file())
                self.assertFalse((fixture.workspace / "cache").exists())
                self.assertFalse((fixture.workspace / "evidence").exists())

    def test_materialized_cache_rejects_pooling_and_embedding_dimension_mutations(
        self,
    ) -> None:
        mutations = ("pooling", "embedding_dimension")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticQualificationFixture(Path(temporary))
                self.declare(fixture)
                feature_root = fixture.write_feature_bundle()
                protocol = json.loads(
                    (fixture.workspace / "protocol.json").read_text(encoding="utf-8")
                )
                text, image, _, _ = qualification._load_bundle(feature_root, protocol)
                cache_root = Path(temporary) / "cache"
                qualification._materialize_cache(
                    cache_root,
                    fixture.panels,
                    fixture.pair,
                    fixture.paths,
                    text,
                    image,
                )
                first = fixture.panels["regression"][0]
                cache_path = cache_root / "regression" / f"{first['sample_id']}.npz"
                if mutation == "pooling":
                    _rewrite_npz(cache_path, pooling=np.asarray(["image_tokens"]))
                else:
                    _rewrite_npz(
                        cache_path,
                        embedding=np.zeros(4095, dtype=np.float64),
                    )
                contract = qualification.dual_eval.cache_contract(fixture.pair)
                expectations = qualification.base.build_feature_cache_expectations(
                    panels=fixture.panels,
                    metadata_info=fixture.metadata,
                    metadata_paths={
                        "regression": fixture.paths.final_metadata,
                        "text_led": fixture.paths.text_led_metadata,
                        "external_benign": fixture.paths.external_benign_metadata,
                    },
                    metadata_root=fixture.run_root,
                    corpus_manifest_sha256=_digest(fixture.paths.manifest),
                )
                provider = qualification.base.ResumableEvaluationFeatureProvider(
                    delegate=qualification.dual_eval.ForbiddenFrozenDelegate(contract),
                    artifact=contract,
                    expectations=expectations,
                    cache_root=cache_root,
                    cache_only=True,
                )
                expected = expectations[first["sample_id"]]
                with self.assertRaises(ValueError):
                    provider._load(cache_path, expected)


class QualificationPrimitiveValidationTests(unittest.TestCase):
    @staticmethod
    def declare(fixture: SyntheticQualificationFixture) -> None:
        with patch.object(
            qualification,
            "_validated_inputs",
            return_value=fixture.validated_inputs(),
        ):
            qualification.declare_workspace(
                target=TARGET,
                run_root=fixture.run_root,
                model_cache_dir=fixture.model_cache,
                development_workspace=fixture.workspace,
            )

    def test_feature_bundle_rejects_order_pooling_dimension_and_config_mutations(
        self,
    ) -> None:
        mutations = (
            (
                "order",
                lambda root, bundle, manifest: _rewrite_npz(
                    bundle,
                    sample_ids=np.asarray(
                        [
                            row["sample_id"]
                            for row in qualification._csv(
                                root / "aligned_source_metadata.csv"
                            )[1]
                        ][::-1]
                    ),
                ),
                "IDs/order",
            ),
            (
                "pooling",
                lambda _root, bundle, _manifest: _rewrite_npz(
                    bundle, pooling=np.asarray(["image_tokens"])
                ),
                "layer/pooling/ID",
            ),
            (
                "dimension",
                lambda _root, bundle, _manifest: _rewrite_npz(
                    bundle,
                    text_embeddings=np.zeros((50, 2047), dtype=np.float64),
                ),
                "dimensions/finiteness",
            ),
            (
                "extractor config",
                lambda _root, _bundle, manifest: manifest["config"].__setitem__(
                    "max_pixels", None
                ),
                "extractor configuration",
            ),
        )
        for name, mutate, message in mutations:
            with self.subTest(mutation=name), tempfile.TemporaryDirectory() as temporary:
                fixture = SyntheticQualificationFixture(Path(temporary))
                self.declare(fixture)
                feature_root = fixture.write_feature_bundle()
                bundle_path = feature_root / "feature_bundle.npz"
                manifest_path = feature_root / "feature_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(feature_root, bundle_path, manifest)
                manifest["feature_bundle_sha256"] = _digest(bundle_path)
                _write_json(manifest_path, manifest)
                protocol = json.loads(
                    (fixture.workspace / "protocol.json").read_text(encoding="utf-8")
                )
                with self.assertRaisesRegex(ValueError, message):
                    qualification._load_bundle(feature_root, protocol)

    def test_feature_bundle_rejects_aligned_metadata_and_protocol_dimension_mutations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)
            feature_root = fixture.write_feature_bundle()
            aligned_path = feature_root / "aligned_source_metadata.csv"
            rows = qualification._csv(aligned_path)[1]
            _write_csv(aligned_path, list(reversed(rows)))
            manifest_path = feature_root / "feature_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["aligned_source_metadata_sha256"] = _digest(aligned_path)
            _write_json(manifest_path, manifest)
            protocol = json.loads(
                (fixture.workspace / "protocol.json").read_text(encoding="utf-8")
            )
            with self.assertRaisesRegex(ValueError, "metadata order"):
                qualification._load_bundle(feature_root, protocol)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticQualificationFixture(Path(temporary))
            self.declare(fixture)
            feature_root = fixture.write_feature_bundle()
            protocol = json.loads(
                (fixture.workspace / "protocol.json").read_text(encoding="utf-8")
            )
            protocol["feature_extraction_contract"]["component_dimension"] = 1024
            with self.assertRaisesRegex(ValueError, "identity/dimension"):
                qualification._load_bundle(feature_root, protocol)


class QualificationCliTests(unittest.TestCase):
    def test_cli_separates_declaration_runtime_inputs_from_subject_consume(self) -> None:
        parser = qualification._parser()
        declared = parser.parse_args(
            [
                "declare",
                TARGET,
                "--run-root",
                "synthetic-run",
                "--model-cache-dir",
                "synthetic-cache",
                "--runtime-image-id",
                "sha256:container",
                "--development-workspace",
                "synthetic-development",
            ]
        )
        self.assertEqual(declared.command, "declare")
        self.assertEqual(declared.model_cache_dir, Path("synthetic-cache"))
        self.assertEqual(declared.runtime_image_id, "sha256:container")
        self.assertFalse(hasattr(declared, "qualification_subject"))

        consumed = parser.parse_args(
            [
                "consume",
                TARGET,
                "--run-root",
                "synthetic-run",
                "--qualification-subject",
                "a" * 64,
            ]
        )
        self.assertEqual(consumed.command, "consume")
        self.assertEqual(consumed.qualification_subject, "a" * 64)
        for forbidden in (
            "model_cache_dir",
            "llava_runtime_model",
            "runtime_image_id",
            "workspace",
            "summary",
        ):
            self.assertFalse(hasattr(consumed, forbidden), forbidden)

    def test_consume_cli_requires_subject_and_rejects_runtime_overrides(self) -> None:
        parser = qualification._parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["consume", TARGET])
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "consume",
                    TARGET,
                    "--qualification-subject",
                    "a" * 64,
                    "--model-cache-dir",
                    "substitution-cache",
                ]
            )


if __name__ == "__main__":
    unittest.main()
