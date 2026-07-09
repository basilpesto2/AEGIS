from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="aegis-smoke-") as directory:
        root = Path(directory)
        metadata_path = root / "metadata.csv"
        embeddings_path = root / "embeddings.npz"
        scores_path = root / "scores.csv"
        detector_path = root / "detector.npz"
        provider_module_path = root / "smoke_provider.py"
        guard_request_path = root / "guard_request.json"
        decisions_path = root / "decisions.csv"
        decisions_summary_path = root / "decisions_summary.json"
        calibrated_policy_path = root / "calibrated_policy.json"
        calibration_summary_path = root / "calibration_summary.json"
        monitoring_summary_path = root / "monitoring_summary.json"
        provider_contract_path = root / "provider_contract.json"
        target_summary_path = root / "target_summary.json"
        target_scores_path = root / "target_scores.csv"
        target_manifest_path = root / "target_manifest.json"
        target_bundle_path = root / "target_bundle.json"
        target_registry_path = root / "target_registry.json"
        target_package_dir = root / "target_package"
        litmus_dir = root / "litmus"

        sample_ids = np.asarray([f"s{idx:02d}" for idx in range(12)])
        labels = np.asarray([0, 0, 1, 1] * 3)
        splits = np.asarray(["fit"] * 4 + ["val"] * 4 + ["test"] * 4)
        signal = np.where(labels == 1, 2.0, -2.0)
        features = np.column_stack(
            [
                signal,
                signal * 0.5 + np.linspace(-0.2, 0.2, len(labels)),
            ]
        )
        metadata = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "label": np.where(labels == 1, "malicious", "benign"),
                "experiment_split": splits,
            }
        )
        metadata.to_csv(metadata_path, index=False)
        np.savez(
            embeddings_path,
            embeddings=features,
            sample_id=sample_ids,
            model_id=np.asarray(["smoke/model"]),
            layer=np.asarray([-1]),
            pooling=np.asarray(["text_tokens"]),
        )
        provider_module_path.write_text(
            "\n".join(
                [
                    "import numpy as np",
                    "",
                    "class SmokeProvider:",
                    "    model_family = 'generic'",
                    "    model_id = 'smoke/model'",
                    "    pooling = 'text_tokens'",
                    "    feature_dim = 2",
                    "",
                    "    def embed(self, request):",
                    "        value = 4.0 if 'malicious' in request.text else -4.0",
                    "        return np.asarray([value, value * 0.5])",
                    "",
                    "provider = SmokeProvider()",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        smoke_env = os.environ.copy()
        smoke_env["PYTHONPATH"] = (
            str(root)
            if not smoke_env.get("PYTHONPATH")
            else str(root) + os.pathsep + smoke_env["PYTHONPATH"]
        )

        commands = [
            [
                sys.executable,
                "-m",
                "AEGIS.cli",
                "score",
                "--embeddings",
                str(embeddings_path),
                "--metadata",
                str(metadata_path),
                "--split-column",
                "experiment_split",
                "--output",
                str(scores_path),
                "--components",
                "1",
                "--malicious-prior",
                "0.5",
            ],
            [
                sys.executable,
                "-m",
                "AEGIS.cli",
                "train-detector",
                "--embeddings",
                str(embeddings_path),
                "--metadata",
                str(metadata_path),
                "--output",
                str(detector_path),
                "--threshold-strategy",
                "f1",
                "--epochs",
                "80",
                "--learning-rate",
                "0.1",
                "--l2",
                "0.0",
                "--source",
                "smoke",
            ],
            [
                sys.executable,
                "-m",
                "AEGIS.cli",
                "inspect-artifact",
                "--detector",
                str(detector_path),
            ],
            [
                sys.executable,
                "-m",
                "AEGIS.cli",
                "doctor",
                "--detector",
                str(detector_path),
                "--cache-dir",
                str(root),
                "--require-cache-dir",
            ],
            [
                sys.executable,
                "-m",
                "AEGIS.cli",
                "guard-request",
                "--provider",
                "smoke_provider:provider",
                "--detector",
                str(detector_path),
                "--text",
                "malicious smoke request",
                "--request-id",
                "smoke-request",
                "--output",
                str(guard_request_path),
                "--require-matching-provenance",
            ],
            [
                sys.executable,
                "scripts/validate_embedding_provider_contract.py",
                "--provider",
                "smoke_provider:provider",
                "--detector",
                str(detector_path),
                "--output",
                str(provider_contract_path),
                "--text",
                "benign smoke contract",
                "--request-id",
                "smoke-contract",
                "--required-modality",
                "text",
            ],
            [
                sys.executable,
                "-m",
                "AEGIS.cli",
                "guard-features",
                "--features",
                str(embeddings_path),
                "--metadata",
                str(metadata_path),
                "--detector",
                str(detector_path),
                "--output",
                str(decisions_path),
                "--summary-output",
                str(decisions_summary_path),
                "--require-matching-provenance",
                "--include-metadata-column",
                "label",
            ],
            [
                sys.executable,
                "scripts/evaluate_detector_features.py",
                "--features",
                str(embeddings_path),
                "--metadata",
                str(metadata_path),
                "--detector",
                str(detector_path),
                "--split-column",
                "experiment_split",
                "--split",
                "test",
                "--source",
                "smoke_test",
                "--output-summary",
                str(target_summary_path),
                "--output-scores",
                str(target_scores_path),
            ],
            [
                sys.executable,
                "scripts/calibrate_guardrail_policy.py",
                "--features",
                str(embeddings_path),
                "--metadata",
                str(metadata_path),
                "--detector",
                str(detector_path),
                "--output-policy",
                str(calibrated_policy_path),
                "--output-summary",
                str(calibration_summary_path),
                "--max-fpr",
                "0.01",
                "--min-recall",
                "0.9",
            ],
            [
                sys.executable,
                "scripts/monitor_guardrail_decisions.py",
                "--decisions",
                str(decisions_path),
                "--output",
                str(monitoring_summary_path),
                "--label-column",
                "label",
            ],
            [
                sys.executable,
                "scripts/build_classify_litmus_panel.py",
                "--output-dir",
                str(litmus_dir),
                "--reference-metadata",
                str(metadata_path),
            ],
        ]

        for command in commands:
            subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                env=smoke_env,
            )

        _write_target_release_fixture(
            detector_path=detector_path,
            policy_path=calibrated_policy_path,
            provider_contract_path=provider_contract_path,
            calibration_summary_path=calibration_summary_path,
            monitoring_summary_path=monitoring_summary_path,
            summary_path=target_summary_path,
            manifest_path=target_manifest_path,
            bundle_path=target_bundle_path,
            registry_path=target_registry_path,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/validate_target_release.py",
                "--bundle",
                str(target_bundle_path),
                "--require-labeled-monitoring",
                "--required-provider-modality",
                "text",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            env=smoke_env,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/validate_target_registry.py",
                "--registry",
                str(target_registry_path),
                "--required-model-family",
                "generic",
                "--required-target-modality",
                "text",
                "--require-labeled-monitoring",
                "--required-provider-modality",
                "text",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            env=smoke_env,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/build_target_deployment_package.py",
                "--bundle",
                str(target_bundle_path),
                "--output-dir",
                str(target_package_dir),
                "--require-labeled-monitoring",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            env=smoke_env,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/validate_target_deployment_package.py",
                "--package-dir",
                str(target_package_dir),
                "--require-labeled-monitoring",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            env=smoke_env,
        )

        litmus_metadata = pd.read_csv(litmus_dir / "metadata.csv")
        guard_request = json.loads(guard_request_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "ok": True,
                    "scores_csv": scores_path.exists(),
                    "detector_npz": detector_path.exists(),
                    "guard_request_json": guard_request_path.exists(),
                    "guard_request_action": guard_request["decisions"][0]["action"],
                    "provider_contract_json": provider_contract_path.exists(),
                    "target_release_bundle_json": target_bundle_path.exists(),
                    "target_registry_json": target_registry_path.exists(),
                    "target_package_dir": target_package_dir.exists(),
                    "target_summary_json": target_summary_path.exists(),
                    "decisions_csv": decisions_path.exists(),
                    "decisions_summary_json": decisions_summary_path.exists(),
                    "calibrated_policy_json": calibrated_policy_path.exists(),
                    "calibration_summary_json": calibration_summary_path.exists(),
                    "monitoring_summary_json": monitoring_summary_path.exists(),
                    "litmus_rows": int(len(litmus_metadata)),
                    "litmus_label_counts": litmus_metadata["label"].value_counts().to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )


def _write_target_release_fixture(
    *,
    detector_path: Path,
    policy_path: Path,
    provider_contract_path: Path,
    calibration_summary_path: Path,
    monitoring_summary_path: Path,
    summary_path: Path,
    manifest_path: Path,
    bundle_path: Path,
    registry_path: Path,
) -> None:
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "artifacts": [
                {"path": str(detector_path)},
                {"path": str(policy_path)},
                {"path": str(provider_contract_path)},
                {"path": str(calibration_summary_path)},
                {"path": str(monitoring_summary_path)},
                {"path": str(summary_path)},
                {"path": str(bundle_path)},
            ],
        },
    )
    _write_json(
        bundle_path,
        {
            "target_name": "smoke-target",
            "model_family": "generic",
            "model_id": "smoke/model",
            "scope": "smoke",
            "status": "production_candidate",
            "intended_modalities": ["text"],
            "detector_path": str(detector_path),
            "policy_path": str(policy_path),
            "summary_paths": [str(summary_path)],
            "provider_contract_path": str(provider_contract_path),
            "calibration_summary_path": str(calibration_summary_path),
            "monitoring_summary_path": str(monitoring_summary_path),
            "manifest_path": str(manifest_path),
            "gate_report": {
                "ok": True,
                "checks": [{"name": "smoke", "ok": True, "detail": "ok"}],
            },
            "notes": "smoke",
        },
    )
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["bundle_path"] = str(bundle_path)
    _write_json(
        registry_path,
        {
            "schema_version": 1,
            "status_counts": {"production_candidate": 1},
            "targets": [bundle],
        },
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
