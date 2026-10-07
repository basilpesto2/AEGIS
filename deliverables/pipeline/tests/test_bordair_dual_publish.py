from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
PIPELINE = ROOT / "deliverables" / "pipeline"
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PIPELINE / "scripts"))

from aegis_research.bordair_dual import load_detector_pair
from aegis_research.bordair_dual_publish import (
    PublicationValidationError,
    PublicationQuarantineError,
    _candidate_runtime_preflight,
    package_files,
    publish_pair_package,
)
from aegis_research.bordair_dual import runtime_detector_identity_sha256
from aegis_research.bordair_paths import RunPaths
from AEGIS.target_profiles import TargetDetectorHead, get_target_profile
from publish_bordair_dual_pair import (
    publish_qualified_dual_pair as publish_qualified_dual_pair_entrypoint,
)


class DualPublishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pair = load_detector_pair(
            ROOT / "models" / "aegis" / "llava05b_v7_dual_or" / "detector_pair_manifest.json",
            expected_target="llava05b",
        )

    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="aegis-publish-test-"))
        self.destination = self.temp / "published"
        self.calls = 0

    def tearDown(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    def validate(self, manifest: Path, image: Path, text: Path) -> dict[str, object]:
        self.calls += 1
        loaded = load_detector_pair(manifest, expected_target="llava05b", image_override=image, text_override=text)
        return {"passed": True, "pair_identity_sha256": loaded.manifest["pair_identity_sha256"]}

    def publish(self, **kwargs: object) -> dict[str, object]:
        return publish_pair_package(
            pair=self.pair,
            destination=self.destination,
            expected_target="llava05b",
            post_validate=self.validate,
            **kwargs,
        )

    def test_publishes_complete_qualification_evidence_tree(self) -> None:
        result = self.publish()
        self.assertTrue(result["published"])
        expected = set(package_files(self.pair))
        actual = {path.relative_to(self.destination) for path in self.destination.rglob("*") if path.is_file()}
        self.assertEqual(expected, actual)
        self.assertTrue(any(str(path).startswith("qualification_evidence") for path in expected))

    def test_exact_destination_is_idempotent(self) -> None:
        self.publish()
        result = self.publish()
        self.assertTrue(result["idempotent"])
        self.assertEqual(self.calls, 2)

    def test_partial_destination_is_rejected_without_overwrite(self) -> None:
        self.destination.mkdir()
        marker = self.destination / "partial.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_destination_mutation_is_rejected(self) -> None:
        self.publish()
        artifact = next(self.destination.glob("*.npz"))
        artifact.write_bytes(artifact.read_bytes() + b"mutation")
        with self.assertRaises(FileExistsError):
            self.publish()

    def test_wrong_target_is_rejected_before_rename(self) -> None:
        with self.assertRaises(ValueError):
            publish_pair_package(
                pair=self.pair,
                destination=self.destination,
                expected_target="qwen25vl3b",
                post_validate=self.validate,
            )
        self.assertFalse(self.destination.exists())

    def test_wrong_target_is_rejected_for_existing_exact_destination(self) -> None:
        self.publish()
        before = {
            path.relative_to(self.destination): path.read_bytes()
            for path in self.destination.rglob("*")
            if path.is_file()
        }
        with self.assertRaisesRegex(ValueError, "target differs"):
            publish_pair_package(
                pair=self.pair,
                destination=self.destination,
                expected_target="qwen25vl3b",
                post_validate=self.validate,
            )
        after = {
            path.relative_to(self.destination): path.read_bytes()
            for path in self.destination.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)
        self.assertFalse((self.temp / ".published.publish.lock").exists())

    def test_crash_before_rename_leaves_no_partial_destination(self) -> None:
        def crash(stage: str) -> None:
            if stage == "before_rename":
                raise RuntimeError("injected crash")

        with self.assertRaises(RuntimeError):
            self.publish(failure_hook=crash)
        self.assertFalse(self.destination.exists())
        self.assertFalse(any(self.temp.glob(".published.stage-*")))
        self.assertFalse((self.temp / ".published.publish.lock").exists())

    def test_failure_after_rename_quarantines_new_destination(self) -> None:
        def crash(stage: str) -> None:
            if stage == "after_rename":
                raise RuntimeError("injected crash")

        with self.assertRaises(PublicationValidationError) as caught:
            self.publish(failure_hook=crash)
        self.assertFalse(self.destination.exists())
        self.assertEqual(caught.exception.state, "published_but_quarantined")
        self.assertTrue(caught.exception.quarantine.is_dir())

    def test_post_validation_failure_quarantines_new_destination(self) -> None:
        def fail(*_args: Path) -> dict[str, object]:
            raise ValueError("not promotable")

        with self.assertRaises(PublicationValidationError) as caught:
            publish_pair_package(
                pair=self.pair,
                destination=self.destination,
                expected_target="llava05b",
                post_validate=fail,
            )
        self.assertFalse(self.destination.exists())
        self.assertTrue(caught.exception.quarantine.is_dir())

    def test_false_post_validation_result_is_not_success(self) -> None:
        with self.assertRaises(PublicationValidationError) as caught:
            publish_pair_package(
                pair=self.pair,
                destination=self.destination,
                expected_target="llava05b",
                post_validate=lambda *_args: {"passed": False},
            )
        self.assertIn("did not pass", str(caught.exception))
        self.assertFalse(self.destination.exists())

    def test_false_validator_does_not_mutate_existing_exact_destination(self) -> None:
        self.publish()
        before = {
            path.relative_to(self.destination): path.read_bytes()
            for path in self.destination.rglob("*")
            if path.is_file()
        }
        with self.assertRaisesRegex(ValueError, "did not pass"):
            publish_pair_package(
                pair=self.pair,
                destination=self.destination,
                expected_target="llava05b",
                post_validate=lambda *_args: {"passed": False},
            )
        after = {
            path.relative_to(self.destination): path.read_bytes()
            for path in self.destination.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)
        self.assertFalse(any(self.temp.glob("*.published_but_quarantined-*")))

    def test_quarantine_rename_failure_retains_canonical_and_lock(self) -> None:
        real_rename = os.rename
        calls = 0

        def rename(source: object, destination: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected quarantine rename failure")
            real_rename(source, destination)

        with patch(
            "aegis_research.bordair_dual_publish.os.rename", side_effect=rename
        ):
            with self.assertRaises(PublicationQuarantineError) as caught:
                publish_pair_package(
                    pair=self.pair,
                    destination=self.destination,
                    expected_target="llava05b",
                    post_validate=lambda *_args: {"passed": False},
                )
        self.assertEqual(
            caught.exception.state,
            "published_validation_failed_quarantine_failed",
        )
        self.assertEqual(caught.exception.canonical, self.destination)
        self.assertTrue(self.destination.is_dir())
        self.assertTrue((self.temp / ".published.publish.lock").is_file())

    def test_extra_empty_directory_is_rejected(self) -> None:
        self.publish()
        (self.destination / "unexpected-empty").mkdir()
        with self.assertRaises(FileExistsError):
            self.publish()

    def test_hardlinked_destination_file_is_rejected(self) -> None:
        self.publish()
        manifest = self.destination / "detector_pair_manifest.json"
        alias = self.temp / "manifest-hardlink.json"
        try:
            alias.hardlink_to(manifest)
        except OSError as error:
            self.skipTest(f"hard links unavailable: {error}")
        with self.assertRaises(FileExistsError):
            self.publish()

    def test_nested_symlink_is_rejected(self) -> None:
        self.publish()
        link = self.destination / "qualification_evidence" / "unexpected-link"
        try:
            link.symlink_to(self.temp, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"directory symlinks unavailable: {error}")
        with self.assertRaises(FileExistsError):
            self.publish()

    def _qwen_candidate(self) -> tuple[object, Path, Path, object]:
        destination = self.temp / "qwen25vl3b_v7_dual_or"
        image_hash = "1" * 64
        text_hash = "2" * 64
        reviews = {"image": 0.31, "text": 0.72}
        identity = runtime_detector_identity_sha256({
            name: {"sha256": digest, "review_threshold": reviews[name]}
            for name, digest in (("image", image_hash), ("text", text_hash))
        })
        shared = {
            "model_family": "qwen25_vl",
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "model_revision": "a" * 40,
            "tokenizer_revision": "a" * 40,
            "layer": -1,
            "base_feature_dim": 2048,
        }
        artifacts = {
            name: {
                "path": f"qwen25vl3b_{name}_head_v1.npz",
                "sha256": digest,
                "review_threshold": reviews[name],
            }
            for name, digest in (("image", image_hash), ("text", text_hash))
        }
        pair = SimpleNamespace(manifest={
            "target": "qwen25vl3b",
            "shared_provenance": shared,
            "artifacts": artifacts,
            "runtime_detector_identity_sha256": identity,
        })
        template = json.loads(
            (ROOT / "configs" / "aegis.llava.deployment.container.json").read_text(
                encoding="utf-8"
            )
        )
        options = dict(template["provider_options"])
        options.update({
            "model_id": shared["model_id"],
            "model_revision": shared["model_revision"],
            "tokenizer_revision": shared["tokenizer_revision"],
            "layer": -1,
            "pooling": "text_image_tokens",
            "feature_dim": 4096,
        })
        heads = tuple(
            TargetDetectorHead(
                name=name,
                detector=str(destination / artifacts[name]["path"]),
                detector_sha256=artifacts[name]["sha256"],
                review_threshold=reviews[name],
            )
            for name in ("text", "image")
        )
        profile = replace(
            get_target_profile("llava05b"),
            name="qwen25vl3b",
            model_family="qwen25_vl",
            provider="test:qwen",
            provider_options=options,
            detector_heads=heads,
        )
        config = self.temp / "qwen-candidate.json"
        template.update({
            "schema_version": 2,
            "base_dir": ".",
            "target_profile": "qwen25vl3b",
            "provider": "test:qwen",
            "provider_options": options,
            "detector": {"mode": "or", "heads": [
                {"name": name, "artifact": str(destination / artifacts[name]["path"]), "review_threshold": reviews[name]}
                for name in ("text", "image")
            ]},
        })
        template["policy"].update(
            {"block_threshold": None, "review_threshold": None, "review_margin": None}
        )
        config.write_text(json.dumps(template), encoding="utf-8")
        return pair, destination, config, profile

    def test_qwen_shaped_preflight_accepts_absent_destination(self) -> None:
        pair, destination, config, profile = self._qwen_candidate()
        self.assertFalse(destination.exists())
        with patch.dict("AEGIS.target_profiles._TARGET_PROFILES", {"qwen25vl3b": profile}):
            _candidate_runtime_preflight(config, pair, destination)
        self.assertFalse(destination.exists())

    def test_qwen_shaped_preflight_rejects_wrong_lexical_path_and_hash(self) -> None:
        pair, destination, config, profile = self._qwen_candidate()
        wrong_path = replace(
            profile,
            detector_heads=(
                replace(profile.detector_heads[0], detector=str(self.temp / "wrong.npz")),
                profile.detector_heads[1],
            ),
        )
        with patch.dict("AEGIS.target_profiles._TARGET_PROFILES", {"qwen25vl3b": wrong_path}):
            with self.assertRaisesRegex(ValueError, "path/hash differs"):
                _candidate_runtime_preflight(config, pair, destination)
        wrong_hash = replace(
            profile,
            detector_heads=(
                replace(profile.detector_heads[0], detector_sha256="3" * 64),
                profile.detector_heads[1],
            ),
        )
        with patch.dict("AEGIS.target_profiles._TARGET_PROFILES", {"qwen25vl3b": wrong_hash}):
            with self.assertRaisesRegex(ValueError, "path/hash differs"):
                _candidate_runtime_preflight(config, pair, destination)

    def test_real_config_parser_rejects_missing_key_and_schema_before_destination(self) -> None:
        pair, destination, config, profile = self._qwen_candidate()
        original = json.loads(config.read_text(encoding="utf-8"))
        for mutation, message in (
            ({key: value for key, value in original.items() if key != "server"}, "missing required keys"),
            ({**original, "schema_version": 1}, "deployment path|schema_version"),
        ):
            config.write_text(json.dumps(mutation), encoding="utf-8")
            with patch.dict("AEGIS.target_profiles._TARGET_PROFILES", {"qwen25vl3b": profile}):
                with self.assertRaisesRegex(ValueError, message):
                    _candidate_runtime_preflight(config, pair, destination)
            self.assertFalse(destination.exists())

    def test_publisher_cli_help_runs_from_repository_and_external_cwd(self) -> None:
        script = (
            ROOT
            / "deliverables"
            / "pipeline"
            / "scripts"
            / "publish_bordair_dual_pair.py"
        )
        for cwd in (ROOT, self.temp):
            completed = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=cwd,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("--destination", completed.stdout)
            self.assertIn("--runtime-config", completed.stdout)
            self.assertFalse(self.destination.exists())

    def test_publisher_entrypoint_validates_against_corpus_manifest(self) -> None:
        paths = RunPaths(self.temp / "run")
        destination = ROOT / "models" / "aegis" / "unused-publisher-test"
        runtime_config = self.temp / "runtime.json"
        pair = SimpleNamespace(manifest={"target": "qwen25vl3b"})
        with (
            patch(
                "publish_bordair_dual_pair.validate_dual_source_candidate",
                return_value=({}, pair, {}),
            ) as source_validator,
            patch("publish_bordair_dual_pair._candidate_runtime_preflight"),
            patch(
                "publish_bordair_dual_pair.validate_dual_promotion_candidate",
                return_value={"passed": True},
            ) as promotion_validator,
            patch(
                "publish_bordair_dual_pair.publish_pair_package",
                return_value={"passed": True},
            ) as publisher,
        ):
            result = publish_qualified_dual_pair_entrypoint(
                target="qwen25vl3b",
                run_paths=paths,
                destination=destination,
                runtime_config=runtime_config,
            )
            post_validate = publisher.call_args.kwargs["post_validate"]
            post_validate(
                self.temp / "pair.json",
                self.temp / "image.npz",
                self.temp / "text.npz",
            )

        self.assertTrue(result["passed"])
        self.assertEqual(
            source_validator.call_args.kwargs["manifest_path"], paths.manifest
        )
        self.assertEqual(
            promotion_validator.call_args.kwargs["manifest_path"], paths.manifest
        )

    def test_publisher_module_imports_from_external_cwd_without_side_effects(self) -> None:
        pipeline = ROOT / "deliverables" / "pipeline"
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(pipeline)!r}); "
            "import aegis_research.bordair_dual_publish as module; "
            "assert callable(module.publish_qualified_dual_pair)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=self.temp,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(list(self.temp.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
