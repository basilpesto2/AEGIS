from __future__ import annotations

from pathlib import Path
import unittest

from deliverables.ablation_report import run_ablation as ablation


class AblationReproductionInstructionTests(unittest.TestCase):
    def test_llava_reproduction_block_is_unchanged(self) -> None:
        lines = ablation._reproduction_lines(
            runtime_target="llava05b",
            metadata=Path(
                "deliverables/runs/current_llava/features/"
                "aligned_source_metadata.csv"
            ),
            features=Path(
                "deliverables/runs/current_llava/features/feature_bundle.npz"
            ),
            output_dir=Path("deliverables/runs/current_llava/ablation"),
        )

        self.assertEqual(
            lines,
            [
                "```powershell",
                '$run = "deliverables/runs/current_llava"',
                "python deliverables/ablation_report/run_ablation.py `",
                '  --metadata "$run/features/aligned_source_metadata.csv" `',
                '  --features "$run/features/feature_bundle.npz" `',
                "  --runtime-target llava05b `",
                '  --output-dir "$run/ablation"',
                "```",
            ],
        )

    def test_qwen_reproduction_block_uses_qwen_target_and_run(self) -> None:
        lines = ablation._reproduction_lines(
            runtime_target="qwen25vl3b",
            metadata=Path(
                "deliverables/runs/current_qwen/features/"
                "aligned_source_metadata.csv"
            ),
            features=Path(
                "deliverables/runs/current_qwen/features/feature_bundle.npz"
            ),
            output_dir=Path("deliverables/runs/current_qwen/ablation"),
        )

        self.assertEqual(lines[1], '$run = "deliverables/runs/current_qwen"')
        self.assertIn("  --runtime-target qwen25vl3b `", lines)
        self.assertIn(
            '  --metadata "$run/features/aligned_source_metadata.csv" `', lines
        )
        self.assertIn(
            '  --features "$run/features/feature_bundle.npz" `', lines
        )
        self.assertIn('  --output-dir "$run/ablation"', lines)
        self.assertNotIn("current_llava", "\n".join(lines))

    def test_external_inputs_render_as_portable_literal_paths(self) -> None:
        metadata = Path("deliverables/benchmark/data/benchmark.csv")
        features = Path("deliverables/external/features.npz")
        lines = ablation._reproduction_lines(
            runtime_target="qwen25vl3b",
            metadata=metadata,
            features=features,
            output_dir=Path("deliverables/runs/current_qwen/ablation"),
        )

        self.assertIn(
            f'  --metadata "{ablation._portable_path(metadata)}" `', lines
        )
        self.assertIn(
            f'  --features "{ablation._portable_path(features)}" `', lines
        )

    def test_invalid_target_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            ablation._reproduction_lines(
                runtime_target="qwen",
                metadata=Path("metadata.csv"),
                features=Path("features.npz"),
                output_dir=Path("deliverables/runs/current_qwen/ablation"),
            )

    def test_empty_run_or_input_path_fails_closed(self) -> None:
        valid = {
            "metadata": Path("metadata.csv"),
            "features": Path("features.npz"),
            "output_dir": Path("deliverables/runs/current_qwen/ablation"),
        }
        for name in valid:
            with self.subTest(path=name):
                arguments = dict(valid)
                arguments[name] = Path(".")
                with self.assertRaises(ValueError):
                    ablation._reproduction_lines(
                        runtime_target="qwen25vl3b", **arguments
                    )

    def test_unsafe_powershell_path_characters_fail_closed(self) -> None:
        unsafe = ("\r", "\n", "`", '"', "$")
        valid = {
            "metadata": Path("deliverables/runs/current_qwen/metadata.csv"),
            "features": Path("deliverables/runs/current_qwen/features.npz"),
            "output_dir": Path("deliverables/runs/current_qwen/ablation"),
        }
        for name in valid:
            for character in unsafe:
                with self.subTest(path=name, character=repr(character)):
                    arguments = dict(valid)
                    if name == "output_dir":
                        arguments[name] = (
                            Path("deliverables/runs")
                            / f"unsafe{character}run"
                            / "ablation"
                        )
                    else:
                        arguments[name] = (
                            Path("deliverables/runs/current_qwen")
                            / f"unsafe{character}{name}"
                        )
                    with self.assertRaises(ValueError):
                        ablation._reproduction_lines(
                            runtime_target="qwen25vl3b", **arguments
                        )


if __name__ == "__main__":
    unittest.main()
