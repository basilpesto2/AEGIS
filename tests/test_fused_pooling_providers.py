from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

import numpy as np

from AEGIS.guardrail import GuardrailRequest
from AEGIS.providers import (
    LlavaOnevisionGuardrailProvider,
    Qwen25VLGuardrailProvider,
    _load_extracted_features,
)


class FusedPoolingProviderTests(unittest.TestCase):
    def test_fused_embedding_is_text_then_image_and_cached_for_both_providers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "request.png"
            image_path.write_bytes(b"test-image")
            request = GuardrailRequest(
                text="inspect this prompt",
                image_paths=(str(image_path),),
                request_id="request-1",
            )
            paths = {
                "image_tokens": Path(directory) / "image.npz",
                "text_tokens": Path(directory) / "text.npz",
            }
            arrays = {
                paths["text_tokens"]: np.asarray([[1.0, 2.0]]),
                paths["image_tokens"]: np.asarray([[3.0, 4.0]]),
            }

            for provider_type in (
                Qwen25VLGuardrailProvider,
                LlavaOnevisionGuardrailProvider,
            ):
                with self.subTest(provider=provider_type.__name__):
                    provider = provider_type(
                        pooling="text_image_tokens",
                        feature_dim=4,
                        environment_overrides=False,
                    )
                    with (
                        patch.object(provider, "_extract", return_value=paths) as extract,
                        patch(
                            "AEGIS.providers.load_embeddings",
                            side_effect=lambda path: (arrays[path].copy(), None),
                        ) as load,
                    ):
                        first = provider.embed(request)
                        first[0] = 99.0
                        second = provider.embed(request)

                    np.testing.assert_array_equal(
                        second,
                        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                    )
                    self.assertEqual(second.shape, (provider.feature_dim,))
                    extract.assert_called_once()
                    self.assertEqual(
                        load.call_args_list,
                        [call(paths["text_tokens"]), call(paths["image_tokens"])],
                    )

    def test_fused_embedding_rejects_component_row_count_mismatch(self) -> None:
        text_path = Path("text.npz")
        image_path = Path("image.npz")
        arrays = {
            text_path: np.zeros((1, 2), dtype=np.float32),
            image_path: np.zeros((2, 2), dtype=np.float32),
        }
        with patch(
            "AEGIS.providers.load_embeddings",
            side_effect=lambda path: (arrays[path], None),
        ):
            with self.assertRaisesRegex(ValueError, "row-count mismatch"):
                _load_extracted_features(
                    {"text_tokens": text_path, "image_tokens": image_path},
                    "text_image_tokens",
                )

    def test_qwen_fused_extraction_uses_one_multi_pooling_adapter_call(self) -> None:
        provider = Qwen25VLGuardrailProvider(
            pooling="text_image_tokens",
            feature_dim=4096,
            environment_overrides=False,
        )
        paths = {
            "text_tokens": Path("text.npz"),
            "image_tokens": Path("image.npz"),
        }
        with (
            patch(
                "AEGIS.adapters.qwen25_vl.extract_qwen25_vl_pooling_embeddings",
                return_value=paths,
            ) as multi_extract,
            patch(
                "AEGIS.adapters.qwen25_vl.extract_qwen25_vl_embeddings"
            ) as single_extract,
        ):
            result = provider._extract(
                metadata_path=Path("request.csv"),
                output_path=Path("embedding.npz"),
                scratch=Path("scratch"),
            )

        self.assertIs(result, paths)
        single_extract.assert_not_called()
        multi_extract.assert_called_once()
        kwargs = multi_extract.call_args.kwargs
        self.assertEqual(kwargs["poolings"], ("text_tokens", "image_tokens"))
        self.assertEqual(kwargs["config"].pooling, "text_tokens")

    def test_llava_fused_extraction_uses_one_multi_pooling_adapter_call(self) -> None:
        provider = LlavaOnevisionGuardrailProvider(
            pooling="text_image_tokens",
            feature_dim=1792,
            environment_overrides=False,
        )
        paths = {
            "text_tokens": Path("text.npz"),
            "image_tokens": Path("image.npz"),
        }
        with patch(
            "AEGIS.adapters.llava_onevision.extract_llava_onevision_pooling_embeddings",
            return_value=paths,
        ) as multi_extract:
            result = provider._extract(
                metadata_path=Path("request.csv"),
                scratch=Path("scratch"),
            )

        self.assertIs(result, paths)
        multi_extract.assert_called_once()
        self.assertEqual(
            multi_extract.call_args.kwargs["poolings"],
            ("text_tokens", "image_tokens"),
        )

    def test_existing_single_pooling_embedding_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "request.png"
            image_path.write_bytes(b"test-image")
            request = GuardrailRequest(
                text="inspect this prompt",
                image_paths=(str(image_path),),
            )
            output_path = Path(directory) / "embedding.npz"

            for provider_type in (
                Qwen25VLGuardrailProvider,
                LlavaOnevisionGuardrailProvider,
            ):
                with self.subTest(provider=provider_type.__name__):
                    provider = provider_type(
                        pooling="text_tokens",
                        feature_dim=2,
                        cache_requests=False,
                        environment_overrides=False,
                    )
                    with (
                        patch.object(provider, "_extract", return_value=output_path),
                        patch(
                            "AEGIS.providers.load_embeddings",
                            return_value=(np.asarray([[5.0, 6.0]]), None),
                        ) as load,
                    ):
                        vector = provider.embed(request)

                    np.testing.assert_array_equal(
                        vector,
                        np.asarray([5.0, 6.0], dtype=np.float32),
                    )
                    load.assert_called_once_with(output_path)


if __name__ == "__main__":
    unittest.main()
