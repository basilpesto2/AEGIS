from __future__ import annotations

from dataclasses import dataclass, field
import csv
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from threading import Lock

import numpy as np

from AEGIS.bounded_cache import BoundedTTLCache
from AEGIS.guardrail import GuardrailRequest
from AEGIS.io import load_embeddings
from AEGIS.provenance import preprocessing_fingerprint


@dataclass
class Qwen25VLGuardrailProvider:
    """Embedding provider for the default Qwen2.5-VL AEGIS detector."""

    model_family: str = "qwen25_vl"
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_revision: str = ""
    tokenizer_revision: str = ""
    pooling: str = "text_tokens"
    feature_dim: int = 2048
    cache_dir: str | None = "models/huggingface"
    layer: int = -1
    torch_dtype: str = "auto"
    device_map: str = "auto"
    min_pixels: int | None = None
    max_pixels: int | None = 200704
    cache_requests: bool = True
    request_cache_max_entries: int = 128
    request_cache_ttl_seconds: float = 300.0
    environment_overrides: bool = True
    local_files_only: bool = False
    preprocessing_sha256: str = field(init=False)
    _feature_cache: BoundedTTLCache[np.ndarray] = field(init=False, repr=False)
    _inference_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.environment_overrides:
            self.model_id = os.getenv("AEGIS_QWEN_MODEL_ID", self.model_id)
            self.model_revision = os.getenv(
                "AEGIS_QWEN_MODEL_REVISION", self.model_revision
            )
            self.tokenizer_revision = os.getenv(
                "AEGIS_QWEN_TOKENIZER_REVISION", self.tokenizer_revision
            )
            cache_dir = os.getenv("AEGIS_QWEN_CACHE_DIR")
            if cache_dir is not None:
                self.cache_dir = cache_dir or None
            self.layer = int(os.getenv("AEGIS_QWEN_LAYER", str(self.layer)))
            self.pooling = os.getenv("AEGIS_QWEN_POOLING", self.pooling)
            self.max_pixels = _optional_env_int(
                "AEGIS_QWEN_MAX_PIXELS", self.max_pixels
            )
            self.request_cache_max_entries = int(
                os.getenv(
                    "AEGIS_REQUEST_CACHE_MAX_ENTRIES",
                    str(self.request_cache_max_entries),
                )
            )
            self.request_cache_ttl_seconds = float(
                os.getenv(
                    "AEGIS_REQUEST_CACHE_TTL_SECONDS",
                    str(self.request_cache_ttl_seconds),
                )
            )
        self._feature_cache = BoundedTTLCache(
            max_entries=self.request_cache_max_entries,
            ttl_seconds=self.request_cache_ttl_seconds,
        )
        self.preprocessing_sha256 = preprocessing_fingerprint(
            self.model_family,
            adapter_schema=1,
            add_generation_prompt=False,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            padding=True,
            chat_template=True,
        )

    def embed(self, request: GuardrailRequest) -> np.ndarray:
        if not request.text.strip() and not request.image_paths:
            raise ValueError("Qwen25VLGuardrailProvider requires text, an image, or both.")
        if len(request.image_paths) > 1:
            raise ValueError("The current Qwen2.5-VL adapter accepts one image per request.")

        cache_key = self._cache_key(request)
        cached = self._feature_cache.get(cache_key) if self.cache_requests else None
        if cached is not None:
            return cached.copy()

        with self._inference_lock:
            cached = self._feature_cache.get(cache_key) if self.cache_requests else None
            if cached is not None:
                return cached.copy()
            with tempfile.TemporaryDirectory(prefix="aegis-qwen25vl-provider-") as directory:
                scratch = Path(directory)
                image_path = self._stage_image(request, scratch)
                metadata_path = scratch / "request.csv"
                output_path = scratch / "embedding.npz"
                with metadata_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=["sample_id", "text", "image_path"]
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "sample_id": request.request_id or "request",
                            "text": request.text,
                            "image_path": image_path,
                        }
                    )

                self._extract(
                    metadata_path=metadata_path,
                    output_path=output_path,
                    scratch=scratch,
                )
                features, _ = load_embeddings(output_path)
                if features.shape[0] != 1:
                    raise ValueError(
                        f"Expected one embedded request, received {features.shape[0]}."
                    )
                vector = np.asarray(features[0], dtype=np.float32)
                if self.cache_requests:
                    self._feature_cache.put(cache_key, vector.copy())
                return vector

    def _stage_image(self, request: GuardrailRequest, scratch: Path) -> str:
        if not request.image_paths:
            return ""
        source = Path(request.image_paths[0])
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Image path does not exist or is not a file: {source}")
        suffix = source.suffix or ".image"
        target = scratch / f"request_image{suffix}"
        shutil.copy2(source, target)
        return target.name

    def _cache_key(self, request: GuardrailRequest) -> str:
        digest = hashlib.sha256()
        for value in (
            self.model_family,
            self.model_id,
            self.model_revision,
            self.tokenizer_revision,
            self.preprocessing_sha256,
            self.pooling,
            str(self.layer),
            str(self.min_pixels),
            str(self.max_pixels),
            request.text,
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        for image_path in request.image_paths:
            path = Path(image_path)
            digest.update(str(path.resolve()).encode("utf-8", errors="replace"))
            digest.update(b"\0")
            if path.exists():
                stat = path.stat()
                digest.update(str(stat.st_size).encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(stat.st_mtime_ns).encode("utf-8"))
                digest.update(b"\0")
        return digest.hexdigest()

    def _extract(self, metadata_path: Path, output_path: Path, scratch: Path) -> None:
        _quiet_transformers_progress()
        from AEGIS.adapters.qwen25_vl import (
            Qwen25VLExtractionConfig,
            extract_qwen25_vl_embeddings,
        )

        config = Qwen25VLExtractionConfig(
            model_id=self.model_id,
            model_revision=self.model_revision or None,
            tokenizer_revision=self.tokenizer_revision or None,
            cache_dir=self.cache_dir,
            layer=self.layer,
            pooling=self.pooling,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            local_files_only=self.local_files_only,
        )
        extract_qwen25_vl_embeddings(
            metadata_path=metadata_path,
            corpus_root=scratch,
            output_path=output_path,
            config=config,
        )


@dataclass
class LlavaOnevisionGuardrailProvider:
    """Embedding provider for LLaVA-OneVision AEGIS detectors."""

    model_family: str = "llava_onevision"
    model_id: str = "models\\huggingface\\llava-onevision-qwen2-0.5b-ov-hf"
    runtime_model_id: str | None = None
    model_revision: str = ""
    tokenizer_revision: str = ""
    pooling: str = "text_tokens"
    feature_dim: int = 896
    cache_dir: str | None = "models/huggingface"
    layer: int = -1
    torch_dtype: str = "auto"
    device_map: str = "auto"
    max_image_edge: int | None = 384
    cache_requests: bool = True
    request_cache_max_entries: int = 128
    request_cache_ttl_seconds: float = 300.0
    environment_overrides: bool = True
    local_files_only: bool = False
    preprocessing_sha256: str = field(init=False)
    _feature_cache: BoundedTTLCache[np.ndarray] = field(init=False, repr=False)
    _inference_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.environment_overrides:
            self.model_id = os.getenv("AEGIS_LLAVA_MODEL_ID", self.model_id)
            self.model_revision = os.getenv(
                "AEGIS_LLAVA_MODEL_REVISION", self.model_revision
            )
            self.tokenizer_revision = os.getenv(
                "AEGIS_LLAVA_TOKENIZER_REVISION", self.tokenizer_revision
            )
            cache_dir = os.getenv("AEGIS_LLAVA_CACHE_DIR")
            if cache_dir is not None:
                self.cache_dir = cache_dir or None
            self.layer = int(os.getenv("AEGIS_LLAVA_LAYER", str(self.layer)))
            self.pooling = os.getenv("AEGIS_LLAVA_POOLING", self.pooling)
            self.max_image_edge = _optional_env_int(
                "AEGIS_LLAVA_MAX_IMAGE_EDGE",
                self.max_image_edge,
            )
            self.request_cache_max_entries = int(
                os.getenv(
                    "AEGIS_REQUEST_CACHE_MAX_ENTRIES",
                    str(self.request_cache_max_entries),
                )
            )
            self.request_cache_ttl_seconds = float(
                os.getenv(
                    "AEGIS_REQUEST_CACHE_TTL_SECONDS",
                    str(self.request_cache_ttl_seconds),
                )
            )
        self._feature_cache = BoundedTTLCache(
            max_entries=self.request_cache_max_entries,
            ttl_seconds=self.request_cache_ttl_seconds,
        )
        self.preprocessing_sha256 = preprocessing_fingerprint(
            self.model_family,
            adapter_schema=1,
            add_generation_prompt=False,
            max_image_edge=self.max_image_edge,
            padding=True,
            chat_template=True,
            image_mode="RGB",
        )

    def embed(self, request: GuardrailRequest) -> np.ndarray:
        if not request.text.strip():
            raise ValueError("LlavaOnevisionGuardrailProvider requires text.")
        if len(request.image_paths) != 1:
            raise ValueError("LLaVA-OneVision provider requires exactly one image per request.")

        cache_key = self._cache_key(request)
        cached = self._feature_cache.get(cache_key) if self.cache_requests else None
        if cached is not None:
            return cached.copy()

        with self._inference_lock:
            cached = self._feature_cache.get(cache_key) if self.cache_requests else None
            if cached is not None:
                return cached.copy()
            with tempfile.TemporaryDirectory(prefix="aegis-llava-provider-") as directory:
                scratch = Path(directory)
                image_path = self._stage_image(request, scratch)
                metadata_path = scratch / "request.csv"
                with metadata_path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=["sample_id", "text", "image_path"]
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "sample_id": request.request_id or "request",
                            "text": request.text,
                            "image_path": image_path,
                        }
                    )

                output_path = self._extract(metadata_path=metadata_path, scratch=scratch)
                features, _ = load_embeddings(output_path)
                if features.shape[0] != 1:
                    raise ValueError(
                        f"Expected one embedded request, received {features.shape[0]}."
                    )
                vector = np.asarray(features[0], dtype=np.float32)
                if self.cache_requests:
                    self._feature_cache.put(cache_key, vector.copy())
                return vector

    def _stage_image(self, request: GuardrailRequest, scratch: Path) -> str:
        source = Path(request.image_paths[0])
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Image path does not exist or is not a file: {source}")
        suffix = source.suffix or ".image"
        target = scratch / f"request_image{suffix}"
        shutil.copy2(source, target)
        return target.name

    def _cache_key(self, request: GuardrailRequest) -> str:
        digest = hashlib.sha256()
        for value in (
            self.model_family,
            self.model_id,
            self.model_revision,
            self.tokenizer_revision,
            self.preprocessing_sha256,
            self.pooling,
            str(self.layer),
            str(self.max_image_edge),
            request.text,
        ):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        path = Path(request.image_paths[0])
        digest.update(str(path.resolve()).encode("utf-8", errors="replace"))
        digest.update(b"\0")
        if path.exists():
            stat = path.stat()
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _extract(self, metadata_path: Path, scratch: Path) -> Path:
        _quiet_transformers_progress()
        from AEGIS.adapters.llava_onevision import (
            LlavaOnevisionExtractionConfig,
            extract_llava_onevision_pooling_embeddings,
        )

        config = LlavaOnevisionExtractionConfig(
            model_id=self.model_id,
            runtime_model_id=self.runtime_model_id,
            model_revision=self.model_revision or None,
            tokenizer_revision=self.tokenizer_revision or None,
            cache_dir=self.cache_dir,
            layer=self.layer,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            max_image_edge=self.max_image_edge,
            local_files_only=self.local_files_only,
        )
        paths = extract_llava_onevision_pooling_embeddings(
            metadata_path=metadata_path,
            corpus_root=scratch,
            output_dir=scratch,
            poolings=[self.pooling],
            config=config,
            output_prefix="request",
        )
        return paths[self.pooling]


def _optional_env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    if not value:
        return None
    return int(value)


def _quiet_transformers_progress() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except ImportError:
        pass
