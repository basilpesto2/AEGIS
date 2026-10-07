from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from AEGIS.provenance import is_sha256
from AEGIS.target_assets import resolve_target_asset
from AEGIS.detector_set import detector_set_identity_sha256


@dataclass(frozen=True)
class TargetModelSource:
    repo_id: str
    revision: str
    storage_mode: str
    local_directory: str | None = None
    expected_content_sha256: str | None = None
    allow_patterns: tuple[str, ...] = (
        "*.json",
        "**/*.json",
        "*.txt",
        "*.model",
        "*.tiktoken",
        "*.safetensors",
        "pytorch_model*.bin",
        "*.index.json",
    )

    def __post_init__(self) -> None:
        if self.storage_mode not in {"huggingface_cache", "local_directory"}:
            raise ValueError("Unsupported target model storage mode.")
        if self.storage_mode == "local_directory" and not self.local_directory:
            raise ValueError("local_directory storage requires a destination path.")
        if self.expected_content_sha256 is not None and not is_sha256(
            self.expected_content_sha256
        ):
            raise ValueError("expected_content_sha256 must be a SHA-256 hex digest.")


@dataclass(frozen=True)
class TargetDetectorHead:
    name: str
    detector: str
    detector_sha256: str
    review_threshold: float

    def __post_init__(self) -> None:
        if not isinstance(self.detector, str) or not self.detector.strip():
            raise ValueError("Target detector-head path must be nonempty.")
        if self.name not in {"text", "image"}:
            raise ValueError("Target detector head name must be 'text' or 'image'.")
        if (
            not is_sha256(self.detector_sha256)
            or self.detector_sha256 != self.detector_sha256.lower()
        ):
            raise ValueError("Target detector-head SHA-256 must be canonical.")
        if isinstance(self.review_threshold, bool) or not 0.0 <= float(
            self.review_threshold
        ) < 1.0:
            raise ValueError("Target detector-head review threshold must be in [0, 1).")


@dataclass(frozen=True)
class TargetProfile:
    name: str
    title: str
    description: str
    status: str
    model_family: str
    intended_modalities: tuple[str, ...]
    detector: str | None
    detector_sha256: str | None
    provider: str
    provider_options: dict[str, object]
    model_source: TargetModelSource
    cache_dir: str
    warmup_request_json: str
    require_cuda: bool
    review_threshold: float | None
    inference_timeout_seconds: float
    worker_startup_timeout_seconds: float
    resources: dict[str, int]
    caveats: tuple[str, ...]
    detector_heads: tuple[TargetDetectorHead, ...] = ()

    def __post_init__(self) -> None:
        if self.detector_heads and (
            self.detector is not None
            or self.detector_sha256 is not None
            or self.review_threshold is not None
        ):
            raise ValueError(
                "Legacy detector fields and detector_heads cannot be configured together."
            )
        if not self.detector_heads and (
            not isinstance(self.detector, str)
            or not is_sha256(self.detector_sha256 or "")
        ):
            raise ValueError("A single target requires a detector path and SHA-256.")
        if self.detector_heads and (
            len(self.detector_heads) != 2
            or {head.name for head in self.detector_heads} != {"text", "image"}
        ):
            raise ValueError("A dual target requires exactly text and image heads.")

    @property
    def detector_mode(self) -> str:
        return "or" if self.detector_heads else "single"

    @property
    def detector_identity_sha256(self) -> str:
        """Runtime identity; equal to the artifact hash for legacy single mode."""

        if not self.detector_heads:
            assert self.detector_sha256 is not None
            return self.detector_sha256
        return detector_set_identity_sha256(
            tuple(
                (head.name, head.detector_sha256, head.review_threshold)
                for head in self.detector_heads
            )
        )

    def summary(self, root: str | Path = ".") -> dict[str, object]:
        root_path = Path(root)
        detector_asset = (
            None
            if self.detector is None
            else resolve_target_asset(self.detector, source_root=root_path)
        )
        warmup_asset = resolve_target_asset(
            self.warmup_request_json,
            source_root=root_path,
        )
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "model_family": self.model_family,
            "intended_modalities": list(self.intended_modalities),
            "detector": self.detector,
            "detector_sha256": self.detector_sha256,
            "detector_identity_sha256": self.detector_identity_sha256,
            "detector_mode": self.detector_mode,
            "detector_asset": None if detector_asset is None else detector_asset.to_dict(),
            "detector_exists": (
                all(
                    resolve_target_asset(head.detector, source_root=root_path).exists
                    for head in self.detector_heads
                )
                if self.detector_heads
                else bool(detector_asset and detector_asset.exists)
            ),
            "detector_heads": [
                {
                    "name": head.name,
                    "detector": head.detector,
                    "detector_sha256": head.detector_sha256,
                    "review_threshold": head.review_threshold,
                    "detector_asset": resolve_target_asset(
                        head.detector, source_root=root_path
                    ).to_dict(),
                }
                for head in self.detector_heads
            ],
            "warmup_asset": warmup_asset.to_dict(),
            "provider": self.provider,
            "model_id": self.provider_options.get("model_id"),
            "model_revision": self.provider_options.get("model_revision"),
            "tokenizer_revision": self.provider_options.get("tokenizer_revision"),
            "model_source": asdict(self.model_source),
            "require_cuda": self.require_cuda,
            "review_threshold": self.review_threshold,
            "inference_timeout_seconds": self.inference_timeout_seconds,
            "worker_startup_timeout_seconds": self.worker_startup_timeout_seconds,
            "resources": self.resources,
            "caveats": list(self.caveats),
        }


_TARGET_PROFILES = {
    "qwen25vl3b": TargetProfile(
        name="qwen25vl3b",
        title="Qwen2.5-VL 3B v7 dual-head guard",
        description="Image-text guard using the Qwen2.5-VL 3B v7 dual-head detector.",
        status="research_candidate",
        model_family="qwen25_vl",
        intended_modalities=("image_text",),
        detector=None,
        detector_sha256=None,
        provider="AEGIS.providers:Qwen25VLGuardrailProvider",
        provider_options={
            "environment_overrides": False,
            "local_files_only": True,
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "tokenizer_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "layer": -1,
            "pooling": "text_image_tokens",
            "feature_dim": 4096,
            "torch_dtype": "auto",
            "device_map": "auto",
            "min_pixels": None,
            "max_pixels": 200704,
            "cache_requests": True,
            "request_cache_max_entries": 128,
            "request_cache_ttl_seconds": 300.0,
        },
        model_source=TargetModelSource(
            repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
            revision="66285546d2b821cf421d4f5eb2576359d3770cd3",
            storage_mode="huggingface_cache",
            expected_content_sha256=(
                "45f7d1afd0ef8e09cb7a79456fd232014d2a482ca975d2008848c0efa77e4ce0"
            ),
        ),
        cache_dir="models/huggingface",
        warmup_request_json="configs/service_warmup_request.example.json",
        require_cuda=True,
        review_threshold=None,
        detector_heads=(
            TargetDetectorHead(
                name="text",
                detector=(
                    "models/aegis/qwen25vl3b_v7_dual_or/"
                    "qwen25vl3b_text_head_v1.npz"
                ),
                detector_sha256=(
                    "8818a75eb0fbe9fb1a0acb9f2035cb8b9f48f083ea1ee7aee03e15ec549533c8"
                ),
                review_threshold=0.9775257227486033,
            ),
            TargetDetectorHead(
                name="image",
                detector=(
                    "models/aegis/qwen25vl3b_v7_dual_or/"
                    "qwen25vl3b_image_head_v1.npz"
                ),
                detector_sha256=(
                    "e17eff8de97b45c22887139417d84bdd32ed067c9615c0d601f9af4467b18deb"
                ),
                review_threshold=0.1490249723273814,
            ),
        ),
        inference_timeout_seconds=120.0,
        worker_startup_timeout_seconds=900.0,
        resources={
            "min_total_physical_bytes": 16106127360,
            "min_available_physical_bytes": 6442450944,
            "min_available_virtual_bytes": 8589934592,
            "min_disk_free_bytes": 12884901888,
            "min_cuda_device_memory_bytes": 7516192768,
            "min_model_cache_bytes": 6442450944,
        },
        caveats=(
            "Validated on neutrally rendered Bordair OCR cases; native non-OCR multimodal images were unavailable.",
            "The dual-head internal test blocked all 100 attacks and falsely blocked 1 of 80 benign cases.",
            "The frozen fixed panel blocked all 10 attacks and 2 of 10 benign controls; the text-led panel blocked all 10 attacks.",
            "The frozen external-benign panel blocked 2 of 20 controls.",
            "This profile is not approved for production blocking.",
        ),
    ),
    "llava05b": TargetProfile(
        name="llava05b",
        title="LLaVA-OneVision 0.5B v7 dual-head guard",
        description="Image-text guard using the LLaVA-OneVision 0.5B v7 dual-head detector.",
        status="research_candidate",
        model_family="llava_onevision",
        intended_modalities=("image_text",),
        detector=None,
        detector_sha256=None,
        provider="AEGIS.providers:LlavaOnevisionGuardrailProvider",
        provider_options={
            "environment_overrides": False,
            "local_files_only": True,
            "model_id": "models\\huggingface\\llava-onevision-qwen2-0.5b-ov-hf",
            "runtime_model_id": (
                "models/huggingface/llava-onevision-qwen2-0.5b-ov-hf"
            ),
            "model_revision": (
                "c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c"
            ),
            "tokenizer_revision": (
                "c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c"
            ),
            "layer": -1,
            "pooling": "text_image_tokens",
            "feature_dim": 1792,
            "torch_dtype": "auto",
            "device_map": "auto",
            "max_image_edge": 384,
            "cache_requests": True,
            "request_cache_max_entries": 128,
            "request_cache_ttl_seconds": 300.0,
        },
        model_source=TargetModelSource(
            repo_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            revision="74dd0bf867a4cda7950c17663794267c60cf4b40",
            storage_mode="local_directory",
            local_directory=(
                "models/huggingface/llava-onevision-qwen2-0.5b-ov-hf"
            ),
            expected_content_sha256=(
                "c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c"
            ),
        ),
        cache_dir="models/huggingface",
        warmup_request_json="configs/llava_service_warmup_request.example.json",
        require_cuda=True,
        review_threshold=None,
        detector_heads=(
            TargetDetectorHead(
                name="text",
                detector=(
                    "models/aegis/llava05b_v7_dual_or/"
                    "llava05b_text_head_v1.npz"
                ),
                detector_sha256=(
                    "cd8502fc73ecaf1e6597d318d63a82fcdf7abae628c1a54e5390b82b2a999a61"
                ),
                review_threshold=0.9773003604375604,
            ),
            TargetDetectorHead(
                name="image",
                detector=(
                    "models/aegis/llava05b_v7_dual_or/"
                    "llava05b_image_head_v1.npz"
                ),
                detector_sha256=(
                    "84b12df0887f420318e2f3f530d0dc4391e8c1e1e5d1e15ff28f4f719c6b80b5"
                ),
                review_threshold=0.41109073768976995,
            ),
        ),
        inference_timeout_seconds=120.0,
        worker_startup_timeout_seconds=600.0,
        resources={
            "min_total_physical_bytes": 8000000000,
            "min_available_physical_bytes": 2147483648,
            "min_available_virtual_bytes": 4294967296,
            "min_disk_free_bytes": 4294967296,
            "min_cuda_device_memory_bytes": 2684354560,
            "min_model_cache_bytes": 1610612736,
        },
        caveats=(
            "Validated on neutrally rendered Bordair OCR cases; native non-OCR multimodal images were unavailable.",
            "The dual-head internal test blocked all 100 attacks and falsely blocked 1 of 80 benign cases.",
            "The frozen fixed panel blocked all 10 attacks and 2 of 10 benign controls; the text-led panel blocked all 10 attacks.",
            "The frozen external-benign panel blocked 2 of 20 controls.",
            "This profile is not approved for production blocking.",
        ),
    ),
}


def target_profile_names() -> tuple[str, ...]:
    return tuple(sorted(_TARGET_PROFILES))


def get_target_profile(name: str) -> TargetProfile:
    try:
        return _TARGET_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown target profile {name!r}; choose one of {list(target_profile_names())}."
        ) from exc


def list_target_profiles(root: str | Path = ".") -> dict[str, object]:
    return {
        "profiles": [
            get_target_profile(name).summary(root) for name in target_profile_names()
        ],
        "safety_note": (
            "Built-in profiles select compatible software artifacts; their status still "
            "requires target-specific operational evidence before blocking use."
        ),
    }
