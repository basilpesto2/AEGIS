from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from AEGIS.target_assets import resolve_target_asset


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


@dataclass(frozen=True)
class TargetProfile:
    name: str
    title: str
    description: str
    status: str
    model_family: str
    intended_modalities: tuple[str, ...]
    detector: str
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

    def summary(self, root: str | Path = ".") -> dict[str, object]:
        root_path = Path(root)
        detector_asset = resolve_target_asset(self.detector, source_root=root_path)
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
            "detector_asset": detector_asset.to_dict(),
            "detector_exists": detector_asset.exists,
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
        title="Qwen2.5-VL 3B controlled guard",
        description="Image-text and text input guard using Qwen2.5-VL 3B hidden states.",
        status="research_candidate",
        model_family="qwen25_vl",
        intended_modalities=("text", "image_text"),
        detector="models/aegis/aegis_qwen25vl3b_text_detector.npz",
        provider="AEGIS.providers:Qwen25VLGuardrailProvider",
        provider_options={
            "environment_overrides": False,
            "local_files_only": True,
            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "model_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "tokenizer_revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "layer": -1,
            "pooling": "text_tokens",
            "feature_dim": 2048,
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
        ),
        cache_dir="models/huggingface",
        warmup_request_json="configs/service_warmup_request.example.json",
        require_cuda=True,
        review_threshold=None,
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
            "The detector predates immutable revision/preprocessing fields and must be rebuilt.",
            "The latest Windows live probe failed because the paging file was too small.",
            "This profile is not approved for production blocking.",
        ),
    ),
    "llava05b": TargetProfile(
        name="llava05b",
        title="LLaVA-OneVision 0.5B controlled guard",
        description="Image-text guard using the local LLaVA-OneVision 0.5B checkpoint.",
        status="research_candidate",
        model_family="llava_onevision",
        intended_modalities=("image_text",),
        detector=(
            "models/aegis/"
            "aegis_llava_onevision_05b_text_detector_provenance_v2.npz"
        ),
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
            "pooling": "text_tokens",
            "feature_dim": 896,
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
        review_threshold=0.05,
        inference_timeout_seconds=120.0,
        worker_startup_timeout_seconds=600.0,
        resources={
            "min_total_physical_bytes": 8589934592,
            "min_available_physical_bytes": 2147483648,
            "min_available_virtual_bytes": 4294967296,
            "min_disk_free_bytes": 4294967296,
            "min_cuda_device_memory_bytes": 2684354560,
            "min_model_cache_bytes": 1610612736,
        },
        caveats=(
            "The provenance-v2 detector still uses the legacy 360/180/180 research split.",
            "The live evasion surrogate is routed to review, but broader adjudicated adversarial coverage is still required.",
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
