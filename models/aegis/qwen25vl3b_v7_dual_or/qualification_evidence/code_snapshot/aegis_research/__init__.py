"""Self-contained research utilities for the AEGIS deliverables."""

from .experiment import ExperimentConfig, run_ablation_suite
from .io import FeatureBundle, load_feature_bundle, load_metadata
from .model import LogisticDetector

__all__ = [
    "ExperimentConfig",
    "FeatureBundle",
    "LogisticDetector",
    "load_feature_bundle",
    "load_metadata",
    "run_ablation_suite",
]
