from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = REPOSITORY / "outputs" / "bordair_retraining_v1"


@dataclass(frozen=True)
class RunPaths:
    """All generated v7 paths derived from one explicit run root.

    Keeping these paths together prevents a command that overrides the corpus root
    from silently reading a summary, feature cache, or evaluation result from the
    repository's default output directory.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.expanduser().resolve())

    @classmethod
    def default(cls) -> "RunPaths":
        return cls(DEFAULT_RUN_ROOT)

    @property
    def manifest(self) -> Path:
        return self.root / "corpus_manifest_v7.json"

    @property
    def development_metadata(self) -> Path:
        return self.root / "development_metadata_v7.csv"

    @property
    def regression_metadata(self) -> Path:
        return self.root / "regression_metadata_v7.csv"

    @property
    def final_metadata(self) -> Path:
        return self.root / "final_metadata_v7.csv"

    @property
    def text_led_metadata(self) -> Path:
        return self.root / "text_led_final_metadata_v7.csv"

    @property
    def external_benign_metadata(self) -> Path:
        return self.root / "external_benign_metadata_v7.csv"

    @property
    def image_root(self) -> Path:
        return self.root / "images_v7"

    @property
    def training_directory(self) -> Path:
        return self.root / "training_v7"

    @property
    def evaluation_directory(self) -> Path:
        return self.root / "evaluation_v7"

    @property
    def feature_cache_directory(self) -> Path:
        return self.root / "features_v7"

    def choose(self, override: Path | None, default: Path) -> Path:
        """Resolve a CLI override, or return its run-root-derived default."""

        return default if override is None else override.expanduser().resolve()

    def corpus_input(
        self,
        override: Path | None,
        default: Path,
        option_name: str,
    ) -> Path:
        """Resolve a corpus input while preserving one strict validation boundary."""

        selected = self.choose(override, default)
        if selected != default:
            raise ValueError(
                f"{option_name} must resolve inside --run-root as {default}. "
                "Set --run-root to the corpus root instead of mixing run trees."
            )
        return selected
