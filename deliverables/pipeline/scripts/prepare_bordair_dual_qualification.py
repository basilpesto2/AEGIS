from __future__ import annotations

from pathlib import Path
import sys


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PIPELINE_ROOT.parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(REPOSITORY))

from aegis_research.bordair_dual_qualification import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
