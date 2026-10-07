from __future__ import annotations

import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_ROOT))

from aegis_research.bordair_training import main  # noqa: E402


if __name__ == "__main__":
    main()
