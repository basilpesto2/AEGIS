from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_prompt_metadata(path: str | Path) -> pd.DataFrame:
    """Read request metadata without pandas changing prompt-like scalar text."""

    return pd.read_csv(
        path,
        engine="python",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
