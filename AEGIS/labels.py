from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


MALICIOUS_STRINGS = {
    "1",
    "true",
    "malicious",
    "attack",
    "adversarial",
    "unsafe",
    "harmful",
    "policy_violation",
    "policy-violation",
}

BENIGN_STRINGS = {
    "0",
    "false",
    "benign",
    "safe",
    "clean",
    "normal",
}


def coerce_binary_labels(values: Iterable[object]) -> np.ndarray:
    """Convert common benign/malicious labels to 0/1 integers."""
    labels = []
    for value in values:
        if pd.isna(value):
            raise ValueError("Labels contain missing values.")
        if isinstance(value, (int, np.integer)):
            if value in (0, 1):
                labels.append(int(value))
                continue
        if isinstance(value, (float, np.floating)):
            if value in (0.0, 1.0):
                labels.append(int(value))
                continue

        normalized = str(value).strip().lower()
        if normalized in MALICIOUS_STRINGS:
            labels.append(1)
        elif normalized in BENIGN_STRINGS:
            labels.append(0)
        else:
            raise ValueError(f"Unrecognized binary label: {value!r}")

    return np.asarray(labels, dtype=np.int64)

