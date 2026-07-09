from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def stratified_sample(
    table: pd.DataFrame,
    group_column: str,
    n_per_group: int | None = None,
    group_counts: Mapping[str, int] | None = None,
    groups: Sequence[str] | None = None,
    seed: int = 0,
    strict: bool = False,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Sample a reproducible subset from a metadata table."""
    if group_column not in table.columns:
        raise ValueError(f"Missing group column {group_column!r}.")
    if group_counts is None and n_per_group is None:
        raise ValueError("Set either n_per_group or group_counts.")
    if group_counts is not None and n_per_group is not None:
        raise ValueError("Set only one of n_per_group or group_counts.")
    if n_per_group is not None and n_per_group < 1:
        raise ValueError("n_per_group must be positive.")

    group_values = table[group_column].astype(str)
    if group_counts is not None:
        requested = {str(group): int(count) for group, count in group_counts.items()}
        if any(count < 1 for count in requested.values()):
            raise ValueError("All group_counts must be positive.")
    else:
        selected_groups = groups or sorted(group_values.unique())
        requested = {str(group): int(n_per_group) for group in selected_groups}

    rng = np.random.default_rng(seed)
    selected_indices: list[np.ndarray] = []
    for group, requested_count in requested.items():
        available = table.index[group_values == group].to_numpy()
        if len(available) == 0:
            raise ValueError(f"Group {group!r} has no rows.")
        if strict and len(available) < requested_count:
            raise ValueError(
                f"Group {group!r} has {len(available)} rows, fewer than requested {requested_count}."
            )
        count = min(requested_count, len(available))
        selected_indices.append(rng.choice(available, size=count, replace=False))

    indices = np.concatenate(selected_indices)
    if shuffle:
        rng.shuffle(indices)
    return table.loc[indices].reset_index(drop=True)


def parse_group_counts(value: str) -> dict[str, int]:
    """Parse strings like 'benign=100,malicious=20'."""
    counts: dict[str, int] = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError("Group counts must use name=count entries separated by commas.")
        group, count = item.split("=", 1)
        group = group.strip()
        if not group:
            raise ValueError("Group name cannot be empty.")
        counts[group] = int(count.strip())
    return counts


def assign_stratified_splits(
    table: pd.DataFrame,
    group_column: str,
    split_fractions: Mapping[str, float],
    split_column: str = "experiment_split",
    seed: int = 0,
) -> pd.DataFrame:
    """Assign split labels within each group while preserving group proportions."""
    if group_column not in table.columns:
        raise ValueError(f"Missing group column {group_column!r}.")
    if not split_fractions:
        raise ValueError("split_fractions cannot be empty.")
    if any(fraction < 0.0 for fraction in split_fractions.values()):
        raise ValueError("Split fractions must be non-negative.")

    total = float(sum(split_fractions.values()))
    if total <= 0.0:
        raise ValueError("Split fractions must sum to a positive value.")

    split_names = [str(name) for name in split_fractions]
    fractions = np.asarray([float(split_fractions[name]) / total for name in split_fractions])
    output = table.copy().reset_index(drop=True)
    output[split_column] = ""

    rng = np.random.default_rng(seed)
    group_values = output[group_column].astype(str)
    for group in sorted(group_values.unique()):
        indices = output.index[group_values == group].to_numpy().copy()
        rng.shuffle(indices)
        counts = _counts_from_fractions(len(indices), fractions)
        start = 0
        for split_name, count in zip(split_names, counts):
            stop = start + count
            output.loc[indices[start:stop], split_column] = split_name
            start = stop

    if (output[split_column] == "").any():
        raise RuntimeError("Some rows were not assigned to a split.")
    return output


def assign_grouped_splits(
    table: pd.DataFrame,
    unit_column: str,
    split_fractions: Mapping[str, float],
    split_column: str = "experiment_split",
    seed: int = 0,
) -> pd.DataFrame:
    """Assign split labels while keeping all rows from each unit together."""
    if unit_column not in table.columns:
        raise ValueError(f"Missing unit column {unit_column!r}.")
    if not split_fractions:
        raise ValueError("split_fractions cannot be empty.")
    if any(fraction < 0.0 for fraction in split_fractions.values()):
        raise ValueError("Split fractions must be non-negative.")

    total = float(sum(split_fractions.values()))
    if total <= 0.0:
        raise ValueError("Split fractions must sum to a positive value.")

    split_names = [str(name) for name in split_fractions]
    fractions = np.asarray([float(split_fractions[name]) / total for name in split_fractions])
    output = table.copy().reset_index(drop=True)
    output[split_column] = ""

    unit_values = output[unit_column].astype(str)
    units = np.asarray(sorted(unit_values.unique()), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    counts = _counts_from_fractions(len(units), fractions)

    start = 0
    for split_name, count in zip(split_names, counts):
        stop = start + count
        selected_units = set(units[start:stop])
        output.loc[unit_values.isin(selected_units), split_column] = split_name
        start = stop

    if (output[split_column] == "").any():
        raise RuntimeError("Some rows were not assigned to a split.")
    return output


def parse_split_fractions(value: str) -> dict[str, float]:
    """Parse strings like 'fit=0.5,val=0.25,test=0.25'."""
    fractions: dict[str, float] = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError("Split fractions must use name=fraction entries separated by commas.")
        split, fraction = item.split("=", 1)
        split = split.strip()
        if not split:
            raise ValueError("Split name cannot be empty.")
        fractions[split] = float(fraction.strip())
    return fractions


def _counts_from_fractions(n_rows: int, fractions: np.ndarray) -> np.ndarray:
    raw_counts = fractions * n_rows
    counts = np.floor(raw_counts).astype(int)
    remainder = int(n_rows - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        counts[order[:remainder]] += 1
    return counts
