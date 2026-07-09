from __future__ import annotations

import numpy as np


def make_family_holdout_masks(
    labels: np.ndarray,
    families: np.ndarray,
    heldout_family: str,
    validation_fraction: float = 0.2,
    seed: int = 81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create fit/validation/test masks with one malicious family held out for test."""
    labels = np.asarray(labels, dtype=np.int64)
    families = np.asarray(families, dtype=str)
    if labels.shape != families.shape:
        raise ValueError("labels and families must have the same shape.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    heldout_malicious = np.where((labels == 1) & (families == heldout_family))[0]
    if len(heldout_malicious) == 0:
        raise ValueError(f"No malicious rows found for held-out family {heldout_family!r}.")

    benign_candidates = np.where(labels == 0)[0]
    if len(benign_candidates) < len(heldout_malicious):
        raise ValueError("Not enough benign rows to balance the held-out test set.")

    rng = np.random.default_rng(seed)
    test_benign = rng.choice(
        benign_candidates,
        size=len(heldout_malicious),
        replace=False,
    )
    test_mask = np.zeros(len(labels), dtype=bool)
    test_mask[heldout_malicious] = True
    test_mask[test_benign] = True

    remaining_indices = np.where(~test_mask & ~((labels == 1) & (families == heldout_family)))[0]
    validation_mask = np.zeros(len(labels), dtype=bool)
    grouped_values = sorted({(int(labels[index]), families[index]) for index in remaining_indices})
    for label, family in grouped_values:
        group = remaining_indices[
            (labels[remaining_indices] == label) & (families[remaining_indices] == family)
        ]
        if len(group) < 2:
            continue
        n_validation = int(round(len(group) * validation_fraction))
        n_validation = min(max(n_validation, 1), len(group) - 1)
        selected = rng.choice(group, size=n_validation, replace=False)
        validation_mask[selected] = True

    fit_mask = ~test_mask & ~validation_mask
    _require_both_classes(labels, fit_mask, "fit")
    _require_both_classes(labels, validation_mask, "validation")
    _require_both_classes(labels, test_mask, "test")
    return fit_mask, validation_mask, test_mask


def make_paired_family_holdout_masks(
    labels: np.ndarray,
    families: np.ndarray,
    pair_ids: np.ndarray,
    heldout_family: str,
    validation_fraction: float = 0.2,
    seed: int = 81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold out all matched pairs from one family and keep remaining pairs together."""
    labels = np.asarray(labels, dtype=np.int64)
    families = np.asarray(families, dtype=str)
    pair_ids = np.asarray(pair_ids, dtype=str)
    if labels.shape != families.shape or labels.shape != pair_ids.shape:
        raise ValueError("labels, families, and pair_ids must have the same shape.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    test_mask = families == heldout_family
    if not np.any(test_mask & (labels == 1)):
        raise ValueError(f"No malicious rows found for held-out family {heldout_family!r}.")
    if not np.any(test_mask & (labels == 0)):
        raise ValueError(f"No benign matched controls found for held-out family {heldout_family!r}.")

    remaining_pairs = np.unique(pair_ids[~test_mask])
    rng = np.random.default_rng(seed)
    validation_pairs = []
    remaining_families = sorted(set(families[~test_mask]))
    for family in remaining_families:
        family_pairs = np.unique(pair_ids[(~test_mask) & (families == family)])
        if len(family_pairs) < 2:
            continue
        n_validation = int(round(len(family_pairs) * validation_fraction))
        n_validation = min(max(n_validation, 1), len(family_pairs) - 1)
        validation_pairs.extend(
            rng.choice(family_pairs, size=n_validation, replace=False).tolist()
        )

    validation_mask = np.isin(pair_ids, np.asarray(validation_pairs, dtype=str))
    fit_mask = np.isin(pair_ids, remaining_pairs) & ~validation_mask
    _require_both_classes(labels, fit_mask, "fit")
    _require_both_classes(labels, validation_mask, "validation")
    _require_both_classes(labels, test_mask, "test")
    return fit_mask, validation_mask, test_mask


def _require_both_classes(labels: np.ndarray, mask: np.ndarray, name: str) -> None:
    selected = set(np.unique(labels[mask]))
    if selected != {0, 1}:
        raise ValueError(f"{name} split must contain both classes, got {sorted(selected)}.")
