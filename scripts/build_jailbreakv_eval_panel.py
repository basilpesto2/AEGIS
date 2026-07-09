from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced JailBreakV binary evaluation panel.")
    parser.add_argument("--jailbreakv-csv", required=True)
    parser.add_argument("--mssbench-csv", required=True)
    parser.add_argument("--vlguard-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--corpus-root", default="data/external")
    parser.add_argument("--mssbench-benign-count", type=int, default=180)
    parser.add_argument("--vlguard-benign-count", type=int, default=180)
    parser.add_argument("--seed", type=int, default=59)
    args = parser.parse_args()

    jailbreakv = pd.read_csv(args.jailbreakv_csv)
    malicious = _prepare_rows(
        jailbreakv,
        image_prefix="JailBreakV-28K",
        role="jailbreakv_malicious",
        split="test",
    )

    mssbench = pd.read_csv(args.mssbench_csv)
    mssbench_benign = _sample_benign(
        mssbench,
        count=args.mssbench_benign_count,
        seed=args.seed,
        group_column="source_subset",
    )
    mssbench_benign = _prepare_rows(
        mssbench_benign,
        image_prefix="MSSBench",
        role="mssbench_benign_control",
        split="test",
    )

    vlguard = pd.read_csv(args.vlguard_csv)
    vlguard_benign = _sample_benign(
        vlguard,
        count=args.vlguard_benign_count,
        seed=args.seed + 1,
        group_column="source_subset",
    )
    vlguard_benign = _prepare_rows(
        vlguard_benign,
        image_prefix="VLGuard/test",
        role="vlguard_benign_control",
        split="test",
    )

    panel = pd.concat([malicious, mssbench_benign, vlguard_benign], ignore_index=True)
    _validate_images(panel, Path(args.corpus_root))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)

    print(f"Wrote {len(panel)} rows to {args.output_csv}")
    print(panel["label"].value_counts().to_string())
    print(panel.groupby(["benchmark_role", "source", "label"]).size().to_string())


def _sample_benign(table: pd.DataFrame, count: int, seed: int, group_column: str) -> pd.DataFrame:
    benign = table[table["label"].eq("benign")].copy()
    if len(benign) < count:
        raise ValueError(f"Requested {count} benign controls, but only {len(benign)} are available.")
    if group_column not in benign.columns:
        return benign.sample(n=count, random_state=seed)

    groups = sorted(benign[group_column].dropna().unique())
    base = count // len(groups)
    remainder = count % len(groups)
    parts = []
    for index, group in enumerate(groups):
        group_rows = benign[benign[group_column].eq(group)]
        take = base + (1 if index < remainder else 0)
        if len(group_rows) < take:
            raise ValueError(f"Group {group!r} has {len(group_rows)} rows, but {take} were requested.")
        parts.append(group_rows.sample(n=take, random_state=seed + index))
    return pd.concat(parts, ignore_index=True)


def _prepare_rows(table: pd.DataFrame, image_prefix: str, role: str, split: str) -> pd.DataFrame:
    prepared = table.copy()
    prepared["original_image_path"] = prepared["image_path"]
    prepared["image_path"] = prepared["image_path"].fillna("").map(
        lambda value: _join_path(image_prefix, str(value))
    )
    prepared["benchmark_role"] = role
    prepared["split"] = split
    return prepared


def _join_path(prefix: str, image_path: str) -> str:
    prefix = prefix.strip("/").replace("\\", "/")
    image_path = image_path.strip().lstrip("/").replace("\\", "/")
    if not image_path:
        return ""
    if image_path == prefix or image_path.startswith(f"{prefix}/"):
        return image_path
    return f"{prefix}/{image_path}"


def _validate_images(table: pd.DataFrame, corpus_root: Path) -> None:
    missing = [
        image_path
        for image_path in table["image_path"].fillna("")
        if not image_path or not (corpus_root / str(image_path)).exists()
    ]
    if missing:
        examples = missing[:5]
        raise ValueError(f"{len(missing)} image paths are missing under {corpus_root}: {examples}")


if __name__ == "__main__":
    main()
