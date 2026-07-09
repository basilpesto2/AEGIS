from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

from AEGIS.litmus import LITMUS_SOURCE, METADATA_FILENAME, build_litmus_panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a held-out multimodal litmus panel for aegis classify."
    )
    parser.add_argument(
        "--output-dir",
        default=f"data/processed/{LITMUS_SOURCE}",
        help="Directory that will receive metadata.csv and generated images.",
    )
    parser.add_argument(
        "--reference-metadata",
        nargs="*",
        default=["data/processed/*metadata.csv", "outputs/*metadata.csv"],
        help="Existing metadata CSVs or glob patterns used to reject prompt overlap.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_metadata = (output_dir / METADATA_FILENAME).resolve()
    reference_paths = _expand_reference_paths(args.reference_metadata, output_metadata)
    table = build_litmus_panel(output_dir, reference_metadata_paths=reference_paths)
    label_counts = table["label"].value_counts().to_dict()
    print(
        f"Wrote {len(table)} {LITMUS_SOURCE} rows to {output_dir / METADATA_FILENAME} "
        f"with label counts {label_counts}."
    )


def _expand_reference_paths(patterns: list[str], output_metadata: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved == output_metadata or resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


if __name__ == "__main__":
    main()
