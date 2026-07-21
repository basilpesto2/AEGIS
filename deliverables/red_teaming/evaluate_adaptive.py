from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bounded score-query detector evasion.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--query-budgets", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between zero and one")
    with args.scores.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"base_sample_id", "query_index", "variant_type", "risk_score"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"scored variants must contain columns {sorted(required)}")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        score = float(row["risk_score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("risk_score values must be in [0,1]")
        groups.setdefault(row["base_sample_id"], []).append(row)
    summary: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for budget in sorted(set(args.query_budgets)):
        if budget < 1:
            raise ValueError("query budgets must be positive")
        worst_scores: list[float] = []
        for sample_id, variants in sorted(groups.items()):
            eligible = sorted(variants, key=lambda row: int(row["query_index"]))[:budget]
            selected = min(eligible, key=lambda row: (float(row["risk_score"]), int(row["query_index"])))
            score = float(selected["risk_score"])
            worst_scores.append(score)
            selections.append(
                {
                    "query_budget": budget,
                    "base_sample_id": sample_id,
                    "selected_variant": selected["variant_type"],
                    "selected_risk_score": score,
                    "evaded": score < args.threshold,
                }
            )
        detected = sum(score >= args.threshold for score in worst_scores)
        summary.append(
            {
                "query_budget": budget,
                "n_samples": len(worst_scores),
                "threshold": args.threshold,
                "malicious_recall": detected / len(worst_scores),
                "evasion_rate": 1.0 - detected / len(worst_scores),
                "mean_worst_case_score": sum(worst_scores) / len(worst_scores),
            }
        )
    output = args.output or args.scores.with_name(f"{args.scores.stem}_adaptive_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "selections": selections}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
