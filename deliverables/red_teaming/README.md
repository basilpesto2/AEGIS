# Red-teaming scripts

These tools create deterministic, safety-redacted prompt variants and evaluate bounded
score-query adaptation. They are intended to test detector robustness without storing
operational harmful instructions.

## Generate the panel

```powershell
python deliverables/red_teaming/generate_variants.py
```

By default, the script transforms every malicious test row in the annotated benchmark
into six variants and writes `generated/variants.csv`. Image-overlay variants receive
new generated cue-card PNGs. Re-running with the same input produces byte-identical
metadata and images.

The committed smoke evaluation is reproducible with:

```powershell
python deliverables/reproducible_pipeline/scripts/build_smoke_features.py `
  --metadata deliverables/red_teaming/generated/variants.csv `
  --output deliverables/red_teaming/generated/variant_smoke_features.npz
python deliverables/reproducible_pipeline/scripts/score_feature_bundle.py `
  --metadata deliverables/red_teaming/generated/variants.csv `
  --features deliverables/red_teaming/generated/variant_smoke_features.npz `
  --detector deliverables/reproducible_pipeline/artifacts/smoke/selected_detector.npz `
  --output deliverables/red_teaming/generated/scored_variants.csv
```

## Evaluate a detector

Score every variant and add a numeric `risk_score` column to a CSV. Then run:

```powershell
python deliverables/red_teaming/evaluate_adaptive.py `
  --scores path/to/scored_variants.csv `
  --threshold 0.5 `
  --query-budgets 1 3 6
```

For each base sample and budget, the evaluator assumes a bounded adversary selects the
lowest-risk variant among the first `budget` deterministic queries. It reports malicious
recall, evasion rate, and mean worst-case score. This is a detector-evasion measure, not
a claim that the protected MLLM generated harmful content.

For a response-level evaluation, record adjudicated `attack_succeeded` labels and guard
actions using `response_judgment_schema.json`, then run
`evaluate_attack_success.py`. The script reports guarded versus unguarded attack-success
rates and absolute/relative reduction without storing response text.

See `SAFE_USE.md` before adding attack families or external red-team data.
