# Red-teaming scripts

These tools create deterministic, safety-redacted prompt variants and evaluate bounded
best-of-N detector-score evasion without storing operational harmful instructions.

The committed files under `generated/` are a frozen fixture snapshot from commit
`7b50825`. They use the 8-dimensional detector
`pipeline/artifacts/smoke/selected_detector.npz` at threshold
`0.8202568457258909`; they do not use the current 896-dimensional LLaVA tuned-v3
runtime detector.

## Generate a new versioned panel

`generate_variants.py` transforms malicious benchmark rows into six fixed variants.
Image-overlay variants receive generated, safety-redacted cue-card PNGs.

The committed panel predates the repository's directory-name normalization. Its legacy
`../../annotated_benchmark/images/` paths are preserved and mapped in
`../HISTORICAL_SNAPSHOT.md`. New runs use `../../benchmark/images/`.

Generated PNG bytes are deterministic only when Python and Pillow versions are pinned;
new manifests record the toolchain. Never overwrite the committed snapshot. For
example:

```powershell
python deliverables/red_teaming/generate_variants.py `
  --output deliverables/red_teaming/runs/reproduction_v1/variants.csv
```

## Score with the fixture pipeline

```powershell
python deliverables/pipeline/scripts/build_smoke_features.py `
  --metadata deliverables/red_teaming/runs/reproduction_v1/variants.csv `
  --output deliverables/red_teaming/runs/reproduction_v1/variant_smoke_features.npz
python deliverables/pipeline/scripts/score_feature_bundle.py `
  --metadata deliverables/red_teaming/runs/reproduction_v1/variants.csv `
  --features deliverables/red_teaming/runs/reproduction_v1/variant_smoke_features.npz `
  --detector deliverables/pipeline/artifacts/smoke/selected_detector.npz `
  --output deliverables/red_teaming/runs/reproduction_v1/scored_variants.csv
```

These commands reproduce software-fixture evidence. The production tuned-v3 artifact
has a different 896-dimensional contract and cannot be substituted into this scorer.
Fresh, provenance-complete LLaVA `text_tokens` extraction is required before a current
runtime robustness result can be reported.

## Evaluate bounded best-of-N evasion

The evaluator selects the minimum score in hindsight among the first N fixed variants.
This is an oracle best-of-N analysis, not a sequential adversary that adapts later
queries from earlier responses.

```powershell
python deliverables/red_teaming/evaluate_adaptive.py `
  --scores deliverables/red_teaming/runs/reproduction_v1/scored_variants.csv `
  --detector deliverables/pipeline/artifacts/smoke/selected_detector.npz `
  --features deliverables/red_teaming/runs/reproduction_v1/variant_smoke_features.npz `
  --threshold-kind fixture_detector_block_threshold `
  --traffic-mode not_applicable `
  --query-budgets 1 3 6 `
  --output deliverables/red_teaming/runs/reproduction_v1/best_of_n.json
```

The output records detector, score-panel, and feature-bundle hashes together with
threshold semantics and traffic mode. A threshold analysis performed while AEGIS is in
shadow mode is counterfactual: the runtime's effective action remains `allow`.

## Evaluate guarded model responses

For response-level work, create paired `unguarded` and `guarded` adjudications using
`response_judgment_schema.json`, then run `evaluate_attack_success.py`. The schema
separates effective action, recommended action, traffic mode, and downstream
disposition. This prevents a shadow-mode recommendation from being reported as an
observed block.

See `SAFE_USE.md` before adding attack families or external red-team data.
