# Ablation and transfer report

The committed `REPORT.md` and `results/` tables are a frozen deterministic fixture
snapshot. They verify the experiment, threshold-selection, low-label, uncertainty,
transfer, and reporting software; they do not measure the current tuned-v3 LLaVA
detector.

`legacy_evidence.csv` separately preserves aggregate values recovered from commit
`3f6e46a`. Its source embeddings and detailed outputs are absent, so those values are
historical context rather than independently rerun evidence.

## Reproduce into a new run

`run_ablation.py` writes to `runs/reproduction_v1/` by default and refuses to overwrite
an existing run unless `--force` is explicit. It does not silently attach the historical
fixture-evasion table to arbitrary feature bundles.

To reproduce the committed fixture protocol into a new versioned directory:

```powershell
python deliverables/ablation_report/run_ablation.py `
  --output-dir deliverables/ablation_report/runs/reproduction_v1 `
  --adaptive-summary deliverables/red_teaming/generated/adaptive_summary.json `
  --adaptive-scores deliverables/red_teaming/generated/scored_variants.csv `
  --adaptive-detector deliverables/pipeline/artifacts/smoke/selected_detector.npz `
  --adaptive-features deliverables/red_teaming/generated/variant_smoke_features.npz
```

Choose a different `--output-dir` for every evidence-producing run. Passing a new
`--features` bundle is not sufficient by itself to establish reportable MLLM evidence:
review its manifest for model ID, model and tokenizer revisions, preprocessing hash,
pooling, feature dimensions, corpus hash, and split compatibility.

The current runtime relationship and its evidence limits are stated in `REPORT.md`,
`../HISTORICAL_SNAPSHOT.md`, and `../runtime_validation/README.md`.
