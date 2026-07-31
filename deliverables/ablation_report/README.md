# Ablation and transfer workflow

`run_ablation.py` generates signal-importance, low-label, uncertainty, attack-family
transfer, and optional bounded-evasion tables from an explicitly supplied
provenance-complete MLLM feature bundle. No generated report or result table is
committed.

The metadata and feature rows must align exactly. Model family, model identifier,
model and tokenizer revisions, layer, pooling, preprocessing fingerprint, and feature
dimensions are recorded in every generated run manifest.

## Generate a report

```powershell
$run = "deliverables/runs/current_llava"

python deliverables/ablation_report/run_ablation.py `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --output-dir "$run/ablation"
```

To include a bounded red-team table, also supply the summary, score table, detector,
and feature bundle from the same current run:

```powershell
python deliverables/ablation_report/run_ablation.py `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --output-dir "$run/ablation_with_redteam" `
  --adaptive-summary "$run/redteam/best_of_n.json" `
  --adaptive-scores "$run/redteam/scored_variants.csv" `
  --adaptive-detector models/aegis/aegis_llava_onevision_05b_text_detector_tuned_v3.npz `
  --adaptive-features "$run/redteam/features/feature_bundle.npz"
```

The ablation runner verifies that the adaptive summary's recorded score, detector,
and feature hashes match the supplied files. It refuses to replace a non-empty output
directory unless `--force` is explicit. Use a new directory for each reportable run.
