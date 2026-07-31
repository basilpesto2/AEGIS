# AEGIS research and validation deliverables

This directory preserves the evidence used to design, test, and evaluate AEGIS. It is
required research documentation for the Final Report, but it is intentionally separate
from the files needed to install and run the CLI, service, or GUI.

The current repository package is AEGIS `0.3.0`. The research evidence and the deployed
runtime do not all make the same kind of claim, so each artifact is labelled by its
evidence boundary.

| Subdirectory | Deliverable | Evidence boundary |
| --- | --- | --- |
| `benchmark/` | Balanced, synthetic, safety-redacted multimodal benchmark; annotations, schema, generated images, CSV/JSON data, and Excel workbook. | Dataset and annotation evidence; not a production-safety claim. |
| `pipeline/` | Feature-bundle contract, detector training, low-label learning, evaluation, and tests. | Deterministic software-smoke results unless a provenance-complete MLLM bundle is supplied. |
| `red_teaming/` | Bounded, safety-redacted attack generation and adaptive detector-evasion evaluation. | Detector robustness evidence; no harmful model responses are stored. |
| `ablation_report/` | Executable ablation protocol, aggregate smoke results, transfer analysis, and historical aggregate context. | Committed tables use deterministic smoke features; legacy MLLM aggregates are not independently reproducible. |
| `runtime_validation/` | Sanitized Docker, API, inference, GUI, and resource-preflight observations from 2026-07-31. | Live validation of the current packaged LLaVA runtime on the recorded machine. |
| `ci/` | Portable pipeline unit tests and deliverable-integrity verification. | Research-artifact CI, separate from product runtime tests. |

`HISTORICAL_SNAPSHOT.md` records the original fixture hashes, obsolete embedded path
names, and the exact boundary between preserved historical results and current runtime
evidence.

## Current runtime relationship

- The bundled LLaVA target now uses
  `models/aegis/aegis_llava_onevision_05b_text_detector_tuned_v3.npz`, with a
  validation-selected block threshold of `0.6909739881800183` and review threshold of
  `0.23579741243702598`.
- The committed pipeline smoke detector in
  `pipeline/artifacts/smoke/selected_detector.npz` is an 8-dimensional deterministic
  fixture artifact. It proves that the research pipeline executes; it is not the
  896-dimensional detector deployed by the LLaVA service.
- The current CLI, authenticated API, CUDA-backed LLaVA inference, runtime traffic-mode
  control, and local GUI were live-tested successfully. The stock LLaVA profile did not
  remain up on the validation machine because Docker exposed only 1.72-1.82 GiB free
  physical memory after initialization, below the configured 2.00 GiB gate. See
  `runtime_validation/README.md`.
- The Qwen profile was not rerun: its 15 GiB total-memory requirement exceeds the
  recorded 7.61 GiB Docker allocation, and its model cache was not prepared.

## Verify the committed evidence

From the repository root, using Python 3.10 or newer:

```powershell
python deliverables/ci/run_checks.py
```

This runs the research-pipeline unit tests and a read-only integrity check over the
canonical committed evidence. Use `python deliverables/verify_deliverables.py --update`
only when intentionally refreshing `MANIFEST.json` and `verification_report.json`
after a reviewed deliverable change.

## Reproduce into a new versioned run

Never overwrite the committed historical scores or result tables. The following
example writes a complete deterministic reproduction under a new run directory:

```powershell
$run = "deliverables/reproductions/smoke_v1"

python deliverables/pipeline/scripts/build_smoke_features.py `
  --metadata deliverables/benchmark/data/benchmark.csv `
  --output "$run/benchmark_smoke_features.npz"
python deliverables/pipeline/scripts/run_experiment.py `
  --metadata deliverables/benchmark/data/benchmark.csv `
  --features "$run/benchmark_smoke_features.npz" `
  --output-dir "$run/pipeline"

python deliverables/red_teaming/generate_variants.py `
  --output "$run/red_team/variants.csv"
python deliverables/pipeline/scripts/build_smoke_features.py `
  --metadata "$run/red_team/variants.csv" `
  --output "$run/red_team/variant_smoke_features.npz"
python deliverables/pipeline/scripts/score_feature_bundle.py `
  --metadata "$run/red_team/variants.csv" `
  --features "$run/red_team/variant_smoke_features.npz" `
  --detector "$run/pipeline/selected_detector.npz" `
  --output "$run/red_team/scored_variants.csv"
python deliverables/red_teaming/evaluate_adaptive.py `
  --scores "$run/red_team/scored_variants.csv" `
  --detector "$run/pipeline/selected_detector.npz" `
  --features "$run/red_team/variant_smoke_features.npz" `
  --threshold-kind fixture_detector_block_threshold `
  --traffic-mode not_applicable `
  --query-budgets 1 3 6 `
  --output "$run/red_team/best_of_n.json"

python deliverables/ablation_report/run_ablation.py `
  --metadata deliverables/benchmark/data/benchmark.csv `
  --features "$run/benchmark_smoke_features.npz" `
  --output-dir "$run/ablation" `
  --adaptive-summary "$run/red_team/best_of_n.json" `
  --adaptive-scores "$run/red_team/scored_variants.csv" `
  --adaptive-detector "$run/pipeline/selected_detector.npz" `
  --adaptive-features "$run/red_team/variant_smoke_features.npz"
```

`benchmark/build_workbook.mjs` additionally produces the Excel workbook and QA
previews. It requires the bundled Codex spreadsheet runtime; the canonical
machine-readable benchmark remains `benchmark/data/benchmark.csv`.

Model downloads and GPU extraction are deliberately excluded from this deterministic
smoke workflow. To produce new MLLM evidence, use
`pipeline/scripts/extract_mllm_features.py`, retain its provenance manifest, and rerun
the same experiment and ablation entry points.

## Safety and disclosure

All newly authored adversarial examples use explicit redaction placeholders instead
of operational harmful content. Third-party dataset content and model weights are not
redistributed. See `benchmark/DATASHEET.md` and `red_teaming/SAFE_USE.md` before
extending the corpus.
