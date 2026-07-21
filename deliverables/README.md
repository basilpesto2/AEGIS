# AEGIS research deliverables

This directory contains the research artifacts of the AEGIS project. 
It is deliberately self-contained: the deployment service remains at
the repository root, while the material needed to inspect, reproduce, and extend the
research lives here.

| Subdirectory | Deliverable |
| --- | --- |
| `reproducible_pipeline/` | Feature-bundle contract, detector training, low-label learning, evaluation, and tests. |
| `red_teaming/` | Deterministic, bounded, safety-redacted attack generation and adaptive evaluation tools. |
| `annotated_benchmark/` | Self-contained balanced multimodal benchmark, images, annotations, schema, and data documentation. |
| `ablation_report/` | Executable ablation protocol, aggregate results, transfer analysis, and report. |
| `ci/` | Portable unit-test and 35-check verification entry point plus an opt-in workflow template. |

## Reproduce everything

From the repository root, using Python 3.10 or newer:

```powershell
python deliverables/annotated_benchmark/build_assets.py
python deliverables/red_teaming/generate_variants.py
python deliverables/reproducible_pipeline/scripts/build_smoke_features.py
python deliverables/ablation_report/run_ablation.py
python deliverables/verify_deliverables.py
```

`build_workbook.mjs` additionally produces the Excel version of the
benchmark. It uses the bundled Codex spreadsheet runtime; the canonical machine input
remains `annotated_benchmark/data/benchmark.csv`.

The committed smoke features and ablation numbers are pipeline-verification evidence,
not a substitute for rerunning MLLM hidden-state extraction.  Historical MLLM results
are reported separately and clearly marked as legacy evidence.

## Safety and disclosure

All newly authored adversarial examples use explicit redaction placeholders instead
of operational harmful content. Third-party dataset content and model weights are not
redistributed. See `annotated_benchmark/DATASHEET.md` and
`red_teaming/SAFE_USE.md` before extending the corpus.
