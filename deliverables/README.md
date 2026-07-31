# AEGIS research and validation deliverables

This directory contains the material that represents AEGIS `0.3.0`.
It is separate from the files required to install and run the CLI, service, or GUI.
Only current datasets, reusable evaluation tools, and validation records are retained.

| Subdirectory | Deliverable | Evidence boundary |
| --- | --- | --- |
| `benchmark/` | Balanced, synthetic, safety-redacted multimodal benchmark; annotations, schema, images, CSV/JSON data, and Excel workbook. | Dataset and annotation evidence; not a production-safety claim. |
| `pipeline/` | Provenance-aware MLLM feature extraction, detector training, scoring, and unit tests. | Results are reportable only when generated from a current, provenance-complete feature bundle. |
| `red_teaming/` | Safety-redacted variant generation and bounded detector-evasion evaluation. | Tools only; no detector robustness result is committed. |
| `ablation_report/` | Executable signal-importance, low-label, uncertainty, and transfer-analysis workflow. | Tools only; reports must be generated from an explicitly supplied current feature bundle. |
| `runtime_validation/` | Sanitized Docker, API, inference, and GUI observations from the current packaged LLaVA runtime. | Functional validation; not an accuracy, robustness, latency, or capacity benchmark. |
| `ci/` | Portable research-tool unit tests and deliverable-integrity verification. | Research-artifact CI, separate from product runtime tests. |

## Current runtime relationship

The bundled LLaVA target uses
`models/aegis/aegis_llava_onevision_05b_text_detector_tuned_v3.npz`.
It is a 896-dimensional `text_tokens` detector with a validation-selected block
threshold of `0.6909739881800183` and review threshold of
`0.23579741243702598`.

Research outputs are intentionally not committed unless their feature bundle records
the model family, model identifier, exact model and tokenizer revisions, layer,
pooling, preprocessing fingerprint, aligned sample identifiers, and feature
dimensions. The retained scripts reject missing or incompatible provenance.

## Verify the deliverables

From the repository root, using Python 3.10 or newer:

```powershell
python deliverables/ci/run_checks.py
```

This runs the research-pipeline unit tests and a read-only integrity check. Use
`python deliverables/verify_deliverables.py --update` only after a reviewed
deliverable change.

## Generate current research outputs

Create a new run directory and extract features using the same LLaVA family,
checkpoint revisions, layer, pooling, and preprocessing contract as the tuned-v3
detector:

```powershell
$run = "deliverables/runs/current_llava"

python deliverables/pipeline/scripts/extract_mllm_features.py `
  --model-family llava_onevision `
  --metadata deliverables/benchmark/data/benchmark.csv `
  --corpus-root deliverables/benchmark `
  --output-dir "$run/features" `
  --model-id "models\huggingface\llava-onevision-qwen2-0.5b-ov-hf" `
  --runtime-model-id "models/huggingface/llava-onevision-qwen2-0.5b-ov-hf" `
  --model-revision c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c `
  --tokenizer-revision c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c `
  --local-files-only

python deliverables/pipeline/scripts/run_experiment.py `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --output-dir "$run/pipeline"

python deliverables/ablation_report/run_ablation.py `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --output-dir "$run/ablation"
```

For a current red-team run, generate a new panel, extract aligned features from that
panel, and score them with the tuned-v3 detector. The feature-bundle provenance must
match the detector before scoring is allowed. See `pipeline/README.md` and
`red_teaming/README.md` for the exact commands.

`benchmark/build_workbook.mjs` produces the Excel workbook and QA previews. The
canonical machine-readable benchmark is `benchmark/data/benchmark.csv`.

## Safety and disclosure

All authored adversarial examples use explicit redaction placeholders instead of
operational harmful content. Third-party dataset content and model weights are not
redistributed. See `benchmark/DATASHEET.md` and `red_teaming/SAFE_USE.md` before
extending the corpus.
