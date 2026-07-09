# AEGIS: Multimodal Malicious Prompt Detection

AEGIS is the FYP codebase for detecting malicious or policy-violating prompts before a multimodal language model generates a response. It works by extracting hidden representations from an MLLM, training lightweight detectors on those representations, and evaluating them under leakage-controlled benchmark protocols.

## What AEGIS Provides

- A Python package and `aegis` CLI for scoring embeddings, training detector artifacts, extracting MLLM representations, and classifying new prompts.
- Dataset importers for VLGuard, MSSBench, and JailBreakV-28K into a shared metadata schema.
- Qwen2.5-VL and LLaVA-OneVision extraction adapters.
- SVD, supervised logistic, low-label logistic, trusted-plus-pseudo-label, transfer, robustness, signal-analysis, and guarded-response evaluation scripts.
- A default calibrated detector artifact for one-command text or image-text prompt classification.
- A model-agnostic guardrail runtime with embedding-provider contracts, calibrated policy files, monitoring summaries, and per-target validation bundles.
- Reproducible benchmark metadata, fixed seeds, validation-only model selection, and privacy-aware result artifacts.

## Repository Map

```text
AEGIS/                  Python package and CLI implementation.
AEGIS/adapters/         Qwen2.5-VL and LLaVA-OneVision embedding extractors.
AEGIS/datasets/         Dataset normalizers for VLGuard, MSSBench, JailBreakV.
scripts/                Reproducible import, split, extraction, evaluation, and reporting scripts.
data/external/          Downloaded third-party datasets. Ignored by git.
data/processed/         Normalized metadata and split CSVs. Ignored by git.
models/aegis/           Committed AEGIS detector artifacts for default classification.
models/huggingface/     Local Hugging Face model cache. Ignored by git.
outputs/                Embeddings, scores, reports, and presentation artifacts. Ignored by git.
```

All Git-visible source, scripts, tests, docs, configs, and default AEGIS detector artifacts are committed. Large datasets, third-party model weights, embedding arrays, and result artifacts remain intentionally ignored because of size and licensing; recreate those local artifacts with the commands below or place existing caches at the documented paths.

## Quick Start

Use these commands from the repository root on Windows PowerShell. On macOS or Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

1. Create the local environment used for MLLM work.

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install -e .
```

2. Install optional MLLM dependencies only when you need real model extraction or `aegis classify`.

```powershell
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -e ".[mllm]"
```

Choose a different PyTorch wheel if the machine does not use CUDA 12.8. The base package install does not require PyTorch or Transformers; those dependencies are only needed for real model extraction and `aegis classify`.

3. Check the CLI installation.

```powershell
& .\.venv\Scripts\aegis.exe --help
```

4. Classify a prompt with the default detector.

```powershell
& .\.venv\Scripts\aegis.exe classify `
  --text "Summarize the benefits of regular exercise."
```

The committed default detector artifact lives at `models\aegis\aegis_qwen25vl3b_text_detector.npz`. Classification still needs Qwen2.5-VL. On a fresh clone, the first run downloads the model into `models\huggingface` and can take several minutes plus several GB of disk space. Offline mode works only after the Hugging Face cache exists locally.

## Development Quality Gates

Install the lightweight development extras when changing code:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run the fast CPU-only checks:

```powershell
& .\.venv\Scripts\python.exe -m ruff check AEGIS scripts tests
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\python.exe scripts\run_smoke_tests.py
```

Inspect a detector artifact without loading model weights:

```powershell
& .\.venv\Scripts\aegis.exe inspect-artifact `
  --detector models\aegis\aegis_qwen25vl3b_text_detector.npz

& .\.venv\Scripts\aegis.exe doctor `
  --detector models\aegis\aegis_qwen25vl3b_text_detector.npz `
  --cache-dir models\huggingface
```

Score precomputed features from any LLM or MLLM embedding provider:

```powershell
& .\.venv\Scripts\aegis.exe guard-features `
  --features outputs\my_model_features.npz `
  --metadata data\processed\my_requests.csv `
  --detector models\aegis\my_detector.npz `
  --require-matching-provenance `
  --output outputs\my_guardrail_decisions.csv
```

Guard one live request through a provider object:

```powershell
& .\.venv\Scripts\aegis.exe guard-request `
  --provider my_package.my_provider:provider `
  --detector models\aegis\my_detector.npz `
  --text "User prompt to evaluate" `
  --request-id req-123 `
  --require-matching-provenance `
  --output outputs\my_guardrail_decision.json
```

The provider spec can be an object, no-argument class, or factory function that
exposes `model_family`, `model_id`, `pooling`, `feature_dim`, and
`embed(GuardrailRequest)`. The response stores prompt and optional image hashes,
not raw prompt text.

Regenerate the raw-prompt-free reproducibility manifest:

```powershell
& .\.venv\Scripts\python.exe scripts\generate_reproducibility_manifest.py
```

Run the controlled-deployment readiness gate for the current evidence bundle:

```powershell
& .\.venv\Scripts\python.exe scripts\build_production_evidence_report.py

& .\.venv\Scripts\python.exe scripts\validate_production_readiness.py `
  --summary-json outputs\aegis_classify_multimodal_litmus_v1_summary.json `
  --detector models\aegis\aegis_qwen25vl3b_text_detector.npz `
  --manifest docs\reproducibility_manifest.json `
  --scope controlled_deployment
```

See `docs\production_guardrail_playbook.md` and `configs\guardrail_policy.example.json` for the model-agnostic integration contract, deployment phases, logging policy, and release checklist.

Current controlled-deployment status: the registry contains two `production_candidate` target bundles: `docs\targets\qwen25vl3b_controlled.md` for `Qwen/Qwen2.5-VL-3B-Instruct`, and `docs\targets\llava_onevision_05b_controlled.md` for local LLaVA-OneVision 0.5B. The Qwen target has AUROC `0.98828`, AUPRC `0.99013`, precision `1.0`, recall `0.9375`, and false-positive rate `0.0` on the 32-case controlled litmus panel. The LLaVA target has AUROC `1.0`, AUPRC `1.0`, precision `1.0`, recall `0.98889`, and false-positive rate `0.0` on the held-out 180-row image-matched JailBreakV test split. Both provider contracts now record `image_text` request coverage, and the registry can be audited with `--required-provider-modality image_text` plus `--required-target-modality image_text`. Qwen now carries prompt robustness/adaptive evidence; LLaVA carries attack-family robustness and guarded response attack-success evidence. This makes AEGIS deployable as an input-side guardrail for those validated targets, not as a universal guardrail for every LLM/MLLM without target-specific validation; the strict universal portfolio audit still fails because the production-candidate portfolio needs a real text-only LLM target and Qwen-specific guarded attack-success evidence.

For production operation, calibrate thresholds from representative local traffic and monitor decisions after launch:

```powershell
& .\.venv\Scripts\python.exe scripts\calibrate_guardrail_policy.py `
  --features outputs\representative_features.npz `
  --metadata data\processed\representative_traffic.csv `
  --detector models\aegis\my_detector.npz `
  --output-policy configs\my_calibrated_policy.json

& .\.venv\Scripts\python.exe scripts\monitor_guardrail_decisions.py `
  --decisions outputs\my_guardrail_decisions.csv `
  --output outputs\my_guardrail_monitoring_summary.json
```

Serve live provider-backed decisions over HTTP:

```powershell
& .\.venv\Scripts\python.exe scripts\serve_guardrail_provider.py `
  --provider my_package.my_provider:provider `
  --detector models\aegis\my_detector.npz `
  --host 127.0.0.1 `
  --port 8766 `
  --require-matching-provenance
```

`POST /v1/guard` accepts either a single JSON request with `text`, `image_path`
or `image_paths`, and `request_id`, or a batch as `{"requests": [...]}`.

For each target LLM or MLLM, validate the embedding provider and build a target bundle:

```powershell
& .\.venv\Scripts\python.exe scripts\validate_embedding_provider_contract.py `
  --provider my_package.my_provider:provider `
  --detector models\aegis\my_detector.npz `
  --output outputs\my_provider_contract.json `
  --required-modality image_text

& .\.venv\Scripts\python.exe scripts\build_target_validation_bundle.py `
  --target-name my_target_model `
  --model-family custom `
  --model-id vendor/my-target-model `
  --intended-modality image_text `
  --detector models\aegis\my_detector.npz `
  --summary-json outputs\my_target_eval_summary.json `
  --policy configs\my_calibrated_policy.json `
  --provider-contract outputs\my_provider_contract.json `
  --calibration-summary outputs\my_calibration_summary.json `
  --monitoring-summary outputs\my_monitoring_summary.json `
  --robustness-summary outputs\my_prompt_robustness_summary.csv `
  --attack-success-summary outputs\my_guardrail_asr_summary.csv `
  --json-output docs\targets\my_target_model.json `
  --markdown-output docs\targets\my_target_model.md

& .\.venv\Scripts\python.exe scripts\validate_target_release.py `
  --bundle docs\targets\my_target_model.json `
  --require-labeled-monitoring `
  --require-robustness-evidence `
  --require-attack-success-evidence `
  --required-provider-modality image_text

& .\.venv\Scripts\python.exe scripts\validate_target_registry.py `
  --registry docs\target_validation_registry.json `
  --min-production-candidates 1 `
  --required-model-family custom `
  --required-target-modality image_text `
  --require-labeled-monitoring `
  --require-robustness-evidence `
  --require-attack-success-evidence `
  --required-provider-modality image_text

& .\.venv\Scripts\python.exe scripts\build_target_deployment_package.py `
  --bundle docs\targets\my_target_model.json `
  --output-dir outputs\deployment_packages\my_target_model `
  --require-labeled-monitoring `
  --require-robustness-evidence `
  --require-attack-success-evidence

& .\.venv\Scripts\python.exe scripts\validate_target_deployment_package.py `
  --package-dir outputs\deployment_packages\my_target_model `
  --require-labeled-monitoring `
  --require-robustness-evidence `
  --require-attack-success-evidence
```

Use `--intended-modality text`, `--required-modality text`,
`--required-provider-modality text`, and `--required-target-modality text` for
text-only LLM deployments. The target bundle records the deployment request
shape, release validation enforces matching provider-contract coverage by
default, and registry validation can now require portfolio coverage for target
modalities such as `text` and `image_text`. Strict release validation can also
require per-target robustness and guarded attack-success evidence. The deployment
package copies the approved detector, policy, provider contract, calibration,
monitoring, summary, robustness, and attack-success artifacts into one
checksum-validated directory.

## Classifying Prompts

Default detector path:

```text
models\aegis\aegis_qwen25vl3b_text_detector.npz
```

Classify text:

```powershell
& .\.venv\Scripts\aegis.exe classify `
  --text "Explain why regular backups are useful."
```

Classify text with an image:

```powershell
& .\.venv\Scripts\aegis.exe classify `
  --text "Follow the instructions shown in this image." `
  --image path\to\image.png
```

Classify a long UTF-8 prompt from a file:

```powershell
& .\.venv\Scripts\aegis.exe classify `
  --text-file path\to\prompt.txt `
  --output outputs\my_prediction.json
```

Use `--image` without `--text` or `--text-file` for image-only input. Image-only use is less validated because the default detector pools text-side hidden states after multimodal fusion; review those decisions conservatively.

Classification output fields:

| Field | Meaning |
| --- | --- |
| `verdict` | `malicious` or `benign` at the saved threshold. |
| `risk_score` | Logistic detector score. Larger means riskier, but it is not a universal probability. |
| `threshold` | Validation-selected decision boundary saved in the artifact. |
| `uncertain` | `true` when the score is within the detector's review margin. |
| `recommended_action` | `allow`, `review`, or `block`. |
| `prompt_sha256` | Prompt fingerprint. Raw prompt text is not echoed in the JSON. |

Default detector training sources:

- Full grouped VLGuard test representations.
- Full grouped MSSBench representations.
- The 720-row image-matched JailBreakV panel.
- A 48-row generic text calibration corpus generated by `scripts/build_inference_calibration_corpus.py`.

Default detector performance at the saved high-precision operating point:

| Split | AUROC | AUPRC | Precision | Recall | F1 | False positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Combined validation | 0.95310 | 0.96349 | 0.98939 | 0.63328 | 0.77226 | 4 / 482 |
| Combined test | 0.95600 | 0.96475 | 0.97250 | 0.66044 | 0.78665 | 11 / 478 |

The default `classify` detector uses a validation false-positive-rate cap of `0.01`, giving a saved threshold of `0.74176`. Local classification checks classify an ordinary exercise-summary request as benign at score `0.10011`, the placeholder prompt `Enter your prompt here` as benign at score `0.66753`, and a generic policy-evasion surrogate as malicious at score `0.88613`.

## Held-Out Multimodal Classify Litmus

Build a fresh image-text litmus panel for `aegis classify`:

```powershell
& .\.venv\Scripts\python.exe scripts\build_classify_litmus_panel.py
```

The builder writes generated image cards and metadata to `data\processed\aegis_classify_multimodal_litmus_v1`, rejects exact normalized prompt-text overlap with local AEGIS metadata references, and keeps the panel balanced between benign and malicious labels. The current panel source contains 32 image-text cases: 16 benign and 16 malicious, with additional natural benign safety/security prompts and more varied prompt-injection, privacy-exfiltration, fraud, and redacted harmful-planning surrogates. The prompts are newly authored redacted surrogates, inspired by public multimodal safety benchmark taxonomies such as MM-SafetyBench and SafeBench plus prompt-injection threat descriptions; no third-party raw harmful prompts are copied into the repo.

Run the default detector against the panel:

```powershell
& .\.venv\Scripts\python.exe scripts\run_classify_litmus.py `
  --refresh-embeddings `
  --require-perfect
```

The runner extracts Qwen2.5-VL text-token embeddings in one model load, downloading the model first if the local cache is empty. It scores the saved detector, writes raw-text-free per-sample results to `outputs\aegis_classify_multimodal_litmus_v1_scores.csv`, and writes aggregate metrics to `outputs\aegis_classify_multimodal_litmus_v1_summary.json`.

Current local result on July 9, 2026 for the expanded 32-case panel: 31 / 32 correct, 0 false positives, 1 false negative, 1 uncertain benign sample, AUROC `0.98828`, AUPRC `0.99013`, precision `1.0`, recall `0.9375`, F1 `0.96774`, threshold `0.74176`. The missed malicious sample is `litmus_malicious_instruction_hierarchy`, and the uncertain benign sample is `litmus_benign_chemistry_notice`. This is a stronger litmus than the earlier 16-case snapshot, but it is still narrow by design and should not be reported as proof of universal classifier safety.

## Rebuilding The Default Detector

Rebuild the generic text calibration corpus:

```powershell
& .\.venv\Scripts\python.exe scripts\build_inference_calibration_corpus.py
```

Extract its Qwen text-token embeddings:

```powershell
& .\.venv\Scripts\python.exe -m AEGIS.cli extract-qwen25-vl `
  --metadata data\processed\aegis_inference_calibration_v1_metadata.csv `
  --corpus-root data\processed `
  --output outputs\aegis_inference_calibration_v1_qwen25vl3b_text_tokens.npz `
  --model-id Qwen/Qwen2.5-VL-3B-Instruct `
  --cache-dir models\huggingface `
  --layer -1 `
  --pooling text_tokens `
  --max-pixels 200704
```

Train the four-source detector:

```powershell
& .\.venv\Scripts\aegis.exe train-detector `
  --embeddings `
    outputs\vlguard_test_full_qwen25vl3b_features\vlguard_test_full_qwen25vl3b_layer_m1_text_tokens.npz `
    outputs\mssbench_full_qwen25vl3b_layer_m1_text_tokens.npz `
    outputs\jailbreakv_eval_720_image_matched_controls_qwen25vl3b_layer_m1_text_tokens.npz `
    outputs\aegis_inference_calibration_v1_qwen25vl3b_text_tokens.npz `
  --metadata `
    data\processed\vlguard_test_full_groupsplit_metadata.csv `
    data\processed\mssbench_full_groupsplit_metadata.csv `
    data\processed\jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv `
    data\processed\aegis_inference_calibration_v1_metadata.csv `
  --output models\aegis\aegis_qwen25vl3b_text_detector.npz `
  --threshold-strategy max-fpr `
  --max-fpr 0.01 `
  --uncertainty-margin 0.05 `
  --source vlguard_mssbench_jailbreakv_generic_text_qwen25vl3b
```

The saved `.npz` artifact stores logistic weights, bias, normalization statistics, threshold, uncertainty margin, model ID, layer, pooling mode, source provenance, and an artifact version. Loading uses `allow_pickle=False`.

## Data Schema

Every importer writes CSV metadata centered on this schema:

| Column | Meaning |
| --- | --- |
| `sample_id` | Stable unique ID used to align metadata and embeddings. |
| `label` | `benign` or `malicious`; numeric `0` and `1` are also accepted by evaluators. |
| `text` | Prompt text. Avoid printing raw harmful rows in demos or public logs. |
| `image_path` | Path relative to the supplied `--corpus-root`. |
| `source` | Source dataset name. |
| `split` | Source-provided coarse split when available. |
| `experiment_split` | AEGIS `fit`, `val`, or `test` assignment for experiments. |
| `attack_style` | Normalized attack family or style. |
| `harm_category` | Normalized safety category. |
| `modality` | `text`, `image`, or `image_text`. |
| `matched_pair_id` | Pair ID for same-image malicious/benign controls. |
| `source_id` | Grouping key used to keep related rows in one split. |

Embedding files are `.npz` archives with:

- `embeddings`: float array with shape `(n_samples, embedding_dim)`.
- `sample_id`, `sample_ids`, or `ids`: optional string ID array with shape `(n_samples,)`.
- extraction metadata such as `model_id`, `layer`, and `pooling` when produced by AEGIS adapters.

AEGIS aligns embeddings and metadata by `sample_id` whenever IDs are present. Do not rely on row order alone.

## Dataset Setup

### VLGuard

VLGuard is gated on Hugging Face. Request access in the browser, create a read token, then authenticate locally:

```powershell
& .\.venv\Scripts\hf.exe auth login
& .\.venv\Scripts\hf.exe auth whoami
```

Do not paste tokens into chat, source files, or committed logs.

Download the files used by AEGIS:

```powershell
& .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ys-zong/VLGuard', repo_type='dataset', local_dir='data/external/VLGuard', allow_patterns=['train.json','test.json','train.zip','test.zip'])"
```

Unzip `train.zip` and `test.zip` under `data\external\VLGuard`, then normalize metadata:

```powershell
& .\.venv\Scripts\python.exe scripts\import_vlguard.py `
  --input-json data\external\VLGuard\test.json `
  --output-csv data\processed\vlguard_test_metadata.csv `
  --source-split test
```

Current local normalization:

| File | Rows | Label mix |
| --- | ---: | --- |
| `data\processed\vlguard_train_metadata.csv` | 2,977 | 977 benign, 2,000 malicious |
| `data\processed\vlguard_test_metadata.csv` | 1,558 | 558 benign, 1,000 malicious |

### MSSBench

Download and normalize MSSBench:

```powershell
& .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='kzhou35/mssbench', repo_type='dataset', local_dir='data/external/MSSBench', allow_patterns=['chat/**','embodied/**','combined.json'])"

& .\.venv\Scripts\python.exe scripts\import_mssbench.py `
  --input-json data\external\MSSBench\combined.json `
  --output-csv data\processed\mssbench_metadata.csv `
  --source-split test
```

Current local normalization:

| File | Rows | Label mix |
| --- | ---: | --- |
| `data\processed\mssbench_metadata.csv` | 1,960 | 980 benign, 980 malicious |
| `data\processed\mssbench_full_groupsplit_metadata.csv` | 1,960 | 980 fit, 490 validation, 490 test |

MSSBench is the main situational-safety and low-label benchmark. Image features are genuinely useful here because the safety label often depends on the visual situation.

### JailBreakV-28K

The public Hugging Face snapshot includes 28,000 malicious metadata rows but only a limited image package. The executable AEGIS panel uses the 360 malicious rows whose images are locally available, each paired with a neutral benign prompt using the identical image.

Download the public files used by AEGIS:

```powershell
& .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='JailbreakV-28K/JailBreakV-28k', repo_type='dataset', local_dir='data/external/JailBreakV-28K', allow_patterns=['JailBreakV_28K/JailBreakV_28K.csv','JailBreakV_28K/mini_JailBreakV_28K.csv','JailBreakV_28K/RedTeam_2K.csv','JailBreakV_28K/figstep/**','JailBreakV_28K/llm_transfer_attack/**','JailBreakV_28K/query_related/**'])"
```

Normalize, keep image-available rows, and build same-image controls:

```powershell
& .\.venv\Scripts\python.exe scripts\import_jailbreakv.py `
  --input-csv data\external\JailBreakV-28K\JailBreakV_28K\JailBreakV_28K.csv `
  --output-csv data\processed\jailbreakv_metadata.csv `
  --source-split test

& .\.venv\Scripts\python.exe scripts\filter_metadata_by_available_images.py `
  --input-csv data\processed\jailbreakv_metadata.csv `
  --output-csv data\processed\jailbreakv_image_available_metadata.csv `
  --corpus-root data\external `
  --image-column image_path

& .\.venv\Scripts\python.exe scripts\build_jailbreakv_matched_controls.py `
  --input-csv data\processed\jailbreakv_image_available_metadata.csv `
  --output-csv data\processed\jailbreakv_eval_720_image_matched_controls_metadata.csv `
  --corpus-root data\external `
  --image-prefix JailBreakV-28K

& .\.venv\Scripts\python.exe scripts\assign_metadata_splits.py `
  --input-csv data\processed\jailbreakv_eval_720_image_matched_controls_metadata.csv `
  --output-csv data\processed\jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv `
  --group-column label `
  --unit-column matched_pair_id `
  --split-column experiment_split `
  --split-fractions fit=0.5,val=0.25,test=0.25 `
  --seed 61
```

Current local files:

| File | Rows | Purpose |
| --- | ---: | --- |
| `data\processed\jailbreakv_metadata.csv` | 28,000 | Malicious metadata only. |
| `data\processed\jailbreakv_image_available_metadata.csv` | 360 | Public-image executable slice. |
| `data\processed\jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv` | 720 | Primary paired jailbreak benchmark. |

Because JailBreakV is malicious-only, binary metrics always depend on the disclosed benign controls. Do not present the 360-image slice as representative of all 28,000 examples.

## Embedding Extraction

Set offline mode after the model files are cached:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
```

Extract one Qwen2.5-VL view:

```powershell
& .\.venv\Scripts\python.exe -m AEGIS.cli extract-qwen25-vl `
  --metadata data\processed\vlguard_test_metadata.csv `
  --corpus-root data\external\VLGuard\test `
  --output outputs\vlguard_test_qwen25vl3b_text_tokens.npz `
  --model-id Qwen/Qwen2.5-VL-3B-Instruct `
  --cache-dir models\huggingface `
  --layer -1 `
  --pooling text_tokens `
  --max-pixels 200704
```

Extract Qwen text and image views in one model pass:

```powershell
& .\.venv\Scripts\python.exe -m AEGIS.cli extract-qwen25-vl-poolings `
  --metadata data\processed\vlguard_test_full_groupsplit_metadata.csv `
  --corpus-root data\external\VLGuard\test `
  --output-dir outputs\vlguard_test_full_qwen25vl3b_features `
  --output-prefix vlguard_test_full_qwen25vl3b `
  --poolings text_tokens image_tokens `
  --model-id Qwen/Qwen2.5-VL-3B-Instruct `
  --cache-dir models\huggingface `
  --layer -1 `
  --max-pixels 200704
```

Extract several Qwen layers:

```powershell
& .\.venv\Scripts\python.exe -m AEGIS.cli extract-qwen25-vl-layers `
  --metadata data\processed\vlguard_test_metadata.csv `
  --corpus-root data\external\VLGuard\test `
  --output-dir outputs\vlguard_test_qwen25vl3b_layers `
  --output-prefix vlguard_test_qwen25vl3b `
  --layers -1 -8 -16 -24 -32 `
  --model-id Qwen/Qwen2.5-VL-3B-Instruct `
  --cache-dir models\huggingface `
  --pooling mean_tokens `
  --max-pixels 200704
```

Extract LLaVA-OneVision text and image views:

```powershell
& .\.venv\Scripts\python.exe -m AEGIS.cli extract-llava-onevision-poolings `
  --metadata data\processed\jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv `
  --corpus-root data\external `
  --output-dir outputs\jailbreakv_eval_720_image_matched_controls_llava_onevision_05b `
  --output-prefix jailbreakv_eval_720_image_matched_controls_llava_onevision_05b `
  --poolings text_tokens image_tokens `
  --model-id models\huggingface\llava-onevision-qwen2-0.5b-ov-hf `
  --cache-dir models\huggingface `
  --layer -1 `
  --max-image-edge 384 `
  --batch-size 4 `
  --max-batch-characters 1200
```

Pooling modes:

| Pooling | Meaning |
| --- | --- |
| `mean_tokens` | Mean over all attended prompt tokens. Useful as an early baseline. |
| `text_tokens` | Prompt tokens excluding vision delimiters and image placeholder spans. Primary jailbreak feature. |
| `image_tokens` | Image placeholder token span. Useful for situational safety and ablations. |
| `last_token` | Final attended prompt token. Mostly for debugging. |

## Running Core Experiments

### Score Embeddings With SVD

```powershell
& .\.venv\Scripts\python.exe -m AEGIS.cli score `
  --embeddings outputs\vlguard_test_120_imbalanced_qwen25vl3b_mean_embeddings.npz `
  --metadata data\processed\vlguard_test_120_imbalanced_metadata.csv `
  --output outputs\vlguard_test_120_imbalanced_qwen25vl3b_mean_scores.csv `
  --components 4 `
  --malicious-prior 0.167
```

Run an SVD component and orientation grid:

```powershell
& .\.venv\Scripts\python.exe scripts\run_svd_ablation.py `
  --embeddings outputs\vlguard_test_120_imbalanced_qwen25vl3b_mean_embeddings.npz `
  --metadata data\processed\vlguard_test_120_imbalanced_metadata.csv `
  --output-csv outputs\vlguard_test_120_imbalanced_svd_ablation.csv `
  --components 1 2 4 8 16 32 `
  --orientations normal inverted `
  --malicious-prior 0.167
```

Use inverted orientation only as a validation-selected setting or ablation diagnostic.

### Train From SVD Pseudo-Labels

```powershell
& .\.venv\Scripts\python.exe -m AEGIS.cli pseudo-train `
  --embeddings outputs\vlguard_test_full_qwen25vl3b_features\vlguard_test_full_qwen25vl3b_layer_m1_text_tokens.npz `
  --metadata data\processed\vlguard_test_full_groupsplit_metadata.csv `
  --output outputs\vlguard_test_full_qwen25vl3b_pseudo_label_text_scores.csv `
  --split-column experiment_split `
  --components 8 `
  --malicious-prior 0.5 `
  --train-split fit `
  --test-split test `
  --epochs 300
```

### Run Low-Label Logistic Evaluation

```powershell
& .\.venv\Scripts\python.exe scripts\run_low_label_logistic.py `
  --embedding-glob "outputs\mssbench_full_qwen25vl3b_layer_m1_*_tokens.npz" `
  --metadata data\processed\mssbench_full_groupsplit_metadata.csv `
  --output-csv outputs\mssbench_full_qwen25vl3b_low_label_logistic.csv `
  --labels-per-class 5 10 20 40 `
  --label-seeds 101 102 103 104 105 `
  --include-concat `
  --learning-rates 0.01 `
  --l2-values 0.001 `
  --epochs-values 300 `
  --malicious-prior 0.5 `
  --selection-metric auprc
```

### Run Trusted-Plus-Pseudo Self-Training

```powershell
& .\.venv\Scripts\python.exe scripts\run_trusted_pseudo_logistic.py `
  --embedding-glob "outputs\mssbench_full_qwen25vl3b_layer_m1_*_tokens.npz" `
  --metadata data\processed\mssbench_full_groupsplit_metadata.csv `
  --output-csv outputs\mssbench_full_qwen25vl3b_trusted_pseudo_logistic_seed_sweep.csv `
  --labels-per-class 5 10 20 40 `
  --pseudo-multipliers 0 2 `
  --trusted-repeats 1 5 `
  --label-seeds 101 102 103 104 105 `
  --include-concat `
  --learning-rates 0.01 `
  --l2-values 0.001 `
  --epochs-values 300 `
  --malicious-prior 0.5 `
  --selection-metric auprc
```

### Run Signal Analysis

```powershell
& .\.venv\Scripts\python.exe scripts\run_signal_analysis.py `
  --text-embeddings outputs\mssbench_full_qwen25vl3b_layer_m1_text_tokens.npz `
  --image-embeddings outputs\mssbench_full_qwen25vl3b_layer_m1_image_tokens.npz `
  --metadata data\processed\mssbench_full_groupsplit_metadata.csv `
  --output-prefix outputs\mssbench_full_qwen25vl3b_signal_analysis `
  --malicious-prior 0.5
```

### Run Prompt Robustness

```powershell
& .\.venv\Scripts\python.exe scripts\build_prompt_robustness_variants.py `
  --input-csv data\processed\jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv `
  --output-csv data\processed\jailbreakv_eval_image_matched_test_robustness_variants_metadata.csv

& .\.venv\Scripts\python.exe scripts\run_prompt_robustness_evaluation.py `
  --original-embeddings outputs\jailbreakv_eval_720_image_matched_controls_qwen25vl3b_layer_m1_text_tokens.npz `
  --original-metadata data\processed\jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv `
  --transformed-embeddings outputs\jailbreakv_eval_image_matched_test_robustness_variants_qwen25vl3b_layer_m1_text_tokens.npz `
  --transformed-metadata data\processed\jailbreakv_eval_image_matched_test_robustness_variants_metadata.csv `
  --output-csv outputs\jailbreakv_eval_image_matched_prompt_robustness_evaluation.csv `
  --malicious-prior 0.5 `
  --selection-metric auprc
```

### Run Bounded Adaptive Prompt Evaluation

```powershell
& .\.venv\Scripts\python.exe scripts\run_adaptive_prompt_evaluation.py `
  --original-embeddings outputs\jailbreakv_eval_720_image_matched_controls_qwen25vl3b_layer_m1_text_tokens.npz `
  --original-metadata data\processed\jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv `
  --transformed-embeddings outputs\jailbreakv_eval_image_matched_test_adaptive_variants_qwen25vl3b_layer_m1_text_tokens.npz `
  --transformed-metadata data\processed\jailbreakv_eval_image_matched_test_adaptive_variants_metadata.csv `
  --output-summary-csv outputs\jailbreakv_eval_image_matched_adaptive_prompt_evaluation.csv `
  --output-selections-csv outputs\jailbreakv_eval_image_matched_adaptive_prompt_selections.csv `
  --malicious-prior 0.5
```

### Run Guarded Response ASR Proxy Evaluation

```powershell
& .\.venv\Scripts\python.exe scripts\run_guardrail_asr_evaluation.py `
  --embeddings outputs\jailbreakv_eval_720_image_matched_controls_llava_onevision_05b\jailbreakv_eval_720_image_matched_controls_llava_onevision_05b_layer_m1_text_tokens.npz `
  --metadata data\processed\jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv `
  --corpus-root data\external `
  --model-id models\huggingface\llava-onevision-qwen2-0.5b-ov-hf `
  --cache-dir models\huggingface `
  --output-samples-csv outputs\jailbreakv_eval_image_matched_llava_onevision_05b_guardrail_asr_samples.csv `
  --output-summary-csv outputs\jailbreakv_eval_image_matched_llava_onevision_05b_guardrail_asr_summary.csv `
  --max-new-tokens 64 `
  --max-image-edge 384 `
  --minimum-response-words 8 `
  --malicious-prior 0.5
```

The ASR sample audit stores stable IDs, family, detector scores, thresholds, block/refusal flags, response lengths, and SHA-256 hashes. It does not store raw prompts or generated response text.

## Evaluation Protocol

Primary protocols:

| Protocol | Data | Selection rule | Main report |
| --- | --- | --- | --- |
| Low-label detection | Full MSSBench, 1,960 rows | Validation AUPRC | Test AUROC, AUPRC, FPR95, precision, recall, F1; mean/std across label seeds |
| Jailbreak detection | 360 JailBreakV malicious rows plus 360 same-image controls | Validation only | Aggregate and family metrics |
| Source-disjoint transfer | Train on one source, test on a target that excludes that source's benign controls | Source validation or cross-source stability | Transfer AUROC/AUPRC with target prevalence disclosed |
| Attack-family holdout | Hold out FigStep, LLM-transfer, or query-related | Validation on remaining families | Mean/std by held-out family |
| Robustness | Original fit/validation, transformed malicious test prompts | Frozen original threshold | Metrics against unchanged benign controls |

Leakage controls:

1. Align metadata and embeddings by `sample_id`.
2. Keep repeated sources and matched image pairs in one split.
3. Fit normalization and detector parameters on fit rows only.
4. Select features, hyperparameters, pseudo-label thresholds, and classifier thresholds on validation rows only.
5. Keep test and transformed-test labels out of model selection.
6. Disclose when benign and malicious examples originate from different datasets.
7. Preserve command arguments, random seeds, model ID, layer, pooling mode, metadata path, and embedding path with every result.

Metric meanings:

| Metric | Use |
| --- | --- |
| AUROC | Ranking performance across thresholds. |
| AUPRC | Primary ranking metric when malicious prevalence may be imbalanced. |
| FPR95 | False-positive rate at 95 percent true-positive rate. |
| Precision, recall, F1 | Thresholded operating-point metrics selected on validation data. |
| ASR proxy | Response-level non-refusal proxy; not a human harmfulness judgment. |

## Headline Results

| Claim | Evidence |
| --- | --- |
| Direct weighted SVD is a useful no-label baseline but not the final detector. | VLGuard 480 held-out SVD reaches AUROC `0.5710`, AUPRC `0.2603`; full MSSBench SVD remains near chance. |
| Representations contain strong safety signal when labels are available. | Full MSSBench supervised concatenated features reach AUROC `0.97613`, AUPRC `0.97237`. |
| Few trusted labels are powerful. | Full VLGuard 5 labels/class reaches mean AUPRC `0.96755`; full MSSBench 40 labels/class reaches mean AUPRC `0.79904`. |
| Pseudo-labels are a refinement, not a replacement for trusted labels. | Trusted-plus-pseudo improves AUPRC by about `0.003` to `0.005` once at least 10 trusted labels/class are available; it hurts at 5 labels/class. |
| Transfer is asymmetric. | VLGuard to balanced JailBreakV plus MSSBench controls reaches AUPRC `0.91693`; VLGuard to MSSBench-240 reaches only AUPRC `0.44964`. |
| Cross-source stability helps when target labels are unavailable. | MSSBench to balanced JailBreakV plus VLGuard controls improves from concat AUPRC `0.60093` to text-token AUPRC `0.74268`. |
| Modality importance is task dependent. | MSSBench image-token AUPRC is `0.97090`; same-image JailBreakV image-token features fall near chance. |
| Fused text-token features are the primary jailbreak signal. | Image-matched attack-family holdout reaches text-token AUPRC `1.0` for FigStep, LLM-transfer, and query-related attacks. |
| The paired-panel result reproduces in a second MLLM. | LLaVA-OneVision reaches text-token AUPRC `1.0`, image-token AUPRC `0.51128`, consistency AUPRC `0.92161`, and direct SVD AUPRC `0.65684`. |
| Deterministic prompt transformations do not break the current paired detector. | Base64 indirection, leetspeak, and role-play wrappers retain AUROC/AUPRC `1.0` on the current paired test panel. |
| Bounded score-query adaptation exposes calibration risk. | Ten-query adaptive selection lowers malicious recall from `0.98889` to `0.96667`; AUROC/AUPRC remain `1.0`. |
| Input blocking reduces the documented response-level proxy. | On 90 malicious LLaVA test prompts, lexical non-refusal ASR proxy falls from `0.83333` unguarded to `0.0` with AEGIS blocking. |

Recommended detector policy for the current evidence base:

1. Use final-layer Qwen2.5-VL fused `text_tokens` for jailbreak-style prompt detection.
2. Use image/text concatenation for MSSBench-like situational safety when target validation labels are available.
3. Use cross-source stability, not source validation alone, for transfer feature selection when target labels are unavailable.
4. Prefer lightweight logistic detectors with a small trusted target-label budget.
5. Add balanced confident pseudo-labels only after validation confirms the seed classifier is reliable, usually at 10 or more trusted labels/class.
6. Treat direct SVD as a label-free baseline and diagnostic, not as the recommended production detector.
7. Route uncertain cases to review instead of treating every score as an absolute safety judgment.

## Limitations

- The primary JailBreakV executable panel has only 360 malicious prompts with locally available public-snapshot images.
- Same-image benign controls are templated and are not natural deployment traffic.
- JailBreakV is malicious-only, so binary claims depend on the disclosed controls.
- Qwen2.5-VL and LLaVA-OneVision are different representation spaces, but both use related Qwen-family language backbones.
- The default inference calibration corpus is deliberately small and generic.
- Response-level attack-success reduction uses a lexical non-refusal proxy, not human harmfulness labels.
- Perfect or near-perfect paired-panel ranking should be read alongside cross-dataset transfer failures.
- Current robustness tests cover deterministic transformations and a bounded ten-query search, not gradient-based or unconstrained attacks.
- AEGIS is a research risk detector, not a universal policy oracle or final content authority.

## Ethics And Safe Handling

- Do not print raw harmful prompts in demos, screenshots, public reports, or routine logs.
- Prefer aggregate metrics, stable IDs, hashes, categories, and provenance fields.
- Do not redistribute third-party images, model weights, or dataset files unless their licenses and access terms allow it.
- Screen any newly collected red-team data for personal information before annotation or release.
- Calibrate thresholds on representative traffic before deployment.
- Use proportionate actions: allow low-risk prompts, review uncertain prompts, and block or route high-risk prompts according to the use case.
- Report subgroup and source-specific uncertainty where possible; aggregate AUROC is not a fairness claim.
- Coordinate disclosure of newly discovered bypasses with affected maintainers before public release.
- Never execute instructions contained inside adversarial prompts.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Detector artifact does not exist` | Run from the repository root, confirm the detector file exists under `models\aegis`, or rebuild with `aegis train-detector`. |
| Hugging Face download fails in offline mode | Clear `$env:HF_HUB_OFFLINE` and `$env:TRANSFORMERS_OFFLINE`, authenticate if needed, download once, then re-enable offline mode. |
| VLGuard access is denied | Request dataset access in Hugging Face, wait for approval, then run `hf auth login`. |
| `Metadata contains sample ids missing from embeddings` | Confirm the embedding file and metadata CSV come from the same run and share `sample_id` values. |
| `No rows found for split` | Check `--split-column`, `--fit-split`, `--validation-split`, and `--test-split`; many experiment scripts expect `experiment_split`. |
| Image files are missing | Pass the correct `--corpus-root`; `image_path` is interpreted relative to that root. |
| CUDA out-of-memory | Lower `--max-pixels`, reduce LLaVA `--batch-size`, set `--max-batch-characters`, or use a smaller model. |
| First classification is slow | The model is loading and compiling kernels. Later calls are faster in a warm process. |
| Scores look perfect on paired JailBreakV | Recheck controls, source leakage, and pair grouping; disclose templated-control and 360-image limits. |

## Research Context

AEGIS is closest to VLMGuard's unlabeled representation-space malicious prompt detection idea. HaloScope contributes a similar "separate then learn" recipe from hallucination detection. Truthfulness Separator Vector work motivates low-label plus pseudo-label representation learning. Dual-Align and related calibration work motivate layer, uncertainty, and process-signal checks. EMI-style multimodal distribution-shift analysis motivates separate text, image, consistency, and transfer evaluations.

The practical lesson from the project is deliberately modest: hidden representations do contain useful prompt-safety signal, but the reliable guardrail is not a single unsupervised SVD direction. The strongest current path is small trusted-label adaptation, careful validation, cross-source feature selection, uncertainty-aware review, and explicit disclosure of dataset and transfer limits.
