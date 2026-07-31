# Reproducible representation-space detector pipeline

This package trains lightweight safety detectors over precomputed MLLM feature
bundles. It supports representation, cross-modal consistency, perturbation-attribution,
and uncertainty signals; validation-only threshold selection; low-label seed sets;
balanced pseudo-labeling; and aggregate evaluation.

## Feature bundle contract

An `.npz` bundle must contain:

- `sample_ids`: unique string identifiers aligned to benchmark rows.
- `text_embeddings`: `(n, d)` pooled hidden representations.
- `image_embeddings`: `(n, d)` pooled image-token representations. Text-only rows use zero vectors.
- `attribution_features`: `(n, a)` safe aggregate attribution statistics, never raw gradients or prompts.
- `feature_source`: one string such as `qwen25_vl_hidden_states` or `deterministic_smoke_fixture`.

The authoritative JSON contract is `schemas/feature_bundle.schema.json`.

## Run

```powershell
$run = "deliverables/reproductions/smoke_v1"

python deliverables/pipeline/scripts/build_smoke_features.py `
  --output "$run/benchmark_smoke_features.npz"
python deliverables/pipeline/scripts/run_experiment.py `
  --metadata deliverables/benchmark/data/benchmark.csv `
  --features "$run/benchmark_smoke_features.npz" `
  --output-dir "$run/pipeline"
```

The committed files under `fixtures/` and `artifacts/smoke/` are frozen historical
evidence. Writers refuse to replace existing outputs unless `--force` is explicitly
supplied; normal reproductions should always use a new versioned directory.
`requirements.txt` pins the NumPy and Pillow versions used by the 2026-07-31
reproduction check; new manifests also record the active Python and library versions.

For a real experiment, replace the smoke bundle with hidden states exported from an
MLLM and record model ID, revision, layer, pooling, preprocessing hash, and extraction
command in the accompanying manifest.

The repository's Qwen2.5-VL and LLaVA-OneVision adapters are wrapped by
`scripts/extract_mllm_features.py`. The wrapper extracts text/image token pools and a
fixed masked-perturbation attribution summary into the same bundle contract:

Real extraction runs from the repository checkout and additionally requires the root
AEGIS package with its `mllm` optional dependencies, plus the target's compatible
PyTorch/CUDA installation. The small research `requirements.txt` alone is sufficient
only for deterministic fixture reproduction. Both model and tokenizer revisions are
mandatory for an MLLM extraction.

```powershell
$run = "deliverables/reproductions/llava_v1"

python deliverables/pipeline/scripts/extract_mllm_features.py `
  --model-family llava_onevision `
  --metadata deliverables/benchmark/data/benchmark.csv `
  --corpus-root deliverables/benchmark `
  --output-dir "$run/features" `
  --model-id "models\huggingface\llava-onevision-qwen2-0.5b-ov-hf" `
  --runtime-model-id models/huggingface/llava-onevision-qwen2-0.5b-ov-hf `
  --model-revision c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c `
  --tokenizer-revision c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c `
  --local-files-only
```

Model downloads and GPU execution are intentionally not triggered by the committed
smoke workflow. LLaVA extraction emits an aligned image-text subset because that
adapter requires one image per row.

## Relationship to the deployed detector

The committed `artifacts/smoke/selected_detector.npz` is trained from the deterministic
fixture and has an 8-dimensional attribution feature view. It is not copied into
`models/aegis/` and is not used by the service.

The current LLaVA service uses the 896-dimensional `text_tokens` artifact
`models/aegis/aegis_llava_onevision_05b_text_detector_tuned_v3.npz`. Its immutable
model, tokenizer, and preprocessing provenance is checked during readiness. New
research runs should record the same fields before their metrics are compared with the
runtime detector.

## Leakage controls

- Splits come from benchmark group IDs; paired benign/adversarial examples never cross splits.
- Feature-view and classifier settings are selected using training/validation rows only.
- Thresholds are selected on validation data and frozen before test scoring.
- The test split is used once for final aggregate metrics.
- Pseudo-labels are drawn only from the training split and balanced by predicted class.

## Interpretation

The uncertainty signal is normalized binary entropy of the detector score. Attribution
features are aggregate changes induced by documented input perturbations or model-native
attribution summaries. Cross-modal consistency uses normalized text/image representation
agreement. These signals should be reported separately so a strong text representation
cannot conceal weak multimodal behavior.
