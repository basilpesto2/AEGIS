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
python deliverables/reproducible_pipeline/scripts/build_smoke_features.py
python deliverables/reproducible_pipeline/scripts/run_experiment.py `
  --metadata deliverables/annotated_benchmark/data/benchmark.csv `
  --features deliverables/reproducible_pipeline/fixtures/smoke_features.npz `
  --output-dir deliverables/reproducible_pipeline/artifacts/smoke
```

For a real experiment, replace the smoke bundle with hidden states exported from an
MLLM and record model ID, revision, layer, pooling, preprocessing hash, and extraction
command in the accompanying manifest.

The repository's Qwen2.5-VL and LLaVA-OneVision adapters are wrapped by
`scripts/extract_mllm_features.py`. The wrapper extracts text/image token pools and a
fixed masked-perturbation attribution summary into the same bundle contract:

```powershell
python deliverables/reproducible_pipeline/scripts/extract_mllm_features.py `
  --model-family qwen25_vl `
  --metadata deliverables/annotated_benchmark/data/benchmark.csv `
  --corpus-root deliverables/annotated_benchmark `
  --output-dir deliverables/reproducible_pipeline/artifacts/qwen_run `
  --model-revision YOUR_PINNED_REVISION `
  --tokenizer-revision YOUR_PINNED_REVISION
```

Model downloads and GPU execution are intentionally not triggered by the committed
smoke workflow. LLaVA extraction emits an aligned image-text subset because that
adapter requires one image per row.

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
