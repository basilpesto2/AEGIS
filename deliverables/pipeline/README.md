# Provenance-aware representation-space detector pipeline

This package extracts MLLM representations, trains lightweight safety detectors, and
scores aligned datasets. It supports representation, cross-modal consistency,
perturbation-attribution, uncertainty, low-label learning, balanced pseudo-labeling,
and validation-only threshold selection.

No generated detector results are committed. Every research run must use a new output
directory and a provenance-complete feature bundle.

## Feature bundle contract

An `.npz` bundle must contain:

- `sample_ids`: unique identifiers in exact metadata-row order.
- `text_embeddings`: finite `(n, d)` pooled hidden representations.
- `image_embeddings`: finite `(n, d)` pooled image-token representations; text-only
  rows use zero vectors.
- `attribution_features`: finite `(n, a)` aggregate attribution statistics.
- `feature_source`: a non-empty extractor/model description.
- `model_family`, `model_id`, `model_revision`, and `tokenizer_revision`.
- `layer`, `pooling`, `id_column`, and the 64-character
  `preprocessing_sha256`.

The authoritative logical contract is `schemas/feature_bundle.schema.json`.

## Extract current LLaVA features

The extraction wrapper requires exact model and tokenizer revisions. It emits
`feature_bundle.npz`, `feature_manifest.json`, and
`aligned_source_metadata.csv`. LLaVA output contains only rows with an image because
the adapter requires one image per request.

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
```

The root package with its `mllm` dependencies and a compatible PyTorch runtime is
required for extraction. The small research `requirements.txt` covers the
post-extraction analysis tools only.

## Train a research detector

```powershell
python deliverables/pipeline/scripts/run_experiment.py `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --output-dir "$run/pipeline"
```

Writers reject non-empty output locations unless `--force` is explicit. Prefer a new
run directory to replacing an existing result.

## Score with the deployed tuned-v3 detector

`score_feature_bundle.py` supports both research detectors containing
`metadata_json` and the deployed tuned-v3 artifact. For a deployed artifact with
`pooling=text_tokens`, it uses the bundle's `text_representation` view and verifies
model family, model ID, revisions, layer, pooling, preprocessing fingerprint, and
feature dimension before scoring.

```powershell
python deliverables/pipeline/scripts/score_feature_bundle.py `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --detector models/aegis/aegis_llava_onevision_05b_text_detector_tuned_v3.npz `
  --output "$run/scored.csv"
```

## Leakage controls

- Splits come from benchmark group IDs; paired benign/adversarial examples never
  cross splits.
- Feature-view and classifier settings are selected using training/validation rows
  only.
- Thresholds are selected on validation data before test scoring.
- The test split is used once for final aggregate metrics.
- Pseudo-labels are drawn only from the training split and balanced by predicted
  class.

## Interpretation

The uncertainty signal is normalized binary entropy of the detector score.
Attribution features are aggregate changes induced by documented input perturbations
or model-native summaries. Cross-modal consistency uses normalized text/image
representation agreement. Report these signals separately so a strong text
representation cannot conceal weak multimodal behavior.
