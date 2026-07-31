# Red-teaming tools

These tools generate deterministic, safety-redacted prompt variants and evaluate
bounded best-of-N detector-score evasion without storing operational harmful
instructions. No generated panel or detector score is committed.

## Generate a new panel

`generate_variants.py` transforms malicious benchmark rows into six fixed variants.
Image-overlay variants receive generated, safety-redacted cue-card PNGs. An explicit
output path is required so each run has a deliberate location.

```powershell
$run = "deliverables/runs/current_redteam"

python deliverables/red_teaming/generate_variants.py `
  --output "$run/variants.csv"
```

The generated manifest records the benchmark and output hashes, selected split,
variant ordering, and toolchain. Choose a new output path for each evidence-producing
run.

## Extract and score current features

The extractor accepts either `sample_id` or `variant_id`. For LLaVA it emits an
image-aligned metadata table alongside the provenance-complete feature bundle.

```powershell
python deliverables/pipeline/scripts/extract_mllm_features.py `
  --model-family llava_onevision `
  --metadata "$run/variants.csv" `
  --corpus-root "$run" `
  --output-dir "$run/features" `
  --model-id "models\huggingface\llava-onevision-qwen2-0.5b-ov-hf" `
  --runtime-model-id "models/huggingface/llava-onevision-qwen2-0.5b-ov-hf" `
  --model-revision c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c `
  --tokenizer-revision c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c `
  --local-files-only

python deliverables/pipeline/scripts/score_feature_bundle.py `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --detector models/aegis/aegis_llava_onevision_05b_text_detector_tuned_v3.npz `
  --output "$run/scored_variants.csv"
```

Scoring stops if the feature-bundle identifier column, row order, provenance, or
dimension is incompatible with the metadata or detector.

## Evaluate bounded best-of-N evasion

The evaluator selects the minimum score in hindsight among the first N fixed variants.
This is an oracle best-of-N analysis, not a sequential adversary that adapts later
queries from earlier responses.

```powershell
python deliverables/red_teaming/evaluate_adaptive.py `
  --scores "$run/scored_variants.csv" `
  --detector models/aegis/aegis_llava_onevision_05b_text_detector_tuned_v3.npz `
  --features "$run/features/feature_bundle.npz" `
  --threshold-kind detector_artifact_block_threshold `
  --traffic-mode shadow `
  --query-budgets 1 3 6 `
  --output "$run/best_of_n.json"
```

The evaluator requires the score table's ordered `variant_id` values to match the
feature bundle exactly. Its output records detector, score-panel, and feature-bundle
hashes together with threshold semantics and traffic mode. In shadow mode,
threshold-based blocking is counterfactual because the effective runtime action
remains `allow`.

## Evaluate guarded model responses

For response-level work, create paired `unguarded` and `guarded` adjudications using
`response_judgment_schema.json`, then run `evaluate_attack_success.py`. The schema
separates effective action, recommended action, traffic mode, and downstream
disposition.

See `SAFE_USE.md` before adding attack families or external red-team data.
