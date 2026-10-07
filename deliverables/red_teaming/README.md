# Red-teaming tools

These tools generate deterministic, safety-redacted prompt variants and evaluate
bounded best-of-N detector-score evasion without storing operational harmful
instructions. Generated panels and detector scores are run evidence and are not
committed.

Each current target evaluates its model-specific v7 detector pair: one text-token
head and one image-token head applied to a single fused `text_image_tokens` feature
extraction. The runtime combines their recommended actions with OR precedence:
`block > review > allow`.

| Target | Runtime detector identity | Pair identity | Pair-manifest SHA-256 |
| --- | --- | --- | --- |
| `llava05b` | `25cebcf1e91125f4823176111d88cddae7379a0f70a1d8cc8af5ff07e22f08f6` | `358750a038e8f28f1b1c4250ed7a6d879b81c16e70c78bc423cfb421586c3d2b` | `5385783a98c8a4ce909efb3321a8732bf2d71a3e286ea61a3897ad763e1e2c68` |
| `qwen25vl3b` | `960f53c6fa3e584bfd2ff16a6ec3cd2d54f592e640e32da55748c2b8facd3a8e` | `7d44bf903ec880082aabc569b3468ef719746f87023422c0c721cf3463ff1174` | `3045793da674120f5c46358380278ba65673b8b17e88c86ccb91c9afc59787ca` |

## Generate a new panel

`generate_variants.py` transforms malicious benchmark rows into six fixed variants.
Image-overlay variants receive generated, safety-redacted cue-card PNGs. Use a new
output directory for every evidence-producing run.

Resolve the immutable deployment image first. Choose one target-specific run path;
use a different path when evaluating the other target.

```powershell
$image = "aegis-mllm-guard:local"
$imageId = (docker image inspect --format '{{.Id}}' $image).Trim()
$project = (Resolve-Path ".").Path

# LLaVA. For Qwen, use /workspace/deliverables/runs/current_redteam_qwen instead.
$run = "/workspace/deliverables/runs/current_redteam_llava"

docker run --rm `
  --entrypoint python `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --workdir /workspace `
  $imageId `
  deliverables/red_teaming/generate_variants.py `
  --output "$run/variants.csv"
```

The generated manifest records the benchmark and output hashes, selected split,
variant ordering, and toolchain.

## Extract fused v7 features

Extract both primitive views in one model forward. The bundle must be keyed by
`variant_id`, declare `pooling=text_image_tokens`, and contain aligned
`text_embeddings` and `image_embeddings` arrays. Extraction must use the same
inspected image ID and the single prepared Compose model-cache volume, not an
arbitrary host Python environment:

```powershell
$cacheVolumes = @(
  docker volume ls `
    --filter "label=com.docker.compose.volume=aegis_model_cache" `
    --format '{{.Name}}'
)
if ($cacheVolumes.Count -ne 1) {
  throw "Expected exactly one Compose aegis_model_cache volume; found $($cacheVolumes.Count)."
}
$cacheVolume = $cacheVolumes[0]
```

For LLaVA:

```powershell
docker run --rm --gpus all `
  --entrypoint python `
  -e HF_HUB_OFFLINE=1 `
  -e TRANSFORMERS_OFFLINE=1 `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --mount "type=volume,source=$cacheVolume,target=/app/models/huggingface,readonly" `
  --workdir /workspace `
  $imageId `
  deliverables/pipeline/scripts/extract_mllm_features.py `
  --model-family llava_onevision `
  --metadata "$run/variants.csv" `
  --corpus-root "$run" `
  --output-dir "$run/features" `
  --model-id "models\huggingface\llava-onevision-qwen2-0.5b-ov-hf" `
  --runtime-model-id /app/models/huggingface/llava-onevision-qwen2-0.5b-ov-hf `
  --cache-dir /app/models/huggingface `
  --model-revision c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c `
  --tokenizer-revision c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c `
  --pooling text_image_tokens `
  --local-files-only
```

For Qwen, use its pinned model revision and runtime pixel limit:

```powershell
# Set $run to /workspace/deliverables/runs/current_redteam_qwen and rerun the
# variant-generation command before extracting Qwen features.
docker run --rm --gpus all `
  --entrypoint python `
  -e HF_HUB_OFFLINE=1 `
  -e TRANSFORMERS_OFFLINE=1 `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --mount "type=volume,source=$cacheVolume,target=/app/models/huggingface,readonly" `
  --workdir /workspace `
  $imageId `
  deliverables/pipeline/scripts/extract_mllm_features.py `
  --model-family qwen25_vl `
  --metadata "$run/variants.csv" `
  --corpus-root "$run" `
  --output-dir "$run/features" `
  --model-id Qwen/Qwen2.5-VL-3B-Instruct `
  --cache-dir /app/models/huggingface `
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 `
  --tokenizer-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 `
  --max-pixels 200704 `
  --pooling text_image_tokens `
  --local-files-only
```

The extractor writes `aligned_source_metadata.csv` and `feature_bundle.npz` under
`$run/features`. Use a separate run directory for each target.

## Score the v7 detector pair

Use the pair manifest, not either head in isolation. The scorer verifies the pair
identity, both artifact hashes and thresholds, fused feature layout, shared model
provenance, primitive dimensions, row identifiers, and row order.

```powershell
# LLaVA
$pair = "models/aegis/llava05b_v7_dual_or/detector_pair_manifest.json"

# For Qwen, use this value instead:
# $pair = "models/aegis/qwen25vl3b_v7_dual_or/detector_pair_manifest.json"

docker run --rm `
  --entrypoint python `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --workdir /workspace `
  $imageId `
  deliverables/pipeline/scripts/score_feature_bundle.py `
  --mode dual_or `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --pair-manifest $pair `
  --output "$run/scored_variants.csv"
```

Each score row records both head scores, block and review thresholds, head actions,
the OR-composed recommended action, and the decisive head. It also binds the runtime
detector identity and richer pair identity.

## Evaluate bounded best-of-N evasion

The evaluator independently reloads and validates the pair, then recomputes both
head scores from the two primitive views. It rejects a score table if any recorded
score, threshold, action, decisive head, identity, feature provenance, identifier,
or row position differs from the recomputed result.

```powershell
docker run --rm `
  --entrypoint python `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --workdir /workspace `
  $imageId `
  deliverables/red_teaming/evaluate_adaptive.py `
  --mode dual_or `
  --scores "$run/scored_variants.csv" `
  --features "$run/features/feature_bundle.npz" `
  --pair-manifest $pair `
  --traffic-mode shadow `
  --query-budgets 1 3 6 `
  --output "$run/best_of_n.json"
```

For each query budget, the oracle considers only the first N fixed variants and
selects the least restrictive composed action in hindsight. Ties use normalized
progress within the decisive head's current action band, then the earliest query.
This avoids directly comparing raw text-head and image-head scores, whose scales and
thresholds differ.

The schema-version 3 output reports OR block recall and evasion rate, review-or-block
recall, action rates and counts, per-head block recall, decisive-head counts, and each
selected variant's nested head decisions. Its provenance binds the score table,
feature bundle, pair manifest, both artifacts, both detector identities, target, and
traffic mode. These are recommended-action metrics. In `shadow` mode, block and
review remain counterfactual because the effective runtime action is `allow`.

## Select the matching v7 target

Use the LLaVA pair only with LLaVA features and the Qwen pair only with Qwen features.
The commands are otherwise identical. Never evaluate a LLaVA bundle with a Qwen
artifact, or vice versa; the provenance checks reject a mismatch in model family,
model ID, revision, preprocessing, feature layout, dimensions, or artifact identity.

## Historical single-head mode

`legacy_single` is retained only for reproducing historical single-artifact
experiments, including v6. It is not representative of either current v7 runtime.
The v6 detector files were retired from the current checkout. Restore the historical
artifacts from the v6 revision before scoring:

```powershell
git restore --source=2649dda --worktree -- `
  models/aegis/aegis_llava_onevision_05b_bordair_ocr_v6.npz `
  models/aegis/aegis_qwen25vl3b_bordair_ocr_v6.npz
```

Check their hashes against the historical runtime-validation record as described in
the [pipeline reproduction instructions](../pipeline/README.md). Both v6 heads
require `image_tokens` provenance. Use a new target-specific run directory, such as
`/workspace/deliverables/runs/legacy_redteam_llava_image_tokens` or
`/workspace/deliverables/runs/legacy_redteam_qwen_image_tokens`. Set `$run` to that
directory, repeat the panel-generation command above, then repeat the matching
target's extraction with `--pooling image_tokens` instead of `text_image_tokens`.
Keep the matching model identifier, pinned revisions, and preprocessing settings.
Score this separate single-view bundle; the fused v7 bundle is incompatible with the
historical artifact.

```powershell
# Historical LLaVA v6. For Qwen v6, use
# legacy_redteam_qwen_image_tokens and the Qwen artifact instead.
$run = "/workspace/deliverables/runs/legacy_redteam_llava_image_tokens"
$legacyDetector = "models/aegis/aegis_llava_onevision_05b_bordair_ocr_v6.npz"
# Qwen alternative:
# $legacyDetector = "models/aegis/aegis_qwen25vl3b_bordair_ocr_v6.npz"

docker run --rm `
  --entrypoint python `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --workdir /workspace `
  $imageId `
  deliverables/pipeline/scripts/score_feature_bundle.py `
  --mode legacy_single `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --detector $legacyDetector `
  --output "$run/legacy_scored_variants.csv"

docker run --rm `
  --entrypoint python `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --workdir /workspace `
  $imageId `
  deliverables/red_teaming/evaluate_adaptive.py `
  --mode legacy_single `
  --scores "$run/legacy_scored_variants.csv" `
  --features "$run/features/feature_bundle.npz" `
  --detector $legacyDetector `
  --threshold-kind detector_artifact_block_threshold `
  --traffic-mode shadow `
  --query-budgets 1 3 6 `
  --output "$run/legacy_best_of_n.json"
```

## Evaluate guarded model responses

For response-level work, create paired `unguarded` and `guarded` adjudications using
`response_judgment_schema.json`, then run `evaluate_attack_success.py`. The schema
separates effective action, recommended action, traffic mode, and downstream
disposition.

See `SAFE_USE.md` before adding attack families or external red-team data.
