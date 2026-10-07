# Provenance-aware representation-space detector pipeline

This package extracts MLLM representations, trains lightweight safety detectors, and
scores aligned datasets. It supports representation, cross-modal consistency,
perturbation-attribution, uncertainty, low-label learning, balanced pseudo-labeling,
and validation-only threshold selection.

Generated intermediate detector results are not committed. Every research run must
use a new output directory and a provenance-complete feature bundle. Qualified v7
runtime packages under `models/aegis/` are the explicit exception: they retain the
artifacts, pair manifest, and bound qualification-evidence closure used to qualify
the published pair. Online AEGIS inference loads the packaged heads without running
these research validators.

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

## Durable Bordair detector workflow

The maintained Bordair corpus, training, evaluation, and promotion entry points live
in this tracked pipeline. Generated images, feature caches, candidate heads, and
evaluation evidence remain under the ignored `outputs/bordair_retraining_v1`
workspace.

The tracked code is not a self-contained copy of the Bordair data or model weights.
Before preparing a corpus from a clean checkout, restore the exact pinned inputs to
the following ignored locations:

- `outputs/bordair_eval/source/text_image_001.json` through
  `text_image_013.json` and `outputs/bordair_eval/cases.json`;
- `outputs/bordair_retraining_v1/source/bordair_benign_multimodal_text_image.json`;
- the immutable schema-3 reuse archive at
  `outputs/bordair_retraining_v1/features_v7_trials/schema3_pre_additional_hard_negatives/`,
  including its pinned `archive_manifest.json` and every file/tree it binds;
- the pinned LLaVA checkpoint or Qwen Hugging Face snapshot required by the selected
  target.

Preparation and validation fail closed when an input is absent or its SHA-256 does
not match the constants in the tracked corpus code. Obtain these inputs from the
authorised pinned dataset/model source; the workflow does not fabricate or silently
download substitutes. Rendering selects installed compatible TrueType fonts in a
deterministic preference order and records each selected font path and hash. Exact
byte-for-byte regeneration of an existing image corpus additionally requires the
same recorded font files.

The maintained entry points are `prepare_bordair_corpus.py`,
`validate_bordair_corpus.py`, `train_bordair_detector.py`, and
`validate_bordair_promotion.py`. Treat these names as a workflow map, not as host
Python setup instructions. Preparation, model-backed training, qualification, and
publication must be invoked through the inspected Docker image and, where model
loading is involved, the read-only Compose model-cache volume documented below.

Training and extraction intentionally use the canonical hard-wired
`outputs/bordair_retraining_v1` run root and do not accept `--run-root`. For
evaluation and both validators, `--run-root` is the single source of defaults
for the manifest, held-out metadata, feature cache, training summaries, and runtime
evidence. Corpus-input overrides cannot mix files from a different run tree; point
`--run-root` at that tree instead. The strict tracked corpus/disjointness validator
runs before any frozen-panel evaluation or promotion check.

Do not execute a second frozen evaluation after one-shot evidence has been consumed.
Promotion reparses the complete copied one-shot evidence closure and verifies all
bound hashes without invoking a model or reading feature-cache embeddings. The dual
frozen evaluator is also deliberately cache-only for a separately predeclared first
execution: it requires exactly 50 valid
pre-existing fused cache entries and fails before model inference or writes on any
miss, invalid entry, rebuild request, or disabled cache. Promotion additionally
requires a schema-2 deployment config that passes the live AEGIS config loader and
matches the target profile, both artifact bytes, per-head thresholds, fused provider
pooling/dimension, shared provenance, and runtime detector identity.

### Qwen v7 one-shot qualification (declare, then one atomic GPU run)

Any new Qwen pair must use the tracked one-shot orchestrator. Declaration must
run in the same image with the same read-only model-cache volume that will be used by
the atomic attempt. It validates the development pair and strict corpus, fixes the
ordered 50-case protocol, copies those metadata/images and the small training
evidence closure, hashes the exact pinned model/tokenizer snapshot, snapshots the
extractor/evaluator/validator implementation, and creates the sole append-only
subject claim. It does not extract or inspect frozen scores.

From the repository root in PowerShell, after `aegis-mllm-guard:local` and the Qwen
cache have been prepared, run exactly:

```powershell
$image = "aegis-mllm-guard:local"
$imageId = (docker image inspect --format '{{.Id}}' $image).Trim()
$project = (Resolve-Path ".").Path
$runRoot = "/workspace/outputs/bordair_retraining_v1"
$cacheVolumes = @(
  docker volume ls `
    --filter "label=com.docker.compose.volume=aegis_model_cache" `
    --format '{{.Name}}'
)
if ($cacheVolumes.Count -ne 1) {
  throw "Expected exactly one Compose aegis_model_cache volume; found $($cacheVolumes.Count)."
}
$cacheVolume = $cacheVolumes[0]

# Declare before any extraction from the frozen 50-case panel.
$declareJson = docker run --rm --gpus all `
  --entrypoint python `
  -e HF_HUB_OFFLINE=1 `
  -e TRANSFORMERS_OFFLINE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --mount "type=volume,source=$cacheVolume,target=/app/models/huggingface,readonly" `
  --workdir /workspace `
  $imageId `
  deliverables/pipeline/scripts/prepare_bordair_dual_qualification.py `
  declare qwen25vl3b `
  --run-root $runRoot `
  --model-cache-dir /app/models/huggingface `
  --runtime-image-id $imageId
$declared = $declareJson | ConvertFrom-Json
$subject = $declared.qualification_subject_sha256

# One process now performs extraction -> cache-only evaluation -> independent
# validation -> evidence packaging. There is no external features hand-off.
$consumeJson = docker run --rm --gpus all `
  --entrypoint python `
  -e HF_HUB_OFFLINE=1 `
  -e TRANSFORMERS_OFFLINE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --mount "type=volume,source=$cacheVolume,target=/app/models/huggingface,readonly" `
  --workdir /workspace `
  $imageId `
  deliverables/pipeline/scripts/prepare_bordair_dual_qualification.py `
  consume qwen25vl3b `
  --run-root $runRoot `
  --qualification-subject $subject
$consumed = $consumeJson | ConvertFrom-Json
$consumed
```

The canonical workspace is
`outputs/bordair_retraining_v1/training_v7/qwen25vl3b/dual_or/frozen_evaluation_once_generic_v4/<subject-sha256>`.
After a passing consume, bind the copied evidence into a regenerated byte-identical
pair in the same pinned Docker numerical and serialization runtime. The command
recomputes only the already-declared development training; it does not rescore the
frozen panels or retune from their results, and it fails if the head bytes differ
from the qualified pair:

```powershell
$workspaceContainer = (
  "$runRoot/training_v7/qwen25vl3b/dual_or/" +
  "frozen_evaluation_once_generic_v4/$subject"
)
docker run --rm --gpus all `
  --entrypoint python `
  -e HF_HUB_OFFLINE=1 `
  -e TRANSFORMERS_OFFLINE=1 `
  -e HF_HUB_DISABLE_TELEMETRY=1 `
  -e TOKENIZERS_PARALLELISM=false `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --mount "type=volume,source=$cacheVolume,target=/app/models/huggingface,readonly" `
  --workdir /workspace `
  $imageId `
  deliverables/pipeline/scripts/train_bordair_detector.py qwen25vl3b `
  --artifact-mode dual_or `
  --cache-dir /app/models/huggingface `
  --qwen-chunk-size 32 `
  --candidate-poolings text_tokens image_tokens `
  --benign-weight 1 `
  --counterfactual-paired-benign-weight 2 `
  --dual-hard-negative-weight 3.5 `
  --candidate-image-only-malicious-weights 0.75 1 1.1 1.25 1.5 2 3 4 `
  --candidate-text-led-malicious-weights 1 2 4 `
  --candidate-learning-rates 0.005 0.01 0.02 0.05 `
  --candidate-l2-values 0.0001 0.0003 0.001 0.01 0.1 `
  --dual-qualification-evidence-dir "$workspaceContainer/evidence"
```

Do not bind on a host with a different Python, NumPy, operating-system, or NPZ
serialization runtime. Even negligible floating-point or archive-metadata drift
changes the artifact bytes and must fail the byte-identical binding requirement.
Declaration, consumption, and binding therefore all use the exact inspected Docker
image and read-only cache volume shown above.

After binding succeeds, stop the AEGIS services and verify that the Qwen target
profile and schema-2 deployment config contain the exact pair-derived image/text
paths, SHA-256 values, review thresholds, fused `text_image_tokens` pooling, and
feature dimension. The publisher rejects any stale single-head or mismatched dual
profile/config before it changes the destination. Then publish and validate the
package:

```powershell
$destination = "/workspace/models/aegis/qwen25vl3b_v7_dual_or"
docker run --rm `
  --entrypoint python `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --workdir /workspace `
  $imageId `
  deliverables/pipeline/scripts/publish_bordair_dual_pair.py qwen25vl3b `
  --run-root $runRoot `
  --destination $destination `
  --runtime-config /workspace/configs/aegis.deployment.container.json
```

This command installs the pair manifest, both NPZ heads, and the complete bound
`qualification_evidence/` tree. The destination must be absent or already exactly
byte-identical; partial or divergent contents are never overwritten. A new package
is staged and renamed atomically, after which the existing promotion validator must
pass against the current schema-2 config, target profile, and published paths before
the command reports success. A failed post-publication validation atomically moves a
new package out of the runtime path to a sibling
`.qwen25vl3b_v7_dual_or.published_but_quarantined-*` directory and reports that
state; an existing byte-identical destination is never mutated on validation
failure. After an actual process or host crash, inspect and preserve any sibling
`.publish.lock`, `.stage-*`, or quarantine directory for diagnosis. Remove a stale
lock/stage only after confirming no publisher is running and no canonical
destination is partial. If quarantine itself fails, the command reports
`published_validation_failed_quarantine_failed` and intentionally retains both the
canonical path and `.publish.lock`; stop all services and resolve that state
manually before removing the lock.

The bundled Qwen v7 package is published and passed that post-publication validation.
Its pair manifest has SHA-256
`3045793da674120f5c46358380278ba65673b8b17e88c86ccb91c9afc59787ca`,
runtime detector identity
`960f53c6fa3e584bfd2ff16a6ec3cd2d54f592e640e32da55748c2b8facd3a8e`,
and pair identity
`7d44bf903ec880082aabc569b3468ef719746f87023422c0c721cf3463ff1174`.
Its `qualification_evidence/` tree contains 209 files, including exactly 50 frozen
cache entries. The pair manifest binds
`cache_entry_tree_sha256=f7577b99b3f9e4cfff888fd9fa10f337be9d4fd880eb80a0f73065bc4e5864b7`
and
`corpus_manifest_sha256=ffd961e91771ec95b2781a66c8992fd0edafe94dfaf61f5fe4062dde02127220`.
This cache is qualification evidence, not the Hugging Face model cache and not an
AEGIS runtime request cache. `promotion_authorized_by_evidence: false` is intentional:
evidence passing alone cannot authorize operator publication. For any future
package, claim runtime publication only after the publisher returns `passed: true`.

Publication does not require a GPU or the Hugging Face cache volume: it copies and
validates already-bound files. It must nevertheless use `$imageId` so numerical and
NPZ serialization behavior matches declaration, consumption, and binding.

For a new LLaVA qualification, declaration additionally requires the exact local
checkpoint path inside the same cache mount via
`--llava-runtime-model /app/models/huggingface/llava-onevision-qwen2-0.5b-ov-hf`.
The existing qualified LLaVA evidence is a preserved legacy package and must not be
rerun or rewritten.

The consume command creates its exclusive burn marker before loading the model or
writing any extracted feature. A crash, validation error, model-tree mutation, or
partial output permanently consumes that subject; do not delete the claim or retry
under another path. Restore the underlying defect and begin only with a genuinely
new pair/corpus/protocol subject. `--development-workspace` exists solely for
synthetic tests and always produces non-promotable evidence.

This is a fail-closed local ledger while its files and permissions are preserved,
not external attestation. The image ID is caller-inspected and recorded, not
independently proven by the Python process. Promotion binds the complete copied
code/input/model-manifest/runtime/results/validation closure and rejects any byte
substitution, but a privileged filesystem owner can rewrite local history; use an
external append-only or signed store if that adversary is in scope.

The thin scripts import their implementation from `aegis_research`; tests therefore
exercise tracked code and do not import Python files from the ignored output tree.

## Extract current dual-head features

The extraction wrapper requires exact model and tokenizer revisions. It emits
`feature_bundle.npz`, `feature_manifest.json`, and
`aligned_source_metadata.csv`. The current dual-head workflow requires
`--pooling text_image_tokens` for both targets. This preserves the text and image
primitive matrices in the bundle and declares their canonical text-then-image fused
provider representation. Each input row must contain an image and non-empty text;
fused extraction rejects text-only rows before model inference.

Resolve the immutable deployment image and the single prepared Compose model-cache
volume first. Model-backed extraction must use this pinned environment; an arbitrary
host Python environment is not an equivalent publication or reproduction runtime.

```powershell
$image = "aegis-mllm-guard:local"
$imageId = (docker image inspect --format '{{.Id}}' $image).Trim()
$project = (Resolve-Path ".").Path
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

The canonical benchmark has 64 rows, including 20 text-only rows. Define this
selection step once and run it for each target below. It writes the 44 image-text
rows into that target's new run directory in their original order, preserving
sample identifiers, all metadata columns, and image paths relative to
`deliverables/benchmark`. The canonical benchmark remains unchanged.

```powershell
$selectImageText = @'
import csv
import sys
from pathlib import Path

source = Path("deliverables/benchmark/data/benchmark.csv")
destination = Path(sys.argv[1])
if destination.exists():
    raise FileExistsError(f"choose a new run directory: {destination}")
with source.open(encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames
    rows = [row for row in reader if row["image_path"].strip()]
if not rows or any(not row["prompt_text"].strip() for row in rows):
    raise ValueError("image-text metadata requires non-empty prompt_text")
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"Selected {len(rows)} image-text rows -> {destination}")
'@
```

LLaVA dual-head extraction uses:

```powershell
$run = "/workspace/deliverables/runs/current_llava"

$selectImageText | docker run --rm -i `
  --entrypoint python `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --workdir /workspace `
  $imageId - "$run/input/benchmark_image_text.csv"
if ($LASTEXITCODE -ne 0) { throw "Image-text metadata selection failed." }

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
  --metadata "$run/input/benchmark_image_text.csv" `
  --corpus-root deliverables/benchmark `
  --output-dir "$run/features" `
  --model-id "models\huggingface\llava-onevision-qwen2-0.5b-ov-hf" `
  --runtime-model-id /app/models/huggingface/llava-onevision-qwen2-0.5b-ov-hf `
  --cache-dir /app/models/huggingface `
  --model-revision c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c `
  --tokenizer-revision c2cd35a65b8059c8add9e8901550c9e29d62d189cc7c2a5a1f6f715d7e05bb1c `
  --pooling text_image_tokens `
  --local-files-only
```

Qwen dual-head extraction uses:

```powershell
$run = "/workspace/deliverables/runs/current_qwen"

$selectImageText | docker run --rm -i `
  --entrypoint python `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --workdir /workspace `
  $imageId - "$run/input/benchmark_image_text.csv"
if ($LASTEXITCODE -ne 0) { throw "Image-text metadata selection failed." }

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
  --metadata "$run/input/benchmark_image_text.csv" `
  --corpus-root deliverables/benchmark `
  --output-dir "$run/features" `
  --model-id Qwen/Qwen2.5-VL-3B-Instruct `
  --cache-dir /app/models/huggingface `
  --model-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 `
  --tokenizer-revision 66285546d2b821cf421d4f5eb2576359d3770cd3 `
  --max-pixels 200704 `
  --pooling text_image_tokens `
  --local-files-only
```

The inspected deployment image supplies the `mllm`, CUDA/PyTorch, and Transformers
stack used for extraction. The small research `requirements.txt` covers only
post-extraction analysis and is not a substitute for that pinned model runtime.

## Train a research detector

```powershell
docker run --rm `
  --entrypoint python `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --workdir /workspace `
  $imageId `
  deliverables/pipeline/scripts/run_experiment.py `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --output-dir "$run/pipeline"
```

Writers reject non-empty output locations unless `--force` is explicit. Prefer a new
run directory to replacing an existing result.

## Score a current v7 dual-head pair

The maintained scoring mode consumes a qualified `detector_pair_manifest.json`.
The feature bundle must have `pooling=text_image_tokens` and contain aligned
`text_embeddings` and `image_embeddings` primitive views. The scorer validates
the pair's bound artifact bytes and qualification evidence, exact shared model
provenance, canonical text-then-image representation, and both primitive dimensions
before either head is scored.

```powershell
$run = "/workspace/deliverables/runs/current_llava"
$pair = "models/aegis/llava05b_v7_dual_or/detector_pair_manifest.json"

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
  --output "$run/scored.csv"
```

To score Qwen instead, select its feature run and pair manifest:

```powershell
$run = "/workspace/deliverables/runs/current_qwen"
$pair = "models/aegis/qwen25vl3b_v7_dual_or/detector_pair_manifest.json"

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
  --output "$run/scored.csv"
```

Each output row records the text and image scores, the artifact block threshold and
manifest review threshold for each head, each head's action, and the OR-composed
action. The top-level score and thresholds come from the decisive head: first by
action severity (`block` over `review` over `allow`), then by normalized progress
within that action band, then by head name for an exact tie. The pair and runtime
detector identities are also copied into every row.

Single-head scoring remains available only through the explicit `legacy_single`
label. The retired v6 detector files are absent from the current checkout. For
historical reproduction, restore them from the v6 revision without staging changes:

```powershell
git restore --source=2649dda --worktree -- `
  models/aegis/aegis_llava_onevision_05b_bordair_ocr_v6.npz `
  models/aegis/aegis_qwen25vl3b_bordair_ocr_v6.npz
```

Verify the restored hashes against
`deliverables/runtime_validation/v6_compose_validation_2026-08-29.json`. Both v6
heads require `image_tokens` provenance. Repeat the matching target's extraction
above with a new run directory, such as
`/workspace/deliverables/runs/legacy_llava_image_tokens`, and change only the
extraction's pooling argument to `--pooling image_tokens`. Do not pass a fused v7
feature bundle to a v6 head. Historical scoring evaluates the restored detector on
the supplied corpus; it does not recreate the observations in the archived Compose
record.

For LLaVA, score the separate image-token bundle:

```powershell
$run = "/workspace/deliverables/runs/legacy_llava_image_tokens"

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
  --detector models/aegis/aegis_llava_onevision_05b_bordair_ocr_v6.npz `
  --output "$run/legacy_single_scored.csv"
```

For Qwen, use a separate `legacy_qwen_image_tokens` run extracted with Qwen's
`--pooling image_tokens` configuration, then select
`models/aegis/aegis_qwen25vl3b_bordair_ocr_v6.npz` in `legacy_single` mode. These
historical paths do not represent either target's current v7 runtime.

In either mode, use only a feature bundle extracted with the matching model family,
model ID, revisions, layer, pooling, preprocessing fingerprint, and dimensions.

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
