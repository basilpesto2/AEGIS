# AEGIS research and validation deliverables

This directory contains the material that represents AEGIS `0.3.0`.
It is separate from the files required to install and run the CLI, service, or GUI.
Current datasets and reusable evaluation tools are retained together with the
current partial v7 runtime-validation record and the historical v6 record.

| Subdirectory | Deliverable | Evidence boundary |
| --- | --- | --- |
| `benchmark/` | Balanced, synthetic, safety-redacted multimodal benchmark; annotations, schema, images, CSV/JSON data, and Excel workbook. | Dataset and annotation evidence; not a production-safety claim. |
| `pipeline/` | Provenance-aware MLLM feature extraction, detector training, scoring, and unit tests. | Results are reportable only when generated from a current, provenance-complete feature bundle. |
| `red_teaming/` | Safety-redacted variant generation and bounded detector-evasion evaluation. | Tools only; no detector robustness result is committed. |
| `ablation_report/` | Executable signal-importance, low-label, uncertainty, and transfer-analysis workflow. | Tools only; reports must be generated from an explicitly supplied current feature bundle. |
| `runtime_validation/` | Current 2026-08-30 v7 and historical 2026-08-29 v6 Compose/CUDA validation records. | Functional validation only; the v7 record is partial, the v6 record is superseded, and neither is an accuracy, robustness, latency, or capacity benchmark. |
| `ci/` | Root product regression tests, research-tool unit tests, and deliverable-integrity verification. | Functional and artifact-integrity checks; not a model-accuracy benchmark. |

## Current runtime relationship

The two built-in targets accept only image-text requests with non-empty text and
exactly one image. Both use model-specific v7 dual-head detector pairs:

| Target / head | Artifact | Pooling | Dimension | Block threshold | Review threshold |
| --- | --- | --- | ---: | ---: | ---: |
| `llava05b` v7 text | `models/aegis/llava05b_v7_dual_or/llava05b_text_head_v1.npz` | `text_tokens` | 896 | `0.9973003604375604` | `0.9773003604375604` |
| `llava05b` v7 image | `models/aegis/llava05b_v7_dual_or/llava05b_image_head_v1.npz` | `image_tokens` | 896 | `0.43109073768976996` | `0.41109073768976995` |
| `qwen25vl3b` v7 text | `models/aegis/qwen25vl3b_v7_dual_or/qwen25vl3b_text_head_v1.npz` | `text_tokens` | 2048 | `0.9975257227486033` | `0.9775257227486033` |
| `qwen25vl3b` v7 image | `models/aegis/qwen25vl3b_v7_dual_or/qwen25vl3b_image_head_v1.npz` | `image_tokens` | 2048 | `0.1690249723273814` | `0.1490249723273814` |

| Target | Runtime detector identity | Pair identity | Pair-manifest SHA-256 |
| --- | --- | --- | --- |
| `llava05b` | `25cebcf1e91125f4823176111d88cddae7379a0f70a1d8cc8af5ff07e22f08f6` | `358750a038e8f28f1b1c4250ed7a6d879b81c16e70c78bc423cfb421586c3d2b` | `5385783a98c8a4ce909efb3321a8732bf2d71a3e286ea61a3897ad763e1e2c68` |
| `qwen25vl3b` | `960f53c6fa3e584bfd2ff16a6ec3cd2d54f592e640e32da55748c2b8facd3a8e` | `7d44bf903ec880082aabc569b3468ef719746f87023422c0c721cf3463ff1174` | `3045793da674120f5c46358380278ba65673b8b17e88c86ccb91c9afc59787ca` |

All heads use layer `-1`. Each target performs one fused `text_image_tokens`
extraction, splits it into the two primitive views, and combines head recommendations
with OR precedence (`block > review > allow`). For both LLaVA and Qwen, the dual-head
internal test blocked all 100 attacks and falsely blocked 1 of 80 benign cases. Each
frozen fixed panel blocked 10 of 10 malicious cases and 2 of 10 benign controls, each
text-led panel blocked all 10 attacks, and each external-benign panel blocked 2 of 20
controls. These detectors were developed on neutrally rendered Bordair OCR cases;
the synthetic benchmark in this directory is a separate pipeline and annotation
asset, not a production-safety benchmark.

The current `runtime_validation/v7_compose_validation_2026-08-30.json` record has
`partial` status. LLaVA v7 passed build/start, CUDA loading, warmup, authenticated
status with runtime detector identity
`25cebcf1e91125f4823176111d88cddae7379a0f70a1d8cc8af5ff07e22f08f6`,
one benign and one malicious fixed-panel dashboard inference, and graceful shutdown.
The malicious case recommended `block` but remained effectively `allow` in shadow
mode. Qwen's image was built and its detector package was present, but the unchanged
preflight failed closed before model loading on total RAM (8,172,244,992 observed versus
16,106,127,360 required) and free VRAM (7,440,695,296 versus 7,516,192,768).
Qwen load, warmup, and inference remain pending. This is functional evidence, not an
accuracy, robustness, latency, or capacity benchmark.

The immutable `runtime_validation/v6_compose_validation_2026-08-29.json` record
validated the then-current LLaVA v6 path and is now historical; it does not verify
either current v7 pair. Its Qwen path was likewise resource-blocked before model
load. Interpret that record only from its embedded v6 artifacts and observations.

The published Qwen package contains 209 qualification-evidence files, including
exactly 50 frozen cache entries. Its pair manifest binds
`cache_entry_tree_sha256=f7577b99b3f9e4cfff888fd9fa10f337be9d4fd880eb80a0f73065bc4e5864b7`
and
`corpus_manifest_sha256=ffd961e91771ec95b2781a66c8992fd0edafe94dfaf61f5fe4062dde02127220`.
This is a qualification evidence cache, not the Hugging Face model cache and not an
AEGIS runtime request cache. The recorded
`promotion_authorized_by_evidence: false` is intentional: passing evidence does not
itself authorize an operator to publish a package. The package is current because the
publisher subsequently installed and validated it against the schema-2 runtime
configuration.

Unpublished research outputs are intentionally not committed unless their feature
bundle records the model family, model identifier, exact model and tokenizer
revisions, layer, pooling, preprocessing fingerprint, aligned sample identifiers,
and feature dimensions. Qualified detector packages under `models/aegis/` are the
explicit exception: they retain the evidence closure required to validate the
bundled pair. The retained scripts reject missing or incompatible provenance.

## Verify the deliverables

From the repository root, using Python 3.10 or newer:

```powershell
python deliverables/ci/run_checks.py
```

This runs the root product regression suite, research-pipeline unit tests, and a
read-only integrity check. Use
`python deliverables/verify_deliverables.py --update` only after a reviewed
deliverable change.

## Generate current research outputs

Model-backed extraction must run in the inspected deployment image, not with an
arbitrary host Python environment. Build `aegis-mllm-guard:local` and prepare the
selected target's model cache first, then resolve the immutable image ID and the
single Compose cache volume from the repository root:

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

The canonical benchmark has 64 rows, including 20 text-only rows. Fused v7
extraction requires an image and non-empty text for every input row. Define this
selection step once, then run it in each target's new run directory before
extraction. It writes the 44 image-text rows in their original order, preserving
sample identifiers, all metadata columns, and image paths relative to
`deliverables/benchmark`; it does not change the canonical benchmark.

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

Create the selected metadata and extract the fused feature bundle for LLaVA:

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

docker run --rm `
  --entrypoint python `
  -e PYTHONDONTWRITEBYTECODE=1 `
  --mount "type=bind,source=$project,target=/workspace" `
  --workdir /workspace `
  $imageId `
  deliverables/ablation_report/run_ablation.py `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --runtime-target llava05b `
  --output-dir "$run/ablation"
```

For Qwen v7, extract the same fused layout with its pinned model revision and pixel
limit:

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

The feature-bundle provenance must match the selected detector before scoring is
allowed. Use the matching v7 pair manifest rather than an individual head. See
`pipeline/README.md` and `red_teaming/README.md` for exact scoring and evaluation
commands. Historical v6 single-head reproduction requires restoring the retired
artifact from Git history and extracting a separate feature bundle with matching
provenance and pooling before using the explicitly labelled `legacy_single` mode.

`benchmark/build_workbook.mjs` produces the Excel workbook and QA previews. The
canonical machine-readable benchmark is `benchmark/data/benchmark.csv`.

## Safety and disclosure

All authored adversarial examples use explicit redaction placeholders instead of
operational harmful content. Full third-party source datasets and model weights are
not redistributed. The qualified Qwen package retains a bounded, provenance-bound
snapshot of derived Bordair metadata, rendered inputs, and frozen feature-cache
entries needed to validate that published pair; it is not a copy of the full source
datasets. See `benchmark/DATASHEET.md` and `red_teaming/SAFE_USE.md` before extending
the corpus.
