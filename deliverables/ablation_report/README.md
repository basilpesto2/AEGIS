# Ablation, transfer, and deployed-detector scoring

`run_ablation.py` generates signal-importance, low-label, uncertainty, and
attack-family-transfer tables from an explicitly supplied, provenance-complete
MLLM feature bundle. It also compares that bundle with the selected current
runtime profile.

When the bundle exactly matches a v7 runtime pair, the runner scores the
`text_tokens` and `image_tokens` heads separately and composes their actions with
the runtime rule:

```text
Block > Review > Allow
```

It writes `runtime_detector_scores.csv` with both head decisions and the decisive
aggregate action, plus `runtime_detector_summary.csv` with per-head and combined
malicious-block and benign-block results. Model family, model identifier, model
and tokenizer revisions, preprocessing fingerprint, layer, fused pooling, and
both primitive dimensions must match exactly before these files are produced.
An incompatible research bundle can still be used for signal ablations, but the
manifest and report explicitly record why runtime scoring was skipped.

## Generate a v7 report

Use a fused feature bundle extracted with `text_image_tokens` pooling. The bundle
stores matching `n x d` text and image primitive matrices; the runtime provider
consumes their `n x 2d` text-then-image concatenation.

```powershell
$run = "deliverables/runs/current_llava"

python deliverables/ablation_report/run_ablation.py `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --runtime-target llava05b `
  --output-dir "$run/ablation"
```

For a Qwen feature bundle, set `$run` to
`deliverables/runs/current_qwen` and use `--runtime-target qwen25vl3b`. The
selected built-in target resolves to its current v7 dual-head pair. Legacy
single-head compatibility is used only when explicitly requested for historical
evidence. The generated report's reproduction block uses the actual runtime target,
input paths, and parent of `--output-dir`; inputs inside that run root are rendered
as `$run/...`, while external inputs retain their repository-relative or absolute
portable paths.

The runner refuses to replace a non-empty output directory unless `--force` is
explicit. Prefer a new versioned directory for each reportable run.

## Attach v7 bounded-evasion evidence

Supply the schema-3 adaptive summary, its exact dual-head score table, the exact
fused feature bundle, and the canonical pair manifest that produced them:

```powershell
# LLaVA
$pair = "models/aegis/llava05b_v7_dual_or/detector_pair_manifest.json"

# For Qwen, use both of these values instead:
# $pair = "models/aegis/qwen25vl3b_v7_dual_or/detector_pair_manifest.json"
# $runtimeTarget = "qwen25vl3b"
$runtimeTarget = "llava05b"

python deliverables/ablation_report/run_ablation.py `
  --metadata "$run/features/aligned_source_metadata.csv" `
  --features "$run/features/feature_bundle.npz" `
  --runtime-target $runtimeTarget `
  --output-dir "$run/ablation_with_redteam" `
  --adaptive-summary "$run/redteam/best_of_n.json" `
  --adaptive-scores "$run/redteam/scored_variants.csv" `
  --adaptive-pair-manifest $pair `
  --adaptive-features "$run/redteam/features/feature_bundle.npz"
```

The importer independently reloads both artifacts through the pair manifest,
checks artifact hashes and identities, checks shared provenance and primitive
dimensions, recomputes both head scores, recomputes `Block > Review > Allow`, and
rejects any inconsistent score row or evidence binding.

## Historical single-head compatibility

`--adaptive-detector PATH` is retained only for current schema-3 summaries whose
evaluator mode is explicitly `legacy_single`. It is mutually exclusive with
`--adaptive-pair-manifest`, and every generated report labels the imported
evidence `legacy_single`. The importer independently recomputes the artifact
scores, fixed-query ordering, selected variants, and best-of-N summary; older
schema-2 summaries are rejected. New v7 evidence should always use the
pair-manifest command above.
