# Runtime validation records

This directory retains two immutable records. The current
`v7_compose_validation_2026-08-30.json` record covers the active v7 dual-head
packages and has `partial` status. The earlier
`v6_compose_validation_2026-08-29.json` record remains historical evidence for the
then-current Bordair OCR v6 targets. A field marked `not_run_resource_preflight` is
pending evidence and must not be interpreted as a passed check. Authentication,
fingerprint, and dashboard-session secrets are omitted.

## Current v7 contract on 2026-08-30

The current record has SHA-256
`0496760f8d69a1c4c306622b4d139b18302311231e87074c008435ead5d20b7c`.

| Target / head | Artifact | Pooling | Dimension | Block | Review | Live evidence |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `llava05b` text | `llava05b_v7_dual_or/llava05b_text_head_v1.npz` | `text_tokens` | 896 | `0.9973003604375604` | `0.9773003604375604` | LLaVA v7 functional path passed |
| `llava05b` image | `llava05b_v7_dual_or/llava05b_image_head_v1.npz` | `image_tokens` | 896 | `0.43109073768976996` | `0.41109073768976995` | LLaVA v7 functional path passed |
| `qwen25vl3b` text | `qwen25vl3b_v7_dual_or/qwen25vl3b_text_head_v1.npz` | `text_tokens` | 2048 | `0.9975257227486033` | `0.9775257227486033` | Resource-blocked before model load |
| `qwen25vl3b` image | `qwen25vl3b_v7_dual_or/qwen25vl3b_image_head_v1.npz` | `image_tokens` | 2048 | `0.1690249723273814` | `0.1490249723273814` | Resource-blocked before model load |

LLaVA v7 passed Compose build/start, startup resource preflight, CUDA model loading,
warmup, readiness, and authenticated status. The loaded runtime reported OR mode,
two heads, zero worker restarts, and detector identity
`25cebcf1e91125f4823176111d88cddae7379a0f70a1d8cc8af5ff07e22f08f6`.
The local dashboard passed one benign fixed-panel case (`BTI-05697`, recommended and
effective `allow`) and one malicious case (`TI-01093`, recommended `block` but
effective `allow` because traffic mode was `shadow`). Dashboard and container
shutdown were graceful, the container exited with code 0, and it was not OOM-killed.

A separate doctor run after the model was already loaded and serving passed 38 of 39
required checks. Available virtual memory had fallen to 4,216,020,992 bytes against
4,294,967,296 required. The same fail-closed resource preflight had passed at startup,
and no requirement was reduced; the record therefore labels this post-load doctor
observation `partial` rather than rewriting it as a pass.

The Qwen image and published detector package were present, but the service failed
closed before model loading on the unchanged resource gate. The Docker VM exposed
8,172,244,992 bytes of total RAM against 16,106,127,360 required and 7,440,695,296
bytes of free VRAM against 7,516,192,768 required. Available physical memory,
available virtual memory, disk, and model-cache size passed their configured minima.
Qwen CUDA model load, warmup, and authenticated image-text inference remain listed in
`pending_checks`; none is claimed as passed.

## Historical v6 record

The v6 JSON records the configuration paths used at the time. Those configurations
have since been updated to v7, so reproduce or interpret the v6 record from its
embedded artifacts, pooling, thresholds, and hashes—not from the current contents of
the referenced configuration files. The record is immutable.
The JSON itself remains unchanged. It is
superseded for both active runtimes and does not verify either v7 runtime. Its
`partial` status means the common Compose image and complete LLaVA v6 CUDA path
passed, while Qwen failed closed at the unmodified resource preflight.

## Recorded v6 contract on 2026-08-29

| Target | Artifact | Pooling | Dimension | Block | Review | Live evidence |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `llava05b` | `aegis_llava_onevision_05b_bordair_ocr_v6.npz` | `image_tokens` | 896 | `0.2938954606972983` | `0.2792006876624334` | Pass |
| `qwen25vl3b` | `aegis_qwen25vl3b_bordair_ocr_v6.npz` | `image_tokens` | 2048 | `0.36330026547530514` | `0.34513525220153984` | Resource-blocked before model load |

Both targets accept only image-text requests with non-empty text and exactly one
image. The rebuilt image was
`sha256:f4faef3c4e66d8470802728197fd3074577cd5221290660aa645273f9d4c7a6d`.

The then-current LLaVA v6 target passed all 38 required doctor checks, including
detector SHA-256, exact model and tokenizer revisions, layer, pooling, preprocessing
fingerprint, and model-content SHA-256. Its isolated CUDA worker loaded and completed
warmup with zero restarts. A
timed restart reached ready state in 29.664 seconds, and an authenticated uncached
image-text request returned HTTP 200 in 0.435 seconds using the v6 detector source.
The public readiness response omitted internal diagnostics; detailed status and
metrics required the evaluation bearer. Runtime code, configuration, detector, and
model cache were not writable by the service account, while the audit volume remained
writable. The container stopped gracefully in 4.581 seconds with exit code 0.

The Qwen model-cache preflight passed against the exact configured snapshot revision:
11 files, 7,520,892,432 bytes, and runtime fingerprint
`45f7d1afd0ef8e09cb7a79456fd232014d2a482ca975d2008848c0efa77e4ce0`.
No download was performed. This proves cache readiness only, not CUDA model loading
or inference.

Qwen then failed closed before model loading. The Docker VM exposed 8,172,244,992
bytes of RAM against a required 16,106,127,360 and 7,440,695,296 bytes of free VRAM
against a required 7,516,192,768. Its cache, available physical-memory, and
virtual-memory checks passed, but warmup and authenticated inference were not run. No
resource requirement was reduced to obtain a more favorable result. The three unrun Qwen
checks remain listed in the record's `pending_checks` field.

## Evidence limits

Each record is functional validation evidence, not an accuracy, robustness, latency,
or production-capacity benchmark. The two LLaVA v7 fixed-panel inferences and the
single LLaVA v6 inference must not be generalized into detector-performance claims;
their recorded durations are observations, not latency benchmarks. Qwen model
loading, warmup, and inference remain unclaimed in both records because each
configured resource preflight failed closed.
