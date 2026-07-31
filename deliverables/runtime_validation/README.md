# Current runtime validation

This directory records a sanitized live smoke test of the cleaned AEGIS `0.3.0`
checkout on 2026-07-31. It exists to connect the restored research evidence to the
current CLI, Docker service, tuned-v3 LLaVA detector, and GUI.

The machine-readable record is `docker_smoke_2026-07-31.json`. Authentication and
fingerprint secrets are deliberately omitted.

The live validation ran on
`b0a4aca0eadfdf61e2eecd615de0417689729e00` before that local commit was
squashed with the deliverable restoration and refresh. The record retains that former
execution hash as provenance. The surviving tracked-operational source anchor is
`71a183bd10a102324a7c782155075e977d0e83ae`, recorded as
`validated_tracked_operational_source_commit`. Every tracked path in
`validated_tracked_operational_scope` is byte-identical between that source anchor and
the post-squash checkout.

The root `README.md` is not part of that byte-identity claim. The Dockerfile copies it
into the image and the package build uses it as metadata. Its content changed during
deliverable cleanup and restoration, so a rebuild can have different documentation,
package metadata, image bytes, and image ID. It does not change AEGIS executable
modules, configs, or model artifacts. The recorded Docker image ID belongs to the
pre-squash execution and is not a promised rebuild identity at the surviving source
anchor.

The ignored Hugging Face model cache is also outside Git's tracked-path comparison;
its model revision, tokenizer revision, preprocessing fingerprint, and cache size are
recorded separately. The combined commit changes documentation and research tooling,
not the operational runtime that was smoke-tested.

## What was demonstrated

- A fresh Docker image was built from the cleaned checkout.
- The image exposed AEGIS `0.3.0`, the CLI demo worked, GUI assets were packaged, CUDA
  detected the NVIDIA GeForce RTX 4060 Laptop GPU, and deleted test/research directories
  were absent from the runtime image.
- The packaged LLaVA model cache and tuned-v3 detector passed readiness and immutable
  model/tokenizer/preprocessing provenance checks.
- An unauthenticated guard request returned HTTP 401; authenticated metrics and guard
  requests returned HTTP 200.
- One benign image-text smoke request returned `verdict=benign`, `action=allow`, and
  `modality=image_text` in shadow mode.
- Runtime traffic mode was changed from `shadow` to `review` and restored to `shadow`.
- The local GUI connected to the authenticated service, rendered its home page,
  completed one image-text evaluation, persisted one history record, and exited cleanly
  after its authenticated shutdown endpoint returned HTTP 202.

After the deliverables were restored, Docker was checked again. The engine was
running; the completed LLaVA container recorded authentication enabled, shadow mode,
successful warmup, five passing health checks, about 33 minutes of runtime, and exit
code 0. Ports 8766 and 8767 were closed. This post-restoration check inspected Docker
state and logs; it did not repeat the earlier API or GUI requests.

## Resource-qualified result

The stock `llava05b` Compose profile did not remain up with this Docker Desktop
allocation. It requires at least 2.00 GiB available physical memory at preflight; the
observed range after initialization was 1.72-1.82 GiB. Total memory, virtual memory,
disk, model-cache size, and GPU-memory checks passed, and the container was not
OOM-killed.

To isolate code/runtime correctness from that narrow host-memory gate, the same image,
model cache, detector, request limits, policy, CUDA requirement, and worker settings
were started in an ephemeral validation container with:

- `target_profile=null`, so the built-in profile budget was not misrepresented; and
- only `min_available_physical_bytes` lowered from 2.00 GiB to 1.50 GiB.

That non-production validation container became ready and completed the API and GUI
checks above. No repository configuration was changed by the override.

## Evidence limits

This is a functional smoke test, not an accuracy, robustness, latency, or production
capacity benchmark. The single benign decision and GUI duration must not be generalized
into a performance claim. The deterministic research metrics elsewhere in
`deliverables/` use a separate fixture detector and likewise do not measure the current
tuned-v3 runtime.

The Qwen profile was not rerun because its 15 GiB total-memory requirement exceeds the
recorded 7.61 GiB Docker allocation and its model cache was not prepared.

The temporary validation container, test-only audit file, GUI history database, and GUI
process were removed after the check. Ports 8766 and 8767 were closed.
