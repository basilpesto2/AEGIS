# Current runtime validation

This directory records a sanitized live functional smoke test of AEGIS `0.3.0` on
2026-07-31. It covers the CLI, Docker service, tuned-v3 LLaVA detector, authenticated
API, runtime traffic control, and local GUI.

The machine-readable record is `docker_smoke_2026-07-31.json`. Authentication and
fingerprint secrets are deliberately omitted.

## What was demonstrated

- A fresh Docker image was built and exposed AEGIS `0.3.0`.
- The CLI demo worked, GUI assets were packaged, and LLaVA inference initialized.
- The packaged LLaVA model and tuned-v3 detector passed readiness and immutable
  model, tokenizer, and preprocessing provenance checks.
- An unauthenticated guard request returned HTTP 401; authenticated metrics and guard
  requests returned HTTP 200.
- One benign image-text smoke request returned `verdict=benign`, `action=allow`, and
  `modality=image_text` in shadow mode.
- Runtime traffic mode was changed from `shadow` to `review` and restored to `shadow`.
- The local GUI connected to the authenticated service, rendered its home page,
  completed one image-text evaluation, persisted one history record, and exited cleanly
  after its authenticated shutdown endpoint returned HTTP 202.

Docker was checked again after the functional smoke test. The completed LLaVA
container recorded authentication enabled, shadow mode, successful warmup, five
passing health checks, and exit code 0. Ports 8766 and 8767 were closed. This recheck
inspected Docker state and logs; it did not repeat the API or GUI requests.

## Evidence limits

This is a functional smoke test, not an accuracy, robustness, latency, or production
capacity benchmark. The single benign decision must not be generalized into a
performance claim.

The temporary validation container, test-only audit file, GUI history database, and GUI
process were removed after the check. Ports 8766 and 8767 were closed.
