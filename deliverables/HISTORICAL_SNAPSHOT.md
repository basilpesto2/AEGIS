# Historical snapshot preservation and path relocation

The committed research outputs under `ablation_report/results/`,
`pipeline/artifacts/smoke/`, and `red_teaming/generated/` originated in commit
`7b50825` ("Added project deliverables"). Their numeric results are retained as
historical evidence and are not silently recalculated against the current tuned-v3
runtime.

## Directory-name relocation

The original generators and embedded metadata used two intended directory names that
did not match the paths committed by `7b50825`:

| Historical embedded path | Current repository path |
| --- | --- |
| `deliverables/annotated_benchmark/` | `deliverables/benchmark/` |
| `deliverables/reproducible_pipeline/` | `deliverables/pipeline/` |

Consequently, 40 historical `image_path` values in
`red_teaming/generated/variants.csv` and `scored_variants.csv` contain the prefix
`../../annotated_benchmark/images/`. Those fields are preserved byte-for-byte for
provenance. When resolving an image, substitute
`../../benchmark/images/`. New runs produced by the repaired generator use the current
path directly.

The historical ablation `run_manifest.json` also contains the original author's
absolute Windows paths. Treat those strings as a snapshot of the old run environment,
not as portable commands. New ablation runs record repository-relative paths and write
to a versioned output directory.

The repository `.gitattributes` keeps these frozen evidence directories byte-preserving
so their original SHA-256 identities survive checkout and recommit. Newly authored text
artifacts use LF endings for cross-platform reproducibility.

## Frozen fixture identities

| Artifact | SHA-256 | Meaning |
| --- | --- | --- |
| `pipeline/fixtures/smoke_features.npz` | `93cd720edfc6bfbd55a94eba05dbe3079a3d0e350d390ac8ebe5273b38fa32a2` | 64-row deterministic benchmark fixture; 32-D text, 32-D image, and 8-D attribution inputs. |
| `pipeline/artifacts/smoke/selected_detector.npz` | `c008be65092b3b113832d91c968cf4c3842745c78f38544b4cede17468ba7b71` | 8-D attribution fixture detector, threshold `0.8202568457258909`; not deployed. |
| `red_teaming/generated/variants.csv` | `c807e6d12676a367783165ecd950bff8c2eca7f3b357e8e3dd47a8c0c5458ba7` | Frozen 48-row, six-variant safety-redacted panel. |
| `red_teaming/generated/variant_smoke_features.npz` | `f6a3ba6d712289b395737041c4fa25afebbb1ed3d4bd053a509a239fb5fe2b5c` | Deterministic feature bundle aligned to the frozen variant panel. |
| `red_teaming/generated/scored_variants.csv` | `5a0b3dc9830317c223b2da9f62551b326216f42e96fb384dabcf3484475d5950` | Frozen fixture-detector scores used by the best-of-N table. |

The current deployed LLaVA tuned-v3 artifact is separately identified in
`runtime_validation/docker_smoke_2026-07-31.json`. Its 896-dimensional feature
contract is incompatible with the frozen 8-dimensional fixture detector. Applying
current block or review thresholds to these frozen scores would not produce valid
current-runtime evidence.

`ablation_report/legacy_evidence.csv` is an additional, older aggregate-only snapshot
from commit `3f6e46a`. Its source embeddings and detailed output tables are absent; it
must remain labelled as historical context rather than independently reproducible
evidence.
