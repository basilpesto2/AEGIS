# AEGIS signal-importance and transferability ablation report

## Evidence status

This run is classified as **`software_smoke_test_only`** and uses feature source **`deterministic_smoke_fixture_not_mllm_evidence`**. The deterministic fixture verifies the complete training, validation, thresholding, low-label, uncertainty, transfer, and reporting path. It is not evidence of MLLM safety performance.

## Relationship to the current runtime

The current AEGIS `0.3.0` LLaVA profile uses
`models/aegis/aegis_llava_onevision_05b_text_detector_tuned_v3.npz` (SHA-256
`9816ed0b706b3d0e01162701310ad77bc795c7dd99f7e2e6da4133d6fa299587`), an
896-dimensional `text_tokens` detector with block threshold `0.6909739881800183`
and review threshold `0.23579741243702598`. The tables below use a separate
8-dimensional deterministic fixture detector at threshold `0.8202568457258909`;
they must not be cited as tuned-v3 accuracy, attack-success, or robustness
measurements. Current functional runtime evidence is recorded in
`deliverables/runtime_validation/`.

## Protocol

Matched benign/adversarial groups remain in one split. Classifier fitting uses training rows, threshold selection uses validation rows, and the test split is not used for selection. The fixture experiment detector is class-balanced logistic regression over pooled representations and compact signals.

## Signal importance

| feature_view | feature_dim | auroc | auprc | precision | recall | f1 | mean_test_uncertainty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text_representation | 32 | 0.8750 | 0.9129 | 0.7778 | 0.8750 | 0.8235 | 0.2978 |
| image_representation | 32 | 0.3516 | 0.4333 | 0.4615 | 0.7500 | 0.5714 | 0.9962 |
| joint_representation | 64 | 0.8906 | 0.9308 | 0.7778 | 0.8750 | 0.8235 | 0.2793 |
| cross_modal_consistency | 8 | 0.3438 | 0.5130 | 0.5333 | 1.0000 | 0.6957 | 0.9657 |
| attribution | 8 | 1.0000 | 1.0000 | 0.8889 | 1.0000 | 0.9412 | 0.2009 |
| joint_plus_consistency | 72 | 0.8750 | 0.9129 | 0.7778 | 0.8750 | 0.8235 | 0.2857 |
| all_input_signals | 80 | 0.9844 | 0.9861 | 1.0000 | 0.7500 | 0.8571 | 0.1885 |

`all_input_signals` combines text and image representations, cross-modal consistency, and perturbation-attribution summaries. Comparisons are diagnostic because the smoke extractor is intentionally lightweight and its lexical cues are not MLLM hidden states.

## Low-label and pseudo-label ablation

| trusted_per_class | pseudo_per_class | seeds | mean_auroc | std_auroc | mean_auprc | std_auprc | mean_f1 | mean_pseudo_selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0 | 5 | 0.8688 | 0.0272 | 0.9011 | 0.0190 | 0.7814 | 0.0000 |
| 4 | 4 | 5 | 0.9125 | 0.0538 | 0.9376 | 0.0347 | 0.8237 | 8.0000 |
| 8 | 0 | 5 | 0.9219 | 0.0474 | 0.9429 | 0.0246 | 0.8379 | 0.0000 |
| 8 | 4 | 5 | 0.9219 | 0.0261 | 0.9437 | 0.0113 | 0.8379 | 8.0000 |

Pseudo-labeling is accepted only when both predicted classes contribute the same number of high-confidence training rows. A zero `mean_pseudo_selected` records that the confidence gate correctly declined to invent supervision.

## Attack-family transfer

| feature_view | held_out_attack_style | n_test | auroc | auprc | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| text_representation | cross_modal_misalignment | 8 | 0.9375 | 0.9500 | 0.7500 | 0.7500 | 0.7500 |
| text_representation | direct_policy_violation | 8 | 1.0000 | 1.0000 | 0.8000 | 1.0000 | 0.8889 |
| text_representation | image_embedded_instruction | 8 | 0.5000 | 0.7321 | 0.6667 | 0.5000 | 0.5714 |
| text_representation | indirect_prompt_injection | 8 | 0.6875 | 0.7929 | 1.0000 | 0.2500 | 0.4000 |
| text_representation | instruction_hierarchy | 8 | 0.9375 | 0.9500 | 0.8000 | 1.0000 | 0.8889 |
| text_representation | role_play | 8 | 0.9375 | 0.9500 | 1.0000 | 0.7500 | 0.8571 |
| text_representation | safety_camouflage | 8 | 1.0000 | 1.0000 | 0.8000 | 1.0000 | 0.8889 |
| text_representation | text_obfuscation | 8 | 0.9375 | 0.9500 | 0.8000 | 1.0000 | 0.8889 |
| all_input_signals | cross_modal_misalignment | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| all_input_signals | direct_policy_violation | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| all_input_signals | image_embedded_instruction | 8 | 0.8125 | 0.8542 | 0.0000 | 0.0000 | 0.0000 |
| all_input_signals | indirect_prompt_injection | 8 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| all_input_signals | instruction_hierarchy | 8 | 1.0000 | 1.0000 | 0.8000 | 1.0000 | 0.8889 |
| all_input_signals | role_play | 8 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.6667 |
| all_input_signals | safety_camouflage | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| all_input_signals | text_obfuscation | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Each transfer row holds out all four matched groups of one attack family. Training and threshold selection use different attack families. With only eight held-out rows per family, these estimates have high variance and must not be generalized to deployment.

## Uncertainty and review coverage

| review_fraction | reviewed_rows | retained_rows | coverage | retained_accuracy | mean_retained_uncertainty |
| --- | --- | --- | --- | --- | --- |
| 0.0000 | 0 | 16 | 1.0000 | 0.8750 | 0.1885 |
| 0.1250 | 2 | 14 | 0.8750 | 0.9286 | 0.1065 |
| 0.2500 | 4 | 12 | 0.7500 | 1.0000 | 0.0638 |
| 0.5000 | 8 | 8 | 0.5000 | 1.0000 | 0.0289 |

Rows are ranked for review by normalized binary entropy. This table exposes the coverage/accuracy trade-off rather than treating near-threshold decisions as certain.

## Bounded best-of-N fixture evasion

| query_budget | n_samples | threshold | malicious_recall | evasion_rate | mean_worst_case_score |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 0.8203 | 0.7500 | 0.2500 | 0.8509 |
| 3 | 8 | 0.8203 | 0.3750 | 0.6250 | 0.6473 |
| 6 | 8 | 0.8203 | 0.0000 | 1.0000 | 0.5732 |

The evaluator selects the lowest detector score in hindsight among the first N fixed,
deterministic variants. This is a bounded best-of-N oracle analysis, not a sequential
adaptive attack policy. These smoke results intentionally reveal that the lightweight
fixture detector is brittle; they measure fixture-detector evasion, not harmful MLLM
response generation.

## Legacy MLLM evidence

`legacy_evidence.csv` preserves aggregate values reported in commit `3f6e46a`. The
underlying processed data, embeddings, and output tables were ignored and are absent,
so these values are historical context only - not independently rerun results. Those
legacy values describe earlier Qwen/JailBreakV experiments and an earlier LLaVA
artifact, not the current tuned-v3 detector.

## Conclusions

The framework measures representation, modality, consistency, attribution,
uncertainty, label-efficiency, and held-family transfer independently. Scientific
claims remain conditional on replacing the smoke fixture with provenance-complete
MLLM features and rerunning the report into a new versioned output directory.

## Reproduction

```powershell
$run = "deliverables/reproductions/ablation_v1"

python deliverables/pipeline/scripts/build_smoke_features.py `
  --output "$run/benchmark_smoke_features.npz"
python deliverables/ablation_report/run_ablation.py `
  --features "$run/benchmark_smoke_features.npz" `
  --output-dir "$run/results"
```

This creates a new ablation run without altering the tables above. See
`deliverables/README.md` for the full versioned workflow that also regenerates and
provenance-checks the red-team best-of-N analysis.
