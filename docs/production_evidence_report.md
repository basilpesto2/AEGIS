# AEGIS Production Evidence Report

Overall status: `partial`

## Requirement Matrix

| Requirement | Status | Evidence | Notes |
| --- | --- | --- | --- |
| balanced_corpus | `satisfied` | C:\Users\Eugene\Documents\FYP\data\processed\vlguard_test_full_groupsplit_metadata.csv<br>C:\Users\Eugene\Documents\FYP\data\processed\mssbench_full_groupsplit_metadata.csv<br>C:\Users\Eugene\Documents\FYP\data\processed\jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv<br>C:\Users\Eugene\Documents\FYP\data\processed\aegis_classify_multimodal_litmus_v1\metadata.csv | Dataset metadata exists locally; third-party raw data remains untracked by design. |
| red_teaming_attack_styles | `satisfied` | C:\Users\Eugene\Documents\FYP\data\processed\aegis_classify_multimodal_litmus_v1\metadata.csv<br>C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_prompt_robustness_evaluation.csv<br>C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_adaptive_prompt_evaluation.csv | Expanded litmus and robustness/adaptive variants provide safe redacted coverage. |
| representation_detector | `satisfied` | C:\Users\Eugene\Documents\FYP\models\aegis\aegis_qwen25vl3b_text_detector.npz<br>C:\Users\Eugene\Documents\FYP\docs\reproducibility_manifest.json | Detector artifact and manifest are present. |
| uncertainty_and_review | `satisfied` | C:\Users\Eugene\Documents\FYP\outputs\aegis_classify_multimodal_litmus_v1_summary.json<br>C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_qwen25vl3b_signal_analysis_feature_comparison.csv | Runtime supports review decisions; litmus summary reports uncertain samples. |
| signal_importance | `satisfied` | C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_qwen25vl3b_signal_analysis_feature_comparison.csv<br>C:\Users\Eugene\Documents\FYP\outputs\mssbench_full_qwen25vl3b_signal_analysis_feature_comparison.csv<br>C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_llava_onevision_05b_signal_analysis_feature_comparison.csv | Signal-analysis result files exist across jailbreak and MSSBench settings. |
| low_label_learning | `satisfied` | C:\Users\Eugene\Documents\FYP\outputs\vlguard_test_full_qwen25vl3b_low_label_seed_sweep_summary.csv<br>C:\Users\Eugene\Documents\FYP\outputs\mssbench_full_qwen25vl3b_trusted_pseudo_logistic_seed_sweep_summary.csv | Low-label and trusted-plus-pseudo summaries exist. |
| diverse_models | `satisfied` | C:\Users\Eugene\Documents\FYP\outputs\aegis_classify_multimodal_litmus_v1_summary.json<br>C:\Users\Eugene\Documents\FYP\outputs\llava_onevision_05b_controlled_summary.json<br>C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_720_image_matched_controls_llava_onevision_05b_attack_family_holdout_summary.csv<br>C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_llava_onevision_05b_guardrail_asr_summary.csv | Qwen2.5-VL and LLaVA-OneVision controlled/release evidence are present. |
| diverse_datasets | `satisfied` | C:\Users\Eugene\Documents\FYP\data\processed\vlguard_test_full_groupsplit_metadata.csv<br>C:\Users\Eugene\Documents\FYP\data\processed\mssbench_full_groupsplit_metadata.csv<br>C:\Users\Eugene\Documents\FYP\data\processed\jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv<br>C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_source_disjoint_transfer_summary.csv | VLGuard, MSSBench, JailBreakV, and transfer summaries are present. |
| robustness | `satisfied` | C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_prompt_robustness_evaluation.csv<br>C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_adaptive_prompt_evaluation.csv | Prompt robustness and bounded adaptive evaluation files exist. |
| attack_success_reduction | `satisfied` | C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_llava_onevision_05b_guardrail_asr_summary.csv | ASR proxy summary exists for LLaVA-OneVision guarded response evaluation. |
| production_operations | `satisfied` | C:\Users\Eugene\Documents\FYP\docs\production_guardrail_playbook.md<br>C:\Users\Eugene\Documents\FYP\configs\guardrail_policy.example.json<br>C:\Users\Eugene\Documents\FYP\AEGIS\guardrail.py<br>C:\Users\Eugene\Documents\FYP\AEGIS\provider_contract.py<br>C:\Users\Eugene\Documents\FYP\AEGIS\deployment_package.py<br>C:\Users\Eugene\Documents\FYP\AEGIS\target_release.py<br>C:\Users\Eugene\Documents\FYP\scripts\serve_guardrail_provider.py<br>C:\Users\Eugene\Documents\FYP\scripts\evaluate_detector_features.py<br>C:\Users\Eugene\Documents\FYP\scripts\validate_target_release.py<br>C:\Users\Eugene\Documents\FYP\scripts\validate_target_registry.py<br>C:\Users\Eugene\Documents\FYP\scripts\build_target_deployment_package.py<br>C:\Users\Eugene\Documents\FYP\scripts\validate_target_deployment_package.py<br>C:\Users\Eugene\Documents\FYP\outputs\deployment_packages\qwen25vl3b_controlled\deployment_package_report.json<br>C:\Users\Eugene\Documents\FYP\outputs\deployment_packages\llava_onevision_05b_controlled\deployment_package_report.json<br>C:\Users\Eugene\Documents\FYP\configs\llava_onevision_05b_controlled_policy.json<br>C:\Users\Eugene\Documents\FYP\outputs\llava_onevision_05b_provider_contract.json<br>C:\Users\Eugene\Documents\FYP\outputs\llava_onevision_05b_controlled_calibration_summary.json<br>C:\Users\Eugene\Documents\FYP\outputs\llava_onevision_05b_controlled_monitoring_summary.json<br>C:\Users\Eugene\Documents\FYP\docs\targets\llava_onevision_05b_controlled.json<br>C:\Users\Eugene\Documents\FYP\configs\qwen25vl3b_controlled_policy.json<br>C:\Users\Eugene\Documents\FYP\outputs\qwen25vl3b_provider_contract.json<br>C:\Users\Eugene\Documents\FYP\outputs\qwen25vl3b_controlled_calibration_summary.json<br>C:\Users\Eugene\Documents\FYP\outputs\qwen25vl3b_controlled_monitoring_summary.json<br>C:\Users\Eugene\Documents\FYP\docs\targets\qwen25vl3b_controlled.json<br>C:\Users\Eugene\Documents\FYP\docs\reproducibility_manifest.json<br>C:\Users\Eugene\Documents\FYP\docs\target_validation_registry.json | Operational docs/config, provider contracts, calibrated policies, monitoring baselines, modality-declared target bundles, release validators, deployment packages, and manifest exist. |
| universal_any_llm_mllm_claim | `partial` | C:\Users\Eugene\Documents\FYP\docs\production_guardrail_playbook.md<br>C:\Users\Eugene\Documents\FYP\docs\targets\qwen25vl3b_controlled.json<br>C:\Users\Eugene\Documents\FYP\docs\targets\llava_onevision_05b_controlled.json<br>C:\Users\Eugene\Documents\FYP\docs\target_validation_registry.json<br>C:\Users\Eugene\Documents\FYP\outputs\target_registry_universal_portfolio_report.json | The registry portfolio audit is executable and currently records image-text production coverage; this remains partial because a real text-only LLM target, Qwen target-specific attack-success evidence, and arbitrary additional target models still need their own provider contract, calibrated policy, representative traffic evidence, robustness evidence, and guarded attack-success evidence. |

## Evidence Inventory

- `default_detector`: yes - `C:\Users\Eugene\Documents\FYP\models\aegis\aegis_qwen25vl3b_text_detector.npz`
- `llava_detector`: yes - `C:\Users\Eugene\Documents\FYP\models\aegis\aegis_llava_onevision_05b_text_detector.npz`
- `reproducibility_manifest`: yes - `C:\Users\Eugene\Documents\FYP\docs\reproducibility_manifest.json`
- `expanded_litmus_summary`: yes - `C:\Users\Eugene\Documents\FYP\outputs\aegis_classify_multimodal_litmus_v1_summary.json`
- `expanded_litmus_metadata`: yes - `C:\Users\Eugene\Documents\FYP\data\processed\aegis_classify_multimodal_litmus_v1\metadata.csv`
- `vlguard_metadata`: yes - `C:\Users\Eugene\Documents\FYP\data\processed\vlguard_test_full_groupsplit_metadata.csv`
- `mssbench_metadata`: yes - `C:\Users\Eugene\Documents\FYP\data\processed\mssbench_full_groupsplit_metadata.csv`
- `jailbreakv_metadata`: yes - `C:\Users\Eugene\Documents\FYP\data\processed\jailbreakv_eval_720_image_matched_controls_groupsplit_metadata.csv`
- `qwen_low_label`: yes - `C:\Users\Eugene\Documents\FYP\outputs\vlguard_test_full_qwen25vl3b_low_label_seed_sweep_summary.csv`
- `trusted_pseudo_low_label`: yes - `C:\Users\Eugene\Documents\FYP\outputs\mssbench_full_qwen25vl3b_trusted_pseudo_logistic_seed_sweep_summary.csv`
- `prompt_robustness`: yes - `C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_prompt_robustness_evaluation.csv`
- `adaptive_prompt`: yes - `C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_adaptive_prompt_evaluation.csv`
- `guardrail_asr`: yes - `C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_llava_onevision_05b_guardrail_asr_summary.csv`
- `qwen_signal_analysis`: yes - `C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_qwen25vl3b_signal_analysis_feature_comparison.csv`
- `mssbench_signal_analysis`: yes - `C:\Users\Eugene\Documents\FYP\outputs\mssbench_full_qwen25vl3b_signal_analysis_feature_comparison.csv`
- `llava_signal_analysis`: yes - `C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_image_matched_llava_onevision_05b_signal_analysis_feature_comparison.csv`
- `qwen_transfer`: yes - `C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_source_disjoint_transfer_summary.csv`
- `llava_attack_holdout`: yes - `C:\Users\Eugene\Documents\FYP\outputs\jailbreakv_eval_720_image_matched_controls_llava_onevision_05b_attack_family_holdout_summary.csv`
- `production_playbook`: yes - `C:\Users\Eugene\Documents\FYP\docs\production_guardrail_playbook.md`
- `policy_config`: yes - `C:\Users\Eugene\Documents\FYP\configs\guardrail_policy.example.json`
- `guardrail_runtime`: yes - `C:\Users\Eugene\Documents\FYP\AEGIS\guardrail.py`
- `provider_contract_runtime`: yes - `C:\Users\Eugene\Documents\FYP\AEGIS\provider_contract.py`
- `deployment_package_runtime`: yes - `C:\Users\Eugene\Documents\FYP\AEGIS\deployment_package.py`
- `target_release_runtime`: yes - `C:\Users\Eugene\Documents\FYP\AEGIS\target_release.py`
- `provider_guardrail_server`: yes - `C:\Users\Eugene\Documents\FYP\scripts\serve_guardrail_provider.py`
- `detector_feature_evaluator`: yes - `C:\Users\Eugene\Documents\FYP\scripts\evaluate_detector_features.py`
- `target_release_validator`: yes - `C:\Users\Eugene\Documents\FYP\scripts\validate_target_release.py`
- `target_registry_validator`: yes - `C:\Users\Eugene\Documents\FYP\scripts\validate_target_registry.py`
- `target_deployment_package_builder`: yes - `C:\Users\Eugene\Documents\FYP\scripts\build_target_deployment_package.py`
- `target_deployment_package_validator`: yes - `C:\Users\Eugene\Documents\FYP\scripts\validate_target_deployment_package.py`
- `qwen_deployment_package_report`: yes - `C:\Users\Eugene\Documents\FYP\outputs\deployment_packages\qwen25vl3b_controlled\deployment_package_report.json`
- `llava_deployment_package_report`: yes - `C:\Users\Eugene\Documents\FYP\outputs\deployment_packages\llava_onevision_05b_controlled\deployment_package_report.json`
- `llava_policy`: yes - `C:\Users\Eugene\Documents\FYP\configs\llava_onevision_05b_controlled_policy.json`
- `llava_provider_contract`: yes - `C:\Users\Eugene\Documents\FYP\outputs\llava_onevision_05b_provider_contract.json`
- `llava_calibration_summary`: yes - `C:\Users\Eugene\Documents\FYP\outputs\llava_onevision_05b_controlled_calibration_summary.json`
- `llava_monitoring_summary`: yes - `C:\Users\Eugene\Documents\FYP\outputs\llava_onevision_05b_controlled_monitoring_summary.json`
- `llava_target_bundle`: yes - `C:\Users\Eugene\Documents\FYP\docs\targets\llava_onevision_05b_controlled.json`
- `llava_controlled_summary`: yes - `C:\Users\Eugene\Documents\FYP\outputs\llava_onevision_05b_controlled_summary.json`
- `qwen_policy`: yes - `C:\Users\Eugene\Documents\FYP\configs\qwen25vl3b_controlled_policy.json`
- `qwen_provider_contract`: yes - `C:\Users\Eugene\Documents\FYP\outputs\qwen25vl3b_provider_contract.json`
- `qwen_calibration_summary`: yes - `C:\Users\Eugene\Documents\FYP\outputs\qwen25vl3b_controlled_calibration_summary.json`
- `qwen_monitoring_summary`: yes - `C:\Users\Eugene\Documents\FYP\outputs\qwen25vl3b_controlled_monitoring_summary.json`
- `qwen_target_bundle`: yes - `C:\Users\Eugene\Documents\FYP\docs\targets\qwen25vl3b_controlled.json`
- `target_validation_registry`: yes - `C:\Users\Eugene\Documents\FYP\docs\target_validation_registry.json`
- `target_portfolio_universal_report`: yes - `C:\Users\Eugene\Documents\FYP\outputs\target_registry_universal_portfolio_report.json`
