# Target Validation Bundle: llava_onevision_05b_controlled

Status: `production_candidate`
Scope: `controlled_deployment`
Model family: `llava_onevision`
Model ID: `models\huggingface\llava-onevision-qwen2-0.5b-ov-hf`
Intended modalities: `image_text`

## Artifacts

- Detector: `models\aegis\aegis_llava_onevision_05b_text_detector.npz`
- Policy: `configs\llava_onevision_05b_controlled_policy.json`
- Provider contract: `outputs\llava_onevision_05b_provider_contract.json`
- Calibration summary: `outputs\llava_onevision_05b_controlled_calibration_summary.json`
- Monitoring summary: `outputs\llava_onevision_05b_controlled_monitoring_summary.json`
- Manifest: `docs\reproducibility_manifest.json`

## Evaluation Summaries

- `outputs\llava_onevision_05b_controlled_summary.json`

## Robustness Evidence

- `outputs\jailbreakv_eval_720_image_matched_controls_llava_onevision_05b_attack_family_holdout_summary.csv`

## Attack-Success Evidence

- `outputs\jailbreakv_eval_image_matched_llava_onevision_05b_guardrail_asr_summary.csv`

## Gate

- Gate OK: `True`
- Coverage: `{'summary_files': 1, 'model_families': ['llava_onevision'], 'sources': ['jailbreakv_image_matched_llava_onevision_05b_test']}`

## Notes

Production-candidate bundle for the LLaVA-OneVision 0.5B controlled target on the held-out image-matched JailBreakV test split. Live provider contract, calibrated policy, calibration summary, monitoring baseline, attack-family robustness evidence, guarded response attack-success reduction, and release evidence are attached.
