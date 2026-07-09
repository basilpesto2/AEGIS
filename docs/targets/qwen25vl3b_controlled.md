# Target Validation Bundle: qwen25vl3b_controlled

Status: `production_candidate`
Scope: `controlled_deployment`
Model family: `qwen25_vl`
Model ID: `Qwen/Qwen2.5-VL-3B-Instruct`
Intended modalities: `image_text`

## Artifacts

- Detector: `models\aegis\aegis_qwen25vl3b_text_detector.npz`
- Policy: `configs\qwen25vl3b_controlled_policy.json`
- Provider contract: `outputs\qwen25vl3b_provider_contract.json`
- Calibration summary: `outputs\qwen25vl3b_controlled_calibration_summary.json`
- Monitoring summary: `outputs\qwen25vl3b_controlled_monitoring_summary.json`
- Manifest: `docs\reproducibility_manifest.json`

## Evaluation Summaries

- `outputs\aegis_classify_multimodal_litmus_v1_summary.json`

## Robustness Evidence

- `outputs\jailbreakv_eval_image_matched_prompt_robustness_evaluation.csv`
- `outputs\jailbreakv_eval_image_matched_adaptive_prompt_evaluation.csv`

## Attack-Success Evidence

- `missing`

## Gate

- Gate OK: `True`
- Coverage: `{'summary_files': 1, 'model_families': ['qwen25_vl'], 'sources': ['aegis_classify_multimodal_litmus_v1']}`

## Notes

Production-candidate bundle for the current Qwen2.5-VL controlled target. Provider contract is single-pass because repeated full Qwen reloads can fail under local Accelerate offload; live embedding, provenance, finite vector shape, runtime handoff, calibrated policy, monitoring baseline, and prompt robustness/adaptive evidence are attached. Guarded response attack-success evidence is still missing for this target.
