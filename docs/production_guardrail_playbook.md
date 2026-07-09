# AEGIS Production Guardrail Playbook

AEGIS is an input-side guardrail component. A production deployment should treat it
as a policy decision engine around a detector artifact and a model-specific
embedding provider.

## Integration Contract

For any LLM or MLLM, implement an embedding provider that returns the same feature
space used to train the detector:

```python
from AEGIS.guardrail import GuardrailRequest, GuardrailRuntime
from AEGIS.detector_artifact import load_detector_artifact

artifact = load_detector_artifact("models/aegis/my_detector.npz")
runtime = GuardrailRuntime(artifact)

request = GuardrailRequest(
    request_id="req-123",
    text="user prompt",
    image_paths=("image.png",),
)
decision = runtime.evaluate_request(request, provider)
```

The provider must expose `model_family`, `model_id`, `pooling`, `feature_dim`, and
an `embed(request)` method. For batch systems, write AEGIS-format NPZ files and
run:

```powershell
python -m AEGIS.cli guard-features `
  --features outputs\my_model_features.npz `
  --metadata data\processed\my_requests.csv `
  --detector models\aegis\my_detector.npz `
  --require-matching-provenance `
  --output outputs\my_guardrail_decisions.csv
```

For live request systems, use the same provider contract directly:

```powershell
python -m AEGIS.cli guard-request `
  --provider my_package.my_provider:provider `
  --detector models\aegis\my_detector.npz `
  --text "user prompt" `
  --request-id req-123 `
  --require-matching-provenance
```

## Required Deployment Phases

1. Shadow mode: score representative traffic without blocking.
2. Calibration: choose thresholds on local benign, malicious, and ambiguous traffic.
3. Review mode: route `review` and high-risk `block` candidates to human or stronger automated review.
4. Limited blocking: block only categories and operating points with validated low false-positive risk.
5. Production monitoring: track score drift, action rates, appeal outcomes, latency, and false-negative audits.

Calibrate a policy from labeled representative traffic:

```powershell
python scripts\calibrate_guardrail_policy.py `
  --features outputs\representative_features.npz `
  --metadata data\processed\representative_traffic.csv `
  --detector models\aegis\my_detector.npz `
  --output-policy configs\my_calibrated_policy.json `
  --output-summary outputs\my_calibration_summary.json `
  --max-fpr 0.01 `
  --min-recall 0.90
```

Monitor shadow or live decisions:

```powershell
python scripts\monitor_guardrail_decisions.py `
  --decisions outputs\my_guardrail_decisions.csv `
  --output outputs\my_guardrail_monitoring_summary.json
```

Run a local feature-vector sidecar for integrations that prefer HTTP:

```powershell
python scripts\serve_guardrail_features.py `
  --detector models\aegis\my_detector.npz `
  --host 127.0.0.1 `
  --port 8765
```

The sidecar accepts `POST /v1/guard` JSON payloads with `features` plus optional
`sample_ids`, `request_ids`, `prompt_sha256`, and `modalities`. It does not
extract embeddings itself; that boundary keeps AEGIS model-agnostic.

Run a local provider-backed sidecar when the guardrail process should call the
embedding provider itself:

```powershell
python scripts\serve_guardrail_provider.py `
  --provider my_package.my_provider:provider `
  --detector models\aegis\my_detector.npz `
  --host 127.0.0.1 `
  --port 8766 `
  --require-matching-provenance
```

This sidecar accepts `POST /v1/guard` payloads with `text`, optional
`image_path` or `image_paths`, and optional `request_id`, or a batch in
`requests`. The response includes action, verdict, risk score, provenance, and
hashes rather than raw prompt text.

## Production Validation Checklist

- Detector artifact loads with `allow_pickle=False`.
- Feature provenance matches the target embedding provider.
- Evaluation covers representative benign traffic, adversarial traffic, and ambiguous edge cases.
- Metrics meet the release gate for each deployment scope.
- Adaptive prompt tests and response-level attack-success proxy are refreshed for the target model.
- Privacy-safe logging is enabled; raw harmful prompts and generated harmful responses are not stored by default.
- Deployment is layered with output-side safety checks and downstream policy enforcement.
- Rollback is available if false positives, false negatives, or latency exceed limits.

## Release Gate

Build the requirement-by-requirement evidence report:

```powershell
python scripts\build_production_evidence_report.py
```

Run the local gate against the current controlled-deployment evidence bundle:

```powershell
python scripts\validate_production_readiness.py `
  --summary-json outputs\aegis_classify_multimodal_litmus_v1_summary.json `
  --detector models\aegis\aegis_qwen25vl3b_text_detector.npz `
  --manifest docs\reproducibility_manifest.json `
  --scope controlled_deployment
```

For a broader multi-model release, make the gate stricter by requiring multiple
summary files and explicit model families:

```powershell
python scripts\validate_production_readiness.py `
  --summary-json outputs\qwen_summary.json outputs\llava_summary.json `
  --min-summary-files 2 `
  --min-model-families 2 `
  --min-sources 2 `
  --required-model-family qwen25_vl `
  --required-model-family llava_onevision `
  --required-source qwen_litmus `
  --required-source llava_holdout
```

Passing this gate validates only the supplied scope and evidence. It is not a
claim that the detector is universally safe across every MLLM, LLM, language,
domain, or deployment environment.

## Target Validation Bundles

Every target LLM or MLLM should have its own validation bundle. A bundle joins
the detector, evaluation summaries, policy, provider contract report,
calibration, monitoring baseline, manifest, and readiness-gate result.

```powershell
python scripts\validate_embedding_provider_contract.py `
  --provider my_package.my_provider:provider `
  --detector models\aegis\my_detector.npz `
  --output outputs\my_provider_contract.json `
  --required-modality image_text

python scripts\build_target_validation_bundle.py `
  --target-name my_target_model `
  --model-family custom `
  --model-id vendor/my-target-model `
  --intended-modality image_text `
  --detector models\aegis\my_detector.npz `
  --summary-json outputs\my_target_eval_summary.json `
  --policy configs\my_calibrated_policy.json `
  --provider-contract outputs\my_provider_contract.json `
  --calibration-summary outputs\my_calibration_summary.json `
  --monitoring-summary outputs\my_monitoring_summary.json `
  --robustness-summary outputs\my_prompt_robustness_summary.csv `
  --attack-success-summary outputs\my_guardrail_asr_summary.csv `
  --json-output docs\targets\my_target_model.json `
  --markdown-output docs\targets\my_target_model.md

python scripts\validate_target_release.py `
  --bundle docs\targets\my_target_model.json `
  --require-labeled-monitoring `
  --require-robustness-evidence `
  --require-attack-success-evidence `
  --required-provider-modality image_text

python scripts\validate_target_registry.py `
  --registry docs\target_validation_registry.json `
  --min-production-candidates 1 `
  --required-model-family custom `
  --required-target-modality image_text `
  --require-labeled-monitoring `
  --require-robustness-evidence `
  --require-attack-success-evidence `
  --required-provider-modality image_text

python scripts\build_target_deployment_package.py `
  --bundle docs\targets\my_target_model.json `
  --output-dir outputs\deployment_packages\my_target_model `
  --require-labeled-monitoring `
  --require-robustness-evidence `
  --require-attack-success-evidence

python scripts\validate_target_deployment_package.py `
  --package-dir outputs\deployment_packages\my_target_model `
  --require-labeled-monitoring `
  --require-robustness-evidence `
  --require-attack-success-evidence
```

Use `--intended-modality text`, `--required-modality text`,
`--required-provider-modality text`, and `--required-target-modality text` for
text-only LLM deployments. Target bundles record the intended request shape,
release validation enforces matching provider-contract coverage by default, and
registry validation can require portfolio coverage across modalities such as
`text` and `image_text`. Strict release validation can also require per-target
robustness and guarded attack-success evidence. Deployment packages then copy
the approved detector, policy, provider contract, calibration, monitoring,
summary, robustness, and attack-success artifacts into one checksum-validated
directory with a portable `bundle.json`.

The registry at `docs\target_validation_registry.json` is the auditable list of
target models that have been validated. A target can be called a production
candidate only when its bundle includes a passing gate, provider contract,
calibrated policy, calibration summary, and monitoring baseline.
The release validator reopens the bundle artifacts and fails if the detector,
provider contract, policy threshold, calibration summary, monitoring baseline,
evaluation summaries, or reproducibility manifest disagree.
The registry validator then runs that release audit for every registered
production candidate and enforces the requested target, modality, and
model-family coverage.

## Current Status

The current target registry contains two `production_candidate` controlled
targets: `docs\targets\qwen25vl3b_controlled.md` for Qwen2.5-VL and
`docs\targets\llava_onevision_05b_controlled.md` for local LLaVA-OneVision
0.5B. Each includes a passing gate, live embedding provider contract, calibrated
policy, calibration summary, monitoring baseline, manifest entry, and release
audit. Both provider contracts record `image_text` request coverage, so the
registry audit now enforces multimodal provider coverage directly from the
target bundles' declared `intended_modalities`. Qwen carries prompt
robustness/adaptive evidence, while LLaVA carries attack-family robustness and
guarded response attack-success reduction evidence. Portable deployment packages
exist under `outputs\deployment_packages\qwen25vl3b_controlled` and
`outputs\deployment_packages\llava_onevision_05b_controlled`, and both pass
checksum plus packaged-release validation.

This means AEGIS can be deployed as an input-side guardrail for those validated
targets. It is not yet a universal production guardrail because broader natural
traffic calibration is needed before high-impact rollout, the current production
candidate registry covers `image_text` but not a real text-only LLM target,
Qwen does not yet have target-specific guarded attack-success evidence, and each
new LLM/MLLM embedding space requires its own provider contract, trained
detector artifact, calibrated policy, release audit, robustness evidence,
attack-success evidence, and target validation bundle.
