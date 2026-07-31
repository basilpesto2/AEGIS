# Safe-use and disclosure protocol

- Keep committed examples redacted and non-operational. Do not replace placeholders
  with real exploit code, harmful procedures, credentials, slurs, or personal data.
- Store higher-risk extensions in access-controlled infrastructure, never in ordinary
  Git history or public artifacts.
- Use bounded query budgets and preserve every transformation, seed, corpus/panel hash,
  feature-bundle hash, detector path and hash, model/tokenizer revision, preprocessing
  hash, threshold value and semantics, traffic mode, and selection rule.
- Never overwrite a historical score panel or reuse its filename for a different
  detector, threshold, model revision, preprocessing pipeline, or traffic mode.
- Distinguish effective runtime action from recommended action. In shadow mode,
  threshold-based blocking is counterfactual unless a downstream system actually
  reviewed or blocked the request.
- Do not send generated attacks to third-party services without authorization and a
  documented data-retention review.
- Treat a successful detector evasion as sensitive until the affected maintainer has
  had reasonable time to reproduce and mitigate it.
- Report aggregate results and redacted minimal reproductions. Coordinate disclosure
  when an evasion could materially increase real-world abuse.
- Screen results for differential false positives across language, topic, and protected
  groups; aggregate AUROC alone is not evidence of fairness.
