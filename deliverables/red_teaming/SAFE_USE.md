# Safe-use and disclosure protocol

- Keep committed examples redacted and non-operational. Do not replace placeholders
  with real exploit code, harmful procedures, credentials, slurs, or personal data.
- Store higher-risk extensions in access-controlled infrastructure, never in ordinary
  Git history or public artifacts.
- Use bounded query budgets and record every transformation, seed, detector version,
  model revision, threshold, and selection rule.
- Do not send generated attacks to third-party services without authorization and a
  documented data-retention review.
- Treat a successful detector evasion as sensitive until the affected maintainer has
  had reasonable time to reproduce and mitigate it.
- Report aggregate results and redacted minimal reproductions. Coordinate disclosure
  when an evasion could materially increase real-world abuse.
- Screen results for differential false positives across language, topic, and protected
  groups; aggregate AUROC alone is not evidence of fairness.
