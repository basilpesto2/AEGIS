# Continuous verification

`run_checks.py` is the local and CI entry point for the Final Report evidence. It
runs independently of the product CLI/GUI test suite and does not mutate committed
artifacts.

```powershell
python deliverables/ci/run_checks.py
```

The command runs the research detector unit tests followed by the read-only,
schema-aware artifact audit. `workflow.example.yml` is an example workflow definition;
copy and review it under `.github/workflows/` if repository-hosted CI is desired. It is
not itself an enabled workflow.

`reproduction_validation_2026-07-31.json` records the isolated end-to-end research
workflow, benchmark rebuild, workbook inspection, and paired-response evaluator checks
performed while refreshing the Final Report evidence. Temporary outputs were removed;
the committed historical scores were not overwritten.
