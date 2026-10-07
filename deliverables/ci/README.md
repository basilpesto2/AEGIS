# Continuous verification

`run_checks.py` is the local and CI entry point. It runs the research-pipeline unit
tests and the deliverable-integrity audit without
mutating committed artifacts.

```powershell
python deliverables/ci/run_checks.py
```

The command runs `deliverables/pipeline/tests/`, followed by the read-only,
schema-aware artifact audit. `workflow.example.yml` is an example workflow definition;
copy and review it under `.github/workflows/` if repository-hosted CI is desired. It
is not an enabled workflow.
