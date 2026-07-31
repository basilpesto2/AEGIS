# Continuous verification

`run_checks.py` is the local and CI entry point. It runs
independently of the product CLI/GUI test suite and does not mutate committed
artifacts.

```powershell
python deliverables/ci/run_checks.py
```

The command runs the research-pipeline unit tests followed by the read-only,
schema-aware artifact audit. `workflow.example.yml` is an example workflow definition;
copy and review it under `.github/workflows/` if repository-hosted CI is desired. It
is not an enabled workflow.
