# Continuous verification

`run_checks.py` is the CI entry point for the deliverables. It deliberately
lives inside `deliverables/` so the complete handoff remains in the single folder
requested for this project.

```powershell
python deliverables/ci/run_checks.py
```

The command runs the detector unit tests followed by the 35-check artifact audit.
