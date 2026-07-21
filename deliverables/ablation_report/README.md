# Ablation and transfer report

`run_ablation.py` regenerates every table in `REPORT.md` from the committed benchmark
and feature bundle. The report separates:

1. freshly executed deterministic smoke-test results, which verify the experiment
   software and reporting path; and
2. legacy MLLM results recovered from Git commit `3f6e46a`, whose underlying ignored
   output files are no longer present and therefore are not treated as rerun evidence.

Run:

```powershell
python deliverables/ablation_report/run_ablation.py
```

Outputs are written under `results/`. Replace the smoke feature bundle with aligned
MLLM features and pass `--features` to obtain reportable model evidence.
