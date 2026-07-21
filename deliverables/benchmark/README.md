# Annotated benchmark

The canonical dataset is `data/benchmark.csv`; `benchmark.xlsx` is a formatted review
copy with summary formulas and a data dictionary. Generated PNG cue cards live in
`images/`. `data/asset_manifest.json` records counts and content hashes.

Rebuild in two stages:

```powershell
python deliverables/annotated_benchmark/build_assets.py
node deliverables/annotated_benchmark/build_workbook.mjs
```

Review `DATASHEET.md` and `ANNOTATION_GUIDELINES.md` before using or extending the
benchmark. The data is project-internal.
