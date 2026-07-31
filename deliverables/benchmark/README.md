# Annotated benchmark

The canonical dataset is `data/benchmark.csv`; `benchmark.xlsx` is a formatted review
copy with summary formulas and a data dictionary. Generated PNG cue cards live in
`images/`. `data/asset_manifest.json` records counts and content hashes.

Rebuild in two stages. The Python stage produces the canonical CSV/JSON data and cue
cards; the Node stage produces only the formatted workbook and visual QA:

```powershell
python deliverables/benchmark/build_assets.py
node deliverables/benchmark/build_workbook.mjs
```

Review `DATASHEET.md` and `ANNOTATION_GUIDELINES.md` before using or extending the
benchmark. The data is project-internal.

The workbook's Summary sheet explicitly identifies this as a synthetic, redacted
benchmark. It is dataset evidence and must not be cited as measured performance of the
currently deployed LLaVA tuned-v3 detector.

The committed workbook and QA previews were regenerated and visually reviewed on
2026-07-31 with the bundled `@oai/artifact-tool` 2.8.36 runtime. The inspection
transcript in `qa/workbook_inspect.ndjson` confirms that the summary formulas reconcile
the dataset and contain no spreadsheet error values.
