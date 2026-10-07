# Annotated benchmark

The canonical dataset is `data/benchmark.csv`; `benchmark.xlsx` is a formatted review
copy with summary formulas and a data dictionary. Generated PNG cue cards live in
`images/`. `data/asset_manifest.json` records counts and content hashes.

Rebuild in two stages. The Python stage produces the canonical CSV/JSON data and cue
cards; the Node stage produces only the formatted workbook and visual QA.

The Node stage requires the bundled `@oai/artifact-tool` package to be available to
the Node runtime's module resolution. Use that prepared runtime when rebuilding the
workbook. The committed `benchmark.xlsx` and `qa/` previews are directly usable
without this package or regeneration.

Run both stages from the repository root:

```powershell
python deliverables/benchmark/build_assets.py
node deliverables/benchmark/build_workbook.mjs
```

Review `DATASHEET.md` and `ANNOTATION_GUIDELINES.md` before using or extending the
benchmark. The data is project-internal.

The workbook's Summary sheet explicitly identifies this as a synthetic, redacted
benchmark. It is dataset evidence, not detector-training data, and must not be cited as
measured performance of any maintained LLaVA or Qwen target.

The committed workbook and QA previews were regenerated and visually reviewed on
2026-08-30 with the bundled `@oai/artifact-tool` runtime. The inspection
transcript in `qa/workbook_inspect.ndjson` confirms that the summary formulas reconcile
the dataset and contain no spreadsheet error values.
