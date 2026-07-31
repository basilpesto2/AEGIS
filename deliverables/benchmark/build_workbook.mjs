import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const root = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.join(root, "data");
const rows = JSON.parse(await fs.readFile(path.join(dataDir, "benchmark_rows.json"), "utf8"));
if (!Array.isArray(rows) || rows.length === 0) {
  throw new Error("benchmark_rows.json is empty; run build_assets.py first");
}

const headers = Object.keys(rows[0]);
const matrix = [headers, ...rows.map((row) => headers.map((header) => row[header] ?? ""))];
const lastColumn = columnName(headers.length);
const lastRow = matrix.length;
const tailStartRow = Math.max(2, lastRow - 11);
const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const benchmark = workbook.worksheets.add("Benchmark");
const dictionary = workbook.worksheets.add("Data Dictionary");

benchmark.getRangeByIndexes(0, 0, matrix.length, headers.length).values = matrix;
benchmark.tables.add(`A1:${lastColumn}${lastRow}`, true, "BenchmarkTable");
benchmark.freezePanes.freezeRows(1);
benchmark.freezePanes.freezeColumns(4);
benchmark.showGridLines = false;
benchmark.getRange(`A1:${lastColumn}1`).format = {
  fill: "#17324D",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  verticalAlignment: "center",
};
benchmark.getRange(`A2:${lastColumn}${lastRow}`).format = {
  font: { color: "#172B3A" },
  verticalAlignment: "top",
};
benchmark.getRange(`E2:E${lastRow}`).format.numberFormat = "0";
benchmark.getRange(`P2:P${lastRow}`).format.numberFormat = "0.00";
benchmark.getRange(`A1:${lastColumn}1`).format.rowHeight = 34;
benchmark.getRange("A:A").format.columnWidth = 25;
benchmark.getRange("B:F").format.columnWidth = 15;
benchmark.getRange("G:G").format.columnWidth = 54;
benchmark.getRange("H:N").format.columnWidth = 24;
benchmark.getRange("O:X").format.columnWidth = 30;
benchmark.getRange(`A1:${lastColumn}${lastRow}`).format.wrapText = true;
benchmark.getRange(`A2:${lastColumn}${lastRow}`).format.autofitRows();
benchmark.getRange(`D2:D${lastRow}`).conditionalFormats.add("containsText", {
  text: "malicious",
  format: { fill: "#FCE8E6", font: { color: "#A61B1B", bold: true } },
});
benchmark.getRange(`D2:D${lastRow}`).conditionalFormats.add("containsText", {
  text: "benign",
  format: { fill: "#E7F5EC", font: { color: "#176B3A" } },
});

summary.showGridLines = false;
summary.getRange("A1:H1").merge();
summary.getRange("A1").values = [["AEGIS Synthetic Redacted Benchmark v1.0.0"]];
summary.getRange("A1:H1").format = {
  fill: "#17324D",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  rowHeight: 30,
  verticalAlignment: "center",
};
summary.getRange("A3:B7").values = [
  ["Quality check", "Value"],
  ["Total rows", null],
  ["Benign rows", null],
  ["Malicious rows", null],
  ["Balanced?", null],
];
summary.getRange("B4").formulas = [[`=COUNTA('Benchmark'!$A$2:$A$${lastRow})`]];
summary.getRange("B5").formulas = [[`=COUNTIF('Benchmark'!$D$2:$D$${lastRow},"benign")`]];
summary.getRange("B6").formulas = [[`=COUNTIF('Benchmark'!$D$2:$D$${lastRow},"malicious")`]];
summary.getRange("B7").formulas = [["=IF(B5=B6,\"PASS\",\"FAIL\")"]];

summary.getRange("D3:F7").values = [
  ["Split", "Benign", "Malicious"],
  ["train", null, null],
  ["validation", null, null],
  ["test", null, null],
  ["Total", null, null],
];
for (let row = 4; row <= 6; row += 1) {
  summary.getRange(`E${row}`).formulas = [[`=COUNTIFS('Benchmark'!$B$2:$B$${lastRow},D${row},'Benchmark'!$D$2:$D$${lastRow},"benign")`]];
  summary.getRange(`F${row}`).formulas = [[`=COUNTIFS('Benchmark'!$B$2:$B$${lastRow},D${row},'Benchmark'!$D$2:$D$${lastRow},"malicious")`]];
}
summary.getRange("E7").formulas = [["=SUM(E4:E6)"]];
summary.getRange("F7").formulas = [["=SUM(F4:F6)"]];

const attackStyles = [
  "direct_policy_violation",
  "role_play",
  "text_obfuscation",
  "instruction_hierarchy",
  "image_embedded_instruction",
  "cross_modal_misalignment",
  "indirect_prompt_injection",
  "safety_camouflage",
];
summary.getRange("A10:B18").values = [
  ["Malicious attack style", "Rows"],
  ...attackStyles.map((style) => [style, null]),
];
for (let row = 11; row <= 18; row += 1) {
  summary.getRange(`B${row}`).formulas = [[`=COUNTIF('Benchmark'!$J$2:$J$${lastRow},A${row})`]];
}
summary.getRange("D10:E13").values = [
  ["Modality", "Rows"],
  ["text", null],
  ["image_text", null],
  ["Total", null],
];
summary.getRange("E11").formulas = [[`=COUNTIF('Benchmark'!$F$2:$F$${lastRow},D11)`]];
summary.getRange("E12").formulas = [[`=COUNTIF('Benchmark'!$F$2:$F$${lastRow},D12)`]];
summary.getRange("E13").formulas = [["=SUM(E11:E12)"]];

for (const range of ["A3:B3", "D3:F3", "A10:B10", "D10:E10"]) {
  summary.getRange(range).format = {
    fill: "#315B7D",
    font: { bold: true, color: "#FFFFFF" },
  };
}
summary.getRange("A3:F18").format.borders = { preset: "inside", style: "thin", color: "#D9E2EA" };
summary.getRange("A:A").format.columnWidth = 32;
summary.getRange("B:B").format.columnWidth = 14;
summary.getRange("D:F").format.columnWidth = 17;
summary.getRange("B7").conditionalFormats.add("containsText", {
  text: "PASS",
  format: { fill: "#E7F5EC", font: { color: "#176B3A", bold: true } },
});
summary.getRange("A20:H20").merge();
summary.getRange("A21:H22").merge();
summary.getRange("A20").values = [["Evidence boundary"]];
summary.getRange("A21").values = [[
  "This workbook documents a synthetic, safety-redacted dataset. It is not a performance report for the deployed LLaVA tuned-v3 detector; current live-system evidence is recorded under deliverables/runtime_validation/."
]];
summary.getRange("A20:H20").format = {
  fill: "#315B7D",
  font: { bold: true, color: "#FFFFFF" },
};
summary.getRange("A21:H22").format = {
  fill: "#EEF4F8",
  font: { color: "#172B3A", italic: true },
  wrapText: true,
  verticalAlignment: "top",
};
summary.getRange("A21:H22").format.autofitRows();
summary.freezePanes.freezeRows(1);

const descriptions = {
  sample_id: "Unique benchmark row identifier.",
  split: "Leakage-controlled train, validation, or test assignment.",
  group_id: "Matched benign/adversarial pair identifier; groups do not cross splits.",
  label: "Human-readable binary decision.",
  label_id: "Machine-readable label: benign=0, malicious=1.",
  modality: "text or image_text.",
  prompt_text: "User-visible text prompt; harmful operational details are redacted.",
  image_path: "Relative generated PNG path; empty for text-only rows.",
  harm_category: "Requested policy harm, or none for benign rows.",
  attack_style: "Observed adversarial mechanism, or none for benign rows.",
  paired_attack_style: "Attack family represented by both rows in the matched pair.",
  scenario_theme: "Benign topic used to construct the matched scenario.",
  obfuscation_type: "Redaction/obfuscation marker.",
  policy_rationale: "One-sentence annotation justification.",
  annotation_method: "How the label was authored and validated.",
  annotation_confidence: "Confidence from 0 to 1.",
  annotation_status: "Synthetic annotation maturity marker.",
  pii_status: "Personal-information screening outcome.",
  content_safety: "Safe-handling state of the sample.",
  source: "Dataset provenance identifier.",
  license: "Current usage/release status.",
  benchmark_version: "Semantic dataset version.",
  prompt_sha256: "SHA-256 of normalized prompt text.",
  image_sha256: "SHA-256 of image bytes, blank for text-only rows.",
};
const dictionaryRows = [["Column", "Description", "Required"]];
for (const header of headers) {
  dictionaryRows.push([header, descriptions[header] ?? "See schema.json.", "yes"]);
}
dictionary.getRangeByIndexes(0, 0, dictionaryRows.length, 3).values = dictionaryRows;
dictionary.tables.add(`A1:C${dictionaryRows.length}`, true, "DictionaryTable");
dictionary.getRange("A1:C1").format = { fill: "#17324D", font: { bold: true, color: "#FFFFFF" } };
dictionary.getRange("A:A").format.columnWidth = 30;
dictionary.getRange("B:B").format.columnWidth = 72;
dictionary.getRange("C:C").format.columnWidth = 12;
dictionary.getRange(`A1:C${dictionaryRows.length}`).format.wrapText = true;
dictionary.getRange("A1:C1").format.rowHeight = 28;
dictionary.getRange(`A2:C${dictionaryRows.length}`).format.autofitRows();
dictionary.freezePanes.freezeRows(1);
dictionary.showGridLines = false;

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
const workbookPath = path.join(root, "benchmark.xlsx");
await xlsx.save(workbookPath);

const qaDir = path.join(root, "qa");
await fs.mkdir(qaDir, { recursive: true });
for (const previewSpec of [
  { sheetName: "Summary", range: "A1:H22", file: "summary.png" },
  { sheetName: "Benchmark", range: `A1:${lastColumn}12`, file: "benchmark_head.png" },
  { sheetName: "Benchmark", range: `A${tailStartRow}:${lastColumn}${lastRow}`, file: "benchmark_tail.png" },
  { sheetName: "Data Dictionary", range: `A1:C${dictionaryRows.length}`, file: "data_dictionary.png" },
]) {
  const preview = await workbook.render({ sheetName: previewSpec.sheetName, range: previewSpec.range, scale: 1, format: "png" });
  await fs.writeFile(path.join(qaDir, previewSpec.file), new Uint8Array(await preview.arrayBuffer()));
}

const check = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:H22",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 8,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(qaDir, "workbook_inspect.ndjson"), `${check.ndjson}\n${errors.ndjson}\n`, "utf8");
await fs.rm(`${workbookPath}.inspect.ndjson`, { force: true });
console.log(JSON.stringify({ rows: rows.length, columns: headers.length, source: "data/benchmark_rows.json", workbook: "benchmark.xlsx" }));


function columnName(count) {
  let value = count;
  let name = "";
  while (value > 0) {
    value -= 1;
    name = String.fromCharCode(65 + (value % 26)) + name;
    value = Math.floor(value / 26);
  }
  return name;
}
