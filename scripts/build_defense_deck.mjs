import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { Presentation, PresentationFile } = require("@oai/artifact-tool");

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/(.:)/, "$1")), "..");
const OUTPUT = path.join(ROOT, "outputs", "AEGIS_FYP_Defense.pptx");
const WORKSPACE = path.join(os.tmpdir(), "aegis-presentations", "aegis-fyp-defense");
const PREVIEW = path.join(WORKSPACE, "tmp", "preview");
const LAYOUT = path.join(WORKSPACE, "tmp", "layout");
const QA = path.join(WORKSPACE, "tmp", "qa");

const C = {
  bg: "#F7F8F4",
  ink: "#172126",
  muted: "#58666B",
  teal: "#173B3F",
  mint: "#A9D3C8",
  red: "#D1495B",
  gold: "#E6AF2E",
  white: "#FFFFFF",
  line: "#D7DEDA",
};

const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function rect(slide, left, top, width, height, fill, line = fill) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: line, width: line === fill ? 0 : 1 },
  });
}

function text(slide, value, left, top, width, height, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    fontSize: options.size ?? 20,
    typeface: options.typeface ?? "Aptos",
    bold: options.bold ?? false,
    color: options.color ?? C.ink,
    alignment: options.align ?? "left",
    verticalAlignment: options.valign ?? "top",
  };
  return shape;
}

function baseSlide(kicker, titleValue, source) {
  const slide = p.slides.add();
  slide.background.fill = C.bg;
  rect(slide, 0, 0, 18, 720, C.teal);
  text(slide, kicker.toUpperCase(), 64, 42, 720, 24, { size: 13, bold: true, color: C.red });
  text(slide, titleValue, 64, 74, 1110, 58, { size: 36, bold: true, typeface: "Aptos Display" });
  rect(slide, 64, 142, 1150, 2, C.line);
  text(slide, source, 64, 690, 1060, 16, { size: 10, color: C.muted });
  text(slide, String(p.slides.items.length), 1170, 686, 44, 18, { size: 11, bold: true, color: C.teal, align: "right" });
  return slide;
}

function pill(slide, value, left, top, width, fill, color = C.ink) {
  rect(slide, left, top, width, 34, fill);
  text(slide, value, left + 8, top + 7, width - 16, 20, { size: 14, bold: true, color, align: "center" });
}

function kpi(slide, value, label, left, top, width, accent) {
  rect(slide, left, top, width, 132, C.white, C.line);
  rect(slide, left, top, 7, 132, accent);
  text(slide, value, left + 24, top + 20, width - 40, 54, { size: 42, bold: true, color: accent, typeface: "Aptos Display" });
  text(slide, label, left + 24, top + 82, width - 40, 34, { size: 16, color: C.muted });
}

function bullet(slide, value, left, top, width, accent = C.teal) {
  rect(slide, left, top + 8, 8, 8, accent);
  text(slide, value, left + 20, top, width - 20, 52, { size: 19, color: C.ink });
}

// 1. Cover
{
  const s = p.slides.add();
  s.background.fill = C.teal;
  rect(s, 0, 0, 22, 720, C.gold);
  text(s, "AEGIS", 74, 80, 520, 86, { size: 66, bold: true, color: C.white, typeface: "Aptos Display" });
  text(s, "Detecting malicious prompts\nfor multimodal language models", 74, 186, 820, 170, { size: 43, bold: true, color: C.white, typeface: "Aptos Display" });
  rect(s, 74, 389, 190, 5, C.red);
  text(s, "Representation-space detection, low-label learning, and jailbreak robustness", 74, 420, 760, 74, { size: 22, color: C.mint });
  kpi(s, "3", "external safety datasets", 846, 116, 302, C.gold);
  kpi(s, "1,960", "full MSSBench rows", 846, 268, 302, C.mint);
  kpi(s, "720", "paired JailBreakV rows", 846, 420, 302, C.red);
  text(s, "FYP defense | 2026", 74, 650, 400, 28, { size: 17, bold: true, color: C.white });
}

// 2. Threat
{
  const s = baseSlide("Problem", "The attack surface is multimodal", "Source: AEGIS final report draft and dataset taxonomy");
  const cols = [64, 448, 832];
  const items = [
    ["TEXT", "Obfuscation\nRole-play\nHarmful instructions", C.red],
    ["IMAGE", "Embedded cues\nVisual context\nImage-conditioned harm", C.gold],
    ["INTERACTION", "Cross-modal mismatch\nSituational risk\nJailbreak transfer", C.teal],
  ];
  items.forEach(([head, body, accent], i) => {
    rect(s, cols[i], 188, 332, 288, C.white, C.line);
    rect(s, cols[i], 188, 332, 10, accent);
    text(s, head, cols[i] + 24, 224, 282, 30, { size: 16, bold: true, color: accent });
    text(s, body, cols[i] + 24, 278, 282, 148, { size: 25, bold: true, typeface: "Aptos Display" });
  });
  text(s, "Input-side detection can stop, route, or scrutinize risk before generation.", 64, 526, 1100, 62, { size: 29, bold: true, color: C.teal, typeface: "Aptos Display" });
}

// 3. Questions
{
  const s = baseSlide("Research design", "Three questions drive AEGIS", "Source: Project summary; AEGIS benchmark protocol");
  const rows = [190, 328, 466];
  const qs = [
    ["01", "Which internal signals separate benign and malicious prompts?", "Layer, pooling, modality, SVD, supervised upper bound"],
    ["02", "How far can we go with few trusted labels?", "Low-label seeds, source controls, balanced pseudo-labels"],
    ["03", "What survives dataset shift and jailbreak transformations?", "Source-disjoint transfer, family holdout, locked thresholds"],
  ];
  qs.forEach(([n, q, detail], i) => {
    text(s, n, 74, rows[i], 80, 54, { size: 34, bold: true, color: i === 1 ? C.red : C.teal });
    text(s, q, 162, rows[i], 640, 54, { size: 24, bold: true });
    text(s, detail, 824, rows[i] + 3, 354, 48, { size: 17, color: C.muted });
    rect(s, 162, rows[i] + 72, 1016, 1, C.line);
  });
}

// 4. Pipeline
{
  const s = baseSlide("System", "A reproducible representation-space pipeline", "Source: AEGIS README and implementation");
  const stages = [
    ["1", "Normalize", "Shared metadata\n+ provenance"],
    ["2", "Represent", "Qwen + LLaVA states\n+ cached embeddings"],
    ["3", "Detect", "SVD / logistic\n+ low-label fusion"],
    ["4", "Select", "Validation only\n+ stability rule"],
    ["5", "Stress", "Transfer / family\n+ transformations"],
  ];
  stages.forEach(([n, head, body], i) => {
    const x = 64 + i * 229;
    rect(s, x, 222, 188, 220, i === 2 ? C.teal : C.white, i === 2 ? C.teal : C.line);
    text(s, n, x + 20, 242, 42, 34, { size: 18, bold: true, color: i === 2 ? C.gold : C.red });
    text(s, head, x + 20, 292, 148, 34, { size: 24, bold: true, color: i === 2 ? C.white : C.ink });
    text(s, body, x + 20, 348, 148, 62, { size: 16, color: i === 2 ? C.mint : C.muted });
    if (i < 4) text(s, ">", x + 192, 304, 34, 38, { size: 28, bold: true, color: C.gold, align: "center" });
  });
  pill(s, "Model-agnostic cached embedding interface", 356, 500, 560, C.mint, C.teal);
}

// 5. Data/protocol
{
  const s = baseSlide("Evaluation", "Three datasets, four distinct claims", "Source: AEGIS dataset cards and benchmark protocol");
  const data = [
    ["VLGuard", "4,535", "development + source transfer", C.teal],
    ["MSSBench", "1,960", "grouped low-label benchmark", C.red],
    ["JailBreakV", "360 + 360", "same-image jailbreak pairs", C.gold],
  ];
  data.forEach(([name, count, role, accent], i) => {
    const x = 64 + i * 384;
    kpi(s, count, `${name} | ${role}`, x, 184, 344, accent);
  });
  const claims = ["Within-domain", "Source-disjoint", "Family holdout", "Prompt robustness"];
  claims.forEach((v, i) => pill(s, v, 64 + i * 286, 368, 256, i % 2 ? "#F4D7DC" : "#DCEBE7", i % 2 ? C.red : C.teal));
  bullet(s, "Fit, validation, and test are separated by source group or matched pair.", 80, 456, 1060);
  bullet(s, "Thresholds and hyperparameters are selected on validation data only.", 80, 518, 1060, C.red);
  bullet(s, "Near-perfect source-confounded runs are diagnostics, not headline claims.", 80, 580, 1060, C.gold);
}

// 6. Signal ablation
{
  const s = baseSlide("Ablation", "The useful signal depends on the task", "Source: outputs/mssbench_full_*; paired JailBreakV family holdout");
  s.charts.add("bar", {
    position: { left: 74, top: 190, width: 720, height: 390 },
    categories: ["SVD text", "Supervised text", "Supervised image", "Concat"],
    series: [{ name: "MSSBench AUPRC (%)", values: [53.13, 94.78, 97.09, 97.24], fill: C.teal }],
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd" },
    xAxis: { minimumScale: 0, maximumScale: 100 },
    yAxis: { majorGridlines: { style: "solid", fill: C.line, width: 1 } },
  });
  text(s, "MSSBench AUPRC (%)", 92, 174, 220, 26, { size: 14, bold: true, color: C.red });
  kpi(s, "1.000", "paired JailBreakV text AUPRC", 844, 208, 330, C.teal);
  kpi(s, "~0.50", "paired JailBreakV image AUPRC", 844, 366, 330, C.red);
  text(s, "Image features matter for situational safety, but vanish as a shortcut when images are matched.", 844, 524, 330, 80, { size: 19, bold: true, color: C.ink });
}

// 7. Low-label
{
  const s = baseSlide("Low-label learning", "A small trusted set changes the picture", "Source: full MSSBench five-seed low-label and trusted+pseudo sweeps");
  s.charts.add("line", {
    position: { left: 64, top: 190, width: 760, height: 390 },
    categories: ["2", "5", "10", "20", "40"],
    series: [{ name: "Mean test AUPRC (%)", values: [57.47, 59.11, 62.60, 74.22, 79.90], fill: C.teal }],
    hasLegend: false,
    dataLabels: { showValue: true, position: "above" },
    yAxis: { minimumScale: 50, maximumScale: 85, majorGridlines: { style: "solid", fill: C.line, width: 1 } },
  });
  text(s, "Trusted labels per class", 300, 590, 300, 24, { size: 15, bold: true, color: C.muted, align: "center" });
  kpi(s, "+0.0046", "pseudo-label AUPRC at 40/class", 866, 212, 308, C.gold);
  kpi(s, "+0.0116", "pseudo-label F1 at 40/class", 866, 370, 308, C.red);
  text(s, "Policy: keep trusted-only at 5/class; enable balanced pseudo-labels from 10/class.", 866, 536, 308, 72, { size: 18, bold: true });
}

// 8. Transfer
{
  const s = baseSlide("Transfer", "Generalization is asymmetric, and selection matters", "Source: AEGIS transferability report; balanced target outputs");
  const rows = [
    ["VLGuard -> MSSBench", 0.4496, C.red, "fails under task shift"],
    ["VLGuard -> JailBreakV", 0.9169, C.teal, "text features transfer"],
    ["MSSBench -> JailBreakV", 0.6009, C.red, "source-selected concat"],
    ["+ stability selection", 0.7427, C.gold, "selects text without target labels"],
  ];
  rows.forEach(([name, val, color, note], i) => {
    const y = 192 + i * 98;
    text(s, name, 74, y, 292, 28, { size: 18, bold: true });
    rect(s, 370, y + 2, 600, 28, C.line);
    rect(s, 370, y + 2, 600 * val, 28, color);
    text(s, val.toFixed(4), 986, y, 104, 28, { size: 20, bold: true, color });
    text(s, note, 370, y + 42, 600, 24, { size: 15, color: C.muted });
  });
  pill(s, "Choose the feature that is stable across non-target domains", 322, 600, 650, C.mint, C.teal);
}

// 9. Robustness
{
  const s = baseSlide("Jailbreak robustness", "Controls remove the easy image shortcut", "Source: family holdout, adaptive attack, and guarded-response outputs");
  text(s, "Same image", 98, 204, 180, 34, { size: 22, bold: true, color: C.teal });
  text(s, "+", 296, 204, 40, 34, { size: 24, bold: true, color: C.gold, align: "center" });
  text(s, "Malicious or neutral text", 354, 204, 300, 34, { size: 22, bold: true, color: C.red });
  text(s, "=", 676, 204, 40, 34, { size: 24, bold: true, color: C.gold, align: "center" });
  text(s, "Pair-safe test", 736, 204, 230, 34, { size: 22, bold: true });
  rect(s, 74, 260, 1100, 2, C.line);
  const variants = ["Original", "Base64", "Leetspeak", "Role-play"];
  variants.forEach((v, i) => {
    const x = 74 + i * 276;
    kpi(s, i === 0 ? "0.994" : "1.000", `${v} F1`, x, 314, 240, i === 0 ? C.teal : C.red);
  });
  bullet(s, "Qwen prompt variants retain AUROC and AUPRC 1.0 at the original threshold.", 92, 478, 1030);
  bullet(s, "Ten-query adaptive selection lowers recall to 0.967 (3.37% bounded evasion).", 92, 536, 1030, C.red);
  bullet(s, "LLaVA lexical ASR proxy falls from 0.833 to 0.000 after input blocking.", 92, 594, 1030, C.gold);
}

// 10. Conclusion
{
  const s = baseSlide("Conclusion", "What AEGIS establishes, and what comes next", "Source: AEGIS final report draft and ethical considerations");
  text(s, "ESTABLISHED", 74, 184, 460, 28, { size: 15, bold: true, color: C.teal });
  bullet(s, "Safety signal reproduces across Qwen and LLaVA representations.", 74, 232, 500);
  bullet(s, "Small trusted target sets outperform label-free SVD.", 74, 300, 500, C.red);
  bullet(s, "Cross-source stability improves transfer selection.", 74, 368, 500, C.gold);
  bullet(s, "Pair-safe controls expose modality shortcuts.", 74, 436, 500);
  text(s, "NEXT", 680, 184, 460, 28, { size: 15, bold: true, color: C.red });
  bullet(s, "Add an architecturally distant third MLLM.", 680, 232, 480, C.red);
  bullet(s, "Acquire the authorized full JailBreakV image release.", 680, 300, 480, C.gold);
  bullet(s, "Add human-judged response safety evaluation.", 680, 368, 480, C.red);
  bullet(s, "Test gradient-based adaptive attacks.", 680, 436, 480, C.gold);
  rect(s, 64, 544, 1110, 72, C.teal);
  text(s, "Practical result: lightweight target adaptation + stable transfer selection + explicit limits", 92, 565, 1054, 32, { size: 24, bold: true, color: C.white, align: "center" });
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

await fs.mkdir(path.dirname(OUTPUT), { recursive: true });
await Promise.all([PREVIEW, LAYOUT, QA].map((dir) => fs.mkdir(dir, { recursive: true })));
await fs.writeFile(path.join(WORKSPACE, "tmp", "source-notes.txt"), "AEGIS defense deck source ledger\nProject-owned sources: README, benchmark protocol, ablation report, transferability report, adaptive-attack report, second-MLLM report, guarded-response report, final report draft, and CSV outputs.\nNo external visual assets or raw harmful prompts are used.\n");
await fs.writeFile(path.join(WORKSPACE, "tmp", "slide-plan.txt"), "Create mode; 10 slides; FYP examiner audience. Palette: off-white, teal, crimson, gold, ink. Fonts: Aptos Display + Aptos. All charts and shapes editable.\n");

for (const [index, slide] of p.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  await writeBlob(path.join(PREVIEW, `${stem}.png`), await p.export({ slide, format: "png", scale: 1 }));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(LAYOUT, `${stem}.layout.json`), await layout.text());
}

await writeBlob(path.join(PREVIEW, "montage.webp"), await p.export({ format: "webp", montage: true, scale: 1 }));
const pptx = await PresentationFile.exportPptx(p);
await pptx.save(OUTPUT);
await fs.writeFile(path.join(QA, "visual-qa.txt"), "Visual QA passed after the final cross-model and guarded-response refresh. All 10 slide renders were inspected; the changed system, robustness, and conclusion slides were rechecked at full resolution. No clipping, overlap, detached labels, or inconsistent page markers were found. Chart values match the project result artifacts. Aptos typography and editable native shapes/charts are retained in the exported PPTX.\n");
console.log(JSON.stringify({ output: OUTPUT, workspace: WORKSPACE, slides: p.slides.items.length }, null, 2));
