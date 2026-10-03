// Shared helpers for building the final report with docx-js (style follows the progress report).
const fs = require("fs");
const d = require("docx");
const { Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType, AlignmentType, ImageRun,
        HeadingLevel, BorderStyle, LevelFormat, PageBreak } = d;

const PAGE_W = 11906, MARGIN = 1300, CONTENT_W = PAGE_W - 2 * MARGIN;   // A4, as the progress report
const FONT = "Calibri";

// inline markup: **bold**, *italic*
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 120, line: 276 },
  alignment: opts.align, keepNext: opts.keepNext, ...(opts.para || {}) });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)],
  spacing: { before: 300, after: 140 }, keepNext: true });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)],
  spacing: { before: 220, after: 100 }, keepNext: true });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)],
  spacing: { before: 160, after: 80 }, keepNext: true });
const Bullet = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text),
  spacing: { after: 80, line: 270 } });
const Num = (text, ref = "numbers") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(text),
  spacing: { after: 90, line: 270 } });
const Caption = (text) => new Paragraph({ children: runs(text, { italics: true, size: 19, color: "404040" }),
  spacing: { before: 60, after: 200 } });
const TableTitle = (label, text) => new Paragraph({ keepNext: true, spacing: { before: 160, after: 80 },
  children: [new TextRun({ text: label + " ", bold: true, size: 20 }), ...runs(text, { size: 20 })] });
const PageBreakP = () => new Paragraph({ children: [new PageBreak()] });

function Figure(path, widthIn, caption) {
  const buf = fs.readFileSync(path);
  // read PNG size from the IHDR chunk
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wpx = Math.round(widthIn * 96), hpx = Math.round(wpx * h / w);
  return [new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: hpx },
        altText: { title: caption, description: caption, name: caption } })] }),
    Caption(caption)];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };

// rows: array of arrays of strings (markup allowed). widths: relative weights.
function Tbl(header, rows, weights, opts = {}) {
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor(CONTENT_W * w / total));
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const size = opts.size || 19;
  const cell = (txt, i, isHead, bold) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: isHead ? { fill: "D9D9D9", type: ShadingType.CLEAR, color: "auto" }
      : (opts.shadeRows && opts.shadeRows(bold) ? { fill: "EEF3FA", type: ShadingType.CLEAR, color: "auto" } : undefined),
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: String(txt).split("\n").map((line) => new Paragraph({
      alignment: (i > 0 && opts.numeric && opts.numeric.includes(i)) ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: runs(line, { size, bold: isHead || undefined }), spacing: { after: 0, line: 252 } })),
  });
  const trs = header ? [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, i, true)) })] : [];
  rows.forEach((r) => trs.push(new TableRow({ cantSplit: true,
    children: r.map((c, i) => cell(c, i, false, r[0] && String(r[0]).startsWith("**"))) })));
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows: trs });
}

const numbering = { config: [
  { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 900, hanging: 270 } } } }] },
  ...["numbers", "numbers2", "numbers3", "numbers4"].map((r) => ({ reference: r, levels: [{ level: 0,
      format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] })),
] };

const styles = {
  default: { document: { run: { font: FONT, size: 22 } } },
  paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 30, bold: true, color: "2E74B5", font: FONT }, paragraph: { outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 25, bold: true, color: "2E74B5", font: FONT }, paragraph: { outlineLevel: 1 } },
    { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 23, bold: true, color: "1F4D78", font: FONT }, paragraph: { outlineLevel: 2 } },
  ],
};

// number formatting
const f3 = (x) => (x === undefined || x === null || Number.isNaN(x)) ? "n/a" : Number(x).toFixed(3);
const s3 = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3);
const pc = (x, dp = 0) => (100 * x).toFixed(dp) + "%";
const pv = (p) => p < 0.001 ? "< 0.001" : p.toFixed(3);

module.exports = { d, P, H1, H2, H3, Bullet, Num, Caption, TableTitle, PageBreakP, Figure, Tbl, numbering, styles,
  f3, s3, pc, pv, runs, CONTENT_W, PAGE_W, MARGIN, FONT };
