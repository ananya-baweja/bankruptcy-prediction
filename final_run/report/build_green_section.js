// Renders only the Green AI section to a .docx (for checking it, and because the full report needs
// analysis outputs that live on the Drive):  node build_green_section.js [out.docx]
const fs = require("fs");
const L = require("./lib.js");
const G = require("./green_section.js");
const { Document, Packer } = L.d;

if (!G.available()) { console.error(`no green results in ${G.GREEN}`); process.exit(1); }
const doc = new Document({
  creator: "DL/NLP project team", title: "Green AI: accuracy versus energy cost",
  styles: L.styles, numbering: L.numbering,
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 },
    margin: { top: 1300, right: 1300, bottom: 1300, left: 1300 } } }, children: G.build(L) }],
});
const out = process.argv[2] || `${G.GREEN}/Green_AI_section.docx`;
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out, b.length, "bytes"); });
