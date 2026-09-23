"""Extract page-level text from annual report PDFs, with OCR for scanned pages.

Step by step, for each PDF in data/interim/documents.csv:

1. Open with PyMuPDF and read each page's text layer.
2. If a page has fewer than ``min_chars_per_page`` characters it is probably a
   scanned image (common in older small-cap reports). Render it at ``dpi`` and
   run Tesseract OCR on it.
3. If Tesseract is not installed, the page is kept empty and marked
   ``needs_ocr`` so you can see how much text is missing.
4. Save data/interim/pages/<doc_id>.json:
   {doc_id, n_pages, n_text_pages, n_ocr_pages, n_empty_pages, pages: [{page, method, text}]}

Page text is saved separately from sections so that section rules can be
changed and re-run in seconds, without repeating slow OCR.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import pymupdf as fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm

from bpp.config import Paths

log = logging.getLogger(__name__)

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def normalize_page_text(text: str) -> str:
    """Unicode clean-up that keeps line breaks (section detection needs them)."""
    text = unicodedata.normalize("NFKC", text)
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = (text.replace("’", "'").replace("‘", "'").replace("“", '"')
                .replace("”", '"').replace("–", "-").replace("—", "-")
                .replace(" ", " ").replace("­", ""))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


_tesseract_ok: bool | None = None


def tesseract_available(cmd: str | None = None) -> bool:
    global _tesseract_ok
    if _tesseract_ok is not None:
        return _tesseract_ok
    try:
        import pytesseract
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        pytesseract.get_tesseract_version()
        _tesseract_ok = True
    except Exception:  # noqa: BLE001
        _tesseract_ok = False
        log.warning("Tesseract OCR not available: scanned pages will be marked needs_ocr. "
                    "See docs/01_setup.md to install it.")
    return _tesseract_ok


def ocr_page(page: "fitz.Page", dpi: int = 300, lang: str = "eng") -> str:
    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img, lang=lang)


def extract_pdf(pdf_path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    ecfg = cfg["extraction"]
    ocfg = ecfg["ocr"]
    min_chars = ecfg["min_chars_per_page"]
    can_ocr = ocfg["enabled"] and tesseract_available(ocfg.get("tesseract_cmd"))

    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            method = "text"
            if len(text.strip()) < min_chars:
                if can_ocr:
                    try:
                        text = ocr_page(page, ocfg["dpi"], ocfg["lang"])
                        method = "ocr"
                    except Exception as exc:  # noqa: BLE001
                        log.warning("OCR failed on %s page %d: %s", pdf_path.name, i + 1, exc)
                        method = "needs_ocr"
                else:
                    method = "needs_ocr" if page.get_images() else "empty"
            pages.append({"page": i + 1, "method": method, "text": normalize_page_text(text)})

    counts = pd.Series([p["method"] for p in pages]).value_counts().to_dict() if pages else {}
    return {
        "n_pages": len(pages),
        "n_text_pages": counts.get("text", 0),
        "n_ocr_pages": counts.get("ocr", 0),
        "n_needs_ocr_pages": counts.get("needs_ocr", 0),
        "n_empty_pages": counts.get("empty", 0),
        "n_chars": sum(len(p["text"]) for p in pages),
        "pages": pages,
    }


def run_extract_text(cfg: dict[str, Any], paths: Paths, doc_ids: list[str] | None = None,
                     force: bool = False) -> pd.DataFrame:
    from bpp.scrape.annual_reports import load_manifest, resolve_local_path

    man = load_manifest(paths)
    man = man[man["status"].isin(["downloaded", "registered"])]
    if doc_ids:
        man = man[man["doc_id"].isin(doc_ids)]
    summary = []
    for _, r in tqdm(man.iterrows(), total=len(man), desc="extract text"):
        out = paths.pages / f"{r['doc_id']}.json"
        if out.exists() and not force:
            continue
        pdf = resolve_local_path(r["local_path"])
        try:
            res = extract_pdf(pdf, cfg)
        except Exception as exc:  # noqa: BLE001 - one corrupt PDF must not stop the batch
            log.error("could not read %s: %s", pdf, exc)
            summary.append({"doc_id": r["doc_id"], "error": str(exc)})
            continue
        res = {"doc_id": r["doc_id"], "firm_id": r["firm_id"], "fy": int(float(r["fy"])),
               "source_pdf": r["local_path"], **res}
        out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        summary.append({k: v for k, v in res.items() if k != "pages"})
    df = pd.DataFrame(summary)
    if len(df):
        log.info("extracted %d PDFs (OCR pages: %s, pages still needing OCR: %s)", len(df),
                 int(df.get("n_ocr_pages", pd.Series([0])).sum()),
                 int(df.get("n_needs_ocr_pages", pd.Series([0])).sum()))
    else:
        log.info("nothing to extract (use --force to redo existing files)")
    return df
