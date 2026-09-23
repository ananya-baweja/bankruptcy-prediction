"""Page text with pypdfium2 instead of PyMuPDF - same output as ``pdf_text.py``.

``pdf_text.py`` (Phase 2) uses PyMuPDF. Where PyMuPDF cannot be installed (the
cloud sessions that process the pilot) this module produces the identical
``interim/pages/<doc_id>.json`` from pypdfium2, so every later phase reads the
same file whichever backend made it. Phase 2's own code is not changed.

Differences worth knowing:

* pdfium orders text by the content stream, PyMuPDF by blocks; on multi-column
  pages the line order can differ slightly. Section detection works on headings
  and is not sensitive to it; the ``backend`` field records which one was used.
* OCR renders the page with pdfium at ``dpi`` and runs Tesseract, exactly as the
  PyMuPDF path does.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

try:  # share the exact clean-up rules with the PyMuPDF path when it is importable
    from bpp.extract.pdf_text import normalize_page_text, tesseract_available
except ImportError:  # PyMuPDF missing: same rules, copied
    import re
    import unicodedata

    _LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}

    def normalize_page_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        for k, v in _LIGATURES.items():
            text = text.replace(k, v)
        text = (text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"')
                    .replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-")
                    .replace("\u00a0", " ").replace("\u00ad", ""))
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    _tesseract_ok: bool | None = None

    def tesseract_available(cmd: str | None = None) -> bool:
        global _tesseract_ok
        if _tesseract_ok is None:
            try:
                import pytesseract
                if cmd:
                    pytesseract.pytesseract.tesseract_cmd = cmd
                pytesseract.get_tesseract_version()
                _tesseract_ok = True
            except Exception:  # noqa: BLE001
                _tesseract_ok = False
        return _tesseract_ok


def _ocr(page: Any, dpi: int, lang: str) -> str:
    import pytesseract

    img = page.render(scale=dpi / 72).to_pil()
    return pytesseract.image_to_string(img, lang=lang)


def extract_pdf_pdfium(pdf_path: Path, cfg: dict[str, Any], ocr_pages: set[int] | None = None) -> dict[str, Any]:
    """Page texts of one PDF. ``ocr_pages`` (1-based) limits OCR to those pages."""
    import pypdfium2 as pdfium

    ecfg = cfg["extraction"]
    ocfg = ecfg["ocr"]
    min_chars = ecfg["min_chars_per_page"]
    can_ocr = ocfg["enabled"] and tesseract_available(ocfg.get("tesseract_cmd"))
    pages = []
    t0 = time.time()
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                tp = page.get_textpage()
                text = tp.get_text_range() or ""
                tp.close()
            except Exception as exc:  # noqa: BLE001
                log.warning("text layer unreadable on %s page %d: %s", pdf_path.name, i + 1, exc)
                text = ""
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            method = "text"
            if len(text.strip()) < min_chars:
                wanted = ocr_pages is None or (i + 1) in ocr_pages
                if can_ocr and wanted:
                    try:
                        text = _ocr(page, ocfg["dpi"], ocfg["lang"])
                        method = "ocr"
                    except Exception as exc:  # noqa: BLE001
                        log.warning("OCR failed on %s page %d: %s", pdf_path.name, i + 1, exc)
                        method = "needs_ocr"
                else:
                    has_image = any(obj.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE for obj in page.get_objects())
                    method = "needs_ocr" if has_image else "empty"
            page.close()
            pages.append({"page": i + 1, "method": method, "text": normalize_page_text(text)})
    finally:
        pdf.close()
    counts: dict[str, int] = {}
    for p in pages:
        counts[p["method"]] = counts.get(p["method"], 0) + 1
    return {
        "n_pages": len(pages),
        "n_text_pages": counts.get("text", 0),
        "n_ocr_pages": counts.get("ocr", 0),
        "n_needs_ocr_pages": counts.get("needs_ocr", 0),
        "n_empty_pages": counts.get("empty", 0),
        "n_chars": sum(len(p["text"]) for p in pages),
        "backend": "pypdfium2",
        "seconds": round(time.time() - t0, 1),
        "pages": pages,
    }


def write_pages_json(out_path: Path, doc_id: str, firm_id: str, fy: int, source_pdf: str,
                     res: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"doc_id": doc_id, "firm_id": firm_id, "fy": int(fy), "source_pdf": source_pdf, **res}
    out_path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
