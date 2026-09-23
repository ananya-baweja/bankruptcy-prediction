import pytest

from bpp.extract.pdf_text import extract_pdf
from bpp.extract.sections import _classify_heading, segment_document
from bpp.synthetic import write_report_pdf


@pytest.mark.parametrize("line,expected", [
    ("MANAGEMENT DISCUSSION AND ANALYSIS REPORT", "mdna"),
    ("Management's Discussion & Analysis", "mdna"),
    ("Annexure II - Management Discussion and Analysis", "mdna"),
    ("DIRECTORS' REPORT", "directors_report"),
    ("Board's Report", "directors_report"),
    ("Report of the Board of Directors", "directors_report"),
    ("INDEPENDENT AUDITORS' REPORT", "auditor_report"),
    ("Annexure 'A' to the Independent Auditor's Report", "caro_annexure"),
    ("ANNEXURE - B TO THE INDEPENDENT AUDITORS' REPORT", "ifc_annexure"),
    ("Report on Corporate Governance", "corporate_governance"),
    ("Balance Sheet as at 31st March, 2023", "financial_statements"),
])
def test_heading_variants(line, expected):
    found = _classify_heading(line)
    assert found is not None and found[0] == expected


def test_sentence_is_not_a_heading():
    found = _classify_heading("Board's report is prepared in compliance with the Act")
    from bpp.extract.sections import _is_heading_rest
    assert found is None or not _is_heading_rest(found[1])


@pytest.fixture
def distressed_pdf(tmp_path):
    p = tmp_path / "FY2019.pdf"
    write_report_pdf(p, "Demo Distressed Limited", 2019, distressed=True, mention_cirp=True,
                     scanned_mdna_page=False, seed=1)
    return p


def test_segment_synthetic_report(cfg, distressed_pdf):
    cfg["extraction"]["ocr"]["enabled"] = False
    pages = extract_pdf(distressed_pdf, cfg)["pages"]
    seg = segment_document(pages, cfg)
    s = seg["sections"]
    assert s["mdna"]["heading"].startswith("MANAGEMENT DISCUSSION")        # not the cross-reference sub-heading
    assert s["mdna"]["start_page"] > 2                                     # not the contents page
    assert "Board's Report" not in s["mdna"]["text"][:200]
    assert s["directors_report"]["heading"] == "BOARD'S REPORT"
    assert "Corporate Governance" in s["directors_report"]["text"]         # sub-heading did not cut it
    assert s["auditor_report"]["scope"] == "standalone"
    assert "Consolidated" not in s["auditor_report"]["text"][:400]
    assert s["caro_annexure"]["text"].startswith("ANNEXURE 'A'")
    assert "defaulted in repayment" in s["caro_annexure"]["text"]
    assert s["going_concern"] and "material uncertainty" in s["going_concern"]["text"].lower()
    assert s["basis_for_modified_opinion"]
    assert seg["audit_opinion"] == "qualified"
    assert seg["leakage"]["cirp_specific_mentions"] >= 1


def test_healthy_report_unmodified(cfg, tmp_path):
    cfg["extraction"]["ocr"]["enabled"] = False
    p = tmp_path / "h.pdf"
    write_report_pdf(p, "Demo Healthy Limited", 2019, distressed=False, mention_cirp=False,
                     scanned_mdna_page=False, seed=2)
    seg = segment_document(extract_pdf(p, cfg)["pages"], cfg)
    assert seg["audit_opinion"] == "unmodified"
    assert seg["sections"]["going_concern"] is None
    assert seg["leakage"]["cirp_specific_mentions"] == 0


def test_scanned_page_marked_or_ocrd(cfg, tmp_path):
    from bpp.extract.pdf_text import tesseract_available
    p = tmp_path / "s.pdf"
    write_report_pdf(p, "Demo Scan Limited", 2019, distressed=True, mention_cirp=False,
                     scanned_mdna_page=True, seed=3)
    res = extract_pdf(p, cfg)
    if tesseract_available():
        assert res["n_ocr_pages"] == 1
        ocr_page = next(pg for pg in res["pages"] if pg["method"] == "ocr")
        assert "RISKS" in ocr_page["text"].upper()
    else:
        assert res["n_needs_ocr_pages"] == 1
