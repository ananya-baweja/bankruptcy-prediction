"""The table-extraction path, on a real PDF.

``parse_statement_lines`` (the text walk) is what must always work, because a
scanned statement has no table structure at all. pdfplumber and camelot are an
accuracy improvement on top of it, so these tests check that the two paths agree
rather than that either is used.
"""

import pytest

from bpp.extract.pdf_text import extract_pdf
from bpp.features.statements import (best_hits, extract_table_rows, locate_statements,
                                     map_statement, parse_statement_lines, rows_to_line_items)
from bpp.synthetic import write_report_pdf, year_figures

pdfplumber = pytest.importorskip("pdfplumber", reason="pdfplumber is optional")


@pytest.fixture
def statements_pdf(tmp_path):
    path = tmp_path / "FY2019.pdf"
    write_report_pdf(path, "Zenthra Infraprojects Limited", 2019, distressed=True,
                     mention_cirp=False, scanned_mdna_page=False, seed=1,
                     financial_statements=True)
    return path


def test_the_text_walk_reads_the_statements(cfg, statements_pdf):
    cfg["extraction"]["ocr"]["enabled"] = False
    pages = extract_pdf(statements_pdf, cfg)["pages"]
    locations = locate_statements(pages, cfg, report_fy=2019)
    assert "balance_sheet" in locations
    assert locations["balance_sheet"].scope == "standalone"

    hits = []
    for location in locations.values():
        hits += map_statement(parse_statement_lines(pages, location), location)
    values = {(h.field, h.fy): h.value for h in best_hits(hits).values()}
    expected = year_figures(2019, True, seed=1)
    assert values[("total_assets", 2019)] == pytest.approx(expected["total_assets"])
    assert values[("total_equity", 2019)] == pytest.approx(expected["total_equity"])


def test_table_extraction_agrees_with_the_text_walk(cfg, statements_pdf):
    """Two independent readings of the same page must give the same number."""
    cfg["extraction"]["ocr"]["enabled"] = False
    pages = extract_pdf(statements_pdf, cfg)["pages"]
    location = locate_statements(pages, cfg, report_fy=2019)["balance_sheet"]

    rows = extract_table_rows(statements_pdf, range(location.start_page, location.end_page + 1),
                              flavours=("pdfplumber",))
    if not rows:
        pytest.skip("this PDF has no ruled tables; the text walk covers it")

    from_tables = {(h.field, h.fy): h.value for h in
                   best_hits(map_statement(rows_to_line_items(rows, location), location)).values()}
    from_text = {(h.field, h.fy): h.value for h in
                 best_hits(map_statement(parse_statement_lines(pages, location), location)).values()}
    shared = set(from_tables) & set(from_text)
    assert shared, "the two paths found no field in common"
    for key in shared:
        assert from_tables[key] == pytest.approx(from_text[key]), f"paths disagree on {key}"


def test_table_extraction_never_raises_on_a_missing_file(tmp_path):
    """Every table dependency is optional; a failure must degrade, not stop the batch."""
    assert extract_table_rows(tmp_path / "does_not_exist.pdf", [1]) == []


def test_table_extraction_handles_an_empty_page_list(statements_pdf):
    assert extract_table_rows(statements_pdf, []) == []
