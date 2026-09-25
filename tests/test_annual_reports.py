import io
import zipfile

import pandas as pd

from bpp.scrape.annual_reports import (parse_bse_listing, parse_nse_listing, register_manual_reports,
                                       save_pdf_bytes)

PDF = b"%PDF-1.4\n%fake\n"


def test_parse_nse_listing():
    # field names as in the nse package's sample response (src/samples/annual_reports.json)
    payload = {"data": [{"companyName": "HDFC Bank Limited", "fromYr": "2023", "toYr": "2024",
                         "submission_type": "-", "broadcast_dttm": "18-JUL-2024 18:34:53",
                         "disseminationDateTime": "18-JUL-2024 00:00:00", "timeTaken": "-18:-34:-53",
                         "fileName": "https://nsearchives.nseindia.com/annual_reports/AR_HDFCBANK_2024.pdf"}]}
    out = parse_nse_listing(payload)
    assert out == [{"fy": 2024, "url": "https://nsearchives.nseindia.com/annual_reports/AR_HDFCBANK_2024.pdf",
                    "pub_date": "2024-07-18"}]


def test_parse_bse_listing_relative_file():
    payload = [{"Year": "2018-2019", "file_name": "1234abcd.pdf", "Dt_tm": "2019-08-30T10:00:00"}]
    out = parse_bse_listing(payload, "500325", "https://www.bseindia.com/bseplus/AnnualReport")
    assert out[0]["fy"] == 2019
    assert out[0]["url"] == "https://www.bseindia.com/bseplus/AnnualReport/500325/1234abcd.pdf"
    assert out[0]["pub_date"] == "2019-08-30"


def test_save_pdf_bytes_handles_zip_and_rejects_html(tmp_path):
    assert save_pdf_bytes(PDF, tmp_path / "a.pdf")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("report.pdf", PDF)
    assert save_pdf_bytes(buf.getvalue(), tmp_path / "b.pdf")
    assert (tmp_path / "b.pdf").read_bytes() == PDF
    assert not save_pdf_bytes(b"<html>blocked</html>", tmp_path / "c.pdf")


def test_register_manual_reports(paths):
    d = paths.raw_reports / "BSE1"
    d.mkdir(parents=True)
    (d / "FY2019.pdf").write_bytes(PDF)
    pd.DataFrame([{"firm_id": "BSE1", "fy": "2018-19", "local_path": "", "pub_date": "30-08-2019",
                   "source_url": "https://company.example/ar.pdf", "note": ""}]).to_csv(paths.manual_reports, index=False)
    man = register_manual_reports(paths).set_index("doc_id")
    assert man.loc["BSE1_FY2019", "pub_date"] == "2019-08-30"
    assert man.loc["BSE1_FY2019", "status"] == "registered"


def test_ocr_scale_caps_giant_pages():
    from bpp.extract.pdf_text_pdfium import MAX_OCR_SIDE_PX, ocr_scale
    assert ocr_scale(595, 842, 300) == 300 / 72                    # A4 renders at 300 dpi
    s = ocr_scale(3750, 5000, 300)                                  # a scan declared at its pixel size
    assert abs(5000 * s - MAX_OCR_SIDE_PX) < 1e-6 and s < 300 / 72
