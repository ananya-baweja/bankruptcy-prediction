import pandas as pd

from bpp.scrape.ibbi import combine_cached_pages, filter_cirp_announcements, parse_announcement_page
from bpp.synthetic import ibbi_html

ROWS = [
    {"type": "Corporate Insolvency Resolution Process", "date": "14-08-2019", "debtor": "ZENTHRA INFRAPROJECTS LTD."},
    {"type": "Corporate Insolvency Resolution Process", "date": "02-09-2019", "debtor": "Zenthra Infraprojects Limited"},
    {"type": "Corporate Insolvency Resolution Process", "date": "11-09-2019", "debtor": "Brisk Logistics Pvt. Ltd."},
    {"type": "Liquidation Process", "date": "01-02-2022", "debtor": "Mirovan Shipyards Ltd"},
]


def test_parse_page_maps_columns_by_header():
    rows = parse_announcement_page(ibbi_html(ROWS))
    assert len(rows) == 4
    assert rows[0]["corporate_debtor"] == "ZENTHRA INFRAPROJECTS LTD."
    assert rows[0]["announcement_date"] == "14-08-2019"
    assert rows[0]["pa_pdf_url"].startswith("https://ibbi.gov.in/uploads/announcement/")


def test_parse_page_without_table_returns_empty():
    assert parse_announcement_page("<html><body>No records</body></html>") == []


def test_filter_keeps_earliest_cirp_non_private(cfg, paths):
    (paths.ibbi_pages / "page_00001.html").write_text(ibbi_html(ROWS), encoding="utf-8")
    df = filter_cirp_announcements(cfg, paths, combine_cached_pages(paths))
    assert len(df) == 1                                    # private + liquidation dropped, duplicates merged
    assert df.iloc[0]["cirp_announcement_date"] == pd.Timestamp("2019-08-14")
    assert df.iloc[0]["n_announcements"] == 2


def test_parse_page_with_td_header_row():
    html = ("<table><tr><td>Type of PA</td><td>Date of Announcement</td><td>Last date of Submission</td>"
            "<td>Name of Corporate Debtor</td><td>Name of Applicant</td><td>Name of Insolvency Professional</td>"
            "<td>Public Announcement</td><td>Remarks</td></tr>"
            "<tr><td>Corporate Insolvency Resolution Process</td><td>01-01-2020</td><td>15-01-2020</td>"
            "<td>ABC Limited</td><td>Bank</td><td>IP</td><td><a href='https://ibbi.gov.in//uploads/x.pdf'>PA</a></td>"
            "<td></td></tr></table>")
    rows = parse_announcement_page(html)
    assert len(rows) == 1 and rows[0]["corporate_debtor"] == "ABC Limited"
