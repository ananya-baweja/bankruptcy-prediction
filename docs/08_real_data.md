# Real data — how it is collected and turned into the project's tables

CMIE Prowess is not available. This is the pipeline that replaces it with public sources, built and run
on 2026-09-23 for the 27-pair pilot.

```
IBBI export (CIN)  ─┐
BSE/NSE lists      ─┼─► exchange_pipeline.py parse ─► ibbi_listed_matches.csv
                    │
BSE ComHeadernew   ─┐                                   (industry)
BSE Result_Arch_ng ─┼─► exchange_pipeline.py select ───► pilot_distressed.csv
BSE AnnualReport   ─┘                                   (reports + XBRL size year)
                        exchange_pipeline.py candidates ► pilot_peer_candidates.csv
XBRL (size year)   ───► exchange_pipeline.py peers ─────► candidate XBRL job
                        exchange_pipeline.py finalize ──► cohort.csv, sample_frame.csv, report job
PDFs + XBRL        ───► process_reports.py text ────────► interim/pages/*.json   (pypdfium2 + Tesseract)
                        exchange_pipeline.py xbrl ──────► interim/xbrl_financials_exchange.csv
                        process_reports.py phases ──────► sections, labels, financials, ratios, language
                        process_reports.py cincheck ────► interim/report_cin_check.csv
```

## Why a download agent on a laptop

IBBI, BSE and NSE refuse requests from cloud addresses. A team laptop in India can reach all three.
`bpp_fetch.py` (in the laptop's `bpp-data` folder, started with `START_DOWNLOADER.bat`) watches a
`jobs\` folder and downloads what each job file lists. It uses only the Python standard library, only
contacts the hosts in its allow-list (IBBI, BSE, NSE, and the Notre Dame / Google Drive link of the
Loughran-McDonald dictionary), only writes inside its own folder, and never runs anything it downloads.
Every request is logged in `logs\<job>.csv` with its HTTP status, size and SHA-256. It restarts itself
when a new version of the script is saved into the folder.

Small API responses are collected as JSON lines (`raw/**/*.jsonl`, one line per request, the raw body
kept); PDFs are saved as `raw/annual_reports/<firm_id>/FY<year>.pdf`, the layout Phase 2 already uses.

## Sources and what each gives

| Source | Endpoint | Gives |
| --- | --- | --- |
| IBBI | `public-announcement?...&export_excel=export_excel` | every CIRP public announcement **with the debtor's CIN** |
| BSE | `ListofScripData` (Active / Suspended / Delisted) | the listed universe with ISIN |
| NSE | `EQUITY_L.csv`, `namechange.csv`, `symbolchange.csv` | NSE listings and renames |
| BSE | `ComHeadernew` | Sector › Industry › Group › Sub-group per company |
| BSE | `GetINDUSTRYWATCHLIST_ng` | the 186 sub-groups with hierarchical codes |
| BSE | `Result_Arch_ng` | every results filing, with the standalone/consolidated XBRL link and filing time |
| BSE | `AnnualReport_New` | annual-report PDFs with filing time (recorded only from ~FY2023) |
| BSE/NSE | the XBRL instance | the audited standalone annual results, tagged (`in-bse-fin`) |

**XBRL coverage:** BSE from FY2019, NSE from FY2018. Earlier years come from the PDFs.

## Things that are not what they seem (found on the real data)

- **A CIRP announcement is not always insolvency.** Container Corporation of India (2019), Titagarh
  Wagons (2022) and Zee Entertainment (2023) were admitted on operational creditors' claims and stayed
  within days. The pilot excludes government companies and BSE group A companies for this reason —
  **a provisional rule the team must confirm**.
- **Companies file XBRL in the wrong unit.** Total assets of ₹7.5 crore were filed as ₹7,53,311 crore.
  `xbrl_unit_checks` compares each XBRL filing with the report and drops a filing that is off by a
  power of ten when the report states its own unit.
- **Small companies print statements in rupees with no caption.** The unit is now read from a
  "Rupees / Amount (Rs.)" column header, or inferred from large whole numbers.
- **The directors' report reprints a "Profit and Loss" table.** The P&L is now the one after the
  balance sheet.
- **BSE's industry classification is empty for many delisted companies**, and filing dates are
  missing for almost every report before FY2023 (the FY end + 183 days rule applies).
- **One annual report was 229 MB** (a scanned FY2018 report), too large to move in one transfer. The
  agent (1.5) can now cut a file into numbered parts (`split` items); the processing session joins
  them and checks the SHA-256.
- **A download can stop early and still say HTTP 200.** One report arrived as 98 KB of 3.9 MB. The
  agent (1.4) keeps a file only when it matches `Content-Length` and a PDF ends with `%%EOF`.
- **An XBRL filing can repeat last year's profit and loss** to the rupee (with the current year typed
  into the quarter column). Such filings are not used for flow figures.
- **XBRL and the report disagree on some balance-sheet lines** in both directions. The value that
  makes the balance sheet add up is kept.
- **Older reports (FY2016-2017, pre-Ind AS) print no section totals**, only one `Total` per side.
  Section totals are built by adding the rows and kept only when both sides balance to rounding.
- **Contents and corporate-information pages look like statements** to a figure counter (PIN codes,
  phone numbers, page references). Headings must end in a date, and contents entries are ignored.
- **The CARO annexure is missed in about a third of reports** by the Phase 2 section extractor even
  though they contain it. Fixing it is a Phase 2 change and waits for the team's approval.

## Pilot results (27 pairs, 24 Sep 2026)

| | |
| --- | --- |
| Reports read | 210 (23,636 pages, 1,624 OCR'd), no failures |
| Insolvent firms confirmed by the CIN in their own report | 27 of 27 |
| Company-years | 216: 140 from XBRL, 67 from the report, 7 from next year's report, 2 missing |
| Report reader vs XBRL, within 1% | 87.9% of 1,934 figures; 91.2% leaving out 6 reports detected as read at the wrong scale |
| Unrecoverable company-years (a core figure missing) | distressed 10 of 108, healthy 9 of 108 - almost all FY2016-2018 |
| Modelling rows | 160 included; 135 keep usable financials after the pair rule |

`python scripts/pilot_report.py --data-dir D` writes the details to `interim/qa/pilot/`.

## Commands (processing session)

```bash
python scripts/exchange_pipeline.py --data-dir D parse
python scripts/exchange_pipeline.py --data-dir D select --n 40 --years 2019-2025 --jobs-out J
python scripts/exchange_pipeline.py --data-dir D candidates --jobs-out J
python scripts/exchange_pipeline.py --data-dir D peers --jobs-out J
python scripts/exchange_pipeline.py --data-dir D finalize --n-pairs 30 --jobs-out J
python scripts/exchange_pipeline.py --data-dir D xbrl
python scripts/exchange_pipeline.py --data-dir D manifest
python scripts/process_reports.py --data-dir D text --workers 2
python scripts/process_reports.py --data-dir D phases
python scripts/process_reports.py --data-dir D cincheck
python scripts/pilot_report.py --data-dir D
```

`J` is the folder of job files copied into the laptop's `bpp-data\jobs\`.
