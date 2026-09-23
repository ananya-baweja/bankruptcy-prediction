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
- **One annual report was 229 MB** (a scanned FY2018 report) and could not be moved to the processing
  session in the time allowed per transfer.

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
```

`J` is the folder of job files copied into the laptop's `bpp-data\jobs\`.
