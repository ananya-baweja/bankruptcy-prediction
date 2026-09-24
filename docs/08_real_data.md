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
- **The CARO annexure was missed in about a third of reports** by the Phase 2 section extractor even
  though they contain it (annexure letters vary, and the IFC annexure was taken for CARO). Fixed on
  24 Sep with the team's approval: the annexure's own text decides what it is (185 of 210 found).
- **Scanned reports carry their own OCR text layer**, with figures split by stray spaces
  (`1, 255. 53`, `70 .68`); the reader closes these up before it assigns columns.
- **A report can be damaged on the exchange itself.** BSE526550's FY2022 report is 13.9 MB of which
  everything after the first 0.6 MB is zero bytes, on every download; it is treated as missing.
- **Collected JSON is big.** The peers' size-year XBRL filings came to 540 MB of text (the 2025+
  "integrated filing" format is ~200 KB per company-year). Agent 1.6 gzips before handing over (13x smaller).
- **The unit caption is wherever the PDF put it.** On the full cohort 66 company-years were first read a
  power of ten off the XBRL filing: the caption `(₹ in Lakhs)` came after the signature block in the
  text layer, or inside the heading line (`Balance sheet as at 31 March 2019 (₹ in Lakhs)`), or only on
  the profit and loss, or only in the accounting-policy note, or in a font whose text comes out as
  `(`LQ/DNKV` (every letter shifted 29 code points). Some reports print in **hundreds**; one printed a
  boilerplate "(All amounts in lacs)" over rupee figures, another a bare "Rs." header over lakh
  figures. The reader now looks in all these places (`docs/06_phase3_financials.md`); 63 of the 66
  read right, and PDF-vs-XBRL agreement within 1% went from 80.0% to 91.6%.
- **Two XBRL filings in a row can both be wrong.** Rane (Madras) filed FY2022 and FY2023 100x too
  small; each looked consistent with the other. The unit check now also compares with the firm's
  filings outside the sample years and with the report's own prior-year column.
- **Company names can tie after normalisation.** Asian Hotels (East), (North) and (West) all
  normalise to "ASIAN HOTELS"; IBBI's insolvent Asian Hotels (West) was auto-matched to Asian Hotels
  (East). The CIN printed in the report caught it (the only mismatch among 136 insolvent firms).
- **A peer can be sized on a slipped filing.** P0100's peer was chosen on FY2023 total assets of
  ₹11.78 crore from an XBRL filing 100x too small; the company is ~₹1,178 crore.

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

## Full cohort results (136 pairs, 25 Sep 2026)

| | |
| --- | --- |
| Pairs | 136 (the pilot's 27 + 109 new); 39 insolvent firms found no peer within ±30% |
| Reports read | 1,143 (147,662 pages, 7,436 OCR'd), no failures; 1,048 belong to the cohort |
| Insolvent firms confirmed by the CIN in their own report | 135 of 136 (127 exact, 6 same registration number, 2 AP to TG); P0093 is a wrong match |
| Leakage review | 62 reports read, 38 excluded (discuss a petition against the company itself) |
| Company-years | 1,088: 727 from XBRL, 320 from the report, 26 from next year's report, 15 missing |
| Report reader vs XBRL, within 1% | 91.6% of 10,176 figures (93.0% within 5%); 5 company-years detected as read at the wrong scale, 7 XBRL filings |
| Unrecoverable company-years (a core figure missing) | distressed 67 of 544, healthy 49 of 544 - 95 of the 116 are FY2016-2018 |
| Modelling rows | 729 included (367 distressed / 362 healthy); 604 keep usable financials after the pair rule |

`python scripts/pilot_report.py --data-dir D --name full` writes the details to `interim/qa/full/`.

### Full cohort (from 24 Sep 2026)

One step per download round; the pilot's 27 pairs are kept as they are:

```bash
python scripts/exchange_pipeline.py --data-dir D full-select --jobs-out J              # 011: insolvent firms' size-year XBRL
python scripts/exchange_pipeline.py --data-dir D full-candidates --jobs-out J          # 012: possible peers' filing lists
python scripts/exchange_pipeline.py --data-dir D full-distressed-reports --jobs-out J  # 019: insolvent firms' reports
python scripts/exchange_pipeline.py --data-dir D full-sizing --jobs-out J              # 013: peers' size-year XBRL
python scripts/exchange_pipeline.py --data-dir D full-shortlist --jobs-out J           # 014: report lists within +/-30%
python scripts/exchange_pipeline.py --data-dir D full-finalize --jobs-out J            # pairs, cohort.csv; 015: peers' reports
python scripts/pilot_report.py --data-dir D --name full                                # after text, xbrl, phases, cincheck
```

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
