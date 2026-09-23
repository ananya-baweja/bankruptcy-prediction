# Progress log

What was done each working session, newest first. Keep entries short: what, result, next.

## 2026-09-23 — Phase 3 built (financial features, `bpp financials`)

**Done**
- `bpp financials` and `bpp financials-score`, plus `src/bpp/features/{numbers,statements,ratios,financials}.py`
  and `docs/06_phase3_financials.md`.
- Locates the **standalone** balance sheet, P&L and cash flow statement in Phase 2's page text (no
  re-OCR), reads the unit and period columns from the header, walks the lines tracking Schedule III
  sub-headings, and maps line items to 20 standard fields. Ruled tables go through pdfplumber or
  camelot when available; the text walk always runs, so scanned statements still work.
- Both the current-year and prior-year columns are read. Figures are stored **as first published**;
  the next year's comparative cross-checks them (`restated`) or fills a missing year.
- Validation: both balance-sheet identities, subtotal containment, unit-scale checks against the
  cohort's own assets and against last year, and a confidence score per figure.
- The 12 stream-C ratios of Table 7, including Altman EM Z'' with its zones. Ratios with a negative
  or zero denominator return empty with a reason, and the condition is kept as its own feature.
- Missing-data flag per company-year (rows kept, never dropped), the gap-filling order, the
  pair-exclusion rule, and the unrecoverable count **by class** printed every run.
- Spot-check sheet: a random 10% of company-years with page numbers, scored by `bpp financials-score`.
- `synthetic.py` gained `financial_statements=False` (off by default, so Phases 1–2 are unchanged),
  which writes proper Schedule III statements with a prior-year column, a stated unit, bracketed
  negatives, a note column and a consolidated set to discriminate against.

**Checked on synthetic data (not real results)**
- 173 new tests pass, covering: units (crore / lakh / million / thousand / rupee all normalising to
  the same ₹ crore figure), Indian and Western digit grouping, bracketed negatives, Nil/NA/em-dash,
  OCR damage, refusal on unrecoverable cells, standalone-vs-consolidated preference, `Borrowings`
  disambiguated by sub-heading, both year columns, as-first-published, restatement detection,
  gap-filling from a comparative and from the XBRL CSV, unrecoverable years, pair exclusion, the
  spot-check sheet, and every ratio hand-computed.
- The extracted balance sheet balances exactly on the synthetic reports, and the table path
  (pdfplumber) and the text path agree figure for figure.
- An adversarial code review was run over the module and found eleven ways a figure could come out
  wrong but plausible, all of which the first round of tests had missed because the fixtures were
  too clean. All are fixed and each has a regression test in `tests/test_financials_robustness.py`.
  The ones worth knowing about: a blank money column let a note reference become this year's figure;
  a unit word in prose ("turnover crossed Rs. 500 crore") overrode the real "(Rs. in lakhs)" caption
  and made every figure 100x too large; the auditor's report quotes the balance sheet's title and
  was winning the statement's location; a merged table cell "500 400" silently became 500400; and
  the Altman safe/grey/distress cutoffs were being read against the +3.25 rating-equivalent scale
  rather than the discriminant, which called a failing firm "safe".

**Not verified yet (needs real reports)**
- **Step 9 of the brief — the run on 5 real reports (2 insolvent, 3 healthy) has not happened**,
  because Phase 2 has not yet been run on real PDFs. Nothing here has met a real annual report. The
  line-item patterns and the statement-page locator are the parts most likely to need work; expect
  to add patterns to `FIELD_PATTERNS` after the first real batch.
- Whether `retained_earnings` appears on the face of the balance sheet often enough, or whether
  `other_equity` will stand in for nearly every firm.

**Next**
1. Run Phase 2 on real reports, then `bpp financials` on 5 of them (2 insolvent, 3 healthy) and
   record what failed here before scaling up.
2. Fill the spot-check sheet and record the accuracy per field.
3. Promoter pledge % from the exchange shareholding filings — the column is reserved and empty.

## 2026-09-21 — Progress report for the professor

**Done**
- Wrote `docs/Progress_Report_Bankruptcy_Prediction.docx` (13 pages): the ten reference papers, methodology step by step, pipeline and architecture, evaluation plan, progress so far, timeline and risks.
- Drew five diagrams (`docs/figures/`): project overview, cohort building, label/leakage timeline, document processing, model architecture. Reusable in slides.
- Verified every citation against the publisher or arXiv page before including it.

**The ten papers**
Altman (1968) · Loughran & McDonald (2011) · Mai, Tian, Lee & Ma (2019) · Cohen, Malloy & Nguyen (2020) · Huang, Wang & Yang (2023, FinBERT) · Arno et al. (2024, ECL) · Mancisidor & Aas (2022/24) · Gupta (2022, IBC ratios) · Gupta & Banerjee (2023, IBC text) · Ganin & Lempitsky (2015, gradient reversal).

**Revised the report the same day (risk of over-promising)**
- Reframed the project from "text beats ratios" to "how much signal language carries, which section carries it, and how early" — stated on page 1 and in the framing row.
- Added Section 7.4 (four hypotheses with what we report if each fails), 7.5 (deliverables that stand regardless of the result) and 7.6 (steps that give the language stream a fair chance).
- Added two experiments: MD&A-only vs auditor-only vs both, and the subgroup of firms whose ratios look healthy.
- Added a pilot signal check in weeks 3–4 (tone, hedging, going-concern and CARO flags on the first ~30 pairs) so we learn early whether language separates the groups.
- Added two risks: language adding little beyond ratios, and too few insolvent firms to prove a small gain.

**Next**
- Same as the 16 Sep entry: environment setup, IBBI scrape, Prowess access, name-match review.
- Run the pilot signal check as soon as ~30 pairs of reports exist.

## 2026-09-16 — Phases 0–2 code built (with Claude in Cowork)

**Done**
- Repo structure, `pyproject.toml`, `requirements.txt`, `environment.yml`, `.gitignore`, `configs/config.yaml`.
- `bpp` command with 16 sub-commands (`bpp --help`), including `status` and `demo`.
- Phase 1 code: IBBI announcement scraper (cached, polite), CIRP filter, BSE/NSE universe (active + suspended + delisted), fuzzy name matching with a human-review file, healthy-peer matching, firm-year sample frame.
- Phase 2 code: NSE/BSE report listing + download, manual registration, PDF text with Tesseract OCR fallback, section extraction (MD&A, Board's report, standalone auditor's report + qualified opinion / going concern / emphasis of matter, CARO annexure), labels with leakage rules and horizons, QA sheet + scoring.
- Synthetic data generator and 35 automated tests (all passing).
- Docs: plan, setup, Phase 1 and Phase 2 guides, data dictionary, decisions log.

**Checked on synthetic data (not real results)**
- Demo: 4 pairs, 32 firm-years, 31 PDFs (1 deliberately missing), 1 scanned page OCR'd.
- Sections found in 31/31 synthetic reports; 22 firm-years included (11 distressed / 11 healthy) after leakage rules.
- A near-duplicate name ("Tarang" vs "Tarangi") correctly went to human review after raising the auto-accept threshold to 98.

**Not verified yet (needs the real sites, from a laptop in India)**
- IBBI page markup and pagination on the live site.
- BSE annual-report endpoint (`reports.bse_annual_report_endpoint`) — returned 403 from a cloud IP.
- NSE `annual_reports` response fields (parser is defensive; test on one symbol).

**Next**
1. Everyone: environment setup, `pytest`, `bpp demo` (docs/01_setup.md).
2. Data lead: `bpp ibbi-scrape --max-pages 5`, check `public_announcements.csv`, then the full scrape.
3. Confirm CMIE Prowess access with the college library (decides how `firm_financials.csv` is filled).
4. Split name-match review among the team.
