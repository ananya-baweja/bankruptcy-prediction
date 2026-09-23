# Progress log

What was done each working session, newest first. Keep entries short: what, result, next.

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
