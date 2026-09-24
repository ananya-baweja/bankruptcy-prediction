# Progress log

What was done each working session, newest first. Keep entries short: what, result, next.

## 2026-09-23/24 — Real data: download agent, 27-pair pilot, Phases 2–4 on 210 real reports

**Done**
- **Data without CMIE.** A small download agent (`bpp_fetch.py`, standard library only) runs on a team
  laptop in India and fetches what job files list: IBBI's CIRP export (with CINs), the BSE/NSE listed
  universe, BSE industry codes, annual-report PDFs and the standalone annual XBRL results. Every request
  is logged with its SHA-256. Agent 1.4 keeps a file only when complete; 1.5 can split a file too large
  to hand over (one report was 229 MB). See `docs/08_real_data.md`.
- **Cohort from exchange data** (`scripts/exchange_pipeline.py`): 455 insolvent companies matched to a
  listing by CIN; 175 eligible under the pilot rules; a 27-pair pilot sampled across admission years
  2019–2025, peers from the same BSE sub-group within ±30% total assets. **All 27 insolvent firms are
  confirmed by the CIN printed in their own reports** (26 exact, 1 via the registration number).
- **All 210 pilot reports read** (pypdfium2 + Tesseract): 23,636 pages, 1,624 of them OCR'd, no failures.
- **Phase 3 on real reports — what failed, and the fixes** (each has a regression test in
  `tests/test_financials_real_reports.py`, 31 tests):
  - statements in rupees with no caption read as crore (10^7 too large); a directors'-report P&L summary
    taken for the P&L; cash-flow movements read as balances; bracketed expenses negative; pre-exceptional
    PBT; a note number read as a figure; figures printed on the line after the label; unlabelled or bare
    `Total` section totals; pre-Ind AS sheets with no section totals (now summed, kept only if the sheet
    balances to rounding); "Other current liabilities" fuzzy-matched to the total; opening cash taken as
    closing cash; the auditor's report and contents pages taken for the balance sheet; OCR slips (lost
    `(`, spaced commas).
  - XBRL is not always right either: 2 filings in the wrong unit, 2 that repeat last year's P&L, 1 with a
    mistyped balance-sheet tag. Handled by evidence-based unit checks, a stale-copy rule and a
    balance-sheet-identity arbitration; 6 reports read at the wrong scale are detected the same way.
- **Result:** the PDF reader agrees with the XBRL filing within 1% on **87.9%** of 1,934 figures where both
  exist (91.2% excluding the reports detected as read at the wrong scale). Unrecoverable
  company-years (a core field missing after every source): **distressed 10 of 108, healthy 9 of 108**,
  almost all FY2016–2018. Of the 160 modelling rows, 135 keep usable financials after the pair rule.
- **Phase 4 on real reports:** tone for 201 of 210 reports (LM Master Dictionary 1993–2025), readability,
  auditor flags, drift. Leakage review of the 7 flagged reports done (`interim/qa/leakage_review.csv`):
  3 recommended for exclusion, not applied.
- **Pilot signal check** (same pair, same horizon, modelling rows only; descriptive): the ratios separate
  strongly (current ratio lower for the insolvent firm in 53 of 61 pairs, Altman Z'' in 44 of 51); language modestly
  (MD&A hedging higher in 47 of 71, auditor's-report negative tone in 48 of 74, MD&A negative tone in 44 of 71;
  going concern 6 pairs to 0). Drift shows nothing yet. `interim/qa/pilot/signal_check.csv`.

**Not done / needs the team**
- Confirm the provisional exclusions (financial sector, government companies, BSE group A).
- Leakage: exclude the 3 reports that discuss the company's own insolvency petition?
- **Phase 2 change request:** the CARO annexure is missed in 66 of 210 reports and the audit opinion in
  35; fixing it means changing Phase 2 code.
- Approve the full-cohort downloads (an estimated ~1,400 reports, ~9–10 GB, plus ~4,900 small industry
  lookups). 13 pilot firms found no peer among the ≤30 traded members of their sub-group; the full
  industry list would fix that.
- Fill the financials spot-check sheet (`interim/qa/financials_spot_check.csv`, 22 company-years).

**Next**
1. Team decisions above, then the full cohort.
2. Phase 2 CARO/opinion fix once approved; re-run Phase 4.
3. Stream A (FinBERT) and the models, on Colab.

## 2026-09-23 — Phase 4 stream B built (language features, `bpp language-features`)

**Done**
- `bpp language-features`, plus `src/bpp/nlp/{lexicon,readability,lm,drift,features}.py` and
  `docs/07_phase4_language.md`.
- Tone (Loughran-McDonald, seven categories), hedging density, an India/IBC distress phrase **seed**
  list, Gunning Fog readability and length — computed for the **MD&A and the auditor's report
  separately**, so the Phase 7 "MD&A only vs auditor only vs both" ablation is possible.
- Auditor flags read from Phase 2's sections: going concern, emphasis of matter, opinion severity as
  an ordinal, and the two CARO clauses that matter (loan default, unpaid statutory dues).
- Perplexity under an interpolated **Kneser-Ney** model trained on healthy-firm MD&A, written here
  because nltk is not a dependency. Left empty unless a training set is named, so the reference model
  cannot be fitted across folds.
- Year-on-year **drift** (contribution A): Jaccard, cosine, new-word share, and deltas in tone,
  hedging and readability, always against the same firm's previous year.
- `CORE_LANGUAGE_FEATURES` names the 25 values the model takes, matching Table 7; the table carries
  more for the ablations. **The model code should read that list, not hard-code 25.**

**Checked on synthetic data (not real results)**
- 111 new tests pass. The whole suite is 314 passing.
- An adversarial review was run again after the tests first came back green, and found 16 more
  defects — the same lesson as Phase 3, that a green suite on clean fixtures proves very little.
  The ones worth knowing:
  - **Both CARO flags were 1.0 for every company.** A clean annexure says "has **not** defaulted",
    "**Neither** the Company **nor** its promoters ... wilful defaulter" and "there are **no**
    undisputed statutory dues outstanding"; some auditors also reproduce the Order's own wording,
    "**whether** the company has defaulted ... if yes". Now judged clause by clause.
  - **Fog was ~38 for any section of headings and bullets** — which is most PDF-extracted MD&A —
    because the splitter needed a full stop and returned one sentence for the whole section.
  - "going concern" in the distress seed list made `auditor_distress_phrase_density` constant: SA 570
    puts it in every clean auditor's report.
  - Drift was keyed on `(firm, fy)` while rows are per `doc_id`, so two documents in one firm-year
    swapped drift; and a fiscal year stored as a string silently disabled drift for every row.
  - Perplexity used raw counts at the lower orders, making it absolute discounting with a
    Kneser-Ney unigram rather than Kneser-Ney, and an `order: 1` config gave every document the
    same score.
  - Hyphenated compounds ("non-performing") could never match the dictionary.
  All fixed, each with a regression test in `tests/test_language_robustness.py`.

**Not verified yet (needs real reports)**
- Nothing here has met a real annual report, for the same reason as Phase 3.
- The Loughran-McDonald dictionary has not been downloaded, so tone has only been exercised against
  a small hand-written list. Someone must fetch it from sraf.nd.edu into `data/manual/`.
- The distress phrase list is a **seed**, not contribution C's mined lexicon.

**Next**
1. Download the LM dictionary; note its version in `decisions_log.md`.
2. After the first real batch: group every flag by `label` and confirm the classes differ. A
   constant feature is the failure mode of this phase.
3. Stream A (FinBERT), NER masking, coreference, SVO triplets and the mined lexicon — all Colab.

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
