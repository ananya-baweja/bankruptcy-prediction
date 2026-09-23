# Phase 2 — Collecting annual reports and extracting sections (weeks 2–4)

**Goal:** for every firm-year in `sample_frame.csv`, a PDF on disk, its page text, the sections
we model (MD&A, Board's report, auditor's report + sub-sections, CARO annexure), and a label with
leakage rules applied.

**End products**

| File | What it is |
| --- | --- |
| `data/interim/documents.csv` | manifest: every PDF, where it came from, publication date, checksum |
| `data/interim/pages/<doc_id>.json` | text of every page, with how it was read (text layer / OCR) |
| `data/interim/sections/<doc_id>.json` | the extracted sections, audit opinion, leakage counts, warnings |
| `data/interim/extraction_report.csv` | one row per report: characters per section, OCR pages, warnings |
| `data/processed/documents_labeled.csv` | **the modelling table**: label, horizon (t-1/t-2/t-3), included or exclude reason |
| `data/processed/missing_reports.csv` | firm-years with no report (late/missing filing is a feature later) |
| `data/interim/qa/section_qa_scores.csv` | hand-checked extraction accuracy (goes in the paper) |

`doc_id` is always `<firm_id>_FY<year>`, e.g. `BSE532532_FY2017`.

---

## Step 2.1 — Find out which reports exist online

```powershell
bpp reports-list --firm-id BSE500325 --source nse     # test on ONE company first
bpp reports-list                                       # then all cohort firms
```

**What it does.** For each cohort firm, asks NSE (via the open-source `nse` package, which
handles NSE's cookies) and BSE which annual reports they host, and caches the answers in
`data/raw/annual_reports/_listings/`. Output: `data/interim/report_listings.csv`
(firm_id, source, fy, url, pub_date).

**Known limits (be ready for these).**
- The **BSE annual-report endpoint is unverified**: BSE blocks requests without browser headers and changes endpoints. If BSE returns errors, use NSE plus manual collection, or fix `reports.bse_annual_report_endpoint` in the config after checking the request your browser makes (F12 → Network) on a company's BSE "Annual Reports" page.
- NSE/BSE often block cloud IPs: run this on a laptop, not Colab.
- Suspended/delisted firms often have no reports on the exchange sites any more — see Step 2.3.

## Step 2.2 — Download the PDFs

```powershell
bpp reports-download --limit 10     # small batch first, then without --limit
```

Files are saved as `data/raw/annual_reports/<firm_id>/FY<year>.pdf`. Anything already on disk
is skipped, and manually collected files are never overwritten. Zipped downloads are unzipped;
HTML "blocked" pages are rejected (a file must start with `%PDF`).

## Step 2.3 — Collect the rest by hand, then register them

For firm-years still missing (`bpp status`, or compare `sample_frame.csv` with `documents.csv`):

1. Look on: the company's website (Investors → Annual Reports), BSE/NSE company pages, MCA filings, the Wayback Machine (web.archive.org) for old company sites.
2. Save the PDF as `data/raw/annual_reports/<firm_id>/FY<year>.pdf`, **or** anywhere and list it in `data/manual/manual_reports.csv`.
3. In `manual_reports.csv` also record `pub_date` (the date it was filed with the exchange or dated on the AGM notice) and `source_url`. Publication dates matter for the leakage rules.
4. Run:

```powershell
bpp reports-register
```

## Step 2.4 — Extract page text (with OCR for scanned pages)

```powershell
bpp extract-text
```

**What it does.** Opens each PDF with PyMuPDF and reads the text layer page by page. A page with
fewer than 50 characters is probably a scanned image (common in older small-cap reports), so it
is rendered at 300 dpi and read with **Tesseract OCR**. If Tesseract is not installed, the page is
marked `needs_ocr` and the run continues. Page text is saved separately from sections, so section
rules can be changed and re-run in seconds without repeating slow OCR.

Check `n_needs_ocr_pages` in the extraction report; if it is not 0, install Tesseract (see `01_setup.md`)
and run `bpp extract-text --force`.

## Step 2.5 — Split reports into sections

```powershell
bpp extract-sections              # add --force after changing rules
```

**How it works** (rule-based, explainable in a viva — code in `src/bpp/extract/sections.py`):

1. **Find heading lines.** Short lines (or a short line joined with the next, for wrapped
   headings) are matched against patterns for our sections and for "boundary" sections that end
   them (Report on Corporate Governance, Secretarial Audit Report, Business Responsibility Report,
   Annexure B, Balance Sheet, AGM Notice). Indian variants are covered: *Board's Report /
   Directors' Report / Report of the Board of Directors*, *Annexure 'A' to the Independent
   Auditors' Report*, *Annexure II – Management Discussion and Analysis*, numbering like "1." or "(iv)".
2. **Remove false headings.**
   - Contents pages: an early page that names 3+ different sections.
   - Contents-style lines: dotted leaders or trailing page numbers.
   - Sentences: a "heading" followed on the same line by words like *is, are, forms, annexed*.
   - **Cross references:** inside the Board's Report there is usually a sub-heading
     "Management Discussion and Analysis" followed by "…is presented in a separate section forming part
     of this Annual Report". Such hits are dropped when the real heading exists elsewhere.
3. **Runs.** Consecutive hits for the same section (e.g. running page headers) merge; a section ends where
   a different one starts. A short run sandwiched inside another section is absorbed as a sub-heading.
4. **Pick.** MD&A and Board's report: the longest run. Auditor's report and CARO: the first long run
   that is about the **standalone** (not consolidated) financial statements.
5. **Inside the auditor's report:** sub-sections *Basis for Qualified/Adverse Opinion / Disclaimer*,
   *Material Uncertainty Related to Going Concern*, *Emphasis of Matter*, and the opinion type
   (`unmodified / qualified / adverse / disclaimer`).
6. **Leakage counts:** mentions of IBC/NCLT (generic) and CIRP-specific terms (resolution professional,
   committee of creditors, Section 7/9/10 application…).

Then open `data/interim/extraction_report.csv`: sort by `mdna_chars` and `warnings` to find failures fast.
Typical MD&A is 8,000–60,000 characters; under ~2,000 usually means a wrong split.

## Step 2.6 — Labels, horizons and leakage exclusions

```powershell
bpp assign-labels
```

**Rules, in order** (config `labels:`):

| Rule | Why |
| --- | --- |
| Publication date = real filing date; if unknown, FY end + 183 days (`pub_date_source = assumed`) | the label depends on when the market could read the report, not the fiscal year |
| Exclude reports published **after the petition date** (when known) | they describe the NCLT case openly |
| Exclude reports published **within 180 days before admission** (or after) | same, when the petition date is unknown |
| If a distressed firm's FY is excluded/missing, drop the peer's same FY (`pair_partner_excluded`) | both classes cover the same years, so the model cannot learn "year" as a shortcut |
| Rank remaining FYs of each pair newest first → `t-1`, `t-2`, `t-3`; drop older (`beyond_n_years_before`) | enables the separate t-1 vs t-2 experiment (the 12–24-month claim) |
| Flag distressed reports that still contain CIRP terms (`needs_leakage_review`) | a person reads them and decides; add the decision to `decisions_log.md` |

Also added: `months_before_reference`, `within_12m`, `within_24m`, `filing_delay_days`,
`audit_opinion`, `has_<section>` flags.

## Step 2.7 — Measure extraction accuracy (report this number)

```powershell
bpp qa-sample            # 30 reports x 4 sections -> data/interim/qa/section_qa_sheet.csv
# fill found_correct / start_correct / end_correct with Y or N, looking at the PDF pages given
bpp qa-score
```

Split the 30 reports among the team (e.g. 10 each). If a section scores below ~85%, look at the
`notes` column for patterns, add or adjust a regex in `sections.py`, run `bpp extract-sections --force`,
and re-check only the failed rows.

---

### Team checklist for Phase 2

- [ ] NSE listing tested on 1 firm, then run for all
- [ ] Downloads done; missing list shared and split among the team
- [ ] Manual PDFs registered with publication dates
- [ ] Tesseract installed; `n_needs_ocr_pages` = 0
- [ ] Extraction report reviewed; obvious failures fixed
- [ ] `needs_leakage_review` rows read and decided
- [ ] QA sheet filled; accuracy per section recorded in `progress_log.md`
- [ ] `data/interim/sections/` and `data/processed/` uploaded to the shared Google Drive for Phase 4 on Colab
