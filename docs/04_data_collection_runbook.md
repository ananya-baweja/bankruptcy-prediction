# Data Collection Runbook

Everything needed to go from an empty `data/` folder to a labelled dataset. Follow the stages in order.
Commands are run from the project folder with the `bpp` environment active.

**Target when you are finished**

| Item | Target | File that proves it |
| --- | --- | --- |
| Insolvent listed companies, verified | 100 or more (150+ if data allows) | `data/manual/ibbi_listed_matches_reviewed.csv` |
| Matched healthy peers | One per insolvent firm | `data/processed/cohort.csv` |
| Firm-years needed | About 800 (200 firms x 4 years) | `data/processed/sample_frame.csv` |
| Annual reports collected | 70% or more of the firm-years | `data/interim/documents.csv` |
| Reports with all four sections found | 85% or more | `data/interim/extraction_report.csv` |
| Labelled rows for modelling | Balanced, pairs aligned | `data/processed/documents_labeled.csv` |

Check progress at any time with `bpp status`.

---

## Stage 0 — Setup (Day 1, everyone, 1 hour)

1. Install the environment on each laptop: `conda env create -f environment.yml`, then `conda activate bpp`.
2. Install Tesseract OCR and set `extraction.ocr.tesseract_cmd` in `configs/config.yaml`.
3. Run `pytest` (35 tests should pass) and `bpp demo` (runs the whole pipeline on fake data).
4. Create a shared Google Drive folder named `BPP-data` with the same structure as `data/`.
   One person owns the master copy; the others send files to that person rather than editing the same file.
5. Agree roles: **Data lead** (Stages 1, 2, 4), **Financials** (Stage 3), **Reports** (Stage 5, everyone helps),
   **Extraction and QA** (Stage 6).

---

## Stage 1 — Insolvency records: the labels (Day 1–3, data lead)

### 1.1 Test the scraper on a few pages

```
bpp ibbi-scrape --max-pages 3
```

Open `data/raw/ibbi/public_announcements.csv`. You should see columns for type of announcement,
date, corporate debtor, applicant, insolvency professional and the PDF link. If the file is empty or the
columns are shifted, stop and report it before running the full scrape.

**What the site looks like (checked 22 September 2026):** `?page=0` is the newest page, each page holds
16 to 20 announcements, and page 5 was already one month back. Going back to December 2016 is roughly
700 to 900 pages. With the built-in 2-second delay the full run takes 30 to 45 minutes.

### 1.2 Run the full scrape

```
bpp ibbi-scrape
```

Leave it running. Pages are cached in `data/raw/ibbi/pages/`, so if the laptop sleeps or the connection
drops, running the command again continues from where it stopped.

### 1.3 Filter to the cases that matter

```
bpp ibbi-filter
```

This keeps insolvency cases only, removes private limited companies, LLPs and one-person companies, and
keeps the earliest announcement per company. Expect a few thousand rows in
`data/interim/ibbi_cirp_filtered.csv`. Most are unlisted; the next stage finds the listed ones.

### 1.4 Sanity check (10 minutes, do not skip)

Pick five rows at random, search the company name on the IBBI website and confirm the date matches.
If the dates are wrong, everything downstream is wrong.

**If the scraper fails:** use the IBBI Corporate Debtor search at `ibbi.gov.in/claims/claims`, which can
return the complete record set, or collect the last three years by hand from the announcement pages.
A smaller, correct list beats a large, broken one.

---

## Stage 2 — Find the listed companies and verify them (Day 2–4, everyone)

### 2.1 Download the company lists

```
bpp listed-fetch
```

Check `data/interim/listed_universe.csv`. It must include **suspended and delisted** companies, not only
active ones. If the download fails, download the equity lists from the BSE and NSE websites by hand and
put the rows into `data/manual/listed_companies_extra.csv` (columns: company_name, bse_code, nse_symbol,
isin, status, industry), then re-run.

### 2.2 Match names

```
bpp match-names
```

Produces `data/interim/ibbi_listed_matches.csv` with three groups: `auto_accepted` (score 98 or above),
`needs_review` (80 to 97) and `no_match`.

### 2.3 Verify every match by hand — the most important manual step

Split the rows between the team (for example 4 people, 60 rows each). For each row:

1. Search the matched company on the BSE or NSE website. Confirm it is the same legal entity, not a
   similarly named one. Watch for "India" in the name, and for group companies with near-identical names.
2. Open the Public Announcement PDF from the `pa_pdf_url` column. It states the **insolvency commencement
   date**. Copy it into `admission_date_verified` as `YYYY-MM-DD`.
3. If you can find the NCLT order (nclt.gov.in) or the company's own exchange filing, note the date the
   petition was **filed** in `petition_date`. This is optional but valuable: it sharpens the exclusion rule.
4. Fill `business_group` (for example the parent group name) where one exists; leave blank otherwise.
5. Set `accept` to `Y` or `N`.

Save the completed file as `data/manual/ibbi_listed_matches_reviewed.csv`.

**Rule of thumb:** about 20 to 40 minutes per 10 rows once you have a rhythm. Aim for at least 100 accepted
firms; accept more if you find them, since a larger sample narrows the confidence intervals.

---

## Stage 3 — Financial data (Day 3–8, financials owner, runs in parallel with Stage 5)

The file to produce is `data/manual/firm_financials.csv` with these exact columns:

```
firm_id, company_name, fy, industry_code, total_assets
```

`fy` is the year the financial year ends (FY2019 = April 2018 to March 2019). Use the same unit
(₹ crore) for every company. `firm_id` must match the ids in `listed_universe.csv` (for example `BSE532532`).

### Option A — CMIE Prowess (if the college has access)

Export, for all listed companies and FY2012 onwards: company name, ISIN, NIC industry code, total assets.
Map ISIN to `firm_id` using `listed_universe.csv`. One export covers everything.

### Option B — No Prowess (the realistic path)

1. **Peer pool in one shot.** Use a stock screening site that allows exporting a company list with total
   assets and industry. One export gives a few thousand companies, which is your peer pool.
2. **Insolvent firms one by one.** These are usually suspended and often missing from screening sites, so
   take total assets from the balance sheet of each annual report you collect in Stage 5. One number per
   company-year.
3. **Industry code.** Use the industry field from the exchange list. Consistency matters more than the
   classification system: the same label must mean the same industry for every company.

**Minimum needed to start:** four years of total assets for each insolvent firm, and the same four years
for at least five candidate peers in its industry. You do not need the full ratio set yet; that comes in
the modelling phase.

---

## Stage 4 — Build the cohort (Day 8, data lead, 15 minutes)

```
bpp build-cohort
```

Then check:

- `data/processed/cohort.csv` — the matched pairs. Count them. Target 75 or more.
- `data/processed/cohort_unmatched.csv` — firms with no peer, and the reason.
- `data/processed/sample_frame.csv` — the firm-years to collect. This is your shopping list for Stage 5.

If many firms fail with "no peer in industry within asset tolerance", widen `cohort.asset_tolerance` from
0.30 to 0.50 in `configs/config.yaml`, re-run, and record the change in `docs/decisions_log.md`.

---

## Stage 5 — Annual reports (Day 5–20, everyone; this is the long pole)

### 5.1 Automatic first

```
bpp reports-list --firm-id <one firm> --source nse     # test on one company
bpp reports-list                                        # then all firms
bpp reports-download --limit 10                         # test
bpp reports-download                                    # then the rest
```

Run this from a laptop on a normal Indian internet connection, not from Colab. Files land in
`data/raw/annual_reports/<firm_id>/FY<year>.pdf`.

### 5.2 Then by hand, in this order of ease

For each missing firm-year (see `bpp status` and `data/processed/missing_reports.csv`):

1. **BSE company page** — the Financials or Annual Reports tab for that scrip.
2. **NSE company page** — the Financials tab, Annual Reports section.
3. **Company website** — Investors or Investor Relations, then Annual Reports. Best source for older years.
4. **Stock information sites** that host exchange copies of annual reports on the company page.
5. **Wayback Machine (web.archive.org)** — paste the old company website address. This is how you recover
   reports for delisted companies whose sites are gone.
6. **MCA portal** — paid, a last resort for a few critical firms.

Save each file as `data/raw/annual_reports/<firm_id>/FY<year>.pdf`, using the FY in which the year **ends**.

### 5.3 Record the publication date

In `data/manual/manual_reports.csv` add one row per file:

```
firm_id, fy, local_path, pub_date, source_url, note
```

The publication date, in order of preference:

1. The date the exchange lists against the annual report filing (shown on the BSE or NSE page).
2. The date of the AGM notice or the board's report inside the PDF.
3. Leave blank — the pipeline then assumes the financial year end plus six months and marks it `assumed`.

This date decides the label, so it is worth two extra minutes per report.

### 5.4 Keep the work moving

- Assign firms, not years: one person takes 25 companies and collects all four years for each.
- A realistic pace is 15 to 20 reports per person per day once the routine is set. Four people for five
  working days covers 300 to 400 reports.
- Register what has been collected so far at the end of each day:

```
bpp reports-register
bpp status
```

- If a company yields nothing at all, mark it in your tracking sheet and move on. Missing reports are
  recorded by the pipeline and are themselves a feature.

---

## Stage 6 — Extract, label and check (Day 12–22, extraction owner)

```
bpp extract-text          # PDF text, with OCR for scanned pages
bpp extract-sections      # MD&A, Board's Report, Auditor's Report, CARO annexure
bpp assign-labels         # exclusion rules, horizons, labels
```

Then review `data/interim/extraction_report.csv`:

- Sort by `mdna_chars`. Anything under about 2,000 characters usually means a wrong split — open the PDF
  and check.
- Check `n_needs_ocr_pages`. If it is not zero, Tesseract is missing or misconfigured; fix it and run
  `bpp extract-text --force`.
- Check the `warnings` column for reports where a section was not found.

Then measure accuracy properly:

```
bpp qa-sample             # creates a sheet of 30 reports x 4 sections
# split the sheet between the team; open each PDF at the pages shown and fill
# found_correct / start_correct / end_correct with Y or N
bpp qa-score
```

Target 85% or better per section. Below that, look at the `notes` column for a pattern, adjust the heading
rules in `src/bpp/extract/sections.py`, re-run `bpp extract-sections --force`, and re-check only the rows
that failed.

Finally, read every report flagged `needs_leakage_review` in `documents_labeled.csv` and decide whether it
gives away the insolvency outcome. Record each decision in `docs/decisions_log.md`.

---

## Stage 7 — Pilot signal check (as soon as 30 pairs have reports)

Do not wait for the full dataset. With about 30 insolvent firms and their peers:

1. Compute negative tone, uncertainty words, hedging density, the going-concern flag and the CARO default
   flag for the t-2 reports.
2. Compare the two groups. The published Indian study found 2.21% against 1.30% negative words; if your
   numbers point the same way, the language signal is present in your sample.
3. Run a simple logistic regression three ways: text features only, ratios only, both.

This is the early answer to the question "does the text actually help", and it arrives weeks before the
full model.

---

## Definition of done

- [ ] `ibbi_listed_matches_reviewed.csv` complete, every row marked Y or N, dates verified
- [ ] `firm_financials.csv` covers all insolvent firms and their candidate peers for four years
- [ ] `cohort.csv` has 75 or more pairs
- [ ] 70% or more of `sample_frame.csv` firm-years have a PDF in `documents.csv`
- [ ] Publication dates recorded for most reports (`pub_date_source = filing`, not `assumed`)
- [ ] `n_needs_ocr_pages` is zero across the extraction report
- [ ] QA accuracy measured and recorded, 85% or better per section
- [ ] Leakage-flagged reports read and decided
- [ ] `data/` uploaded to the shared Drive folder

---

## Common problems and what to do

| Problem | Fix |
| --- | --- |
| IBBI scrape returns no rows | Open a cached page from `data/raw/ibbi/pages/` in a browser. If the table changed, update `HEADER_MAP` in `src/bpp/scrape/ibbi.py`. Fall back to the IBBI corporate debtor search. |
| BSE or NSE download fails or returns 403 | Run from a laptop, not Colab; wait and retry; collect those firms by hand. |
| Two companies with almost the same name | Compare CIN or ISIN, not the name. If still unclear, mark `accept = N` and add a note. |
| A company changed its name before insolvency | Search the old name too; IBBI often writes "formerly known as". Record both names in `notes`. |
| Annual report is a scanned image | Keep it. OCR handles it, and the OCR page count is reported per document. |
| Report covers 15 months or a changed year end | Record the FY in which the period ends and add a note; check the date logic for that firm by hand. |
| Team members overwrite each other's CSV | One owner per file. Others send their portion as a separate file for the owner to merge. |

---

## Data use

All sources used here are public: IBBI announcements, exchange filings and company annual reports. The
scrapers wait between requests so the sites are not overloaded. Collected PDFs are kept for the project
and are not redistributed. If the college provides CMIE Prowess, its licence terms apply to that data.
