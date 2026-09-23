# Phase 1 — Building the cohort (weeks 1–3)

**Goal:** a list of ~75–100 listed companies admitted to CIRP under the IBC, each paired
with a similar healthy company, plus the list of firm-years whose annual reports we need.

**End products**

| File | What it is |
| --- | --- |
| `data/processed/cohort.csv` | one row per firm: pair id, role (distressed/healthy), reference (admission) date, industry, assets, match quality |
| `data/processed/sample_frame.csv` | one row per firm × fiscal year to collect (4 FYs before admission per firm) |
| `data/processed/cohort_unmatched.csv` | distressed firms we could not pair, with the reason |

```
IBBI announcements ──► CIRP debtors ──► match to listed firms ──► HUMAN REVIEW ──► peer matching ──► sample frame
   (1.1 scrape)         (1.2 filter)      (1.3 + 1.4 names)          (1.5)          (1.6 + 1.7)        (1.7)
```

Run `bpp status` at any time to see which of these files exist and what to do next.

---

## Step 1.1 — Download IBBI public announcements

```powershell
bpp ibbi-scrape --max-pages 5     # try a few pages first
bpp ibbi-scrape                   # then everything (about 750 pages, ~30 min with the 2 s delay)
```

**What it does.** When the NCLT admits a company into CIRP, the Interim Resolution
Professional publishes a Public Announcement (Form A) within about 3 days. IBBI lists every
announcement at <https://ibbi.gov.in/en/public-announcement>. The scraper downloads each
listing page, saves the raw HTML in `data/raw/ibbi/pages/`, and parses the table
(Type of PA, Date of Announcement, Name of Corporate Debtor, Applicant, IP, PDF link).

**Why it is built this way.**
- Pages are cached, so if your laptop sleeps or the site times out, re-running continues where it stopped.
- Columns are found by their header text, not position, so a small layout change on the site does not silently mix up columns.
- A 2-second delay between requests (config `ibbi.request_delay_s`) keeps us from being blocked.

**Output:** `data/raw/ibbi/public_announcements.csv`

**If it breaks:** open one saved page in a browser. If the table has different headers, update
`HEADER_MAP` in `src/bpp/scrape/ibbi.py`. To force a fresh download, delete `data/raw/ibbi/pages/`
or use `--no-resume`.

## Step 1.2 — Keep CIRP announcements for companies that could be listed

```powershell
bpp ibbi-filter
```

**What it does.**
1. Keeps rows whose type contains "Corporate Insolvency Resolution" (drops liquidation, voluntary liquidation, pre-pack).
2. Drops Private Limited companies, OPCs and LLPs (they cannot be listed).
3. Merges duplicate announcements of the same debtor and keeps the **earliest** date as the admission proxy.

**Output:** `data/interim/ibbi_cirp_filtered.csv` (one row per debtor)

## Step 1.3 — Build the universe of listed companies

```powershell
bpp listed-fetch
```

**What it does.** Downloads the BSE list of scrips for **Active, Suspended and Delisted**
companies (most CIRP firms are suspended or delisted, so an active-only list would miss them),
and NSE's equity list. They are merged on ISIN into one table with a stable `firm_id`
(`BSE<scrip code>`, or `NSE_<symbol>` if there is no BSE code).

**Manual fallback.** Company missing? Add it to `data/manual/listed_companies_extra.csv`
(copy the template) and re-run. If the BSE/NSE download is blocked, the whole universe can come
from that file.

**Output:** `data/interim/listed_universe.csv`

## Step 1.4 — Match IBBI debtor names to listed companies

```powershell
bpp match-names
```

**What it does.** Names differ in small ways ("Jaypee Infratech Ltd." vs "JAYPEE INFRATECH
LIMITED"), so names are normalised (upper case, punctuation and "Limited/Ltd/The" removed,
"formerly known as" aliases also tried) and compared with fuzzy matching
(rapidfuzz `token_sort_ratio`, 0–100).

| Score | Status | Meaning |
| --- | --- | --- |
| ≥ 98 | `auto_accepted` | practically identical; `accept` is pre-filled `Y` |
| 80–97 | `needs_review` | a person must decide (e.g. "Tarang Infra" vs "Tarangi Infra" are different companies) |
| < 80 | `no_match` | most IBBI debtors are unlisted; ignore |

**Output:** `data/interim/ibbi_listed_matches.csv`

## Step 1.5 — HUMAN REVIEW of matches (do not skip)

1. Open `data/interim/ibbi_listed_matches.csv` in Excel.
2. For every `needs_review` row, set `accept` to `Y` or `N` (check the company on BSE/NSE or
   news: same city, same promoter, same industry?). Spot-check ~10 `auto_accepted` rows too.
3. For accepted firms, fill in as much as you can:
   - `admission_date_verified` — the insolvency commencement date from the Public Announcement PDF (link in `pa_pdf_url`) or the NCLT order. Format `YYYY-MM-DD`.
   - `petition_date` — date the Section 7/9/10 application was filed, from the NCLT order. Format `YYYY-MM-DD`. Reports published after this are excluded later (leakage).
   - `business_group` — needed for the adversarial debiasing contribution (e.g. "Jaypee Group", or blank for standalone firms).
4. **Save As** `data/manual/ibbi_listed_matches_reviewed.csv` (keep CSV format).

Why: a wrong match puts a healthy company in the distressed class, which silently poisons every model.

## Step 1.6 — Fill in firm financials (industry + total assets)

The peer-matching step needs, for every candidate company and fiscal year:

```
firm_id, company_name, fy, industry_code, total_assets [, business_group]
```

Save as `data/manual/firm_financials.csv` (template in the same folder).

- **With CMIE Prowess:** export NIC industry code and total assets for all listed companies, FY2012 onwards; map company codes to our `firm_id` using the ISIN or BSE code in `listed_universe.csv`.
- **Without Prowess:** start with the distressed firms and 5–10 same-industry candidates each; take total assets from the balance sheet in the annual report (or screener-style sites), industry from the BSE industry field.
- `fy` is the year the fiscal year **ends** (FY2019 = Apr 2018–Mar 2019). Use the same unit (₹ crore) for everyone.

## Step 1.7 — Match healthy peers and build the sample frame

```powershell
bpp build-cohort
# quick unreviewed trial run only:  bpp build-cohort --use-auto-matches
```

**What it does, for each distressed firm D admitted on date A:**

1. **Reference year:** the latest FY that **ended before** A and has financials.
2. **Eligibility:** D needs at least 3 FYs of financials before A; admission after 2017-06-01.
3. **Healthy candidates** must
   - never appear in the CIRP match file (auto or needs-review, unless a reviewer marked them `N`),
   - have the **same industry code** (fallback: same first 2 digits, recorded as `industry_prefix_2`),
   - have total assets within **±30%** of D in the reference year,
   - have financials for the same 3 years.
4. **Closest wins:** smallest |log(assets ratio)|. Firms with the fewest candidates are matched
   first so scarce peers are not used up; each healthy firm is used at most once.
5. **Sample frame:** for each firm in a pair, the **4 FYs** ending before A. We need only 3 (t-1…t-3)
   but collect 4, because the most recent report is often published too close to admission and is excluded in Phase 2.

All of these numbers are in `configs/config.yaml` under `cohort:`.

**Checks before moving on**
- `cohort_unmatched.csv`: if many firms fail on "no peer in industry within asset tolerance", widen `asset_tolerance` to 0.5 or use the 2-digit fallback, and **write the change in `docs/decisions_log.md`**.
- Count pairs: aim for ≥ 75. Look at `match_quality` — report the share of exact-industry matches in the paper.

---

### Try it on fake data first

`bpp demo` runs all of Phase 1 and 2 on invented companies in `data_demo/`. Open
`data_demo/interim/ibbi_listed_matches.csv` to see a `needs_review` row and
`data_demo/processed/cohort.csv` to see the pairs.
