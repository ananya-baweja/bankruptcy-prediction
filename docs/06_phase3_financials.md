# Phase 3 — financial features (`bpp financials`)

Turns the annual-report PDFs Phase 2 collected into the 12 stream-C ratios, one row per
company-year. CMIE Prowess is not available to this project, so every figure comes out of the
statements themselves. **MCA21 is not used.**

```powershell
bpp financials          # extract -> validate -> ratios -> spot-check sheet
bpp financials-score    # accuracy, once the spot-check sheet is filled in
```

Phase 2 must have run first: Phase 3 reads `data/interim/pages/<doc_id>.json`, so OCR is never
repeated.

---

## Step 3.1 — What it does, per report

| # | Step | Detail |
| --- | --- | --- |
| 1 | **Locate** | Heading patterns find the Balance Sheet, Statement of Profit and Loss and Cash Flow Statement, and decide whether each is **standalone** or consolidated. |
| 2 | **Read the header** | The unit (`(Rs. in crore)`, `(₹ in lakhs)`…) and the period columns (`As at 31 March 2019 | 2018`) are read once and applied to every line beneath. |
| 3 | **Walk the lines** | Sub-headings are tracked as context, so `Borrowings` under *Non-current liabilities* becomes long-term debt and under *Current liabilities* short-term debt. |
| 4 | **Map** | Line items are matched to the standard fields by pattern first, fuzzy similarity second. The score is kept. |
| 5 | **Both columns** | The current-year **and** prior-year columns are read, so one report covers two years. |

**Standalone, not consolidated.** Consolidated statements include subsidiaries, so they are the
wrong denominator for every ratio. The same preference already applies to the auditor's report and
CARO (`decisions_log.md`, 2026-09-16). A report that only has consolidated statements is used and
flagged `consolidated_only`.

**Ruled tables.** When pdfplumber or camelot can read the statement as a table, the column geometry
is used instead of whitespace. Both are optional (`pip install -e ".[tables]"`); the text walk is
what always runs, because a scanned statement has no table structure at all.

## Step 3.2 — Standard fields

`current_assets`, `current_liabilities`, `inventories`, `cash_and_equivalents`, `total_assets`,
`non_current_assets`, `non_current_liabilities`, `borrowings_long_term`, `borrowings_short_term`,
`total_equity`, `equity_share_capital`, `other_equity`, `retained_earnings`, `total_liabilities`,
`total_equity_and_liabilities`, `revenue`, `total_income`, `pbt`, `finance_costs`, `depreciation`,
`net_profit`.

`total_liabilities` (non-current + current) and `total_equity_and_liabilities` (= total assets) are
different quantities and are kept apart; conflating them makes the balance check fail on a correct
report.

Everything is stored in **₹ crore**, the unit the rest of the project already uses. The figure as
printed and its declared unit are kept alongside, so a scale error stays detectable and the
spot-check sheet can show the number the way the report showed it.

Schedule III usually prints only `other_equity` on the face of the balance sheet, with retained
earnings inside it in the notes. When `retained_earnings` is absent, `other_equity` stands in and
the substitution is recorded.

## Step 3.3 — As first published, and the gap-filling order

A figure is stored **as first published**: the number printed in that year's own report. Next
year's comparative never overwrites it — it is used to cross-check, and a difference beyond
`restatement_tolerance` sets `restated = True`.

When a company-year has no usable figures, in this order:

| # | Source | `financials_source` |
| --- | --- | --- |
| 1 | the next year's report, whose prior-year column covers this year | `next_report_comparative` |
| 2 | `data/manual/xbrl_financials.csv` — NSE/BSE XBRL annual results, entered by hand | `manual_xbrl` |
| 3 | mark missing and apply the pair-exclusion rule | `missing` |

Step 2 is a manual CSV rather than a scraper because the exchange endpoints block cloud addresses
and the BSE one is still unverified (`configs/config.yaml`). Columns:
`firm_id, fy, field, value_cr, source_url, note` — one row per figure, already in ₹ crore.
`bpp init` writes a template. It only ever **fills gaps**; it never overrides a published figure.

**Missing reports are a feature, not rows to drop.** Late and absent filing is itself a distress
signal (project plan, issue 7). Every expected company-year keeps its row, flagged
`financials_missing`. The pair-exclusion rule then mirrors Phase 2's `keep_pairs_aligned`: if a
distressed firm's year drops out, its healthy peer's same year drops out too
(`pair_partner_missing_financials`), so the model cannot learn the calendar year as a shortcut.

Every run prints **the count of unrecoverable company-years by class**. That number decides whether
the pair-matched design survives; if it is large on the distressed side, say so in
`progress_log.md` rather than quietly shrinking the cohort.

## Step 3.4 — Validation and confidence

| Check | Flag when it fails |
| --- | --- |
| `total_assets` = `total_equity_and_liabilities` (within `balance_tolerance`) | `balance_sheet_does_not_balance` |
| `total_assets` = equity + non-current + current liabilities | `equity_plus_liabilities_mismatch` |
| a subtotal contains its parts | `current_assets_smaller_than_its_parts`, `current_assets_exceed_total_assets` |
| assets vs the cohort's own `total_assets`, and vs last year | `unit_scale_differs_from_cohort`, `unit_scale_jump_vs_previous_year` |
| …and the gap matches a unit conversion | `possible_unit_confusion` |
| figures that cannot be negative | `negative_<field>` |
| the page carries too few figures to be a statement | `few_money_lines_on_page` |
| two different scales both look like captions | `unit_ambiguous_read_as_<unit>` |
| the first money column is not the report's own year | `first_column_is_not_the_report_year` |

A wrong unit is the error that still looks plausible — every ratio computes, the firm is simply
100× too large — so it is checked against two independent anchors. A firm can double in a year; it
cannot grow exactly hundredfold.

`confidence` per figure multiplies: number-parse confidence × label-match score × a unit factor,
reduced for a comparative source, a restatement, OCR'd pages and any validation flag.

## Step 3.5 — The 12 ratios (Table 7, stream C)

| Ratio | Definition |
| --- | --- |
| Current ratio | current assets / current liabilities |
| Quick ratio | (current assets − inventories) / current liabilities |
| Cash to assets | cash and equivalents / total assets |
| EBITDA margin | EBITDA / revenue |
| ROCE | EBIT / capital employed |
| ROA | **net profit** / total assets |
| Debt to equity | total borrowings / total equity |
| Debt to EBITDA | total borrowings / EBITDA |
| Interest coverage | EBIT / finance costs |
| Retained earnings to assets | retained earnings (or other equity) / total assets |
| Promoter pledge | *empty for now* — exchange shareholding filings, a later step |
| Altman Z'' | emerging-market form, below |

EBIT = PBT + finance costs. EBITDA = EBIT + depreciation. Capital employed is the reported figure
when there is one, else total assets − current liabilities.

**Altman emerging-market Z''** (Altman, Hartzell & Peck 1995):

```
Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4 + 3.25
X1 working capital / total assets      X3 EBIT / total assets
X2 retained earnings / total assets    X4 book value of equity / total liabilities
```

Sales/total assets is dropped to remove the industry effect. **Two scales, and they are easy to
confuse.** `Z''` above is the discriminant; adding 3.25 gives the *rating-equivalent* EMS scale, on
which 0 corresponds to a bond rating of D. The constant carries no information.

The published zones — **> 2.6 safe · 1.1–2.6 grey · < 1.1 distress** — belong to `Z''`, *without*
the constant. Many secondary sources print them next to the +3.25 formula, which lifts every firm by
3.25: a firm with negative working capital and near-zero EBIT then scores "safe". So `ratios.csv`
carries both, `altman_z_em` (with the constant, the stream-C feature) and `altman_z_dprime` (without
it), and `altman_zone` is taken from `altman_z_dprime`.

**Why a ratio can be empty.** A distressed Indian firm routinely has negative equity, negative
EBITDA and no finance cost in its worst year. Dividing by any of those inverts the meaning — a
debt-to-equity of −5.3 on wiped-out equity sorts as healthier than +8.0 on thin equity. Those
ratios return empty with a reason in `ratio_reasons`, and the condition is kept as its own feature:
`negative_equity`, `negative_ebitda`, `zero_finance_costs`, `negative_working_capital`,
`negative_capital_employed`. Altman X4 is the exception — Z'' is linear, so a negative term
correctly pushes the score down.

Ratios are stored **raw**. Winsorising at the 1st/99th percentile and scaling belong inside the
cross-validation folds (`ratios.winsorise_bounds`); bounds learnt over the whole table would leak
the test fold's distribution into training.

## Step 3.6 — Spot check (report this number)

```powershell
bpp financials            # writes data/interim/qa/financials_spot_check.csv
# open each PDF at the page given and fill in value_correct / page_correct (Y or N)
bpp financials-score
```

A random 10% of company-years (`financials.spot_check_fraction`), every extracted field listed with
the **page number** and the label it matched. Fill `value_correct`, `page_correct`, and
`true_value_cr` where wrong. Split it among the team as with the Phase 2 section QA sheet. Sort by
`confidence` ascending to find the failures fastest; if a field scores below ~90%, add a pattern to
`FIELD_PATTERNS` in `features/statements.py`, re-run, and re-check only the failed rows.

## Outputs

| File | Content |
| --- | --- |
| `processed/financials_figures.csv` | one row per figure: value, source, page, statement, label, match score, parse flags, confidence, restatement |
| `processed/financials_extracted.csv` | one row per company-year, one column per standard field, plus validation and missing flags |
| `processed/ratios.csv` | the stream-C table: 12 ratios, Altman zone, condition indicators |
| `processed/financials_missing.csv` | the company-years with no usable figures |
| `interim/qa/financials_spot_check.csv` | the hand-check sheet |
| `interim/qa/financials_spot_check_scores.csv` | accuracy per field, from `bpp financials-score` |

`data/manual/firm_financials.csv` is the team's **Phase 1 input** (total assets for peer matching)
and is never written to by Phase 3. Phase 3 does read it, as one of the two anchors for the
unit-scale check.

---

### Team checklist for Phase 3

- [ ] `bpp financials` run; the unrecoverable-by-class counts recorded in `progress_log.md`
- [ ] `consolidated_only` reports listed and checked — is a standalone set really absent?
- [ ] `balance_sheet_does_not_balance` and `possible_unit_confusion` rows opened and fixed
- [ ] `restated = True` figures reviewed; genuine restatements noted in `decisions_log.md`
- [ ] gaps filled from `manual/xbrl_financials.csv` where the numbers are obtainable
- [ ] spot-check sheet filled by the team; `bpp financials-score` accuracy recorded
- [ ] `processed/ratios.csv` uploaded to the shared Google Drive for Phases 5–6 on Colab
