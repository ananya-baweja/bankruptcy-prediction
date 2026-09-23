# Data dictionary

All paths are relative to the data folder (`data/` by default, or `--data-dir` / `BPP_DATA_DIR`).
Conventions: `firm_id` = `BSE<scrip code>` (or `NSE_<symbol>`), `fy` = year the fiscal year **ends**
(FY2019 = Apr 2018–Mar 2019), `doc_id` = `<firm_id>_FY<fy>`, dates `YYYY-MM-DD`.

## Filled in by the team (`manual/`)

| File | Columns | Notes |
| --- | --- | --- |
| `ibbi_listed_matches_reviewed.csv` | all columns of `interim/ibbi_listed_matches.csv`; you edit `accept` (Y/N), `admission_date_verified`, `petition_date`, `business_group`, `notes` | source of truth for which firms are distressed |
| `firm_financials.csv` | `firm_id, company_name, fy, industry_code, total_assets` (+ `business_group`) | one row per firm-year; assets in ₹ crore for everyone |
| `manual_reports.csv` | `firm_id, fy, local_path, pub_date, source_url, note` | `local_path` may be blank if the PDF follows the naming convention |
| `listed_companies_extra.csv` | `company_name, bse_code, nse_symbol, isin, status, industry` | companies missing from the BSE/NSE lists |
| `loughran_mcdonald.csv` | the Master Dictionary as published, or a simple `word,category` file | Phase 4 tone features. Downloaded once from sraf.nd.edu; **not** in the repo, because it is republished yearly and a silent copy makes a tone number irreproducible. Record the version in `decisions_log.md` |
| `xbrl_financials.csv` | `firm_id, fy, field, value_cr, source_url, note` | Phase 3 gap-fill, step 2: figures typed in from NSE/BSE XBRL annual results, already in ₹ crore. One row per figure. Fills gaps only — never overrides a figure read from a report |

## Raw (`raw/`)

| File | Created by | Content |
| --- | --- | --- |
| `ibbi/pages/page_NNNNN.html` | `ibbi-scrape` | cached listing pages |
| `ibbi/public_announcements.csv` | `ibbi-scrape`, `ibbi-filter` | `pa_type, announcement_date, last_submission_date, corporate_debtor, applicant, insolvency_professional, pa_pdf_url, remarks, source_page` |
| `listed/bse_scrips.csv`, `listed/nse_equity_list.csv` | `listed-fetch` | exchange lists as downloaded |
| `annual_reports/<firm_id>/FY<year>.pdf` | `reports-download` or by hand | the reports |
| `annual_reports/_listings/<src>_<code>.json` | `reports-list` | cached exchange responses |

## Interim (`interim/`)

| File | Key columns |
| --- | --- |
| `ibbi_cirp_filtered.csv` | `name_norm, corporate_debtor, cirp_announcement_date, n_announcements, applicant, pa_pdf_url` |
| `listed_universe.csv` | `firm_id, company_name, name_norm, bse_code, nse_symbol, isin, status, industry, sources` |
| `ibbi_listed_matches.csv` | `corporate_debtor, cirp_announcement_date, pa_pdf_url, match_status, score, firm_id, matched_name, bse_code, nse_symbol, isin, listing_status, industry, other_candidates, accept, petition_date, admission_date_verified, business_group, notes` |
| `report_listings.csv` | `firm_id, source, fy, url, pub_date` |
| `documents.csv` | `doc_id, firm_id, fy, source, url, local_path, pub_date, sha256, n_bytes, status, note` |
| `pages/<doc_id>.json` | `n_pages, n_text_pages, n_ocr_pages, n_needs_ocr_pages, n_empty_pages, n_chars, pages[{page, method, text}]` |
| `sections/<doc_id>.json` | `sections{mdna, directors_report, auditor_report, caro_annexure, basis_for_modified_opinion, going_concern, emphasis_of_matter}` each `{heading, start_page, end_page, n_chars, text}` (or null); `audit_opinion`; `leakage{ibc_generic_mentions, cirp_specific_mentions}`; `runs`; `warnings` |
| `extraction_report.csv` | `<section>_chars` per section, `n_ocr_pages`, `audit_opinion`, `auditor_scope`, leakage counts, `warnings` |
| `qa/section_qa_sheet.csv` | predicted pages/heading/snippets + `found_correct, start_correct, end_correct, true_start_page, true_end_page, notes` |
| `qa/section_qa_scores.csv` | per section: `n_checked, found_accuracy, start_accuracy, end_accuracy, exact_span_accuracy` |
| `qa/financials_spot_check.csv` | a random 10% of company-years x every extracted field: `firm_id, fy, field, value_cr, source, source_doc_id, page, statement, label, printed, unit, confidence, parse_flags` + `value_correct, page_correct, true_value_cr, notes` to fill in |
| `qa/financials_spot_check_scores.csv` | per field: `n_checked, value_accuracy, page_accuracy` |

## Processed (`processed/`)

### `cohort.csv` — one row per firm
`pair_id, firm_id, company_name, role (distressed|healthy), label (1|0), reference_date (admission date of the pair's distressed firm), admission_date_source (verified|ibbi_pa_date), petition_date, reference_fy, industry_code, total_assets_ref_fy, asset_ratio_to_distressed, match_quality (exact_industry|industry_prefix_2), business_group`

### `sample_frame.csv` — one row per firm × FY to collect
`pair_id, firm_id, company_name, role, label, reference_date, fy, fy_end_date, fy_rank_before_reference (1 = latest FY before admission), expected_pub_date`

### `documents_labeled.csv` — the modelling table
| Column | Meaning |
| --- | --- |
| `doc_id, pair_id, firm_id, role, label, fy` | identifiers; `label` 1 = distressed firm |
| `pub_date, pub_date_source` | filing date, or FY end + 183 days if `assumed` |
| `months_before_reference, days_before_reference` | time from publication to admission |
| `filing_delay_days` | publication − FY end (late filing is a distress signal) |
| `exclude_reason` | blank if included; else `missing_document`, `published_within_180d_of_admission`, `published_after_petition`, `pair_partner_excluded`, `beyond_n_years_before` |
| `included` | True for rows used in modelling |
| `horizon` | `t-1`, `t-2`, `t-3` (rank of the FY within the pair, newest first) |
| `within_12m, within_24m` | distressed and published ≤12 / ≤24 months before admission |
| `has_<section>, audit_opinion, cirp_specific_mentions, ibc_generic_mentions` | from section extraction |
| `needs_leakage_review` | distressed, included, and mentions CIRP terms → read it |

### `financials_figures.csv` — one row per extracted figure (Phase 3 audit trail)
`firm_id, fy, field, value_cr, source (report_current_year | next_report_comparative | manual_xbrl),
source_doc_id, page, statement, statement_scope, label, match_how, match_score, unit,
unit_confidence, printed, parse_confidence, parse_flags, ocr_pages, restated, restatement_diff,
has_comparative, validation_flags, confidence`

`value_cr` is ₹ crore; `printed` is the figure as the report showed it, in `unit`. `restated` means
the next year's comparative disagreed with the figure as first published — the first-published one
is kept either way.

### `financials_extracted.csv` — one row per company-year
The standard fields (`total_assets, current_assets, current_liabilities, inventories,
cash_and_equivalents, non_current_assets, non_current_liabilities, borrowings_long_term,
borrowings_short_term, total_equity, equity_share_capital, other_equity, retained_earnings,
total_liabilities, total_equity_and_liabilities, revenue, total_income, pbt, finance_costs,
depreciation, net_profit`),
all in ₹ crore, plus:

| Column | Meaning |
| --- | --- |
| `pair_id, role, label, horizon, company_name` | carried from `documents_labeled.csv` |
| `balance_check, components_check` | relative gap in the two balance-sheet identities (0 = exact) |
| `scale_vs_cohort, scale_vs_previous_year` | total assets against the two unit-scale anchors |
| `validation_flags, n_validation_flags` | see `docs/06_phase3_financials.md` |
| `missing_fields, n_fields_found, has_core_financials` | what was and was not recovered |
| `financials_source` | `report_current_year` / `next_report_comparative` / `manual_xbrl` / `missing` |
| `financials_missing` | True when a required field is absent — the row is still kept |
| `financials_exclude_reason` | blank, `missing_financials`, or `pair_partner_missing_financials` |
| `included_financials` | True for rows usable in modelling |

### `ratios.csv` — the stream-C table
`firm_id, fy` + the 12 ratios (`current_ratio, quick_ratio, cash_to_assets, ebitda_margin, roce,
roa, debt_to_equity, debt_to_ebitda, interest_coverage, retained_earnings_to_assets,
promoter_pledge, altman_z_em`), plus `altman_z_dprime` (the same score without Altman's +3.25
rating-equivalent constant) and `altman_zone (safe|grey|distress)`, which is read off
`altman_z_dprime`. Then the condition indicators (`borrowings_partial, negative_equity,
negative_ebitda, zero_finance_costs, negative_working_capital, negative_capital_employed`),
`ratio_reasons`, and the identifiers carried from `financials_extracted.csv`.

`promoter_pledge` is empty until the exchange shareholding filings are collected. An empty ratio
means it could not be computed honestly, not zero — `ratio_reasons` says why. Ratios are raw:
winsorise and scale inside the CV folds.

### `language_features.csv` — the stream-B table (Phase 4)
One row per report. Per section, for `mdna_` and `auditor_` separately: `lm_negative_ratio`,
`lm_positive_ratio`, `lm_uncertainty_ratio`, `lm_litigious_ratio`, `lm_strong_modal_ratio`,
`lm_weak_modal_ratio`, `lm_constraining_ratio`, `hedge_density`, `hedge_distinct`,
`hedge_phrase_density`, `distress_phrase_density`, `distress_phrase_distinct`, `fog_index`,
`avg_sentence_length`, `complex_word_ratio`, `n_words`, `n_sentences`.

| Column | Meaning |
| --- | --- |
| `doc_id, firm_id, fy` + `pair_id, role, label, horizon, included` | identifiers, carried from `documents_labeled.csv` |
| `has_going_concern, has_emphasis_of_matter, has_modified_opinion_basis` | from Phase 2's auditor sub-sections |
| `audit_opinion_severity` | 0 unmodified, 1 qualified, 2 adverse, 3 disclaimer; empty when the opinion could not be read |
| `audit_opinion_found` | whether the opinion was identified at all |
| `caro_default_flag, caro_statutory_dues_flag` | CARO clauses ix and vii, **negation-aware** — a clean annexure denies default, so a naive match is 1 for every company |
| `has_mdna, has_auditor_report, has_caro_annexure` | section coverage |
| `ibc_generic_mentions, cirp_specific_mentions` | leakage counts from Phase 2 |
| `perplexity` | under the healthy-firm reference model; empty unless a training set was named |
| `perplexity_source` | `fold_training_set` or `not_fitted_no_training_set_given` |
| `perplexity_in_training` | 1 when the document was itself used to fit the model, so its score is optimistic |
| `drift_jaccard, drift_cosine, drift_new_word_share, drift_length_ratio` | MD&A against the **same firm's** previous year |
| `drift_negative_delta, drift_hedge_delta, drift_fog_delta` | change in tone, hedging and readability |
| `has_previous_year, drift_previous_doc_id` | whether a prior report existed, and which one was used |

Tone columns are **empty, not zero**, when no dictionary was supplied. Drift is empty for a firm's
first year. `CORE_LANGUAGE_FEATURES` in `nlp/features.py` names the 25 values the model is trained
on; the rest are kept for the Phase 7 ablations.

### `financials_missing.csv`
the `financials_extracted.csv` rows where `financials_missing` is True

### `missing_reports.csv`
`pair_id, firm_id, company_name, role, fy, expected_pub_date`

### `cohort_unmatched.csv`
the distressed firm's match row + `reason`
