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

### `missing_reports.csv`
`pair_id, firm_id, company_name, role, fy, expected_pub_date`

### `cohort_unmatched.csv`
the distressed firm's match row + `reason`
