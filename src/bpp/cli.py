"""The ``bpp`` command. Run ``bpp --help`` or ``bpp <command> --help``.

Phase 1 - cohort
    ibbi-scrape        download IBBI public announcement pages (cached)
    ibbi-filter        keep CIRP announcements for possibly-listed debtors
    listed-fetch       BSE + NSE company lists (active, suspended, delisted)
    match-names        fuzzy-match IBBI debtors to listed companies  -> human review
    build-cohort       healthy peer matching + firm-year sample frame

Phase 2 - documents
    reports-list       ask NSE/BSE which annual reports exist
    reports-download   download missing PDFs
    reports-register   register PDFs you downloaded by hand
    extract-text       PDF -> page text (OCR for scanned pages)
    extract-sections   page text -> MD&A, Board's report, auditor's report, CARO
    assign-labels      publication dates, leakage exclusions, horizons, labels
    qa-sample          sheet for hand-checking section extraction
    qa-score           accuracy from the filled sheet

Phase 3 - financial features
    financials         statements -> standard fields -> validation -> 12 ratios
    financials-score   accuracy from the filled spot-check sheet

Phase 4 - language features (stream B; FinBERT/stream A runs on Colab)
    language-features  tone, hedging, readability, auditor flags, drift, perplexity

Other
    init               create data folders and manual-input templates
    demo               run Phases 1-2 end to end on synthetic data (offline)
    status             what exists so far and what to do next
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

from bpp.common import setup_logging
from bpp.config import PROJECT_ROOT, Paths, get_paths, load_config

log = logging.getLogger("bpp.cli")


# ----------------------------------------------------------------------------- commands
def cmd_init(cfg, paths: Paths, args) -> None:
    templates = {
        "firm_financials_TEMPLATE.csv":
            "firm_id,company_name,fy,industry_code,total_assets,business_group\n"
            "BSE500000,Example Industries Limited,2019,24103,1523.4,Example Group\n",
        "ibbi_listed_matches_reviewed_TEMPLATE.csv":
            "corporate_debtor,cirp_announcement_date,firm_id,matched_name,match_status,score,accept,"
            "petition_date,admission_date_verified,business_group,notes\n",
        "manual_reports_TEMPLATE.csv":
            "firm_id,fy,local_path,pub_date,source_url,note\n"
            "BSE500000,2019,data/raw/annual_reports/BSE500000/FY2019.pdf,2019-08-30,https://...,from company website\n",
        "listed_companies_extra_TEMPLATE.csv":
            "company_name,bse_code,nse_symbol,isin,status,industry\n",
        "loughran_mcdonald_TEMPLATE.csv":
            "word,category\n"
            "# replace this file with the real Master Dictionary from sraf.nd.edu\n"
            "loss,negative\nmay,weak_modal\nuncertain,uncertainty\n",
        "xbrl_financials_TEMPLATE.csv":
            "firm_id,fy,field,value_cr,source_url,note\n"
            "BSE500000,2019,total_assets,1523.4,https://www.nseindia.com/...,XBRL annual results\n",
    }
    for name, content in templates.items():
        f = paths.manual / name
        if not f.exists():
            f.write_text(content, encoding="utf-8")
    log.info("data folders ready under %s; templates in %s", paths.data, paths.manual)


def cmd_ibbi_scrape(cfg, paths, args) -> None:
    from bpp.scrape.ibbi import scrape_public_announcements
    scrape_public_announcements(cfg, paths, max_pages=args.max_pages, resume=not args.no_resume,
                                start_page=args.start_page)


def cmd_ibbi_filter(cfg, paths, args) -> None:
    from bpp.scrape.ibbi import combine_cached_pages, filter_cirp_announcements
    df = combine_cached_pages(paths)
    filter_cirp_announcements(cfg, paths, df)


def cmd_listed_fetch(cfg, paths, args) -> None:
    from bpp.scrape.listed import fetch_listed_universe
    fetch_listed_universe(cfg, paths, args.sources)


def cmd_match_names(cfg, paths, args) -> None:
    from bpp.cohort.name_match import run_name_matching
    run_name_matching(cfg, paths)


def cmd_build_cohort(cfg, paths, args) -> None:
    from bpp.cohort.peers import run_build_cohort
    run_build_cohort(cfg, paths, use_auto=args.use_auto_matches)


def _frame_and_universe(paths: Paths):
    frame = pd.read_csv(paths.sample_frame)
    universe = pd.read_csv(paths.listed_universe, dtype={"bse_code": "string", "nse_symbol": "string"})
    return frame, universe


def cmd_reports_list(cfg, paths, args) -> None:
    from bpp.scrape.annual_reports import list_available_reports
    frame, universe = _frame_and_universe(paths)
    list_available_reports(cfg, paths, frame, universe, args.source or cfg["reports"]["sources"], args.firm_id)


def cmd_reports_download(cfg, paths, args) -> None:
    from bpp.scrape.annual_reports import download_reports
    frame, _ = _frame_and_universe(paths)
    listing = pd.read_csv(paths.report_listing_table)
    download_reports(cfg, paths, frame, listing, args.source or cfg["reports"]["sources"], args.limit)


def cmd_reports_register(cfg, paths, args) -> None:
    from bpp.scrape.annual_reports import register_manual_reports
    frame = pd.read_csv(paths.sample_frame) if paths.sample_frame.exists() else None
    register_manual_reports(paths, frame)


def cmd_extract_text(cfg, paths, args) -> None:
    from bpp.extract.pdf_text import run_extract_text
    run_extract_text(cfg, paths, args.doc_id, args.force)


def cmd_extract_sections(cfg, paths, args) -> None:
    from bpp.extract.sections import run_extract_sections
    run_extract_sections(cfg, paths, args.doc_id, args.force)


def cmd_assign_labels(cfg, paths, args) -> None:
    from bpp.cohort.labels import run_assign_labels
    run_assign_labels(cfg, paths)


def cmd_qa_sample(cfg, paths, args) -> None:
    from bpp.extract.qa import make_qa_sheet
    make_qa_sheet(cfg, paths, args.n, args.overwrite)


def cmd_qa_score(cfg, paths, args) -> None:
    from bpp.extract.qa import score_qa_sheet
    print(score_qa_sheet(paths).to_string(index=False))


def cmd_financials(cfg, paths, args) -> None:
    from bpp.features.financials import run_financials
    run_financials(cfg, paths, args.doc_id, spot_check=not args.no_spot_check,
                   overwrite_spot_check=args.overwrite_spot_check)


def cmd_financials_score(cfg, paths, args) -> None:
    from bpp.features.financials import score_spot_check
    print(score_spot_check(paths).to_string(index=False))


def cmd_language_features(cfg, paths, args) -> None:
    from bpp.nlp.features import run_language_features
    run_language_features(cfg, paths, args.doc_id, args.lm_train_doc_id)


def cmd_status(cfg, paths, args) -> None:
    steps = [
        ("IBBI announcements", paths.ibbi_announcements, "bpp ibbi-scrape"),
        ("CIRP debtors (non-private)", paths.cirp_listed_candidates, "bpp ibbi-filter"),
        ("Listed universe", paths.listed_universe, "bpp listed-fetch"),
        ("Name matches (auto)", paths.name_matches, "bpp match-names"),
        ("Name matches (REVIEWED by team)", paths.reviewed_matches, "review data/interim/ibbi_listed_matches.csv"),
        ("Firm financials (manual)", paths.financials, "fill data/manual/firm_financials.csv"),
        ("Cohort", paths.cohort, "bpp build-cohort"),
        ("Sample frame", paths.sample_frame, "bpp build-cohort"),
        ("Report listings", paths.report_listing_table, "bpp reports-list"),
        ("Documents manifest", paths.documents, "bpp reports-download / reports-register"),
        ("Extraction report", paths.extraction_report, "bpp extract-text && bpp extract-sections"),
        ("Labelled documents", paths.documents_labeled, "bpp assign-labels"),
        ("QA sheet", paths.qa_sheet, "bpp qa-sample"),
        ("Financials (extracted)", paths.financials_extracted, "bpp financials"),
        ("Ratios (stream C)", paths.ratios, "bpp financials"),
        ("Financials spot-check sheet", paths.financials_qa_sheet, "bpp financials"),
        ("Loughran-McDonald dictionary", paths.lm_dictionary, "download to data/manual/ (docs/07_phase4_language.md)"),
        ("Language features (stream B)", paths.language_features, "bpp language-features"),
    ]
    print(f"data folder: {paths.data}\n")
    next_step = None
    for name, f, how in steps:
        if f.exists():
            try:
                n = len(pd.read_csv(f))
                info = f"{n} rows"
            except Exception:  # noqa: BLE001
                info = "exists"
            print(f"  [x] {name:<34} {info}")
        else:
            print(f"  [ ] {name:<34} -> {how}")
            next_step = next_step or how
    print(f"\n  pages extracted: {len(list(paths.pages.glob('*.json')))}, "
          f"sections extracted: {len(list(paths.sections.glob('*.json')))}")
    if next_step:
        print(f"\nNext: {next_step}")


def cmd_demo(cfg, paths, args) -> None:
    """Run every Phase 1-2 step on synthetic data in a separate folder."""
    from bpp.cohort.labels import run_assign_labels
    from bpp.cohort.name_match import run_name_matching
    from bpp.cohort.peers import run_build_cohort
    from bpp.extract.pdf_text import run_extract_text
    from bpp.extract.qa import make_qa_sheet
    from bpp.extract.sections import run_extract_sections
    from bpp.scrape.annual_reports import register_manual_reports
    from bpp.scrape.ibbi import combine_cached_pages, filter_cirp_announcements
    from bpp.synthetic import make_synthetic_inputs, make_synthetic_reports

    demo_dir = Path(args.out) if args.out else PROJECT_ROOT / "data_demo"
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    dp = get_paths(cfg, demo_dir)
    log.info("=== DEMO on synthetic data in %s ===", demo_dir)
    make_synthetic_inputs(dp)
    log.info("--- Phase 1.1 IBBI announcements -> CIRP debtors")
    filter_cirp_announcements(cfg, dp, combine_cached_pages(dp))
    log.info("--- Phase 1.2 match debtor names to listed companies")
    run_name_matching(cfg, dp)
    log.info("--- Phase 1.3 healthy peers + sample frame (auto matches, no human review in the demo)")
    run_build_cohort(cfg, dp, use_auto=True)
    log.info("--- Phase 2.1 collect reports (synthetic PDFs stand in for NSE/BSE downloads)")
    make_synthetic_reports(dp, cfg)
    register_manual_reports(dp, pd.read_csv(dp.sample_frame))
    log.info("--- Phase 2.2 text extraction (+OCR)")
    run_extract_text(cfg, dp)
    log.info("--- Phase 2.3 section extraction")
    run_extract_sections(cfg, dp)
    log.info("--- Phase 2.4 labels, horizons, leakage exclusions")
    run_assign_labels(cfg, dp)
    log.info("--- Phase 2.5 QA sheet")
    make_qa_sheet(cfg, dp, n=4, overwrite=True)
    log.info("=== DEMO finished. Open %s to look at every output. ===", demo_dir)


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bpp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="path to a config YAML (default configs/config.yaml)")
    p.add_argument("--data-dir", help="override the data folder (e.g. a Google Drive path on Colab)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(func=fn)
        return sp

    add("init", cmd_init, "create data folders and manual templates")
    sp = add("ibbi-scrape", cmd_ibbi_scrape, "download IBBI public announcement pages")
    sp.add_argument("--max-pages", type=int)
    sp.add_argument("--start-page", type=int, default=0)
    sp.add_argument("--no-resume", action="store_true", help="re-download pages even if cached")
    add("ibbi-filter", cmd_ibbi_filter, "combine cached pages and keep CIRP, non-private debtors")
    sp = add("listed-fetch", cmd_listed_fetch, "download BSE/NSE company lists")
    sp.add_argument("--sources", nargs="+", default=["bse", "nse"], choices=["bse", "nse"])
    add("match-names", cmd_match_names, "fuzzy match IBBI debtors to listed companies")
    sp = add("build-cohort", cmd_build_cohort, "match healthy peers and build the firm-year frame")
    sp.add_argument("--use-auto-matches", action="store_true", help="skip human review (quick test only)")
    sp = add("reports-list", cmd_reports_list, "list annual reports available on NSE/BSE")
    sp.add_argument("--source", nargs="+", choices=["nse", "bse"])
    sp.add_argument("--firm-id", nargs="+")
    sp = add("reports-download", cmd_reports_download, "download annual report PDFs")
    sp.add_argument("--source", nargs="+", choices=["nse", "bse"])
    sp.add_argument("--limit", type=int)
    add("reports-register", cmd_reports_register, "register PDFs saved by hand")
    for name, fn, h in [("extract-text", cmd_extract_text, "PDF -> page text with OCR fallback"),
                        ("extract-sections", cmd_extract_sections, "page text -> sections")]:
        sp = add(name, fn, h)
        sp.add_argument("--doc-id", nargs="+")
        sp.add_argument("--force", action="store_true", help="redo documents already processed")
    add("assign-labels", cmd_assign_labels, "labels, horizons and leakage exclusions")
    sp = add("qa-sample", cmd_qa_sample, "create the hand-check sheet")
    sp.add_argument("--n", type=int)
    sp.add_argument("--overwrite", action="store_true")
    add("qa-score", cmd_qa_score, "score the filled hand-check sheet")
    sp = add("financials", cmd_financials,
             "statements -> standard fields -> validation -> the 12 stream-C ratios")
    sp.add_argument("--doc-id", nargs="+", help="only these documents")
    sp.add_argument("--no-spot-check", action="store_true", help="skip the hand-check sheet")
    sp.add_argument("--overwrite-spot-check", action="store_true",
                    help="replace the spot-check sheet even if it has hand checks in it")
    add("financials-score", cmd_financials_score, "score the filled financials spot-check sheet")
    sp = add("language-features", cmd_language_features,
             "tone, hedging, readability, auditor flags, drift and perplexity")
    sp.add_argument("--doc-id", nargs="+", help="only these documents")
    sp.add_argument("--lm-train-doc-id", nargs="+",
                    help="documents to fit the healthy-firm reference model on "
                         "(training folds only; without this, perplexity is left empty)")
    add("status", cmd_status, "show progress and the next step")
    sp = add("demo", cmd_demo, "run Phases 1-2 on synthetic data")
    sp.add_argument("--out", help="folder for demo data (default data_demo/)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    cfg = load_config(args.config)
    paths = get_paths(cfg, args.data_dir)
    try:
        args.func(cfg, paths, args)
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
