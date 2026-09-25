"""Configuration loading and project paths.

All commands call :func:`load_config` and :func:`get_paths`, so every file
location in the project is defined in exactly one place (here).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config. Falls back to configs/config.yaml."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(cfg_path)
    return cfg


@dataclass(frozen=True)
class Paths:
    """Every folder and file the pipeline reads or writes."""

    data: Path

    # ---- raw inputs (downloaded) ----
    @property
    def raw_ibbi(self) -> Path:
        return self.data / "raw" / "ibbi"

    @property
    def ibbi_pages(self) -> Path:
        return self.raw_ibbi / "pages"

    @property
    def ibbi_announcements(self) -> Path:
        return self.raw_ibbi / "public_announcements.csv"

    @property
    def raw_listed(self) -> Path:
        return self.data / "raw" / "listed"

    @property
    def raw_reports(self) -> Path:
        return self.data / "raw" / "annual_reports"

    @property
    def report_listings(self) -> Path:
        return self.raw_reports / "_listings"

    # ---- manual inputs (filled in by the team) ----
    @property
    def manual(self) -> Path:
        return self.data / "manual"

    @property
    def financials(self) -> Path:
        return self.manual / "firm_financials.csv"

    @property
    def xbrl_financials(self) -> Path:
        """Phase 3 gap-fill: figures taken by hand from NSE/BSE XBRL annual results."""
        return self.manual / "xbrl_financials.csv"

    @property
    def xbrl_exchange(self) -> Path:
        """Figures parsed from the exchanges' XBRL annual results (bpp xbrl-financials)."""
        return self.interim / "xbrl_financials_exchange.csv"

    @property
    def lm_dictionary(self) -> Path:
        """Phase 4: the Loughran-McDonald dictionary, downloaded once by the team.

        Not vendored: it is republished annually, and a silent copy in the repo
        would make a tone result impossible to reproduce. See
        ``docs/07_phase4_language.md``.
        """
        return self.manual / "loughran_mcdonald.csv"

    @property
    def leakage_review(self) -> Path:
        """Reports read for leakage: doc_id, decision (exclude|keep), reason."""
        return self.manual / "leakage_review.csv"

    @property
    def reviewed_matches(self) -> Path:
        return self.manual / "ibbi_listed_matches_reviewed.csv"

    @property
    def manual_reports(self) -> Path:
        return self.manual / "manual_reports.csv"

    @property
    def extra_listed(self) -> Path:
        return self.manual / "listed_companies_extra.csv"

    # ---- interim (generated, re-creatable) ----
    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def cirp_listed_candidates(self) -> Path:
        return self.interim / "ibbi_cirp_filtered.csv"

    @property
    def listed_universe(self) -> Path:
        return self.interim / "listed_universe.csv"

    @property
    def name_matches(self) -> Path:
        return self.interim / "ibbi_listed_matches.csv"

    @property
    def report_listing_table(self) -> Path:
        return self.interim / "report_listings.csv"

    @property
    def documents(self) -> Path:
        return self.interim / "documents.csv"

    @property
    def pages(self) -> Path:
        return self.interim / "pages"

    @property
    def sections(self) -> Path:
        return self.interim / "sections"

    @property
    def qa(self) -> Path:
        return self.interim / "qa"

    @property
    def qa_sheet(self) -> Path:
        return self.qa / "section_qa_sheet.csv"

    @property
    def financials_qa_sheet(self) -> Path:
        return self.qa / "financials_spot_check.csv"

    @property
    def extraction_report(self) -> Path:
        return self.interim / "extraction_report.csv"

    # ---- processed (final tables used for modelling) ----
    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def cohort(self) -> Path:
        return self.processed / "cohort.csv"

    @property
    def sample_frame(self) -> Path:
        return self.processed / "sample_frame.csv"

    @property
    def documents_labeled(self) -> Path:
        return self.processed / "documents_labeled.csv"

    @property
    def missing_reports(self) -> Path:
        return self.processed / "missing_reports.csv"

    # ---- Phase 3: financial statements and ratios ----
    # Note: ``financials`` above is the team's manual input (total assets for
    # peer matching). These are what Phase 3 produces, and never overwrite it.
    @property
    def financials_figures(self) -> Path:
        """Audit trail: one row per figure, with page, label and confidence."""
        return self.processed / "financials_figures.csv"

    @property
    def financials_extracted(self) -> Path:
        """One row per company-year, one column per standard field."""
        return self.processed / "financials_extracted.csv"

    @property
    def ratios(self) -> Path:
        """The stream-C table: 12 ratios per company-year."""
        return self.processed / "ratios.csv"

    @property
    def financials_missing(self) -> Path:
        return self.processed / "financials_missing.csv"

    # ---- Phase 4: language features (stream B) ----
    @property
    def language_features(self) -> Path:
        """One row per report: tone, hedging, readability, flags, drift, perplexity."""
        return self.processed / "language_features.csv"

    def ensure(self) -> "Paths":
        """Create all folders (safe to call repeatedly)."""
        for p in [
            self.ibbi_pages, self.raw_listed, self.report_listings, self.manual,
            self.pages, self.sections, self.qa, self.processed,
        ]:
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(cfg: dict[str, Any], data_dir: str | Path | None = None) -> Paths:
    """Resolve the data folder: --data-dir > BPP_DATA_DIR env var > config."""
    chosen = data_dir or os.environ.get("BPP_DATA_DIR") or cfg["paths"]["data_dir"]
    p = Path(chosen)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return Paths(data=p).ensure()
