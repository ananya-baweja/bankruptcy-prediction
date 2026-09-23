"""Phase 3 - build the financial table and the stream-C ratios from annual reports.

CMIE Prowess is not available to this project, so every figure comes out of the
annual-report PDFs Phase 2 already collected. MCA21 is not used.

What this does, per company-year
--------------------------------
1. Reads Phase 2's ``interim/pages/<doc_id>.json`` and finds the **standalone**
   balance sheet, statement of profit and loss and cash flow statement.
2. Reads the figures off them -- both the current-year and the prior-year
   column -- and maps the line items to the standard fields.
3. Stores each figure **as first published**: the number printed in that year's
   own report. Next year's comparative is never allowed to overwrite it; it is
   used to cross-check (and raise a restatement flag when the two disagree) or
   to fill the year in when its own report is missing.
4. Validates: the balance sheet must balance, sub-totals must contain their
   parts, the unit scale is checked against the year before and against the
   cohort's own ``total_assets``, and every field carries a confidence score.
5. Computes the 12 ratios of Table 7, including the emerging-market Altman Z''.

Gap-filling order, when a company-year has no usable figures
------------------------------------------------------------
1. the next year's report, whose prior-year column covers this year
2. ``data/manual/xbrl_financials.csv`` -- figures taken from the NSE/BSE XBRL
   annual results (see ``docs/06_phase3_financials.md`` for how to fill it)
3. mark the company-year missing and apply the pair-exclusion rule

A missing company-year is a feature, not a row to delete: late and absent
filing is itself a distress signal (Mancisidor & Aas, and issue 7 of the
project plan). Rows are kept and flagged, never dropped. The pair-exclusion
rule then mirrors Phase 2's ``keep_pairs_aligned``: if a distressed firm's year
drops out of modelling, its healthy peer's same year drops out too, so the
model cannot learn the calendar year as a shortcut.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from bpp.common import read_csv, write_csv
from bpp.config import Paths
from bpp.features.numbers import TO_CRORE
from bpp.features.ratios import (INDICATOR_FIELDS, RATIO_FIELDS, compute_ratios)
from bpp.features.statements import (FIELD_STATEMENT, STANDARD_FIELDS, FieldHit,
                                     best_hits, extract_table_rows, locate_statements,
                                     map_statement, parse_statement_lines, rows_to_line_items)

log = logging.getLogger(__name__)

#: fields a company-year needs before any ratio is worth computing
CORE_FIELDS = ["total_assets", "current_assets", "current_liabilities", "total_equity"]

#: where a figure ended up coming from, best first
SOURCE_PRIMARY = "report_current_year"
SOURCE_COMPARATIVE = "next_report_comparative"
SOURCE_XBRL = "manual_xbrl"
SOURCE_NONE = "missing"


# --------------------------------------------------------------------------- per document
def extract_document(pages_json: dict[str, Any], cfg: dict[str, Any],
                     pdf_path: str | None = None) -> tuple[list[FieldHit], dict[str, Any]]:
    """Pull every figure out of one report. Returns (hits, per-document report)."""
    fcfg = cfg["financials"]
    pages = pages_json["pages"]
    report_fy = int(pages_json.get("fy") or 0) or None

    locations = locate_statements(pages, cfg, report_fy)
    hits: list[FieldHit] = []
    found: dict[str, Any] = {}

    for name, loc in locations.items():
        items = parse_statement_lines(pages, loc)

        # Ruled tables give real column geometry; use them when they add rows.
        if fcfg["use_table_extraction"] and pdf_path:
            rows = extract_table_rows(pdf_path, range(loc.start_page, loc.end_page + 1),
                                      flavours=fcfg["table_flavours"])
            table_items = rows_to_line_items(rows, loc)
            if len(table_items) > len(items):
                items = table_items + items

        hits.extend(map_statement(items, loc))
        found[name] = {
            "scope": loc.scope, "start_page": loc.start_page, "end_page": loc.end_page,
            "unit": loc.unit, "unit_confidence": loc.unit_confidence,
            "unit_source": loc.unit_source, "period_fys": loc.period_fys,
            "period_confidence": loc.period_confidence, "ocr_pages": loc.ocr_pages,
            "flags": loc.flags, "heading": loc.heading,
        }

    report = {
        "doc_id": pages_json.get("doc_id"),
        "firm_id": pages_json.get("firm_id"),
        "fy": report_fy,
        "statements": found,
        "missing_statements": [s for s in ("balance_sheet", "profit_and_loss") if s not in found],
        "n_hits": len(hits),
    }
    return hits, report


# --------------------------------------------------------------------------- assembling years
def _hit_row(hit: FieldHit, doc_id: str, report_fy: int | None, statements: dict) -> dict[str, Any]:
    info = statements.get(hit.statement, {})
    # What makes a figure first-published is that it appears in the report for
    # its own fiscal year -- not which column it sat in. A report whose columns
    # run oldest-first would otherwise have its own year filed as a comparative.
    is_primary = report_fy is not None and hit.fy == report_fy
    return {
        "firm_id": doc_id.rsplit("_FY", 1)[0] if "_FY" in doc_id else "",
        "fy": hit.fy,
        "field": hit.field,
        "value_cr": hit.value,
        "source": SOURCE_PRIMARY if is_primary else SOURCE_COMPARATIVE,
        "source_doc_id": doc_id,
        "source_report_fy": report_fy,
        "column": hit.column,
        "page": hit.page,
        "statement": hit.statement,
        "statement_scope": info.get("scope"),
        "label": hit.label,
        "match_how": hit.match_how,
        "match_score": hit.match_score,
        "unit": hit.unit,
        "unit_confidence": info.get("unit_confidence", 0.2),
        "printed": hit.printed,
        "parse_confidence": hit.parse_confidence,
        "parse_flags": ";".join(hit.parse_flags),
        "ocr_pages": info.get("ocr_pages", 0),
    }


def collect_figures(paths: Paths, cfg: dict[str, Any],
                    doc_ids: list[str] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Read every available report and return one row per (firm-year, field, source)."""
    files = sorted(paths.pages.glob("*.json"))
    if doc_ids:
        wanted = set(doc_ids)
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise FileNotFoundError(
            "No page files found. Run bpp extract-text first (see docs/03_phase2_documents.md).")

    local_paths: dict[str, str] = {}
    if paths.documents.exists():
        manifest = pd.read_csv(paths.documents)
        if {"doc_id", "local_path"} <= set(manifest.columns):
            local_paths = dict(zip(manifest["doc_id"], manifest["local_path"].fillna("")))

    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for path in tqdm(files, desc="read statements"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pdf = local_paths.get(payload.get("doc_id", ""), "") or payload.get("source_pdf", "")
        try:
            hits, report = extract_document(payload, cfg, pdf or None)
        except Exception as exc:                                   # noqa: BLE001
            log.error("could not read statements in %s: %s", path.stem, exc)
            reports.append({"doc_id": path.stem, "error": str(exc)})
            continue
        chosen = best_hits(hits)
        for hit in chosen.values():
            rows.append(_hit_row(hit, payload["doc_id"], report["fy"], report["statements"]))
        reports.append(report)

    figures = pd.DataFrame(rows)
    if len(figures):
        figures["firm_id"] = figures["source_doc_id"].str.rsplit("_FY", n=1).str[0]
    return figures, reports


def load_xbrl_supplement(paths: Paths) -> pd.DataFrame:
    """Optional gap-fill source: figures keyed by firm_id, fy, field, value_cr.

    NSE and BSE publish XBRL annual results, but their endpoints block cloud
    addresses and the BSE one in this project is still unverified (see
    ``configs/config.yaml``). Rather than ship a scraper nobody has been able to
    run, step 2 of the gap-filling order reads a CSV the team fills from those
    downloads, exactly as ``manual_reports.csv`` works in Phase 2.
    """
    if not paths.xbrl_financials.exists():
        return pd.DataFrame(columns=["firm_id", "fy", "field", "value_cr"])
    df = read_csv(paths.xbrl_financials, required=["firm_id", "fy", "field", "value_cr"])
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
    df["value_cr"] = pd.to_numeric(df["value_cr"], errors="coerce")
    return df.dropna(subset=["fy", "value_cr"])


def resolve_figures(figures: pd.DataFrame, xbrl: pd.DataFrame,
                    cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the as-first-published rule and the gap-filling order.

    Returns ``(resolved, audit)``: one row per (firm_id, fy, field) with the
    chosen value, and the full audit trail with every candidate kept.
    """
    tol = cfg["financials"]["restatement_tolerance"]
    audit = figures.copy()
    resolved_rows: list[dict[str, Any]] = []

    grouped = ([] if figures.empty
               else figures.groupby(["firm_id", "fy", "field"], sort=False))
    for (firm_id, fy, fieldname), group in grouped:
        primary = group[group["source"] == SOURCE_PRIMARY]
        comparative = group[group["source"] == SOURCE_COMPARATIVE]

        restated = False
        difference = np.nan
        if len(primary) and len(comparative):
            first = float(primary.iloc[0]["value_cr"])
            later = float(comparative.iloc[0]["value_cr"])
            scale = max(abs(first), abs(later), 1e-9)
            difference = abs(first - later) / scale
            restated = bool(difference > tol)

        if len(primary):
            chosen, source = primary.iloc[0], SOURCE_PRIMARY
        elif len(comparative):
            chosen, source = comparative.iloc[0], SOURCE_COMPARATIVE
        else:                                                     # pragma: no cover - defensive
            continue

        resolved_rows.append({
            "firm_id": firm_id, "fy": int(fy), "field": fieldname,
            "value_cr": float(chosen["value_cr"]), "source": source,
            "source_doc_id": chosen["source_doc_id"], "page": chosen["page"],
            "statement": chosen["statement"], "statement_scope": chosen["statement_scope"],
            "label": chosen["label"], "match_how": chosen["match_how"],
            "match_score": chosen["match_score"], "unit": chosen["unit"],
            "unit_confidence": chosen["unit_confidence"], "printed": chosen["printed"],
            "parse_confidence": chosen["parse_confidence"], "parse_flags": chosen["parse_flags"],
            "ocr_pages": chosen["ocr_pages"],
            "restated": restated, "restatement_diff": difference,
            "has_comparative": bool(len(comparative)),
        })

    resolved = pd.DataFrame(resolved_rows)

    # Step 2 of the gap-filling order: the manual XBRL supplement. This runs even
    # when no report yielded a figure at all -- an all-scanned batch is exactly
    # when the supplement is the only source there is.
    if len(xbrl):
        have = set(zip(resolved["firm_id"], resolved["fy"], resolved["field"])) if len(resolved) else set()
        extra = [
            {"firm_id": r["firm_id"], "fy": int(r["fy"]), "field": r["field"],
             "value_cr": float(r["value_cr"]), "source": SOURCE_XBRL,
             "source_doc_id": "", "page": np.nan, "statement": "", "statement_scope": "",
             "label": "xbrl", "match_how": "xbrl", "match_score": 1.0, "unit": "crore",
             "unit_confidence": 0.95, "printed": np.nan, "parse_confidence": 1.0,
             "parse_flags": "", "ocr_pages": 0, "restated": False,
             "restatement_diff": np.nan, "has_comparative": False}
            for _, r in xbrl.iterrows()
            if (r["firm_id"], int(r["fy"]), r["field"]) not in have
        ]
        if extra:
            resolved = pd.concat([resolved, pd.DataFrame(extra)], ignore_index=True)
            log.info("gap-filled %d figures from %s", len(extra), "manual/xbrl_financials.csv")

    return resolved, audit


def to_wide(resolved: pd.DataFrame) -> pd.DataFrame:
    """One row per company-year, one column per standard field."""
    if resolved.empty:
        return pd.DataFrame(columns=["firm_id", "fy", *STANDARD_FIELDS])
    wide = resolved.pivot_table(index=["firm_id", "fy"], columns="field",
                                values="value_cr", aggfunc="first").reset_index()
    wide.columns.name = None
    for column in STANDARD_FIELDS:
        if column not in wide.columns:
            wide[column] = np.nan
    return wide


# --------------------------------------------------------------------------- validation
def validate_year(row: pd.Series, cfg: dict[str, Any],
                  reference_assets: float | None = None,
                  previous_assets: float | None = None) -> dict[str, Any]:
    """Run every consistency check on one company-year."""
    fcfg = cfg["financials"]
    tol = fcfg["balance_tolerance"]
    checks: dict[str, Any] = {}
    flags: list[str] = []

    def value(name: str) -> float | None:
        v = row.get(name)
        return None if v is None or pd.isna(v) else float(v)

    total_assets = value("total_assets")
    equity_and_liabilities = value("total_equity_and_liabilities")

    # 1. the balance sheet must balance
    checks["balance_check"] = None
    if total_assets is not None and equity_and_liabilities is not None:
        scale = max(abs(total_assets), abs(equity_and_liabilities), 1e-9)
        gap = abs(total_assets - equity_and_liabilities) / scale
        checks["balance_check"] = round(gap, 6)
        if gap > tol:
            flags.append("balance_sheet_does_not_balance")

    # 2. the same identity from the components, which catches a misread subtotal
    equity, non_current, current = value("total_equity"), value("non_current_liabilities"), value("current_liabilities")
    checks["components_check"] = None
    if None not in (total_assets, equity, non_current, current):
        built = equity + non_current + current
        scale = max(abs(total_assets), abs(built), 1e-9)
        gap = abs(total_assets - built) / scale
        checks["components_check"] = round(gap, 6)
        if gap > tol:
            flags.append("equity_plus_liabilities_mismatch")

    # 3. a subtotal must contain its parts
    current_assets, inventories, cash = value("current_assets"), value("inventories"), value("cash_and_equivalents")
    if current_assets is not None:
        known_parts = [p for p in (inventories, cash) if p is not None]
        # only a comparison against parts we actually have; with none of them
        # the sum is 0 and a negative current-assets figure would "fail" against it
        if known_parts and sum(known_parts) > current_assets * (1 + tol):
            flags.append("current_assets_smaller_than_its_parts")
        if total_assets is not None and current_assets > total_assets * (1 + tol):
            flags.append("current_assets_exceed_total_assets")

    # 4. unit scale. A wrong scale is the error that still looks plausible, so
    #    it is checked against two independent anchors: the cohort's own assets
    #    figure and the previous year of the same firm.
    checks["scale_vs_cohort"] = None
    checks["scale_vs_previous_year"] = None
    for anchor, key, flag in ((reference_assets, "scale_vs_cohort", "unit_scale_differs_from_cohort"),
                              (previous_assets, "scale_vs_previous_year", "unit_scale_jump_vs_previous_year")):
        if total_assets is not None and anchor and anchor > 0:
            ratio = total_assets / anchor
            checks[key] = round(ratio, 4)
            if ratio > 0 and (ratio >= fcfg["scale_jump_factor"] or ratio <= 1 / fcfg["scale_jump_factor"]):
                flags.append(flag)
                if _looks_like_unit_confusion(ratio):
                    flags.append("possible_unit_confusion")

    # 5. figures that cannot be negative
    for name in ("total_assets", "current_assets", "current_liabilities", "inventories",
                 "cash_and_equivalents", "revenue"):
        v = value(name)
        if v is not None and v < 0:
            flags.append(f"negative_{name}")

    checks["validation_flags"] = ";".join(dict.fromkeys(flags))
    checks["n_validation_flags"] = len(set(flags))
    return checks


def field_confidence(row: pd.Series, validation_flags: str) -> float:
    """Combine the things that can go wrong into one score per field.

    Multiplicative, because these are independent ways of being wrong: a
    perfectly parsed number under a misread unit is still wrong, and so is a
    clean number attached to the wrong line item.
    """
    score = float(row.get("parse_confidence", 1.0) or 1.0)
    score *= float(row.get("match_score", 1.0) or 1.0)
    unit_confidence = float(row.get("unit_confidence", 0.2) or 0.2)
    score *= 0.6 + 0.4 * unit_confidence          # an assumed unit caps, never zeroes
    if row.get("source") == SOURCE_COMPARATIVE:
        score *= 0.9                               # a comparative may have been restated
    elif row.get("source") == SOURCE_XBRL:
        score *= 0.95
    if row.get("restated"):
        score *= 0.8
    if row.get("ocr_pages"):
        score *= 0.9
    if validation_flags:
        score *= 0.7
    return round(min(1.0, max(0.0, score)), 4)


#: every ratio one unit scale can be mistaken for another by, e.g. 100 for
#: crore read as lakh. Used to tell a unit error from a genuine jump in size.
_UNIT_RATIOS = sorted({round(a / b, 10) for a in TO_CRORE.values() for b in TO_CRORE.values()
                       if a != b})


def _looks_like_unit_confusion(ratio: float, tolerance: float = 0.15) -> bool:
    """True when a scale discrepancy matches a unit conversion rather than growth.

    A firm can double in a year; it cannot grow exactly hundredfold. A ratio
    sitting on one of the unit conversions is a misread header, not a business.
    """
    return any(abs(ratio - candidate) <= tolerance * candidate for candidate in _UNIT_RATIOS)


# --------------------------------------------------------------------------- the pipeline
def _cohort_context(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The company-years we care about, plus the cohort's own assets anchor."""
    scope = pd.DataFrame()
    if paths.documents_labeled.exists():
        scope = pd.read_csv(paths.documents_labeled)
    elif paths.sample_frame.exists():
        scope = pd.read_csv(paths.sample_frame)

    anchor = pd.DataFrame()
    if paths.financials.exists():
        anchor = pd.read_csv(paths.financials)
    return scope, anchor


def build_financials(paths: Paths, cfg: dict[str, Any],
                     doc_ids: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Run Phase 3 end to end and return every table it produces."""
    figures, reports = collect_figures(paths, cfg, doc_ids)
    resolved, audit = resolve_figures(figures, load_xbrl_supplement(paths), cfg)
    wide = to_wide(resolved)
    scope, anchor = _cohort_context(paths)

    # --- validation, per company-year
    anchor_lookup: dict[tuple[str, int], float] = {}
    if len(anchor) and {"firm_id", "fy", "total_assets"} <= set(anchor.columns):
        for _, r in anchor.iterrows():
            try:
                anchor_lookup[(r["firm_id"], int(r["fy"]))] = float(r["total_assets"])
            except (TypeError, ValueError):
                continue

    previous: dict[tuple[str, int], float] = {}
    if len(wide):
        for _, r in wide.iterrows():
            if not pd.isna(r.get("total_assets")):
                previous[(r["firm_id"], int(r["fy"]))] = float(r["total_assets"])

    validations = []
    for _, row in wide.iterrows() if len(wide) else []:
        key = (row["firm_id"], int(row["fy"]))
        validations.append({
            "firm_id": row["firm_id"], "fy": int(row["fy"]),
            **validate_year(row, cfg,
                            reference_assets=anchor_lookup.get(key),
                            previous_assets=previous.get((row["firm_id"], int(row["fy"]) - 1))),
        })
    validation = pd.DataFrame(validations)
    if len(wide) and len(validation):
        wide = wide.merge(validation, on=["firm_id", "fy"], how="left")
    elif len(wide):
        wide["validation_flags"] = ""
        wide["n_validation_flags"] = 0

    # --- per-figure confidence, now that validation is known
    if len(resolved):
        flag_lookup = ({(r["firm_id"], int(r["fy"])): r["validation_flags"]
                        for _, r in validation.iterrows()} if len(validation) else {})
        resolved["validation_flags"] = [
            flag_lookup.get((r["firm_id"], int(r["fy"])), "") for _, r in resolved.iterrows()]
        resolved["confidence"] = [
            field_confidence(r, r["validation_flags"]) for _, r in resolved.iterrows()]

    # --- missing-data flags and the gap-filling outcome
    wide = _add_missing_flags(wide, resolved, scope, cfg)
    wide = _apply_pair_exclusion(wide, scope, cfg)

    # --- the ratios
    ratios = compute_ratio_table(wide)

    return {"figures": resolved, "figures_audit": audit, "financials": wide,
            "ratios": ratios, "reports": pd.DataFrame(reports)}


def _add_missing_flags(wide: pd.DataFrame, resolved: pd.DataFrame, scope: pd.DataFrame,
                       cfg: dict[str, Any]) -> pd.DataFrame:
    """Record what is missing per company-year, keeping every row."""
    required = cfg["financials"]["required_fields"] or CORE_FIELDS

    # Every company-year we expected, even those with no report at all, and the
    # cohort metadata that goes with it -- ``label`` in particular, because the
    # count of unrecoverable company-years has to be reported per class.
    carry = ["pair_id", "role", "label", "horizon", "company_name"]
    expected = pd.DataFrame(columns=["firm_id", "fy"])
    if len(scope) and {"firm_id", "fy"} <= set(scope.columns):
        keep = ["firm_id", "fy"] + [c for c in carry if c in scope.columns]
        expected = scope[keep].drop_duplicates(subset=["firm_id", "fy"])
        expected["fy"] = pd.to_numeric(expected["fy"], errors="coerce").astype("Int64")
    if len(wide):
        wide["fy"] = pd.to_numeric(wide["fy"], errors="coerce").astype("Int64")
        # Keep exactly the company-years the cohort asked for. A prior-year
        # column often reaches back beyond the sample frame, and keeping those
        # would give one class more years than the other -- the calendar-year
        # shortcut the pair rule exists to prevent.
        if len(expected):
            wide = expected.merge(wide, on=["firm_id", "fy"], how="left")
    elif len(expected):
        wide = expected.copy()
        for column in STANDARD_FIELDS:
            wide[column] = np.nan

    if not len(wide):
        return wide

    sources: dict[tuple[str, int], set[str]] = {}
    if len(resolved):
        for _, r in resolved.iterrows():
            sources.setdefault((r["firm_id"], int(r["fy"])), set()).add(r["source"])

    missing_lists, n_found, has_core, source_col = [], [], [], []
    for _, row in wide.iterrows():
        absent = [f for f in required if pd.isna(row.get(f))]
        present = [f for f in STANDARD_FIELDS if not pd.isna(row.get(f))]
        missing_lists.append(";".join(absent))
        n_found.append(len(present))
        has_core.append(len(absent) == 0)
        got = sources.get((row["firm_id"], int(row["fy"])) if not pd.isna(row["fy"]) else ("", 0), set())
        source_col.append(SOURCE_PRIMARY if SOURCE_PRIMARY in got
                          else SOURCE_COMPARATIVE if SOURCE_COMPARATIVE in got
                          else SOURCE_XBRL if SOURCE_XBRL in got else SOURCE_NONE)

    wide["missing_fields"] = missing_lists
    wide["n_fields_found"] = n_found
    wide["has_core_financials"] = has_core
    wide["financials_source"] = source_col
    wide["financials_missing"] = ~pd.Series(has_core, index=wide.index)
    return wide


def _apply_pair_exclusion(wide: pd.DataFrame, scope: pd.DataFrame,
                          cfg: dict[str, Any]) -> pd.DataFrame:
    """Drop a peer's year from modelling when its distressed partner's year is unusable.

    The same rule Phase 2 applies to documents (``keep_pairs_aligned``): both
    classes must cover the same calendar years, or the model can separate them
    on the year alone.
    """
    if not len(wide):
        return wide
    wide["financials_exclude_reason"] = np.where(
        wide["financials_missing"], "missing_financials", "")

    if not cfg["labels"]["keep_pairs_aligned"]:
        wide["included_financials"] = ~wide["financials_missing"]
        return wide

    merged = wide
    if not {"pair_id", "role"} <= set(wide.columns):
        if not len(scope) or not {"pair_id", "firm_id", "fy", "role"} <= set(scope.columns):
            wide["included_financials"] = ~wide["financials_missing"]
            return wide
        meta = scope[["pair_id", "firm_id", "fy", "role"]].drop_duplicates(subset=["firm_id", "fy"])
        meta["fy"] = pd.to_numeric(meta["fy"], errors="coerce").astype("Int64")
        merged = wide.merge(meta, on=["firm_id", "fy"], how="left")

    broken = {(r["pair_id"], r["fy"]) for _, r in merged.iterrows()
              if r.get("role") == "distressed" and bool(r["financials_missing"])
              and not pd.isna(r.get("pair_id"))}
    partner = [
        bool(not pd.isna(r.get("pair_id")) and (r["pair_id"], r["fy"]) in broken
             and r.get("role") != "distressed")
        for _, r in merged.iterrows()
    ]
    merged["financials_exclude_reason"] = np.where(
        merged["financials_exclude_reason"].astype(bool), merged["financials_exclude_reason"],
        np.where(partner, "pair_partner_missing_financials", ""))
    merged["included_financials"] = merged["financials_exclude_reason"] == ""
    return merged


def compute_ratio_table(wide: pd.DataFrame) -> pd.DataFrame:
    """The stream-C table: 12 ratios and the condition indicators per company-year."""
    if not len(wide):
        return pd.DataFrame(columns=["firm_id", "fy", *RATIO_FIELDS, *INDICATOR_FIELDS])
    rows = []
    for _, row in wide.iterrows():
        figures = {f: (None if pd.isna(row.get(f)) else float(row.get(f)))
                   for f in STANDARD_FIELDS}
        result = compute_ratios(figures)
        record: dict[str, Any] = {"firm_id": row["firm_id"], "fy": row["fy"]}
        record.update({f: result.values.get(f) for f in RATIO_FIELDS})
        # both Altman scales, so the zone can be checked against the score it
        # actually comes from (the discriminant, not the rating-equivalent one)
        record["altman_z_dprime"] = result.derived.get("altman_z_dprime")
        record["altman_zone"] = result.altman_zone
        record.update({f: result.indicators.get(f) for f in INDICATOR_FIELDS})
        record["ratio_reasons"] = ";".join(f"{k}={v}" for k, v in result.reasons.items() if v)
        for passthrough in ("label", "role", "pair_id", "horizon", "financials_source",
                            "financials_missing", "included_financials", "validation_flags",
                            "n_fields_found"):
            if passthrough in row.index:
                record[passthrough] = row[passthrough]
        rows.append(record)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- spot check
def make_spot_check_sheet(paths: Paths, cfg: dict[str, Any], figures: pd.DataFrame,
                          wide: pd.DataFrame, fraction: float | None = None,
                          overwrite: bool = False) -> pd.DataFrame:
    """A hand-verification sheet: a random sample of company-years, with page numbers.

    Every extracted field is listed with the page it came from and the label it
    matched, so a person can open the PDF at that page and tick or correct it.
    Same shape as the Phase 2 section QA sheet, so the workflow is familiar.
    """
    fraction = fraction if fraction is not None else cfg["financials"]["spot_check_fraction"]
    if paths.financials_qa_sheet.exists() and not overwrite:
        raise FileExistsError(
            f"{paths.financials_qa_sheet} exists (it may contain hand checks). "
            "Use --overwrite to replace.")
    if not len(wide):
        raise ValueError("Nothing to spot-check: no company-years were built.")

    years = wide[["firm_id", "fy"]].drop_duplicates()
    n = max(1, int(round(len(years) * fraction)))
    sample = years.sample(min(n, len(years)), random_state=cfg["qa"]["random_seed"])

    rows = []
    for _, key in sample.sort_values(["firm_id", "fy"]).iterrows():
        subset = figures[(figures["firm_id"] == key["firm_id"]) & (figures["fy"] == key["fy"])] \
            if len(figures) else figures
        if not len(subset):
            rows.append({"firm_id": key["firm_id"], "fy": key["fy"], "field": "(nothing extracted)",
                         "value_cr": "", "source": SOURCE_NONE, "source_doc_id": "", "page": "",
                         "statement": "", "label": "", "printed": "", "unit": "",
                         "confidence": "", "parse_flags": "",
                         **{c: "" for c in SPOT_CHECK_FILL_COLS}})
            continue
        for _, r in subset.sort_values("field").iterrows():
            rows.append({
                "firm_id": r["firm_id"], "fy": r["fy"], "field": r["field"],
                "value_cr": r["value_cr"], "source": r["source"],
                "source_doc_id": r["source_doc_id"], "page": r["page"],
                "statement": r["statement"], "label": r["label"],
                "printed": r["printed"], "unit": r["unit"],
                "confidence": r.get("confidence", ""), "parse_flags": r["parse_flags"],
                **{c: "" for c in SPOT_CHECK_FILL_COLS},
            })
    sheet = pd.DataFrame(rows)
    write_csv(sheet, paths.financials_qa_sheet)
    log.info("spot-check sheet: %d company-years (%.0f%%) -> fill in %s",
             len(sample), fraction * 100, SPOT_CHECK_FILL_COLS)
    return sheet


SPOT_CHECK_FILL_COLS = ["value_correct", "page_correct", "true_value_cr", "notes"]


def score_spot_check(paths: Paths) -> pd.DataFrame:
    """Accuracy per field from the filled spot-check sheet."""
    df = pd.read_csv(paths.financials_qa_sheet, dtype=str).fillna("")
    for column in ("value_correct", "page_correct"):
        df[column] = df[column].str.strip().str.upper()
    filled = df[df["value_correct"].isin(["Y", "N"])]
    if filled.empty:
        raise ValueError("No rows filled in yet (value_correct must be Y or N).")

    def rate(series: pd.Series) -> float | None:
        series = series[series.isin(["Y", "N"])]
        return round((series == "Y").mean(), 3) if len(series) else None

    out = [{"field": fieldname, "n_checked": len(group),
            "value_accuracy": rate(group["value_correct"]),
            "page_accuracy": rate(group["page_correct"])}
           for fieldname, group in filled.groupby("field")]
    result = pd.DataFrame(out).sort_values("field")
    write_csv(result, paths.qa / "financials_spot_check_scores.csv")
    return result


# --------------------------------------------------------------------------- entry point
def run_financials(cfg: dict[str, Any], paths: Paths, doc_ids: list[str] | None = None,
                   spot_check: bool = True, overwrite_spot_check: bool = False) -> pd.DataFrame:
    """``bpp financials`` -- extract, validate, compute ratios, write every table."""
    tables = build_financials(paths, cfg, doc_ids)

    write_csv(tables["figures"], paths.financials_figures)
    write_csv(tables["financials"], paths.financials_extracted)
    write_csv(tables["ratios"], paths.ratios)

    missing = tables["financials"]
    if len(missing) and "financials_missing" in missing.columns:
        write_csv(missing[missing["financials_missing"]], paths.financials_missing)

    if spot_check:
        try:
            make_spot_check_sheet(paths, cfg, tables["figures"], tables["financials"],
                                  overwrite=overwrite_spot_check)
        except (FileExistsError, ValueError) as exc:
            log.warning("spot-check sheet not written: %s", exc)

    report_coverage(tables["financials"], tables["ratios"])
    return tables["financials"]


def report_coverage(wide: pd.DataFrame, ratios: pd.DataFrame) -> dict[str, Any]:
    """Log what was recovered and what was not, split by class.

    The count of unrecoverable company-years by class is the number that decides
    whether the pair-matched design survives, so it is printed every run.
    """
    summary: dict[str, Any] = {"company_years": len(wide)}
    if not len(wide):
        log.warning("no company-years built")
        return summary

    log.info("company-years: %d", len(wide))
    if "financials_source" in wide.columns:
        for source, count in wide["financials_source"].value_counts().items():
            log.info("  source %-24s %d", source, count)

    if "label" in wide.columns and "financials_missing" in wide.columns:
        # dropna=False: an unlabelled company-year must not vanish from the count
        # that decides whether the pair-matched design survives
        by_class = wide.groupby("label", dropna=False)["financials_missing"].agg(["sum", "count"])
        for label, row in by_class.iterrows():
            if pd.isna(label):
                name = "unlabelled"
            else:
                name = "distressed" if str(label) in ("1", "1.0", "True") else "healthy"
            log.info("  unrecoverable %-12s %d of %d", name, int(row["sum"]), int(row["count"]))
            summary[f"unrecoverable_{name}"] = int(row["sum"])
    elif "financials_missing" in wide.columns:
        summary["unrecoverable"] = int(wide["financials_missing"].sum())
        log.info("  unrecoverable company-years: %d (no labels yet, so not split by class)",
                 summary["unrecoverable"])

    if "validation_flags" in wide.columns:
        flagged = wide[wide["validation_flags"].astype(str).str.len() > 0]
        log.info("  company-years with validation flags: %d", len(flagged))

    if len(ratios) and "altman_z_em" in ratios.columns:
        usable = ratios["altman_z_em"].notna().sum()
        log.info("  Altman Z'' computed for %d of %d company-years", usable, len(ratios))
    return summary
