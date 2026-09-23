"""Phase 4, stream B - the engineered linguistic features, one row per report.

Reads Phase 2's ``interim/sections/<doc_id>.json``, so the text has already been
split into MD&A, Board's report, auditor's report and CARO annexure, and nothing
here re-reads a PDF.

Features are computed **per section** for the MD&A and the auditor's report,
because Phase 7 has to answer "MD&A only vs auditor only vs both" and that
ablation is impossible if the two are averaged together first. Issue 6 of the
project plan expects the auditor's report to carry the strongest Indian signal,
so it gets its own columns rather than being folded in.

Three things are deliberately absent, and each says why in its own column rather
than arriving as a zero:

* **Tone** is empty when the Loughran-McDonald dictionary has not been supplied.
  A zero would read as a document with no negative words.
* **Perplexity** is empty unless a training set of healthy documents was named.
  Fitting the reference model on every report leaks the test fold into training.
* **Drift** is empty for a firm's first year. A zero would read as "wrote exactly
  the same thing again", which is the opposite of what is true.

Stream A (FinBERT sentence vectors) is not here: it needs torch and a GPU, and
belongs in the Colab notebook. This module is the part that runs on a laptop.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd
from tqdm import tqdm

from bpp.common import write_csv
from bpp.config import Paths
from bpp.nlp import lm as lm_module
from bpp.nlp.drift import NO_PREVIOUS, drift_features
from bpp.nlp.lexicon import (LM_CATEGORIES, Lexicon, distress_phrases, hedging,
                             load_lm_dictionary, tone_ratios, words)
from bpp.nlp.readability import readability

log = logging.getLogger(__name__)

#: sections that get their own feature block
FEATURE_SECTIONS = ["mdna", "auditor_report"]

#: how severe the auditor's opinion is, as an ordinal
OPINION_SEVERITY = {"unmodified": 0.0, "qualified": 1.0, "adverse": 2.0, "disclaimer": 3.0}

# CARO's two reportable failures: loan default (clause ix) and unpaid statutory
# dues (clause vii). Phase 2 extracts the annexure text but does not read it.
#
# Negation is the whole game. A clean annexure says all of these:
#   "the Company has **not** defaulted in repayment of loans or borrowings"
#   "**Neither** the Company **nor** any of its promoters ... has been declared a
#    wilful defaulter"
#   "there are **no** undisputed statutory dues outstanding ... for more than six
#    months"
# and some auditors reproduce the Order's own wording, "**whether** the company
# has defaulted ... if yes, the period and amount of default to be reported".
# A flag that matches the phrase alone is therefore 1 for every company in the
# sample: it looks like a working feature and carries no information at all.
#
# So each occurrence is judged inside its own clause. A negation or a
# hypothetical anywhere before it in that clause cancels it, and a later clause
# can still assert the failure -- one annexure often denies default and then
# reports a wilful defaulter.
_CLAUSE_SPLIT = re.compile(r"[.;\n]+")
_NEGATION_CUE = re.compile(r"\b(?:not|no|never|neither|nor|without|whether|nil|none)\b", re.I)

_CARO_DEFAULT = re.compile(
    r"default(?:ed|s)?\s+in\s+(?:the\s+)?(?:repayment|payment)"
    r"|has\s+defaulted"
    r"|wil{1,2}ful\s+defaulter"
    r"|delay(?:s|ed)?\s+in\s+(?:the\s+)?repayment", re.I)

# Stated as a failure, so a negation cancels it.
_STATUTORY_FAILURE = re.compile(
    r"arrears\s+of\s+statutory\s+dues"
    r"|statutory\s+dues[\w\s,'\-]{0,60}(?:outstanding|in\s+arrears|not\s+been\s+paid)", re.I)
# Stated with a negation *in* it, which is itself the failure, so it is matched
# literally and must not be run through the negation check.
_STATUTORY_IRREGULAR = re.compile(r"\bnot\s+been\s+regular\s+in\s+depositing", re.I)


def _asserted(pattern: re.Pattern[str], text: str) -> bool:
    """True where the text *states* the thing rather than denying or quoting it.

    Judged per clause: "has defaulted" counts, "has not defaulted" does not, and
    "whether the company has defaulted" is the Order's question, not an answer.
    """
    for clause in _CLAUSE_SPLIT.split(text or ""):
        for match in pattern.finditer(clause):
            if not _NEGATION_CUE.search(clause[:match.start()]):
                return True
    return False


def caro_flags(caro_text: str) -> dict[str, float | None]:
    """The two CARO clauses that matter for distress."""
    if not caro_text:
        return {"caro_default_flag": None, "caro_statutory_dues_flag": None}
    statutory = (bool(_STATUTORY_IRREGULAR.search(caro_text))
                 or _asserted(_STATUTORY_FAILURE, caro_text))
    return {
        "caro_default_flag": float(_asserted(_CARO_DEFAULT, caro_text)),
        "caro_statutory_dues_flag": float(statutory),
    }


def _section_text(payload: dict[str, Any], name: str) -> str:
    section = (payload.get("sections") or {}).get(name)
    if not section:
        return ""
    return section.get("text") or ""


def section_features(text: str, lexicon: Lexicon) -> dict[str, float | None]:
    """Tone, hedging, distress phrases and readability for one section."""
    tokens = words(text)
    out: dict[str, float | None] = {}
    out.update(tone_ratios(tokens, lexicon))
    out.update(hedging(text, tokens, lexicon))
    out.update(distress_phrases(text, tokens, lexicon))
    out.update(readability(text))
    return out


def flag_features(payload: dict[str, Any]) -> dict[str, float | None]:
    """The auditor's own statements, which Phase 2 already located."""
    sections = payload.get("sections") or {}
    opinion = payload.get("audit_opinion") or "not_found"
    caro = _section_text(payload, "caro_annexure")
    return {
        "has_going_concern": float(bool(sections.get("going_concern"))),
        "has_emphasis_of_matter": float(bool(sections.get("emphasis_of_matter"))),
        "has_modified_opinion_basis": float(bool(sections.get("basis_for_modified_opinion"))),
        "audit_opinion_severity": OPINION_SEVERITY.get(opinion),
        "audit_opinion_found": float(opinion in OPINION_SEVERITY),
        **caro_flags(caro),
        "has_mdna": float(bool(sections.get("mdna"))),
        "has_auditor_report": float(bool(sections.get("auditor_report"))),
        "has_caro_annexure": float(bool(sections.get("caro_annexure"))),
    }


def document_features(payload: dict[str, Any], lexicon: Lexicon) -> dict[str, Any]:
    """Every per-document feature except perplexity and drift, which need context."""
    row: dict[str, Any] = {
        "doc_id": payload.get("doc_id"),
        "firm_id": payload.get("firm_id"),
        "fy": payload.get("fy"),
    }
    for name in FEATURE_SECTIONS:
        prefix = "mdna" if name == "mdna" else "auditor"
        for key, value in section_features(_section_text(payload, name), lexicon).items():
            row[f"{prefix}_{key}"] = value
    row.update(flag_features(payload))
    leakage = payload.get("leakage") or {}
    row["ibc_generic_mentions"] = float(leakage.get("ibc_generic_mentions", 0))
    row["cirp_specific_mentions"] = float(leakage.get("cirp_specific_mentions", 0))
    return row


# --------------------------------------------------------------------------- the run
def _load_sections(paths: Paths, doc_ids: list[str] | None) -> list[dict[str, Any]]:
    files = sorted(paths.sections.glob("*.json"))
    if doc_ids:
        wanted = set(doc_ids)
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise FileNotFoundError(
            "No section files yet. Run bpp extract-sections first (docs/03_phase2_documents.md).")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def fit_reference_model(payloads: list[dict[str, Any]], train_doc_ids: list[str],
                        cfg: dict[str, Any]) -> lm_module.KneserNeyLM | None:
    """Fit the healthy-firm reference model on an explicitly named training set."""
    if not train_doc_ids:
        return None
    wanted = set(train_doc_ids)
    texts = [_section_text(p, "mdna") for p in payloads if p.get("doc_id") in wanted]
    texts = [t for t in texts if t]
    if not texts:
        log.warning("none of the named perplexity training documents had an MD&A; "
                    "perplexity will be empty")
        return None
    ncfg = cfg["language"]
    model = lm_module.fit(texts, order=ncfg["lm_order"], discount=ncfg["lm_discount"],
                          min_count=ncfg["lm_min_count"])
    log.info("reference LM fitted on %d documents (vocabulary %d)",
             model.n_documents, len(model.vocabulary))
    return model


def build_language_features(paths: Paths, cfg: dict[str, Any],
                            doc_ids: list[str] | None = None,
                            lm_train_doc_ids: list[str] | None = None) -> pd.DataFrame:
    """One row per report: stream-B features, drift and (optionally) perplexity."""
    payloads = _load_sections(paths, doc_ids)
    lexicon = load_lm_dictionary(paths.lm_dictionary)
    if not lexicon.has_lm:
        log.warning("tone features will be empty: no Loughran-McDonald dictionary at %s",
                    paths.lm_dictionary)

    rows = [document_features(p, lexicon) for p in tqdm(payloads, desc="language features")]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # --- perplexity, only against an explicitly named training set
    model = fit_reference_model(payloads, lm_train_doc_ids or [], cfg)
    texts = {p.get("doc_id"): _section_text(p, "mdna") for p in payloads}
    if model is not None:
        trained_on = set(lm_train_doc_ids or [])
        frame["perplexity"] = [lm_module.perplexity(model, texts.get(d, "")) for d in frame["doc_id"]]
        frame["perplexity_source"] = "fold_training_set"
        # A document scored by a model fitted on itself gets an optimistically
        # low perplexity. In fold-wise use the two sets are disjoint; this column
        # makes any overlap visible instead of silently flattering the feature.
        frame["perplexity_in_training"] = [float(d in trained_on) for d in frame["doc_id"]]
        overlap = int(frame["perplexity_in_training"].sum())
        if overlap:
            log.warning("%d documents were scored by a model fitted on them; their perplexity "
                        "is optimistic and must not be used for modelling", overlap)
    else:
        frame["perplexity"] = None
        frame["perplexity_source"] = "not_fitted_no_training_set_given"
        frame["perplexity_in_training"] = 0.0

    # --- drift against the same firm's previous year
    # Keyed by doc_id, not by (firm, fy): two documents can share a firm-year (a
    # revised filing, or a re-extract under a new doc_id), and a (firm, fy) key
    # silently hands one document the other's drift. The fiscal year is taken
    # from the normalised column on both sides of the join -- a raw "2019" string
    # in the JSON would otherwise match nothing and disable drift for every row
    # with no error, leaving only a quiet "drift 0 of N" in the coverage line.
    frame["fy"] = pd.to_numeric(frame["fy"], errors="coerce").astype("Int64")
    text_by_doc = {p.get("doc_id"): _section_text(p, "mdna") for p in payloads}
    row_by_doc = {r["doc_id"]: r for _, r in frame.iterrows()}

    docs_by_firm_year: dict[tuple, list[str]] = {}
    for doc_id, firm_id, fy in zip(frame["doc_id"], frame["firm_id"], frame["fy"]):
        if pd.isna(fy):
            continue
        docs_by_firm_year.setdefault((firm_id, int(fy)), []).append(doc_id)
    duplicated = {k: v for k, v in docs_by_firm_year.items() if len(v) > 1}
    if duplicated:
        log.warning("%d firm-years have more than one document; drift uses the first by "
                    "doc_id: %s", len(duplicated), sorted(duplicated)[:5])

    drift_rows = []
    for _, row in frame.iterrows():
        fy = row["fy"]
        previous_ids = ([] if pd.isna(fy)
                        else sorted(docs_by_firm_year.get((row["firm_id"], int(fy) - 1), [])))
        previous_id = previous_ids[0] if previous_ids else None
        previous_row = row_by_doc.get(previous_id) if previous_id else None
        drift = drift_features(
            text_by_doc.get(row["doc_id"], ""),
            text_by_doc.get(previous_id) if previous_id else None,
            {"lm_negative_ratio": row.get("mdna_lm_negative_ratio"),
             "hedge_density": row.get("mdna_hedge_density"),
             "fog_index": row.get("mdna_fog_index")},
            None if previous_row is None else {
                "lm_negative_ratio": previous_row.get("mdna_lm_negative_ratio"),
                "hedge_density": previous_row.get("mdna_hedge_density"),
                "fog_index": previous_row.get("mdna_fog_index")},
        )
        drift["drift_previous_doc_id"] = previous_id or ""
        drift_rows.append(drift)
    frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame(drift_rows)], axis=1)

    # --- carry the cohort metadata so the table can be modelled directly
    if paths.documents_labeled.exists():
        labelled = pd.read_csv(paths.documents_labeled)
        carry = [c for c in ("pair_id", "role", "label", "horizon", "included", "exclude_reason")
                 if c in labelled.columns]
        if carry and "doc_id" in labelled.columns:
            frame = frame.merge(labelled[["doc_id", *carry]].drop_duplicates("doc_id"),
                                on="doc_id", how="left")

    frame.attrs["lm_dictionary"] = lexicon.lm_source
    frame.attrs["lm_version"] = lexicon.lm_version
    return frame


#: The stream-B vector the model is trained on: 25 values, matching Table 7 and
#: the ``25 -> 64 -> 64`` layer in the progress report. Everything else this
#: module computes is kept in the table for the Phase 7 ablations (MD&A only vs
#: auditor only vs both) but is not fed to the model by default -- with ~100
#: insolvent firms, a 54-wide input into a 64-unit layer is the overfitting the
#: project plan warns about.
CORE_LANGUAGE_FEATURES = [
    # MD&A: tone, hedging, distress phrases, readability, length
    "mdna_lm_negative_ratio", "mdna_lm_positive_ratio", "mdna_lm_uncertainty_ratio",
    "mdna_lm_weak_modal_ratio", "mdna_lm_litigious_ratio",
    "mdna_hedge_density", "mdna_hedge_phrase_density", "mdna_distress_phrase_density",
    "mdna_fog_index", "mdna_avg_sentence_length", "mdna_complex_word_ratio", "mdna_n_words",
    # the reference-model surprise
    "perplexity",
    # the auditor's report, where the plan expects the strongest Indian signal
    "auditor_lm_negative_ratio", "auditor_lm_uncertainty_ratio", "auditor_hedge_density",
    "auditor_distress_phrase_density", "auditor_n_words",
    # what the auditor actually asserted
    "has_going_concern", "audit_opinion_severity", "caro_default_flag",
    "caro_statutory_dues_flag",
    # contribution A: year-on-year drift
    "drift_jaccard", "drift_new_word_share", "drift_negative_delta",
]

#: columns that identify a row rather than describe it
IDENTIFIER_COLUMNS = {
    "doc_id", "firm_id", "fy", "pair_id", "role", "label", "horizon", "included",
    "exclude_reason", "perplexity_source", "has_previous_year", "perplexity_in_training",
    "drift_previous_doc_id",
}


def language_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every numeric feature column, in a stable order, excluding identifiers."""
    return [c for c in frame.columns if c not in IDENTIFIER_COLUMNS]


def report_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    """What was computed and what could not be, so gaps are visible before modelling."""
    summary: dict[str, Any] = {"documents": len(frame)}
    if frame.empty:
        log.warning("no documents")
        return summary
    log.info("documents: %d", len(frame))
    for column, label in (("mdna_lm_negative_ratio", "tone (needs the LM dictionary)"),
                          ("perplexity", "perplexity (needs a training set)"),
                          ("drift_jaccard", "drift (needs a previous year)"),
                          ("mdna_fog_index", "readability (needs an MD&A)")):
        if column in frame.columns:
            have = int(frame[column].notna().sum())
            summary[column] = have
            log.info("  %-40s %d of %d", label, have, len(frame))
    if "audit_opinion_severity" in frame.columns:
        found = int(frame["audit_opinion_severity"].notna().sum())
        log.info("  %-40s %d of %d", "auditor's opinion identified", found, len(frame))
    return summary


def run_language_features(cfg: dict[str, Any], paths: Paths, doc_ids: list[str] | None = None,
                          lm_train_doc_ids: list[str] | None = None) -> pd.DataFrame:
    """``bpp language-features``."""
    frame = build_language_features(paths, cfg, doc_ids, lm_train_doc_ids)
    if not frame.empty:
        write_csv(frame, paths.language_features)
        missing = [c for c in CORE_LANGUAGE_FEATURES if c not in frame.columns]
        log.info("stream-B core vector: %d of %d columns present; %d columns in total",
                 len(CORE_LANGUAGE_FEATURES) - len(missing), len(CORE_LANGUAGE_FEATURES),
                 len(language_feature_columns(frame)))
        if missing:
            log.warning("core features absent from the table: %s", missing)
    report_coverage(frame)
    return frame
