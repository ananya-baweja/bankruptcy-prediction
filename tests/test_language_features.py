"""The stream-B run end to end, and the ways a feature can be silently wrong."""

import json
import random

import pandas as pd
import pytest

from bpp.nlp.features import (CORE_LANGUAGE_FEATURES, build_language_features,
                              document_features, flag_features, language_feature_columns,
                              report_coverage)
from bpp.nlp.lexicon import load_lm_dictionary
from bpp.synthetic import DISTRESS_SENTENCES, HEALTHY_SENTENCES, NEUTRAL_SENTENCES

# A realistic clean annexure, not a two-sentence toy. Every clause here appears
# in ordinary unqualified CARO reports, and each one used to set a flag to 1.
CLEAN_CARO = (
    "(vii)(a) The Company has been regular in depositing undisputed statutory dues. "
    "(vii)(b) There are no undisputed statutory dues outstanding as at 31 March 2023 for a "
    "period of more than six months from the date they became payable. "
    "(ix)(a) The Company has not defaulted in repayment of loans or other borrowings to any "
    "lender. "
    "(ix)(b) Neither the Company nor any of its promoters or directors has been declared a "
    "wilful defaulter by any bank or financial institution or other lender.")
DISTRESSED_CARO = (
    "(vii)(a) The Company has not been regular in depositing undisputed statutory dues. "
    "(vii)(b) Undisputed statutory dues outstanding for more than six months are Rs. 38 crore. "
    "(ix)(a) The Company has defaulted in repayment of loans and interest to banks. "
    "(ix)(b) The lenders have declared the Company a wilful defaulter.")


def _section(text):
    return {"heading": "H", "start_page": 1, "end_page": 2, "n_chars": len(text), "text": text}


def _payload(firm_id, fy, distressed, seed=0, mdna=None):
    rng = random.Random(seed)
    tone = DISTRESS_SENTENCES if distressed else HEALTHY_SENTENCES
    mdna_text = mdna if mdna is not None else " ".join(
        rng.choice(tone + NEUTRAL_SENTENCES) for _ in range(30))
    return {
        "doc_id": f"{firm_id}_FY{fy}", "firm_id": firm_id, "fy": fy,
        "sections": {
            "mdna": _section(mdna_text),
            "auditor_report": _section(("Qualified Opinion. " if distressed else "Opinion. ")
                                       + " ".join(rng.choice(NEUTRAL_SENTENCES) for _ in range(8))),
            "caro_annexure": _section(DISTRESSED_CARO if distressed else CLEAN_CARO),
            "directors_report": _section("The Board presents the annual report."),
            "going_concern": _section("material uncertainty") if distressed else None,
            "emphasis_of_matter": _section("We draw attention"),
            "basis_for_modified_opinion": _section("Basis for Qualified Opinion") if distressed else None,
        },
        "audit_opinion": "qualified" if distressed else "unmodified",
        "leakage": {"ibc_generic_mentions": 2 if distressed else 0,
                    "cirp_specific_mentions": 1 if distressed else 0},
    }


def _write_cohort(paths, years=(2017, 2018, 2019)):
    rows = []
    for firm_id, distressed in (("BSE900101", True), ("BSE900201", False)):
        for fy in years:
            payload = _payload(firm_id, fy, distressed, seed=fy * (2 if distressed else 3))
            (paths.sections / f"{firm_id}_FY{fy}.json").write_text(
                json.dumps(payload), encoding="utf-8")
            rows.append({"doc_id": payload["doc_id"], "firm_id": firm_id, "fy": fy,
                         "pair_id": "P1", "role": "distressed" if distressed else "healthy",
                         "label": 1 if distressed else 0, "horizon": f"t-{2020 - fy}",
                         "included": True})
    pd.DataFrame(rows).to_csv(paths.documents_labeled, index=False)


@pytest.fixture
def lm_dictionary(paths):
    rows = [{"word": w, "category": "negative"} for w in
            ("loss", "losses", "default", "defaulted", "stress", "stressed", "delays",
             "challenges", "constraints", "overruns", "non")]
    rows += [{"word": w, "category": "positive"} for w in ("growth", "strong", "healthy", "improved")]
    rows += [{"word": w, "category": "uncertainty"} for w in ("may", "could", "believes", "subject")]
    rows += [{"word": w, "category": "weak_modal"} for w in ("may", "could", "might")]
    pd.DataFrame(rows).to_csv(paths.lm_dictionary, index=False)
    return paths.lm_dictionary


@pytest.fixture
def features(cfg, paths, lm_dictionary):
    _write_cohort(paths)
    return build_language_features(paths, cfg)


# --------------------------------------------------------------------------- the run
def test_one_row_per_report(features):
    assert len(features) == 6
    assert features["doc_id"].is_unique


def test_the_core_vector_is_twenty_five_values(features):
    """Table 7 and the 25 -> 64 -> 64 layer in the progress report."""
    assert len(CORE_LANGUAGE_FEATURES) == 25
    missing = [c for c in CORE_LANGUAGE_FEATURES if c not in features.columns]
    assert missing == []


def test_the_full_table_keeps_more_than_the_core_for_the_ablations(features):
    assert len(language_feature_columns(features)) > len(CORE_LANGUAGE_FEATURES)


def test_mdna_and_auditor_sections_get_separate_columns(features):
    """Phase 7 asks MD&A only vs auditor only vs both; averaging them first
    would make that ablation impossible."""
    assert "mdna_lm_negative_ratio" in features.columns
    assert "auditor_lm_negative_ratio" in features.columns


def test_cohort_metadata_is_carried_through(features):
    assert set(features.columns) >= {"label", "role", "pair_id", "horizon"}


def test_the_distressed_firm_reads_more_negative_than_its_peer(features):
    """Not a claim about the model -- a check that the signs point the right way."""
    by_label = features.groupby("label")["mdna_lm_negative_ratio"].mean()
    assert by_label.loc[1] > by_label.loc[0]


def test_the_distressed_firm_hedges_more(features):
    by_label = features.groupby("label")["mdna_hedge_density"].mean()
    assert by_label.loc[1] > by_label.loc[0]


# --------------------------------------------------------------------------- flags
def test_a_clean_caro_annexure_does_not_raise_the_default_flag():
    """Every clean report says the Company has *not* defaulted. A flag that
    matches the phrase alone is 1 for everyone and carries no information."""
    assert flag_features(_payload("BSE900201", 2019, False))["caro_default_flag"] == 0.0


def test_a_real_default_raises_the_flag():
    assert flag_features(_payload("BSE900101", 2019, True))["caro_default_flag"] == 1.0


def test_statutory_dues_flag_follows_the_negation_too():
    assert flag_features(_payload("BSE900201", 2019, False))["caro_statutory_dues_flag"] == 0.0
    assert flag_features(_payload("BSE900101", 2019, True))["caro_statutory_dues_flag"] == 1.0


def test_the_auditors_opinion_becomes_an_ordinal():
    assert flag_features(_payload("BSE900101", 2019, True))["audit_opinion_severity"] == 1.0
    assert flag_features(_payload("BSE900201", 2019, False))["audit_opinion_severity"] == 0.0


def test_an_unreadable_opinion_is_missing_not_zero():
    payload = _payload("BSE900101", 2019, True)
    payload["audit_opinion"] = "not_found"
    flags = flag_features(payload)
    assert flags["audit_opinion_severity"] is None
    assert flags["audit_opinion_found"] == 0.0


def test_a_missing_caro_annexure_gives_missing_flags_not_false():
    payload = _payload("BSE900101", 2019, True)
    payload["sections"]["caro_annexure"] = None
    flags = flag_features(payload)
    assert flags["caro_default_flag"] is None
    assert flags["has_caro_annexure"] == 0.0


# --------------------------------------------------------------------------- perplexity
def test_perplexity_is_empty_unless_a_training_set_is_named(features):
    """Fitting the reference model on every report leaks the test fold into training."""
    assert features["perplexity"].isna().all()
    assert (features["perplexity_source"] == "not_fitted_no_training_set_given").all()


def test_perplexity_is_computed_against_a_named_training_set(cfg, paths, lm_dictionary):
    _write_cohort(paths)
    train = ["BSE900201_FY2017", "BSE900201_FY2018"]      # healthy firm only
    frame = build_language_features(paths, cfg, lm_train_doc_ids=train)
    assert frame["perplexity"].notna().all()
    assert (frame["perplexity_source"] == "fold_training_set").all()


def test_documents_used_to_fit_the_model_are_marked(cfg, paths, lm_dictionary):
    """Scoring a document with a model fitted on it flatters it; make that visible."""
    _write_cohort(paths)
    train = ["BSE900201_FY2017"]
    frame = build_language_features(paths, cfg, lm_train_doc_ids=train)
    marked = frame.set_index("doc_id")["perplexity_in_training"]
    assert marked.loc["BSE900201_FY2017"] == 1.0
    assert marked.loc["BSE900101_FY2019"] == 0.0


# --------------------------------------------------------------------------- drift
def test_the_first_year_of_a_firm_has_no_drift(features):
    first = features[features["fy"] == 2017]
    assert (~first["has_previous_year"].astype(bool)).all()
    assert first["drift_jaccard"].isna().all()


def test_later_years_have_drift(features):
    later = features[features["fy"] > 2017]
    assert later["has_previous_year"].astype(bool).all()
    assert later["drift_jaccard"].notna().all()


def test_drift_compares_a_firm_with_itself_not_with_its_peer(cfg, paths, lm_dictionary):
    """A year-on-year feature computed across firms would be meaningless."""
    (paths.sections / "BSE900101_FY2018.json").write_text(
        json.dumps(_payload("BSE900101", 2018, True, mdna="alpha beta gamma delta epsilon")),
        encoding="utf-8")
    (paths.sections / "BSE900101_FY2019.json").write_text(
        json.dumps(_payload("BSE900101", 2019, True, mdna="alpha beta gamma delta epsilon")),
        encoding="utf-8")
    (paths.sections / "BSE900201_FY2018.json").write_text(
        json.dumps(_payload("BSE900201", 2018, False, mdna="completely different words entirely")),
        encoding="utf-8")
    frame = build_language_features(paths, cfg)
    row = frame.set_index("doc_id").loc["BSE900101_FY2019"]
    assert row["drift_jaccard"] == pytest.approx(1.0)       # same firm, identical text


# --------------------------------------------------------------------------- robustness
def test_a_report_with_no_mdna_does_not_crash(cfg, paths, lm_dictionary):
    payload = _payload("BSE900101", 2019, True)
    payload["sections"]["mdna"] = None
    (paths.sections / "BSE900101_FY2019.json").write_text(json.dumps(payload), encoding="utf-8")
    frame = build_language_features(paths, cfg)
    row = frame.iloc[0]
    assert row["has_mdna"] == 0.0
    assert pd.isna(row["mdna_fog_index"])


def test_a_report_with_no_sections_at_all_does_not_crash(cfg, paths, lm_dictionary):
    (paths.sections / "BSE900101_FY2019.json").write_text(
        json.dumps({"doc_id": "BSE900101_FY2019", "firm_id": "BSE900101", "fy": 2019,
                    "sections": {}, "audit_opinion": "not_found"}), encoding="utf-8")
    frame = build_language_features(paths, cfg)
    assert len(frame) == 1
    assert frame.iloc[0]["has_mdna"] == 0.0


def test_tone_is_empty_when_no_dictionary_was_supplied(cfg, paths):
    _write_cohort(paths)
    frame = build_language_features(paths, cfg)
    assert frame["mdna_lm_negative_ratio"].isna().all()
    # ...but everything that does not need the dictionary still works
    assert frame["mdna_fog_index"].notna().all()
    assert frame["caro_default_flag"].notna().all()


def test_no_section_files_is_a_clear_error(cfg, paths):
    with pytest.raises(FileNotFoundError, match="extract-sections"):
        build_language_features(paths, cfg)


def test_coverage_reports_what_could_not_be_computed(features):
    summary = report_coverage(features)
    assert summary["documents"] == 6
    assert summary["drift_jaccard"] == 4          # two firms lose their first year


def test_document_features_never_returns_a_bare_zero_for_a_missing_dictionary():
    row = document_features(_payload("BSE900101", 2019, True), load_lm_dictionary(None))
    assert row["mdna_lm_negative_ratio"] is None
    assert row["mdna_fog_index"] is not None
