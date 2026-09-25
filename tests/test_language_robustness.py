"""Regression tests for features that came out silently wrong, or silently constant.

Every case here passed the first round of tests and was still broken. A feature
that is the same value for every company looks perfectly healthy in the output
table and carries no information at all, so most of these check that the two
classes actually *differ* rather than that a number was produced.
"""

import json

import pandas as pd
import pytest

from bpp.nlp import lm
from bpp.nlp.features import build_language_features, caro_flags
from bpp.nlp.lexicon import (Lexicon, _version_from_name, distress_phrases, hedging,
                             normalise_for_phrases, sentences, words)
from bpp.nlp.readability import count_syllables, is_complex, readability


# --------------------------------------------------------------------------- CARO negation
@pytest.mark.parametrize("clause", [
    "(ix)(a) The Company has not defaulted in repayment of loans or other borrowings to any lender.",
    "(ix)(b) Neither the Company nor any of its promoters or directors has been declared a "
    "wilful defaulter by any bank or financial institution or other lender.",
    "The Company has not, during any part of the year, defaulted in the repayment of loans.",
    "(ix)(a) whether the company has defaulted in repayment of loans or other borrowings, "
    "if yes, the period and the amount of default to be reported",
    "There were no defaults in the repayment of dues to any lender during the year.",
])
def test_clean_annexure_clauses_do_not_raise_the_default_flag(clause):
    """All of these are ordinary unqualified CARO wording. A flag matching the
    phrase alone is 1 for every company in the sample."""
    assert caro_flags(clause)["caro_default_flag"] == 0.0


@pytest.mark.parametrize("clause", [
    "(ix) The Company has defaulted in repayment of loans and interest to banks.",
    "(ix) There has been a delay in repayment of principal to a financial institution.",
    "The Company has not defaulted in repayment of loans. The lenders have declared the "
    "Company a wilful defaulter.",
])
def test_a_real_default_still_raises_the_flag(clause):
    assert caro_flags(clause)["caro_default_flag"] == 1.0


@pytest.mark.parametrize("clause", [
    "(vii) The Company has been regular in depositing undisputed statutory dues.",
    "there are no undisputed statutory dues outstanding as at 31 March 2023 for a period of "
    "more than six months",
    "there were no undisputed statutory dues in arrears as at March 31, 2023",
    "There were no arrears of statutory dues as at 31 March 2023 outstanding for more than six months",
])
def test_clean_statutory_dues_clauses_do_not_raise_the_flag(clause):
    assert caro_flags(clause)["caro_statutory_dues_flag"] == 0.0


@pytest.mark.parametrize("clause", [
    "(vii) The Company has not been regular in depositing undisputed statutory dues.",
    "undisputed statutory dues outstanding for more than six months are Rs. 38 crore",
])
def test_a_real_statutory_failure_raises_the_flag(clause):
    assert caro_flags(clause)["caro_statutory_dues_flag"] == 1.0


def test_an_annexure_can_deny_default_and_still_report_a_wilful_defaulter():
    """One annexure often does both; the later clause must survive the earlier denial."""
    text = ("(ix)(a) The Company has not defaulted in repayment of loans. "
            "(ix)(b) The Company has been declared a wilful defaulter by a consortium bank.")
    assert caro_flags(text)["caro_default_flag"] == 1.0


# --------------------------------------------------------------------------- constant features
def _cohort_payload(firm_id, fy, caro, mdna):
    section = lambda t: {"heading": "H", "start_page": 1, "end_page": 1, "n_chars": len(t), "text": t}
    return {"doc_id": f"{firm_id}_FY{fy}", "firm_id": firm_id, "fy": fy,
            "sections": {"mdna": section(mdna), "auditor_report": section("Opinion. " + mdna),
                         "caro_annexure": section(caro)},
            "audit_opinion": "unmodified", "leakage": {}}


CLEAN = ("(vii)(b) There are no undisputed statutory dues outstanding as at 31 March 2023 for a "
         "period of more than six months. (ix)(a) The Company has not defaulted in repayment of "
         "loans or other borrowings. (ix)(b) Neither the Company nor any of its promoters has "
         "been declared a wilful defaulter by any bank.")
DISTRESSED = ("(vii)(a) The Company has not been regular in depositing undisputed statutory dues. "
              "(ix)(a) The Company has defaulted in repayment of loans to banks.")


def test_the_caro_flags_are_not_constant_across_a_cohort(cfg, paths):
    """The failure mode of this phase: a flag that is 1 everywhere."""
    for firm_id, caro in (("A", CLEAN), ("B", CLEAN), ("C", CLEAN), ("D", DISTRESSED)):
        (paths.sections / f"{firm_id}_FY2019.json").write_text(
            json.dumps(_cohort_payload(firm_id, 2019, caro, "Some management commentary here.")),
            encoding="utf-8")
    frame = build_language_features(paths, cfg)
    assert frame["caro_default_flag"].nunique() == 2
    assert frame["caro_statutory_dues_flag"].nunique() == 2
    assert frame["caro_default_flag"].sum() == 1.0


def test_the_auditors_boilerplate_does_not_look_like_distress(cfg, paths):
    """SA 570 puts "going concern basis of accounting" in every clean report, so
    a distress list containing it measures report length, not distress."""
    boilerplate = ("We conclude on the appropriateness of management's use of the going concern "
                   "basis of accounting and, based on the audit evidence obtained, whether a "
                   "material uncertainty exists.")
    result = distress_phrases(boilerplate, words(boilerplate), Lexicon())
    assert result["distress_phrase_density"] == 0.0


# --------------------------------------------------------------------------- readability
def test_a_section_of_headings_and_bullets_is_not_one_sentence():
    """PDF-extracted MD&A is largely headings and bullets. A splitter that needs a
    full stop returns one sentence, so avg sentence length equals the word count
    and Fog reads ~40 for every such document."""
    text = ("Industry structure and developments\n"
            "The domestic steel industry witnessed subdued demand\n"
            "- volatile raw material prices\n"
            "- muted infrastructure spending\n"
            "Opportunities and threats\n"
            "The Company expects margin pressure to continue")
    assert len(sentences(text)) >= 5
    result = readability(text)
    assert result["avg_sentence_length"] < result["n_words"]
    assert result["fog_index"] < 25


def test_numbered_and_lettered_list_markers_start_a_new_unit():
    assert len(sentences("(i) first point\n(ii) second point\n(iii) third point")) == 3


@pytest.mark.parametrize("word,expected", [
    ("business", 2), ("revenue", 3), ("employee", 3), ("family", 3), ("nation", 2),
])
def test_frequent_words_the_rule_used_to_get_wrong(word, expected):
    assert count_syllables(word) == expected


@pytest.mark.parametrize("word,complex_", [
    ("reported", False),      # three syllables only by taking "-ed"
    ("generally", True),      # an adverb, not a verb ending
    ("borrowing", True), ("borrowings", True),   # must agree with each other
    ("business", False), ("revenue", True),
])
def test_complex_word_classification(word, complex_):
    assert is_complex(word) is complex_


# --------------------------------------------------------------------------- tokens
def test_hyphenated_compounds_split_so_they_can_match_the_dictionary():
    """"non-performing" as one token can never be in the LM dictionary, and it is
    exactly the vocabulary this project is looking for."""
    assert words("non-performing assets") == ["non", "performing", "assets"]


def test_a_hyphen_at_a_line_break_is_rejoined():
    assert words("manage-\nment discussion") == ["management", "discussion"]


def test_hyphen_spelling_does_not_double_count_a_distress_phrase():
    text = "classified as a non-performing asset and as a non performing asset"
    result = distress_phrases(text, words(text), Lexicon())
    assert result["distress_phrase_distinct"] == 1.0


def test_normalisation_makes_hyphenation_irrelevant():
    assert normalise_for_phrases("non-performing") == normalise_for_phrases("non performing")


# --------------------------------------------------------------------------- hedging
def test_an_overlapping_hedge_is_counted_once():
    """"there can be no assurance that..." is one hedge, not two."""
    text = "There can be no assurance that the restructuring will be completed."
    result = hedging(text, words(text), Lexicon())
    assert result["hedge_phrase_density"] == pytest.approx(1 / len(words(text)))


def test_subject_is_counted_as_a_phrase_and_not_also_as_a_word():
    text = "The outcome is subject to the approval of lenders"
    result = hedging(text, words(text), Lexicon())
    assert result["hedge_density"] == 0.0            # "subject" alone is not a hedge term
    assert result["hedge_phrase_density"] > 0        # "subject to" is


# --------------------------------------------------------------------------- dictionary
def test_the_dictionary_version_is_the_last_year_in_a_range():
    """The canonical file is ..._1993-2021.csv; the first year is not the version."""
    assert _version_from_name("LoughranMcDonald_MasterDictionary_1993-2021.csv") == "2021"


# --------------------------------------------------------------------------- language model
def test_the_distribution_sums_to_one_at_every_order():
    model = lm.fit(["the company reported strong growth in revenue and margins"] * 4, order=3)
    for context in [(), ("the",), ("the", "company")]:
        total = sum(lm.probability(model, w, context) for w in model.vocabulary)
        assert total == pytest.approx(1.0, abs=1e-9), f"context {context} sums to {total}"


def test_a_context_with_no_observed_followers_still_sums_to_one():
    """The weight must pass all the mass down, or the distribution leaks away."""
    model = lm.fit(["alpha beta gamma delta"] * 3, order=3)
    total = sum(lm.probability(model, w, ("delta", "</s>")) for w in model.vocabulary)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_continuation_counts_are_actually_built():
    """Without them this is absolute discounting with a Kneser-Ney unigram --
    a different model with the same name."""
    model = lm.fit(["the company reported growth", "the company reported losses"] * 3, order=3)
    assert model.cont_counts.get(1)
    assert model.cont_total.get(2)


def test_a_unigram_model_still_tells_documents_apart():
    """With order 1 every context is empty; a pure continuation base case makes
    perplexity exactly |V| for every document."""
    model = lm.fit(["the company reported strong growth in revenue"] * 4, order=1)
    common = lm.perplexity(model, "the company reported growth")
    unknown = lm.perplexity(model, "zebra xylophone quixotic")
    assert common != unknown


# --------------------------------------------------------------------------- drift join
def _drift_payload(doc_id, firm_id, fy, mdna):
    section = lambda t: {"heading": "H", "start_page": 1, "end_page": 1, "n_chars": len(t), "text": t}
    return {"doc_id": doc_id, "firm_id": firm_id, "fy": fy,
            "sections": {"mdna": section(mdna), "auditor_report": section("Opinion."),
                         "caro_annexure": section(CLEAN)},
            "audit_opinion": "unmodified", "leakage": {}}


def test_a_fiscal_year_stored_as_a_string_does_not_disable_drift(cfg, paths):
    """It used to silently return None for all seven drift columns, leaving only
    a quiet "drift 0 of N" in the coverage line."""
    for fy, text in (("2018", "alpha beta gamma delta"), ("2019", "alpha beta gamma epsilon")):
        (paths.sections / f"D_FY{fy}.json").write_text(
            json.dumps(_drift_payload(f"D_FY{fy}", "D", fy, text)), encoding="utf-8")
    frame = build_language_features(paths, cfg).set_index("doc_id")
    assert bool(frame.loc["D_FY2019", "has_previous_year"]) is True
    assert pd.notna(frame.loc["D_FY2019", "drift_jaccard"])


def test_two_documents_in_one_firm_year_each_get_their_own_drift(cfg, paths):
    """Keyed by (firm, fy) one document silently received the other's drift."""
    (paths.sections / "D_FY2018.json").write_text(
        json.dumps(_drift_payload("D_FY2018", "D", 2018, "one two three four five")), encoding="utf-8")
    (paths.sections / "D_FY2019_a.json").write_text(
        json.dumps(_drift_payload("D_FY2019_a", "D", 2019,
                                  "one two three four five six seven eight nine ten")), encoding="utf-8")
    (paths.sections / "D_FY2019_b.json").write_text(
        json.dumps(_drift_payload("D_FY2019_b", "D", 2019, "alpha beta")), encoding="utf-8")
    frame = build_language_features(paths, cfg).set_index("doc_id")
    assert frame.loc["D_FY2019_a", "drift_length_ratio"] == pytest.approx(2.0)
    assert frame.loc["D_FY2019_b", "drift_length_ratio"] == pytest.approx(0.4)


def test_the_document_drift_was_measured_against_is_recorded(cfg, paths):
    for fy, text in ((2018, "alpha beta"), (2019, "alpha gamma")):
        (paths.sections / f"D_FY{fy}.json").write_text(
            json.dumps(_drift_payload(f"D_FY{fy}", "D", fy, text)), encoding="utf-8")
    frame = build_language_features(paths, cfg).set_index("doc_id")
    assert frame.loc["D_FY2019", "drift_previous_doc_id"] == "D_FY2018"
    assert frame.loc["D_FY2018", "drift_previous_doc_id"] == ""
