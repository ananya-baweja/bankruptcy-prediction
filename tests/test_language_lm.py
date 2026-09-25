"""The healthy-firm reference model, and the year-on-year drift features."""

import pytest

from bpp.nlp import lm
from bpp.nlp.drift import cosine, drift_features, jaccard, new_word_share

HEALTHY_CORPUS = [
    "The Company recorded healthy growth in revenue driven by strong order inflows.",
    "Operating margins improved on account of better realisations and prudent cost management.",
    "The Company continued to maintain a comfortable liquidity position and met all debt obligations.",
    "Capital expenditure during the year was funded through internal accruals.",
    "Credit rating agencies reaffirmed the Company's long term rating with a stable outlook.",
    "The Board is confident of sustaining profitable growth in the coming years.",
    "Working capital cycles remained stable and receivables were collected within the credit period.",
    "The outlook for the sector remains positive supported by steady demand.",
] * 3


@pytest.fixture
def reference_model():
    return lm.fit(HEALTHY_CORPUS, order=3)


def test_the_model_fits(reference_model):
    assert reference_model.fitted
    assert reference_model.n_documents == len(HEALTHY_CORPUS)


def test_distressed_language_is_more_surprising_than_healthy_language(reference_model):
    """Perplexity is meaningless without a reference; the reference is healthy-firm
    disclosure, so a high score reads as departure from normal disclosure."""
    in_domain = reference_model and lm.perplexity(
        reference_model, "The Company maintained a comfortable liquidity position.")
    out_of_domain = lm.perplexity(
        reference_model,
        "The Company has defaulted in repayment of borrowings and lenders invoked the pledge.")
    assert out_of_domain > in_domain


def test_probabilities_form_a_distribution(reference_model):
    total = sum(lm.probability(reference_model, w, ("the", "company"))
                for w in reference_model.vocabulary)
    assert total == pytest.approx(1.0, abs=0.02)


def test_an_unseen_word_still_has_probability(reference_model):
    """Kneser-Ney interpolates down to a continuation floor, so nothing is impossible."""
    assert lm.probability(reference_model, "zzzunseen", ("the", "company")) > 0


def test_an_unfitted_model_returns_no_perplexity():
    assert lm.perplexity(lm.fit([]), "anything at all") is None


def test_empty_text_has_no_perplexity(reference_model):
    assert lm.perplexity(reference_model, "") is None
    assert lm.perplexity(reference_model, "   ") is None


def test_rare_words_become_unk_so_the_vocabulary_is_bounded():
    model = lm.fit(["alpha beta gamma", "alpha beta delta"], order=2, min_count=2)
    assert "alpha" in model.vocabulary and "gamma" not in model.vocabulary
    assert lm.UNK in model.vocabulary


# --------------------------------------------------------------------------- drift
def test_identical_text_is_completely_similar():
    tokens = ["revenue", "growth", "margins", "improved"]
    assert jaccard(tokens, tokens) == pytest.approx(1.0)
    assert cosine(tokens, tokens) == pytest.approx(1.0)
    assert new_word_share(tokens, tokens) == pytest.approx(0.0)


def test_completely_different_text_shares_nothing():
    assert jaccard(["alpha", "beta"], ["gamma", "delta"]) == pytest.approx(0.0)
    assert cosine(["alpha", "beta"], ["gamma", "delta"]) == pytest.approx(0.0)
    assert new_word_share(["alpha", "beta"], ["gamma", "delta"]) == pytest.approx(1.0)


def test_a_rewritten_report_is_less_similar_than_a_copied_one():
    """The Lazy Prices idea: companies copy last year forward, and the paragraphs
    they rewrite are the ones under pressure."""
    last_year = ("The Company recorded healthy growth in revenue driven by strong order "
                 "inflows and improved capacity utilisation across all plants.")
    copied = last_year + " The Board thanks its shareholders."
    rewritten = ("Delays in realisation of receivables stretched the working capital cycle "
                 "and the Company is in discussions with lenders regarding restructuring.")
    copied_drift = drift_features(copied, last_year)
    rewritten_drift = drift_features(rewritten, last_year)
    assert copied_drift["drift_jaccard"] > rewritten_drift["drift_jaccard"]
    assert rewritten_drift["drift_new_word_share"] > copied_drift["drift_new_word_share"]


def test_a_first_year_has_no_drift_rather_than_zero_drift():
    """Zero would read as 'wrote exactly the same thing again' -- the opposite."""
    result = drift_features("Some text this year", None)
    assert result["has_previous_year"] is False
    assert result["drift_jaccard"] is None
    assert result["drift_new_word_share"] is None


def test_deltas_carry_the_direction_of_the_change():
    result = drift_features(
        "text now", "text before",
        {"lm_negative_ratio": 0.05, "hedge_density": 0.04, "fog_index": 20.0},
        {"lm_negative_ratio": 0.02, "hedge_density": 0.01, "fog_index": 18.0})
    assert result["drift_negative_delta"] == pytest.approx(0.03)
    assert result["drift_hedge_delta"] == pytest.approx(0.03)
    assert result["drift_fog_delta"] == pytest.approx(2.0)


def test_a_missing_previous_value_gives_a_missing_delta():
    result = drift_features("text now", "text before",
                            {"lm_negative_ratio": 0.05}, {"lm_negative_ratio": None})
    assert result["drift_negative_delta"] is None


def test_empty_sections_do_not_crash_drift():
    assert drift_features("", "something")["has_previous_year"] is False
    assert drift_features("something", "")["has_previous_year"] is False
