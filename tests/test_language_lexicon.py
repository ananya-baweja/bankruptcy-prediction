import pytest

from bpp.nlp.lexicon import (CLEANED_STOPWORDS, LM_CATEGORIES, Lexicon, cleaned_words,
                             distress_phrases, hedging, load_lm_dictionary, sentences,
                             tone_ratios, words)
from bpp.nlp.readability import count_syllables, is_complex, readability


@pytest.fixture
def lm_file(tmp_path):
    """A small dictionary in the simple two-column form."""
    path = tmp_path / "loughran_mcdonald_2024.csv"
    path.write_text(
        "word,category\n"
        "loss,negative\nlosses,negative\ndefault,negative\ndefaulted,negative\n"
        "growth,positive\nstrong,positive\n"
        "may,uncertainty\nuncertain,uncertainty\napproximately,uncertainty\n"
        "may,weak_modal\ncould,weak_modal\n"
        "will,strong_modal\n"
        "litigation,litigious\n"
        "restrict,constraining\n", encoding="utf-8")
    return path


@pytest.fixture
def master_dictionary_file(tmp_path):
    """The Master Dictionary form: a non-zero year means the word is a member."""
    path = tmp_path / "LoughranMcDonald_MasterDictionary_2024.csv"
    path.write_text(
        "Word,Seq_num,Word Count,Negative,Positive,Uncertainty,Litigious,Strong_Modal,"
        "Weak_Modal,Constraining,Syllables,Source\n"
        "LOSS,1,100,2009,0,0,0,0,0,0,1,12of12inf\n"
        "GROWTH,2,100,0,2009,0,0,0,0,0,1,12of12inf\n"
        "MAY,3,100,0,0,2009,0,0,2009,0,1,12of12inf\n"
        "NEUTRALWORD,4,100,0,0,0,0,0,0,0,3,12of12inf\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- dictionary
def test_the_simple_two_column_form_loads(lm_file):
    lexicon = load_lm_dictionary(lm_file)
    assert lexicon.has_lm
    assert "loss" in lexicon.lm["negative"]
    assert "may" in lexicon.lm["uncertainty"] and "may" in lexicon.lm["weak_modal"]


def test_the_master_dictionary_form_loads(master_dictionary_file):
    """A category column holds the year the word entered it; 0 means not a member."""
    lexicon = load_lm_dictionary(master_dictionary_file)
    assert "loss" in lexicon.lm["negative"]
    assert "growth" in lexicon.lm["positive"]
    assert "may" in lexicon.lm["weak_modal"]
    assert "neutralword" not in lexicon.lm["negative"]
    assert all("neutralword" not in lexicon.lm[c] for c in LM_CATEGORIES)


def test_the_dictionary_version_is_recorded(master_dictionary_file):
    """The dictionary is republished yearly; a result is not reproducible without it."""
    assert load_lm_dictionary(master_dictionary_file).lm_version == "2024"


def test_a_missing_dictionary_is_not_an_error(tmp_path):
    lexicon = load_lm_dictionary(tmp_path / "nope.csv")
    assert not lexicon.has_lm


def test_tone_is_empty_not_zero_without_a_dictionary():
    """A zero would read as a document with no negative words at all."""
    ratios = tone_ratios(words("The Company suffered a loss"), Lexicon())
    assert all(v is None for v in ratios.values())


# --------------------------------------------------------------------------- tone
def test_tone_ratios_are_shares_of_the_full_word_count(lm_file):
    """Loughran and McDonald's own denominator. A stopword-stripped one is not
    comparable with any published figure, including Gupta & Banerjee (2023)."""
    lexicon = load_lm_dictionary(lm_file)
    text = "The Company reported a loss and a further loss"      # 9 words, 2 negative
    ratios = tone_ratios(words(text), lexicon)
    assert ratios["lm_negative_ratio"] == pytest.approx(2 / 9)


def test_a_distressed_text_scores_more_negative_than_a_healthy_one(lm_file):
    lexicon = load_lm_dictionary(lm_file)
    distressed = tone_ratios(words("The Company defaulted and reported a loss"), lexicon)
    healthy = tone_ratios(words("The Company reported strong growth this year"), lexicon)
    assert distressed["lm_negative_ratio"] > healthy["lm_negative_ratio"]
    assert healthy["lm_positive_ratio"] > distressed["lm_positive_ratio"]


def test_empty_text_gives_no_ratios(lm_file):
    assert all(v is None for v in tone_ratios([], load_lm_dictionary(lm_file)).values())


# --------------------------------------------------------------------------- tracks
def test_the_raw_track_keeps_every_word():
    assert words("The Company may not repay") == ["the", "company", "may", "not", "repay"]


def test_the_cleaned_track_drops_stopwords():
    assert "the" not in cleaned_words("The Company may not repay")


def test_hedges_survive_cleaning_because_they_are_measured_on_the_raw_track():
    """'may' and 'could' are stopwords in most lists and are exactly the hedging
    this project sets out to measure -- issue 3 of the project plan."""
    assert "may" not in CLEANED_STOPWORDS
    assert "could" not in CLEANED_STOPWORDS


# --------------------------------------------------------------------------- hedging
def test_hedging_density_counts_modals_and_phrases():
    text = "The Company may possibly recover, subject to the outcome of negotiations."
    result = hedging(text, words(text), Lexicon())
    assert result["hedge_density"] > 0
    assert result["hedge_phrase_density"] > 0        # "subject to"


def test_plain_text_has_no_hedging():
    text = "The Company repaid all borrowings during the year."
    assert hedging(text, words(text), Lexicon())["hedge_density"] == 0.0


def test_hedging_on_empty_text_is_none_not_zero():
    assert hedging("", [], Lexicon())["hedge_density"] is None


# --------------------------------------------------------------------------- distress phrases
def test_distress_phrases_are_found():
    text = "Certain accounts were classified as a non performing asset and a one time settlement was discussed."
    result = distress_phrases(text, words(text), Lexicon())
    assert result["distress_phrase_density"] > 0
    assert result["distress_phrase_distinct"] >= 2


def test_the_distress_list_is_labelled_a_seed():
    """Contribution C mines the real lexicon; nobody should report this as it."""
    assert Lexicon().distress_source == "seed"


# --------------------------------------------------------------------------- readability
@pytest.mark.parametrize("word,expected", [
    ("the", 1), ("provide", 2), ("company", 3), ("nation", 2),
    ("operation", 4), ("insolvency", 4), ("strength", 1),
])
def test_syllable_counting(word, expected):
    assert count_syllables(word) == expected


def test_tion_words_do_not_gain_a_syllable():
    """Corporate prose is full of them; an extra syllable each inflates every Fog score."""
    assert count_syllables("nation") == 2
    assert count_syllables("provision") == 3


def test_complex_words_ignore_common_suffixes():
    assert not is_complex("reported")          # re-port-ed reaches 3 only via "-ed"
    assert is_complex("insolvency")


def test_denser_prose_scores_a_higher_fog_index():
    easy = "The company did well. Sales went up. Cash is fine."
    hard = ("The Company's continued operational underperformance, compounded by the "
            "deterioration of receivable realisation cycles, introduces considerable "
            "uncertainty regarding the sustainability of operations.")
    assert readability(hard)["fog_index"] > readability(easy)["fog_index"]


def test_readability_of_empty_text_is_none_not_zero():
    result = readability("")
    assert result["fog_index"] is None and result["n_words"] == 0.0


def test_sentences_always_returns_at_least_the_text_it_was_given():
    assert len(sentences("No terminal punctuation here")) == 1
    assert sentences("") == []
