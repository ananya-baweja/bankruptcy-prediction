"""Readability and length: how hard the report is to read, and how much of it there is.

The Gunning Fog index is the plan's choice. It is the ratio of long sentences to
long words, so it moves when management starts writing defensively -- longer
sentences, more polysyllabic hedging -- which is the effect the project is
looking for. Length is included in its own right: a report that suddenly gets
much shorter, or much longer, is itself a signal.

Syllables are counted by rule rather than by dictionary. ``nltk`` and ``pyphen``
are not dependencies of this project, and a vowel-group heuristic is accurate
enough for an index that only needs to separate "complex" (3+ syllables) from
the rest. The error it makes is stable across documents, which is what matters
when the feature is a comparison between years.
"""

from __future__ import annotations

import re

from bpp.nlp.lexicon import sentences, words

_VOWEL_GROUP = re.compile(r"[aeiouy]+")
#: vowel pairs the group rule merges but which are really two syllables
#: ("mater-i-al", "negoti-a-tion")
_SPLIT_PAIRS = ("ia", "io", "eo", "ua", "uo")
#: ...except here, where "-tion"/"-sion"/"-cion" is a single syllable. Without
#: this exception every "operation" and "provision" gains a syllable, and
#: corporate prose is full of them, so the Fog index drifts upward on all of it.
_ONE_SYLLABLE_ENDINGS = ("tion", "sion", "cion")


#: Words where the rule is wrong and which are too frequent in these reports for
#: the error to average out. Verified by hand; kept short on purpose -- a long
#: list here would be a dictionary, and the point of a rule is that it
#: generalises to the vocabulary nobody thought of.
_SYLLABLE_EXCEPTIONS = {
    "business": 2, "businesses": 3,
    "revenue": 3, "revenues": 3,
    "employee": 3, "employees": 4,
    "family": 3, "families": 3,
}


def count_syllables(word: str) -> int:
    """Syllables in one English word, by rule. Always at least 1."""
    word = word.lower().strip("'-")
    if not word:
        return 0
    if word in _SYLLABLE_EXCEPTIONS:
        return _SYLLABLE_EXCEPTIONS[word]

    groups = _VOWEL_GROUP.findall(word)
    total = len(groups)
    # A silent final "e" is a vowel group but not a syllable -- but only when it
    # stands alone. In "revenue" the final group is "ue", already one syllable,
    # and subtracting turns a three-syllable word into two.
    if (word.endswith("e") and total > 1 and groups[-1] == "e"
            and not word.endswith(("le", "ee", "ye"))):
        total -= 1
    for pair in _SPLIT_PAIRS:
        for match in re.finditer(pair, word):
            if word[max(0, match.start() - 1):match.start() + 3] in _ONE_SYLLABLE_ENDINGS:
                continue
            total += 1
    return max(1, total)


def is_complex(word: str) -> bool:
    """Fog's "complex word": three or more syllables.

    Gunning excludes words that reach three syllables only by taking a verb
    ending -- "reported" is not complex -- but not adverbs: "generally" is. So
    only ``-ed`` and ``-es`` are stripped, and stripping ``-ly`` as well would
    make "family" simple and disagree with "families".
    """
    stripped = re.sub(r"(?:es|ed)$", "", word.lower())
    return count_syllables(stripped or word) >= 3


def readability(text: str) -> dict[str, float | None]:
    """Fog index plus the two quantities it is built from, and length."""
    tokens = words(text)
    sentence_list = sentences(text)
    n_words, n_sentences = len(tokens), len(sentence_list)
    if n_words == 0 or n_sentences == 0:
        return {"fog_index": None, "avg_sentence_length": None,
                "complex_word_ratio": None, "n_words": 0.0, "n_sentences": 0.0}

    complex_words = sum(1 for t in tokens if is_complex(t))
    avg_sentence_length = n_words / n_sentences
    complex_ratio = complex_words / n_words
    return {
        "fog_index": 0.4 * (avg_sentence_length + 100.0 * complex_ratio),
        "avg_sentence_length": avg_sentence_length,
        "complex_word_ratio": complex_ratio,
        "n_words": float(n_words),
        "n_sentences": float(n_sentences),
    }
