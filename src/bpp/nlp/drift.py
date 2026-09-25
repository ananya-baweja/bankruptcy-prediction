"""Year-on-year change in how a company writes about itself.

Contribution A of the project plan, and the one idea here that is not in the
Indian literature: Cohen, Malloy and Nguyen ("Lazy Prices", 2020) found that
*changes* in a company's filing predict outcomes, while the level does not.
Companies copy last year's report forward; the paragraphs they rewrite are the
ones under pressure.

So the features are comparisons, not levels: how much of last year's MD&A
survived, and which way the tone and length moved. A firm with no prior report
gets ``None`` and a flag -- never a zero, which would read as "wrote exactly the
same thing again", the opposite of the truth.

Similarity is computed on the cleaned track (stopwords out), because otherwise
two documents are ~40% similar on "the" and "of" before anything is said.
Cosine is on plain term frequencies rather than TF-IDF: an IDF term would have
to be fitted on a corpus, and fitting it across all years leaks the test fold
into training exactly as a scaler would.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from bpp.nlp.lexicon import cleaned_words

#: what a drift row looks like when there is no prior year to compare with
NO_PREVIOUS = {
    "drift_jaccard": None,
    "drift_cosine": None,
    "drift_new_word_share": None,
    "drift_length_ratio": None,
    "drift_negative_delta": None,
    "drift_hedge_delta": None,
    "drift_fog_delta": None,
    "has_previous_year": False,
}


def jaccard(current: Sequence[str], previous: Sequence[str]) -> float | None:
    """Share of the combined vocabulary that both years use."""
    a, b = set(current), set(previous)
    if not a and not b:
        return None
    union = a | b
    return len(a & b) / len(union) if union else None


def cosine(current: Sequence[str], previous: Sequence[str]) -> float | None:
    """Cosine similarity of the two term-frequency vectors."""
    a, b = Counter(current), Counter(previous)
    if not a or not b:
        return None
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return None
    return dot / (norm_a * norm_b)


def new_word_share(current: Sequence[str], previous: Sequence[str]) -> float | None:
    """Share of this year's vocabulary that was not used last year.

    The directional half of the Lazy Prices idea: what management started
    saying, as opposed to how much overlap there is.
    """
    a, b = set(current), set(previous)
    if not a:
        return None
    return len(a - b) / len(a)


def _delta(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None:
        return None
    return current - previous


def drift_features(current_text: str, previous_text: str | None,
                   current_values: dict[str, float | None] | None = None,
                   previous_values: dict[str, float | None] | None = None) -> dict:
    """Compare one company-year's MD&A with the same company's previous year."""
    if not previous_text or not current_text:
        return dict(NO_PREVIOUS)

    current_tokens = cleaned_words(current_text)
    previous_tokens = cleaned_words(previous_text)
    if not current_tokens or not previous_tokens:
        return dict(NO_PREVIOUS)

    current_values = current_values or {}
    previous_values = previous_values or {}

    length_ratio = None
    if previous_tokens:
        length_ratio = len(current_tokens) / len(previous_tokens)

    return {
        "drift_jaccard": jaccard(current_tokens, previous_tokens),
        "drift_cosine": cosine(current_tokens, previous_tokens),
        "drift_new_word_share": new_word_share(current_tokens, previous_tokens),
        "drift_length_ratio": length_ratio,
        "drift_negative_delta": _delta(current_values.get("lm_negative_ratio"),
                                       previous_values.get("lm_negative_ratio")),
        "drift_hedge_delta": _delta(current_values.get("hedge_density"),
                                    previous_values.get("hedge_density")),
        "drift_fog_delta": _delta(current_values.get("fog_index"),
                                  previous_values.get("fog_index")),
        "has_previous_year": True,
    }
