"""Perplexity under a healthy-firm language model.

Issue 4 of the project plan: "perplexity" needs a reference model, or the number
means nothing. The reference here is an n-gram model trained on the MD&A of
**healthy firms only**, so perplexity reads as *departure from normal
disclosure* rather than as a property of English.

Interpolated Kneser-Ney, implemented here because ``nltk`` is not a dependency
of this project. Kneser-Ney rather than backoff or absolute discounting because
its lower orders use **continuation counts** -- in how many distinct contexts a
word appears, rather than how often it appears. That is what stops a word like
"crore", frequent but almost always in the same phrase, from looking like a good
guess everywhere. Getting that wrong turns the model into plain absolute
discounting with a Kneser-Ney unigram, which is a different model with the same
name, so the continuation counts are computed explicitly in :func:`fit` and used
at every order below the highest.

Leakage
-------
The model must be fitted on the **training folds only**. Fitting it on every
healthy report and then scoring the test fold leaks the test distribution into
training exactly as a globally fitted scaler would. Phase 4 therefore refuses to
invent a training set: :func:`fit` takes the documents it is given, and the
feature run leaves ``perplexity`` empty with a reason when no training set was
named. Phase 5/6 fits one model per fold. See ``docs/decisions_log.md``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from bpp.nlp.lexicon import words

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


@dataclass
class KneserNeyLM:
    """An interpolated Kneser-Ney n-gram model."""

    order: int = 3
    discount: float = 0.75
    vocabulary: set[str] = field(default_factory=set)
    n_documents: int = 0

    #: counts[k] maps a k-gram to how often it occurred (used at the highest order)
    counts: dict[int, Counter] = field(default_factory=dict)
    #: followers[k][ctx] = how many distinct words follow this k-gram context
    followers: dict[int, dict[tuple, int]] = field(default_factory=dict)

    #: continuation counts, used at every order below the highest.
    #: cont_counts[k][gram] = in how many distinct contexts this k-gram appears
    cont_counts: dict[int, dict[tuple, int]] = field(default_factory=dict)
    #: cont_total[k][ctx]  = the same, summed over every word following ctx
    cont_total: dict[int, dict[tuple, int]] = field(default_factory=dict)
    #: cont_followers[k][ctx] = how many distinct words follow ctx in continuation space
    cont_followers: dict[int, dict[tuple, int]] = field(default_factory=dict)

    total_unigrams: int = 0
    total_bigram_types: int = 0
    #: denominator of the unigram continuation distribution, including one unit
    #: of mass for every vocabulary item that never appears as a continuation
    unigram_denominator: int = 0

    @property
    def fitted(self) -> bool:
        return bool(self.vocabulary)


def _prepare(tokens: Sequence[str], order: int) -> list[str]:
    return [BOS] * (order - 1) + list(tokens) + [EOS]


def fit(texts: Iterable[str], order: int = 3, discount: float = 0.75,
        min_count: int = 2) -> KneserNeyLM:
    """Train on the given documents.

    Words seen fewer than ``min_count`` times become ``<unk>``, so unseen words
    at scoring time have real probability mass rather than falling off the model.
    """
    order = max(1, int(order))
    documents = [words(t) for t in texts if t]
    documents = [d for d in documents if d]
    model = KneserNeyLM(order=order, discount=discount, n_documents=len(documents))
    if not documents:
        return model

    frequency = Counter(token for document in documents for token in document)
    vocabulary = {w for w, c in frequency.items() if c >= min_count}
    if not vocabulary:                      # a very small fold: keep everything
        vocabulary = set(frequency)
    vocabulary |= {BOS, EOS, UNK}
    model.vocabulary = vocabulary

    counts: dict[int, Counter] = {k: Counter() for k in range(1, order + 1)}
    followers: dict[int, dict[tuple, set]] = defaultdict(lambda: defaultdict(set))
    preceders: dict[int, dict[tuple, set]] = defaultdict(lambda: defaultdict(set))

    for document in documents:
        padded = _prepare([w if w in vocabulary else UNK for w in document], order)
        for k in range(1, order + 1):
            for i in range(len(padded) - k + 1):
                gram = tuple(padded[i:i + k])
                counts[k][gram] += 1
                if k > 1:
                    followers[k - 1][gram[:-1]].add(gram[-1])
                    preceders[k][gram[1:]].add(gram[0])

    model.counts = counts
    model.followers = {k: {ctx: len(s) for ctx, s in d.items()} for k, d in followers.items()}
    model.total_unigrams = sum(counts[1].values())
    model.total_bigram_types = len(counts[2]) if order >= 2 else 0

    # Continuation counts for every order below the highest. preceders[k + 1] is
    # keyed by k-grams, so it is exactly N1+(* gram) for those k-grams.
    for k in range(1, order):

        gram_counts = {gram: len(s) if isinstance(s, set) else s
                       for gram, s in preceders.get(k + 1, {}).items()}
        totals: dict[tuple, int] = defaultdict(int)
        distinct: dict[tuple, int] = defaultdict(int)
        for gram, n in gram_counts.items():
            context = gram[:-1]
            totals[context] += n
            if n > 0:
                distinct[context] += 1
        model.cont_counts[k] = gram_counts
        model.cont_total[k] = dict(totals)
        model.cont_followers[k] = dict(distinct)

    # Words that never appear as a continuation (<unk> in a corpus where nothing
    # was rare enough to be folded into it) still need mass. Reserving one unit
    # each *in the denominator* keeps the unigram distribution summing to 1;
    # adding a floor on top of an already-normalised distribution does not.
    unigram_counts = model.cont_counts.get(1, {})
    never_continued = sum(1 for w in model.vocabulary if unigram_counts.get((w,), 0) == 0)
    model.unigram_denominator = model.total_bigram_types + never_continued
    return model


def probability(model: KneserNeyLM, word: str, context: tuple[str, ...] = ()) -> float:
    """P(word | context) under interpolated Kneser-Ney."""
    if not model.fitted:
        return 0.0
    word = word if word in model.vocabulary else UNK
    context = tuple(w if w in model.vocabulary else UNK for w in context)
    context = context[-(model.order - 1):] if model.order > 1 else ()
    return _probability(model, word, context)


def _probability(model: KneserNeyLM, word: str, context: tuple[str, ...]) -> float:
    k = len(context) + 1

    if k == 1:
        return _unigram(model, word)

    if k >= model.order:
        context_total = model.counts.get(k - 1, Counter()).get(context, 0)
        gram_count = model.counts.get(k, Counter()).get(context + (word,), 0)
        distinct_followers = model.followers.get(k - 1, {}).get(context, 0)
    else:
        context_total = model.cont_total.get(k, {}).get(context, 0)
        gram_count = model.cont_counts.get(k, {}).get(context + (word,), 0)
        distinct_followers = model.cont_followers.get(k, {}).get(context, 0)

    lower = _probability(model, word, context[1:])
    if context_total <= 0:
        return lower                       # nothing observed here; defer entirely

    discounted = max(gram_count - model.discount, 0.0) / context_total
    # With no distinct followers the discounted mass is zero for every word, so
    # all of it must pass to the lower order or the distribution leaks away.
    weight = (model.discount * distinct_followers / context_total) if distinct_followers else 1.0
    return discounted + weight * lower


def _unigram(model: KneserNeyLM, word: str) -> float:
    """The base case.

    For a model of order 2 or more this is the Kneser-Ney continuation
    probability: in how many distinct contexts does the word appear, rather than
    how often. For a unigram model there is no context to continue, so it falls
    back to the maximum-likelihood estimate -- otherwise every document would
    score the same perplexity.
    """
    if model.order == 1:
        total = model.total_unigrams or 1
        count = model.counts.get(1, Counter()).get((word,), 0)
        if count == 0:
            return 1.0 / (total + len(model.vocabulary))
        return count / total

    if model.total_bigram_types == 0:
        return 1.0 / max(1, len(model.vocabulary))
    denominator = model.unigram_denominator or model.total_bigram_types
    # a word never seen as a continuation holds the one unit reserved for it
    return max(model.cont_counts.get(1, {}).get((word,), 0), 1) / denominator


def perplexity(model: KneserNeyLM, text: str) -> float | None:
    """Perplexity of one document. Lower = more like the training corpus."""
    if not model.fitted:
        return None
    tokens = words(text)
    if not tokens:
        return None
    padded = _prepare([w if w in model.vocabulary else UNK for w in tokens], model.order)
    start = model.order - 1

    log_sum, n = 0.0, 0
    for i in range(start, len(padded)):
        context = tuple(padded[max(0, i - model.order + 1):i])
        p = _probability(model, padded[i], context)
        if p <= 0:
            p = 1e-12
        log_sum += math.log(p)
        n += 1
    if n == 0:
        return None
    return math.exp(-log_sum / n)
