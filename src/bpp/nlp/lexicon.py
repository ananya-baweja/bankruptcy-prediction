"""Word lists and the counts built from them: tone, hedging, distress phrases.

Three sources of words, kept apart on purpose:

* **Loughran-McDonald** -- the finance-specific sentiment dictionary. General
  word lists misclassify financial text ("liability", "cost" and "capital" are
  not negative in a balance sheet), which is the finding the project cites
  Loughran & McDonald (2011) for. The dictionary is *not* vendored into the
  repo: it is versioned and updated annually, and pinning a silent copy would
  make a result irreproducible. The team downloads it once into
  ``data/manual/`` -- see ``docs/07_phase4_language.md``.
* **Hedging** -- modal and approximating language. This list lives in the code
  because it is small, stable, and specific to the claim the project is
  testing.
* **Indian distress phrases** -- a seed list only. Contribution C of the project
  plan is to *mine* this list from the reports themselves with a log-odds
  prior; until then these few phrases stand in, and the feature is marked
  ``seed`` so nobody reports it as the mined lexicon.

Counting convention
-------------------
Tone ratios use the **raw** token stream, so the denominator is the document's
full word count. That is Loughran and McDonald's own methodology, and ratios
computed against a stopword-stripped denominator are not comparable with any
published figure -- including the 2.21% vs 1.30% in Gupta & Banerjee (2023),
which this project sets out to beat. The cleaned track is for n-grams and word
vectors, where stopword removal helps.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

#: the Loughran-McDonald categories this project uses
LM_CATEGORIES = ["negative", "positive", "uncertainty", "litigious",
                 "strong_modal", "weak_modal", "constraining"]

# In the Master Dictionary each category column holds the *year* a word entered
# that category, and 0 means "not a member" -- not a score.
_LM_COLUMN_ALIASES = {
    "negative": "negative",
    "positive": "positive",
    "uncertainty": "uncertainty",
    "litigious": "litigious",
    "strong_modal": "strong_modal",
    "strongmodal": "strong_modal",
    "weak_modal": "weak_modal",
    "weakmodal": "weak_modal",
    "constraining": "constraining",
}

#: Hedging: the words that say "we are not committing to this". Kept here
#: rather than in the LM dictionary because weak-modal is narrower than hedging.
HEDGE_TERMS = frozenset({
    "may", "might", "could", "would", "should", "possibly", "possible", "perhaps",
    "probable", "probably", "likely", "unlikely", "appear", "appears", "appeared",
    "seem", "seems", "seemed", "suggest", "suggests", "indicate", "indicates",
    "believe", "believes", "believed", "expect", "expects", "expected", "anticipate",
    "anticipates", "anticipated", "assume", "assumes", "assumed", "estimate",
    "estimates", "estimated", "approximately", "roughly", "around", "substantially",
    "generally", "largely", "partially", "somewhat", "relatively", "potential",
    "potentially", "tentative", "uncertain", "pending", "presumably",
})   # NB "subject" is deliberately absent: it is counted as the phrase "subject to",
     # and having it in both lists counts one hedge twice.

#: Multi-word hedges, matched on the raw text
HEDGE_PHRASES = (
    "subject to", "to the extent", "in the event that", "no assurance",
    "cannot be assured", "cannot assure", "depending upon", "depending on",
    "in due course", "at this stage", "as and when",
)   # NB "there can be no" was removed: it overlaps "no assurance", and
    # "there can be no assurance that..." is one hedge, not two.

#: India/IBC distress phrases. A SEED list -- contribution C mines the real one.
DISTRESS_PHRASES_SEED = (
    "one time settlement", "corporate debt restructuring", "strategic debt restructuring",
    "non performing asset", "wilful defaulter", "willful defaulter",
    "default in repayment", "defaulted in repayment", "delay in repayment",
    "invoked the pledge", "pledge invoked", "recall notice", "sarfaesi",
    "debt recovery tribunal", "asset reconstruction company",
    "resolution plan", "net worth has eroded", "erosion of net worth",
    "sick industrial", "restructuring of debt", "moratorium under section",
    "insolvency resolution", "failed to pay", "unable to service",
)

# LM tokenises on runs of letters, so a hyphenated compound becomes two words.
# That matters here: "non-performing" as a single token can never be in the
# dictionary, and it is exactly the vocabulary this project is looking for.
_WORD = re.compile(r"[A-Za-z]+")
#: a hyphen at a line break is PDF layout, not part of the word ("manage-\nment")
_LINE_BREAK_HYPHEN = re.compile(r"-\s*\n\s*")
#: sentence end: terminal punctuation followed by something that starts a sentence
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"\'(\[]?[A-Z0-9])")
#: a list marker starts a new unit of text for readability purposes
_BULLET = re.compile(
    r"^\s*(?:[-\u2022\u00b7*\u25cf\u25aa]|\(?[ivxlc]{1,4}\)|\(?[a-z]\)|\d{1,2}[.)])\s+", re.I)

#: Stopwords for the CLEANED track only. Never applied before tone or hedging:
#: "may", "could" and "should" are stopwords *and* are the hedging being measured.
CLEANED_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "as", "at", "by",
    "for", "from", "in", "into", "of", "on", "to", "with", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "it", "its", "this",
    "that", "these", "those", "which", "who", "whom", "we", "our", "us", "they",
    "their", "them", "he", "she", "his", "her", "not", "no", "so", "such", "also",
})


@dataclass
class Lexicon:
    """The word lists one feature run uses, plus where they came from."""

    lm: dict[str, frozenset[str]] = field(default_factory=dict)
    lm_source: str = ""
    lm_version: str = ""
    hedge_terms: frozenset[str] = HEDGE_TERMS
    hedge_phrases: tuple[str, ...] = HEDGE_PHRASES
    distress_phrases: tuple[str, ...] = DISTRESS_PHRASES_SEED
    distress_source: str = "seed"

    @property
    def has_lm(self) -> bool:
        return any(self.lm.get(c) for c in LM_CATEGORIES)


def load_lm_dictionary(path: Path | str | None) -> Lexicon:
    """Read the Loughran-McDonald dictionary, if the team has supplied it.

    Two formats are accepted:

    * the **Master Dictionary** as published: one row per word, a column per
      category holding the year the word entered it (0 = not a member);
    * a **simple** two-column ``word,category`` file, which is what a test or a
      hand-trimmed list looks like.

    A missing file is not an error. The run continues without tone features and
    says so, because a silent zero would look like a document with no negative
    words rather than a missing dictionary.
    """
    lexicon = Lexicon()
    if path is None:
        return lexicon
    path = Path(path)
    if not path.exists():
        log.warning("Loughran-McDonald dictionary not found at %s -- tone features "
                    "will be empty. See docs/07_phase4_language.md.", path)
        return lexicon

    buckets: dict[str, set[str]] = {c: set() for c in LM_CATEGORIES}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [f.strip().lower().replace(" ", "_") for f in (reader.fieldnames or [])]
        reader.fieldnames = fieldnames
        simple = "category" in fieldnames
        for row in reader:
            word = (row.get("word") or "").strip().lower()
            if not word:
                continue
            if simple:
                category = (row.get("category") or "").strip().lower().replace(" ", "_")
                category = _LM_COLUMN_ALIASES.get(category)
                if category:
                    buckets[category].add(word)
                continue
            for column, category in _LM_COLUMN_ALIASES.items():
                raw = row.get(column)
                if raw is None:
                    continue
                try:
                    # non-zero = the year the word entered the category
                    if float(str(raw).strip() or 0) != 0:
                        buckets[category].add(word)
                except ValueError:
                    continue

    lexicon.lm = {c: frozenset(w) for c, w in buckets.items()}
    lexicon.lm_source = str(path)
    lexicon.lm_version = _version_from_name(path.name)
    if not lexicon.has_lm:
        log.warning("%s parsed but no category words were found -- check the format", path)
    else:
        log.info("Loughran-McDonald: %s", {c: len(lexicon.lm[c]) for c in LM_CATEGORIES})
    return lexicon


def _version_from_name(name: str) -> str:
    """The dictionary is republished yearly; the filename carries the year.

    The canonical file is ``LoughranMcDonald_MasterDictionary_1993-2021.csv``, a
    *range*, so the last year is the version -- taking the first records 1993 for
    every release ever made.
    """
    years = re.findall(r"(?:19|20)\d{2}", name)
    return years[-1] if years else ""


# --------------------------------------------------------------------------- tokens
def words(text: str) -> list[str]:
    """The raw token stream: every alphabetic word, lowercased.

    This is the denominator for every ratio, so it must not be filtered.
    """
    text = _LINE_BREAK_HYPHEN.sub("", text or "")
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def cleaned_words(text: str) -> list[str]:
    """The cleaned track: stopwords removed. For n-grams and word vectors only."""
    return [w for w in words(text) if w not in CLEANED_STOPWORDS and len(w) > 1]


def sentences(text: str) -> list[str]:
    """Split into sentences, counting headings and bullets as their own units.

    Collapsing all whitespace first would be a mistake: an MD&A extracted from a
    PDF is largely headings and bullet points with no terminal punctuation, so a
    splitter that needs a full stop returns *one* sentence for the whole section.
    Average sentence length then equals the word count and the Fog index reads
    ~40 for every such document -- worst exactly where extraction is worst.

    Deliberately simple otherwise: "Rs." and "Ltd." are common enough that a
    cleverer splitter mis-fires more often than it helps, and readability only
    needs the count.
    """
    if not text or not text.strip():
        return []
    units: list[str] = []
    for line in _LINE_BREAK_HYPHEN.sub("", text).split("\n"):
        line = _BULLET.sub("", line.strip())
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line:
            continue
        units.extend(p.strip() for p in _SENTENCE_SPLIT.split(line) if p.strip())
    return units


# --------------------------------------------------------------------------- counts
def _phrase_pattern(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """One alternation, longest phrase first, so matches do not overlap.

    ``finditer`` returns non-overlapping matches, so "there can be no assurance"
    counts once rather than once per sub-phrase.
    """
    ordered = sorted({p.lower() for p in phrases}, key=len, reverse=True)
    return re.compile("|".join(re.escape(p) for p in ordered)) if ordered else re.compile(r"(?!)")


def normalise_for_phrases(text: str) -> str:
    """Lowercase, and make hyphenation irrelevant.

    "non-performing asset" and "non performing asset" are the same phrase; listing
    both spellings would count one concept twice in ``distress_phrase_distinct``.
    """
    text = _LINE_BREAK_HYPHEN.sub("", text or "").lower()
    return re.sub(r"[\s\-\u2010-\u2015]+", " ", text)


def tone_ratios(tokens: list[str], lexicon: Lexicon) -> dict[str, float | None]:
    """Share of the document's words in each Loughran-McDonald category.

    Returns ``None`` per category when the dictionary is absent, so a missing
    dictionary never looks like a document with no negative words.
    """
    total = len(tokens)
    if not lexicon.has_lm or total == 0:
        return {f"lm_{c}_ratio": None for c in LM_CATEGORIES}
    counts = {c: 0 for c in LM_CATEGORIES}
    for token in tokens:
        for category in LM_CATEGORIES:
            if token in lexicon.lm.get(category, ()):
                counts[category] += 1
    return {f"lm_{c}_ratio": counts[c] / total for c in LM_CATEGORIES}


def hedging(text: str, tokens: list[str], lexicon: Lexicon) -> dict[str, float | None]:
    """Hedging density over the raw text: single words plus multi-word hedges."""
    total = len(tokens)
    if total == 0:
        return {"hedge_density": None, "hedge_distinct": None, "hedge_phrase_density": None}
    hits = [t for t in tokens if t in lexicon.hedge_terms]
    phrase_hits = len(_phrase_pattern(lexicon.hedge_phrases)
                      .findall(normalise_for_phrases(text)))
    return {
        "hedge_density": len(hits) / total,
        "hedge_distinct": float(len(set(hits))),
        "hedge_phrase_density": phrase_hits / total,
    }


def distress_phrases(text: str, tokens: list[str], lexicon: Lexicon) -> dict[str, float | None]:
    """Density of the India/IBC distress phrases. Seed list -- see the module docstring."""
    total = len(tokens)
    if total == 0:
        return {"distress_phrase_density": None, "distress_phrase_distinct": None}
    found = _phrase_pattern(lexicon.distress_phrases).findall(normalise_for_phrases(text))
    return {
        "distress_phrase_density": len(found) / total,
        "distress_phrase_distinct": float(len(set(found))),
    }
