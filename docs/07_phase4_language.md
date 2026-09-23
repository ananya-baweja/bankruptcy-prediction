# Phase 4 — language features, stream B (`bpp language-features`)

Turns the sections Phase 2 extracted into the engineered linguistic features of Table 7, stream B:
one row per report. **Stream A (FinBERT sentence vectors) is not here** — it needs torch and a GPU
and belongs in the Colab notebook. This is the part that runs on a laptop, and the part whose
numbers can be checked by hand.

```powershell
bpp language-features
bpp language-features --lm-train-doc-id BSE900201_FY2017 BSE900201_FY2018   # for perplexity
```

Phase 2 must have run first: this reads `data/interim/sections/<doc_id>.json`.

---

## Step 4.1 — Before you start: the dictionary

Tone is computed with the **Loughran-McDonald** finance dictionary, because general sentiment lists
misclassify financial text — "liability", "cost" and "capital" are not negative on a balance sheet.
That is the finding this project cites Loughran & McDonald (2011) for, so using a general list would
undercut the whole stream.

The dictionary is **not in the repo**. It is republished every year, and a silent copy would make a
tone number impossible to reproduce. Download it once:

1. Go to `https://sraf.nd.edu/loughranmcdonald-master-dictionary/`
2. Download the Master Dictionary CSV
3. Save it as **`data/manual/loughran_mcdonald.csv`**
4. Record the year you downloaded in `decisions_log.md` — tone ratios are not comparable across
   dictionary versions

Both formats are accepted: the Master Dictionary as published (a column per category holding the
year the word entered it, `0` = not a member), and a simple `word,category` file.

**Without the dictionary the run still works** — readability, hedging, the auditor flags and drift
are all computed — and the tone columns are left *empty*, not zero. A zero would read as a document
containing no negative words, which is a very different claim from "we have no dictionary".

## Step 4.2 — What is computed

Per section, for the **MD&A** and the **auditor's report** separately (`mdna_*`, `auditor_*`):

| Group | Columns |
| --- | --- |
| Tone | `lm_negative_ratio`, `lm_positive_ratio`, `lm_uncertainty_ratio`, `lm_litigious_ratio`, `lm_strong_modal_ratio`, `lm_weak_modal_ratio`, `lm_constraining_ratio` |
| Hedging | `hedge_density`, `hedge_distinct`, `hedge_phrase_density` |
| Distress phrases | `distress_phrase_density`, `distress_phrase_distinct` |
| Readability | `fog_index`, `avg_sentence_length`, `complex_word_ratio`, `n_words`, `n_sentences` |

The two sections are kept apart because Phase 7 asks **MD&A only vs auditor only vs both**, and that
ablation is impossible if they are averaged together first. Issue 6 of the project plan expects the
auditor's report to carry the strongest Indian signal.

Per document:

| Group | Columns |
| --- | --- |
| Auditor's statements | `has_going_concern`, `has_emphasis_of_matter`, `has_modified_opinion_basis`, `audit_opinion_severity` (0 unmodified → 3 disclaimer), `audit_opinion_found`, `caro_default_flag`, `caro_statutory_dues_flag` |
| Coverage | `has_mdna`, `has_auditor_report`, `has_caro_annexure` |
| Leakage counts | `ibc_generic_mentions`, `cirp_specific_mentions` (from Phase 2) |
| Reference model | `perplexity`, `perplexity_source`, `perplexity_in_training` |
| Drift | `drift_jaccard`, `drift_cosine`, `drift_new_word_share`, `drift_length_ratio`, `drift_negative_delta`, `drift_hedge_delta`, `drift_fog_delta`, `has_previous_year` |

### The 25-value core vector

`CORE_LANGUAGE_FEATURES` in `nlp/features.py` names the 25 values the model is trained on, matching
Table 7 and the `25 → 64 → 64` layer in the progress report. The table carries more columns than
that on purpose — Phase 7 needs them for the ablations — but with ~100 insolvent firms, feeding all
of them into a 64-unit layer is the overfitting the project plan warns about. **The model code
should read `CORE_LANGUAGE_FEATURES`, not hard-code a width.**

### Two text tracks

Issue 3 of the project plan: standard cleaning removes stopwords, but "may", "could" and "subject
to" *are* stopwords and *are* the hedging being measured. So:

- **raw** — every word. Tone ratios, hedging, readability, perplexity.
- **cleaned** — stopwords removed. Drift similarity, and the n-grams and word vectors of later steps.

Tone ratios use the raw word count as the denominator. That is Loughran and McDonald's own
methodology; a ratio computed against a stopword-stripped denominator is not comparable with any
published figure, including the 2.21% vs 1.30% in Gupta & Banerjee (2023) that this project sets out
to beat.

## Step 4.3 — Perplexity needs a reference, and a fold

Issue 4 of the plan: "perplexity" is meaningless without a reference model. Here it is an
interpolated **Kneser-Ney** n-gram model trained on the MD&A of **healthy firms**, so the score
reads as *departure from normal disclosure*. Kneser-Ney rather than backoff because its lower orders
use continuation counts — how many different contexts a word appears in — which stops a word like
"crore", frequent but always in the same phrase, from looking like a good guess everywhere.

It is implemented in `nlp/lm.py` rather than taken from `nltk`, which is not a dependency.

**The model must be fitted on the training folds only.** Fitting it on every healthy report and then
scoring the test fold leaks the test distribution into training exactly as a globally fitted scaler
would. So `bpp language-features` will not invent a training set: without `--lm-train-doc-id` the
`perplexity` column is left empty and `perplexity_source` says why. Phase 5/6 fits one model per
fold. Any document that was itself in the training set is marked `perplexity_in_training = 1`, since
a model fitted on a document flatters it.

## Step 4.4 — Drift is the contribution, not the levels

Contribution A. Cohen, Malloy and Nguyen ("Lazy Prices", 2020) found that *changes* in a filing
predict outcomes while levels do not: companies copy last year's report forward, and the paragraphs
they rewrite are the ones under pressure.

So `drift_*` compares a firm's MD&A with **the same firm's previous year** — never with its peer.
A firm's first year has no drift: the columns are empty and `has_previous_year` is `False`. A zero
would read as "wrote exactly the same thing again", which is the opposite of the truth.

Similarity is on the cleaned track, or two reports are ~40% alike on "the" and "of" before anything
is said. Cosine uses plain term frequencies rather than TF-IDF, because an IDF term fitted across
all years would leak the test fold into training.

## Step 4.5 — Text handling that is not obvious

**Tokens are runs of letters.** "non-performing" becomes two words, which is how Loughran and
McDonald tokenise and the only way the compound can match the dictionary at all. A hyphen at a line
break ("manage-\nment") is rejoined first, or every PDF-extracted word count is inflated.

**Line breaks end sentences.** A PDF-extracted MD&A is largely headings and bullets with no terminal
punctuation. A splitter that needs a full stop returns *one* sentence for the whole section, so
average sentence length equals the word count and the Fog index reads ~38 for every such document —
worst exactly where extraction is worst.

**Syllables are counted by rule**, not by dictionary, with two corrections that matter for corporate
prose: "-tion"/"-sion" is one syllable (otherwise every "operation" and "provision" gains one), and a
silent final "e" is only dropped when it stands alone (otherwise "revenue" loses a syllable). A short
exception list covers a handful of very frequent words the rule still gets wrong.

## Step 4.6 — Negation, and why the CARO flags are not regexes alone

Every clean annexure contains the sentence *"the Company has **not** defaulted in repayment of loans
or borrowings to any lender"*. A flag that matches `defaulted in repayment` is therefore **1 for
every company in the sample** — it looks like a working feature in the output table and carries
exactly no information.

`caro_default_flag` checks each occurrence for a negation in the words before it, and counts the
clause only where the auditor is asserting the failure rather than denying it. `caro_statutory_dues_flag`
is the mirror image: there the *negated* form ("has **not** been regular in depositing") is the
failure, so it is matched explicitly.

This is the kind of error to look for everywhere in this phase: a feature that is constant across
the sample. Check every new flag by grouping on `label` and confirming the two classes differ.

## Outputs

| File | Content |
| --- | --- |
| `processed/language_features.csv` | one row per report: everything above, plus `pair_id`, `role`, `label`, `horizon` carried from `documents_labeled.csv` |

## Not in this phase

Stream A (FinBERT sentence vectors), NER masking, coreference, SVO distress triplets, the Lesk
ablation, Word2Vec/GloVe embeddings and the mined India-specific distress lexicon (contribution C).
They need spaCy, transformers and torch, and run on Colab. The distress phrase list used here is a
**seed** — `distress_source = "seed"` — and must not be reported as the mined lexicon.

---

### Team checklist for Phase 4 (stream B)

- [ ] Loughran-McDonald dictionary downloaded to `data/manual/`, version noted in `decisions_log.md`
- [ ] `bpp language-features` run; coverage line checked for features that could not be computed
- [ ] **Every flag grouped by `label` and confirmed to differ between the classes** — a constant
      feature is the failure mode here
- [ ] Tone difference between classes compared with Gupta & Banerjee's 2.21% vs 1.30%
- [ ] `processed/language_features.csv` uploaded to the shared Drive for Phases 5–6 on Colab
