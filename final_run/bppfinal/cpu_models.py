"""Every model that does not need a GPU, as out-of-fold predictions on the frozen folds.

All preprocessing that learns from data (medians, winsorising limits, scaling, TF-IDF vocabulary,
the mined lexicon) is fitted on the training folds only, exactly as notebook 2 does for the network.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

AUD_FLAGS = ["has_going_concern", "has_emphasis_of_matter", "has_modified_opinion_basis",
             "audit_opinion_severity", "caro_default_flag", "caro_statutory_dues_flag"]


def feature_sets(R, L):
    lang_text = [c for c in L if c != "financials_missing"]
    drift = [c for c in L if c.startswith("drift_")]
    mdna = [c for c in L if c.startswith("mdna_")] + ["has_mdna"] + drift + ["perplexity_fold"]
    aud = ([c for c in L if c.startswith("auditor_")] + AUD_FLAGS +
           ["audit_opinion_found", "has_auditor_report", "has_caro_annexure"])
    assert set(mdna) | set(aud) == set(lang_text), set(lang_text) - set(mdna) - set(aud)
    return dict(ratios=R, lang=L, lang_text=lang_text, mdna=mdna, aud=aud, flags=AUD_FLAGS, drift=drift)


def _num(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def hgb_oof(df, cols, rows=None, seed=0):
    X, y, F = _num(df, cols), df.label.to_numpy(float), df.fold.to_numpy()
    use = np.ones(len(df), bool) if rows is None else rows
    oof = np.full(len(df), np.nan)
    for k in range(5):
        tr, te = (F != k) & use, (F == k) & use
        oof[te] = HistGradientBoostingClassifier(random_state=seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def _lin_prep(Xtr, Xte):
    """Winsorise at the training 1st/99th percentiles, fill with the training median, add
    missing-value indicators, standardise with training mean/sd."""
    lo, hi = np.nanpercentile(Xtr, 1, 0), np.nanpercentile(Xtr, 99, 0)
    med = np.nanmedian(Xtr, 0)
    out = []
    for X in (Xtr, Xte):
        Z = np.clip(X, lo, hi)
        miss = np.isnan(Z)
        Z = np.where(miss, med, Z)
        out.append((Z, miss.astype(float)))
    keep_miss = out[0][1].std(0) > 0
    A = np.hstack([out[0][0], out[0][1][:, keep_miss]])
    B = np.hstack([out[1][0], out[1][1][:, keep_miss]])
    A = np.nan_to_num(A); B = np.nan_to_num(B)
    mu, sd = A.mean(0), A.std(0); sd[sd < 1e-9] = 1
    return (A - mu) / sd, (B - mu) / sd


def logit_oof(df, cols, C=1.0, rows=None):
    X, y, F = _num(df, cols), df.label.to_numpy(float), df.fold.to_numpy()
    use = np.ones(len(df), bool) if rows is None else rows
    oof = np.full(len(df), np.nan)
    for k in range(5):
        tr, te = (F != k) & use, (F == k) & use
        A, B = _lin_prep(X[tr], X[te])
        oof[te] = LogisticRegression(C=C, max_iter=5000).fit(A, y[tr]).predict_proba(B)[:, 1]
    return oof


def altman_oof(df):
    """The classical benchmark: no training. Score = -Z'' (lower Z = higher risk); a missing Z gets
    the training folds' median, i.e. a neutral score."""
    z = pd.to_numeric(df.altman_z_em, errors="coerce").to_numpy(float) - 3.25   # Z'' = Z''(EM) - 3.25
    F = df.fold.to_numpy(); oof = np.full(len(df), np.nan)
    for k in range(5):
        tr, te = F != k, F == k
        zz = np.where(np.isnan(z[te]), np.nanmedian(z[tr]), z[te])
        oof[te] = -zz
    return oof


# ----------------------------------------------------------------------------- text models
_NUM = re.compile(r"\d[\d,.]*")


def clean_text(t):
    return _NUM.sub(" 0 ", (t or "").lower())


def tfidf_oof(df, section, C=1.0, min_firms=5):
    """TF-IDF on unigrams and bigrams + logistic regression, text only. A term must appear in the
    reports of at least `min_firms` different training companies, which keeps company and place
    names out of the vocabulary."""
    y, F = df.label.to_numpy(float), df.fold.to_numpy()
    docs = df[section].map(clean_text).to_numpy()
    has = df[section].str.len().to_numpy() > 0
    oof = np.full(len(df), np.nan)
    for k in range(5):
        tr, te = (F != k), (F == k)
        vocab = firm_vocab(docs[tr & has], df.firm_id.to_numpy()[tr & has], min_firms)
        vec = TfidfVectorizer(vocabulary=vocab, ngram_range=(1, 2), sublinear_tf=True,
                              token_pattern=r"(?u)\b[a-z][a-z]+\b")
        A = vec.fit_transform(docs[tr]); B = vec.transform(docs[te])
        oof[te] = LogisticRegression(C=C, max_iter=5000).fit(A, y[tr]).predict_proba(B)[:, 1]
    return oof


def firm_vocab(docs, firms, min_firms, ngram=(1, 2)):
    cv = CountVectorizer(ngram_range=ngram, token_pattern=r"(?u)\b[a-z][a-z]+\b", binary=True, min_df=3)
    X = cv.fit_transform(docs)
    terms = np.array(cv.get_feature_names_out())
    f_codes, f_idx = np.unique(firms, return_inverse=True)
    # number of distinct firms using each term
    import scipy.sparse as sp
    M = sp.csr_matrix((np.ones(len(f_idx)), (f_idx, np.arange(len(f_idx)))), shape=(len(f_codes), len(f_idx)))
    firm_term = (M @ X) > 0
    n_firms = np.asarray(firm_term.sum(0)).ravel()
    return sorted(terms[n_firms >= min_firms].tolist())


def log_odds(docs_a, docs_b, vocab, prior_scale=None):
    """Monroe, Colaresi & Quinn (2008): log-odds ratio with an informative Dirichlet prior.
    Returns z-scores, positive = more typical of group a (distressed)."""
    # binary=True: a term counts once per report, so one firm repeating a word 200 times (an
    # industry name, a table heading) cannot dominate the list
    cv = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-z][a-z]+\b", binary=True)
    ya = np.asarray(cv.transform(docs_a).sum(0)).ravel().astype(float)
    yb = np.asarray(cv.transform(docs_b).sum(0)).ravel().astype(float)
    a0 = ya + yb
    if prior_scale is None:
        prior_scale = 1.0
    alpha = a0 / a0.sum() * prior_scale * len(vocab)   # prior proportional to the pooled corpus
    alpha = np.maximum(alpha, 1e-3)
    A, na, nb = alpha.sum(), ya.sum(), yb.sum()
    d = (np.log((ya + alpha) / (na + A - ya - alpha)) - np.log((yb + alpha) / (nb + A - yb - alpha)))
    var = 1 / (ya + alpha) + 1 / (yb + alpha)
    return d / np.sqrt(var), ya, yb


def lexicon_oof(df, section="mdna", top=100, min_firms=10):
    """Contribution C, fold-aware: mine distress and healthy term lists from the training folds,
    then score each held-out report by the share of the distress terms it contains minus the share
    of the healthy terms."""
    y, F = df.label.to_numpy(), df.fold.to_numpy()
    docs = df[section].map(clean_text).to_numpy()
    has = df[section].str.len().to_numpy() > 0
    firms = df.firm_id.to_numpy()
    score = np.full(len(df), np.nan)
    lists = {}
    for k in range(5):
        tr, te = (F != k) & has, (F == k)
        vocab = firm_vocab(docs[tr], firms[tr], min_firms)
        z, _, _ = log_odds(docs[tr & (y == 1)], docs[tr & (y == 0)], vocab)
        order = np.argsort(z)
        pos = [vocab[i] for i in order[::-1][:top]]
        neg = [vocab[i] for i in order[:top]]
        lists[k] = (pos, neg)
        cv = CountVectorizer(vocabulary=pos + neg, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-z][a-z]+\b",
                             binary=True)
        C = cv.transform(docs[te]).toarray()
        s = (C[:, :top].sum(1) - C[:, top:].sum(1)) / top     # share of distress terms present minus healthy
        s[~has[te]] = np.nan
        score[te] = s
    return score, lists


def lexicon_full(df, section="mdna", min_firms=10):
    """The published lexicon: fitted on every row (a descriptive output, never used for scoring)."""
    y = df.label.to_numpy()
    docs = df[section].map(clean_text).to_numpy()
    has = df[section].str.len().to_numpy() > 0
    vocab = firm_vocab(docs[has], df.firm_id.to_numpy()[has], min_firms)
    z, ya, yb = log_odds(docs[has & (y == 1)], docs[has & (y == 0)], vocab)
    # in how many distressed / healthy firms' reports each term appears
    cv = CountVectorizer(vocabulary=vocab, ngram_range=(1, 2), token_pattern=r"(?u)\b[a-z][a-z]+\b", binary=True)
    X = cv.transform(docs[has])
    firms = df.firm_id.to_numpy()[has]; lab = y[has]
    import scipy.sparse as sp
    out = pd.DataFrame({"term": vocab, "z": z, "reports_distressed": ya, "reports_healthy": yb})
    for g, nm in [(1, "firms_distressed"), (0, "firms_healthy")]:
        m = lab == g
        f_codes, f_idx = np.unique(firms[m], return_inverse=True)
        M = sp.csr_matrix((np.ones(len(f_idx)), (f_idx, np.arange(len(f_idx)))), shape=(len(f_codes), len(f_idx)))
        out[nm] = np.asarray(((M @ X[m]) > 0).sum(0)).ravel()
    return out.sort_values("z", ascending=False).reset_index(drop=True)
