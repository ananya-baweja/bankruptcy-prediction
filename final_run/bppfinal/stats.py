"""Metrics and tests used in the final report. Every metric is computed per fold and averaged
(notebook 2, section 2), so models are never penalised for fold-to-fold calibration."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata, ttest_rel, wilcoxon
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

K = 5


def rank_norm(p, fold):
    p = np.asarray(p, float); out = np.full(len(p), np.nan)
    for k in range(K):
        m = (fold == k) & ~np.isnan(p)
        if m.sum():
            out[m] = rankdata(p[m]) / (m.sum() + 1)
    return out


def tpr_at_fpr(y, p, fpr_target=0.10):
    """Share of insolvent firms caught when 10% of healthy firms are flagged (a point on the ROC curve)."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))


def pr_at_prevalence(y, p, prevalence=1 / 11):
    """PR-AUC re-weighted so that insolvent firms are 1 in 11 (about 1:10), instead of ~1:1."""
    pos, neg = (y == 1).sum(), (y == 0).sum()
    w_neg = pos / neg * (1 - prevalence) / prevalence
    w = np.where(y == 1, 1.0, w_neg)
    return float(average_precision_score(y, p, sample_weight=w))


METRICS = {
    "roc": lambda y, p: roc_auc_score(y, p),
    "pr": lambda y, p: average_precision_score(y, p),
    "tpr10": tpr_at_fpr,
    "pr_1in11": pr_at_prevalence,
    "brier": lambda y, p: brier_score_loss(y, p),
}


def per_fold(y, p, fold, subset=None, metric="pr"):
    f = METRICS[metric]; out = []
    for k in range(K):
        m = (fold == k) & ~np.isnan(p)
        if subset is not None:
            m &= subset
        if m.sum() < 5 or len(set(y[m])) < 2:
            out.append(np.nan); continue
        out.append(f(y[m], p[m]))
    return np.array(out)


def summarise(y, p, fold, subset=None, metrics=("roc", "pr", "tpr10", "pr_1in11")):
    r = {}
    for m in metrics:
        v = per_fold(y, p, fold, subset, m)
        r[m] = float(np.nanmean(v)); r[m + "_sd"] = float(np.nanstd(v, ddof=1)); r[m + "_folds"] = v.tolist()
    return r


def paired(y, a, b, fold, subset=None, metric="pr"):
    """a minus b, fold by fold: mean difference, paired t and Wilcoxon p, folds won."""
    va, vb = per_fold(y, a, fold, subset, metric), per_fold(y, b, fold, subset, metric)
    d = va - vb
    ok = ~np.isnan(d)
    try:
        pw = float(wilcoxon(va[ok], vb[ok]).pvalue)
    except ValueError:
        pw = float("nan")
    return dict(diff=float(np.nanmean(d)), t_p=float(ttest_rel(va[ok], vb[ok]).pvalue), wilcoxon_p=pw,
                wins=int((d[ok] > 0).sum()), n=int(ok.sum()), folds=d.tolist())


def pair_bootstrap(y, a, b, fold, groups, subset=None, metric="pr", n_boot=2000, seed=0):
    """Resample matched pairs with replacement; difference of pooled metric on within-fold ranks."""
    A, B = rank_norm(a, fold), rank_norm(b, fold)
    use = ~np.isnan(A) & ~np.isnan(B)
    if subset is not None:
        use &= subset
    f = METRICS[metric]
    idx = np.flatnonzero(use)
    g = groups[idx]
    pairs = np.unique(g)
    idx_of = {p: idx[g == p] for p in pairs}
    rng = np.random.default_rng(seed); d = []
    for _ in range(n_boot):
        take = np.concatenate([idx_of[p] for p in rng.choice(pairs, len(pairs), replace=True)])
        if len(set(y[take])) < 2:
            continue
        d.append(f(y[take], A[take]) - f(y[take], B[take]))
    d = np.array(d)
    return dict(point=float(f(y[idx], A[idx]) - f(y[idx], B[idx])), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), p_gt0=float((d > 0).mean()))


# ----------------------------------------------------------------------------- DeLong
def _midrank(x):
    return rankdata(x)


def _delong_cov(y, preds):
    """Fast DeLong (Sun & Xu 2014). preds: (m, n). Returns AUCs and their covariance."""
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m_, n_ = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.array([_midrank(r) for r in pos]); ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m_].sum(1) / m_ / n_ - (m_ + 1) / 2 / n_
    v01 = (tz[:, :m_] - tx) / n_
    v10 = 1 - (tz[:, m_:] - ty) / m_
    cov = np.cov(v01).reshape(k, k) / m_ + np.cov(v10).reshape(k, k) / n_
    return aucs, cov


def delong_stratified(y, a, b, fold, subset=None):
    """DeLong per fold, combined across folds (independent test sets): mean ROC-AUC difference and
    its two-sided p. Assumes reports are independent; the pair bootstrap relaxes that."""
    ds, vs = [], []
    for k in range(K):
        m = (fold == k) & ~np.isnan(a) & ~np.isnan(b)
        if subset is not None:
            m &= subset
        if len(set(y[m])) < 2:
            continue
        aucs, cov = _delong_cov(y[m], np.vstack([a[m], b[m]]))
        ds.append(aucs[0] - aucs[1]); vs.append(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    d, v = np.mean(ds), np.sum(vs) / len(ds) ** 2
    z = d / np.sqrt(v)
    return dict(diff=float(d), z=float(z), p=float(2 * norm.sf(abs(z))))
