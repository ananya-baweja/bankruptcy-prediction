#!/usr/bin/env python3
"""Feature-family importance, the Gupta & Banerjee tone replication, descriptive checks."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "."); sys.path.insert(0, "vendor"); sys.path.append("../src")
from bppfinal.load import table  # noqa: E402
df, R, L = table()
y = df.label.to_numpy(float); F = df.fold.to_numpy()
out = {}

# ---------------------------------------------------------------- 1. feature-family permutation importance
FAM = {
    "Accounting ratios": R,
    "Financial statements missing (flag)": ["financials_missing"],
    "Auditor: formal flags (going concern, opinion, CARO)": ["has_going_concern", "has_emphasis_of_matter",
        "has_modified_opinion_basis", "audit_opinion_severity", "caro_default_flag", "caro_statutory_dues_flag"],
    "Auditor: tone, hedging, distress phrases": [c for c in L if c.startswith("auditor_") and
        any(t in c for t in ["_lm_", "hedge", "distress"])],
    "Auditor: readability and length": [c for c in L if c.startswith("auditor_") and
        any(t in c for t in ["fog", "sentence", "complex", "n_words", "n_sentences"])],
    "MD&A: tone (Loughran-McDonald)": [c for c in L if c.startswith("mdna_lm_")],
    "MD&A: hedging": [c for c in L if c.startswith("mdna_hedge")],
    "MD&A: distress phrases": [c for c in L if c.startswith("mdna_distress")],
    "MD&A: readability": ["mdna_fog_index", "mdna_avg_sentence_length", "mdna_complex_word_ratio"],
    "MD&A: length": ["mdna_n_words", "mdna_n_sentences"],
    "Section found (MD&A, auditor, CARO, opinion)": ["has_mdna", "has_auditor_report", "has_caro_annexure",
                                                      "audit_opinion_found"],
    "Year-on-year drift": [c for c in L if c.startswith("drift_")],
    "Perplexity": ["perplexity_fold"],
}
allc = R + L
assert sorted(sum(FAM.values(), [])) == sorted(allc), set(allc) ^ set(sum(FAM.values(), []))
X = df[allc].apply(pd.to_numeric, errors="coerce").to_numpy(float)
rng = np.random.default_rng(0)
imp = {f: [] for f in FAM}; imp_feat = {c: [] for c in allc}
N_REP = 30
for k in range(5):
    tr, te = F != k, F == k
    m = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
    base = average_precision_score(y[te], m.predict_proba(X[te])[:, 1])
    Xte = X[te]
    for fam, cs in FAM.items():
        ix = [allc.index(c) for c in cs]; drops = []
        for _ in range(N_REP):
            Z = Xte.copy(); perm = rng.permutation(len(Z)); Z[:, ix] = Xte[perm][:, ix]
            drops.append(base - average_precision_score(y[te], m.predict_proba(Z)[:, 1]))
        imp[fam].append(float(np.mean(drops)))
    for c in allc:
        ix = allc.index(c); drops = []
        for _ in range(10):
            Z = Xte.copy(); Z[:, ix] = rng.permutation(Z[:, ix])
            drops.append(base - average_precision_score(y[te], m.predict_proba(Z)[:, 1]))
        imp_feat[c].append(float(np.mean(drops)))
fam_tab = (pd.DataFrame({"family": list(FAM), "n_features": [len(v) for v in FAM.values()],
                         "pr_drop": [np.mean(imp[f]) for f in FAM], "pr_drop_sd": [np.std(imp[f], ddof=1) for f in FAM],
                         "folds_positive": [int((np.array(imp[f]) > 0).sum()) for f in FAM]})
           .sort_values("pr_drop", ascending=False))
feat_tab = (pd.DataFrame({"feature": allc, "pr_drop": [np.mean(imp_feat[c]) for c in allc],
                          "folds_positive": [int((np.array(imp_feat[c]) > 0).sum()) for c in allc]})
            .sort_values("pr_drop", ascending=False))
print(fam_tab.round(4).to_string()); print(feat_tab.head(15).round(4).to_string())
fam_tab.to_csv("results_cpu/importance_families.csv", index=False)
feat_tab.to_csv("results_cpu/importance_features.csv", index=False)

# ---------------------------------------------------------------- 2. tone replication (Gupta & Banerjee 2023)
tone = {}
for c in ["mdna_lm_negative_ratio", "auditor_lm_negative_ratio", "mdna_lm_uncertainty_ratio",
          "mdna_hedge_density", "mdna_lm_positive_ratio", "mdna_fog_index", "auditor_hedge_density"]:
    v = pd.to_numeric(df[c], errors="coerce")
    d, h = v[df.label == 1].dropna(), v[df.label == 0].dropna()
    # within matched pair and fiscal-year rank: distressed minus healthy
    w = df.assign(v=v).pivot_table(index=["pair_id", "horizon"], columns="label", values="v", aggfunc="mean").dropna()
    diff = w[1] - w[0]
    tone[c] = dict(mean_distressed=float(d.mean()), mean_healthy=float(h.mean()),
                   median_distressed=float(d.median()), median_healthy=float(h.median()),
                   mw_p=float(mannwhitneyu(d, h).pvalue),
                   auc=float(roc_auc_score(np.r_[np.ones(len(d)), np.zeros(len(h))], np.r_[d, h])),
                   n_pairs=int(len(diff)), share_pairs_distressed_higher=float((diff > 0).mean()),
                   wilcoxon_p=float(wilcoxon(diff).pvalue))
print(pd.DataFrame(tone).T.round(4).to_string())
out["tone"] = tone

# ---------------------------------------------------------------- 3. descriptives
out["desc"] = dict(
    mdna_present=df.groupby("label").apply(lambda g: float((g.mdna.str.len() > 0).mean())).to_dict(),
    auditor_present=df.groupby("label").apply(lambda g: float((g.auditor_report.str.len() > 0).mean())).to_dict(),
    fin_missing=df.groupby("label").financials_missing.mean().to_dict(),
    going_concern=df.groupby("label").has_going_concern.mean().to_dict(),
    n=len(df), n_pos=int(y.sum()), n_strict=int(df.strict.sum()), n_pairs=int(df.pair_id.nunique()),
    extra_healthy=int((~df.included.astype(bool)).sum()),
)
z = pd.to_numeric(df.altman_z_em, errors="coerce") - 3.25
out["altman_zone"] = dict(
    distress_zone_share_distressed=float((z[df.label == 1] < 1.1).mean()),
    distress_zone_share_healthy=float((z[df.label == 0] < 1.1).mean()),
    safe_zone_share_distressed=float((z[df.label == 1] > 2.6).mean()),
    safe_zone_share_healthy=float((z[df.label == 0] > 2.6).mean()),
    z_missing=int(z.isna().sum()))
print(json.dumps(out["desc"], indent=1, default=str)); print(out["altman_zone"])
json.dump(out, open("results_cpu/extras.json", "w"), indent=1, default=float)
