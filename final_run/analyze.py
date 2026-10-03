#!/usr/bin/env python3
"""Every number in the final report, from the saved out-of-fold predictions.

    python analyze.py [--dl results/]      # CPU predictions always; DL ones if the folder is given
Writes results_final/tables.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bppfinal import stats as st  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--dl", default=None); args = ap.parse_args()
OUT = Path(__import__("os").environ.get("BPP_OUT", "results_final")); OUT.mkdir(exist_ok=True)
from bppfinal.load import table  # noqa: E402
df, _, _ = table()
o = pd.read_csv("results_cpu/cpu_oof.csv")
assert (o.doc_id.values == df.doc_id.values).all()
y = o.label.to_numpy(float); F = o.fold.to_numpy(); S = o.strict.to_numpy().astype(bool)
G = df.pair_id.to_numpy(); H = df.horizon.to_numpy()
P = {c: o[c].to_numpy(float) for c in o.columns if c not in ("doc_id", "label", "fold", "strict", "horizon")}
T = {}

# ------------------------------------------------------------------ deep-learning predictions
HAVE_DL = args.dl is not None
if HAVE_DL:
    D = Path(args.dl)
    d = pd.read_csv(D / "dl_oof.csv")
    assert (d.doc_id.values == df.doc_id.values).all(), "DL predictions are in a different row order"
    info = json.loads((D / "run_info.json").read_text())
    T["dl_run_info"] = {k: v for k, v in info.items() if k != "runs"}
    T["dl_epochs"] = {k: [r["epochs"] for r in v] for k, v in info["runs"].items()}

    def seeds(prefix, n=None):
        cs = sorted([c for c in d.columns if c.startswith(prefix + "_s")], key=lambda c: int(c.split("_s")[-1]))
        return cs[:n] if n else cs

    def ens(prefix, n=None):
        cs = seeds(prefix, n)
        return np.nanmean([st.rank_norm(d[c].to_numpy(float), F) for c in cs], 0), cs

    for name in ["main", "strict", "notext", "text_mdna", "text_aud", "text_both", "fusion_aud", "fusion_both"]:
        if seeds(name):
            P["dl_" + name], cs = ens(name)
            P["dlprob_" + name] = np.nanmean([d[c].to_numpy(float) for c in cs], 0)
    P["dl_main5"], _ = ens("main", 5)
    P["blend"] = 0.5 * st.rank_norm(P["dl_main"], F) + 0.5 * st.rank_norm(P["hgb_both"], F)
    if "dl_strict" in P:
        P["blend_strict"] = 0.5 * st.rank_norm(P["dl_strict"], F) + 0.5 * st.rank_norm(P["hgb_both_strict"], F)
    # per-seed spread of each experiment's 5-fold mean PR-AUC
    T["dl_seed_spread"] = {}
    for name in ["main", "strict", "notext", "text_mdna", "text_aud", "text_both", "fusion_aud", "fusion_both"]:
        sub = S if name == "strict" else None
        v = [np.nanmean(st.per_fold(y, d[c].to_numpy(float), F, sub, "pr")) for c in seeds(name)]
        r = [np.nanmean(st.per_fold(y, d[c].to_numpy(float), F, sub, "roc")) for c in seeds(name)]
        if v:
            T["dl_seed_spread"][name] = dict(pr=v, roc=r, pr_mean=float(np.mean(v)), pr_sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0,
                                             roc_mean=float(np.mean(r)))
    # leak test
    z = np.load(D / "runs" / "leak_s0.npz")
    from sklearn.metrics import roc_auc_score
    T["leak"] = dict(roc=float(roc_auc_score(z["y_shuffled"], z["oof"])),
                     per_fold=[float(roc_auc_score(z["y_shuffled"][F == k], z["oof"][F == k])) for k in range(5)])
    # averaged FinBERT vectors + logistic regression, C chosen inside the training folds (Mai et al. 2019)
    for sec in ["mdna", "auditor"]:
        Emean = np.load(D / f"emb_mean_{sec}.npy")
        for with_tab in [False, True]:
            oof = np.full(len(y), np.nan)
            for k in range(5):
                tr, te = F != k, F == k
                A, B = Emean[tr], Emean[te]
                mu, sd = A.mean(0), A.std(0) + 1e-6
                A, B = (A - mu) / sd, (B - mu) / sd
                pca = PCA(32, random_state=0).fit(A); A, B = pca.transform(A), pca.transform(B)
                if with_tab:
                    from bppfinal.cpu_models import _lin_prep
                    cols = json.load(open("results_cpu/feature_sets.json"))
                    Xt = df[cols["ratios"] + cols["lang"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
                    a2, b2 = _lin_prep(Xt[tr], Xt[te]); A, B = np.hstack([A, a2]), np.hstack([B, b2])
                inner = list(GroupKFold(5).split(A, y[tr], G[tr]))
                m = LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=inner, scoring="average_precision",
                                         max_iter=5000).fit(A, y[tr])
                oof[te] = m.predict_proba(B)[:, 1]
            P[f"emb_lr_{sec}" + ("_tab" if with_tab else "")] = oof
    # gates
    g = pd.read_csv(D / "dl_gates_main.csv")
    gm = g.groupby("doc_id")[["g_text", "g_lang", "g_ratio"]].mean().reindex(df.doc_id)
    T["gates"] = {"all": gm.mean().to_dict(),
                  "distressed": gm[y == 1].mean().to_dict(), "healthy": gm[y == 0].mean().to_dict(),
                  "by_seed": g.groupby("seed")[["g_text", "g_lang", "g_ratio"]].mean().to_dict("index")}

# ------------------------------------------------------------------ subsets
z = pd.to_numeric(df.altman_z_em, errors="coerce") - 3.25
r_rank = st.rank_norm(P["hgb_ratios"], F)
SUBSETS = {
    "all": None, "strict": S,
    "t-1": H == "t-1", "t-2": H == "t-2", "t-3": H == "t-3",
    "z_above_median": ((z > z.median()) & z.notna()).to_numpy(),
    "z_safe_zone": ((z > 2.6) & z.notna()).to_numpy(),
    "ratio_model_low_risk": r_rank <= 0.5,
}
T["subset_sizes"] = {k: dict(n=int(len(y) if v is None else v.sum()), pos=int(y.sum() if v is None else y[v].sum()))
                     for k, v in SUBSETS.items()}

MODELS = list(P)
T["perf"] = {}
for sname, sub in SUBSETS.items():
    T["perf"][sname] = {}
    for m in MODELS:
        if m == "perplexity" and sname != "all":
            continue
        mets = ("roc", "pr", "tpr10", "pr_1in11") + (("brier",) if (m.startswith("hgb") or m.startswith("logit")
                                                                   or m.startswith("dlprob") or m.startswith("emb_lr")) else ())
        T["perf"][sname][m] = st.summarise(y, P[m], F, sub, mets)

# ------------------------------------------------------------------ comparisons
COMP = [
    ("hgb_both", "hgb_ratios"), ("hgb_ratios_fm", "hgb_ratios"), ("hgb_ratios_langtext", "hgb_ratios"),
    ("hgb_both", "hgb_ratios_fm"),
    ("hgb_ratios", "altman"), ("hgb_ratios", "logit_ratios"),
    ("hgb_ratios_mdna", "hgb_ratios"), ("hgb_ratios_aud", "hgb_ratios"), ("hgb_ratios_flags", "hgb_ratios"),
    ("hgb_ratios_langtext", "hgb_ratios_aud"), ("hgb_ratios_langtext", "hgb_ratios_mdna"),
    ("hgb_aud", "hgb_mdna"), ("tfidf_aud", "tfidf_mdna"), ("hgb_flags", "hgb_mdna"),
    ("hgb_both_nodrift", "hgb_both"), ("hgb_both_lex", "hgb_both"), ("hgb_both_noperp", "hgb_both"),
    ("hgb_lang_lex", "hgb_lang"), ("lexicon_aud", "lexicon_mdna"), ("hgb_lang", "logit_lang"),
    ("hgb_both_strict", "hgb_ratios_strict"),
]
if HAVE_DL:
    COMP += [("blend", "hgb_ratios"), ("blend", "hgb_both"), ("blend", "dl_main"), ("dl_main", "hgb_ratios"),
             ("dl_main", "hgb_both"), ("dl_main", "dl_notext"), ("dl_text_aud", "dl_text_mdna"),
             ("dl_main5", "dl_fusion_aud"), ("dl_main5", "dl_fusion_both"), ("dl_text_both", "dl_text_mdna"),
             ("blend", "hgb_ratios_fm"), ("emb_lr_auditor", "emb_lr_mdna"), ("dl_text_mdna", "emb_lr_mdna"),
             ("dl_text_aud", "emb_lr_auditor")]
    if "blend_strict" in P:
        COMP += [("blend_strict", "hgb_ratios_strict"), ("dl_strict", "hgb_ratios_strict")]
T["comp"] = {}
BOOT_SUBSETS = {"all", "strict", "z_above_median", "z_safe_zone", "ratio_model_low_risk"}
for sname in ["all", "strict", "z_above_median", "z_safe_zone", "ratio_model_low_risk", "t-1", "t-2", "t-3"]:
    sub = SUBSETS[sname]; T["comp"][sname] = {}
    for a, b in COMP:
        if sname != "strict" and a.endswith("_strict"):
            continue
        key = f"{a} - {b}"
        r = st.paired(y, P[a], P[b], F, sub, "pr")
        r["roc"] = st.paired(y, P[a], P[b], F, sub, "roc")
        if sname in BOOT_SUBSETS:
            r["boot"] = st.pair_bootstrap(y, P[a], P[b], F, G, sub, "pr", n_boot=1000)
        r["delong"] = st.delong_stratified(y, P[a], P[b], F, sub)
        T["comp"][sname][key] = r

# Altman zone rule as a classifier (Z'' < 1.1 = distress zone; missing Z counts as not flagged)
flag = (z < 1.1).to_numpy()
T["altman_rule"] = dict(recall=float(flag[y == 1].mean()), false_alarm=float(flag[y == 0].mean()))

json.dump(T, open(OUT / "tables.json", "w"), indent=1, default=float)
pd.DataFrame({"doc_id": df.doc_id, **{k: v for k, v in P.items()}}).to_csv(OUT / "all_oof.csv", index=False)
print("models:", len(P), "| DL:", HAVE_DL, "| wrote", OUT / "tables.json")
for s in ["all", "strict", "z_above_median", "z_safe_zone", "ratio_model_low_risk", "t-1"]:
    print(f"\n== {s} {T['subset_sizes'][s]}")
    for m in ["altman", "logit_ratios", "hgb_ratios", "hgb_ratios_fm", "hgb_ratios_langtext", "hgb_both"] + \
             (["dl_main", "blend"] if HAVE_DL else []):
        r = T["perf"][s][m]
        print(f"  {m:20s} ROC {r['roc']:.3f}  PR {r['pr']:.3f}±{r['pr_sd']:.3f}  TPR@10 {r['tpr10']:.3f}  PR1:10 {r['pr_1in11']:.3f}")
