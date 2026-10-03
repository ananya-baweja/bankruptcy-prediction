#!/usr/bin/env python3
"""Fit the final scoring models on all 763 company-years and save what score.py needs.

    python train_final.py --data data_v2 [--dl-oof results_v2/dl_oof.csv]

Run it on the machine that will score (a model saved by one scikit-learn version may not load
in another). Takes about two minutes on a CPU. Writes final_models/:

  hgb.pkl          gradient boosting on ratios + language features, fitted on every row
  lm_fold{k}.pkl.gz  the five healthy-firm trigram language models behind the perplexity feature
  reference.npz    the cross-validated (out-of-fold) scores of every row - a new company's score
                   is placed among them, so "risk score 93" means it scored above 93% of the
                   healthy company-years the models were tested on
  reference.json   columns, band thresholds and what each band caught in cross-validation,
                   healthy-firm feature quantiles (for the explanations), versions
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
sys.path.append(str(HERE.parent / "src"))          # inside the repository: the bpp package itself
from bppfinal.table import build  # noqa: E402

BANDS = {"high": 0.90, "watch": 0.70}   # share of healthy company-years scoring below the band


def pct_against(ref, x):
    """Share of the reference scores strictly below x, plus half the ties (a mid-rank percentile)."""
    ref = np.sort(np.asarray(ref, float)); x = np.atleast_1d(np.asarray(x, float))
    lo = np.searchsorted(ref, x, "left"); hi = np.searchsorted(ref, x, "right")
    return (lo + 0.5 * (hi - lo)) / len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data_v2"))
    ap.add_argument("--dl-oof", default=str(HERE / "results_v2" / "dl_oof.csv"),
                    help="the cross-validated network predictions (run_gpu.py); without them the "
                         "scorer uses gradient boosting alone")
    ap.add_argument("--out", default=str(HERE / "final_models"))
    args = ap.parse_args()

    from sklearn import __version__ as skl_version
    from sklearn.ensemble import HistGradientBoostingClassifier
    from bpp.nlp.lm import fit as lm_fit
    from run_gpu import table_checksum

    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df, RATIOS, LANG = build(Path(args.data), with_perplexity=True, folds_file=HERE / "folds.csv")
    chk = table_checksum(df, RATIOS + LANG)
    cols = RATIOS + LANG
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = df.label.to_numpy(int); fold = df.fold.to_numpy()
    print(f"table checksum {chk}; {len(df)} rows, {len(cols)} features", flush=True)

    # ---- cross-validated scores: the reference a new company is placed against
    oof_hgb = np.full(len(df), np.nan)
    for k in range(5):
        tr, te = fold != k, fold == k
        oof_hgb[te] = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

    dl_seeds, oof_dl_seed = [], None
    dl_path = Path(args.dl_oof)
    if dl_path.exists():
        d = pd.read_csv(dl_path)
        assert (d.doc_id.to_numpy() == df.doc_id.to_numpy()).all(), "dl_oof.csv rows differ from the table"
        seed_cols = sorted([c for c in d.columns if c.startswith("main_s")], key=lambda c: int(c[6:]))
        oof_dl_seed = d[seed_cols].to_numpy(float)
        dl_seeds = [int(c[6:]) for c in seed_cols]
        info = json.loads((dl_path.parent / "run_info.json").read_text()) if (dl_path.parent / "run_info.json").exists() else {}
        if info.get("table_checksum") not in (None, chk):
            sys.exit(f"{dl_path} was made from a different table (checksum {info.get('table_checksum')}, "
                     f"this one {chk}). Use the run made with the same --data.")
        print(f"network reference: {len(seed_cols)} seeds from {dl_path}", flush=True)
    else:
        print(f"no {dl_path}: the scorer will use gradient boosting alone", flush=True)

    # Each model's score becomes a percentile among its own cross-validated scores; the network's
    # seeds are averaged that way too, and the blend is the mean of the two (weight 0.5, as fixed
    # before the evaluation). Pooled percentiles stand in for the within-fold ranks of the report.
    p_hgb = pct_against(oof_hgb, oof_hgb)
    if oof_dl_seed is not None:
        p_dl = np.mean([pct_against(oof_dl_seed[:, j], oof_dl_seed[:, j]) for j in range(oof_dl_seed.shape[1])], 0)
        blend = 0.5 * p_hgb + 0.5 * p_dl
    else:
        p_dl, blend = None, p_hgb
    healthy = y == 0

    # band thresholds and what they caught, measured on the cross-validated scores
    def band_stats(score):
        thr = {b: float(np.quantile(score[healthy], q)) for b, q in BANDS.items()}
        res = {}
        for b, t in thr.items():
            flag = score >= t
            res[b] = dict(threshold=t, healthy_flagged=float(flag[healthy].mean()),
                          insolvent_caught=float(flag[~healthy].mean()),
                          insolvent_caught_by_horizon={h: float(flag[(~healthy) & (df.horizon == h).to_numpy()].mean())
                                                       for h in ["t-1", "t-2", "t-3"]})
        return res
    bands = {"blend" if p_dl is not None else "hgb": band_stats(blend), "hgb": band_stats(p_hgb)}

    from sklearn.metrics import average_precision_score, roc_auc_score
    def fold_mean(f, s):
        return float(np.mean([f(y[fold == k], s[fold == k]) for k in range(5)]))
    cv = {"hgb": dict(pr_auc=fold_mean(average_precision_score, oof_hgb), roc_auc=fold_mean(roc_auc_score, oof_hgb))}
    if p_dl is not None:
        cv["dl"] = dict(pr_auc=fold_mean(average_precision_score, p_dl), roc_auc=fold_mean(roc_auc_score, p_dl))
        cv["blend"] = dict(pr_auc=fold_mean(average_precision_score, blend), roc_auc=fold_mean(roc_auc_score, blend))
    print("cross-validated (pooled-percentile versions):", json.dumps(cv), flush=True)

    # ---- final models, fitted on every row
    hgb = HistGradientBoostingClassifier(random_state=0).fit(X, y)
    with open(out / "hgb.pkl", "wb") as f:
        pickle.dump(hgb, f, protocol=4)
    # Perplexity: the five fold language models, exactly as the training rows' values were made
    # (each on the healthy MD&As of four folds, ~300 reports). A new report is scored by all five
    # and the mean is used. One model fitted on all 375 healthy MD&As would know more text and
    # give systematically lower perplexities than the ones the boosting model learned from.
    for k in range(5):
        train_healthy = (df.fold != k) & (df.label == 0) & (df.mdna.str.len() > 0)
        lm = lm_fit(df.loc[train_healthy, "mdna"].tolist(), order=3, min_count=2)
        with gzip.open(out / f"lm_fold{k}.pkl.gz", "wb") as f:
            pickle.dump(lm, f, protocol=4)
    print("final models fitted (boosting on all rows; the five fold language models saved)", flush=True)

    # ---- healthy-firm feature distributions, for the explanations
    q = {}
    for j, c in enumerate(cols):
        v = X[healthy, j]; v = v[~np.isnan(v)]
        w = X[~healthy, j]; w = w[~np.isnan(w)]
        q[c] = dict(healthy=[float(x) for x in np.quantile(v, [0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95])] if len(v) else None,
                    insolvent_median=float(np.median(w)) if len(w) else None,
                    healthy_rate=float(v.mean()) if len(v) and set(np.unique(v)) <= {0.0, 1.0} else None,
                    insolvent_rate=float(w.mean()) if len(w) and set(np.unique(w)) <= {0.0, 1.0} else None)

    np.savez_compressed(out / "reference.npz", doc_id=df.doc_id.to_numpy().astype(str), label=y, fold=fold,
                        horizon=df.horizon.astype(str).to_numpy(), oof_hgb=oof_hgb,
                        oof_dl_seed=oof_dl_seed if oof_dl_seed is not None else np.zeros((0, 0)),
                        blend=blend, X=X)
    (out / "reference.json").write_text(json.dumps(dict(
        table_checksum=chk, rows=len(df), healthy=int(healthy.sum()), insolvent=int((~healthy).sum()),
        ratios=RATIOS, language=LANG, columns=cols, dl_seeds=dl_seeds, bands=bands, band_quantiles=BANDS,
        cv=cv, quantiles=q, data=str(args.data), dl_oof=str(dl_path) if dl_seeds else None,
        sklearn=skl_version, numpy=np.__version__, pandas=pd.__version__, python=platform.python_version(),
        trained=time.strftime("%Y-%m-%d %H:%M")), indent=1))
    b = bands["blend" if p_dl is not None else "hgb"]
    print(f"bands: High = above {BANDS['high']:.0%} of healthy (caught {b['high']['insolvent_caught']:.0%} of "
          f"insolvent company-years); Watch = above {BANDS['watch']:.0%} (caught {b['watch']['insolvent_caught']:.0%})")
    print(f"DONE in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
