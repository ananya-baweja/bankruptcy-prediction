"""Build the 763-row modelling table from processed/ - the same steps as notebook 1, sections 2-6.

Used both by the CPU analyses and by the GPU runner, so the two can never disagree about
rows, folds or feature lists. Nothing here trains a predictive model.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

RATIOS = ["current_ratio", "quick_ratio", "cash_to_assets", "ebitda_margin", "roce", "roa",
          "debt_to_equity", "debt_to_ebitda", "interest_coverage", "retained_earnings_to_assets",
          "altman_z_em"]

DROP = {"doc_id", "firm_id", "fy", "pair_id", "role", "label", "horizon", "included", "exclude_reason",
        "perplexity_source", "drift_previous_doc_id", "has_previous_year", "strict", "fold",
        "mdna", "auditor_report", "included_financials"}
LEAKY = {"cirp_specific_mentions", "ibc_generic_mentions"}
N_SPLITS = 5


def load_text(data: Path) -> pd.DataFrame:
    rows = []
    for part in ["report_sections_1.jsonl.gz", "report_sections_2.jsonl.gz"]:
        with gzip.open(data / part, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append({"doc_id": r["doc_id"],
                             "mdna": r.get("mdna") or "",
                             "auditor_report": r.get("auditor_report") or ""})
    return pd.DataFrame(rows)


def build(data: Path, repo_src: Path | None = None, with_perplexity: bool = True, verbose: bool = True,
          folds_file: Path | None = None):
    """Return (df, RATIOS, LANG). df carries the text columns 'mdna' and 'auditor_report'.

    folds_file: a CSV (doc_id, fold) that fixes the folds. GroupKFold's assignment depends on how
    numpy breaks ties when sorting pair sizes, which changed between library versions, so the final
    run reads the folds from a file instead of recomputing them.
    """
    data = Path(data)
    dl = pd.read_csv(data / "documents_labeled.csv")
    lf = pd.read_csv(data / "language_features.csv")
    ra = pd.read_csv(data / "ratios.csv")
    text = load_text(data)

    keep = dl.included | (dl.exclude_reason == "pair_partner_excluded")
    base = dl.loc[keep & (dl.has_document == True),
                  ["doc_id", "firm_id", "fy", "pair_id", "label", "included", "horizon"]].copy()
    df = (base
          .merge(lf.drop(columns=[c for c in ["label", "pair_id", "role", "horizon", "included",
                                              "exclude_reason"] if c in lf.columns]),
                 on=["doc_id", "firm_id", "fy"], how="left")
          .merge(ra[["firm_id", "fy"] + RATIOS + ["included_financials"]], on=["firm_id", "fy"], how="left")
          .merge(text, on="doc_id", how="left")
          .fillna({"mdna": "", "auditor_report": ""})
          .reset_index(drop=True))
    df["financials_missing"] = (~df["included_financials"].fillna(False).astype(bool)).astype(float)
    df["strict"] = (df.included & df.included_financials.fillna(False).astype(bool))

    # folds: grouped by matched pair (notebook 1, section 4)
    y, groups = df.label.to_numpy(), df.pair_id.to_numpy()
    fold = np.full(len(df), -1)
    for k, (_, te) in enumerate(GroupKFold(n_splits=N_SPLITS).split(df, y, groups)):
        fold[te] = k
    df["fold"] = fold
    if folds_file is not None:
        fx = pd.read_csv(folds_file)[["doc_id", "fold"]]
        if verbose:
            agree = df[["doc_id", "fold"]].merge(fx, on="doc_id", suffixes=("_gkf", ""))
            print(f"folds from {Path(folds_file).name}; this library's GroupKFold agrees on "
                  f"{(agree.fold_gkf == agree.fold).mean():.0%} of rows")
        df = df.drop(columns="fold").merge(fx, on="doc_id", how="left")
        assert df.fold.notna().all() and len(df) == len(fx), "folds file does not match the table"
        df["fold"] = df.fold.astype(int)
        fold = df.fold.to_numpy()
    assert all(len(set(groups[fold == k]) & set(groups[fold != k])) == 0 for k in range(N_SPLITS))

    LANG = [c for c in lf.columns if c not in DROP and c not in LEAKY and lf[c].dtype.kind in "fib"]
    dead = [c for c in LANG + RATIOS if df[c].isna().all() or df[c].nunique(dropna=True) <= 1]
    LANG = [c for c in LANG if c not in dead] + ["financials_missing"]

    if with_perplexity:
        if repo_src is not None and str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))
        from bpp.nlp.lm import fit as lm_fit, perplexity as lm_perplexity
        ppl = np.full(len(df), np.nan)
        t0 = time.time()
        for k in range(N_SPLITS):
            train_healthy = (df.fold != k) & (df.label == 0) & (df.mdna.str.len() > 0)
            model = lm_fit(df.loc[train_healthy, "mdna"].tolist(), order=3, min_count=2)
            for i in np.flatnonzero((df.fold == k).to_numpy()):
                ppl[i] = lm_perplexity(model, df.mdna.iloc[i]) or np.nan
            if verbose:
                print(f"  perplexity fold {k}: {int(train_healthy.sum())} healthy reports, "
                      f"{time.time()-t0:.0f}s", flush=True)
        df["perplexity_fold"] = ppl
        LANG = LANG + ["perplexity_fold"]
    if verbose:
        print(f"rows {len(df)}  healthy {int((df.label==0).sum())}  distressed {int((df.label==1).sum())}  "
              f"pairs {df.pair_id.nunique()}  strict {int(df.strict.sum())}")
        print(f"{len(LANG)} language columns, {len(RATIOS)} ratios; dropped as dead: {dead}")
    return df, RATIOS, LANG
