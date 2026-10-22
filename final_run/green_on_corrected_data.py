#!/usr/bin/env python3
"""Table 20 of the final report: the Green AI configurations that need no new training (1, 3, 5 or 7
network seeds; no network at all), scored on the corrected data of Section 7 from the saved
out-of-fold predictions of the final run. No GPU needed.

    python final_run/green_on_corrected_data.py "<project folder>/final_run"

The argument is the project's final_run folder (not in git), which holds analysis/results_cpu/,
analysis/results_final/, analysis/folds.csv and gpu_v2/results_v2/. Writes
results/green/green_on_corrected_data.csv. GPU energy for these configurations is in
results/green/green_results.csv: the computation is identical, only the CARO flags differ.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bppfinal import stats as st  # noqa: E402

FR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent.parent / "final_run"
o = pd.read_csv(FR / "analysis/results_cpu/cpu_oof.csv")
d = pd.read_csv(FR / "gpu_v2/results_v2/dl_oof.csv")
f = pd.read_csv(FR / "analysis/folds.csv").set_index("doc_id").loc[o.doc_id]
a = pd.read_csv(FR / "analysis/results_final/all_oof.csv")
assert (d.doc_id.values == o.doc_id.values).all() and (a.doc_id.values == o.doc_id.values).all()
y = o.label.to_numpy(float); F = o.fold.to_numpy(); S = o.strict.to_numpy().astype(bool); G = f.pair_id.to_numpy()

seeds = sorted([c for c in d.columns if c.startswith("main_s")], key=lambda c: int(c.split("_s")[-1]))
ens = lambda n: np.nanmean([st.rank_norm(d[c].to_numpy(float), F) for c in seeds[:n]], 0)
gb, gbr = o.hgb_both.to_numpy(float), o.hgb_ratios.to_numpy(float)
blend = lambda n: 0.5 * st.rank_norm(ens(n), F) + 0.5 * st.rank_norm(gb, F)
cur = blend(7)
assert np.nanmax(abs(cur - a.blend.to_numpy(float))) < 1e-9, "does not reproduce the report's full system"

cands = {"GB ratios only": gbr, "GB ratios+language (chosen)": gb, "Network only 7 seeds": ens(7),
         **{f"Blend {n} seed(s)": blend(n) for n in (1, 3, 5, 7)}}
rows = {}
for k, p in cands.items():
    pr, roc, prs = (st.per_fold(y, p, F, sub, m) for sub, m in ((None, "pr"), (None, "roc"), (S, "pr")))
    r = dict(pr=np.nanmean(pr), pr_sd=np.nanstd(pr, ddof=1), roc=np.nanmean(roc), pr_strict=np.nanmean(prs))
    if k != "Blend 7 seed(s)":
        b = st.pair_bootstrap(y, p, cur, F, G); w = st.paired(y, p, cur, F)
        r.update(diff=w["diff"], lo=b["lo"], hi=b["hi"], wil=w["wilcoxon_p"], wins=w["wins"])
    rows[k] = r
out = pd.DataFrame(rows).T
dest = HERE.parent / "results/green/green_on_corrected_data.csv"
out.to_csv(dest)
print(out.round(4).to_string()); print("wrote", dest)
