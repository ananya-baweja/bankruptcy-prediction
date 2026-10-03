"""Result charts for the final report, from results_final/tables.json."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
T = json.load(open(ROOT / "results_final" / "tables.json"))
OUT = str(ROOT / "report" / "fig")
HAVE_DL = "blend" in T["perf"]["all"]
GREY, BLUE, ORANGE, AQUA = "#a3a19b", "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2 = "#1d1d1b", "#5a5955"
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": "#c9c7c1", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


def style(ax, ylabel=None):
    ax.yaxis.grid(True, color="#e6e4df", lw=0.8); ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def bars(ax, groups, series, colors, labels, sd=None, ylim=(0.3, 0.95), fmt="{:.3f}", label_last=True):
    """Dot plot (position, not length, carries the value, so a non-zero axis is honest)."""
    n = len(series); w = 0.26; x = np.arange(len(groups))
    for i, (vals, c, lab) in enumerate(zip(series, colors, labels)):
        xs = x - w * (n - 1) / 2 + w * i
        e = sd[i] if sd is not None else None
        ax.errorbar(xs, vals, yerr=e, fmt="o", ms=8, color=c, ecolor=c, elinewidth=1.4, capsize=3,
                    label=lab, zorder=3, mec="white", mew=1.2)
        left = (i == 0 and n > 1)
        for j, (xx, v) in enumerate(zip(xs, vals)):
            ax.text(xx - 0.05 if left else xx + 0.05, v, fmt.format(v), ha="right" if left else "left",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(*ylim); ax.set_xlim(-0.6, len(groups) - 0.25)


def get(sub, m, k="pr"):
    return T["perf"][sub][m][k]


# ---------------------------------------------------------------- fig 6: main comparison
models = [("altman", "Altman Z''"), ("logit_ratios", "Logistic\nregression"), ("hgb_ratios", "Boosting:\nratios"),
          ("hgb_both", "Boosting:\nratios + language")]
if HAVE_DL:
    models += [("dl_main", "Deep model\n(7 seeds)"), ("blend", "Blend\n(full system)")]
fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=200)
g = [lab for _, lab in models]
s_all = [get("all", m) for m, _ in models]; s_str = [get("strict", m) for m, _ in models]
sd_all = [get("all", m, "pr_sd") for m, _ in models]; sd_str = [get("strict", m, "pr_sd") for m, _ in models]
bars(ax, g, [s_all, s_str], [GREY, BLUE], ["all 763 company-years", "strict matched 609"],
     sd=[sd_all, sd_str], ylim=(0.5, 0.95))
style(ax, "PR-AUC (mean of 5 folds)")
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9.5)
ax.set_title("Predictive performance by model", loc="left", fontsize=12.5, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_main.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 7: healthy-looking subsets
subs = [("all", "All company-years"), ("z_above_median", "Altman Z'' above\nthe median"),
        ("z_safe_zone", "Altman Z'' in the\n'safe' zone (> 2.6)"), ("ratio_model_low_risk", "Ratio model rates\nlow risk (lower half)")]
ser = [("hgb_ratios", "ratios only", GREY), ("hgb_both", "ratios + language", BLUE)]
if HAVE_DL:
    ser.append(("blend", "full system (blend)", ORANGE))
fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=200)
g = []
for s, lab in subs:
    n, p = T["subset_sizes"][s]["n"], T["subset_sizes"][s]["pos"]
    g.append(f"{lab}\n(n={n}, {p} insolvent)")
bars(ax, g, [[get(s, m) for s, _ in subs] for m, _, _ in ser], [c for _, _, c in ser], [l for _, l, _ in ser],
     sd=None, ylim=(0.2, 0.92))
# base rate markers
for i, (s, _) in enumerate(subs):
    br = T["subset_sizes"][s]["pos"] / T["subset_sizes"][s]["n"]
    ax.plot([i - 0.42, i + 0.42], [br, br], color="#77756f", lw=1.2, ls=(0, (3, 2)), zorder=4)
style(ax, "PR-AUC (mean of 5 folds)")
from matplotlib.lines import Line2D
h, l = ax.get_legend_handles_labels()
h.append(Line2D([0], [0], color="#77756f", lw=1.2, ls=(0, (3, 2)))); l.append("chance (share insolvent)")
ax.legend(h, l, frameon=False, loc="upper right", ncol=2, fontsize=9.5)
ax.set_title("Where the accounts look healthy, language adds the most", loc="left", fontsize=12.5,
             fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig7_healthy.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 8: horizon
hz = [("t-1", "t-1 (last report)"), ("t-2", "t-2"), ("t-3", "t-3 (three years before)")]
fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=200)
g = [f"{lab}\n(n={T['subset_sizes'][h]['n']})" for h, lab in hz]
bars(ax, g, [[get(h, m) for h, _ in hz] for m, _, _ in ser], [c for _, _, c in ser], [l for _, l, _ in ser],
     sd=None, ylim=(0.5, 0.95))
style(ax, "PR-AUC (mean of 5 folds)")
ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=9.5)
ax.set_title("How early the warning appears", loc="left", fontsize=12.5, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig8_horizon.png"); plt.close(fig)

# ---------------------------------------------------------------- fig 9: importance
import pandas as pd
imp = pd.read_csv(ROOT / "results_cpu" / "importance_families.csv").sort_values("pr_drop")
fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
cols = [GREY if f == "Accounting ratios" else BLUE for f in imp.family]
y = np.arange(len(imp))
ax.barh(y, imp.pr_drop, color=cols, height=0.62, zorder=2)
ax.errorbar(imp.pr_drop, y, xerr=imp.pr_drop_sd, fmt="none", ecolor="#77756f", elinewidth=1, capsize=2.5, zorder=3)
for yy, v, fp in zip(y, imp.pr_drop, imp.folds_positive):
    pass
for yy, v, sdv, fp in zip(y, imp.pr_drop, imp.pr_drop_sd, imp.folds_positive):
    ax.text(max(v + sdv, 0.004) + 0.006, yy, f"{v:+.3f}  ({fp}/5 folds)", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels(imp.family, fontsize=9)
ax.axvline(0, color="#8f8d87", lw=1)
ax.xaxis.grid(True, color="#e6e4df", lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("drop in PR-AUC when the family is shuffled (held-out folds)")
ax.set_xlim(-0.03, 0.40)
ax.set_title("What the ratios + language model relies on", loc="left", fontsize=12.5, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig9_importance.png"); plt.close(fig)
print("charts written; DL:", HAVE_DL)
