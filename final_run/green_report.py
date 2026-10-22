#!/usr/bin/env python3
"""Evaluation of the green-AI study, from the files run_green.py saved.

    python green_report.py --check-baseline   # Step 3 gate: the re-run baseline vs the expected PR-AUCs
    python green_report.py                    # green_results.csv, inference.csv, pareto.png, selection.json

Every metric is the per-fold mean (sd across the 5 frozen folds), as in analyze.py. "strict" =
the same out-of-fold predictions restricted to the 609 strict rows.

Energy (DECISIONS.md D4.3, D4.3a): the primary metric is GPU energy, `gpu_kwh` from NVML (codecarbon),
and CO2e computed from it alone (`gpu_kg_co2e`). Whole-machine totals (`kwh_total_ub`,
`kg_co2e_total_ub`, `j_per_doc_ub`) include a CPU share inflated by background load during the
14-15 October runs and are kept only as upper bounds, for the appendix.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bppfinal import green as gr, stats as st     # noqa: E402
from run_green import EMBED, INFER, SEEDS, reuses_baseline_runs   # noqa: E402

EXPECTED = {"gb_ratios": 0.750, "gb_both": 0.786, "dl_finbert_fp16_k7": 0.758, "blend_finbert_fp16_k7": 0.797}
REF_PR, TOL = 0.797, 0.010
CURRENT = "blend_finbert_fp16_k7"
KS = [1, 3, 5, 7]
LABEL = {
    "finbert_fp16": "FinBERT fp16 (current)", "finbert_fp32": "FinBERT fp32", "finbert_bf16": "FinBERT bf16",
    "finbert_fp16_n32": "FinBERT, 32 sent.", "finbert_fp16_n64": "FinBERT, 64 sent.",
    "finbert_fp16_n96": "FinBERT, 96 sent. (= network window)", "findistilroberta_fp16": "Fin-DistilRoBERTa",
    "minilm_fp16": "MiniLM-L6", "distilroberta_fp16": "DistilRoBERTa", "finbert_int8": "FinBERT INT8 (CPU)",
}
# which inference benchmark describes deploying each embedding variant
INFER_OF = {"finbert_fp16": "finbert_gpu_fp16_n200", "finbert_fp32": "finbert_gpu_fp32_n200",
            "finbert_fp16_n64": "finbert_gpu_fp16_n64", "finbert_fp16_n96": "finbert_gpu_fp16_n96",
            "minilm_fp16": "minilm_gpu_fp16_n200", "finbert_int8": "finbert_cpu_int8_n200",
            "findistilroberta_fp16": "findistilroberta_cpu_int8_n200",
            "distilroberta_fp16": "distilroberta_cpu_int8_n200"}


class Study:
    def __init__(self, out):
        self.out = Path(out)
        self.df = pd.read_pickle(self.out / "cache" / "table.pkl")
        self.y = self.df.label.to_numpy(float)
        self.F = self.df.fold.to_numpy()
        self.S = self.df.strict.to_numpy().astype(bool)
        self.G = self.df.pair_id.to_numpy()
        self.E = {p.stem: json.loads(p.read_text()) for p in (self.out / "energy").glob("*.json")}

    def oof(self, stem):
        f = self.out / "runs" / f"{stem}.npz"
        return np.load(f)["oof"].astype(float) if f.exists() else None

    def dl_runs(self, tag, exp="main"):
        """Out-of-fold predictions of the seeds run for `tag`. A sentence cap at or above the
        network's window (N_READ) whose embeddings were a bit-identical prefix of the baseline's has
        no runs of its own (run_green.py reuses the baseline's: same input, same computation)."""
        src = tag
        if reuses_baseline_runs(tag) and not (self.out / "runs" / f"dl_{tag}_{exp}_s0.npz").exists():
            src = "finbert_fp16"
        runs = [self.oof(f"dl_{src}_{exp}_s{s}") for s in SEEDS]
        stages = [f"dl_{src}_{exp}_s{s}" for s in SEEDS]
        n = next((i for i, r in enumerate(runs) if r is None), len(runs))
        return runs[:n], stages[:n], src

    def kwh(self, stages, field="energy_kwh"):
        return float(sum(self.E[s][field] for s in stages))

    def metrics(self, p):
        r = {}
        for sub, m in [("all", None), ("strict", self.S)]:
            for met in ["roc", "pr"]:
                v = st.per_fold(self.y, p, self.F, m, met)
                r[f"{met}_{sub}"] = float(np.nanmean(v)); r[f"{met}_{sub}_sd"] = float(np.nanstd(v, ddof=1))
        return r

    # ------------------------------------------------------------------ configs
    def configs(self):
        rows, preds = [], {}
        base = ["table_build"]

        def add(name, kind, p, stages, **meta):
            preds[name] = p
            r = dict(config=name, kind=kind, **meta, **self.metrics(p))
            r["stages"] = ";".join(stages)
            # primary: GPU energy (NVML) and CO2e from it alone
            r["gpu_kwh"] = self.kwh(stages, "gpu_kwh")
            r["gpu_kg_co2e"] = float(sum(gr.gpu_co2e(self.E[s]) for s in stages))
            r["gpu_kwh_smi"] = self.kwh(stages, "gpu_kwh_smi")
            for part, pred in [("embed", lambda s: s.startswith("embed_")), ("dl", lambda s: s.startswith("dl_")),
                               ("gb", lambda s: s.startswith("gb_")), ("table", lambda s: s == "table_build")]:
                r[f"gpu_kwh_{part}"] = self.kwh([s for s in stages if pred(s)], "gpu_kwh")
            # upper bounds: whole-machine totals, CPU share inflated by background load (D4.3)
            for fld, col in [("energy_kwh", "kwh_total_ub"), ("kg_co2e", "kg_co2e_total_ub"),
                             ("cpu_kwh", "cpu_kwh_ub"), ("ram_kwh", "ram_kwh_ub"), ("wall_s", "wall_s")]:
                r[col] = self.kwh(stages, fld)
            r["gpu_peak_mem_mb"] = max(self.E[s].get("gpu_peak_mem_mb", 0) for s in stages)
            rows.append(r)

        for nm, lab in [("ratios", "GB, ratios only"), ("both", "GB, ratios + language (no neural model)")]:
            add(f"gb_{nm}", "gb", self.oof(f"gb_{nm}"), base + [f"gb_{nm}"], label=lab, variant="g" if nm == "both" else "baseline",
                encoder="none", precision="-", n_sents=0, seeds=0, n_params=0, device="cpu")
        gb_both = self.oof("gb_both")
        for tag, (model, prec, cap, dev, _) in EMBED.items():
            emb = f"embed_{tag}_mdna"
            if emb not in self.E:
                continue
            runs, stages, src = self.dl_runs(tag)
            variant = ("baseline" if tag == "finbert_fp16" else "c" if prec in ("fp32", "bf16") else
                       "d" if cap < 200 else "f" if prec == "int8" else "b")
            for k in KS:
                if len(runs) < k:
                    continue
                dlp = gr.seed_ensemble(runs, self.F, k)
                meta = dict(variant="e" if (tag == "finbert_fp16" and k < 7) else variant, encoder=model,
                            precision=prec, n_sents=cap, seeds=k, n_params=self.E[emb].get("n_params"),
                            device=dev, reused_runs_from=src if src != tag else "")
                lab = LABEL[tag] + ("" if k == 7 else f", {k} seed{'s' if k > 1 else ''}")
                add(f"dl_{tag}_k{k}", "dl", dlp, base + [emb] + stages[:k], label="DL ensemble: " + lab, **meta)
                add(f"blend_{tag}_k{k}", "blend", gr.blend(dlp, gb_both, self.F),
                    base + [emb] + stages[:k] + ["gb_both"], label="Blend: " + lab, **meta)
        return rows, preds

    def strict_trained(self):
        """The baseline's separately trained 609-row network and blend (as in analyze.py)."""
        runs, stages, _ = self.dl_runs("finbert_fp16", "strict")
        if len(runs) < 7:
            return None
        dls = gr.seed_ensemble(runs, self.F, 7)
        b = gr.blend(dls, self.oof("gb_both_strict"), self.F)
        r = {}
        for nm, p, stg in [("dl_strict_trained", dls, stages), ("blend_strict_trained", b, stages + ["gb_both_strict"])]:
            v = {met: st.per_fold(self.y, p, self.F, self.S, met) for met in ["roc", "pr"]}
            all_stg = ["table_build", "embed_finbert_fp16_mdna"] + stg
            r[nm] = dict(roc_strict=float(np.nanmean(v["roc"])), roc_strict_sd=float(np.nanstd(v["roc"], ddof=1)),
                         pr_strict=float(np.nanmean(v["pr"])), pr_strict_sd=float(np.nanstd(v["pr"], ddof=1)),
                         gpu_kwh=self.kwh(all_stg, "gpu_kwh"),
                         gpu_kg_co2e=float(sum(gr.gpu_co2e(self.E[s]) for s in all_stg)),
                         kwh_total_ub=self.kwh(all_stg), kg_co2e_total_ub=self.kwh(all_stg, "kg_co2e"))
        return r


def check_baseline(s: Study):
    """Step 3: re-run baseline vs the expected PR-AUCs (all 763 rows)."""
    runs, _, _ = s.dl_runs("finbert_fp16")
    got = {"gb_ratios": s.oof("gb_ratios"), "gb_both": s.oof("gb_both")}
    if len(runs) == 7:
        got["dl_finbert_fp16_k7"] = gr.seed_ensemble(runs, s.F, 7)
        if got["gb_both"] is not None:
            got["blend_finbert_fp16_k7"] = gr.blend(got["dl_finbert_fp16_k7"], got["gb_both"], s.F)
    res = {}
    for k, exp in EXPECTED.items():
        p = got.get(k)
        if p is None:
            res[k] = dict(expected=exp, got=None, ok=None); continue
        v = float(np.nanmean(st.per_fold(s.y, p, s.F, None, "pr")))
        res[k] = dict(expected=exp, got=round(v, 4), diff=round(v - exp, 4), ok=abs(v - exp) <= 0.01)
    # the same four numbers from the saved run in ~/bpp_final/results, for comparison
    return res


# network runs in the original experiment grid (run_gpu.py EXPS + leak test) and the sections each reads
GRID = {"main": (7, ("mdna",)), "strict": (7, ("mdna",)), "notext": (5, ()), "text_mdna": (5, ("mdna",)),
        "text_aud": (5, ("auditor_report",)), "text_both": (5, ("mdna", "auditor_report")),
        "fusion_aud": (5, ("auditor_report",)), "fusion_both": (5, ("mdna", "auditor_report")), "leak": (1, ("mdna",))}


def caching_savings(E, field="gpu_kwh"):
    """Variant a: embedding once and reusing vs re-embedding inside every network run. In GPU energy
    (the primary metric); `field="energy_kwh"` gives the whole-machine upper bound."""
    e = {sec: E[f"embed_finbert_fp16_{sec}"][field] for sec in ("mdna", "auditor_report")}
    load = E["cache_load_finbert_fp16_mdna"][field]
    n_runs = sum(n for n, _ in GRID.values())
    reembed = sum(n * sum(e[s] for s in secs) for n, secs in GRID.values())
    main7 = 7 * e["mdna"]
    return dict(metric=field, embed_mdna_kwh=e["mdna"], embed_auditor_kwh=e["auditor_report"], cache_load_kwh=load,
                main_7_seeds=dict(cached_kwh=e["mdna"] + 7 * load, reembed_kwh=main7,
                                  saved_kwh=main7 - e["mdna"] - 7 * load),
                full_grid=dict(n_runs=n_runs, cached_kwh=sum(e.values()) + n_runs * load, reembed_kwh=reembed,
                               saved_kwh=reembed - sum(e.values()) - n_runs * load),
                auditor_not_used_by_blend_kwh=e["auditor_report"])


def inference_table(s: Study):
    rows = []
    for name, (model, prec, dev, cap) in INFER.items():
        e = s.E.get(f"infer_{name}")
        if not e:
            continue
        n = e["n_docs"]
        rows.append(dict(benchmark=name, encoder=model or "none (GB only)", precision=prec, device=dev,
                         sentences=cap, n_params=e.get("n_params"), n_docs=n,
                         ms_per_doc=1000 * e["s_per_doc"], ms_per_doc_median=1000 * e["s_per_doc_median"],
                         ms_per_doc_p95=1000 * e["s_per_doc_p95"],
                         gpu_j_per_doc=e["gpu_kwh"] / n * 3.6e6, gpu_mg_co2e_per_doc=gr.gpu_co2e(e) / n * 1e6,
                         j_per_doc_ub=e["kwh_per_doc"] * 3.6e6, mg_co2e_per_doc_ub=e["kg_co2e_per_doc"] * 1e6,
                         # CPU benchmarks ran with ~11 cores taken by background processes (D4.3):
                         # their latency and energy are both upper bounds
                         cpu_bound_under_load=(dev == "cpu"),
                         gpu_peak_mem_mb=e["gpu_peak_mem_mb"], cpu_util_mean=e.get("cpu_util_mean"), method=e["method"]))
    return pd.DataFrame(rows)


def pareto_plot(R, chosen, path):
    """PR-AUC against kWh (log x). Points are numbered; the key beside the plot names each one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = R.assign(kwh=R.gpu_kwh)      # the plot's energy axis is GPU energy (NVML), D4.3a
    show = R[(R.kind == "gb") | ((R.kind == "blend") & ((R.seeds == 7) | R.config.str.startswith("blend_finbert_fp16_k")))
             | (R.config == "dl_finbert_fp16_k7") | R.pareto | (R.config == chosen)].copy()
    show = show.sort_values(["kwh", "pr_all"]).reset_index(drop=True)
    show["num"] = np.arange(1, len(show) + 1)
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    col = {"blend": "#2a78d6", "dl": "#eb6834", "gb": "#1baf7a"}
    name = {"gb": "Gradient boosting only (CPU)", "dl": "Network ensemble only", "blend": "Blend (network + GB)"}
    fig = plt.figure(figsize=(11.5, 6.0), dpi=160)
    ax = fig.add_axes([0.07, 0.10, 0.56, 0.80])
    front = R[R.pareto].sort_values("kwh")
    ax.step(front.kwh, front.pr_all, where="post", color=muted, lw=1.2, ls="--", zorder=2,
            label="Pareto front (over all %d configs)" % len(R))
    for kind in ["gb", "dl", "blend"]:
        g = show[show.kind == kind]
        ax.scatter(g.kwh, g.pr_all, s=46, c=col[kind], label=name[kind], zorder=3, edgecolor="white", linewidth=1.5)
    ch = show[show.config == chosen]
    ax.scatter(ch.kwh, ch.pr_all, marker="*", s=420, facecolor="none", edgecolor=ink, lw=1.2, zorder=4,
               label="chosen green config")
    cur = show[show.config == CURRENT]
    ax.scatter(cur.kwh, cur.pr_all, marker="D", s=120, facecolor="none", edgecolor=ink, lw=1.2, zorder=4,
               label="current blend")
    ax.axhline(REF_PR - TOL, color=muted, ls=":", lw=1)
    ax.text(show.kwh.min() * 1.15, REF_PR - TOL - 0.0007, f"PR-AUC {REF_PR - TOL:.3f} (selection threshold)",
            fontsize=7, color=muted, ha="left", va="top")
    # label placement: try offsets around each point, keep the first that clears earlier labels and points
    ax.set_xscale("log")
    fig.canvas.draw()
    pts = ax.transData.transform(np.c_[show.kwh, show.pr_all])
    placed = [tuple(p) for p in pts]
    cand = [(7, 5), (7, -11), (-14, 5), (-14, -11), (0, 11), (0, -17), (14, 0), (-21, 0), (16, 12), (-23, 12),
            (16, -18), (-23, -18), (0, 22), (0, -28)]
    scale = fig.dpi / 72
    for (i, r), p in zip(show.iterrows(), pts):
        best = cand[0]
        for dx, dy in cand:
            q = (p[0] + dx * scale + 4 * scale, p[1] + dy * scale + 3 * scale)
            if all(abs(q[0] - o[0]) > 11 * scale or abs(q[1] - o[1]) > 8 * scale for o in placed):
                best = (dx, dy); break
        placed.append((p[0] + best[0] * scale + 4 * scale, p[1] + best[1] * scale + 3 * scale))
        ax.annotate(str(r.num), (r.kwh, r.pr_all), textcoords="offset points", xytext=best, fontsize=7, color=ink,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=muted, shrinkA=0, shrinkB=3)
                    if abs(best[0]) + abs(best[1]) > 20 else None)
    ax.set_xlabel("GPU energy for the 5-fold out-of-fold predictions (kWh, log scale; NVML via codecarbon)", fontsize=8.5, color=ink)
    ax.set_ylabel("PR-AUC, all 763 company-years (mean of 5 folds)", fontsize=8.5, color=ink)
    ax.tick_params(labelsize=7.5, colors=muted)
    for sp in ax.spines.values():
        sp.set_color(grid)
    ax.grid(True, which="both", color=grid, lw=0.6)
    ax.legend(fontsize=7, loc="lower center", frameon=False, markerscale=0.6)
    ax.set_title("Accuracy versus GPU energy, same 5 folds and seeds for every configuration", fontsize=9.5, color=ink, loc="left")
    kx = fig.add_axes([0.655, 0.10, 0.34, 0.80]); kx.axis("off")
    kx.text(0, 1.0, f"{'#':<3s} {'configuration':<48s} {'PR-AUC':>6s} {'GPU Wh':>6s}", fontsize=6.6, family="monospace",
            color=muted, va="top")
    kx.text(0, -0.02, "P = on the Pareto front.  GPU Wh = NVML reading, including the\ncard's idle draw (6-7 W) during CPU-only stages.",
            fontsize=6.6, family="monospace", color=muted, va="top")
    for j, (_, r) in enumerate(show.iterrows()):
        lab = r.label.replace("Blend: ", "").replace("DL ensemble: ", "Network only: ").replace("GB, ", "GB: ")
        lab = (lab[:44] + "…") if len(lab) > 45 else lab
        lab = ("P " if r.pareto else "  ") + lab
        kx.text(0, 0.96 - j * 0.96 / max(len(show), 22), f"{r.num:<3d} {lab:<48s} {r.pr_all:6.3f} {1000 * r.kwh:6.1f}",
                fontsize=6.6, family="monospace", color=ink, va="top")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


REPORT_ROWS = ["blend_finbert_fp16_k7", "dl_finbert_fp16_k7", "gb_ratios", "gb_both", "blend_finbert_fp16_k3",
               "blend_minilm_fp16_k7", "blend_distilroberta_fp16_k3", "blend_finbert_fp32_k7",
               "blend_finbert_fp16_n32_k1", "blend_finbert_int8_k7"]


def report_json(R, sel, inf, out):
    """The numbers the report section (report/green_section.js) reads."""
    T = R.set_index("config")
    num = lambda x: None if pd.isna(x) else float(x)      # JSON has no NaN
    rows = [dict(config=c, label=T.loc[c, "label"], **{k: num(T.loc[c, k]) for k in
            ["pr_all", "pr_all_sd", "roc_all", "pr_strict", "gpu_kwh", "gpu_kg_co2e", "gpu_kwh_saving_pct",
             "gpu_kwh_embed", "gpu_kwh_dl", "kwh_total_ub", "kg_co2e_total_ub", "wall_s", "boot_lo", "boot_hi",
             "infer_ms_per_doc", "infer_gpu_j_per_doc", "infer_j_per_doc_ub"]},
            infer_benchmark="" if pd.isna(T.loc[c, "infer_benchmark"]) else str(T.loc[c, "infer_benchmark"]),
            pareto=bool(T.loc[c, "pareto"]), chosen=bool(T.loc[c, "chosen"])) for c in REPORT_ROWS + [sel["chosen"]]
            if c in T.index]
    rows = list({r["config"]: r for r in rows}.values())
    I = inf.set_index("benchmark")
    (Path(out) / "report_numbers.json").write_text(json.dumps(dict(
        rows=rows, selection={k: v for k, v in sel.items() if k != "strict_trained_baseline"},
        n_configs=len(R), energy_metric="gpu_kwh (NVML); whole-machine totals are upper bounds (D4.3)",
        inference={b: dict(ms=float(I.loc[b, "ms_per_doc"]), gpu_j=float(I.loc[b, "gpu_j_per_doc"]),
                           j_ub=float(I.loc[b, "j_per_doc_ub"]), device=str(I.loc[b, "device"]),
                           cpu_bound_under_load=bool(I.loc[b, "cpu_bound_under_load"])) for b in I.index}),
        indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "results" / "green"))
    ap.add_argument("--check-baseline", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ignore-gate", action="store_true", help="testing only (smoke outputs)")
    a = ap.parse_args()
    s = Study(a.out)
    chk = check_baseline(s)
    (s.out / "baseline_check.json").write_text(json.dumps(chk, indent=1))
    print(json.dumps(chk, indent=1))
    if a.check_baseline:
        return
    if not all(v["ok"] for v in chk.values()) and not a.ignore_gate:
        sys.exit("baseline not reproduced within 0.01: see baseline_check.json (Step 3 says stop)")

    rows, preds = s.configs()
    R = pd.DataFrame(rows)
    cur = preds[CURRENT]
    boots, wil = [], []
    for _, r in R.iterrows():
        if r.config == CURRENT:
            boots.append(dict(point=0.0, lo=0.0, hi=0.0, p_gt0=np.nan)); wil.append(dict(diff=0.0, wilcoxon_p=np.nan, wins=0)); continue
        boots.append(st.pair_bootstrap(s.y, preds[r.config], cur, s.F, s.G, None, "pr", n_boot=a.n_boot, seed=0))
        wil.append(st.paired(s.y, preds[r.config], cur, s.F, None, "pr"))
    R["d_pr_vs_current"] = R.pr_all - R.loc[R.config == CURRENT, "pr_all"].iloc[0]
    R["boot_point"] = [b["point"] for b in boots]
    R["boot_lo"] = [b["lo"] for b in boots]
    R["boot_hi"] = [b["hi"] for b in boots]
    R["wilcoxon_p"] = [w["wilcoxon_p"] for w in wil]
    R["folds_won"] = [w["wins"] for w in wil]
    base = R[R.config == CURRENT].iloc[0]
    R["gpu_kwh_saving_pct"] = 100 * (1 - R.gpu_kwh / base.gpu_kwh)
    R["gpu_kwh_multiple"] = base.gpu_kwh / R.gpu_kwh
    R["kwh_total_ub_saving_pct"] = 100 * (1 - R.kwh_total_ub / base.kwh_total_ub)
    R["pareto"] = gr.pareto_front(R.gpu_kwh, R.pr_all)
    inf = inference_table(s)
    if len(inf):
        tag_of = R.config.str.replace(r"^(dl|blend)_", "", regex=True).str.replace(r"_k\d$", "", regex=True)
        m = inf.set_index("benchmark")
        R["infer_benchmark"] = [INFER_OF.get(t, "") if k != "gb" else "gb_only_cpu" for t, k in zip(tag_of, R.kind)]
        R["infer_ms_per_doc"] = [m.ms_per_doc.get(b, np.nan) if b else np.nan for b in R.infer_benchmark]
        R["infer_gpu_j_per_doc"] = [m.gpu_j_per_doc.get(b, np.nan) if b else np.nan for b in R.infer_benchmark]
        R["infer_j_per_doc_ub"] = [m.j_per_doc_ub.get(b, np.nan) if b else np.nan for b in R.infer_benchmark]
    # the pre-registered rule, applied to GPU energy (D4.3a): "lowest kWh" = lowest gpu_kwh
    name, why = gr.choose_config([dict(name=r.config, kwh=r.gpu_kwh, pr_all=r.pr_all, boot_lo=r.boot_lo,
                                       boot_hi=r.boot_hi) for _, r in R.iterrows()], REF_PR, TOL)
    R["eligible"] = [(r.pr_all >= REF_PR - TOL - 1e-12) or (r.boot_lo <= 0 <= r.boot_hi) for _, r in R.iterrows()]
    R["chosen"] = R.config == name
    R = R.sort_values("gpu_kwh").reset_index(drop=True)
    R.to_csv(s.out / "green_results.csv", index=False, float_format="%.6g")
    inf.to_csv(s.out / "inference.csv", index=False, float_format="%.6g")
    pareto_plot(R, name, s.out / "pareto.png")
    c = R[R.config == name].iloc[0]
    pct, mult = gr.reduction(base.gpu_kwh, c.gpu_kwh)
    pct_c, mult_c = gr.reduction(base.gpu_kg_co2e, c.gpu_kg_co2e)
    pct_ub, mult_ub = gr.reduction(base.kwh_total_ub, c.kwh_total_ub)
    MET = ["roc_all", "roc_all_sd", "pr_all", "pr_all_sd", "roc_strict", "roc_strict_sd", "pr_strict", "pr_strict_sd",
           "gpu_kwh", "gpu_kg_co2e", "gpu_kwh_smi", "kwh_total_ub", "kg_co2e_total_ub", "wall_s"]
    sel = dict(rule=f"lowest GPU kWh (NVML) with pr_all >= {REF_PR}-{TOL} or bootstrap CI vs current blend containing 0",
               energy_metric="gpu_kwh", chosen=name, label=c.label, reason=why, current=CURRENT,
               chosen_metrics={k: float(c[k]) for k in MET},
               current_metrics={k: float(base[k]) for k in MET},
               pr_diff=float(c.pr_all - base.pr_all),
               bootstrap=dict(point=float(c.boot_point), lo=float(c.boot_lo), hi=float(c.boot_hi), n_boot=a.n_boot),
               wilcoxon=st.paired(s.y, preds[name], cur, s.F, None, "pr"),
               wilcoxon_roc=st.paired(s.y, preds[name], cur, s.F, None, "roc"),
               energy_saving_pct=pct, energy_multiple=mult, co2_saving_pct=pct_c, co2_multiple=mult_c,
               total_upper_bound=dict(saving_pct=pct_ub, multiple=mult_ub,
                                      note="whole-machine codecarbon totals; CPU share inflated by background load (D4.3)"),
               strict_trained_baseline=s.strict_trained(), n_configs=len(R), n_eligible=int(R.eligible.sum()))
    sel["caching"] = caching_savings(s.E)
    sel["caching_total_upper_bound"] = caching_savings(s.E, "energy_kwh")
    (s.out / "selection.json").write_text(json.dumps(sel, indent=1, default=float))
    report_json(R, sel, inf, s.out)
    print(json.dumps({k: v for k, v in sel.items() if k != "strict_trained_baseline"}, indent=1, default=float))


if __name__ == "__main__":
    main()
