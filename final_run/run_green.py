#!/usr/bin/env python3
"""Green-AI study: the current pipeline and its greener variants, with energy measured per stage.

    python run_green.py --smoke --out /tmp/green_smoke      # ~5 min check of every code path
    python run_green.py                                     # the full study (resumable)
    python run_green.py --stages table,embed,dl,gb,infer    # any subset, in this order

Reads data/ (the processed/ files) and folds.csv exactly as run_gpu.py does. Writes only to
--out (default: results/green/ in the repository): energy records per stage (energy/*.json),
out-of-fold predictions per run (runs/*.npz), embedding caches (cache/, not in git) and the log
(green_run.log). A stage whose output already exists is skipped. The evaluation, the table, the
figure and the selection are made from these files by green_report.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.append(str(REPO / "src"))

from bppfinal import green as gr     # noqa: E402

FINBERT = "ProsusAI/finbert"
FIN_DISTIL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
DISTILROBERTA = "distilroberta-base"

# tag -> (model, precision, cap on sentences per section at embedding time, device, sections).
# The network reads only the first N_READ slots (dl.CFG["n_sents"]), so a cap above N_READ cannot
# change its input: the sentence-cap ladder therefore stops at N_READ, where embedding exactly what
# the network reads is a free saving against the current 200.
EMBED = {
    "finbert_fp16":       (FINBERT, "fp16", 200, "cuda", ("mdna", "auditor_report")),   # current
    "finbert_fp32":       (FINBERT, "fp32", 200, "cuda", ("mdna",)),
    "finbert_bf16":       (FINBERT, "bf16", 200, "cuda", ("mdna",)),
    "finbert_fp16_n32":   (FINBERT, "fp16", 32, "cuda", ("mdna",)),
    "finbert_fp16_n64":   (FINBERT, "fp16", 64, "cuda", ("mdna",)),
    "finbert_fp16_n96":   (FINBERT, "fp16", 96, "cuda", ("mdna",)),     # = N_READ
    "findistilroberta_fp16": (FIN_DISTIL, "fp16", 200, "cuda", ("mdna",)),
    "minilm_fp16":        (MINILM, "fp16", 200, "cuda", ("mdna",)),
    "distilroberta_fp16": (DISTILROBERTA, "fp16", 200, "cuda", ("mdna",)),
    "finbert_int8":       (FINBERT, "int8", 200, "cpu", ("mdna",)),
}
N_SLOTS = 200
N_READ = 96      # dl.CFG["n_sents"]: the network reads slots 0-95 only (asserted in Run.dl_runs)
SEEDS = list(range(7))


def reuses_baseline_runs(tag):
    """Sentence-cap variants of the baseline encoder whose cap is at or above the network window: if
    their embeddings are a bit-identical prefix of the baseline's (checked), training on them is the
    same computation as the baseline's, so the baseline's runs stand for theirs (DECISIONS.md D3.d)."""
    return tag.startswith("finbert_fp16_n") and EMBED[tag][2] >= N_READ
# per-document inference benchmarks: name -> (model or None for GB only, precision, device, sentences)
INFER = {
    "finbert_gpu_fp16_n200": (FINBERT, "fp16", "cuda", 200),      # current
    "finbert_gpu_fp32_n200": (FINBERT, "fp32", "cuda", 200),
    "finbert_gpu_fp16_n96": (FINBERT, "fp16", "cuda", 96),
    "finbert_gpu_fp16_n64": (FINBERT, "fp16", "cuda", 64),
    "finbert_cpu_fp32_n200": (FINBERT, "fp32", "cpu", 200),
    "finbert_cpu_int8_n200": (FINBERT, "int8", "cpu", 200),
    "finbert_cpu_int8_n96": (FINBERT, "int8", "cpu", 96),
    "minilm_gpu_fp16_n200": (MINILM, "fp16", "cuda", 200),
    "minilm_cpu_fp32_n200": (MINILM, "fp32", "cpu", 200),
    "minilm_cpu_int8_n200": (MINILM, "int8", "cpu", 200),
    "findistilroberta_cpu_int8_n200": (FIN_DISTIL, "int8", "cpu", 200),
    "distilroberta_cpu_int8_n200": (DISTILROBERTA, "int8", "cpu", 200),
    "gb_only_cpu": (None, "-", "cpu", 0),
}


class Run:
    def __init__(self, args):
        self.a = args
        self.out = Path(args.out)
        for d in ["energy", "runs", "cache", "codecarbon"]:
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.logf = self.out / "green_run.log"

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(self.logf, "a") as f:
            f.write(line + "\n")

    def energy_path(self, stage):
        return self.out / "energy" / f"{stage}.json"

    def meter(self, stage, **extra):
        return gr.EnergyMeter(stage, self.out / "codecarbon", **extra)

    def done(self, stage, *files):
        return self.energy_path(stage).exists() and all(Path(f).exists() for f in files)

    def save(self, m):
        gr.save_record(m.record, self.energy_path(m.stage))
        r = m.record
        self.log(f"  [{r['stage']}] {r['wall_s']:.0f}s  {r['energy_kwh']*1000:.2f} Wh  "
                 f"{r['kg_co2e']*1000:.2f} g CO2e  GPU peak {r['gpu_peak_mem_mb']:.0f} MB  ({r['method']})")

    # ------------------------------------------------------------------ table
    def table(self):
        from bppfinal.table import build
        from run_gpu import table_checksum
        pkl, cols = self.out / "cache" / "table.pkl", self.out / "cache" / "columns.json"
        if not self.done("table_build", pkl, cols):
            self.log("table build (incl. fold-wise trigram-LM perplexity)")
            with self.meter("table_build") as m:
                df, R, L = build(Path(self.a.data), with_perplexity=True, folds_file=Path(self.a.folds))
            self.save(m)
            df.to_pickle(pkl)
            cols.write_text(json.dumps({"ratios": R, "language": L}, indent=1))
        df = pd.read_pickle(pkl)
        c = json.loads(cols.read_text())
        R, L = c["ratios"], c["language"]
        chk = table_checksum(df, R + L)
        ref = Path(self.a.ref_results) / "run_info.json"
        ref_chk = json.loads(ref.read_text()).get("table_checksum") if ref.exists() else None
        self.log(f"table: {len(df)} rows, strict {int(df.strict.sum())}, checksum {chk} "
                 f"(reference run: {ref_chk}) {'MATCH' if chk == ref_chk else 'DIFFERENT'}")
        if self.a.smoke:
            df = df.copy()
        self.df, self.R, self.L, self.checksum = df, R, L, chk
        return df

    # ------------------------------------------------------------------ embeddings
    def emb_files(self, tag, sec):
        c = self.out / "cache"
        return c / f"emb_{tag}_{sec}.npy", c / f"len_{tag}_{sec}.npy"

    def embed(self, tags=None):
        from bppfinal import dl
        import torch
        df = self.df
        n_docs = 12 if self.a.smoke else len(df)
        for tag, (model, prec, cap, dev, secs) in EMBED.items():
            if tags and tag not in tags:
                continue
            if dev == "cuda" and not torch.cuda.is_available():
                self.log(f"SKIP embed {tag}: no CUDA"); continue
            for sec in secs:
                stage = f"embed_{tag}_{sec}"
                fE, fL = self.emb_files(tag, sec)
                if self.done(stage, fE, fL):
                    continue
                sents = [dl.to_sentences(t, m=cap) for t in df[sec].iloc[:n_docs]]
                self.log(f"embed {tag} {sec}: {model} {prec} on {dev}, first {cap} sentences, "
                         f"{sum(map(len, sents))} sentences in {n_docs} docs")
                with self.meter(stage, model=model, precision=prec, cap=cap, device=dev, section=sec,
                                n_docs=n_docs, n_sentences=int(sum(map(len, sents)))) as m:
                    tok, enc, n_params = gr.load_encoder(model, dev, prec)
                    E = gr.embed_all(sents, tok, enc, dev, N_SLOTS, dl.MAX_TOKENS,
                                     np.float32 if prec == "fp32" else np.float16, log=self.log)
                    del enc
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                m.record["n_params"] = n_params
                m.record["dim"] = int(E.shape[-1])
                if n_docs < len(df):   # smoke: pad to the full table
                    E = np.concatenate([E, np.zeros((len(df) - n_docs,) + E.shape[1:], E.dtype)])
                    sents += [[]] * (len(df) - n_docs)
                np.save(fE, E); np.save(fL, np.array([len(s) for s in sents]))
                self.save(m)
                del E

    def load_emb(self, tag, sec="mdna"):
        fE, fL = self.emb_files(tag, sec)
        return np.load(fE, mmap_mode="r"), np.load(fL)

    def check_embeddings(self):
        """Faithfulness checks that need no training: the fresh FinBERT fp16 embeddings against the
        existing cache, fp16/bf16/int8 against fp32, and capped against full."""
        f = self.out / "embedding_checks.json"
        res = json.loads(f.read_text()) if f.exists() else {}
        ref_cache = Path(self.a.ref_cache)
        E16, L16 = self.load_emb("finbert_fp16")
        if "fresh_vs_existing_cache" not in res:
            idx = json.loads((ref_cache / "embstore_mdna_finbert_200x96.json").read_text())
            pos = {d: i for i, d in enumerate(idx)}
            old = np.load(ref_cache / "embstore_mdna_finbert_200x96.npy", mmap_mode="r")
            Eold = np.stack([old[pos[d]] for d in self.df.doc_id])
            res["fresh_vs_existing_cache"] = dict(
                bitwise_identical=bool(np.array_equal(Eold, np.asarray(E16))),
                max_abs_diff=float(np.max(np.abs(Eold.astype(np.float32) - np.asarray(E16, np.float32)))),
                mean_cos=gr.mean_cosine(Eold, E16, L16)[0])
        E32 = None
        for tag in ["finbert_fp16", "finbert_bf16", "finbert_int8"]:
            key = f"{tag}_vs_fp32"
            if key in res or not self.emb_files(tag, "mdna")[0].exists() or \
                    not self.emb_files("finbert_fp32", "mdna")[0].exists():
                continue
            if E32 is None:
                E32, L32 = self.load_emb("finbert_fp32")
            E, L = self.load_emb(tag)
            mc, mn = gr.mean_cosine(E32, E, L32)
            res[key] = dict(mean_cos=mc, min_cos=mn, passes_0_99=bool(mc > 0.99))
        for tag, (_, _, cap, _, _) in EMBED.items():
            if not tag.startswith("finbert_fp16_n"):
                continue
            key = f"n{cap}_prefix_identical"
            fE = self.emb_files(tag, "mdna")[0]
            if key in res or not fE.exists():
                continue
            Ec = np.load(fE, mmap_mode="r")
            res[key] = bool(np.array_equal(np.asarray(Ec[:, :cap]), np.asarray(E16[:, :cap])))
        f.write_text(json.dumps(res, indent=1))
        self.log(f"embedding checks: {json.dumps(res)}")
        return res

    # ------------------------------------------------------------------ network runs
    def dl_runs(self, tags=None, experiments=("main", "strict")):
        import torch
        from bppfinal import dl
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        assert dl.CFG["n_sents"] == N_READ, f"N_READ={N_READ} but the network reads {dl.CFG['n_sents']} slots"
        df = self.df
        y = df.label.to_numpy().astype(np.float32)
        STRICT = df.strict.to_numpy().astype(bool)
        Xl = df[self.L].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        Xr = df[self.R].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        T = dl.Trainer(y, df.pair_id.to_numpy(), df.fold.to_numpy(), Xl, Xr, dev)
        cfg = dict(dl.CFG, epochs=2) if self.a.smoke else dl.CFG
        seeds = [0] if self.a.smoke else SEEDS
        checks = json.loads((self.out / "embedding_checks.json").read_text()) \
            if (self.out / "embedding_checks.json").exists() else {}
        plan = [(t, "main", None) for t in EMBED] + [("finbert_fp16", "strict", STRICT)]
        for tag, exp, rows in plan:
            if (tags and tag not in tags) or exp not in experiments:
                continue
            cap = EMBED[tag][2]
            if reuses_baseline_runs(tag) and checks.get(f"n{cap}_prefix_identical"):
                self.log(f"dl {tag}: first {cap} slots bit-identical to the full run and the network "
                         f"reads {N_READ} -> reuses finbert_fp16 runs (no training)")
                continue
            if not self.emb_files(tag, "mdna")[0].exists():
                self.log(f"SKIP dl {tag}: no embeddings"); continue
            Emm, Ln = self.load_emb(tag)
            E = np.asarray(Emm)
            M = gr.prefix_mask(Ln, N_SLOTS)
            for s in seeds:
                stage = f"dl_{tag}_{exp}_s{s}"
                fo = self.out / "runs" / f"{stage}.npz"
                if self.done(stage, fo):
                    continue
                t0 = time.time()
                with self.meter(stage, config=tag, experiment=exp, seed=s) as m:
                    oof, gate, attn, info = T.run_cv(E, M, cfg, seed=s, rows=rows, streams=("t", "l", "r"))
                np.savez_compressed(fo, oof=oof, gate=gate, info=json.dumps(info))
                self.save(m)
                pr = _pr(y, oof, df.fold.to_numpy())
                self.log(f"  {stage}: PR {pr:.3f} ({time.time()-t0:.0f}s)")
            del E

    # ------------------------------------------------------------------ gradient boosting
    def gb(self):
        from bppfinal import cpu_models as cm
        df = self.df
        STRICT = df.strict.to_numpy().astype(bool)
        for name, cols, rows in [("ratios", self.R, None), ("both", self.R + self.L, None),
                                 ("ratios_strict", self.R, STRICT), ("both_strict", self.R + self.L, STRICT)]:
            stage = f"gb_{name}"
            fo = self.out / "runs" / f"{stage}.npz"
            if self.done(stage, fo):
                continue
            with self.meter(stage, n_features=len(cols)) as m:
                oof = cm.hgb_oof(df, cols, rows=rows)
            np.savez_compressed(fo, oof=oof)
            self.save(m)
            self.log(f"  {stage}: PR {_pr(df.label.to_numpy(), oof, df.fold.to_numpy()):.3f}")
        # the cost of reading cached embeddings instead of recomputing them
        stage = "cache_load_finbert_fp16_mdna"
        if not self.done(stage) and self.emb_files("finbert_fp16", "mdna")[0].exists():
            with self.meter(stage) as m:
                np.load(self.emb_files("finbert_fp16", "mdna")[0]).sum()
            self.save(m)

    # ------------------------------------------------------------------ Step 3 gate
    def gate(self):
        from green_report import Study, check_baseline
        res = check_baseline(Study(self.out))
        (self.out / "baseline_check.json").write_text(json.dumps(res, indent=1))
        ok = all(v["ok"] for v in res.values())
        self.log("baseline check (PR-AUC, all 763): " + "; ".join(
            f"{k} {v['got']} vs {v['expected']} ({'ok' if v['ok'] else 'OFF'})" for k, v in res.items()))
        if not ok:
            lines = ["# Step 3 stopped: the baseline was not reproduced", "",
                     "| Model | Expected | Re-run | Difference |", "| --- | --- | --- | --- |"]
            lines += [f"| {k} | {v['expected']} | {v['got']} | {v.get('diff')} |" for k, v in res.items()]
            lines += ["", "See DECISIONS.md D0.1 for which data version and saved run the expected values match."]
            (self.out / "REPRODUCTION_REPORT.md").write_text("\n".join(lines) + "\n")
        return ok

    # ------------------------------------------------------------------ per-document inference
    def infer(self, names=None):
        import torch
        from sklearn.ensemble import HistGradientBoostingClassifier
        from bppfinal import dl
        df = self.df
        rng = np.random.default_rng(0)
        sample = rng.choice(len(df), 3 if self.a.smoke else self.a.infer_docs, replace=False)
        X = df[self.R + self.L].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        gbm = HistGradientBoostingClassifier(random_state=0).fit(X, df.label.to_numpy())
        min_s = 3 if self.a.smoke else self.a.infer_seconds
        for name, (model, prec, dev, cap) in INFER.items():
            if names and name not in names:
                continue
            stage = f"infer_{name}"
            if self.done(stage):
                continue
            if dev == "cuda" and not torch.cuda.is_available():
                self.log(f"SKIP {stage}: no CUDA"); continue
            torch.set_num_threads(self.a.cpu_threads)
            if model is not None:
                tok, enc, n_params = gr.load_encoder(model, dev, prec)
                # the live scorer (scorer/models.py) runs one network per seed on every report
                nets = [dl.FusionNet(len(self.L), len(self.R), dim=enc.config.hidden_size).to(dev).eval()
                        for _ in SEEDS]
            else:
                n_params = 0

            def one(i):
                if model is not None:
                    ss = dl.to_sentences(df.mdna.iloc[i], m=cap)
                    e = gr.embed_doc(tok, enc, ss, dev, max(cap, dl.CFG["n_sents"]), dl.MAX_TOKENS,
                                     out_dtype=np.float32)[:dl.CFG["n_sents"]]
                    msk = gr.prefix_mask([len(ss)], dl.CFG["n_sents"])
                    with torch.no_grad():
                        for net in nets:
                            net(torch.from_numpy(e)[None].to(dev), torch.from_numpy(msk).to(dev),
                                torch.zeros(1, len(self.L), device=dev), torch.zeros(1, len(self.R), device=dev))
                gbm.predict_proba(X[i:i + 1])

            for i in sample[:2]:      # warm-up (CUDA kernels, oneDNN), not measured
                one(i)
            n, lat = 0, []
            with self.meter(stage, model=model or "none", precision=prec, device=dev, cap=cap,
                            n_params=n_params, threads=self.a.cpu_threads,
                            n_networks=len(SEEDS) if model else 0) as m:
                t_start = time.time()
                while n < len(sample) or time.time() - t_start < min_s:
                    t0 = time.time(); one(sample[n % len(sample)]); n += 1
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    lat.append(time.time() - t0)
            m.record.update(n_docs=n, s_per_doc=float(np.mean(lat)), s_per_doc_median=float(np.median(lat)),
                            s_per_doc_p95=float(np.percentile(lat, 95)),
                            kwh_per_doc=m.record["energy_kwh"] / n, kg_co2e_per_doc=m.record["kg_co2e"] / n)
            self.save(m)
            self.log(f"  {stage}: {n} docs, {np.mean(lat)*1000:.0f} ms/doc, "
                     f"{m.record['kwh_per_doc']*3.6e6:.1f} J/doc")
            if model is not None:
                del enc, nets
                if dev == "cuda":
                    torch.cuda.empty_cache()


def _pr(y, p, fold):
    from sklearn.metrics import average_precision_score
    v = []
    for k in range(5):
        mk = (fold == k) & ~np.isnan(p)
        if mk.sum() and len(set(y[mk])) == 2:
            v.append(average_precision_score(y[mk], p[mk]))
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data" / "data"))
    ap.add_argument("--folds", default=str(Path.home() / "bpp_final" / "folds.csv"))
    ap.add_argument("--ref-results", default=str(Path.home() / "bpp_final" / "results"))
    ap.add_argument("--ref-cache", default=str(Path.home() / "bpp_final" / "cache"))
    ap.add_argument("--out", default=str(REPO / "results" / "green"))
    ap.add_argument("--stages", default="table,embed,check,dl,gb,infer")
    ap.add_argument("--only", default=None, help="comma list of embedding tags / inference names")
    ap.add_argument("--infer-docs", type=int, default=40)
    ap.add_argument("--infer-seconds", type=float, default=45)
    ap.add_argument("--cpu-threads", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    r = Run(a)
    only = a.only.split(",") if a.only else None
    st = a.stages.split(",")
    r.log(f"==== run_green {' '.join(sys.argv[1:])}")
    t0 = time.time()
    r.table()
    if a.smoke or only:
        if "embed" in st:
            r.embed(only)
        if "check" in st and not a.smoke:
            r.check_embeddings()
        if "dl" in st:
            r.dl_runs(only)
        if "gb" in st:
            r.gb()
        if "infer" in st:
            r.infer(only)
    else:
        # Step 3 first: the current pipeline, then the gate, then the variants
        r.embed(["finbert_fp16"])
        r.dl_runs(["finbert_fp16"], ("main",))
        r.gb()
        if not r.gate():
            r.log("==== STOPPED: baseline not reproduced within 0.01 (see REPRODUCTION_REPORT.md)")
            sys.exit(2)
        r.dl_runs(["finbert_fp16"], ("strict",))
        variants = [t for t in EMBED if t != "finbert_fp16"]
        gpu_first = [t for t in variants if EMBED[t][3] == "cuda"] + [t for t in variants if EMBED[t][3] != "cuda"]
        for t in gpu_first:
            r.embed([t])
            r.check_embeddings()
            r.dl_runs([t], ("main",))
        r.infer()
    r.log(f"==== done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
