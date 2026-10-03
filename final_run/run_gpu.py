#!/usr/bin/env python3
"""Final deep-learning run for the bankruptcy-prediction project (replaces notebook 2's GPU parts).

    python run_gpu.py --smoke     # ~3 min: checks the GPU, FinBERT and the training code
    python run_gpu.py             # the full run (resumable: re-running skips finished runs)
    python run_gpu.py --out results_v2 --reuse results
                                  # re-run after a change to the language features or ratios:
                                  # copies the text-only runs (which read neither) from results/

Reads data/ (the processed/ files) and folds.csv. Writes results/ - send that folder back.
Everything that does not need a GPU (gradient boosting, statistics, the report) is done elsewhere
from these outputs, so this script only trains networks and saves their out-of-fold predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))          # bpp.nlp.lm (standard library only)
sys.path.append(str(HERE.parent / "src"))          # inside the repository: the bpp package itself

from bppfinal.table import build                     # noqa: E402


def table_checksum(df, cols):
    """Same function on both machines: proves the GPU box built the identical table."""
    a = df[cols].apply(pd.to_numeric, errors="coerce").round(6).fillna(-999).to_numpy()
    h = hashlib.sha256(a.tobytes() + df.fold.to_numpy().astype(np.int64).tobytes()
                       + df.label.to_numpy().astype(np.int64).tobytes()).hexdigest()
    return h[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--cache", default=str(HERE / "cache"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--quick", action="store_true", help="3 seeds instead of 5/7 (if GPU time is short)")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--reuse", default=None,
                    help="earlier results folder: copy its text-only runs, which use no language "
                         "features or ratios and so are unchanged by a fix to either")
    args = ap.parse_args()

    import torch
    from bppfinal import dl
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "none"
    print(f"torch {torch.__version__}  device {dev}  GPU {gpu}", flush=True)
    if dev != "cuda" and not args.allow_cpu:
        sys.exit("No CUDA GPU visible to torch. Check `nvidia-smi` and the torch install "
                 "(or pass --allow-cpu, which is very slow).")
    if dev == "cuda":
        cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
        if cap not in torch.cuda.get_arch_list():
            sys.exit(f"This torch build has no kernels for the {gpu} ({cap}). RTX 50-series cards need "
                     "a CUDA 12.8 build: pip install --force-reinstall torch "
                     "--index-url https://download.pytorch.org/whl/cu128")

    out = Path(args.out); (out / "runs").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    df, RATIOS, LANG = build(Path(args.data), with_perplexity=True, folds_file=HERE / "folds.csv")
    chk = table_checksum(df, RATIOS + LANG)
    print(f"table checksum {chk}", flush=True)

    y = df.label.to_numpy().astype(np.float32)
    STRICT = df.strict.to_numpy().astype(bool)
    Xl = df[LANG].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    Xr = df[RATIOS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    T = dl.Trainer(y, df.pair_id.to_numpy(), df.fold.to_numpy(), Xl, Xr, dev)

    # ------------------------------------------------------------------ smoke test
    if args.smoke:
        sub = df.head(8).copy()
        E8, L8 = dl.embed(sub, "mdna", Path(args.cache), dev)
        print(f"FinBERT ok: {E8.shape}, sentences {L8.tolist()}")
        rng = np.random.default_rng(0)
        Efake = rng.standard_normal((len(df), dl.MAX_SENTS, dl.DIM)).astype(np.float16)
        Mfake = np.arange(dl.MAX_SENTS)[None, :] < rng.integers(0, 150, len(df))[:, None]
        cfg = dict(dl.CFG, epochs=2)
        for streams in [("t", "l", "r"), ("t",), ("l", "r")]:
            te, p, g, a, inf = T.train_fold(0, Efake, Mfake, cfg, seed=0, streams=streams)
            print(f"train ok {streams}: {len(te)} test rows, mean p {p.mean():.3f}, gate {g.mean(0).round(2)}, "
                  f"attn rows sum {np.nanmean(a.sum(1)):.2f}")
        te, p, g, a, inf = T.train_fold(0, Efake, Mfake, cfg, seed=0, rows=STRICT)
        print(f"strict ok: {len(te)} test rows")
        print(f"\nSMOKE TEST PASSED in {time.time()-t_start:.0f}s. Now run:  python run_gpu.py")
        return

    # ------------------------------------------------------------------ embeddings
    t0 = time.time()
    E_mdna, L_mdna = dl.embed(df, "mdna", Path(args.cache), dev)
    E_aud, L_aud = dl.embed(df, "auditor_report", Path(args.cache), dev)
    print(f"embeddings done in {time.time()-t0:.0f}s: {E_mdna.shape}", flush=True)
    M_mdna = np.arange(dl.MAX_SENTS)[None, :] < L_mdna[:, None]
    M_aud = np.arange(dl.MAX_SENTS)[None, :] < L_aud[:, None]
    # mean-pooled report vectors (all real sentences) for the averaged-embedding baseline (Mai et al.)
    for nm, E, M in [("mdna", E_mdna, M_mdna), ("auditor", E_aud, M_aud)]:
        s = (E.astype(np.float32) * M[:, :, None]).sum(1)
        np.save(out / f"emb_mean_{nm}.npy", (s / np.maximum(M.sum(1), 1)[:, None]).astype(np.float32))
    # both sections: up to 96 MD&A sentences followed directly by up to 96 auditor sentences
    # (notebook 2, H4). Kept contiguous, because the GRU reads the first `length` slots.
    E_both = np.zeros_like(E_mdna)
    L_both = np.minimum(L_mdna, 96) + np.minimum(L_aud, 96)
    for i in range(len(df)):
        a, b = min(L_mdna[i], 96), min(L_aud[i], 96)
        E_both[i, :a] = E_mdna[i, :a]
        E_both[i, a:a + b] = E_aud[i, :b]
    M_both = np.arange(dl.MAX_SENTS)[None, :] < L_both[:, None]

    # ------------------------------------------------------------------ experiments
    S7 = list(range(3 if args.quick else 7))
    S5 = list(range(3 if args.quick else 5))
    C = dl.CFG
    EXPS = [  # name, E, M, cfg, streams, rows, seeds
        ("main",        E_mdna, M_mdna, C, ("t", "l", "r"), None, S7),
        ("strict",      E_mdna, M_mdna, C, ("t", "l", "r"), STRICT, S7),
        ("notext",      E_mdna, M_mdna, C, ("l", "r"), None, S5),
        ("text_mdna",   E_mdna, M_mdna, C, ("t",), None, S5),
        ("text_aud",    E_aud, M_aud, C, ("t",), None, S5),
        ("text_both",   E_both, M_both, dict(C, n_sents=192), ("t",), None, S5),
        ("fusion_aud",  E_aud, M_aud, C, ("t", "l", "r"), None, S5),
        ("fusion_both", E_both, M_both, dict(C, n_sents=192), ("t", "l", "r"), None, S5),
    ]
    n_runs = sum(len(e[-1]) for e in EXPS) + 1
    reused = []
    if args.reuse:
        import shutil
        src = Path(args.reuse) / "runs"
        prev = json.loads((Path(args.reuse) / "run_info.json").read_text()) if (Path(args.reuse) / "run_info.json").exists() else {}
        for name, E, M, cfg, streams, rows, seeds in EXPS:
            if tuple(streams) != ("t",):
                continue
            for s in seeds:
                f_old, f_new = src / f"{name}_s{s}.npz", out / "runs" / f"{name}_s{s}.npz"
                if f_old.exists() and not f_new.exists():
                    z = np.load(f_old)
                    assert len(z["oof"]) == len(df), f"{f_old} has {len(z['oof'])} rows, table {len(df)}"
                    shutil.copy2(f_old, f_new); reused.append(f_new.stem)
        print(f"reused {len(reused)} text-only runs from {src} (table checksum there: "
              f"{prev.get('table_checksum', '?')})", flush=True)
    done_runs, t_runs = 0, time.time()
    for name, E, M, cfg, streams, rows, seeds in EXPS:
        for s in seeds:
            f = out / "runs" / f"{name}_s{s}.npz"
            done_runs += 1
            if f.exists():
                continue
            t0 = time.time()
            oof, gate, attn, info = T.run_cv(E, M, cfg, seed=s, rows=rows, streams=streams)
            np.savez_compressed(f, oof=oof, gate=gate, attn=attn, info=json.dumps(info))
            from sklearn.metrics import average_precision_score
            prs = [average_precision_score(y[(df.fold == k).to_numpy() & ~np.isnan(oof)],
                                           oof[(df.fold == k).to_numpy() & ~np.isnan(oof)]) for k in range(5)]
            el = time.time() - t_runs
            print(f"[{done_runs}/{n_runs}] {name:12s} seed {s}: PR {np.mean(prs):.3f}  ({time.time()-t0:.0f}s; "
                  f"~{el/done_runs*(n_runs-done_runs)/60:.0f} min left)", flush=True)

    # leak test: shuffled labels, all five folds
    f = out / "runs" / "leak_s0.npz"
    if not f.exists():
        y_sh = y[np.random.default_rng(0).permutation(len(y))]
        oof, gate, attn, info = T.run_cv(E_mdna, M_mdna, dict(C, epochs=15, patience=5), seed=0, labels=y_sh)
        np.savez_compressed(f, oof=oof, gate=gate, attn=attn, info=json.dumps(info), y_shuffled=y_sh)
        from sklearn.metrics import roc_auc_score
        print(f"leak test: shuffled-label ROC-AUC {roc_auc_score(y_sh, oof):.3f} (should be near 0.5)")

    # ------------------------------------------------------------------ collect
    cols = {"doc_id": df.doc_id, "label": df.label, "fold": df.fold, "strict": df.strict}
    gates, info_all = [], {}
    for f in sorted((out / "runs").glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        cols[f.stem] = z["oof"]
        info_all[f.stem] = json.loads(str(z["info"]))
        if f.stem.startswith("main_"):
            g = pd.DataFrame(z["gate"], columns=["g_text", "g_lang", "g_ratio"])
            g.insert(0, "doc_id", df.doc_id.to_numpy()); g.insert(1, "seed", int(f.stem.split("_s")[1]))
            gates.append(g)
            np.save(out / f"attn_{f.stem}.npy", z["attn"])
    pd.DataFrame(cols).to_csv(out / "dl_oof.csv", index=False)
    pd.concat(gates).to_csv(out / "dl_gates_main.csv", index=False)
    pd.DataFrame({"doc_id": df.doc_id, "sent_len_mdna": L_mdna, "sent_len_auditor": L_aud}).to_csv(
        out / "sentence_counts.csv", index=False)
    import sklearn, transformers
    (out / "run_info.json").write_text(json.dumps(dict(
        table_checksum=chk, rows=len(df), n_lang=len(LANG), n_ratios=len(RATIOS), cfg=dl.CFG,
        torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
        pandas=pd.__version__, sklearn=sklearn.__version__, python=platform.python_version(), gpu=gpu,
        minutes=round((time.time() - t_start) / 60, 1), reused_from=args.reuse, reused=reused,
        runs=info_all), indent=1))
    print(f"\nALL DONE in {(time.time()-t_start)/60:.0f} min. Send back the folder: {out}")


if __name__ == "__main__":
    main()
