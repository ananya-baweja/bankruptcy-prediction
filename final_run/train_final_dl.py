#!/usr/bin/env python3
"""Train the deep model on ALL 763 company-years, for scoring new companies (score.py).

The cross-validated runs (run_gpu.py) measure how well the method works; they never train on the
row they score, so none of their 35 fitted networks has seen all the data. This script fits the
final networks the same way - same architecture, loss, settings and early-stopping rule, 7 seeds -
but on every row, and saves them with everything needed to prepare a new company's inputs.

    python train_final_dl.py --data data_v2            # ~5 min on the RTX 5060 Ti
    python train_final_dl.py --data data_v2 --smoke    # 1 seed, 2 epochs: checks the code path

Writes final_models/dl/: fusion_s{seed}.pt (weights), prep_s{seed}.npz (the training medians,
means and standard deviations that turn raw features into network inputs), meta.json.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))
sys.path.append(str(HERE.parent / "src"))          # inside the repository: the bpp package itself
from bppfinal.table import build  # noqa: E402


def prep_stats(X, tr):
    """The statistics dl.prep() computes from the training rows, kept so a new row is treated alike."""
    med = np.nanmedian(X[tr], 0); med = np.where(np.isnan(med), 0, med)
    mu = np.nanmean(X[tr], 0); sd = np.nanstd(X[tr], 0)
    mu = np.where(np.isnan(mu), 0, mu)
    sd = np.where(~np.isfinite(sd) | (sd < 1e-6), 1.0, sd)
    return med.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data_v2"))
    ap.add_argument("--cache", default=str(HERE / "cache"))
    ap.add_argument("--out", default=str(HERE / "final_models" / "dl"))
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupShuffleSplit
    from bppfinal import dl
    from run_gpu import table_checksum

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}  device {dev}", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    df, RATIOS, LANG = build(Path(args.data), with_perplexity=True, folds_file=HERE / "folds.csv")
    chk = table_checksum(df, RATIOS + LANG)
    print(f"table checksum {chk}", flush=True)
    y = df.label.to_numpy().astype(np.float32)
    groups = df.pair_id.to_numpy()
    Xl = df[LANG].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    Xr = df[RATIOS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    E, L = dl.embed(df, "mdna", Path(args.cache), dev)
    M = np.arange(dl.MAX_SENTS)[None, :] < L[:, None]

    cfg = dict(dl.CFG)
    seeds = [0] if args.smoke else list(range(args.seeds))
    if args.smoke:
        cfg["epochs"] = 2
    ns = cfg["n_sents"]
    all_rows = np.arange(len(df))

    for seed in seeds:
        t0 = time.time()
        # the same early-stopping rule as every cross-validated run: 20% of the pairs held out
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(all_rows, y, groups))
        tr, va = all_rows[a], all_rows[b]
        torch.manual_seed(seed); np.random.seed(seed)
        stats = {"lang": prep_stats(Xl, tr), "ratio": prep_stats(Xr, tr)}

        def dataset(idx):
            return TensorDataset(torch.from_numpy(np.ascontiguousarray(E[idx, :ns]).astype(np.float32)),
                                 torch.from_numpy(np.ascontiguousarray(M[idx, :ns])),
                                 torch.from_numpy(dl.prep(tr, idx, Xl)),
                                 torch.from_numpy(dl.prep(tr, idx, Xr)),
                                 torch.from_numpy(y[idx]))

        g = torch.Generator(); g.manual_seed(seed)
        Ltr = DataLoader(dataset(tr), batch_size=cfg["batch"], shuffle=True, drop_last=True, generator=g)
        Lva = DataLoader(dataset(va), batch_size=128)
        model = dl.FusionNet(Xl.shape[1], Xr.shape[1], cfg["gru"], cfg["block"], cfg["attn_hidden"],
                             cfg["drop"]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

        def infer(loader):
            model.eval(); P = []
            with torch.no_grad():
                for e, m, l, r, _ in loader:
                    lo, _, _ = model(e.to(dev), m.to(dev), l.to(dev), r.to(dev))
                    P.append(torch.sigmoid(lo).cpu().numpy())
            return np.concatenate(P)

        best, bad, best_state, epochs_run = -1.0, 0, None, 0
        for ep in range(cfg["epochs"]):
            model.train()
            for e, m, l, r, t in Ltr:
                opt.zero_grad()
                lo, _, _ = model(e.to(dev), m.to(dev), l.to(dev), r.to(dev))
                dl.focal_bce(lo, t.to(dev), cfg["gamma"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            epochs_run = ep + 1
            v = average_precision_score(y[va], infer(Lva))
            if v > best:
                best, bad = v, 0
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
        torch.save(best_state, out / f"fusion_s{seed}.pt")
        np.savez(out / f"prep_s{seed}.npz",
                 lang_med=stats["lang"][0], lang_mu=stats["lang"][1], lang_sd=stats["lang"][2],
                 ratio_med=stats["ratio"][0], ratio_mu=stats["ratio"][1], ratio_sd=stats["ratio"][2])
        print(f"seed {seed}: held-out PR-AUC {best:.3f} after {epochs_run} epochs ({time.time()-t0:.0f}s)",
              flush=True)

        # check: the saved files, reloaded, reproduce the network's outputs
        model.load_state_dict(best_state)
        chk_model = dl.FusionNet(Xl.shape[1], Xr.shape[1], cfg["gru"], cfg["block"], cfg["attn_hidden"],
                                 cfg["drop"]).to(dev)
        chk_model.load_state_dict(torch.load(out / f"fusion_s{seed}.pt", map_location=dev))
        chk_model.eval(); model.eval()
        z = np.load(out / f"prep_s{seed}.npz")
        idx = va[:16]
        lang_in = ((np.where(np.isnan(Xl[idx]), z["lang_med"], Xl[idx]) - z["lang_mu"]) / z["lang_sd"]).astype(np.float32)
        assert np.allclose(lang_in, dl.prep(tr, idx, Xl), atol=1e-5), "saved preprocessing differs"
        with torch.no_grad():
            args_t = (torch.from_numpy(np.ascontiguousarray(E[idx, :ns]).astype(np.float32)).to(dev),
                      torch.from_numpy(np.ascontiguousarray(M[idx, :ns])).to(dev),
                      torch.from_numpy(lang_in).to(dev),
                      torch.from_numpy(dl.prep(tr, idx, Xr)).to(dev))
            a1 = model(*args_t)[0].cpu().numpy(); a2 = chk_model(*args_t)[0].cpu().numpy()
        assert np.allclose(a1, a2, atol=1e-4), "reloaded network gives different outputs"

    (out / "meta.json").write_text(json.dumps(dict(
        table_checksum=chk, rows=len(df), lang=LANG, ratios=RATIOS, cfg=cfg, seeds=seeds,
        model_name=dl.MODEL_NAME, max_sents=dl.MAX_SENTS, max_tokens=dl.MAX_TOKENS,
        torch=torch.__version__, smoke=args.smoke,
        trained=time.strftime("%Y-%m-%d %H:%M"), minutes=round((time.time() - t_start) / 60, 1)), indent=1))
    print(f"\nDONE in {(time.time()-t_start)/60:.1f} min: {len(seeds)} networks in {out}")


if __name__ == "__main__":
    main()
