"""Load the final models and turn one company-year's inputs into a risk score with reasons."""
from __future__ import annotations

import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent

# Feature families for the "what drives the score" breakdown (see Scorer.score).
FAMILIES = {
    "Financial ratios": lambda c: c in RATIO_SET or c == "financials_missing",
    "Auditor's opinion and CARO flags": lambda c: c in {
        "has_going_concern", "has_emphasis_of_matter", "has_modified_opinion_basis", "audit_opinion_severity",
        "audit_opinion_found", "caro_default_flag", "caro_statutory_dues_flag", "has_auditor_report",
        "has_caro_annexure"},
    "Wording of the auditor's report": lambda c: c.startswith("auditor_"),
    "Wording of the MD&A": lambda c: c.startswith("mdna_") or c in {"has_mdna", "perplexity_fold"},
    "Change from last year's MD&A": lambda c: c.startswith("drift_"),
}
RATIO_SET: set[str] = set()

# lower is worse for every ratio except the two leverage ratios
WORSE_IF_HIGH = {"debt_to_equity", "debt_to_ebitda"}
RATIO_NAMES = {
    "current_ratio": "Current ratio", "quick_ratio": "Quick ratio", "cash_to_assets": "Cash / total assets",
    "ebitda_margin": "EBITDA margin", "roce": "Return on capital employed", "roa": "Return on assets",
    "debt_to_equity": "Debt / equity", "debt_to_ebitda": "Debt / EBITDA", "interest_coverage": "Interest coverage",
    "retained_earnings_to_assets": "Retained earnings / assets", "altman_z_em": "Altman Z'' (EM score)",
}


def pct_against(ref, x):
    ref = np.sort(np.asarray(ref, float)); x = np.atleast_1d(np.asarray(x, float))
    lo = np.searchsorted(ref, x, "left"); hi = np.searchsorted(ref, x, "right")
    return (lo + 0.5 * (hi - lo)) / len(ref)


class Scorer:
    def __init__(self, model_dir: Path, use_dl: bool = True, cache: Path | None = None, check: bool = False):
        self.check = check          # --dl-check: run the network path even on smoke-test networks
        self.dir = Path(model_dir)
        self.ref = json.loads((self.dir / "reference.json").read_text())
        z = np.load(self.dir / "reference.npz", allow_pickle=False)
        self.label, self.oof_hgb, self.blend_ref = z["label"], z["oof_hgb"], z["blend"]
        self.oof_dl_seed = z["oof_dl_seed"]
        self.Xref = z["X"]
        self.cols = self.ref["columns"]
        RATIO_SET.clear(); RATIO_SET.update(self.ref["ratios"])
        with open(self.dir / "hgb.pkl", "rb") as f:
            self.hgb = pickle.load(f)
        self.lms = []
        for k in range(5):
            with gzip.open(self.dir / f"lm_fold{k}.pkl.gz", "rb") as f:
                self.lms.append(pickle.load(f))
        self.healthy_median = np.array([np.nan if self.ref["quantiles"][c]["healthy"] is None
                                        else self.ref["quantiles"][c]["healthy"][3] for c in self.cols])
        self.dl = None
        self.dl_note = ""
        if use_dl:
            self._load_dl(cache)

    # ------------------------------------------------------------------ the network
    def _load_dl(self, cache):
        d = self.dir / "dl"
        if not (d / "meta.json").exists():
            self.dl_note = "no trained networks in final_models/dl (run train_final_dl.py on the GPU box)"
            return
        if not self.ref.get("dl_seeds") and not self.check:
            self.dl_note = "networks found, but reference.json has no network scores (re-run train_final.py with --dl-oof)"
            return
        try:
            import torch
        except ImportError:
            self.dl_note = "PyTorch is not installed here: gradient boosting only"
            return
        sys.path.insert(0, str(HERE.parent))
        from bppfinal import dl
        meta = json.loads((d / "meta.json").read_text())
        if self.check:
            # a code-path check only: no matching reference scores yet, so a placeholder stands in
            self.dl_note = "DL CHECK: placeholder reference, the score below is not meaningful"
            if not self.ref.get("dl_seeds") or list(meta["seeds"]) != list(self.ref["dl_seeds"]):
                self.oof_dl_seed = np.random.default_rng(0).uniform(size=(len(self.label), len(meta["seeds"])))
                self.ref["dl_seeds"] = list(meta["seeds"])
            self.ref["bands"].setdefault("blend", self.ref["bands"]["hgb"])
        elif meta.get("smoke"):
            self.dl_note = "the saved networks are from a smoke test (2 epochs): gradient boosting only"
            return
        elif meta["lang"] != self.ref["language"] or meta["ratios"] != self.ref["ratios"]:
            self.dl_note = "the networks were trained on different columns than reference.json: not used"
            return
        elif meta.get("table_checksum") != self.ref.get("table_checksum"):
            self.dl_note = "the networks and the boosting model were trained on different tables: not used"
            return
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        nets, preps = [], []
        cfg = meta["cfg"]
        for s in meta["seeds"]:
            net = dl.FusionNet(len(meta["lang"]), len(meta["ratios"]), cfg["gru"], cfg["block"],
                               cfg["attn_hidden"], cfg["drop"]).to(dev)
            net.load_state_dict(torch.load(d / f"fusion_s{s}.pt", map_location=dev))
            net.eval(); nets.append(net); preps.append(dict(np.load(d / f"prep_s{s}.npz")))
        seeds_ref = self.ref["dl_seeds"]
        if list(meta["seeds"]) != list(seeds_ref):
            self.dl_note = f"network seeds {meta['seeds']} differ from the reference seeds {seeds_ref}: not used"
            return
        self.dl = dict(torch=torch, mod=dl, nets=nets, preps=preps, dev=dev, cfg=cfg, meta=meta,
                       cache=cache, enc=None, tok=None)

    def _embed(self, text):
        """FinBERT vectors of the first 200 sentences of 5+ words, as in training."""
        D = self.dl; torch = D["torch"]; dl = D["mod"]
        if D["enc"] is None:
            from transformers import AutoModel, AutoTokenizer
            D["tok"] = AutoTokenizer.from_pretrained(dl.MODEL_NAME)
            enc = AutoModel.from_pretrained(dl.MODEL_NAME).to(D["dev"]).eval()
            D["enc"] = enc.half() if D["dev"] == "cuda" else enc
        ss = dl.to_sentences(text)
        E = np.zeros((dl.MAX_SENTS, dl.DIM), np.float32)
        with torch.no_grad():
            for a in range(0, len(ss), 64):
                chunk = ss[a:a + 64]
                b = D["tok"](chunk, padding=True, truncation=True, max_length=dl.MAX_TOKENS,
                             return_tensors="pt").to(D["dev"])
                h = D["enc"](**b).last_hidden_state.float()
                mk = b["attention_mask"].unsqueeze(-1).float()
                E[a:a + len(chunk)] = ((h * mk).sum(1) / mk.sum(1).clamp(min=1)).cpu().numpy()
        # training stored the vectors as float16
        return E.astype(np.float16).astype(np.float32), ss

    def _dl_scores(self, X, emb, n_sents):
        """Per-seed probabilities (n_rows x n_seeds), mean gate and attention, for rows X that share
        one report's sentences."""
        D = self.dl; torch = D["torch"]
        n_r, n_l = len(self.ref["ratios"]), len(self.ref["language"])
        Xr, Xl = X[:, :n_r], X[:, n_r:n_r + n_l]        # columns are ratios, then language features
        ns = D["cfg"]["n_sents"]
        B = len(X)
        mask = np.repeat((np.arange(ns) < min(n_sents, ns))[None, :], B, 0)
        e = torch.from_numpy(np.ascontiguousarray(emb[None, :ns])).to(D["dev"]).expand(B, -1, -1).contiguous()
        m = torch.from_numpy(mask).to(D["dev"])
        P, G, A = [], [], []
        with torch.no_grad():
            for net, pp in zip(D["nets"], D["preps"]):
                li = ((np.where(np.isnan(Xl), pp["lang_med"], Xl) - pp["lang_mu"]) / pp["lang_sd"]).astype(np.float32)
                ri = ((np.where(np.isnan(Xr), pp["ratio_med"], Xr) - pp["ratio_mu"]) / pp["ratio_sd"]).astype(np.float32)
                lo, g, a = net(e, m, torch.from_numpy(li).to(D["dev"]), torch.from_numpy(ri).to(D["dev"]))
                P.append(torch.sigmoid(lo).cpu().numpy()); G.append(g[0].cpu().numpy()); A.append(a[0].cpu().numpy())
        return np.stack(P, 1), np.mean(G, 0), np.mean(A, 0)

    # ------------------------------------------------------------------ scoring
    def vector(self, ratios: dict, lang: dict, financials_missing: float) -> np.ndarray:
        vals = {**ratios, **lang, "financials_missing": financials_missing}
        return np.array([np.nan if vals.get(c) is None else float(vals[c]) for c in self.cols], float)

    def _risk(self, X, emb=None, n_sents=0):
        """Risk scores (0-100) for the rows of X, plus the parts of the first row."""
        X = np.atleast_2d(X)
        p_h = self.hgb.predict_proba(X)[:, 1]
        pc_h = pct_against(self.oof_hgb, p_h)
        parts = dict(hgb_prob=float(p_h[0]), hgb_pct=float(pc_h[0]))
        if self.dl is not None and emb is not None:
            P, G, A = self._dl_scores(X, emb, n_sents)
            pc_d = np.mean([pct_against(self.oof_dl_seed[:, j], P[:, j]) for j in range(P.shape[1])], 0)
            b = 0.5 * pc_h + 0.5 * pc_d
            parts.update(dl_prob=float(P[0].mean()), dl_pct=float(pc_d[0]), gate=G.tolist(), attention=A)
        else:
            b = pc_h
        healthy = self.label == 0
        score = pct_against(self.blend_ref[healthy], b) * 100
        parts.update(blend=float(b[0]), risk=float(score[0]),
                     share_insolvent_at_least=float((self.blend_ref[~healthy] >= b[0]).mean()),
                     share_healthy_at_least=float((self.blend_ref[healthy] >= b[0]).mean()))
        return score, parts

    def score(self, ratios: dict, lang: dict, financials_missing: float, mdna_text: str = "") -> dict[str, Any]:
        x = self.vector(ratios, lang, financials_missing)
        emb, sents = (None, [])
        if self.dl is not None and mdna_text:
            emb, sents = self._embed(mdna_text)
        elif self.dl is not None:
            emb = np.zeros((self.dl["mod"].MAX_SENTS, self.dl["mod"].DIM), np.float32)
        score, parts = self._risk(x, emb, len(sents))
        risk = float(score[0])
        bands = self.ref["bands"]["blend" if self.dl is not None else "hgb"]
        band = "High" if risk >= 100 * self.ref["band_quantiles"]["high"] else (
               "Watch" if risk >= 100 * self.ref["band_quantiles"]["watch"] else "Low")

        # What drives it: one family of features at a time is swapped for the values of real healthy
        # company-years (150 drawn from the training data), keeping the rest of this report; the
        # average fall in the score is that family's contribution. Real rows rather than a row of
        # medians, which no company has and which the trees can read as unusually safe.
        rng = np.random.default_rng(0)
        Xh = self.Xref[self.label == 0]
        Xh = Xh[rng.choice(len(Xh), min(150, len(Xh)), replace=False)]
        drivers = []
        for fam, member in FAMILIES.items():
            idx = [i for i, c in enumerate(self.cols) if member(c)]
            X2 = np.repeat(x[None], len(Xh), 0); X2[:, idx] = Xh[:, idx]
            s2, _ = self._risk(X2, emb, len(sents))
            drivers.append(dict(family=fam, effect=risk - float(s2.mean())))
        drivers.sort(key=lambda d: -abs(d["effect"]))

        # each ratio against the healthy company-years
        rq = []
        for c in self.ref["ratios"]:
            v = ratios.get(c)
            j = self.cols.index(c)
            ref_h = self.Xref[self.label == 0, j]; ref_h = ref_h[~np.isnan(ref_h)]
            if v is None:
                rq.append(dict(ratio=c, name=RATIO_NAMES[c], value=None, healthy_median=float(np.median(ref_h)),
                               insolvent_median=self.ref["quantiles"][c]["insolvent_median"], worse_than=None))
                continue
            below = float(pct_against(ref_h, v)[0])
            worse = below if c in WORSE_IF_HIGH else 1 - below      # share of healthy firms it is worse than
            rq.append(dict(ratio=c, name=RATIO_NAMES[c], value=float(v), healthy_median=float(np.median(ref_h)),
                           insolvent_median=self.ref["quantiles"][c]["insolvent_median"], worse_than=worse))

        top_sents = []
        if "attention" in parts and sents:
            a = parts.pop("attention")[:len(sents)]
            for i in np.argsort(-a)[:3]:
                if i < len(sents):
                    # relative to an even spread over the sentences read: 1.0 = an average sentence
                    top_sents.append(dict(index=int(i) + 1, weight=float(a[i]),
                                          relative=float(a[i] * min(len(sents), len(a))), sentence=sents[i]))
        else:
            parts.pop("attention", None)
        return dict(risk=risk, band=band, parts=parts, drivers=drivers, ratios=rq, attended=top_sents,
                    models="gradient boosting + gated network (50/50)" if self.dl is not None else "gradient boosting",
                    dl_note=self.dl_note, band_stats=bands, n_sentences=len(sents))
