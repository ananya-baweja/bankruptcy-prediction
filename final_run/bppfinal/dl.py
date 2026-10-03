"""Deep-learning half of the final run: FinBERT sentence embeddings and the gated fusion network.

The network, loss, preprocessing and training loop are notebook 2's (sections 4-5), with three
changes, each recorded in the final report:

1. The BiGRU now reads only the first `n_sents` sentence slots and skips padding
   (pack_padded_sequence). In notebook 2 the attention was masked to the first 96 sentences, but
   the GRU itself ran over all 200 slots, so its backward direction still saw sentences 97-200 and
   the zero padding.
2. `streams` selects which of the three input blocks the network may use (text, language features,
   ratios). Unused blocks are zero and receive no gate weight. This gives the text-only and
   no-text networks used for H1, H4 and the ablation.
3. `rows` restricts training and testing to a subset (used for the strict 609-row run).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

MODEL_NAME, MAX_SENTS, MAX_TOKENS, DIM = "ProsusAI/finbert", 200, 96, 768
CFG = dict(gru=32, block=64, attn_hidden=32, drop=0.50, lr=1e-3, wd=0.0,
           batch=16, gamma=1.0, n_sents=96, epochs=40, patience=8)

# ----------------------------------------------------------------------------- embeddings
_SPLIT, _WS = re.compile(r"(?<=[.!?])\s+"), re.compile(r"\s+")


def to_sentences(t, m=MAX_SENTS):
    """Split into sentences, keep those with 5+ words (notebook 1, section 7)."""
    out = []
    for s in _SPLIT.split(t or ""):
        s = _WS.sub(" ", s).strip()
        if len(s.split()) >= 5:
            out.append(s)
            if len(out) >= m:
                break
    return out


def embed(df, section, cache: Path, dev):
    """FinBERT mean-pooled sentence vectors, (n, 200, 768) float16, cached by doc_id."""
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"{section}_{MODEL_NAME.split('/')[-1]}_{MAX_SENTS}x{MAX_TOKENS}"
    npy, idx = cache / f"embstore_{tag}.npy", cache / f"embstore_{tag}.json"
    have, store = {}, np.zeros((0, MAX_SENTS, DIM), np.float16)
    if npy.exists() and idx.exists():
        store = np.load(npy)
        have = {d: i for i, d in enumerate(json.loads(idx.read_text()))}
    sents = df[section].map(to_sentences)
    need = [d for d in df.doc_id if d not in have]
    print(f"[{section}] {len(have)} cached, {len(need)} to embed", flush=True)
    if need:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        enc = AutoModel.from_pretrained(MODEL_NAME).to(dev).eval()
        if dev == "cuda":
            enc = enc.half()
        sent_of = dict(zip(df.doc_id, sents))
        new = np.zeros((len(need), MAX_SENTS, DIM), np.float16)
        t0 = time.time()
        with torch.no_grad():
            for j, d in enumerate(need):
                ss = sent_of[d]
                for a in range(0, len(ss), 64):
                    chunk = ss[a:a + 64]
                    b = tok(chunk, padding=True, truncation=True, max_length=MAX_TOKENS,
                            return_tensors="pt").to(dev)
                    h = enc(**b).last_hidden_state.float()
                    mk = b["attention_mask"].unsqueeze(-1).float()
                    new[j, a:a + len(chunk)] = ((h * mk).sum(1) / mk.sum(1).clamp(min=1)).cpu().numpy()
                if j % 100 == 0:
                    print(f"   {j}/{len(need)}  {time.time()-t0:.0f}s", flush=True)
        store = np.concatenate([store, new], 0)
        order = list(have.keys()) + need
        np.save(npy, store)
        idx.write_text(json.dumps(order))
        have = {d: i for i, d in enumerate(order)}
        del enc
        if dev == "cuda":
            torch.cuda.empty_cache()
    E = store[[have[d] for d in df.doc_id]]
    lengths = sents.map(len).to_numpy()
    return E, lengths


# ----------------------------------------------------------------------------- network
class AdditiveAttention(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.proj, self.v = nn.Linear(d, hidden), nn.Linear(hidden, 1, bias=False)

    def forward(self, h, mask):
        score = self.v(torch.tanh(self.proj(h))).squeeze(-1)
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        w = torch.softmax(score, 1).unsqueeze(-1)
        return (h * w).sum(1), w.squeeze(-1)


def mlp(i, h, o, drop, bn=False):
    layers = [nn.Linear(i, h)]
    if bn:
        layers.append(nn.BatchNorm1d(h))
    return nn.Sequential(*(layers + [nn.ReLU(), nn.Dropout(drop), nn.Linear(h, o), nn.ReLU()]))


class FusionNet(nn.Module):
    def __init__(self, n_lang, n_ratio, gru=32, block=64, attn_hidden=32, drop=0.5, dim=DIM,
                 streams=("t", "l", "r")):
        super().__init__()
        self.streams = tuple(streams)
        self.gru = nn.GRU(dim, gru, batch_first=True, bidirectional=True)
        self.attn = AdditiveAttention(2 * gru, attn_hidden)
        self.text_proj = nn.Sequential(nn.Linear(2 * gru, block), nn.ReLU())
        self.lang = mlp(max(n_lang, 1), block, block, drop)
        self.ratio = mlp(max(n_ratio, 1), block, block, drop, bn=True)
        self.gate = nn.Linear(3 * block, 3)
        self.head = nn.Sequential(nn.Linear(3 * block, block), nn.ReLU(), nn.Dropout(drop), nn.Linear(block, 1))
        # gate logits of unused streams are pushed to -inf, so they get exactly zero weight
        self.register_buffer("gate_off", torch.tensor([s not in self.streams for s in "tlr"]))

    def forward(self, emb, mask, lang, ratio):
        B, ns = mask.shape
        lengths = mask.sum(1)
        if "t" in self.streams:
            packed = pack_padded_sequence(emb, lengths.clamp(min=1).cpu(), batch_first=True,
                                          enforce_sorted=False)
            out, _ = self.gru(packed)
            h, _ = pad_packed_sequence(out, batch_first=True, total_length=ns)
            pooled, attn_w = self.attn(h, mask)
            t = self.text_proj(pooled) * (lengths > 0).unsqueeze(1).float()
        else:
            attn_w = torch.zeros(B, ns, device=mask.device)
            t = torch.zeros(B, self.text_proj[0].out_features, device=mask.device)
        l = self.lang(lang) if "l" in self.streams else torch.zeros_like(t)
        r = self.ratio(ratio) if "r" in self.streams else torch.zeros_like(t)
        logits = self.gate(torch.cat([t, l, r], -1)).masked_fill(self.gate_off, float("-inf"))
        g = torch.softmax(logits, -1).unsqueeze(-1)
        fused = (torch.stack([t, l, r], 1) * g).flatten(1)
        return self.head(fused).squeeze(-1), g.squeeze(-1), attn_w


def focal_bce(logit, target, gamma=1.0, ce_weight=0.1):
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    if gamma == 0:
        return ce.mean()
    return ((1 - torch.exp(-ce)) ** gamma * ce).mean() + ce_weight * ce.mean()


def prep(train_idx, idx, X):
    """Fill gaps with the TRAINING median and scale by TRAINING mean/sd (notebook 2, section 5)."""
    med = np.nanmedian(X[train_idx], 0); med = np.where(np.isnan(med), 0, med)
    mu = np.nanmean(X[train_idx], 0); sd = np.nanstd(X[train_idx], 0)
    mu = np.where(np.isnan(mu), 0, mu)
    sd = np.where(~np.isfinite(sd) | (sd < 1e-6), 1.0, sd)
    return ((np.where(np.isnan(X[idx]), med, X[idx]) - mu) / sd).astype(np.float32)


class Trainer:
    """Holds the arrays shared by every run, so one call trains one fold."""

    def __init__(self, y, groups, fold, Xl, Xr, dev, n_splits=5):
        self.y = np.asarray(y, np.float32)
        self.groups, self.FOLD = np.asarray(groups), np.asarray(fold)
        self.Xl = np.asarray(Xl, np.float32)
        self.Xr = np.asarray(Xr, np.float32)
        self.dev, self.K = dev, n_splits

    def train_fold(self, k, E, MASK, cfg=CFG, seed=0, labels=None, rows=None, streams=("t", "l", "r")):
        """Train on every fold but k, predict fold k. Returns (test_idx, prob, gate, attention)."""
        yy = self.y if labels is None else np.asarray(labels, np.float32)
        use = np.ones(len(yy), bool) if rows is None else np.asarray(rows, bool)
        te = np.flatnonzero((self.FOLD == k) & use)
        dv = np.flatnonzero((self.FOLD != k) & use)
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed)
                    .split(dv, yy[dv], self.groups[dv]))
        tr, va = dv[a], dv[b]
        torch.manual_seed(seed); np.random.seed(seed)
        ns = cfg["n_sents"]

        def dataset(idx):
            return TensorDataset(torch.from_numpy(np.ascontiguousarray(E[idx, :ns]).astype(np.float32)),
                                 torch.from_numpy(np.ascontiguousarray(MASK[idx, :ns])),
                                 torch.from_numpy(prep(tr, idx, self.Xl)),
                                 torch.from_numpy(prep(tr, idx, self.Xr)),
                                 torch.from_numpy(yy[idx]))

        g = torch.Generator(); g.manual_seed(seed)
        Ltr = DataLoader(dataset(tr), batch_size=cfg["batch"], shuffle=True, drop_last=True, generator=g)
        Lva = DataLoader(dataset(va), batch_size=128)
        Lte = DataLoader(dataset(te), batch_size=128)
        model = FusionNet(self.Xl.shape[1], self.Xr.shape[1], cfg["gru"], cfg["block"], cfg["attn_hidden"],
                          cfg["drop"], streams=streams).to(self.dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

        def infer(loader, full=False):
            model.eval(); P, G, A = [], [], []
            with torch.no_grad():
                for e, m, l, r, _ in loader:
                    lo, gg, aw = model(e.to(self.dev), m.to(self.dev), l.to(self.dev), r.to(self.dev))
                    P.append(torch.sigmoid(lo).cpu().numpy())
                    if full:
                        G.append(gg.cpu().numpy()); A.append(aw.cpu().numpy())
            if full:
                return np.concatenate(P), np.concatenate(G), np.concatenate(A)
            return np.concatenate(P)

        best, bad, best_state, epochs_run = -1.0, 0, None, 0
        for ep in range(cfg["epochs"]):
            model.train()
            for e, m, l, r, t in Ltr:
                opt.zero_grad()
                lo, _, _ = model(e.to(self.dev), m.to(self.dev), l.to(self.dev), r.to(self.dev))
                focal_bce(lo, t.to(self.dev), cfg["gamma"]).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            epochs_run = ep + 1
            v = average_precision_score(yy[va], infer(Lva))
            if v > best:
                best, bad = v, 0
                best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
        model.load_state_dict(best_state)
        p, gate, attn = infer(Lte, full=True)
        return te, p, gate, attn, dict(best_val_pr=best, epochs=epochs_run)

    def run_cv(self, E, MASK, cfg=CFG, seed=0, labels=None, rows=None, streams=("t", "l", "r"), folds=None):
        n = len(self.y)
        ns = cfg["n_sents"]
        m = np.asarray(MASK)[:, :ns]
        assert (m == (np.arange(ns)[None, :] < m.sum(1)[:, None])).all(), \
            "sentence masks must mark a prefix of real sentences (the GRU reads the first `length` slots)"
        oof = np.full(n, np.nan, np.float32)
        gate = np.full((n, 3), np.nan, np.float32)
        attn = np.full((n, cfg["n_sents"]), np.nan, np.float16)
        info = []
        for k in (range(self.K) if folds is None else folds):
            te, p, g, a, inf = self.train_fold(k, E, MASK, cfg, seed, labels, rows, streams)
            oof[te], gate[te], attn[te] = p, g, a
            info.append(dict(fold=int(k), **inf))
        return oof, gate, attn, info
