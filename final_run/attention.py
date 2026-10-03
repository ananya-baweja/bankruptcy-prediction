#!/usr/bin/env python3
"""Which MD&A sentences the network attends to, and whether they are boilerplate.

Reads results/attn_main_s*.npy (attention over the first 96 sentences, per report, from each
seed's held-out fold) and the report text. Writes results_final/attention.json.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "vendor")); sys.path.append(str(HERE.parent / "src"))
from bppfinal.load import table  # noqa: E402

DLDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results_gpu"
_SPLIT, _WS = re.compile(r"(?<=[.!?])\s+"), re.compile(r"\s+")


def to_sentences(t, m=200):            # identical to bppfinal.dl.to_sentences
    out = []
    for s in _SPLIT.split(t or ""):
        s = _WS.sub(" ", s).strip()
        if len(s.split()) >= 5:
            out.append(s)
            if len(out) >= m:
                break
    return out


def norm(s):
    return re.sub(r"[^a-z ]", "", re.sub(r"\d+", "", s.lower())).strip()[:120]


df, _, _ = table()
files = sorted(DLDIR.glob("attn_main_s*.npy"))
A = np.nanmean([np.load(f).astype(np.float32) for f in files], 0)        # (n, 96)
sents = df.mdna.map(lambda t: to_sentences(t)[:96])
y = df.label.to_numpy()

# boilerplate: share of a sentence's word 5-grams that appear in the MD&A of >= 10 other companies
firm_of = df.firm_id.to_numpy()
toks = lambda x: re.findall(r"[a-z]+", x.lower())
def grams(x):
    t = toks(x); return {tuple(t[i:i + 5]) for i in range(max(len(t) - 4, 0))}
firm_grams = {}
for i, ss in enumerate(sents):
    g = firm_grams.setdefault(firm_of[i], set())
    for x in ss: g |= grams(x)
gdf = Counter(g for gs in firm_grams.values() for g in gs)
def boiler_score(x, firm):
    g = grams(x)
    if not g: return 0.0
    return float(np.mean([(gdf[t] - (t in firm_grams[firm])) >= 10 for t in g]))
seen = {}
rows, top_share_boiler, all_share_boiler, entropy, top_pos = [], [], [], [], []
top_mean_bs, all_mean_bs = [], []
for i, ss in enumerate(sents):
    k = len(ss)
    if k < 5 or np.isnan(A[i, :k]).any():
        continue
    w = A[i, :k] / A[i, :k].sum()
    entropy.append(float(-(w * np.log(w + 1e-12)).sum() / np.log(k)))     # 1 = uniform
    order = np.argsort(w)[::-1][:3]
    top_pos.extend((order / k).tolist())
    bs = [boiler_score(ss[j], firm_of[i]) for j in range(k)]
    boiler = [b >= 0.5 for b in bs]                                       # mostly shared phrasing
    top_share_boiler.append(np.mean([boiler[j] for j in order]))
    all_share_boiler.append(np.mean(boiler))
    top_mean_bs.append(np.mean([bs[j] for j in order])); all_mean_bs.append(np.mean(bs))
    for r, j in enumerate(order):
        rows.append(dict(doc_id=df.doc_id.iloc[i], label=int(y[i]), rank=r + 1, weight=float(w[j]),
                         position=int(j), n_sents=k, boiler=float(bs[j]), sentence=ss[j][:220]))

# the sentences that most often receive the top attention weight, across all reports
top1 = Counter(norm(r["sentence"])[:70] for r in rows if r["rank"] == 1)
example = {}
for r in rows:
    if r["rank"] == 1:
        example.setdefault(norm(r["sentence"])[:70], r["sentence"])
out = dict(
    seeds=len(files), reports=len(entropy),
    mean_normalised_entropy=float(np.mean(entropy)),
    share_top3_boilerplate=float(np.mean(top_share_boiler)),
    share_all_boilerplate=float(np.mean(all_share_boiler)),
    mean_relative_position_top3=float(np.mean(top_pos)),
    mean_shared_phrasing_top3=float(np.mean(top_mean_bs)), mean_shared_phrasing_all=float(np.mean(all_mean_bs)),
    most_common_top1=[dict(sentence=example[k], reports=v) for k, v in top1.most_common(8)],
)
# a few top sentences from distressed reports that are NOT boilerplate (used by < 10 companies)
specific = [r for r in rows if r["label"] == 1 and r["rank"] <= 2 and r["boiler"] < 0.2]
specific.sort(key=lambda r: -r["weight"])
out["specific_distressed_examples"] = specific[:12]
(HERE / "results_final").mkdir(exist_ok=True)
json.dump(out, open(HERE / "results_final" / "attention.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("specific_distressed_examples",)}, indent=1)[:3000])
