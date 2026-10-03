#!/usr/bin/env python3
"""Score a company's annual report for insolvency risk.

    python score.py path/to/FY2022.pdf
    python score.py FY2022.pdf --prev FY2021.pdf          # adds the year-on-year change features
    python score.py FY2022.pdf --financials figures.csv   # supply figures the reader missed (Rs crore)
    python score.py FY2022.pdf --no-dl                    # gradient boosting only (no PyTorch needed)

Prints a summary and writes <report>_risk.html and <report>_risk.json next to --out (default: the
current folder). A previous-year report named FY<year-1>.pdf in the same folder is picked up
automatically.

The score is a ranking against the 763 company-years the models were tested on, not a
probability: "risk score 93" means the report scored above 93% of the healthy company-years.
Bands: High = above 90% of healthy company-years, Watch = above 70%, Low = the rest. How many
insolvent company-years each band caught in cross-validation is printed with the result.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scorer"))
sys.path.insert(0, str(HERE))

import reader  # noqa: E402
from models import Scorer  # noqa: E402

BAND_TEXT = {"High": "High risk", "Watch": "Watch list", "Low": "Low risk"}
BAND_COLOUR = {"High": "#c0392b", "Watch": "#d4880f", "Low": "#2e7d4f"}


def load_override(path):
    if not path:
        return {}
    df = pd.read_csv(path)
    if {"field", "value_cr"} <= set(df.columns):              # long form: field,value_cr
        return {r.field: float(r.value_cr) for r in df.itertuples() if pd.notna(r.value_cr)}
    return {k: float(v) for k, v in df.iloc[0].items() if pd.notna(v) and k in reader.STANDARD_FIELDS}


def find_prev(pdf: Path, fy):
    m = re.search(r"FY\s*-?(\d{4})", pdf.stem, re.I)
    year = int(m.group(1)) if m else fy
    if not year:
        return None
    for cand in pdf.parent.glob("*.pdf"):
        mm = re.search(r"FY\s*-?(\d{4})", cand.stem, re.I)
        if mm and int(mm.group(1)) == year - 1:
            return cand
    return None


def fmt(v, kind="ratio"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if kind == "pct":
        return f"{100 * v:.0f}%"
    return f"{v:,.2f}"


def text_summary(r):
    out = []
    a = out.append
    a("")
    a(f"  {r['company'] or Path(r['pdf']).name}   FY{r['fy']}")
    a(f"  RISK SCORE {r['risk']:.0f}/100   {BAND_TEXT[r['band']].upper()}")
    p = r["parts"]
    a(f"  scored above {r['risk']:.0f}% of healthy company-years. In testing, {p['share_insolvent_at_least']:.0%} of "
      f"insolvent and {p['share_healthy_at_least']:.0%} of healthy company-years scored this high or higher")
    m = f"  models: {r['models']}  (boosting percentile {p['hgb_pct']:.0%}"
    if "dl_pct" in p:
        m += f", network percentile {p['dl_pct']:.0%}"
    a(m + ")")
    if r["dl_note"]:
        a(f"  note: {r['dl_note']}")
    a("")
    a("  What drives the score (score change if this part looked like a typical healthy company):")
    for d in r["drivers"]:
        if abs(d["effect"]) >= 1:
            a(f"    {d['effect']:+5.0f}  {d['family']}")
    if r["evidence"]:
        a("")
        a("  What the auditor wrote:")
        for e in r["evidence"]:
            a(f"    - {e['title']} ({e['where']})")
            if e["quote"]:
                a(f"      \"{e['quote'][:260]}\"")
    a("")
    a("  Ratios (worse than x% of healthy company-years):")
    for q in r["ratios"]:
        w = "" if q["worse_than"] is None else f"worse than {q['worse_than']:.0%}"
        a(f"    {q['name']:28s} {fmt(q['value']):>10s}   healthy median {fmt(q['healthy_median']):>8s}   {w}")
    if r["attended"]:
        a("")
        a("  MD&A sentences the network weighted most (its attention is nearly even, so these are only mildly favoured):")
        for s in r["attended"]:
            a(f"    [{s['index']}, x{s.get('relative', 0):.1f} an average sentence] {s['sentence'][:200]}")
    a("")
    a("  Reading the report:")
    for lvl, msg in r["coverage"]:
        a(f"    [{lvl}] {msg}")
    a("")
    return "\n".join(out)


def html_card(r):
    e = html.escape
    p = r["parts"]
    col = BAND_COLOUR[r["band"]]
    bs = r["band_stats"]
    drivers = "".join(
        f"<div class='drv'><span class='lab'>{e(d['family'])}</span><span class='bar'>"
        f"<i style='width:{min(abs(d['effect']), 60) / 60 * 100:.0f}%;background:{'#c0392b' if d['effect'] > 0 else '#2e7d4f'}'></i>"
        f"</span><span class='num'>{d['effect']:+.0f}</span></div>" for d in r["drivers"])
    ev = "".join(f"<li><b>{e(x['title'])}</b> <span class='muted'>({e(x['where'])})</span>"
                 + (f"<blockquote>{e(x['quote'])}</blockquote>" if x["quote"] else "") + "</li>" for x in r["evidence"])
    ev = ev or "<li class='muted'>None: no going-concern doubt, modified opinion, emphasis of matter, loan default or unpaid statutory dues reported.</li>"
    rows = ""
    for q in r["ratios"]:
        w = q["worse_than"]
        tag = "" if w is None else ("bad" if w >= 0.9 else "mid" if w >= 0.7 else "")
        rows += (f"<tr class='{tag}'><td>{e(q['name'])}</td><td>{fmt(q['value'])}</td><td>{fmt(q['healthy_median'])}</td>"
                 f"<td>{fmt(q['insolvent_median'])}</td><td>{'-' if w is None else f'{100 * w:.0f}%'}</td></tr>")
    att = "".join(f"<li><span class='muted'>sentence {s['index']}, {s.get('relative', 0):.1f}&times; an average sentence</span> {e(s['sentence'])}</li>" for s in r["attended"])
    cov = "".join(f"<li class='{lvl}'>{e(msg)}</li>" for lvl, msg in r["coverage"])
    fin = r["financials"]["fields"]
    figs = " &middot; ".join(f"{e(k.replace('_', ' '))} {fmt(v)}" for k, v in fin.items() if v is not None)
    model_line = f"Gradient boosting percentile {100 * p['hgb_pct']:.0f}"
    if "dl_pct" in p:
        g = p.get("gate") or [0, 0, 0]
        model_line += (f" &middot; gated network percentile {100 * p['dl_pct']:.0f} (gate: text {g[0]:.2f}, "
                       f"language {g[1]:.2f}, ratios {g[2]:.2f}) &middot; averaged 50/50")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Insolvency risk: {e(r['company'] or Path(r['pdf']).stem)}</title>
<style>
body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1d1d1f;background:#f6f5f2;margin:0;padding:24px}}
.card{{max-width:980px;margin:0 auto;background:#fff;border-radius:14px;padding:28px 32px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
h1{{font-size:22px;margin:0}} h2{{font-size:16px;margin:28px 0 10px;border-bottom:1px solid #eee;padding-bottom:6px}}
.muted{{color:#777}} .top{{display:flex;gap:28px;align-items:center;flex-wrap:wrap;margin-top:14px}}
.score{{font-size:64px;font-weight:700;color:{col};line-height:1}} .score small{{font-size:22px;color:#999}}
.band{{display:inline-block;background:{col};color:#fff;border-radius:999px;padding:4px 14px;font-weight:600}}
.drv{{display:flex;align-items:center;gap:10px;margin:6px 0}} .lab{{width:280px}} .bar{{flex:1;background:#f0efeb;height:12px;border-radius:6px;overflow:hidden}}
.bar i{{display:block;height:100%}} .num{{width:44px;text-align:right;font-variant-numeric:tabular-nums}}
blockquote{{margin:6px 0 12px;padding:8px 12px;background:#faf7f2;border-left:3px solid #d4880f;font-size:14px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} td,th{{padding:6px 8px;border-bottom:1px solid #f0f0f0;text-align:right}}
td:first-child,th:first-child{{text-align:left}} tr.bad td{{background:#fbeaea}} tr.mid td{{background:#fdf3e1}}
li.warn{{color:#a4590b}} li.ok{{color:#2e7d4f}} li.info{{color:#555}} ul{{padding-left:20px}}
.foot{{font-size:13px;color:#777;margin-top:26px}}
</style></head><body><div class="card">
<h1>{e(r['company'] or Path(r['pdf']).stem)} &middot; FY{r['fy']}</h1>
<div class="muted">{e(Path(r['pdf']).name)} &middot; {r['n_pages']} pages &middot; scored {e(r['scored_at'])}</div>
<div class="top"><div class="score">{r['risk']:.0f}<small>/100</small></div>
<div><span class="band">{BAND_TEXT[r['band']]}</span>
<p style="margin:8px 0 0">Scored above <b>{r['risk']:.0f}%</b> of healthy company-years. In testing, <b>{100 * p['share_insolvent_at_least']:.0f}%</b> of
company-years that went insolvent within three years scored this high or higher, and {100 * p['share_healthy_at_least']:.0f}% of healthy ones.</p>
<p class="muted" style="margin:4px 0 0">{model_line}</p></div></div>
<h2>What drives the score</h2>
<p class="muted" style="margin-top:0">Change in the score if this part of the report looked like a typical healthy company's (red raises risk).</p>
{drivers}
<h2>What the auditor reported</h2><ul>{ev}</ul>
<h2>Financial ratios</h2>
<table><tr><th>Ratio</th><th>This report</th><th>Healthy median</th><th>Insolvent median</th><th>Worse than (healthy)</th></tr>{rows}</table>
<p class="muted">Figures read (Rs crore): {figs or 'none'}</p>
{'<h2>MD&amp;A sentences the network weighted most</h2><p class="muted" style="margin-top:0">The network spreads its attention almost evenly over the sentences, so these are only mildly favoured; the reasons above carry more weight.</p><ul>' + att + '</ul>' if att else ''}
<h2>Reading the report</h2><ul>{cov}</ul>
<p class="foot">Bands: High = above {100 * r['band_quantiles']['high']:.0f}% of healthy company-years (caught {100 * bs['high']['insolvent_caught']:.0f}% of insolvent company-years in
cross-validation); Watch = above {100 * r['band_quantiles']['watch']:.0f}% (caught {100 * bs['watch']['insolvent_caught']:.0f}%). The models were trained on 135 Indian listed companies admitted
to insolvency (IBC) and 135 matched healthy peers, FY2016-2025. The score ranks a report against that sample; it is not a probability, and the sample is
1:1 where insolvency is far rarer. A course project, not investment or credit advice.</p>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", nargs="+")
    ap.add_argument("--prev", help="the previous year's report (for the year-on-year change features)")
    ap.add_argument("--fy", type=int, help="the report's fiscal year (read from the report if omitted)")
    ap.add_argument("--financials", help="CSV of figures in Rs crore that replace what the reader found "
                                         "(columns named like total_assets, or rows field,value_cr)")
    ap.add_argument("--models", default=str(HERE / "final_models"))
    ap.add_argument("--no-dl", action="store_true", help="gradient boosting only")
    ap.add_argument("--out", default=".")
    ap.add_argument("--dl-check", action="store_true",
                    help="run the network path on whatever networks exist (a code check; score not meaningful)")
    args = ap.parse_args()

    t0 = time.time()
    sc = Scorer(Path(args.models), use_dl=not args.no_dl, check=args.dl_check)
    if args.dl_check and sc.dl is None:
        sys.exit(f"DL CHECK FAILED: {sc.dl_note}")
    cfg = reader.config()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for pdf in map(Path, args.pdf):
        t1 = time.time()
        doc = reader.read_pdf(pdf, cfg, args.fy)
        prev_path = Path(args.prev) if args.prev else find_prev(pdf, doc["fy"])
        prev = reader.read_pdf(prev_path, cfg, (doc["fy"] - 1) if doc["fy"] else None) if prev_path else None
        lang = reader.language(doc, sc.lms, prev)
        fin = reader.financials(doc, cfg, load_override(args.financials))
        mdna = (doc["payload"]["sections"].get("mdna") or {}).get("text") or ""
        res = sc.score(fin["ratios"], lang, fin["financials_missing"], mdna)
        res.update(pdf=str(pdf), company=doc["company"], fy=doc["fy"], n_pages=doc["n_pages"],
                   evidence=reader.evidence(doc), coverage=reader.coverage(doc, fin, prev is not None),
                   financials=dict(fields=fin["fields"], source=fin["source"], altman_zone=fin["altman_zone"],
                                   altman_z_dprime=fin["altman_z_dprime"], validation_flags=fin["validation_flags"]),
                   previous_report=str(prev_path) if prev_path else None,
                   features={k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v)
                             for k, v in {**lang, **fin["ratios"], "financials_missing": fin["financials_missing"]}.items()
                             if k in sc.cols},
                   band_quantiles=sc.ref["band_quantiles"], scored_at=time.strftime("%d %b %Y %H:%M"),
                   seconds=round(time.time() - t1, 1))
        print(text_summary(res))
        stem = pdf.parent.name + "_" + pdf.stem if pdf.stem.upper().startswith("FY") else pdf.stem
        (out_dir / f"{stem}_risk.html").write_text(html_card(res), encoding="utf-8")
        (out_dir / f"{stem}_risk.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print(f"  -> {out_dir / (stem + '_risk.html')}   ({res['seconds']:.0f}s)")
        summary.append((stem, res["company"], res["fy"], res["risk"], res["band"]))
    if len(summary) > 1:
        print("\n  " + "\n  ".join(f"{s:28s} {str(c)[:34]:34s} FY{fy}  {r:5.0f}  {b}" for s, c, fy, r, b in summary))
    if args.dl_check:
        print("\nDL CHECK OK: FinBERT and the saved networks ran on a new report.")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
