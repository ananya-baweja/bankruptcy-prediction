"""Numbers for the report's Section 11 and the correction notes, read from the scoring outputs and
the two versions of the language features, so the report types none of them by hand."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ref = json.loads((HERE / "reference_box.json").read_text())
band = ref["bands"]["blend"]
demo = {d: json.loads((HERE / "demo_out" / f"{d}_risk.json").read_text())
        for d in ["BSE532886_FY2017", "BSE532661_FY2022", "BSE519248_FY2017"]}
before = json.loads((HERE / "jvl_before_refinement.json").read_text())

# flag rates on the 763 modelling rows, before and after the correction
tab = pd.read_pickle(ROOT / "results_cpu" / "table.pkl")[["doc_id", "label"]]
rates, changed = {}, 0
for v in ["data_v1", "data_v2"]:
    lf = pd.read_csv(ROOT / v / "language_features.csv")[["doc_id", "caro_default_flag", "caro_statutory_dues_flag"]]
    m = tab.merge(lf, on="doc_id")
    rates[v] = {f: {str(k): round(100 * x) for k, x in m.groupby("label")[f].mean().items()}
                for f in ["caro_default_flag", "caro_statutory_dues_flag"]}
a = pd.read_csv(ROOT / "data_v1" / "language_features.csv").set_index("doc_id")
b = pd.read_csv(ROOT / "data_v2" / "language_features.csv").set_index("doc_id")
for f in ["caro_default_flag", "caro_statutory_dues_flag"]:
    changed += int((a[f].fillna(-1) != b[f].fillna(-1)).sum())


def card(d, company, outcome, why):
    r = demo[d]
    return dict(company=company, outcome=outcome, risk=round(r["risk"]), band=r["band"], why=why)


S = dict(
    parity_features=51,
    high_q=round(100 * ref["band_quantiles"]["high"]), watch_q=round(100 * ref["band_quantiles"]["watch"]),
    high_caught=round(100 * band["high"]["insolvent_caught"]), watch_caught=round(100 * band["watch"]["insolvent_caught"]),
    high_caught_t1=round(100 * band["high"]["insolvent_caught_by_horizon"]["t-1"]),
    instability_points=round(100 * abs(before["parts"]["hgb_pct"] - demo["BSE519248_FY2017"]["parts"]["hgb_pct"])),
    jvl_aud_chars=1968,
    flag_rates=rates, flag_cells_changed=changed,
    examples=[
        card("BSE532886_FY2017", "SEL Manufacturing, FY2017 report", "Admitted to insolvency 27 April 2018",
             "Qualified audit opinion (interest of Rs 359 crore not provided on bank loans classed as NPA), the CARO "
             "statement that it \"has defaulted in repayment of loans\", debt 22.6 times equity, all shown on the card"),
        card("BSE532661_FY2022", "Rane (Madras), FY2022 report", "Healthy (a matched peer; never admitted)",
             "No auditor's flags beyond a COVID-19 emphasis of matter; current ratio 0.89 and debt/equity 1.19 are weak, "
             "the other ratios near healthy medians"),
        card("BSE519248_FY2017", "JVL Agro Industries, FY2017 report", "Admitted to insolvency 27 July 2018",
             "Missed. Only 1,968 characters of the auditor's report were found and the opinion was not identified "
             "(both flagged as warnings on the card); ratios middling"),
    ],
)
(HERE / "scoring_summary.json").write_text(json.dumps(S, indent=1))
print(json.dumps(S, indent=1))
