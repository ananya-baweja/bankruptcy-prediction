"""Measure section-extraction accuracy by hand (the number you report in the paper).

Workflow
--------
1. ``bpp qa-sample`` picks ``sample_size`` documents at random (stratified so
   distressed and healthy firms are both included when labels exist) and writes
   data/interim/qa/section_qa_sheet.csv with one row per document x section.
2. A team member opens each PDF at the predicted pages and fills in:
      found_correct   Y / N   - is there really such a section, and did we find it?
      start_correct   Y / N   - does our text start at the right heading?
      end_correct     Y / N   - does it end at the right place (within ~1 paragraph)?
      true_start_page, true_end_page, notes   (optional, helps fix rules)
   For a section that truly does not exist in the report, and we also found
   nothing, write found_correct = Y and leave start/end blank.
3. ``bpp qa-score`` reads the filled sheet and prints accuracy per section.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from bpp.common import write_csv
from bpp.config import Paths
from bpp.extract.sections import TARGET_SECTIONS

log = logging.getLogger(__name__)

QA_FILL_COLS = ["found_correct", "start_correct", "end_correct", "true_start_page", "true_end_page", "notes"]


def make_qa_sheet(cfg: dict[str, Any], paths: Paths, n: int | None = None, overwrite: bool = False) -> pd.DataFrame:
    n = n or cfg["qa"]["sample_size"]
    if paths.qa_sheet.exists() and not overwrite:
        raise FileExistsError(f"{paths.qa_sheet} exists (it may contain hand labels). Use --overwrite to replace.")
    files = sorted(paths.sections.glob("*.json"))
    if not files:
        raise FileNotFoundError("No section files yet. Run bpp extract-sections first.")
    docs = pd.DataFrame({"doc_id": [f.stem for f in files]})
    if paths.documents_labeled.exists():
        lab = pd.read_csv(paths.documents_labeled)[["doc_id", "label"]].drop_duplicates("doc_id")
        docs = docs.merge(lab, on="doc_id", how="left")
    else:
        docs["label"] = -1
    seed = cfg["qa"]["random_seed"]
    n = min(n, len(docs))
    docs["label"] = docs["label"].fillna(-1)
    groups = [g for _, g in docs.groupby("label")]
    per = max(1, n // max(1, len(groups)))
    sample = pd.concat([g.sample(min(len(g), per), random_state=seed) for g in groups])
    if len(sample) < n:
        rest = docs[~docs["doc_id"].isin(sample["doc_id"])]
        sample = pd.concat([sample, rest.sample(min(len(rest), n - len(sample)), random_state=seed)])
    rows = []
    for did in sorted(sample["doc_id"]):
        js = json.loads((paths.sections / f"{did}.json").read_text(encoding="utf-8"))
        for s in TARGET_SECTIONS:
            sec = js["sections"].get(s)
            rows.append({
                "doc_id": did, "source_pdf": js.get("source_pdf"), "section": s,
                "predicted_found": bool(sec),
                "predicted_start_page": sec["start_page"] if sec else "",
                "predicted_end_page": sec["end_page"] if sec else "",
                "predicted_heading": sec["heading"] if sec else "",
                "predicted_first_200_chars": (sec["text"][:200].replace("\n", " ") if sec else ""),
                "predicted_last_200_chars": (sec["text"][-200:].replace("\n", " ") if sec else ""),
                **{c: "" for c in QA_FILL_COLS},
            })
    df = pd.DataFrame(rows)
    write_csv(df, paths.qa_sheet)
    log.info("QA sheet for %d documents -> fill in %s", len(sample), QA_FILL_COLS[:3])
    return df


def score_qa_sheet(paths: Paths) -> pd.DataFrame:
    df = pd.read_csv(paths.qa_sheet, dtype=str).fillna("")
    for c in ["found_correct", "start_correct", "end_correct"]:
        df[c] = df[c].str.strip().str.upper()
    filled = df[df["found_correct"].isin(["Y", "N"])]
    if filled.empty:
        raise ValueError("No rows filled in yet (found_correct must be Y or N).")

    def rate(s: pd.Series) -> float | None:
        s = s[s.isin(["Y", "N"])]
        return round((s == "Y").mean(), 3) if len(s) else None

    out = []
    for sec, g in filled.groupby("section"):
        found = g[g["predicted_found"].str.lower() == "true"]
        both = found[found["start_correct"].isin(["Y", "N"]) & found["end_correct"].isin(["Y", "N"])]
        out.append({
            "section": sec, "n_checked": len(g),
            "found_accuracy": rate(g["found_correct"]),
            "start_accuracy": rate(found["start_correct"]),
            "end_accuracy": rate(found["end_correct"]),
            "exact_span_accuracy": round(((both["start_correct"] == "Y") & (both["end_correct"] == "Y")).mean(), 3)
            if len(both) else None,
        })
    res = pd.DataFrame(out)
    write_csv(res, paths.qa / "section_qa_scores.csv")
    return res
