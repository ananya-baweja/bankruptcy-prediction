"""process_reports.py export-evidence: processed/ carries the files behind its README's checks."""

import gzip
import hashlib
import importlib.util
from pathlib import Path

import pandas as pd

from bpp.config import Paths

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("process_reports", ROOT / "scripts" / "process_reports.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _data(tmp_path: Path, xbrl_firms: list[str]) -> Paths:
    D = tmp_path / "data"
    paths = Paths(D)
    paths.processed.mkdir(parents=True)
    pd.DataFrame({"pair_id": ["P0001", "P0001"], "firm_id": ["BSE1", "BSE2"],
                  "role": ["distressed", "healthy"]}).to_csv(paths.cohort, index=False)
    (D / "interim/qa/full").mkdir(parents=True)
    (D / "interim/qa/full/summary.md").write_text("# Checks\n- within 1%: 91.8%\n", encoding="utf-8")
    pd.DataFrame({"field": ["total_assets"], "n": [10], "within_1pct": [0.9],
                  "within_5pct": [1.0]}).to_csv(D / "interim/qa/full/pdf_vs_xbrl_by_field.csv", index=False)
    paths.manual.mkdir(parents=True)
    pd.DataFrame({"doc_id": ["BSE1_FY2019"], "decision": ["exclude"],
                  "reason": ["own_petition: p6, a creditor's petition\nlisted before NCLT"],
                  "reviewed_by": ["Gaurav (approved)"]}).to_csv(paths.leakage_review, index=False)
    pd.DataFrame({"firm_id": xbrl_firms, "fy": [2019] * len(xbrl_firms), "field": ["total_assets"] * len(xbrl_firms),
                  "value_cr": [10.5 + k for k in range(len(xbrl_firms))],
                  "note": ["filed, in lakh"] * len(xbrl_firms)}).to_csv(paths.xbrl_exchange, index=False)
    return paths


def test_export_evidence_copies_the_checks_and_lists_them(tmp_path):
    pr = _script()
    paths = _data(tmp_path, ["BSE1", "BSE2", "BSE3"])
    ev = paths.processed / "evidence"
    stale = ev / "cohort/amend_pairs.csv"                  # left by an earlier run
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")

    man = pr.step_export_evidence(paths).set_index("file")

    assert (ev / "qa/summary.md").read_bytes() == (paths.qa / "full/summary.md").read_bytes()
    assert (ev / "documents/leakage_review.csv").read_bytes() == paths.leakage_review.read_bytes()
    assert sorted(pd.read_csv(ev / "xbrl/xbrl_financials_exchange.csv")["firm_id"]) == ["BSE1", "BSE2"]
    with gzip.open(ev / "xbrl/xbrl_financials_all_listed.csv.gz", "rb") as fh:
        assert fh.read() == paths.xbrl_exchange.read_bytes()
    assert not stale.exists()                              # this run made no amend_pairs.csv

    listed = pd.read_csv(ev / "MANIFEST.csv").set_index("file")
    assert listed.loc["qa/summary.md", "sha256"] == hashlib.sha256((ev / "qa/summary.md").read_bytes()).hexdigest()
    assert listed.loc["documents/leakage_review.csv", "rows"] == 1          # a quoted line break is one row
    assert listed.loc["xbrl/xbrl_financials_exchange.csv", "rows"] == 2
    assert listed.loc["xbrl/xbrl_financials_all_listed.csv.gz", "rows"] == 3
    assert listed.loc["cohort/amend_pairs.csv", "note"] == "source not found"
    assert pd.isna(listed.loc["cohort/amend_pairs.csv", "sha256"])
    assert set(man.index) == set(listed.index)


def test_export_evidence_is_reproducible_and_skips_a_cohort_only_table(tmp_path):
    pr = _script()
    paths = _data(tmp_path, ["BSE1", "BSE2", "BSE3"])
    first = pr.step_export_evidence(paths).set_index("file")
    again = pr.step_export_evidence(paths).set_index("file")
    assert (first["sha256"] == again["sha256"]).all()      # the gzip carries no timestamp

    only_cohort = _data(tmp_path / "second", ["BSE1", "BSE2"])
    man = pr.step_export_evidence(only_cohort).set_index("file")
    assert not (only_cohort.processed / "evidence/xbrl/xbrl_financials_all_listed.csv.gz").exists()
    assert man.loc["xbrl/xbrl_financials_all_listed.csv.gz", "note"] == "the XBRL table holds only the cohort's firms"
