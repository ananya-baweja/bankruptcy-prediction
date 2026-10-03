"""One place that builds (or reloads) the modelling table for the CPU analyses.

The table is built from data/ (the processed/ files) with the folds frozen in folds.csv, exactly as
run_gpu.py builds it, and cached in results_cpu/table.pkl."""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent


def table(data=None):
    cache, colsf = HERE / "results_cpu" / "table.pkl", HERE / "results_cpu" / "columns.json"
    if cache.exists() and colsf.exists():
        c = json.loads(colsf.read_text())
        return pd.read_pickle(cache), c["ratios"], c["language"]
    sys.path.insert(0, str(HERE / "vendor")); sys.path.append(str(HERE.parent / "src"))
    from bppfinal.table import build
    df, R, L = build(Path(data or HERE / "data"), folds_file=HERE / "folds.csv")
    cache.parent.mkdir(exist_ok=True)
    df.to_pickle(cache); colsf.write_text(json.dumps({"ratios": R, "language": L}, indent=1))
    return df, R, L
