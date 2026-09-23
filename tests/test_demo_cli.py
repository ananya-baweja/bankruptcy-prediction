import pandas as pd

from bpp.cli import main


def test_demo_runs_end_to_end(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out)]) == 0
    lab = pd.read_csv(out / "processed" / "documents_labeled.csv")
    inc = lab[lab["included"]]
    assert (inc["label"] == 1).sum() == (inc["label"] == 0).sum() > 0      # balanced, aligned pairs
    assert set(inc["horizon"]) <= {"t-1", "t-2", "t-3"}
    rep = pd.read_csv(out / "interim" / "extraction_report.csv")
    assert (rep["mdna_chars"] > 3000).all() and (rep["caro_annexure_chars"] > 0).all()
    assert main(["--data-dir", str(out), "status"]) == 0
    assert "Labelled documents" in capsys.readouterr().out
