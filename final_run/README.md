# final_run: the final models, the evaluation and the scoring tool

Everything behind the final report (`docs/Final_Report_Bankruptcy_Prediction.docx`). It works on the
dataset folder `processed/` (Google Drive), not on `data/` in this repository, and nothing in it
writes to the dataset.

| File | What it does |
| --- | --- |
| `run_gpu.py` | The deep-learning experiments: FinBERT embeddings, 45 network runs on the frozen folds, out-of-fold predictions saved per run (resumable) |
| `run_cpu.py`, `extras_cpu.py` | Every model that needs no GPU (Altman, logistic regression, gradient boosting, TF-IDF, mined lexicon), permutation importance, tone replication |
| `attention.py`, `analyze.py` | Attention analysis; every metric, paired test, pair bootstrap and DeLong test, written to `results_final/tables.json` |
| `report/` | The final report, generated from `tables.json` (`npm install docx` once) |
| `train_final.py`, `train_final_dl.py` | The final models on all 763 company-years, for scoring new companies |
| `score.py`, `scorer/` | Scores one annual report PDF: risk score, band, reasons |
| `bppfinal/` | The modelling table (`table.py`), the network (`dl.py`), the CPU models and the statistics |

## Inputs (not in git)

- `data/`: the dataset's `processed/` folder (`documents_labeled.csv`, `language_features.csv`,
  `ratios.csv`, `report_sections_1/2.jsonl.gz`). `language_features.csv` must be the version with the
  corrected CARO flags (regenerate it with `bpp language-features`, or take
  `final_run/bpp_final/data_v2/language_features.csv` from the Drive).
- `folds.csv`: the frozen fold of every company-year (Drive, `final_run/bpp_final/folds.csv`).
  scikit-learn's GroupKFold assigns ties differently across versions; recomputing the folds would
  move 79% of the rows.
- For `score.py`: `scorer/loughran_mcdonald.csv` (or `data/manual/` of this repository) and the
  trained models in `final_models/`.

## Running it

```bash
pip install -r requirements.txt pypdfium2 pyyaml requests     # torch with CUDA for the GPU steps
python run_gpu.py --smoke                                      # ~3 min check on the GPU
python run_gpu.py --data data --out results                    # ~3 h on an RTX 5060 Ti
python run_cpu.py && python extras_cpu.py                      # minutes on a CPU
python attention.py results && python analyze.py --dl results
cd report && python make_narrative.py && python charts.py && python diagrams.py && node build_report.js
```

After a change to the language features or ratios, `run_gpu.py --out results_v2 --reuse results`
re-runs only the 30 runs that read them and copies the 15 text-only runs.

## Scoring a new company

```bash
python train_final_dl.py && python train_final.py              # once, on the computer that scores
python score.py path/to/FY2023.pdf                              # FY2022.pdf beside it is used for drift
python score.py report.pdf --financials figures.csv             # figures the reader missed, in Rs crore
python score.py report.pdf --no-dl                              # gradient boosting only
```

The risk score is the share of healthy company-years (in cross-validation) that scored lower. High
means riskier than 90% of them, Watch riskier than 70%. It is a ranking against the study's 1:1
sample, not a probability.
