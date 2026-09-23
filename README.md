# Bankruptcy Prediction Project

Multimodal prediction of corporate insolvency (NCLT admission under India's Insolvency and
Bankruptcy Code, 2016) for BSE/NSE-listed companies. The model combines annual-report language
(MD&A, Board's report, auditor's report, CARO annexure) with accounting ratios.
It is a 3rd-year project for the **Natural Language Processing** and **Deep Learning** courses.

> **Status:** Phases 0–2 (setup, cohort building, document collection and section extraction) are
> built and tested on synthetic data. Phases 3–8 are planned in [`docs/00_project_plan.md`](docs/00_project_plan.md).

## Quick start

```powershell
conda env create -f environment.yml
conda activate bpp
pytest                 # automated tests
bpp demo               # whole Phase 1-2 pipeline on fake companies -> data_demo/
bpp status             # what exists in data/ and what to do next
```

## The pipeline so far

| Phase | Step | Command | Output |
| --- | --- | --- | --- |
| 1 | IBBI public announcements | `bpp ibbi-scrape` | `data/raw/ibbi/public_announcements.csv` |
| 1 | CIRP, non-private debtors | `bpp ibbi-filter` | `data/interim/ibbi_cirp_filtered.csv` |
| 1 | Listed companies (active/suspended/delisted) | `bpp listed-fetch` | `data/interim/listed_universe.csv` |
| 1 | Match debtor names to listed firms | `bpp match-names` | `data/interim/ibbi_listed_matches.csv` → **review by hand** |
| 1 | Healthy peers + firm-year frame | `bpp build-cohort` | `data/processed/cohort.csv`, `sample_frame.csv` |
| 2 | Which annual reports exist | `bpp reports-list` | `data/interim/report_listings.csv` |
| 2 | Download / register PDFs | `bpp reports-download`, `bpp reports-register` | `data/interim/documents.csv` |
| 2 | Page text (+OCR) | `bpp extract-text` | `data/interim/pages/*.json` |
| 2 | Sections | `bpp extract-sections` | `data/interim/sections/*.json`, `extraction_report.csv` |
| 2 | Labels, horizons, leakage rules | `bpp assign-labels` | `data/processed/documents_labeled.csv` |
| 2 | Extraction accuracy | `bpp qa-sample`, `bpp qa-score` | `data/interim/qa/` |

Every command accepts `--data-dir <folder>` (e.g. a Google Drive folder), and every setting is in
[`configs/config.yaml`](configs/config.yaml).

## Repository layout

```
configs/config.yaml     all settings
src/bpp/
  scrape/               IBBI, BSE/NSE lists, annual reports
  cohort/               name matching, peers, sample frame, labels
  extract/              PDF text + OCR, sections, QA
  features/ nlp/ models/ eval/   Phases 3-8 (to be built)
  cli.py                the bpp command
  synthetic.py          fake data for the demo and tests
data/                   raw/ interim/ processed/ (not in git), manual/ (templates in git)
docs/                   plan, setup, phase guides, data dictionary, decisions and progress logs
notebooks/              setup + smoke test (local or Colab)
tests/                  pytest suite
```

## Documentation

1. [`docs/00_project_plan.md`](docs/00_project_plan.md) — problem, design decisions, methodology, contributions, timeline
2. [`docs/01_setup.md`](docs/01_setup.md) — Anaconda, VS Code, Tesseract, Git/GitHub, Google Drive, Colab
3. [`docs/02_phase1_cohort.md`](docs/02_phase1_cohort.md) — every Phase 1 step: what, why, command, manual work
4. [`docs/03_phase2_documents.md`](docs/03_phase2_documents.md) — every Phase 2 step
5. [`docs/data_dictionary.md`](docs/data_dictionary.md) — every file and column
6. [`docs/decisions_log.md`](docs/decisions_log.md) and [`docs/progress_log.md`](docs/progress_log.md)

## Data and ethics

Only public sources (IBBI, BSE, NSE, company annual reports) and licensed databases the college
provides are used. Scrapers wait between requests; check each site's terms before large downloads.
Raw data and Prowess exports are never committed to Git. `bpp demo` uses invented companies only.
