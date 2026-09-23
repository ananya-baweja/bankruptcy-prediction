# Setup — local (VS Code + Anaconda) and Google Colab

## Where each phase runs

| Phase | Where | Why |
| --- | --- | --- |
| 0–3 (scraping, PDFs, OCR, sections, ratios) | **Laptop, VS Code** | NSE/BSE often block cloud IPs; the work is CPU and file heavy |
| 4 (FinBERT, spaCy transformer, fastcoref) | **Colab GPU** | FinBERT on ~800 reports takes hours on CPU, minutes on a T4 |
| 5 (baselines) | either | small |
| 6–7 (fusion model, experiments) | **Colab GPU** | training + Optuna |

Code is shared through **GitHub**; data through a shared **Google Drive** folder (`data/` is never committed).

---

## A. Local setup on Windows (once per team member)

### 1. Python environment

Open **Anaconda Prompt**, go to the project folder, create the environment:

```powershell
cd "V:\Projects\Bankruptcy Prediction Project"
conda env create -f environment.yml
conda activate bpp
```

This installs all Phase 0–2 libraries and the project itself in editable mode (`pip install -e .`),
so code changes take effect immediately and the `bpp` command is available.

Updating later (after someone adds a dependency): `pip install -r requirements.txt`.

### 2. Tesseract OCR (for scanned PDFs)

1. Download the Windows installer from the UB Mannheim Tesseract page (github.com/UB-Mannheim/tesseract/wiki).
2. Install to the default `C:\Program Files\Tesseract-OCR\`.
3. In `configs/config.yaml` set:
   ```yaml
   extraction:
     ocr:
       tesseract_cmd: "C:/Program Files/Tesseract-OCR/tesseract.exe"
   ```

Without Tesseract everything still runs; scanned pages are marked `needs_ocr`.

### 3. Check it works

```powershell
pytest            # ~35 tests, under a minute
bpp demo          # full Phase 1-2 run on fake companies -> data_demo/
bpp --data-dir data_demo status
```

### 4. VS Code

1. **File → Open Folder** → the project folder.
2. Install the **Python** and **Jupyter** extensions.
3. `Ctrl+Shift+P` → **Python: Select Interpreter** → pick `bpp` (conda).
4. Use the built-in terminal (`` Ctrl+` ``); run `conda activate bpp` if the prompt does not show `(bpp)`.

### 5. Git and GitHub (one person does this once)

1. VS Code **Source Control** panel → **Initialize Repository**.
2. Stage all → commit message "Phase 0-2 pipeline".
3. **Publish Branch** → private GitHub repository → add teammates as collaborators.
4. Teammates: **Clone Repository** in VS Code, then steps 1–4 above.

Working rules:
- Pull before you start, push when a step works.
- One branch per person/feature (`phase3-ratios`, `nlp-ner`), merge via pull request.
- Never commit `data/` (the `.gitignore` already blocks it) or API keys.

### 6. Shared data on Google Drive

1. Create a shared Drive folder `BPP-data` with the same structure as `data/`.
2. Install **Google Drive for desktop** and either work directly in the synced folder:
   ```powershell
   bpp --data-dir "G:\My Drive\BPP-data" status
   ```
   or set it once for the session: `$env:BPP_DATA_DIR = "G:\My Drive\BPP-data"`.
3. Decide who "owns" each file to avoid two people editing the same CSV (see the task tracker in the shared doc).

---

## B. Google Colab

Use `notebooks/00_setup_and_smoke_test.ipynb`, or these cells:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<your-account>/<your-repo>.git /content/bpp
%cd /content/bpp
!pip install -q -r requirements.txt && pip install -q -e .
!apt-get -qq install -y tesseract-ocr        # only if you OCR on Colab

import os
os.environ["BPP_DATA_DIR"] = "/content/drive/MyDrive/BPP-data"
!bpp status
```

- **Private repo:** create a GitHub fine-grained token (read-only) and clone with
  `https://<token>@github.com/...`; store the token in Colab **Secrets**, never in the notebook.
- **GPU:** Runtime → Change runtime type → T4 GPU (needed from Phase 4).
- Colab sessions reset: anything not in Drive or GitHub is lost. Save outputs to `BPP_DATA_DIR`.

---

## C. Common problems

| Problem | Fix |
| --- | --- |
| `bpp` is not recognized | `conda activate bpp`; or run `python -m bpp.cli ...` |
| `ModuleNotFoundError: nse` | `pip install nse` |
| NSE/BSE requests fail with 401/403 | run on the laptop (not Colab); wait and retry; collect manually |
| `TesseractNotFoundError` | set `tesseract_cmd` in the config (step A2) |
| Excel shows dates as `30-08-2019` and saves them that way | fine: dd-mm-yyyy and yyyy-mm-dd are both understood |
| Excel breaks BSE codes or ISINs | import CSV via **Data → From Text/CSV** and set those columns to Text |
