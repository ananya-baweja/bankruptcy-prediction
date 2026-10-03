# Final deep-learning run on the GPU box

Copy this whole `bpp_final` folder onto the GPU box (over Remote Desktop: copy it in File
Explorer on the laptop, paste it on the box's Desktop). It contains the code, the frozen folds and
the 5 data files it needs (`data/`).

What it does: rebuilds the 763-row table with the folds frozen in `folds.csv`, embeds the MD&A and
auditor's report with FinBERT (downloads ProsusAI/finbert, ~440 MB, the first time), and trains
45 network runs. Each run is saved as soon as it finishes, so the script can be stopped and
re-started; it skips what is done. ~1.5-2 h on a T4-class GPU, less on a bigger one.
`--quick` uses 3 seeds per experiment instead of 5/7 (about half the time).

## Windows (PowerShell on the GPU box)

    cd $HOME\Desktop\bpp_final
    nvidia-smi
    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

If that prints `True` (and torch is 2.7 or newer, needed for RTX 50-series cards), use that Python and run
`pip install transformers pandas scipy scikit-learn`. Otherwise make an environment:

    python -m venv .venv
    Set-ExecutionPolicy -Scope Process Bypass
    .\.venv\Scripts\Activate.ps1
    pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50-series needs cu128
    pip install transformers pandas scipy scikit-learn

Then:

    python run_gpu.py --smoke                        # ~3 min, must end "SMOKE TEST PASSED"
    python run_gpu.py 2>&1 | Tee-Object run.log      # the full run; leave the window open

Closing the Remote Desktop window (disconnect) keeps it running; do not sign out.
When it prints ALL DONE, copy the `results` folder and `run.log` back to the laptop.

## Linux

    cd ~/bpp_final && python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python run_gpu.py --smoke
    nohup python run_gpu.py > run.log 2>&1 &
