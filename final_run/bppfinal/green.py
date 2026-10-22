"""Green-AI study: energy measurement, encoder variants and the selection rule.

Used by run_green.py. The pure functions at the top (integration, cosine similarity, seed
ensembles, the Pareto front, the selection rule) import nothing heavy, so the tests can run them
without torch or a GPU.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

KWH_PER_J = 1 / 3.6e6
# Fallback grid factor for India (kg CO2e per kWh) if codecarbon's own table cannot be read:
# CEA CO2 Baseline Database v19 (2023-24), weighted average of the grid.
IND_KG_PER_KWH_FALLBACK = 0.727


# ----------------------------------------------------------------------------- pure helpers
def integrate_kwh(t, watts):
    """Trapezoid integral of power samples (seconds, watts) -> kWh. One sample -> 0."""
    t, w = np.asarray(t, float), np.asarray(watts, float)
    if len(t) < 2:
        return 0.0
    return float(np.sum((w[1:] + w[:-1]) / 2 * np.diff(t)) * KWH_PER_J)


def mean_cosine(A, B, lengths):
    """Mean cosine similarity between matching sentence vectors of A and B (n, s, d), over the
    first lengths[i] (real) sentences of each document only."""
    A, B = np.asarray(A, np.float32), np.asarray(B, np.float32)
    s = A.shape[1]
    m = np.arange(s)[None, :] < np.minimum(np.asarray(lengths), s)[:, None]
    a, b = A[m], B[m]
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 0
    return float(np.mean(num[ok] / den[ok])), float(np.min(num[ok] / den[ok]))


def prefix_mask(lengths, n_slots, cap=None):
    """Boolean (n, n_slots) mask of the real sentences, optionally keeping only the first `cap`."""
    L = np.asarray(lengths)
    if cap is not None:
        L = np.minimum(L, cap)
    return np.arange(n_slots)[None, :] < L[:, None]


def seed_ensemble(oofs, fold, k=None):
    """Mean of within-fold rank-normalised predictions of the first k seeds (analyze.py: ens)."""
    from .stats import rank_norm
    use = list(oofs)[:k] if k else list(oofs)
    return np.nanmean([rank_norm(np.asarray(p, float), fold) for p in use], 0)


def blend(dl, gb, fold):
    """The current system: 0.5 * rank(DL ensemble) + 0.5 * rank(GB ratios + language), per fold."""
    from .stats import rank_norm
    return 0.5 * rank_norm(dl, fold) + 0.5 * rank_norm(gb, fold)


def pareto_front(cost, score):
    """Mask of points not dominated by another (lower-or-equal cost AND higher-or-equal score,
    strictly better in at least one)."""
    c, s = np.asarray(cost, float), np.asarray(score, float)
    keep = np.ones(len(c), bool)
    for i in range(len(c)):
        dom = (c <= c[i]) & (s >= s[i]) & ((c < c[i]) | (s > s[i]))
        keep[i] = not dom.any()
    return keep


def choose_config(rows, ref_pr, tol=0.010, energy_tie=0.05):
    """The pre-registered rule (DECISIONS.md D2.3). rows: dicts with name, kwh, pr_all, boot_lo,
    boot_hi (CI of PR-AUC minus the current blend). Returns (name, reason) of the lowest-kWh row
    that is within `tol` of ref_pr or whose CI includes 0; (None, reason) if none qualifies.
    Energies within `energy_tie` (relative) of the lowest are a tie, broken by the higher pr_all
    (D2.3a: the GPU metric separates configs with no GPU work only by seconds of idle draw)."""
    ok = []
    for r in rows:
        within = r["pr_all"] >= ref_pr - tol - 1e-12
        ci0 = (r.get("boot_lo") is not None and not math.isnan(r["boot_lo"])
               and r["boot_lo"] <= 0 <= r["boot_hi"])
        if within or ci0:
            ok.append((r["kwh"], -r["pr_all"], r["name"], "within %.3f of %.3f" % (tol, ref_pr) if within
                       else "bootstrap CI vs current blend includes 0"))
    if not ok:
        return None, "no config qualifies"
    ok.sort()
    low = ok[0][0]
    tied = sorted([o for o in ok if o[0] <= low * (1 + energy_tie) + 1e-15], key=lambda o: (o[1], o[0]))
    best = tied[0]
    why = best[3] + ("" if len(tied) == 1 else
                     "; energy tie within %d%% with %s, broken by PR-AUC" % (round(100 * energy_tie),
                     ", ".join(o[2] for o in tied if o[2] != best[2])))
    return best[2], why


def gpu_co2e(rec):
    """kg CO2e of one stage's GPU energy alone (DECISIONS.md D4.3a): the record's own grid factor
    (kg_co2e / energy_kwh, codecarbon's India value) applied to gpu_kwh (NVML). CPU and RAM energy,
    which background load inflates, are left out."""
    e = float(rec.get("energy_kwh") or 0.0)
    factor = float(rec["kg_co2e"]) / e if e > 0 else 0.0
    return float(rec.get("gpu_kwh") or 0.0) * factor


def reduction(base, new):
    """(percent saved, multiple) of `new` against `base`."""
    if new <= 0:
        return 100.0, float("inf")
    return 100 * (1 - new / base), base / new


# ----------------------------------------------------------------------------- energy meter
def _codecarbon_ind_factor():
    try:
        import codecarbon
        p = Path(codecarbon.__file__).parent / "data" / "private_infra" / "global_energy_mix.json"
        d = json.loads(p.read_text())["IND"]
        return float(d["carbon_intensity"]) / 1000, "codecarbon global_energy_mix.json (IND)"
    except Exception:
        return IND_KG_PER_KWH_FALLBACK, "CEA CO2 baseline database (fallback constant)"


class _SmiSampler(threading.Thread):
    """nvidia-smi power draw once per second (+ CPU utilisation via psutil, for the fallback)."""

    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period, self.t, self.w, self.cpu = period, [], [], []
        self._halt = threading.Event()
        self.ok = shutil.which("nvidia-smi") is not None

    def sample(self):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            return sum(float(x) for x in out.split() if x.replace(".", "", 1).isdigit())
        except Exception:
            return None

    def run(self):
        import psutil
        psutil.cpu_percent(None)
        while not self._halt.is_set():
            t0 = time.time()
            w = self.sample() if self.ok else None
            if w is not None:
                self.t.append(time.time()); self.w.append(w)
            self.cpu.append(psutil.cpu_percent(None))
            self._halt.wait(max(0.0, self.period - (time.time() - t0)))

    def stop(self):
        self._halt.set(); self.join(timeout=5)
        if self.ok:   # close the integral at the stop time
            w = self.sample()
            if w is not None:
                self.t.append(time.time()); self.w.append(w)


class EnergyMeter:
    """Context manager around one stage. After exit, `.record` holds wall_s, energy_kwh,
    kg_co2e, the CPU/GPU/RAM split, peak GPU memory, and the nvidia-smi cross-check.

    method = "codecarbon" normally; "nvidia-smi+tdp" if codecarbon could not start or stop."""

    def __init__(self, stage, out_dir=None, country="IND", cpu_tdp_w=140.0, use_codecarbon=True, **extra):
        self.stage, self.out_dir, self.country, self.cpu_tdp_w = stage, out_dir, country, cpu_tdp_w
        self.use_cc, self.extra, self.record = use_codecarbon, extra, None
        self.cc = None

    def __enter__(self):
        try:
            import torch
            self._cuda = torch.cuda.is_available()
            if self._cuda:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        except ImportError:
            self._cuda = False
        self.smi = _SmiSampler(); self.smi.start()
        if self.use_cc:
            try:
                import logging
                from codecarbon import OfflineEmissionsTracker
                logging.getLogger("codecarbon").setLevel(logging.ERROR)
                kw = dict(country_iso_code=self.country, project_name=self.stage, measure_power_secs=1,
                          log_level="error", save_to_file=self.out_dir is not None)
                if self.out_dir is not None:
                    Path(self.out_dir).mkdir(parents=True, exist_ok=True)
                    kw["output_dir"] = str(self.out_dir)
                self.cc = OfflineEmissionsTracker(**kw)
                self.cc.start()
            except Exception as e:     # noqa: BLE001 - any failure means: use the fallback
                self.cc, self.cc_error = None, repr(e)
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        if self._cuda:
            import torch
            torch.cuda.synchronize()
        wall = time.time() - self.t0
        rec = dict(stage=self.stage, wall_s=wall, **self.extra)
        cc_ok = False
        if self.cc is not None:
            try:
                kg = self.cc.stop()
                d = self.cc.final_emissions_data
                rec.update(method="codecarbon", energy_kwh=float(d.energy_consumed), kg_co2e=float(kg),
                           cpu_kwh=float(d.cpu_energy), gpu_kwh=float(d.gpu_energy), ram_kwh=float(d.ram_energy),
                           cpu_power_w=float(d.cpu_power), gpu_power_w=float(d.gpu_power),
                           cpu_model=str(d.cpu_model), gpu_model=str(d.gpu_model))
                cc_ok = True
            except Exception as e:     # noqa: BLE001
                self.cc_error = repr(e)
        self.smi.stop()
        gpu_smi = integrate_kwh(self.smi.t, self.smi.w)
        rec["gpu_kwh_smi"] = gpu_smi
        rec["cpu_util_mean"] = float(np.mean(self.smi.cpu)) if self.smi.cpu else float("nan")
        if not cc_ok:
            factor, src = _codecarbon_ind_factor()
            cpu = self.cpu_tdp_w * (rec["cpu_util_mean"] / 100 if self.smi.cpu else 1.0) * wall * KWH_PER_J
            rec.update(method="nvidia-smi+tdp", energy_kwh=gpu_smi + cpu, cpu_kwh=cpu, gpu_kwh=gpu_smi,
                       ram_kwh=0.0, kg_co2e=(gpu_smi + cpu) * factor, grid_factor_source=src,
                       codecarbon_error=getattr(self, "cc_error", ""))
        rec["gpu_peak_mem_mb"] = (__import__("torch").cuda.max_memory_allocated() / 2**20) if self._cuda else 0.0
        self.record = rec
        return False


def save_record(rec, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=1, default=float))


# ----------------------------------------------------------------------------- encoders
def load_encoder(name, dev, precision="fp16"):
    """(tokenizer, encoder, n_params). precision: fp32 | fp16 | bf16 (GPU) | int8 (CPU dynamic)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    enc = AutoModel.from_pretrained(name).eval()
    n_params = sum(p.numel() for p in enc.parameters())
    if precision == "int8":
        enc = torch.ao.quantization.quantize_dynamic(enc, {torch.nn.Linear}, dtype=torch.qint8)
        return tok, enc, n_params
    enc = enc.to(dev)
    if precision == "fp16":
        enc = enc.half()
    elif precision == "bf16":
        enc = enc.to(torch.bfloat16)
    return tok, enc, n_params


def embed_doc(tok, enc, sents, dev, n_slots, max_tokens=96, chunk=64, out_dtype=np.float16):
    """Masked-mean-pooled last-layer vectors for one document's sentences -> (n_slots, d).
    Identical loop to bppfinal.dl.embed (chunks of 64, truncation, fp32 pooling)."""
    import torch
    d = enc.config.hidden_size
    out = np.zeros((n_slots, d), out_dtype)
    with torch.no_grad():
        for a in range(0, min(len(sents), n_slots), chunk):
            ch = sents[a:min(a + chunk, n_slots)]
            b = tok(ch, padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(dev)
            h = enc(**b).last_hidden_state.float()
            mk = b["attention_mask"].unsqueeze(-1).float()
            out[a:a + len(ch)] = ((h * mk).sum(1) / mk.sum(1).clamp(min=1)).cpu().numpy()
    return out


def embed_all(sent_lists, tok, enc, dev, n_slots=200, max_tokens=96, out_dtype=np.float16, log=None):
    E = np.zeros((len(sent_lists), n_slots, enc.config.hidden_size), out_dtype)
    t0 = time.time()
    for i, ss in enumerate(sent_lists):
        E[i] = embed_doc(tok, enc, ss, dev, n_slots, max_tokens, out_dtype=out_dtype)
        if log and i % 100 == 0:
            log(f"   {i}/{len(sent_lists)}  {time.time()-t0:.0f}s")
    return E
