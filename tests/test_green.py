"""Tests for the green-AI study helpers (final_run/bppfinal/green.py)."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final_run"))
from bppfinal import green as gr  # noqa: E402


def test_integrate_kwh_constant_power():
    # 100 W for 36 s = 3600 J = 0.001 kWh
    t = np.arange(0, 37, 1.0)
    assert gr.integrate_kwh(t, np.full(len(t), 100.0)) == pytest.approx(0.001)


def test_integrate_kwh_trapezoid_and_degenerate():
    assert gr.integrate_kwh([0, 3600], [0, 2000]) == pytest.approx(1.0)   # ramp 0->2 kW over 1 h
    assert gr.integrate_kwh([5.0], [300.0]) == 0.0
    assert gr.integrate_kwh([], []) == 0.0


def test_mean_cosine_ignores_padding():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 5, 8)).astype(np.float32)
    B = A.copy()
    B[:, 3:] = -A[:, 3:]                 # slots 3-4 are padding and must not count
    mc, mn = gr.mean_cosine(A, B, [3, 2, 3])
    assert mc == pytest.approx(1.0, abs=1e-6) and mn == pytest.approx(1.0, abs=1e-6)
    mc_all, _ = gr.mean_cosine(A, B, [5, 5, 5])
    assert mc_all < 0.9


def test_mean_cosine_skips_zero_vectors_and_handles_fp16():
    A = np.ones((2, 2, 4), np.float16)
    B = A.copy(); B[0, 1] = 0            # a zero vector in B: undefined cosine, skipped
    mc, _ = gr.mean_cosine(A, B, [2, 2])
    assert mc == pytest.approx(1.0)


def test_prefix_mask_and_cap():
    m = gr.prefix_mask([0, 2, 5], 4)
    assert m.tolist() == [[False] * 4, [True, True, False, False], [True] * 4]
    assert gr.prefix_mask([5, 1], 4, cap=2).sum(1).tolist() == [2, 1]


def _folds(n=50):
    return np.arange(n) % 5


def test_seed_ensemble_matches_rank_mean_and_k():
    from bppfinal.stats import rank_norm
    rng = np.random.default_rng(1)
    F = _folds()
    oofs = [rng.random(50) for _ in range(4)]
    full = gr.seed_ensemble(oofs, F)
    assert np.allclose(full, np.mean([rank_norm(p, F) for p in oofs], 0))
    one = gr.seed_ensemble(oofs, F, k=1)
    assert np.allclose(one, rank_norm(oofs[0], F))
    assert np.allclose(gr.seed_ensemble(oofs, F, k=10), full)     # k beyond the list = all


def test_blend_is_half_half_of_within_fold_ranks():
    from bppfinal.stats import rank_norm
    rng = np.random.default_rng(2)
    F = _folds()
    a, b = rng.random(50), rng.random(50) * 100     # scale must not matter
    assert np.allclose(gr.blend(a, b, F), 0.5 * rank_norm(a, F) + 0.5 * rank_norm(b, F))
    assert np.allclose(gr.blend(a, a * 3, F), rank_norm(a, F))


def test_pareto_front():
    cost = [1, 2, 3, 4, 2]
    score = [0.5, 0.7, 0.6, 0.8, 0.7]
    # (3, 0.6) is dominated by (2, 0.7); the two (2, 0.7) points do not dominate each other
    assert gr.pareto_front(cost, score).tolist() == [True, True, False, True, True]


def test_choose_config_rule():
    rows = [
        dict(name="current", kwh=10.0, pr_all=0.797, boot_lo=0.0, boot_hi=0.0),
        dict(name="close", kwh=5.0, pr_all=0.790, boot_lo=-0.02, boot_hi=-0.001),      # within 0.010
        dict(name="ci0", kwh=1.0, pr_all=0.780, boot_lo=-0.03, boot_hi=0.004),          # CI includes 0
        dict(name="bad", kwh=0.1, pr_all=0.700, boot_lo=-0.12, boot_hi=-0.05),          # neither
    ]
    name, why = gr.choose_config(rows, 0.797, 0.010)
    assert name == "ci0" and "includes 0" in why
    name, why = gr.choose_config(rows[:2] + rows[3:], 0.797, 0.010)
    assert name == "close" and "within" in why
    assert gr.choose_config([rows[3]], 0.797)[0] is None
    # exactly on the threshold counts; a NaN interval never does
    edge = dict(name="edge", kwh=1.0, pr_all=0.787, boot_lo=math.nan, boot_hi=math.nan)
    assert gr.choose_config([edge], 0.797, 0.010)[0] == "edge"
    edge["pr_all"] = 0.7869
    assert gr.choose_config([edge], 0.797, 0.010)[0] is None


def test_reduction():
    pct, mult = gr.reduction(10.0, 2.5)
    assert pct == pytest.approx(75.0) and mult == pytest.approx(4.0)
    assert gr.reduction(1.0, 0.0) == (100.0, float("inf"))


def test_energy_meter_fallback_records_everything(tmp_path):
    pytest.importorskip("psutil")
    with gr.EnergyMeter("unit", None, use_codecarbon=False, cpu_tdp_w=100.0, tag="x") as m:
        sum(i * i for i in range(200000))
    r = m.record
    assert r["method"] == "nvidia-smi+tdp" and r["stage"] == "unit" and r["tag"] == "x"
    for k in ["wall_s", "energy_kwh", "kg_co2e", "cpu_kwh", "gpu_kwh", "gpu_kwh_smi", "gpu_peak_mem_mb"]:
        assert k in r and r[k] >= 0
    assert r["energy_kwh"] == pytest.approx(r["cpu_kwh"] + r["gpu_kwh"])
    # CPU estimate can never exceed TDP x wall time
    assert r["cpu_kwh"] <= 100.0 * r["wall_s"] / 3.6e6 + 1e-12
    gr.save_record(r, tmp_path / "e" / "unit.json")
    assert json.loads((tmp_path / "e" / "unit.json").read_text())["stage"] == "unit"


def test_india_grid_factor_is_plausible():
    f, src = gr._codecarbon_ind_factor()
    assert 0.5 < f < 1.0 and src


def _tiny_encoder(tmp_path):
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("transformers")
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + "the company has not defaulted in repayment of loans".split()
    (tmp_path / "vocab.txt").write_text("\n".join(words))
    tok = tr.BertTokenizer(str(tmp_path / "vocab.txt"))
    torch.manual_seed(0)
    enc = tr.BertModel(tr.BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=32)).eval()
    return torch, tok, enc


def test_embed_doc_masked_mean_and_padding(tmp_path):
    torch, tok, enc = _tiny_encoder(tmp_path)
    sents = ["the company has not defaulted", "repayment of loans", "the company"]
    E = gr.embed_doc(tok, enc, sents, "cpu", n_slots=5, max_tokens=8, chunk=2, out_dtype=np.float32)
    assert E.shape == (5, 16) and np.all(E[3:] == 0)
    # each slot equals the sentence embedded on its own (padding in a chunk must not leak in)
    for i, s in enumerate(sents):
        alone = gr.embed_doc(tok, enc, [s], "cpu", n_slots=1, max_tokens=8, out_dtype=np.float32)[0]
        assert np.allclose(E[i], alone, atol=1e-5)
    # a cap below the number of sentences keeps only the first n_slots
    E2 = gr.embed_doc(tok, enc, sents, "cpu", n_slots=2, max_tokens=8, chunk=64, out_dtype=np.float32)
    assert np.allclose(E2, E[:2], atol=1e-5)


def test_int8_dynamic_quantisation_close_to_fp32(tmp_path):
    torch, tok, enc = _tiny_encoder(tmp_path)
    if "qnnpack" not in torch.backends.quantized.supported_engines and \
            "fbgemm" not in torch.backends.quantized.supported_engines:
        pytest.skip("no quantized engine")
    q = torch.ao.quantization.quantize_dynamic(enc, {torch.nn.Linear}, dtype=torch.qint8)
    sents = ["the company has not defaulted", "repayment of loans"]
    a = gr.embed_doc(tok, enc, sents, "cpu", 2, 8, out_dtype=np.float32)
    b = gr.embed_doc(tok, q, sents, "cpu", 2, 8, out_dtype=np.float32)
    assert gr.mean_cosine(a[None], b[None], [2])[0] > 0.95


def test_trainer_accepts_smaller_encoder_width():
    torch = pytest.importorskip("torch")
    from bppfinal import dl
    rng = np.random.default_rng(0)
    n, d = 60, 12
    y = np.tile([0, 1], n // 2).astype(np.float32)
    groups = np.repeat(np.arange(n // 2), 2)
    fold = (groups % 5)
    T = dl.Trainer(y, groups, fold, rng.standard_normal((n, 3)), rng.standard_normal((n, 2)), "cpu")
    E = rng.standard_normal((n, 10, d)).astype(np.float16)
    M = gr.prefix_mask(rng.integers(1, 10, n), 10)
    cfg = dict(dl.CFG, n_sents=8, epochs=1, batch=8)
    te, p, g, a, info = T.train_fold(0, E, M, cfg, seed=0)
    assert len(te) == (fold == 0).sum() and np.all((p >= 0) & (p <= 1)) and a.shape == (len(te), 8)


# ----------------------------------------------------------------------------- study configuration
def test_sentence_cap_ladder_stops_at_the_network_window():
    """A cap above dl.CFG["n_sents"] cannot change the network's input, so it would duplicate the
    full run's predictions while measuring embedding work nothing reads."""
    import run_green as rg
    caps = {tag: v[2] for tag, v in rg.EMBED.items() if tag.startswith("finbert_fp16_n")}
    assert caps, "no sentence-cap variants"
    assert all(c <= rg.N_READ for c in caps.values()), caps
    assert rg.N_READ in caps.values()                       # embedding exactly what is read is a variant
    assert all(tag == f"finbert_fp16_n{c}" for tag, c in caps.items())
    dl = pytest.importorskip("bppfinal.dl")                 # needs torch
    assert dl.CFG["n_sents"] == rg.N_READ


def test_inference_benchmark_embeds_what_the_variant_embeds():
    import run_green as rg
    import green_report as rep
    for tag, bench in rep.INFER_OF.items():
        assert bench in rg.INFER, (tag, bench)
        assert rg.INFER[bench][3] == rg.EMBED[tag][2], (tag, bench)
        assert rg.INFER[bench][0] == rg.EMBED[tag][0], (tag, bench)


def test_study_reuses_baseline_runs_only_for_caps_at_the_window(tmp_path):
    import green_report as rep
    import run_green as rg
    (tmp_path / "runs").mkdir()
    for s in rg.SEEDS:
        np.savez(tmp_path / "runs" / f"dl_finbert_fp16_main_s{s}.npz", oof=np.full(5, 0.5))
    s = rep.Study.__new__(rep.Study)
    s.out = tmp_path
    runs, stages, src = s.dl_runs("finbert_fp16_n96")       # cap = window, no runs of its own
    assert src == "finbert_fp16" and len(runs) == len(rg.SEEDS) and stages[0] == "dl_finbert_fp16_main_s0"
    runs, stages, src = s.dl_runs("finbert_fp16_n64")       # below the window: must have its own runs
    assert src == "finbert_fp16_n64" and runs == [] and stages == []
    runs, stages, src = s.dl_runs("finbert_fp32")           # cap 200 but a different encoder precision
    assert src == "finbert_fp32" and runs == []
    np.savez(tmp_path / "runs" / "dl_finbert_fp16_n96_main_s0.npz", oof=np.full(5, 0.1))
    runs, stages, src = s.dl_runs("finbert_fp16_n96")       # trained after all (prefix not identical)
    assert src == "finbert_fp16_n96" and len(runs) == 1


def test_gpu_co2e_uses_the_records_own_grid_factor():
    rec = dict(energy_kwh=0.010, kg_co2e=0.00713, gpu_kwh=0.004, cpu_kwh=0.005, ram_kwh=0.001)
    assert gr.gpu_co2e(rec) == pytest.approx(0.004 * 0.713)
    assert gr.gpu_co2e(dict(energy_kwh=0.0, kg_co2e=0.0, gpu_kwh=0.0)) == 0.0   # no division by zero
    assert gr.gpu_co2e(dict(energy_kwh=0.01, kg_co2e=0.00713)) == 0.0             # no GPU field: nothing


def test_choose_config_breaks_energy_ties_by_accuracy():
    rows = [
        dict(name="current", kwh=10.0, pr_all=0.797, boot_lo=0.0, boot_hi=0.0),
        dict(name="ratios_only", kwh=0.202, pr_all=0.747, boot_lo=-0.094, boot_hi=0.002),   # CI clause only
        dict(name="ratios_lang", kwh=0.206, pr_all=0.782, boot_lo=-0.035, boot_hi=0.015),
        dict(name="n32_k1", kwh=1.27, pr_all=0.787, boot_lo=-0.019, boot_hi=0.010),
    ]
    name, why = gr.choose_config(rows, 0.797, 0.010)
    assert name == "ratios_lang" and "tie" in why and "ratios_only" in why
    assert gr.choose_config(rows, 0.797, 0.010, energy_tie=0.0)[0] == "ratios_only"   # the original rule
    # the band never reaches the next configuration
    assert gr.choose_config(rows, 0.797, 0.010, energy_tie=0.05)[0] == "ratios_lang"
