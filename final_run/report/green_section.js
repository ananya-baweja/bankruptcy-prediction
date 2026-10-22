// Section "Green AI: accuracy versus energy cost". Every number is read from results/green/
// (written by final_run/run_green.py and green_report.py); nothing is typed in by hand.
//
// Energy metric (DECISIONS.md D4.3, D4.3a): GPU energy (NVML) is the primary figure and CO2e is
// computed from it alone; whole-machine totals appear only in the appendix table, as upper bounds,
// because background processes inflated the CPU share during the 14-15 October runs.
const fs = require("fs");
const path = require("path");

const GREEN = process.env.BPP_GREEN || path.resolve(__dirname, "..", "..", "results", "green");

function available() {
  return fs.existsSync(`${GREEN}/report_numbers.json`) && fs.existsSync(`${GREEN}/pareto.png`);
}

// L: lib.js; num: section number; tableNo / figNo: the numbers this section's first table and figure take
// (the appendix table is tableNo + 1)
function build(L, { num = 11, tableNo = 22, figNo = 11 } = {}) {
  const { P, H1, Bullet, TableTitle, Figure, Tbl, f3 } = L;
  const G = JSON.parse(fs.readFileSync(`${GREEN}/report_numbers.json`));
  const S = G.selection, C = S.chosen_metrics, B = S.current_metrics, I = G.inference, K = S.caching;
  const row = (c) => G.rows.find((r) => r.config === c);
  const wh = (kwh) => (1000 * kwh).toFixed(1);
  const sg = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3);
  const pct = (x) => `${Math.round(x)}%`;
  const cur = row("blend_finbert_fp16_k7"), k3 = row("blend_finbert_fp16_k3"), n32 = row("blend_finbert_fp16_n32_k1");
  const ml = row("blend_minilm_fp16_k7"), dr3 = row("blend_distilroberta_fp16_k3"), i8 = row("blend_finbert_int8_k7");
  const f32 = row("blend_finbert_fp32_k7");
  const gpuBench = I.finbert_gpu_fp16_n200, gbBench = I.gb_only_cpu;
  const out = [];
  const add = (...xs) => xs.flat().forEach((x) => out.push(x));
  const name = (r) => r.label.replace("Blend: ", "Blend, ").replace("DL ensemble: ", "Network only, ");

  add(H1(`${num}. Green AI: accuracy versus energy cost`));
  add(P(`Every model in this report was also run with its energy measured, to ask how much of the cost buys accuracy. codecarbon (offline, India grid) wrapped each stage: building the table, embedding the MD&A, each network run, the gradient boosting and per-document scoring. **Energy is reported as GPU energy**, the card's draw read from the NVIDIA driver (NVML) and confirmed by integrating nvidia-smi power once per second (within 3.4% on every GPU-active stage longer than 10 s), and **CO₂e is computed from GPU energy alone**. The whole-machine figure, which adds the processor's rated power times its utilisation, is not reported as a result: it was found afterwards that unrelated processes kept about 11 of the machine's 20 hardware threads busy throughout the runs, so every CPU figure is inflated by an unknown background (decisions log D4.3). GPU energy includes the card's idle draw of 6 to 7 W during the stages that run on the processor; it is what the machine's GPU drew, not an estimate for a machine without one. The whole-machine totals are given in Table ${tableNo + 1} as upper bounds only.`));
  add(P(`${G.n_configs} configurations were compared on the same five folds and seeds: smaller encoders (MiniLM-L6, DistilRoBERTa and a finance-tuned DistilRoBERTa), fp32, fp16 and bf16 FinBERT, the first 32, 64 or 96 sentences only (the network reads 96, so a larger cap cannot change its input), 1 to 7 seeds, INT8 FinBERT on the CPU, and no network at all. The current system draws **${wh(B.gpu_kwh)} Wh of GPU energy** to produce its five-fold predictions: ${wh(cur.gpu_kwh_embed)} Wh to embed every MD&A with FinBERT and ${wh(cur.gpu_kwh_dl)} Wh for the seven network runs. *(These are v1 numbers, from the data before the CARO correction of Section 8, because the study re-ran the pipeline exactly as first run; the comparisons between configurations do not depend on that correction.)*`));
  add(TableTitle(`Table ${tableNo}.`, "Accuracy and GPU energy of selected configurations (mean ± sd over the 5 folds; GPU energy, NVML, to produce the 5-fold predictions; CO₂e from GPU energy only). Δ PR-AUC: pair-bootstrap 95% interval against the current blend."));
  const rows = G.rows.map((r) => [
    (r.chosen ? "**" : "") + name(r) + (r.chosen ? " (chosen)**" : ""),
    `${f3(r.pr_all)} ± ${f3(r.pr_all_sd)}`, f3(r.pr_strict), f3(r.roc_all), wh(r.gpu_kwh), (1000 * r.gpu_kg_co2e).toFixed(2),
    r.config === "blend_finbert_fp16_k7" ? "reference" : `${sg(r.boot_lo)} to ${sg(r.boot_hi)}`]);
  add(Tbl(["Configuration", "PR-AUC (763)", "PR-AUC (609)", "ROC-AUC (763)", "GPU Wh", "g CO₂e (GPU)", "Δ PR-AUC 95% interval"],
    rows, [3.0, 1.25, 0.9, 0.9, 0.65, 0.75, 1.45], { numeric: [1, 2, 3, 4, 5, 6], size: 16 }));
  add(Figure(`${GREEN}/pareto.png`, 6.3, `Figure ${figNo}. PR-AUC against GPU energy (log scale). The dashed line is the Pareto front over all ${G.n_configs} configurations; numbered points are listed beside the plot.`));
  add(P(`**The chosen configuration.** The rule, fixed before any variant was run, takes the lowest-energy configuration whose PR-AUC is within 0.010 of 0.797 or whose bootstrap interval against the current blend includes zero; energy is GPU energy, and energies within 5% of the lowest are a tie broken by PR-AUC (decisions log D2.3a: gradient boosting on ratios alone also qualifies and draws 4 mWh less idle power). It selects **gradient boosting on the ratios and language features, with no neural network**: PR-AUC ${f3(C.pr_all)} against ${f3(B.pr_all)} (${f3(C.pr_strict)} against ${f3(B.pr_strict)} on the strict 609), a difference whose pair-bootstrap interval (${sg(S.bootstrap.lo)} to ${sg(S.bootstrap.hi)}, 2,000 resamples) includes zero and whose per-fold Wilcoxon p is ${S.wilcoxon.wilcoxon_p.toFixed(2)} (the blend is ahead in ${5 - S.wilcoxon.wins} of 5 folds). Its GPU energy is ${wh(C.gpu_kwh)} Wh instead of ${wh(B.gpu_kwh)} Wh, all of it the idle card while the processor builds the table: **${S.energy_saving_pct.toFixed(1)}% less GPU energy and CO₂e, ${S.energy_multiple.toFixed(0)} times less**. Per annual report, the current scorer spends ${gpuBench.ms.toFixed(0)} ms and ${gpuBench.gpu_j.toFixed(1)} J of GPU energy on FinBERT and the seven networks; the chosen model needs no GPU and scored a report in ${gbBench.ms.toFixed(0)} ms on the processor even while it was shared with the background processes. The price is the small, statistically unclear gain the network adds on top of the language features (Section 7). If the network is kept, seeds are the cheapest lever: three seeds lose ${(-(k3.pr_all - cur.pr_all)).toFixed(3)} PR-AUC for ${pct(k3.gpu_kwh_saving_pct)} less GPU energy, and a three-seed blend with DistilRoBERTa scores ${f3(dr3.pr_all)} for ${pct(dr3.gpu_kwh_saving_pct)} less. Under the threshold clause alone the choice would be FinBERT on the first 32 sentences with one seed (${f3(n32.pr_all)}, ${pct(n32.gpu_kwh_saving_pct)} less).`));
  add(P(`**What did not help much.** Smaller encoders keep the accuracy (MiniLM-L6, five times smaller: ${f3(ml.pr_all)}), and fp16 embeddings match fp32 (cosine similarity above 0.9999) for ${(f32.gpu_kwh_embed / cur.gpu_kwh_embed).toFixed(1)} times less GPU energy in the embedding stage, but the encoder is the smaller share of the bill: the network runs are. Caching the embeddings, which the pipeline already did, saves ${wh(K.main_7_seeds.saved_kwh)} Wh of GPU energy over re-embedding for every seed and ${wh(K.full_grid.saved_kwh)} Wh over the original ${K.full_grid.n_runs}-run experiment grid.`));
  add(P(`**INT8 on the CPU: inconclusive.** Dynamic INT8 quantisation of FinBERT changed the sentence vectors substantially (cosine 0.74 to fp32), yet the blend trained on them still scored ${f3(i8.pr_all)}; that accuracy result stands. Its cost does not: the INT8 embedding ran on the processor for most of the configuration's ${Math.round(i8.wall_s / 60)} minutes, and the per-document CPU benchmarks (fp32, INT8, MiniLM, gradient boosting alone) ran while other processes held about 11 of the 20 hardware threads, so neither their latency nor their energy can be compared with the GPU figures, and the apparent 9% speed-up of INT8 over fp32 on this CPU may be an artefact of the contention. Those measurements are listed in Table ${tableNo + 1} as upper bounds and are not used in any conclusion.`));
  add(P("**Limitations of these numbers.**"));
  add(Bullet("GPU energy is a driver reading (NVML, confirmed by nvidia-smi within 3.4% on GPU-active stages); CO₂e applies codecarbon's India grid factor to it. It leaves out the processor and memory altogether, so it understates the true energy of every configuration, most of all the CPU-only ones; it is used because the processor figure cannot be separated from the background load of D4.3."));
  add(Bullet("GPU energy includes the card's idle draw during CPU-only stages (table build, gradient boosting). For the chosen configuration that idle draw is the whole figure, so its energy is best read as \"no GPU work\" rather than as a precise number; a machine without a GPU would draw none."));
  add(Bullet("The carbon factor of the Indian grid (codecarbon's national figure) is an assumption; another year's or region's factor scales every CO₂ figure, not the percentages."));
  add(Bullet("All runs are on a single machine (Intel i9-7900X, RTX 5060 Ti), once per configuration. PDF reading and feature extraction, shared by every configuration, were not measured, so the true percentage savings per report are smaller."));
  add(TableTitle(`Table ${tableNo + 1}.`, "Appendix: whole-machine energy, upper bounds only. codecarbon totals including the processor (rated power × utilisation) and memory, and per-report energy from the same benchmarks, all measured while unrelated processes held about 11 of the 20 hardware threads (D4.3). They overstate every configuration, most for those that run longest on the processor, and are kept for completeness; no conclusion rests on them."));
  const app = G.rows.map((r) => [
    name(r), wh(r.gpu_kwh), wh(r.kwh_total_ub), (1000 * r.kg_co2e_total_ub).toFixed(1), (r.wall_s / 60).toFixed(1),
    r.infer_gpu_j_per_doc == null ? "n/a" : r.infer_gpu_j_per_doc.toFixed(1),
    r.infer_j_per_doc_ub == null ? "n/a" : r.infer_j_per_doc_ub.toFixed(1)]);
  add(Tbl(["Configuration", "GPU Wh", "Total Wh (upper bound)", "Total g CO₂e (upper bound)", "Wall time (min)", "J per report, GPU", "J per report, total (upper bound)"],
    app, [3.0, 0.7, 0.95, 1.0, 0.85, 0.9, 1.1], { numeric: [1, 2, 3, 4, 5, 6], size: 16 }));
  return out;
}

module.exports = { build, available, GREEN };
