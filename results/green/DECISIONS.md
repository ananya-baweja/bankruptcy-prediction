# Green AI study: decisions log

Every choice made during the green-AI study, in the order it was made. Written by the agent that
ran the study (branch `green-ai`). Sections 1-3 were written before any variant was run.

## 0. Inputs and where they came from

| Input | Location used | sha256 (first 16) | Note |
| --- | --- | --- | --- |
| folds.csv | `~/bpp_final/folds.csv` | `98004781cee7e722` | Not in git (`final_run/.gitignore`). 763 rows, folds of 153/153/153/152/152. Read as-is, never recomputed. |
| documents_labeled.csv | `data/data/` | `c78e344c44a829dd` | Identical (md5) to `~/bpp_final/data/` |
| language_features.csv | `data/data/` | `06325875b103259f` | **v1** = before the CARO-flag correction (see D0.1) |
| ratios.csv | `data/data/` | `c1405fcc351a1695` | |
| report_sections_1/2.jsonl.gz | `data/data/` | `398ead59...` / `c3199726...` | |
| Existing DL results | `~/bpp_final/results/` (v1) and `~/bpp_final/results_v2/` (v2) | | Read only, never written |
| Existing FinBERT cache | `~/bpp_final/cache/embstore_*_finbert_200x96.npy` | | Read only; used to check the re-embedding is identical |

**D0.1 Which data version is "the baseline".** The prompt's expected PR-AUCs (ratios 0.750,
ratios+language 0.786, DL 0.758, blend 0.797) were checked against both saved runs, using the
repository's own code (`bppfinal.table.build`, `cpu_models.hgb_oof`, the 7-seed rank-averaged
`main` ensemble and the 50/50 rank blend from `analyze.py`):

| Model | Expected | v1 (`data/data`, `results/`) | v2 (`data_v2`, `results_v2/`) |
| --- | --- | --- | --- |
| GB ratios | 0.750 | 0.7475 (Δ −0.003) | 0.7475 (Δ −0.003) |
| GB ratios + language | 0.786 | 0.7820 (Δ −0.004) | 0.7950 (Δ +0.009) |
| DL ensemble (7 seeds) | 0.758 | 0.7546 (Δ −0.003) | **0.7829 (Δ +0.025)** |
| Blend | 0.797 | 0.7915 (Δ −0.006) | 0.8069 (Δ +0.010) |

All four v1 numbers are within 0.01, while v2's DL ensemble misses by 0.025. v1 is also the version
in `data/` that the prompt names. **So the baseline is v1**, i.e. the data before the CARO-flag
correction. Note that the published final report (and README) quote v2 numbers (blend 0.807).
Every green variant changes only the text encoder, its precision, its input length, the number of
seeds or the presence of the network, and none of these touches the language-feature CSV. The
*relative* accuracy and energy comparisons therefore do not depend on the CARO fix, but the
absolute numbers in the Green AI section are v1 numbers. The report section says so.

**D0.2 Where outputs go.** The repository has no top-level `results/`; the existing results are
in `~/bpp_final/`. Everything new goes to `results/green/` in this repository. Large arrays
(embedding caches) go to `results/green/cache/`, which is git-ignored. Small per-run
out-of-fold predictions (`results/green/runs/*.npz`, ~3 KB each) and energy records are committed.

**D0.3 Environment.** The `bpp` conda env lacked `codecarbon`, `matplotlib`, `pytest` and the
`bpp` package itself. All four were installed with pip (codecarbon 3.3.1, matplotlib 3.11.2,
pytest 9.1.1, `pip install -e .[dev]`). torch 2.11.0+cu128, transformers 5.18.0, sklearn 1.9.1,
the same versions as the saved v2 run (`run_info.json`).
Machine: Intel i9-7900X (10 cores), 15 GB RAM, NVIDIA RTX 5060 Ti 16 GB.

**D0.4 Pre-existing test failure.** Before any change, `pytest` gives one failure:
`tests/test_demo_cli.py::test_demo_runs_end_to_end` (the synthetic demo's CARO annexure is empty
for some documents). See section 7 for the investigation. "Existing tests must still pass" is
read as: no test that passed before may fail after.

## 1. How the pipeline works (Step 1)

Sources: `final_run/run_gpu.py`, `final_run/bppfinal/{dl,table,cpu_models,stats}.py`,
`final_run/analyze.py`, `notebooks/1_NLP_text_to_features.ipynb`,
`notebooks/2_DL_model_and_evaluation.ipynb`.

**Table.** `bppfinal.table.build` joins documents_labeled, language_features, ratios and the
report sections into 763 company-years (401 healthy, 362 later insolvent) in 135 matched pairs.
The folds come from `folds.csv` (grouped by matched pair). `strict` = the row is itself included
and has financials: 609 rows. Features: 11 ratios, 53 language columns (incl.
`financials_missing` and a fold-wise trigram-LM perplexity of the MD&A, `perplexity_fold`,
fitted on the training folds' healthy reports).

**Text encoder.** `ProsusAI/finbert` (BERT-base, 12 layers, 768-d, ~110 M parameters), frozen,
never fine-tuned. Run in **fp16 on the GPU** already (`enc.half()` in `dl.embed`).

**Sentence embedding and pooling.** Each section (MD&A, auditor's report) is split on `.!?`
followed by whitespace; sentences with fewer than 5 words are dropped; the first 200 sentences
are kept (`MAX_SENTS=200`). Sentences are tokenised in chunks of 64, truncated to 96 tokens
(`MAX_TOKENS`), and each sentence vector is the attention-masked **mean of FinBERT's last hidden
layer** over its tokens. Result: (763, 200, 768) float16 per section, cached on disk by doc_id
(`cache/embstore_<section>_finbert_200x96.npy`). So embeddings are **already computed once and
reused** by every seed and every experiment.

**Document pooling inside the network.** `FusionNet` reads only the first `n_sents=96` sentence
vectors: a BiGRU (32 units per direction, packed so padding is skipped), additive attention
(32 hidden) over the GRU states gives a document vector, projected to 64-d. Language features
and ratios each go through a 2-layer MLP (64-d; batch-norm on the ratio branch). A softmax gate
over the three 64-d blocks weights them, and a 2-layer head gives the logit.

**DL training.** Loss: focal BCE (γ=1) + 0.1·BCE. AdamW, lr 1e-3, no weight decay, batch 16,
dropout 0.5, gradient clipping at 1.0. Up to 40 epochs, early stopping with patience 8 on the
validation PR-AUC. For each outer fold k (5, from folds.csv), the other four folds are split by
`GroupShuffleSplit(test_size=0.2, random_state=seed)` on matched pairs into train/validation;
features are median-filled and standardised with training statistics. Seeds: the main model
(MD&A + language + ratios, "main") is trained with **7 seeds (0-6)**; the strict-subset model
also with 7; the ablations with 5. The **DL ensemble** = mean over seeds of the within-fold
rank-normalised out-of-fold probabilities (`analyze.py: ens`).

**GB models.** `HistGradientBoostingClassifier(random_state=0)`, sklearn defaults, trained per
fold on the other four folds: "ratios only" = 11 ratios (`hgb_ratios`), "ratios + language" =
11 ratios + 53 language columns (`hgb_both`). No tuning.

**Blend.** `0.5·rank_norm(DL ensemble) + 0.5·rank_norm(hgb_both)`, ranks taken within each fold.

**Metric.** ROC-AUC and PR-AUC computed per fold and averaged over the 5 folds (sd across folds).

## 2. Study design (written before any variant result was seen)

**D2.1 What a "config" is and what its energy covers.** A config is everything needed to produce
the out-of-fold predictions behind its all-763 PR-AUC:

    E_config = E(embed MD&A, all 763 docs) + Σ_seeds E(DL main run, 5 folds) + E(GB ratios+language, 5 folds)

* Only the **MD&A** embedding is counted, because the main network and the blend read only the
  MD&A. The current script also embeds the auditor's report (used only by ablations); that cost
  is measured and reported separately as a saving in its own right, not hidden in the baseline.
* The 609-row strict numbers are computed, for every config, from the same out-of-fold
  predictions restricted to the 609 strict rows (the all-763-trained model scored on its strict
  rows). This is one identical evaluation for every config. For the baseline only, the separately
  trained 7-seed `strict` network is also re-run and reported (it is what the final report calls
  the strict model), so its energy is known too.
* The table build (incl. the trigram-LM perplexity feature) is shared by every config including
  "no neural model"; it is measured once and reported as a separate line, and included in the
  "GB only" config's total (variant g) so that number is the full CPU cost.
* Upstream PDF reading and language-feature extraction are shared by all configs and were not
  re-run (their outputs are the input CSVs).

**D2.1a Amendment (before any variant result existed).** The table build (dominated by the
fold-wise trigram-LM perplexity) is needed by every config, since the network and both GB models
read the language features. It is therefore added to **every** config's total, not only to g's.
This slightly overstates the ratios-only GB model, which does not need the perplexity.

**D2.1b Measurement hygiene.** codecarbon measures the whole machine (CPU power is TDP × load,
because RAPL counters are not readable without root here). So nothing else heavy is run while
the study runs: the test suite and other CPU work wait until it has finished.

**D2.2 Seeds and folds.** Every variant uses the same `folds.csv`, the same seeds 0-6 and the same
`GroupShuffleSplit` validation splits as the baseline. GB uses `random_state=0` as in the repo.

**D2.3 Pre-registered selection rule (Step 5).** Among all configs that produce all-763
out-of-fold predictions, choose the one with the **lowest E_config (kWh)** such that

1. its all-763 PR-AUC ≥ 0.797 − 0.010 = **0.787** (0.797 is the prompt's reference value for
   the current blend), **or**
2. the 95% paired pair-bootstrap interval (2,000 resamples of matched pairs,
   `bppfinal.stats.pair_bootstrap`) of its PR-AUC minus the current blend's includes 0.

The chosen config is then reported against the re-run current blend with that bootstrap and a
per-fold Wilcoxon signed-rank test (`bppfinal.stats.paired`), and its energy and CO₂ reduction
as a percentage and as a multiple. Configs that only measure deployment inference (variant f
latency) are not eligible because they have no accuracy of their own; f's accuracy is checked by
training the network on INT8 embeddings (D3.f).

**D2.4 Energy measurement.** `codecarbon.OfflineEmissionsTracker(country_iso_code="IND")`
around every stage, one tracker per stage (embedding per section, each DL run, each GB model,
the inference benchmark). Recorded per stage: wall time, kWh (and its CPU/GPU/RAM split),
kg CO₂e, peak GPU memory (`torch.cuda.max_memory_allocated`, reset per stage), and the encoder's
parameter count. In parallel, nvidia-smi power draw is sampled once per second and integrated
(trapezoid rule) as an independent check of the GPU share. If codecarbon fails to start, the
fallback is that sampler for the GPU plus a CPU estimate of TDP × mean CPU utilisation, with
the India grid factor taken from codecarbon's own data. Which method was used is stored in each
record (`method`).

## 3. Variant-specific choices (written before running them)

* **a. Caching.** Already done (D1). Quantified as: E(embed MD&A once) vs the counterfactual of
  re-embedding inside each of the 7 seeds (7×), and for the whole original experiment grid
  (45 network runs incl. the leak test, each reading none, one or both sections).
* **b. Smaller encoders.** Order of attempts: a distilled finance-domain model, then
  `sentence-transformers/all-MiniLM-L6-v2`, then `distilroberta-base`. All are run through the
  same splitter, truncation (96 tokens) and masked mean pooling as FinBERT, so only the encoder
  changes; FusionNet's input width follows the encoder (`dim = E.shape[-1]`, a one-line change
  in `dl.Trainer` that leaves the 768-d case identical). MiniLM's sentence-transformers recipe
  also uses mean pooling (it then L2-normalises; that step is skipped to keep pooling identical).
* **c. Precision.** The baseline is already fp16. Variant c therefore embeds in **fp32** (the
  "non-green" reference) and **bf16**, checks the mean cosine similarity of each against fp32
  (> 0.99 required), and trains the network on each.
* **d. Input reduction.** N = 32, 64, 128 first sentences per section at embedding time. The
  network reads only the first 96 slots, so N=128 yields the same network input as the full 200;
  if the N=128 vectors are bit-identical to the first 128 of the full run, the baseline's network
  predictions are reused for N=128 (documented) and only the embedding energy changes.
  *Amended 2026-10-16 (D5.4a): N=128 replaced by N=96, the network's window.*
* **e. Seeds.** k = 1, 3, 5, 7 using the first k seeds (0..k−1) of the re-run baseline; energy =
  embedding + the measured energy of those k runs + GB. Because it costs no extra compute, k is
  also crossed with every embedding variant (b, c, d, f); all those combinations are configs in
  `green_results.csv` and are eligible under D2.3. (Amended before any variant result existed.)
* **f. INT8.** `torch.ao.quantization.quantize_dynamic(encoder, {nn.Linear}, torch.qint8)` on CPU.
  Per-document latency and energy measured on a fixed sample of documents for: GPU fp16
  (current), CPU fp32, CPU INT8, and the small encoders. Accuracy: the network is trained on
  MD&A embeddings produced by the INT8 encoder.
* **g. No neural model.** GB ratios+language alone on CPU; energy = table build (incl. perplexity)
  + 5-fold GB. Also GB ratios-only for reference.

## 4. Step 3 outcome (baseline reproduction)

Re-run with measurement on (`baseline_check.json`), PR-AUC on all 763 rows:

| Model | Expected | Re-run | Difference |
| --- | --- | --- | --- |
| GB ratios | 0.750 | 0.7475 | −0.0025 |
| GB ratios + language | 0.786 | 0.7820 | −0.0040 |
| DL ensemble (7 seeds) | 0.758 | 0.7546 | −0.0034 |
| Blend | 0.797 | 0.7915 | −0.0055 |

All within 0.01, so the study continued. The 7 re-run networks' out-of-fold predictions are
**bit-identical** to `~/bpp_final/results/dl_oof.csv` (max |difference| = 0 for every seed), and
the table checksum matches (`2a66afb78a14db79`). The expected values are 0.003-0.006 above the
exact re-run values, consistent with rounding or another copy of the numbers. The selection
threshold stays at the pre-registered 0.797 − 0.010 = 0.787, which is slightly *stricter* than
using the re-run 0.7915.

**D4.1 Run order.** The first launch embedded every variant before training any network, so the
Step 3 gate would only have been checked after ~2 h of variant compute. It was stopped after
two finished stages (table build, FinBERT MD&A embedding; both kept, as the run is resumable) and
restarted as: baseline embedding → 7 baseline networks → GB → automatic gate
(`run_green.py: Run.gate`, which writes REPRODUCTION_REPORT.md and exits if any value is off by
more than 0.01) → strict networks → variants (GPU ones first, INT8 CPU last) → inference
benchmarks.

**D4.2 Contaminated baseline energy, re-measured.** A whole-disk `find /` started during the
initial survey (before the study) kept running until 22:52, overlapping the measured table build,
the FinBERT MD&A embedding and the 7 baseline network runs (22:16-22:43). Since codecarbon's CPU
figure is TDP × machine-wide load, these records may be inflated. After the study finished, those
stages were re-run on a quiet machine; the first records are kept in `energy/contaminated/` for
comparison and the clean ones are used everywhere. Predictions are unaffected (bit-identical).

**D4.3 (2026-10-16) The whole 14-15 October study ran under the same background load.** After
the foreign processes of D5.4a were found, the saved records were checked for their presence on
14-15 October. The machine was not rebooted between 13 Oct 17:14 and 16 Oct 07:51, so the study
(14 Oct 22:16 to 15 Oct 03:52, including the D4.2 re-measurement) ran in one boot session; the two
processes start within 20 s of boot today, i.e. they are persistent, and the process table of the
earlier session is gone. The energy records settle it: codecarbon's sampler stores the machine-wide
CPU utilisation of every stage, and the stages that do almost no CPU work of their own read
64-68% (every GPU embedding stage), 71% (the 0.6 s cache-load probe) and 74-83% (GPU inference)
on 14-15 October, while the network runs read 90% at 107 W. Today, with the processes present and
the idle machine at 61% utilisation, the same stages read 69.6% (embedding) and 90.4% at 107.5 W
(network runs): the same numbers. On a quiet machine an embedding loop reads 5-15% and a cache
load is idle. The 14-15 October measurements are therefore **all contaminated in absolute terms**,
the D4.2 "quiet" re-run included (the `find /` of D4.2 was a further 7% on top).

Size of the effect, from codecarbon's own CPU model: the background accounts for about 45-58 W of
the reported CPU power, i.e. roughly 16-20 Wh of the current blend's 47.4 Wh (35-43%) and 1.3-1.7 Wh
of the chosen configuration's 1.85 Wh (70-90%), and about 10 J of the 29 J per GPU-scored report.
GPU energy (NVML, cross-checked by nvidia-smi) is unaffected. Consequences: (i) every absolute Wh and
g CO2e figure in SUMMARY.md, green_results.csv, inference.csv and the report section is an
over-estimate; (ii) the ranking of configurations with similar wall time is barely affected, since
the bias is background power x wall time; (iii) the headline saving of the chosen configuration is
**under**-stated, not over-stated (removing the background from both sides gives roughly 30 Wh
against 0.4 Wh); (iv) the Pareto front and the selection, which depend on the ordering, are the same.
No record is corrected by hand. The study should be re-measured on a clean machine (`run_green.py`
is resumable: delete `energy/*.json` and re-run; predictions are bit-identical) before any absolute
number is quoted; until then the report's energy figures carry this caveat.

**D2.3a (2026-10-16, post hoc) Energy ties.** Under the GPU metric of D4.3a the configurations
with no GPU work (gradient boosting alone) differ only by the card's idle draw over a few seconds
of wall time: GB on ratios alone reads 0.202 Wh and GB on ratios + language 0.206 Wh. Both
qualify under D2.3 (the ratios-only model through the lenient CI clause, as the sensitivity note
already recorded), so the rule as written would now pick the model that is 0.035 PR-AUC worse
because of 4 mWh of idle draw. D2.3 is therefore amended: energies within 5% of the lowest are a
tie, broken by the higher PR-AUC (`choose_config(..., energy_tie=0.05)`). The band is far below
the repeatability of any stage (GPU energy of the same network run varies by more than 5% between
seeds because early stopping changes the epoch count) and far above the gap to the next cheapest
configuration (1.27 Wh), so the choice is not sensitive to it. With the whole-machine metric the
amended rule gives the same answer as the original one.

**D4.3a (2026-10-16) Reporting only uncontaminated numbers.** Following D4.3 the write-up
(SUMMARY.md, pareto.png, the report section, selection.json) uses **GPU energy** as the primary
metric: `gpu_kwh`, codecarbon's NVML reading per stage (within 4% of the nvidia-smi integral on
every GPU-active stage), with **CO2e computed from GPU energy alone** (`gpu_kg_co2e`, same India
grid factor). GPU energy includes the card's idle draw (6-7 W) during CPU-only stages such as the
table build and gradient boosting; it is what the machine's GPU drew, not an estimate of a
GPU-less deployment, and the text says so. The pre-registered rule D2.3 is applied unchanged with
"lowest kWh" read as lowest GPU kWh; the Pareto front is over GPU kWh. Whole-machine totals
(`kwh_total_ub`, `kg_co2e_total_ub`, `j_per_doc_ub`) remain in `green_results.csv`,
`inference.csv` and one appendix table, labelled as upper bounds. The INT8-on-CPU result (variant
f) is restated as inconclusive: its embedding (84 minutes of CPU) and its per-document benchmarks
ran while other processes held about 11 of the 20 hardware threads, so neither its latency nor its
energy can be compared with the GPU figures; its accuracy result stands. The ordering of the GPU
configurations is unchanged by the metric (the table build and GB contribute only idle GPU draw).

## 5. Findings during the run that changed nothing in the design

**D5.1 Precision checks (variant c).** Mean cosine similarity to FinBERT fp32 over all real
MD&A sentences (`embedding_checks.json`): fp16 0.9999996 (min 0.99998), bf16 0.99998 (min 0.9992):
both pass > 0.99. The freshly computed fp16 embeddings are bit-identical to the existing cache.

**D5.2 INT8 (variant f) fails the faithfulness check.** `quantize_dynamic(FinBERT, {nn.Linear},
qint8)` gives mean cosine **0.743** (min 0.48) against fp32. A diagnostic on one 64-sentence
MD&A (not part of the evaluation): every quantized engine available (x86, fbgemm, onednn,
qnnpack) gives 0.74-0.78; leaving only the FFN output projections (`BertOutput.dense`) in fp32
gives 0.984. This is the known activation-outlier problem of BERT's feed-forward outputs under
per-tensor dynamic activation quantization. The variant was kept exactly as specified (all Linear
layers) and the network was trained on its embeddings. The CPU (i9-7900X, Skylake-X) has AVX-512
but no AVX-512 VNNI, so INT8 matrix products get little hardware speed-up: INT8 was only ~9%
faster than fp32 on CPU per document.

**D5.3 N=32 cap is not a prefix of the full run.** With 32 sentences the first (only) chunk is
padded to the longest of 32 sentences, not 64, which changes fp16 rounding slightly, so the N=32
vectors are not bit-identical to the first 32 of the full run (N=64 and N=128 are; N=96 is not,
for the same reason: its second chunk holds 32 sentences). Irrelevant to the variant: its network
was trained on its own embeddings.

**D5.4 N=128 reuses the baseline networks**, as pre-registered (D3.d): its first 128 slots are
bit-identical to the full run's and the network reads 96, so its out-of-fold predictions are
those of the baseline; only its embedding energy differs. *Superseded by D5.4a.*

**D5.4a (2026-10-16) N=128 replaced by N=96.** Reviewing the table showed the N=128 rows equal to
the current blend to every decimal (by construction, D5.4) although 36% of the MD&As have more
than 128 sentences, and their per-document inference figures were those of the `n96` benchmark.
The cause is not a computation error but an incoherent variant: the network reads only the
first 96 slots (`dl.CFG["n_sents"]`), so (i) a cap of 128 cannot change its input and the
"variant" is the baseline under another name, (ii) 32 of its 128 slots per document (10,233
sentences, 16% of its embedding work) were embedded and never read, and (iii) its training energy
was that of a 128-cap while its inference benchmark embedded 96. The only cap at or above the
window that is a real deployment option is 96 itself: embed exactly what the network reads. The
variant was therefore redefined as N=96, with the ladder now 32, 64, 96 (`run_green.py`: caps
checked against `N_READ = dl.CFG["n_sents"]`, test `test_sentence_cap_ladder_stops_at_the_network_window`),
`INFER_OF` mapping it to `finbert_gpu_fp16_n96` (test `test_inference_benchmark_embeds_what_the_variant_embeds`),
and the run-reuse rule of D3.d generalised to any cap ≥ 96 whose vectors are a bit-identical
prefix of the full run (`reuses_baseline_runs`). This is a post-hoc design change; it cannot
affect the chosen configuration (every N=96 config includes at least one 6 Wh network run, against
the chosen 1.85 Wh) nor the sensitivity analysis (N=96 with 1 seed embeds more than N=32 with 1 seed,
the threshold-clause pick at 6.9 Wh), and the N=128 record is kept in `energy/superseded/`.

Outcome: the N=96 vectors are **not** a bit-identical prefix of the full run (D5.3: the second
chunk is padded to the longest of 32 sentences), so the rule did not apply and the 7 networks were
trained on them, as for N=32. The accuracy numbers in `green_results.csv` are theirs.

**Energy of the N=96 stages: measured under background load, parked.** When the N=96 stages ran
(2026-10-16, 17:37 onwards) the machine carried two foreign processes started at boot
(`/root/.b0s`, root, ~6.5 cores, and an "nginx: worker process" owned by user `user`, ~4.4 cores;
load average 19 on 20 threads, 62% CPU busy before the study job started; reported to the owner).
Under the D4.2 standard those records are contaminated: the embedding read 1.94 Wh (CPU 69.6%
busy, 56.9 W, against 65% and 50.9 W for the clean 64- and 128-cap runs; the GPU share, 1.01 Wh,
scales exactly with the sentence count and is sound), and the CPU-bound network runs would be
inflated by more than the difference they are meant to show. The records are in
`energy/contaminated/` and are **not used**: `green_report.py` therefore skips the N=96 configs
until they are re-measured on a quiet machine with

    cd final_run && python run_green.py --only finbert_fp16_n96 --stages embed,check,dl
    python green_report.py && cd report && node build_green_section.js

(the resumable run re-embeds and re-trains because the energy records are absent; outputs were
bit-identical on re-run in D4.2). After that, fill the Wh and g CO2e cells of the N=96 row and the
configuration count (74 → 82) in SUMMARY.md.

**D4.2 outcome.** Re-measured on a quiet machine: table build 2.62 → 1.80 Wh, FinBERT MD&A
embedding 2.92 → 2.55 Wh, the 7 baseline networks 49.9 → 43.0 Wh in total (per run −1% to −26%),
GB models within ±0.02 Wh. All re-run outputs (table, embeddings, 15 out-of-fold prediction sets)
are bit-identical to the first run. The clean records are used; the old ones are in
`energy/contaminated/`. Strict seeds 4-6 and every variant ran after 22:52 and were not re-run.

## 6. Step 5 outcome

Rule D2.3 applied by `green_report.py` (`bppfinal.green.choose_config`) to all 82 configs (74 while the N=96 energy is parked, D5.4a)
(`selection.json`): **chosen `gb_both`**, GB on ratios + language, no neural model, qualifying
through the bootstrap clause (PR-AUC 0.782 < 0.787, but the pair-bootstrap 95% CI against the
re-run current blend is −0.035 to +0.015). Against the current blend: per-fold Wilcoxon p = 0.31
(PR-AUC) and 0.31 (ROC-AUC), the blend ahead in 4 of 5 folds; energy 1.85 vs 47.4 Wh = −96.1%,
25.6×; CO2e identical in relative terms (one grid factor). The rule was applied as written, but the
report and SUMMARY also state that the CI clause is lenient (GB ratios-only also passes it) and
give the threshold-only choice (FinBERT, 32 sentences, 1 seed: 0.787, 6.9 Wh, −85%) and the cheapest
no-loss blends (DistilRoBERTa 3 seeds 0.796 at 18.9 Wh; MiniLM-L6 3 seeds 0.791 at 16.3 Wh).

**D6.1 "Current blend" in the comparisons** is the re-run current blend (PR-AUC 0.7915, bit-identical
to the saved run), while the threshold uses the prompt's 0.797, as pre-registered.

**D6.2 Inference benchmark design.** 40 documents drawn with `numpy.random.default_rng(0)`, 2 warm-up
documents unmeasured, looped until ≥ 45 s and ≥ 1 pass. Per document: sentence split, encoder on
the MD&A, all 7 seed networks (as `scorer/models.py` runs them; random weights, since the cost does
not depend on the values), GB `predict_proba`. Encoder loading is excluded. Each benchmark is
machine-level, so CPU benchmarks include the idle GPU's draw.

## 7. Report, viva, tests

* **Report.** New section "11. Green AI: accuracy versus energy cost" in
  `final_run/report/green_section.js`, called from `build_report.js` after Section 10. Conclusion,
  Reproducibility and References became 12-14, cross-references updated, the files table became
  Table 23 and lists the green files. Every number is read from `results/green/report_numbers.json`
  and `selection.json`. The full report could **not** be rebuilt here (see section 8); the section
  alone was rendered with `node build_green_section.js` to `results/green/Green_AI_section.docx`.
* **Viva.** No viva Q&A document existed (searched the repository, both report .docx files and
  `~/bpp_final`). Created `docs/viva_QA.md` with the 4 Green AI questions.
* **Tests.** `tests/test_green.py`: 20 tests (integration, cosine, masks, seed ensemble, blend,
  Pareto front, selection rule incl. edge cases, reduction, energy meter fallback path, India factor,
  embedding loop vs single-sentence embedding, INT8 on a tiny BERT, Trainer with a 12-d encoder;
  added 2026-10-16: the sentence-cap ladder stops at the network window, every inference benchmark
  embeds what its variant embeds, run reuse applies only to caps at the window; GPU-only CO2e from a
  record's own grid factor, energy ties broken by PR-AUC).
  Full suite: 424 passed, 1 skipped, 1 failed. The failure is the pre-existing
  `test_demo_cli.py::test_demo_runs_end_to_end`, which fails identically with all changes stashed
  (code as on `main`). Its assertion is that every synthetic demo document has a non-empty CARO
  annexure; some come out empty in this environment (freshly installed pymupdf 1.28.2, no
  Tesseract). Unrelated to this work and not changed.

## 8. Failed or skipped, with reasons

| Item | Status | Reason |
| --- | --- | --- |
| Full report rebuild (.docx) | Skipped | `build_report.js` needs `results_final/tables.json`, `results_cpu/extras.json`, `scoring/scoring_summary.json` and `archive_v1/results_final/tables.json`, none of which is on this machine (they live on the Drive). The repo has no PDF build script; the builder makes a .docx. The new section was rendered alone. |
| INT8 faithfulness check (cosine > 0.99) | Failed | 0.743 (D5.2). Variant kept as specified and evaluated. |
| codecarbon fallback (nvidia-smi + TDP) | Not needed | codecarbon worked for every stage; the fallback path is covered by a unit test only. |
| Strict-609 networks for variants | Not trained | By design (D2.1): variants are scored on the 609 rows from their 763-row predictions. The baseline's separately trained strict network was re-run (blend 0.818 on 609, `selection.json`). |
| N=128 variant | Replaced by N=96 | A cap above the network's 96-slot window is the baseline under another name (D5.4a). Its embedding record is in `energy/superseded/`. |
| N=96 energy | Measured under background load, not used | Foreign root processes kept the machine at load 19 (D5.4a); records parked in `energy/contaminated/`, re-measure on a quiet machine. The N=96 predictions are valid and kept. |
| The 30 ablation runs of the original grid (notext, text_*, fusion_*, leak) | Not re-run | Not part of the four baseline numbers or any green config; used only to quantify caching (counted, not run). |
| Upstream PDF reading / language-feature extraction energy | Not measured | Shared by all configs; inputs are the processed CSVs. Stated as a limitation. |
| bf16 and N=32 per-document inference | Not benchmarked | Inference covered fp16/fp32 GPU, N=64/96, CPU fp32/INT8; bf16 was indistinguishable from fp16 in embedding and accuracy. |
| `test_demo_cli` | Fails (pre-existing) | See section 7. |
| Data version | Note | The study uses v1 data (matches the expected baseline); the final report's other sections use v2. |
