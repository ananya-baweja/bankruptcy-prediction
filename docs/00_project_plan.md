# Project plan — Multimodal insolvency prediction for Indian listed firms

Courses: Natural Language Processing, Deep Learning (3rd year).
This file is the agreed plan. Changes to it go in `decisions_log.md` with a date and reason.

## 1. Problem

Accounting ratios lag: by the time the balance sheet shows distress, the firm is already close to
insolvency, and statements can be window-dressed or filed late. Before NCLT admission under the IBC
(2016), management language shifts: more hedging, defensive wording about working capital, auditor
qualifications, and debt refinancing struggles. We fuse annual-report narratives with ratios to
predict NCLT admission 12–24 months ahead, and test that claim directly.

## 2. Design decisions that fix weaknesses in the original statement

| # | Issue | Decision |
| --- | --- | --- |
| 1 | 150–200 firms is too few documents for deep learning | Firm-year panel (t-1, t-2, t-3 per firm → ~500–800 reports); pretrained embeddings (FinBERT); GroupKFold by company |
| 2 | Label timing undefined → leakage | Label by publication date vs NCLT admission date; drop reports published after the petition / within 180 days of admission (Phase 2 code) |
| 3 | Stopword removal destroys hedges, pronouns, syntax | Two text tracks: **raw** (parsing, coref, hedging, perplexity, transformers) and **cleaned** (n-grams, LM dictionary counts, Word2Vec/GloVe) |
| 4 | "Perplexity" needs a reference model | Kneser-Ney n-gram LM trained on **healthy-firm** MD&A (training folds only); perplexity = departure from normal disclosure |
| 5 | Lesk is weak for finance terms | Keep as an ablation (static embeddings ± Lesk vs contextual FinBERT); entity masking justified as leakage prevention |
| 6 | Missing strongest Indian signal | Add Independent Auditor's Report (qualified opinion, going concern, emphasis of matter) and CARO annexure; add promoter pledge % and rating downgrades to tabular stream |
| 7 | Single data source | Prowess if available, else XBRL/annual reports; OCR for scanned PDFs; missing/late filing as a feature |
| 8 | Balanced 1:1 sample overstates accuracy | Report PR-AUC, recall at 10% FPR, Brier; one test at realistic prevalence (~1:10) |
| 9 | "12–24 months" and "no business-group overfitting" untested | Separate t-1 / t-2 models; adversarial business-group head with group-prediction accuracy reported |
| 10 | No baselines/ablations/explainability | Altman Z″, logistic regression, XGBoost; text-only / ratios-only / fused; DeLong tests; SHAP + attention/IG |

## 3. Methodology (Phases 4–6)

**Stream A — narrative text.** Sentences from MD&A, Board's report and auditor's report (after
entity masking) → frozen FinBERT (768-d) → BiGRU with attention pooling (attention weights = sentence-level
explanations). Syllabus comparison: Word2Vec/GloVe + 1D-CNN.

**Stream B — engineered linguistic features (~20–30).** Loughran-McDonald negative/uncertainty/weak-modal
ratios, hedging density, perplexity under the healthy-firm LM, distress SVO triplet counts, going-concern /
qualified-opinion / CARO-default flags, Fog index, length → small MLP.

**Stream C — ratios.** ~10 ratios + promoter pledge % + Altman Z″ → MLP (BatchNorm, Dropout, L2).

**Fusion.** Learned gate weights each stream before concatenation (gate values show text vs number reliance).
Focal loss, AdamW, early stopping on validation PR-AUC, 5-fold GroupKFold by company, Optuna nested in
training folds. Honest baseline to beat: XGBoost on B + C + averaged FinBERT vectors.

## 4. Differentiating contributions

- **A. Linguistic drift (main).** Year-over-year change in MD&A embeddings, hedging, tone and perplexity;
  temporal GRU over t-3, t-2, t-1 document vectors ("Lazy Prices" idea applied to Indian insolvency).
- **B. Adversarial debiasing.** Gradient-reversal head predicting business group/industry; show group accuracy
  drops to chance while distress AUC holds.
- **C. Indian IBC distress lexicon.** Log-odds with informative Dirichlet prior (Monroe et al.) over n-grams;
  publish on GitHub.

## 5. Phases and timeline

| Phase | Weeks | Content | Status |
| --- | --- | --- | --- |
| 0 Setup | 1 | repo, config, CLI, tests, synthetic demo | **code done** |
| 1 Cohort | 1–3 | IBBI CIRP list → listed firms → reviewed matches → healthy peers → firm-year frame | **code done**, data pending |
| 2 Documents | 2–4 | download/collect PDFs, OCR, section extraction, labels + leakage rules, QA | **code done**, data pending |
| 3 Ratios | 3–4 | liquidity, profitability, leverage, solvency, pledge %, Altman Z″; winsorise; fold-wise scaling | not started |
| 4 NLP engine | 4–8 | NER masking, coref, SVO + hedging, LM dictionary, n-gram LM perplexity, Lesk ablation, FinBERT + Word2Vec embeddings, drift features, lexicon mining | not started |
| 5 Baselines | 8 | Altman Z″, logistic regression, XGBoost (ratios; ratios + linguistic) | not started |
| 6 Deep model | 9–11 | GatedFusionNet, adversarial head, temporal drift variant, CNN comparison | not started |
| 7 Experiments | 11–12 | modality ablations, embedding/WSD, masking, adversarial, drift, horizon t-1 vs t-2; DeLong | not started |
| 8 Explain + write-up | 12–14 | SHAP, attention sentences, gate values, lexicon release, optional Streamlit demo | not started |

### Experiment grid (Phase 7)

| Experiment | Purpose |
| --- | --- |
| Ratios only / text only / linguistic only / fused | modality complementarity |
| Word2Vec+CNN vs Word2Vec+Lesk+CNN vs FinBERT+BiGRU | embedding and WSD choices |
| With vs without entity masking | leakage from names |
| With vs without adversarial head (+ group accuracy) | contribution B |
| Single year vs three-year drift | contribution A |
| Horizon t-1 vs t-2 | the 12–24-month claim |

Metrics for every run: AUC, PR-AUC, F1, recall at 10% FPR, Brier (mean ± std over folds).

## 6. Key references

- Mai et al. (2019), *EJOR* — deep learning with textual disclosures for bankruptcy prediction; averaged embeddings are a strong baseline.
- Arno et al. (2023), ECL dataset (github.com/henriarnoUG/ECL) — modalities are complementary; horizon ambiguity.
- Cohen, Malloy & Nguyen (2020), "Lazy Prices", *Journal of Finance* — changes in 10-K language predict outcomes incl. bankruptcy.
- Business Perspectives (2023) — 57 IBC firms vs 55 matched solvent firms; dictionary tone and themes (our motivation).
- Journal of Prediction Markets (2022) and Meena et al. (2024, SSRN) — ratio-only IBC prediction benchmarks.
- Tools/data: ProsusAI/finBERT, yya518/FinBERT, Loughran-McDonald Master Dictionary, IBBI CIRP announcements.

## 7. Team roles (3–4 people)

Data & scraping · NLP engine · Models & experiments · Baselines, explainability & report (rotate to bottlenecks).
Phases 1–2 are the timeline risk: start them immediately and in parallel.
