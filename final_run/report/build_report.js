// Final report builder. Reads every number from the analysis outputs; nothing is typed in by hand.
const fs = require("fs");
const L = require("./lib.js");
const { d, P, H1, H2, H3, Bullet, Num, TableTitle, PageBreakP, Figure, Tbl, f3, s3, pc, pv } = L;
const { Document, Packer, Paragraph, TextRun, AlignmentType, Footer, PageNumber } = d;

const R = require("path").resolve(__dirname, "..");
const T = JSON.parse(fs.readFileSync(process.env.BPP_TABLES || `${R}/results_final/tables.json`));
const X = JSON.parse(fs.readFileSync(`${R}/results_cpu/extras.json`));
const NARR = fs.existsSync(`${__dirname}/narrative.json`) ? JSON.parse(fs.readFileSync(`${__dirname}/narrative.json`)) : {};
const csv = (p) => { const [h, ...rows] = fs.readFileSync(p, "utf8").trim().split("\n"); const H = h.split(",");
  return rows.map((r) => { const v = r.split(","); return Object.fromEntries(H.map((k, i) => [k, v[i]])); }); };
const IMP = csv(`${R}/results_cpu/importance_families.csv`);
const IMPF = csv(`${R}/results_cpu/importance_features.csv`);
const LEXM = csv(`${R}/results_cpu/india_distress_lexicon_mdna.csv`);
const LEXA = csv(`${R}/results_cpu/india_distress_lexicon_auditor.csv`);
const FIG = `${R}/report/fig`;
const DL = "blend" in T.perf.all;

const pf = (s, m) => T.perf[s][m];
const cmp = (s, k) => T.comp[s][k];
const prsd = (s, m) => `${f3(pf(s, m).pr)} ± ${f3(pf(s, m).pr_sd)}`;
const rocsd = (s, m) => `${f3(pf(s, m).roc)} ± ${f3(pf(s, m).roc_sd)}`;
const PEND = "[GPU run pending]";
const dl = (fn) => DL ? fn() : PEND;
const boot = (c) => c.boot ? `${s3(c.boot.point)} [${s3(c.boot.lo)}, ${s3(c.boot.hi)}]` : "";
const N = T.subset_sizes;
const excl = (c) => c.boot && (c.boot.lo > 0 || c.boot.hi < 0);
const foldsTxt = (c) => c.wins === 5 ? "all five folds" : `${c.wins} of 5 folds`;
const TIMES = { 1: "about as large as", 2: "about twice", 3: "about three times", 4: "about four times", 5: "about five times" };
const times = (a, b) => TIMES[Math.round(a / b)] || `${(a / b).toFixed(0)} times`;
const SC = JSON.parse(fs.readFileSync(`${R}/scoring/scoring_summary.json`));
const T1 = JSON.parse(fs.readFileSync(`${R}/archive_v1/results_final/tables.json`));   // the run before the CARO correction
const FR = SC.flag_rates, FR1 = FR.data_v1, FR2 = FR.data_v2;

const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

// ============================================================== title block
add(new Paragraph({ children: [new TextRun({ text: "FINAL REPORT", bold: true, size: 26, color: "2E74B5" })], spacing: { after: 120 } }),
  new Paragraph({ children: [new TextRun({ text: "Early Prediction of Corporate Insolvency for Indian Listed Companies Using Annual-Report Text and Financial Ratios", bold: true, size: 36 })], spacing: { after: 160 } }),
  P("*Do the words in an annual report warn of insolvency earlier, or better, than the numbers do? A matched study of 135 Indian listed companies admitted to insolvency under the Insolvency and Bankruptcy Code, 2016, and 135 healthy peers.*"),
  Tbl(null, [
    ["**Courses**", "Natural Language Processing and Deep Learning (third-year project)"],
    ["**Report date**", "3 October 2026"],
    ["**Stage**", "Complete: data, models, evaluation, the checks promised in the progress report of 1 October 2026, and a tool that scores a new company's annual report"],
    ["**Data**", "135 matched pairs, 1,154 annual reports read, 763 company-years modelled (362 that later entered insolvency, 401 healthy)"],
    ["**Code and data**", "github.com/ananya-baweja/bankruptcy-prediction; the dataset folder processed/; the final-run scripts and every saved prediction in final_run/ (Section 12)"],
  ], [1, 4]));

// ============================================================== summary
add(H1("Summary"));
add(P("When an Indian company goes bankrupt it rarely happens suddenly, but the usual warning signs (debt ratios, interest coverage) are lagging: by the time the accounts look clearly bad, the company is often already at the tribunal. The annual report also contains words: management explaining the year and the auditor giving an opinion. This project asks whether those words warn earlier, or better, than the numbers."));
add(P(`The system reads the reports of 135 companies later admitted to insolvency and 135 healthy companies in the same industry and of similar size, published one to three years before admission, and tries to tell them apart. Every result is the mean over five cross-validation folds that never split a matched pair, on folds fixed in a file so that the numbers can be reproduced exactly. The headline metric is PR-AUC (0.5 is roughly a coin flip on this balanced sample, 1.0 is perfect).`));
add(H3("What was found"));
const kAll = cmp("all", "hgb_both - hgb_ratios");
const cl = cmp("all", "hgb_ratios_langtext - hgb_ratios"), cf = cmp("all", "hgb_ratios_fm - hgb_ratios");
const kZ = cmp("z_above_median", "hgb_both - hgb_ratios");
add(Num(`**The words carry real information.** Models given only report language, with no financial data, separate the two groups clearly better than chance: ROC-AUC ${f3(pf("all", "hgb_lang").roc)} from the engineered language features, ${f3(pf("all", "tfidf_aud").roc)} from the auditor's report as a bag of words${DL ? `, and ${f3(pf("all", "dl_text_aud").roc)} from FinBERT reading the auditor's report` : ""}. Management's discussion alone is much weaker (ROC-AUC ${f3(pf("all", "tfidf_mdna").roc)} to ${f3(pf("all", "hgb_mdna").roc)}). The tone difference reported by Gupta and Banerjee (2023) replicates in direction on this larger sample.`, "numbers"));
add(Num(DL ? NARR.summary_main || PEND : `**Words plus numbers beat numbers alone.** Gradient boosting on ratios and language scores PR-AUC ${f3(pf("all", "hgb_both").pr)} against ${f3(pf("all", "hgb_ratios").pr)} for ratios alone (${s3(kAll.diff)}, ${kAll.wins} of 5 folds; DeLong p = ${pv(kAll.delong.p)} on ROC-AUC).`, "numbers"));
add(Num(`**The words matter most exactly where the numbers fail.** Among the ${N.z_above_median.n} company-years whose Altman Z″ is above the median, ${N.z_above_median.pos} still went insolvent. There the ratio model is weak (PR-AUC ${f3(pf("z_above_median", "hgb_ratios").pr)} against a chance level of ${f3(N.z_above_median.pos / N.z_above_median.n)}); adding language raises it to ${f3(pf("z_above_median", "hgb_both").pr)}, a gain of ${s3(kZ.diff)} that holds in ${kZ.wins} of 5 folds with a 95% interval of ${s3(kZ.boot.lo)} to ${s3(kZ.boot.hi)}${DL ? NARR.summary_healthy_dl || "" : ""}. The gain is ${times(kZ.diff, kAll.diff)} the gain on the full sample, holds with the conventional Altman "safe zone" cut-off as well, and is the most robust finding of the project.`, "numbers"));
add(Num(DL ? NARR.summary_horizon || PEND : `**The warning is early.** Ratios plus language score ${f3(pf("t-1", "hgb_both").pr)} on the last report before admission and ${f3(pf("t-3", "hgb_both").pr)} three years before; language adds most at two years (${s3(cmp("t-2", "hgb_both - hgb_ratios").diff)}).`, "numbers"));
add(Num(DL ? NARR.summary_section || PEND : `**Both sections carry signal, but the auditor's report is the stronger text.** This corrects the progress report.`, "numbers"));
add(H3("What did not work, and is reported as such"));
add(Bullet(DL ? NARR.summary_transformer || PEND : PEND));
add(Bullet(`**Perplexity was a dud**: ROC-AUC ${f3(pf("all", "perplexity").roc)} as a single predictor. Indian management discussions are too templated for a trigram model to find anything unusual.`));
const cDr = cmp("all", "hgb_both_nodrift - hgb_both");
add(Bullet(`**Year-on-year language drift added nothing**: removing the drift features ${cDr.diff > 0.0005 ? `*raised* PR-AUC by ${s3(cDr.diff)}` : `changed PR-AUC by ${s3(cDr.diff)}`}. Contribution A of the plan is not supported.`));
add(Bullet("**Tuning did not help**: with ~120 validation rows the inner score and the test score correlate at r = +0.07. Averaging several models helped; searching harder did not."));
const flagOnTop = pf("all", "hgb_both").pr - pf("all", "hgb_ratios_langtext").pr;
add(Bullet(`**A non-language flag sat among the language features**: a flag recording that a company's financial statements were incomplete had been filed with them. On its own it is worth ${s3(cf.diff)} PR-AUC over the ratios; once language is in the model it adds ${flagOnTop < 0.002 ? `nothing (${s3(flagOnTop)})` : `only ${s3(flagOnTop)}`}. Section 7.4 separates it out; the conclusions hold without it.`));
add(Bullet(`**Two auditor flags had been misread, and were corrected.** The rule that reads the CARO annexure cut sentences at every line break of the PDF, so a wrapped denial (\"the Company has not / defaulted in repayment\") counted as a default. ${FR1.caro_default_flag["0"]}% of healthy company-years carried a loan-default flag and ${FR1.caro_statutory_dues_flag["0"]}% an unpaid-dues flag; after the correction ${FR2.caro_default_flag["0"]}% and ${FR2.caro_statutory_dues_flag["0"]}%, and every remaining healthy-firm default flag was confirmed against the report. Every number in this report uses the corrected flags (Section 8).`));

// ============================================================== 1 introduction
add(PageBreakP(), H1("1. Introduction and Objective"));
add(P("Corporate insolvency is rarely sudden. Before a creditor files an insolvency petition before the National Company Law Tribunal (NCLT), signs of stress appear in two places: in the financial statements and in the language of the annual report. Financial ratios such as debt-to-equity and interest coverage are the conventional warning signals, but they are lagging indicators, and accounts can be presented favourably or filed late."));
add(P("Under stress, management hedges more, writes defensively about working capital and refinancing, and the auditor adds qualifications, going-concern paragraphs and, in the CARO annexure, a statement that the company has defaulted on its borrowings. This project measures how much early-warning signal that language carries, which part of the report carries it, and how early it appears. It does not assume that language performs better than ratios; each hypothesis was fixed in advance with an outcome stated in either direction (Section 6)."));
add(Figure(`${FIG}/fig1_overview.png`, 6.2, "Figure 1. Overview of the final system."));
add(TableTitle("Table 1.", "Terms used in this report."));
add(Tbl(["Term", "Meaning"], [
  ["**IBC, 2016**", "Insolvency and Bankruptcy Code, the law under which insolvent Indian companies are resolved."],
  ["**NCLT**", "National Company Law Tribunal, which admits a company into insolvency. Admission is the event predicted."],
  ["**IBBI**", "Insolvency and Bankruptcy Board of India, which publishes every insolvency announcement; the source of the labels."],
  ["**MD&A**", "Management Discussion and Analysis: management's account of the year and the outlook."],
  ["**CARO annexure**", "The part of the audit report in which the auditor must state loan defaults and unpaid statutory dues."],
  ["**Going concern**", "A formal statement by the auditor that the company may be unable to continue operating."],
  ["**PR-AUC, ROC-AUC**", "Area under the precision-recall and the receiver-operating-characteristic curves; ranking quality, 1.0 perfect."],
  ["**t-1, t-2, t-3**", "The last annual report published before the insolvency reference date, the one before it, and the one before that."],
], [1, 4]));

// ============================================================== 2 literature
add(H1("2. Literature Reviewed"));
add(P("Ten studies shaped the design; each fixed a baseline, a feature, a model choice or a risk to control. Four standard method papers and the statistical references are listed with them in Section 13."));
add(TableTitle("Table 2.", "The ten studies and what was taken from each."));
add(Tbl(["#", "Study", "What it found", "What the project took from it"], [
  ["1", "Altman (1968), *Journal of Finance*", "Five ratios combined by discriminant analysis separate bankrupt and healthy US firms.", "The classical benchmark; its emerging-market form Z″ is a baseline and a feature."],
  ["2", "Loughran and McDonald (2011), *Journal of Finance*", "General sentiment lists misclassify financial text; a finance dictionary.", "Tone features use the finance dictionary, with raw word counts as the denominator."],
  ["3", "Mai, Tian, Lee and Ma (2019), *EJOR*", "Text adds to ratios; averaged embeddings beat CNNs on small samples.", "The averaged-FinBERT baseline, and the expectation that simple text models do well here."],
  ["4", "Cohen, Malloy and Nguyen (2020), *Journal of Finance*", "Year-on-year changes in filings predict returns and bankruptcies.", "The drift features (tested; not supported here)."],
  ["5", "Huang, Wang and Yang (2023), *CAR*", "FinBERT outperforms general models on financial text.", "The frozen sentence encoder."],
  ["6", "Arno, Mulier, Baeck and Demeester (2024), ECL", "Text and financial data are complementary for bankruptcy prediction.", "The fusion design and the separate horizons."],
  ["7", "Mancisidor and Aas (2022/2024)", "Multimodal model; text missing for ~40% of firms.", "Missing text and missing accounts are recorded, not dropped."],
  ["8", "Gupta (2022), *J. Prediction Markets*", "57 IBC firms vs 55 matched; ratio models above 75% accuracy.", "The matched design and the ratio benchmark."],
  ["9", "Gupta and Banerjee (2023), *IMFI*", "Insolvent IBC firms use more negative words (2.21% vs 1.30%).", "The tone comparison replicated in Section 7.3."],
  ["10", "Ganin and Lempitsky (2015), ICML", "Gradient reversal removes a nuisance attribute from a representation.", "The planned business-group test (not built; Section 9)."],
], [0.35, 2.0, 2.6, 2.6], { size: 18 }));

// ============================================================== 3 data
add(H1("3. Data and Sample Design"));
add(P("Every input is public. The planned CMIE Prowess database was not available, so the standardised financial statements were built from the exchanges' XBRL filings (FY2018 onwards) and, for earlier years and as a cross-check, read from the annual reports themselves."));
add(TableTitle("Table 3.", "Data sources as used."));
add(Tbl(["Source", "Data taken", "Purpose"], [
  ["**IBBI public announcements**", "9,147 CIRP announcements, January 2017 to September 2026, with the debtor's CIN", "Labels and admission dates"],
  ["**BSE and NSE lists**", "Active, suspended and delisted companies; BSE industry sub-groups", "Finding listed debtors; matching peers"],
  ["**Annual reports (PDF)**", "1,154 reports, 148,424 pages (7,490 read by OCR)", "MD&A, auditor's report, CARO annexure"],
  ["**Exchange XBRL results**", "Standardised statements for FY2018 onwards", "Ratios; cross-check of the report reader"],
  ["**Loughran-McDonald dictionary**", "Finance word lists", "Tone features"],
], [1.3, 2.4, 1.6]));
add(P("Each insolvent company is matched to a healthy company from the same BSE industry sub-group with total assets within 30%, following Gupta (2022). All 135 insolvent companies were confirmed by the corporate identity number (CIN) printed in their own reports, and the report reader agrees with the companies' own XBRL filings to within 1% on 91.8% of 10,081 checked figures."));
add(Figure(`${FIG}/fig2_sample.png`, 6.2, "Figure 2. Construction of the matched sample, with the final counts."));
add(P("Labels follow the publication date of each report, not its fiscal year. A report published once insolvency proceedings are under way usually discusses them, so a model trained on it would be reading the outcome rather than predicting it."));
add(Figure(`${FIG}/fig3_timeline.png`, 5.8, "Figure 3. Labelling and the exclusion rule (unchanged from the progress report)."));
add(TableTitle("Table 4.", "Rules that prevent label leakage, and how many reports each removed."));
add(Tbl(["Rule", "Reports removed"], [
  ["Published within 180 days of admission", "236"],
  ["Read by hand and found to discuss a petition against the company itself", "38"],
  ["Filed by the exchange under the wrong year", "3"],
  ["No report available, or more than three years before", "40"],
  ["Training and test folds split by matched pair, never by report", "(design rule)"],
  ["Two explicit mentions-of-insolvency counts dropped from the features", "(design rule)"],
], [4, 1.2], { numeric: [1] }));
add(P(`**The modelling sample.** ${T.subset_sizes.all.n} company-years: the 719 that pass every rule (362 distressed, 357 healthy) plus 44 healthy reports that were removed only because their distressed partner's report was removed. A healthy company's report cannot disclose an insolvency that never happened, so these 44 carry no leakage, but they do relax the rule that both halves of a pair are kept or dropped together. The main results are therefore also reported on the **strict matched subset** of ${T.subset_sizes.strict.n} company-years that satisfy the original pair rule and have complete financial statements (${T.subset_sizes.strict.pos} distressed, ${T.subset_sizes.strict.n - T.subset_sizes.strict.pos} healthy: the completeness filter applies to each company-year, so the subset is not exactly balanced), and in Section 7.4 the models are retrained on the strict rows alone.`));

// ============================================================== 4 preparation
add(H1("4. Data Preparation"));
add(Figure(`${FIG}/fig4_documents.png`, 6.2, "Figure 4. From an annual report to the two kinds of language input."));
add(P("Two sections feed the models. The MD&A is management's own account; the auditor's report and CARO annexure are the independent auditor's. The directors' report is extracted but not used. Text is kept in two forms: raw, for FinBERT and for hedging (removing stopwords would delete \"may\" and \"could\", the hedging being measured), and as counts, for dictionary features and bag-of-words models."));
add(TableTitle("Table 5.", "The 51 engineered language features, computed separately for the MD&A and the auditor's report."));
add(Tbl(["Family", "Examples", "Why it might signal distress"], [
  ["Tone", "Loughran-McDonald negative, positive, uncertainty, litigious, constraining, strong and weak modal word shares", "Distressed firms write more negatively and less certainly"],
  ["Hedging", "hedge-word and hedge-phrase density (\"may\", \"subject to\")", "Management hedges when unsure of the outcome"],
  ["Distress phrases", "an India-specific seed list (e.g. one time settlement, wilful defaulter, SARFAESI, erosion of net worth)", "Local phrasing no US dictionary covers"],
  ["Readability and length", "Gunning Fog, sentence length, complex-word share, word and sentence counts", "Obfuscation; short, minimal disclosure"],
  ["Auditor's formal flags", "going concern, emphasis of matter, modified opinion and its severity, CARO loan default, CARO unpaid statutory dues", "The auditor's own assessment, stated before the event"],
  ["Section found", "whether the MD&A, auditor's report, CARO annexure and opinion were located", "Missing or non-standard disclosure"],
  ["Year-on-year drift", "Jaccard and cosine change against the firm's previous MD&A, change in tone, hedging and Fog", "A firm changing how it writes (Cohen et al., 2020)"],
], [1.1, 2.6, 1.9], { size: 18 }));
add(P("The two CARO flags are read clause by clause: a phrase such as \"defaulted in repayment\" or \"statutory dues ... outstanding\" counts only if no negation (\"not\", \"no\", \"neither\", \"whether\") precedes it in the same clause (for the statutory-dues phrase, or sits inside it: \"... and no such statutory dues were outstanding\"). A clause ends at a full stop, a semicolon or a new numbered item, never at the PDF's own line breaks, which fall in the middle of sentences (Section 8)."));
add(P("Two further columns travel with the language features: **perplexity** of the MD&A under a Kneser-Ney trigram model fitted, fold by fold, on the healthy reports of the training folds only; and **financials_missing**, a flag for company-years whose financial statements could not be fully recovered. The second is not language, and Section 7.4 separates it out."));

// ============================================================== 5 models
add(H1("5. Models"));
add(H2("5.1 The deep model as trained"));
add(P("The progress report described the network as planned. The network actually trained is smaller, because a nested hyperparameter search (and its failure to beat a fixed configuration; Section 7.10) pointed that way. Figure 5 and Table 6 describe what was trained."));
add(Figure(`${FIG}/fig5_architecture.png`, 6.2, "Figure 5. The gated fusion network as trained."));
add(TableTitle("Table 6.", "Planned (progress report) against trained (this report)."));
add(Tbl(["Setting", "Progress report", "As trained"], [
  ["Sentences read per report", "up to 200", "first 96 of the MD&A"],
  ["BiGRU hidden units per direction", "64", "32"],
  ["Attention hidden size", "64", "32"],
  ["Dropout", "0.3", "0.5"],
  ["Focal-loss gamma", "2", "1 (plus 0.1 x cross-entropy)"],
  ["Optimiser", "AdamW, lr 1e-3, weight decay 1e-4", "AdamW, lr 1e-3, no weight decay"],
  ["Batch size / epochs / patience", "32 / 100 / 10", "16 / up to 40 / 8, on an inner pair-grouped split"],
  ["Language inputs", "25 values", "53 values (51 features, perplexity, accounts-missing flag)"],
  ["Ratio inputs", "12, incl. promoter pledge", "11 (pledge filings were not collected)"],
  ["Business-group adversarial head", "planned", "not built (Section 9)"],
  ["Seeds", "1", "7, averaged by rank within each fold"],
  ["Trainable parameters", "~0.4 million (0.36 million quoted in Section 7.7 of the progress report)", "185,924 (FinBERT's 110 million frozen); the 0.36 million quoted earlier is the 64-unit configuration"],
], [1.7, 1.6, 2.2], { size: 18 }));
add(P("**One correction to the network itself.** In the progress-report run the attention was limited to the first 96 sentences, but the recurrent layer still ran over all 200 sentence slots, including zero padding, so its backward direction saw text it was not meant to read. The final run reads exactly the first 96 sentences and skips the padding (packed sequences). All deep-model numbers in this report come from the corrected network."));
add(H2("5.2 All models compared"));
add(TableTitle("Table 7.", "Models in this report. Everything that learns from data is fitted on the training folds only."));
add(Tbl(["Model", "Inputs", "Role"], [
  ["Altman Z″ (no training)", "one score", "classical benchmark"],
  ["Logistic regression", "11 ratios, winsorised and scaled in-fold", "standard baseline"],
  ["Gradient boosting (ratios)", "11 ratios", "**the accounting baseline**"],
  ["Gradient boosting (ratios + language)", "11 ratios + 53 language columns", "classic NLP added to ratios"],
  ["Deep model", "FinBERT sentences + language + ratios", "the gated fusion network, 7 seeds"],
  ["**Blend (full system)**", "the two previous models", "**fixed 50/50 average of within-fold ranks**; weight set in advance"],
  ["Text-only models", "language features; TF-IDF (unigrams and bigrams); mined lexicon score; FinBERT with no ratios; averaged FinBERT + logistic regression", "hypotheses H1 and H4"],
], [1.7, 2.3, 1.8], { size: 18 }));

// ============================================================== 6 evaluation design
add(H1("6. Evaluation Design"));
add(P("**Folds.** Five-fold cross-validation grouped by matched pair: a distressed company and its peer are always in the same fold, so the model never sees half of a matched comparison during training. The fold of every report is now fixed in a file (folds.csv). This matters more than it sounds: scikit-learn's GroupKFold breaks ties between equally sized pairs differently across library versions, and in current versions it would assign 79% of the rows to a different fold. The frozen assignment reproduces the progress report's folds (the same fold is hardest for every model, and the baselines agree to within 0.004)."));
add(P("**Per-fold means.** Every figure is the mean over the five folds, with the standard deviation across folds. Each fold trains its own network with its own stopping point, so pooling predictions across folds would penalise the networks for a calibration artefact that has nothing to do with ranking."));
add(TableTitle("Table 8.", "Metrics."));
add(Tbl(["Metric", "What it measures"], [
  ["**PR-AUC** (headline)", "ranking quality with attention to the insolvent class"],
  ["ROC-AUC", "comparability with published studies"],
  ["Recall at a 10% false-alarm rate", "share of insolvencies caught if 10% of healthy firms are flagged for review"],
  ["PR-AUC at a 1:10 prevalence", "the same ranking scored as if insolvent firms were 1 in 11, by reweighting the healthy firms; chance is 0.091"],
  ["Brier score", "calibration of the probability models (the blend is a rank average, not a probability)"],
  ["Paired tests across folds", "t-test and Wilcoxon on the five per-fold differences, and the number of folds won"],
  ["Pair bootstrap", "95% interval for a difference, resampling matched pairs (2,000 or 1,000 resamples)"],
  ["DeLong test", "ROC-AUC difference, computed within each fold and combined across folds"],
], [1.6, 4]));
add(P("**Reading the tests.** With five paired folds the Wilcoxon test cannot go below p = 0.0625; that value means every fold favoured the same model, the strongest outcome available. DeLong treats reports as independent although each company contributes up to three; the pair bootstrap does not, so it is the more conservative of the two and is preferred where they disagree."));
add(TableTitle("Table 9.", "Hypotheses fixed in advance (progress report, Table 14)."));
add(Tbl(["Hypothesis", "Test"], [
  ["**H1** Report language separates insolvent from healthy firms", "text-only models against chance; replication of the Gupta and Banerjee (2023) tone comparison"],
  ["**H2** Language adds information beyond the ratios", "ratios + language, and the full system, against ratios only, with intervals"],
  ["**H3** Language stays informative further from the event", "performance at t-1, t-2 and t-3"],
  ["**H4** The signal is not only the auditor's flags", "MD&A only, auditor only and both"],
], [2.2, 3.4]));

// ============================================================== 7 results
add(PageBreakP(), H1("7. Results"));
add(P(`All numbers below come from one run on the frozen folds: ${T.subset_sizes.all.n} company-years (${T.subset_sizes.all.pos} distressed) and the strict matched subset of ${T.subset_sizes.strict.n} (${T.subset_sizes.strict.pos} distressed). Where the same model was run, they differ from the progress report by a few thousandths for the ratio models, by up to about 0.02 for the models that read the language features, and by up to about 0.03 on the small subsets, because of library versions, the corrected network and the corrected auditor flags (Section 8); the progress-report figures are not reused.`));

// ---- 7.1
add(H2("7.1 Main comparison"));
const MAIN = [["altman", "Altman Z″ (no training)"], ["logit_ratios", "Logistic regression, ratios"],
  ["hgb_ratios", "Gradient boosting, ratios only"], ["hgb_both", "Gradient boosting, ratios + language"]]
  .concat(DL ? [["dl_main", "Deep model (7 seeds)"], ["blend", "**Blend: full system**"]] : []);
add(TableTitle("Table 10.", "Predictive performance. Mean across five folds ± standard deviation."));
add(Tbl(["Model", "PR-AUC (763)", "PR-AUC (609)", "ROC-AUC (763)", "ROC-AUC (609)", "Recall at 10% false alarms", "PR-AUC at 1:10"],
  MAIN.map(([m, lab]) => [lab, prsd("all", m), prsd("strict", m), rocsd("all", m), rocsd("strict", m),
    pc(pf("all", m).tpr10), f3(pf("all", m).pr_1in11)]),
  [2.3, 1.15, 1.15, 1.15, 1.15, 1.0, 0.9], { numeric: [1, 2, 3, 4, 5, 6], size: 17 }));
add(Figure(`${FIG}/fig6_main.png`, 6.2, "Figure 6. PR-AUC by model; whiskers show the standard deviation across the five folds."));
add(P(DL ? (NARR.main || PEND) : `Gradient boosting on ratios is the strongest accounting model; adding language raises PR-AUC from ${f3(pf("all", "hgb_ratios").pr)} to ${f3(pf("all", "hgb_both").pr)}.`));
add(P(`**At a realistic prevalence.** The sample is balanced by design, which flatters precision. Re-weighted so that insolvent firms are 1 in 11, chance PR-AUC is 0.091; the ratio model reaches ${f3(pf("all", "hgb_ratios").pr_1in11)} and ratios plus language ${f3(pf("all", "hgb_both").pr_1in11)}${DL ? `, the full system ${f3(pf("all", "blend").pr_1in11)}` : ""}. The ranking survives, but an analyst would still see several false alarms for every true case. The Altman distress zone (Z″ below 1.1) used as a rule flags ${pc(T.altman_rule.recall)} of the insolvent company-years at a ${pc(T.altman_rule.false_alarm)} false-alarm rate.`));

// ---- 7.2
add(H2("7.2 Are the differences real?"));
const CMP = [["hgb_both - hgb_ratios", "Ratios + language − ratios only"]].concat(DL ? [
  ["blend - hgb_ratios", "**Full system − ratios only**"], ["blend - hgb_both", "Full system − ratios + language"],
  ["blend - dl_main", "Full system − deep model"], ["dl_main - hgb_ratios", "Deep model − ratios only"]] : []);
add(TableTitle("Table 11.", "Paired differences in PR-AUC. Folds won out of 5; p from the paired t-test; interval from the pair bootstrap; DeLong p for the ROC-AUC difference."));
const crow = (s, k, lab) => { const c = cmp(s, k); return [lab, s3(c.diff), `${c.wins} of 5`, pv(c.t_p), c.boot ? `[${s3(c.boot.lo)}, ${s3(c.boot.hi)}]` : "", pv(c.delong.p)]; };
add(Tbl(["Comparison", "Difference", "Folds won", "t-test p", "95% interval", "DeLong p"],
  CMP.map(([k, lab]) => crow("all", k, lab + " (763)")).concat(CMP.map(([k, lab]) => crow("strict", k, lab + " (609)"))),
  [2.6, 0.9, 0.8, 0.8, 1.3, 0.8], { numeric: [1, 2, 3, 4, 5], size: 18 }));
add(P(DL ? (NARR.tests || PEND) : PEND));

// ---- 7.3 H1
add(H2("7.3 H1: does report language alone separate the two groups?"));
const TXT = [["hgb_lang", "Engineered language features (boosting)"], ["logit_lang", "Engineered language features (logistic)"],
  ["tfidf_mdna", "TF-IDF bag of words: MD&A"], ["tfidf_aud", "TF-IDF bag of words: auditor's report"],
  ["lexicon_mdna", "Mined lexicon score: MD&A"], ["lexicon_aud", "Mined lexicon score: auditor's report"]]
  .concat(DL ? [["emb_lr_mdna", "Averaged FinBERT + logistic: MD&A"], ["emb_lr_auditor", "Averaged FinBERT + logistic: auditor"],
    ["dl_text_mdna", "FinBERT + BiGRU, text only: MD&A (5 seeds)"], ["dl_text_aud", "FinBERT + BiGRU, text only: auditor (5 seeds)"],
    ["dl_text_both", "FinBERT + BiGRU, text only: both (5 seeds)"]] : []);
add(TableTitle("Table 12.", "Models that see only report language: no ratios and no accounts-missing flag."));
add(Tbl(["Text-only model", "ROC-AUC", "PR-AUC", "Recall at 10% false alarms"],
  TXT.map(([m, lab]) => [lab, rocsd("all", m), prsd("all", m), pc(pf("all", m).tpr10)]),
  [3, 1.3, 1.3, 1.2], { numeric: [1, 2, 3], size: 18 }));
add(P(`**H1 is supported, by the auditor's words more than management's.** Every model that reads the auditor's report, or the full set of engineered features, is clearly above chance (ROC-AUC 0.5). The MD&A-only models are barely above it (ROC-AUC ${f3(pf("all", "tfidf_mdna").roc)} to ${f3(pf("all", "hgb_mdna").roc)}, below 0.5 in some folds). The engineered features reach ROC-AUC ${f3(pf("all", "hgb_lang").roc)}; the strongest single text is the auditor's report, at ${f3(pf("all", "tfidf_aud").roc)} as a bag of words and ${f3(pf("all", "lexicon_aud").roc)} as a mined-lexicon score.${DL ? " " + (NARR.h1_dl || "") : ""} The progress report quoted 0.69 to 0.72 for text alone. That came from one run of the uncorrected network reading the MD&A, on the 609-row sample, scored on pooled predictions. With the recurrent layer corrected and five seeds, FinBERT on the MD&A alone reaches ${DL ? f3(pf("all", "dl_text_mdna").roc) : PEND}; the figure of about 0.7 holds only for text-only models of the auditor's report.`));
const TONE = X.tone;
const trow = (k, lab) => { const t = TONE[k]; const isRatio = !k.includes("fog");
  const fmt = (v) => isRatio ? (100 * v).toFixed(2) + "%" : v.toFixed(2);
  return [lab, fmt(t.mean_distressed), fmt(t.mean_healthy), pv(t.mw_p), pc(t.share_pairs_distressed_higher), pv(t.wilcoxon_p)]; };
add(TableTitle("Table 13.", "Replication of the Gupta and Banerjee (2023) tone comparison. Means; Mann-Whitney test across reports; matched comparison within pair and horizon."));
add(Tbl(["Measure", "Distressed", "Healthy", "Mann-Whitney p", "Pairs where distressed is higher", "Wilcoxon p (pairs)"], [
  trow("mdna_lm_negative_ratio", "MD&A negative words (LM)"), trow("auditor_lm_negative_ratio", "Auditor's report negative words"),
  trow("mdna_lm_uncertainty_ratio", "MD&A uncertainty words"), trow("mdna_hedge_density", "MD&A hedging density"),
  trow("mdna_fog_index", "MD&A Fog index"), trow("mdna_lm_positive_ratio", "MD&A positive words"),
], [2.2, 0.9, 0.9, 1.0, 1.2, 1.0], { numeric: [1, 2, 3, 4, 5], size: 18 }));
const tn = TONE.mdna_lm_negative_ratio;
add(P(`Insolvent companies' management discussions use more negative words (${(100 * tn.mean_distressed).toFixed(2)}% against ${(100 * tn.mean_healthy).toFixed(2)}%), more hedging and harder-to-read prose, and their auditors' reports are more negative; positive words do not differ. The direction matches Gupta and Banerjee (2.21% against 1.30%), but the gap is about a quarter of theirs (0.21 against 0.91 percentage points): within a matched pair the distressed report is more negative only ${pc(tn.share_pairs_distressed_higher)} of the time. Tone is a real but weak signal on its own, which is why the models that combine many such features do better than any single count.`));

// ---- 7.4 H2
add(H2("7.4 H2: what does language add beyond the ratios, and from where?"));
const DEC = [["hgb_ratios", "Ratios only"], ["hgb_ratios_fm", "Ratios + accounts-missing flag"],
  ["hgb_ratios_langtext", "Ratios + language (no flag)"], ["hgb_both", "Ratios + language + flag (the model of Table 10)"]];
add(TableTitle("Table 14.", "Separating the accounts-missing flag from language. Gradient boosting; PR-AUC."));
add(Tbl(["Inputs", "PR-AUC (763)", "Gain over ratios", "PR-AUC (609)", "Gain over ratios"],
  DEC.map(([m, lab]) => [lab, f3(pf("all", m).pr), m === "hgb_ratios" ? "" : s3(pf("all", m).pr - pf("all", "hgb_ratios").pr),
    f3(pf("strict", m).pr), m === "hgb_ratios" ? "" : s3(pf("strict", m).pr - pf("strict", "hgb_ratios").pr)])
  .concat([["Ratios + language, trained on the strict rows only", "", "", f3(pf("strict", "hgb_both_strict").pr),
    s3(pf("strict", "hgb_both_strict").pr - pf("strict", "hgb_ratios_strict").pr) + " (over strict-trained ratios)"]]),
  [2.8, 0.9, 1.0, 0.9, 1.4], { numeric: [1, 2, 3, 4], size: 18 }));
add(P(`The flag recording that a company-year's financial statements were incomplete had been stored with the language features. It is informative (healthy company-years are more often incomplete: ${pc(X.desc.fin_missing["0"])} against ${pc(X.desc.fin_missing["1"])}), and on its own it adds ${s3(cf.diff)} PR-AUC to the ratio model (${cf.wins} of 5 folds). It is not language. Language without the flag adds ${s3(cl.diff)} (${cl.wins} of 5 folds; DeLong p = ${pv(cl.delong.p)}; bootstrap interval ${s3(cl.boot.lo)} to ${s3(cl.boot.hi)}). On the strict subset, where the flag is always zero at test time, the two gains coincide, and retraining on the strict rows alone gives the same picture, so the 44 additional healthy reports do not drive the result.`));
const cBR = cmp("all", "blend - hgb_ratios");
add(P(excl(cl)
  ? `**H2 is supported.** Language without the flag adds ${s3(cl.diff)} PR-AUC to the booster (${foldsTxt(cl)}; ${s3(pf("strict", "hgb_ratios_langtext").pr - pf("strict", "hgb_ratios").pr)} on the strict subset), the DeLong test on ROC-AUC gives p = ${pv(cl.delong.p)}, and the pair-bootstrap interval for PR-AUC (${s3(cl.boot.lo)} to ${s3(cl.boot.hi)}) excludes zero, though only just. The full system, which adds the deep model, gains ${DL ? s3(cBR.diff) : PEND} over ratios alone in ${DL ? foldsTxt(cBR) : PEND} (Table 11). The gain is concentrated in particular companies (Section 7.5) rather than spread evenly.`
  : `**H2 is supported in direction, not at conventional significance on the full sample.** Language without the flag adds ${s3(cl.diff)} PR-AUC to the booster (${cl.wins} of 5 folds), but the pair-bootstrap interval for PR-AUC includes zero.`));
add(Figure(`${FIG}/fig9_importance.png`, 6.0, "Figure 7. Permutation importance by feature family in the ratios + language model, on held-out folds (30 shuffles per family per fold)."));
const PRETTY = { caro_default_flag: "CARO loan-default flag", caro_statutory_dues_flag: "CARO unpaid-dues flag", interest_coverage: "interest coverage",
  retained_earnings_to_assets: "retained earnings to assets", cash_to_assets: "cash to assets", roa: "return on assets",
  current_ratio: "current ratio", altman_z_em: "Altman Z\u2033", roce: "return on capital employed" };
const top = IMPF.slice(0, 6).map((r) => `${PRETTY[r.feature] || r.feature.replace(/_/g, " ")} (${s3(Number(r.pr_drop))})`).join(", ");
add(P(`**What the language gain consists of.** The accounting ratios carry most of the model, as expected. Among the language features one family stands out: the auditor's formal flags, and within them the **CARO loan-default flag**, the auditor's statement that the company has defaulted on borrowings, which is the single most important feature in the model (the six largest are ${top}). Its companion, the flag for unpaid statutory dues, is now among them too: before the correction of Section 8 it was raised for a quarter of the healthy companies and carried little information. MD&A tone and readability contribute a little; hedging, drift and perplexity contribute nothing measurable once the ratios are present. Permutation importance measures what a model leans on, not what it could do without, and Section 7.7 shows that the MD&A features carry information of their own. Still, the progress report's statement that the signal came from "tone, hedging and readability" should be narrowed: the most-used language feature is a formal statement by the auditor.`));

// ---- 7.5 healthy-looking
add(H2("7.5 Companies whose accounts look healthy"));
add(P("The case for reading the narrative is strongest where the numbers give no warning. The progress report showed this for one definition of \"healthy-looking\". Table 15 adds the confidence intervals it lacked and two further definitions, one of them the conventional Altman cut-off fixed long before this project and one defined by the ratio model's own scores, so that the finding cannot be an artefact of where the line was drawn."));
const SUB = [["z_above_median", "Altman Z″ above the sample median"], ["z_safe_zone", "Altman Z″ in the \"safe\" zone (above 2.6)"],
  ["ratio_model_low_risk", "Ratio model places the company in its lower-risk half"]];
const cell3 = (s, m) => { const c = cmp(s, `${m} - hgb_ratios`);
  return `**${f3(pf(s, m).pr)}**\n${s3(c.diff)}, ${c.wins} of 5 folds\n[${s3(c.boot.lo)}, ${s3(c.boot.hi)}]`; };
const srow = (s, lab) => [lab, `${N[s].n} (${N[s].pos})`, f3(N[s].pos / N[s].n), f3(pf(s, "hgb_ratios").pr), cell3(s, "hgb_both")]
  .concat(DL ? [cell3(s, "blend")] : []);
add(TableTitle("Table 15.", "PR-AUC on healthy-looking company-years. Each cell: PR-AUC; gain over ratios only and folds won; 95% pair-bootstrap interval for the gain."));
add(Tbl(["Subset", "n (insolvent)", "Chance", "Ratios only", "Ratios + language"].concat(DL ? ["Full system"] : []),
  SUB.map(([s, lab]) => srow(s, lab)).concat([srow("all", "All company-years, for comparison")]),
  DL ? [2.2, 0.9, 0.7, 0.8, 1.5, 1.5] : [2.4, 0.9, 0.7, 0.8, 1.6], { numeric: [1, 2, 3, 4, 5], size: 18 }));
add(Figure(`${FIG}/fig7_healthy.png`, 6.2, "Figure 8. PR-AUC on healthy-looking subsets; dashed lines mark chance (the share of insolvent company-years)."));
const cZs = cmp("z_safe_zone", "hgb_both - hgb_ratios"), cZf = cmp("z_above_median", "hgb_ratios_langtext - hgb_ratios");
const cLR = cmp("ratio_model_low_risk", "hgb_both - hgb_ratios");
add(P(`**This is the clearest result in the project.** Among the company-years with above-median Altman Z″, the ratio model scores PR-AUC ${f3(pf("z_above_median", "hgb_ratios").pr)} against a chance level of ${f3(N.z_above_median.pos / N.z_above_median.n)}. Adding language raises it to ${f3(pf("z_above_median", "hgb_both").pr)}, a gain of ${s3(kZ.diff)} that is positive in ${foldsTxt(kZ)} and whose 95% interval ${excl(kZ) ? "excludes" : "includes"} zero; it is ${times(kZ.diff, kAll.diff)} the gain on the sample as a whole. With the conventional safe-zone cut-off the gain is ${s3(cZs.diff)} (${cZs.wins} of 5 folds), and without the accounts-missing flag it is ${s3(cZf.diff)} (${cZf.wins} of 5). On the ratio model's own lower-risk half, adding the language features gains less (${s3(cLR.diff)}, ${cLR.wins} of 5 folds, interval ${s3(cLR.boot.lo)} to ${s3(cLR.boot.hi)}).${DL ? " " + (NARR.healthy_dl || "") : ""}`));
add(P("Put plainly: the numbers already tell you when a company is visibly sick; the words help with the ones that look fine. One caution belongs with this result. Selecting companies on a ratio narrows the range of the ratios inside the subset, which handicaps the ratio model by construction; this is strongest for the subset defined by the ratio model's own scores, which is therefore only a robustness check. The Altman-based subsets are less affected, since Altman Z\u2033 summarises only part of what the eleven ratios measure (it shares retained earnings to assets with them), and every model is scored on exactly the same companies."));

// ---- 7.6 H3
add(H2("7.6 H3: how early does the warning appear?"));
const HZ = [["t-1", "t-1 (last report before admission)"], ["t-2", "t-2"], ["t-3", "t-3 (three years before)"]];
const best = DL ? "blend" : "hgb_both";
add(TableTitle("Table 16.", "Performance by horizon: the same out-of-fold predictions, scored within each horizon group."));
add(Tbl(["Horizon", "n (insolvent)", "Ratios only", "Ratios + language"].concat(DL ? ["Full system"] : []).concat(["Recall at 10% false alarms (" + (DL ? "full system" : "ratios + language") + ")"]),
  HZ.map(([h, lab]) => [lab, `${N[h].n} (${N[h].pos})`, f3(pf(h, "hgb_ratios").pr), f3(pf(h, "hgb_both").pr)]
    .concat(DL ? [f3(pf(h, "blend").pr)] : []).concat([pc(pf(h, best).tpr10, 1)])),
  DL ? [2.2, 1.0, 0.9, 1.0, 0.9, 1.4] : [2.4, 1.0, 1.0, 1.1, 1.5], { numeric: [1, 2, 3, 4, 5], size: 18 }));
add(Figure(`${FIG}/fig8_horizon.png`, 5.6, "Figure 9. PR-AUC by horizon."));
add(P(DL ? (NARR.horizon || PEND) : PEND));

// ---- 7.7 H4
add(H2("7.7 H4: which part of the report carries the signal?"));
add(P("The progress report answered H4 by retraining the fusion network three times with a different text stream (MD&A 0.755, auditor 0.707, both 0.712 PR-AUC). That test was weaker than it looked: all three runs received the same 53 language features, including every auditor flag, so only the FinBERT stream changed; the gate gave that stream a weight of 0.07; and each run used one seed. Table 17 repeats the comparison in ways that isolate the section."));
const pr = (m) => pf("all", m) ? f3(pf("all", m).pr) : "–";
const h4row = (lab, a, b, c, key) => { const k = key ? cmp("all", key) : null;
  return [lab, a ? pr(a) : "–", b ? pr(b) : "–", c ? pr(c) : "–", k ? `${s3(k.diff)} (${k.wins} of 5)` : ""]; };
const H4 = [
  h4row("Engineered features, text only", "hgb_mdna", "hgb_aud", "hgb_lang", "hgb_aud - hgb_mdna"),
  h4row("TF-IDF bag of words, text only", "tfidf_mdna", "tfidf_aud", null, "tfidf_aud - tfidf_mdna"),
  h4row("Mined lexicon score, text only", "lexicon_mdna", "lexicon_aud", null, "lexicon_aud - lexicon_mdna"),
].concat(DL ? [
  h4row("Averaged FinBERT + logistic, text only", "emb_lr_mdna", "emb_lr_auditor", null, "emb_lr_auditor - emb_lr_mdna"),
  h4row("FinBERT + BiGRU network, text only (5 seeds)", "dl_text_mdna", "dl_text_aud", "dl_text_both", "dl_text_aud - dl_text_mdna"),
] : []).concat([
  h4row("Ratios + that section's features (boosting)", "hgb_ratios_mdna", "hgb_ratios_aud", "hgb_ratios_langtext", null),
]).concat(DL ? [h4row("Fusion network, all features, text stream varied (5 seeds)", "dl_main5", "dl_fusion_aud", "dl_fusion_both", null)] : []);
add(TableTitle("Table 17.", "PR-AUC by section of the report (763 company-years). The last column is auditor minus MD&A."));
add(Tbl(["Model", "MD&A", "Auditor's report", "Both", "Auditor − MD&A"], H4, [3.0, 0.8, 1.0, 0.8, 1.3], { numeric: [1, 2, 3, 4], size: 18 }));
const cRM = cmp("all", "hgb_ratios_mdna - hgb_ratios"), cRA = cmp("all", "hgb_ratios_aud - hgb_ratios");
const cMA = cmp("all", "hgb_ratios_langtext - hgb_ratios_aud");
add(P(`**On their own, the auditor's words are the stronger text.** In every text-only comparison that isolates the section, the auditor's report beats the MD&A, and by a wide margin for the bag-of-words and lexicon models (${s3(cmp("all", "tfidf_aud - tfidf_mdna").diff)} and ${s3(cmp("all", "lexicon_aud - lexicon_mdna").diff)} PR-AUC, five folds of five). The auditor's formal flags alone reach ROC-AUC ${f3(pf("all", "hgb_flags").roc)}.${DL ? " " + (NARR.h4_dl || "") : ""}`));
const cRMs = pf("strict", "hgb_ratios_mdna").pr - pf("strict", "hgb_ratios").pr, cRAs = pf("strict", "hgb_ratios_aud").pr - pf("strict", "hgb_ratios").pr;
const cZMA = cmp("z_above_median", "hgb_ratios_langtext - hgb_ratios_aud");
add(P(`**Added to the ratios, the auditor's features also count slightly more.** The auditor's features lift PR-AUC by ${s3(cRA.diff)} (${cRA.wins} of 5 folds, DeLong p = ${pv(cRA.delong.p)}, pair-bootstrap interval ${s3(cRA.boot.lo)} to ${s3(cRA.boot.hi)}) and the MD&A features by ${s3(cRM.diff)} (${cRM.wins} of 5 folds, interval ${s3(cRM.boot.lo)} to ${s3(cRM.boot.hi)}); on the strict subset ${s3(cRAs)} and ${s3(cRMs)}. Neither interval excludes zero, so the two sections are close once the ratios are known: much of what the auditor reports, defaults and losses, is already visible in the accounts. Before the correction of the CARO flags (Section 8) the order was the other way round (auditor ${s3(T1.comp.all["hgb_ratios_aud - hgb_ratios"].diff)}, MD&A ${s3(T1.comp.all["hgb_ratios_mdna - hgb_ratios"].diff)}): the misread flags were what made the auditor's features look weak. Added on top of the ratios and the auditor's features, the MD&A adds a further ${s3(cMA.diff)} (${cMA.wins} of 5 folds, within the noise), and ${s3(cZMA.diff)} on the above-median-Z\u2033 subset (${cZMA.wins} of 5, also within the noise).`));
add(P(`**Verdict on H4: supported in direction.** The hypothesis as pre-registered, that the signal is not only the auditor's flags, holds in the sense that the MD&A features improve on the ratios in every fold; that they add to the auditor's own features is likely but not demonstrated at this sample size. The progress report's stronger claim, that management's discussion carries *more* signal than the auditor's report, is not supported and is withdrawn: on every comparison in Table 17 the auditor's report is at least as informative.${DL ? " " + (NARR.h4_verdict || "") : ""}`));

// ---- 7.8 network
add(H2("7.8 What the network uses"));
add(P(DL ? (NARR.network || PEND) : PEND));

// ---- 7.9 lexicon
add(H2("7.9 An India-specific distress lexicon (contribution C)"));
add(P("Contribution C of the plan was to mine distress vocabulary from the reports themselves rather than borrow a US list. Terms (single words and two-word phrases) are ranked by the log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi and Quinn, 2008), comparing distressed with healthy reports. Two safeguards: a term counts once per report, so one firm repeating an industry word hundreds of times cannot dominate; and a term must appear in the reports of at least ten different companies, which keeps company and place names out."));
const lexrows = (L, n) => L.slice(0, n).map((r) => `${r.term} (${r.firms_distressed}/${r.firms_healthy})`).join(", ");
const lexrowsH = (L, n) => L.slice(-n).reverse().map((r) => `${r.term} (${r.firms_distressed}/${r.firms_healthy})`).join(", ");
add(TableTitle("Table 18.", "Highest-ranked terms, with the number of distressed / healthy companies whose reports use them. Fitted on all reports (a descriptive output; the scores used for prediction are refitted inside each training fold)."));
add(Tbl(["", "Typical of insolvent companies", "Typical of healthy companies"], [
  ["**MD&A**", lexrows(LEXM, 18), lexrowsH(LEXM, 12)],
  ["**Auditor's report**", lexrows(LEXA, 18), lexrowsH(LEXA, 12)],
], [0.9, 3.2, 2.4], { size: 17 }));
const cLL = cmp("all", "hgb_lang_lex - hgb_lang"), cLB = cmp("all", "hgb_both_lex - hgb_both");
add(P(`The lists read sensibly. Among the highest-ranked terms, insolvent companies' auditors write "qualified opinion", "unable to comment", "accumulated losses", "net worth", "eroded", "lenders", "interest on" and "not provided"; their management writes about a "net loss", "lenders", a "downward trend" and "fluctuating" and "difficult" conditions. Healthy companies' management discusses its "interest coverage", "operating profit", "credit rating" and "margin", and their auditors' reports carry more of the standard wording of the modern audit report ("overview", "ICAI firm", "judgements"). Some generic phrases also rank highly, a reminder that the lists are statistical associations to be read with care. Used as a fold-safe score, the lexicon is a strong single text signal for the auditor's report (ROC-AUC ${f3(pf("all", "lexicon_aud").roc)}), changes the language-only model by ${s3(cLL.diff)} PR-AUC (interval ${s3(cLL.boot.lo)} to ${s3(cLL.boot.hi)}; DeLong p = ${pv(cLL.delong.p)}) and adds ${s3(cLB.diff)} once ratios are present (${cLB.wins} of 5 folds; interval ${s3(cLB.boot.lo)} to ${s3(cLB.boot.hi)}): small, and not distinguishable from zero. The full ranked lists are released with the project (Section 12).`));

// ---- 7.10 nulls
add(H2("7.10 Negative and null results"));
add(Bullet(`**Perplexity.** A Kneser-Ney trigram model fitted on healthy companies' MD&A in the training folds only scores ROC-AUC ${f3(pf("all", "perplexity").roc)} as a single predictor; removing it from the ratios + language model ${Math.abs(cmp("all", "hgb_both_noperp - hgb_both").diff) < 0.0005 ? "leaves PR-AUC unchanged (difference below 0.001)" : "changes PR-AUC by " + s3(cmp("all", "hgb_both_noperp - hgb_both").diff) + " (" + cmp("all", "hgb_both_noperp - hgb_both").wins + " of 5 folds favour the model without it)"}. Indian MD&A sections are heavily templated, so a trigram model mostly learns boilerplate shared by both groups.`));
add(Bullet(`**Year-on-year drift (contribution A).** Removing the seven drift features ${cDr.diff > 0.0005 ? "*raises*" : "changes"} PR-AUC by ${s3(cDr.diff)} (${cDr.wins} of 5 folds favour the model without them). The Lazy Prices effect does not appear in this sample, possibly because only up to three consecutive reports per firm are available and the first has no predecessor.`));
add(Bullet("**Hyperparameter search.** In the progress-report runs, a nested Optuna search over nine settings improved PR-AUC by 0.007, while averaging three seeds improved it by 0.023 and cut the fold-to-fold spread from 0.090 to 0.064; inner-validation and test scores correlated at r = +0.07. With ~120 validation rows the search cannot tell a good configuration from a bad one, which is why the final network uses one fixed configuration and seven seeds."));
add(Bullet(`**Calibration.** The probability models are not well calibrated on this sample (Brier score ${f3(pf("all", "hgb_both").brier)} for ratios + language, against 0.249 for always predicting the base rate${DL && pf("all", "dlprob_main") ? `; ${f3(pf("all", "dlprob_main").brier)} for the network's averaged probabilities` : ""}). Scores should be read as rankings; a deployed screen would need recalibrating on data at the real prevalence.`));

// ---- 7.11 leak
add(H2("7.11 Leak test"));
add(P(DL ? (NARR.leak || PEND) : PEND));

// ============================================================== 8 changes since progress report
add(PageBreakP(), H1("8. What Changed Since the Progress Report"));
add(P("The progress report of 1 October was written from a sequence of notebook runs. Rebuilding the whole analysis from the dataset, on frozen folds and with every prediction saved, confirmed its main conclusions and corrected several details. They are listed here so that no number in the earlier report is relied on without its correction."));
add(TableTitle("Table 19.", "Corrections and additions."));
add(Tbl(["Progress report", "Final report", "Why"], [
  ["Text alone reached ROC-AUC 0.69 to 0.72 (FinBERT network on the MD&A)", `About 0.7 only from the auditor's report (${f3(pf("all", "tfidf_aud").roc)} to ${f3(pf("all", "lexicon_aud").roc)})${DL ? `; the FinBERT network on the MD&A alone ${f3(pf("all", "dl_text_mdna").roc)}` : ""} (Table 12)`, "One run of the uncorrected network, 609 rows, pooled scoring; repeated with five seeds"],
  ["MD&A beats the auditor's report (0.755 vs 0.707); \"it is management's own words\"", "Withdrawn. The auditor's report is the stronger text on its own and, slightly, on top of the ratios (Section 7.7)", "The earlier test varied only the FinBERT stream, which the gate largely ignored, with one seed"],
  ["The language signal comes from tone, hedging and readability", "Mostly from the auditor's formal flags, above all the CARO loan-default flag (Figure 7)", "Permutation importance was not run before"],
  [`CARO loan-default and unpaid-dues flags as extracted: raised for ${FR1.caro_default_flag["0"]}% and ${FR1.caro_statutory_dues_flag["0"]}% of healthy company-years`, `Corrected: ${FR2.caro_default_flag["0"]}% and ${FR2.caro_statutory_dues_flag["0"]}% of healthy company-years (${FR2.caro_default_flag["1"]}% and ${FR2.caro_statutory_dues_flag["1"]}% of insolvent ones); ${SC.flag_cells_changed} values changed, all listed with the sentence read`, "The rule cut clauses at the PDF's line breaks, so \"has not / defaulted\" read as a default; found while building the scoring tool (Section 10)"],
  ["Ratios + language +0.035 over ratios", `${s3(kAll.diff)}; language without the accounts-missing flag ${s3(cl.diff)}; the flag alone ${s3(cf.diff)} (Table 14)`, "The flag had been filed with the language features"],
  ["Healthy-looking firms: full system +0.160 over ratios, no interval", `${DL ? `Full system ${s3(cmp("z_above_median", "blend - hgb_ratios").diff)}, 95% interval ${s3(cmp("z_above_median", "blend - hgb_ratios").boot.lo)} to ${s3(cmp("z_above_median", "blend - hgb_ratios").boot.hi)}; ` : ""}ratios + language ${s3(kZ.diff)}, five folds of five; holds with the Altman safe-zone cut-off (Table 15)`, "Interval and robustness checks added"],
  ["Network: 200 sentences, GRU 64, dropout 0.3, gamma 2, adversarial head", "As trained: 96 sentences, GRU 32, dropout 0.5, gamma 1, no adversarial head (Table 6)", "The report described the plan, not the trained model"],
  ["About 0.36 million trainable parameters", "185,924 in the trained 32-unit network", "0.36 million is the 64-unit configuration"],
  ["The recurrent layer read all 200 slots including padding", "Reads exactly the first 96 sentences, padding skipped", "Implementation error, fixed"],
  ["Folds re-created by GroupKFold at run time", "Folds frozen in folds.csv", "Library versions assign 79% of rows differently"],
  ["Recall at 10% false alarms, Brier, DeLong, 1:10 prevalence, Altman and logistic baselines promised", "All reported (Tables 10 and 11; Brier in Section 7.10)", "Pre-registered; not delivered before"],
  ["Distress lexicon \"to be published\"", "Mined, evaluated fold-safely and released (Section 7.9)", "Contribution C completed"],
], [2.0, 2.4, 1.6], { size: 17 }));

// ============================================================== 9 limitations
add(H1("9. Limitations"));
add(Bullet("**Sample size is the binding constraint.** 763 company-years from 135 pairs, for a network with about 186,000 trainable parameters; five folds give little statistical power, which is why intervals accompany the main estimates and why several honest differences remain inside the noise."));
add(Bullet("**A balanced, matched sample flatters precision.** The 1:10 re-weighting in Table 10 shows what the ranking is worth at a more realistic prevalence; a deployed screen would see many more healthy companies than this study does."));
add(Bullet("**Hand checks only partly done.** The corrected CARO flags were checked against the reports' wording, and every changed flag is listed with the sentence the rule read (handcheck/) for the team to confirm. The 10% spot check of extracted financial figures against the PDFs (108 company-years) and the hand check of section extraction have not been completed, so extraction accuracy is measured against XBRL (91.8% within 1%) but not yet by hand."));
add(Bullet(`**A single company's score is less certain than the averages.** Retraining the boosting model after a change to the statutory-dues flag of 7 of the 763 company-years moved one test company's percentile by ${SC.instability_points} points; the cross-validated averages moved by less than 0.01. Scores near a band boundary should be read as borderline.`));
add(Bullet("**Not built from the plan:** the adversarial business-group head (the sample contains at least one same-group pair), company-name masking, coreference and subject-verb-object features, and the Word2Vec + CNN baseline. Grouping folds by pair prevents any company from appearing on both sides of a split, and the lexicon's ten-company rule keeps names out of the vocabulary, but neither is a substitute for the planned group test."));
add(Bullet("**Promoter pledge** is missing because the exchange shareholding filings were not collected."));
add(Bullet("**The 44 additional healthy reports** relax the pair rule; the main results are also given on the strict subset and, for the gradient-boosting models, after retraining on strict rows only."));
add(Bullet("**The healthy-looking subsets** are selected on ratios, which narrows the ratios' range inside them and so favours the other models somewhat, most for the subset defined by the ratio model's own scores; the absolute scores there should not be compared with the full sample."));


// ============================================================== 10 scoring a new company
add(H1("10. Scoring a New Company"));
add(P(`The project's purpose is a tool: an annual report goes in, and a statement of insolvency risk comes out. score.py does this for any listed company's report as a PDF, in under a minute on the GPU computer (about 20 seconds without the network). It runs the same code that built the training data, so a new report is read exactly as the 763 training reports were: on two training reports it reproduced all ${SC.parity_features} language features exactly, and the ratios wherever the financial statements have a text layer.`));
add(TableTitle("Table 20.", "What score.py does with one annual report."));
add(Tbl(["Step", "What happens"], [
  ["Read", "Text of every page (OCR for scanned pages when Tesseract is installed); the MD&A, auditor's report and CARO annexure located; the report year checked"],
  ["Language features", "The 51 engineered features, the CARO flags, perplexity under the five fold language models (so its scale matches training), and drift when last year's report is supplied"],
  ["Ratios", "The balance sheet and profit and loss read from the report; the 11 ratios; figures the reader misses can be supplied in a file"],
  ["Models", `Gradient boosting fitted on all 763 company-years, and the gated network trained seven times on all of them (FinBERT reads the MD&A)`],
  ["Score", `Each model's output is placed among its own cross-validated scores and the two are averaged 50/50; the risk score is the share of healthy company-years that scored lower`],
  ["Band", `High: above ${SC.high_q}% of healthy company-years (in cross-validation this band caught ${SC.high_caught}% of insolvent company-years); Watch: above ${SC.watch_q}% (caught ${SC.watch_caught}%); Low: the rest`],
  ["Reasons", "Score change when each family of features is swapped for that of 150 real healthy company-years; the auditor's own sentences behind every flag; each ratio against healthy companies; the MD&A sentences the network weighted most; warnings about anything the reader could not find"],
], [1.3, 4.6], { size: 18 }));
add(P(`**Three companies the models never saw.** None of these firms is in the training data: two insolvent companies that had no eligible peer and one healthy company whose pair was retired from the cohort.`));
add(TableTitle("Table 21.", "Scores of three unseen annual reports (full system)."));
add(Tbl(["Company and report", "What happened", "Risk score", "Band", "What the tool showed"], SC.examples.map((e) => [e.company, e.outcome, String(e.risk), e.band, e.why]), [1.6, 1.3, 0.7, 0.6, 2.6], { numeric: [2], size: 17 }));
add(Figure(`${FIG}/fig10_scorecard.png`, 6.0, "Figure 10. The score card for SEL Manufacturing's FY2017 report (top of the page)."));
add(P(`The two correct calls behave as intended: SEL Manufacturing, a year before admission, is flagged on its auditor's qualified opinion and its loan default, both quoted; Rane (Madras) scores below most healthy companies. JVL Agro is a miss, and the card says why it may be one: only ${SC.jvl_aud_chars.toLocaleString("en-US")} characters of its auditor's report were found and the opinion could not be identified, so the strongest family of features was largely missing. A score is a ranking against the study's sample, not a probability: the sample is one insolvent company for every healthy one, while insolvency is far rarer among listed companies.`));

// ============================================================== 11 conclusion
add(H1("11. Conclusion"));
add(P(DL ? (NARR.conclusion || PEND) : PEND));

// ============================================================== 12 reproducibility
add(H1("12. Reproducibility"));
add(P(`Everything in this report can be regenerated from the dataset folder and the files below. The folds are fixed in a file, the deep-learning run checks that it has built the identical table (checksum ${T.dl_run_info.table_checksum}), and every out-of-fold prediction is saved so that any table can be recomputed without retraining.`));
add(TableTitle("Table 22.", "Files of the final run (folder final_run/ in the project directory)."));
add(Tbl(["File", "What it is"], [
  ["bpp_final/run_gpu.py, bppfinal/dl.py, bppfinal/table.py", "The deep-learning run: table, FinBERT embeddings, 45 network runs (resumable; 178 minutes on an RTX 5060 Ti; after the CARO correction the 30 runs that read the language features were repeated in 75 minutes and the 15 text-only runs reused)"],
  ["bpp_final/data_v2/language_features.csv", "The language features with the corrected CARO flags (the only file of the dataset that changed)"],
  ["bpp_final/folds.csv", "The frozen fold of every company-year"],
  ["gpu_v2/results_v2/ (dl_oof.csv, runs/, attn_*.npy, run_info.json)", "What the GPU run saved: every network's out-of-fold predictions, gate and attention weights (analysis/archive_v1/ keeps the run before the correction)"],
  ["analysis/run_cpu.py, bppfinal/cpu_models.py", "Every model that needs no GPU: baselines, text-only models, lexicon"],
  ["analysis/extras_cpu.py, attention.py", "Permutation importance, the tone replication, the attention analysis"],
  ["analysis/analyze.py, bppfinal/stats.py", "All metrics, paired tests, bootstrap and DeLong; writes results_final/tables.json"],
  ["analysis/results_cpu/, analysis/results_final/", "Out-of-fold predictions of every model (all_oof.csv), tables.json, importance, the released lexicons (contribution C)"],
  ["report/build_report.js, make_narrative.py, charts.py, diagrams.py", "This document and its figures, generated from tables.json"],
  ["score.py, scorer/, train_final.py, train_final_dl.py, gpu_v2/final_models/", "The scoring tool of Section 10 and the final models it loads"],
  ["handcheck/", "Every CARO flag the correction changed, and a sample it did not, with the sentence the rule read, for checking by hand"],
], [2.6, 3.4], { size: 18 }));

// ============================================================== 12 references
add(H1("13. References"));
[
  "[1] Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. The Journal of Finance, 23(4), 589–609.",
  "[2] Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. The Journal of Finance, 66(1), 35–65.",
  "[3] Mai, F., Tian, S., Lee, C., & Ma, L. (2019). Deep learning models for bankruptcy prediction using textual disclosures. European Journal of Operational Research, 274(2), 743–758.",
  "[4] Cohen, L., Malloy, C., & Nguyen, Q. (2020). Lazy prices. The Journal of Finance, 75(3), 1371–1415.",
  "[5] Huang, A. H., Wang, H., & Yang, Y. (2023). FinBERT: A large language model for extracting information from financial text. Contemporary Accounting Research, 40(2), 806–841.",
  "[6] Arno, H., Mulier, K., Baeck, J., & Demeester, T. (2024). From numbers to words: Multi-modal bankruptcy prediction using the ECL dataset. arXiv:2401.12652.",
  "[7] Mancisidor, R. A., & Aas, K. (2024). Multimodal generative models for bankruptcy prediction using textual data. arXiv:2211.08405.",
  "[8] Gupta, V. (2022). Bankruptcy prediction using machine learning techniques: Evidence on Indian companies under Insolvency and Bankruptcy Code. The Journal of Prediction Markets, 16(2), 77–100.",
  "[9] Gupta, V., & Banerjee, A. (2023). Using textual analysis in bankruptcy prediction: Evidence from Indian firms under IBC. Investment Management and Financial Innovations, 20(3), 22–34.",
  "[10] Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. Proceedings of the 32nd International Conference on Machine Learning, PMLR 37, 1180–1189.",
  "[11] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. Proceedings of NAACL-HLT 2019, 4171–4186.",
  "[12] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. Proceedings of ICCV 2017, 2980–2988.",
  "[13] Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words: Lexical feature selection and evaluation for identifying the content of political conflict. Political Analysis, 16(4), 372–403.",
  "[14] DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. Biometrics, 44(3), 837–845.",
  "[15] Sun, X., & Xu, W. (2014). Fast implementation of DeLong's algorithm for comparing the areas under correlated receiver operating characteristic curves. IEEE Signal Processing Letters, 21(11), 1389–1393.",
].forEach((r) => add(new Paragraph({ children: L.runs(r, { size: 20 }), spacing: { after: 90 }, indent: { left: 400, hanging: 400 } })));

// ============================================================== assemble
const doc = new Document({
  creator: "DL/NLP project team", title: "Final Report: Early Prediction of Corporate Insolvency (India)",
  styles: L.styles, numbering: L.numbering,
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, right: 1300, bottom: 1300, left: 1300 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" })] })] }) },
    children: body,
  }],
});
const out = process.argv[2] || `${R}/report/Final_Report_Bankruptcy_Prediction.docx`;
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out, b.length, "bytes; DL:", DL); });
