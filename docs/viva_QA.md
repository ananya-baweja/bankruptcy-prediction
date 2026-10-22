# Viva questions and answers: Green AI

Prepared answers for the Green AI part of the viva (final report, Section 11). The full Q&A for the
rest of the project is the PDF `Viva_QA_Insolvency_Early_Warning.pdf`, whose last two pages repeat
these answers. Energy is GPU energy read from the NVIDIA driver (NVML); CO₂e is computed from it with
codecarbon's India grid factor (0.713 kg per kWh). Accuracy is on the corrected data of Section 7
unless marked "full study". Sources: `results/green/` and `results/green/green_on_corrected_data.csv`.

## One-line answers

| Question | Answer |
| --- | --- |
| What did you do for Green AI? | Measured the energy of every stage for 82 configurations, chose the cheapest one statistically level with the full system, and made the NLP notebook skip work nothing reads. |
| Where does the energy go? | Training the seven networks: 7.1 of 8.7 Wh (81%). FinBERT embedding is 1.5 Wh. |
| What is the green choice? | Gradient boosting on ratios + language, no network: PR-AUC 0.795 against 0.807, about 42 times less GPU energy and CO₂e. |
| Is it really as good? | Statistically level, not identical: −0.012, interval −0.031 to +0.009, ahead in 3 of 5 folds (Wilcoxon p = 0.81). |
| And if you keep the network? | Fewer seeds: 3 instead of 7 gives 0.805 for 51% less energy; 5 seeds 0.808 for 24% less. |
| What changed in the code? | Notebook 1 embeds only the 96 sentences per section the network reads (fp16, cached) and logs its energy. The deep-learning notebook and all results are unchanged. |
| Did a smaller encoder help? | Accuracy yes (MiniLM-L6, 5 times smaller, matches FinBERT in the full study); energy little, because the encoder is a small part of the bill. |
| fp16 or fp32? | fp16: the same vectors (cosine above 0.9999) for 4.4 times less embedding energy. |
| Did INT8 help? | Inconclusive: accuracy held (0.793, full study) but the processor timings were taken on a busy machine. |

## In more detail

**How did you measure energy, and why only GPU energy?** codecarbon (offline, India grid) wrapped
every stage. Its own logs showed the processor 64-68% busy even in stages that only use the GPU, so
unrelated background processes were running on the GPU computer throughout, which inflates its
processor estimate (rated power × utilisation) by an unknown amount. The GPU reading comes from the
driver and is unaffected; integrating nvidia-smi power once a second agrees within 3.4%. Whole-machine
totals are kept only as upper bounds (report Table 24).

**Why are the full study's accuracies lower than the rest of the report?** The 82-configuration study
ran on the data before the CARO-flag correction (its blend scores 0.792, not 0.807). Every configuration
saw the same data, so comparisons between them hold. The configurations the conclusion depends on were
rescored on the corrected data from the saved predictions (report Table 22); their energy does not
depend on the flags.

**Was the selection rule fixed in advance?** Yes: the lowest-energy configuration within 0.010 PR-AUC
of the current system, or whose bootstrap interval against it includes zero. One change was made
afterwards: energies within 5% are a tie broken by accuracy, because gradient boosting on ratios alone
drew 4 mWh less idle power. On the corrected data it does not matter: ratios alone fail the rule
(interval −0.111 to −0.009).

**Why is the green model's energy 0.2 Wh and not zero?** It is the GPU idling at 6-7 W while the
processor fits the model for about 106 s: read it as "no GPU work". It scores a report in about 2 ms,
against 213 ms and 10 J of GPU energy for the current scorer.

**Why does the 96-sentence cap not change the deep model?** The network reads exactly the first 96
sentence slots of each section, so the other slots were embedded and never used (24,522 of 77,523 MD&A
sentences in the study). Notebook 1 keeps the 200-slot arrays, with the unused slots left at zero, so
notebook 2 runs unchanged.

**Limitations?** One computer, one run per configuration; a single network run's energy varies by up
to 1.8 times between seeds because of early stopping. GPU energy leaves out the processor and memory.
PDF reading and feature extraction were not measured, so the true saving per report is smaller.
