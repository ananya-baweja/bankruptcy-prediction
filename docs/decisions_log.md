# Decisions log

Every choice that changes the data or results, with a date and reason. Examiners ask "why?"; this file answers.
Newest first. Add a row whenever you change `configs/config.yaml` or a rule in the code.

| Date | Decision | Reason | Changed by |
| --- | --- | --- | --- |
| 2026-09-16 | Admission date proxy = earliest IBBI CIRP public-announcement date; replace with verified insolvency commencement date where checked | IRP must publish Form A within ~3 days of appointment; IBBI list is complete and public | Claude (setup) |
| 2026-09-16 | Name matching: auto-accept only at fuzzy score ≥ 98; 80–97 goes to human review | In testing, "Tarang Infra" vs "Tarangi Infra" scored 96; a wrong match puts a healthy firm in the distressed class | Claude (setup) |
| 2026-09-16 | Any firm appearing in the CIRP match list (auto or needs-review, unless reviewed N) is excluded from healthy peers | Conservative: a possibly-distressed firm must never be labelled healthy | Claude (setup) |
| 2026-09-16 | Peers: same industry code (fallback first 2 digits), total assets ±30% in the last FY before admission, closest by log assets, each peer used once, scarcest firms matched first | Standard matched-sample design (industry + size), as in the Business Perspectives (2023) IBC study | Claude (setup) |
| 2026-09-16 | Collect 4 FYs per firm, keep 3 (t-1..t-3) | The most recent report is often excluded by the leakage rule | Claude (setup) |
| 2026-09-16 | Exclude reports published after the petition date, or within 180 days before admission when the petition date is unknown | Such reports can openly discuss the insolvency case (label leakage) | Claude (setup) |
| 2026-09-16 | Keep pairs aligned: drop the peer's FY when the distressed firm's FY is excluded or missing | Prevents the model from learning calendar year as a shortcut | Claude (setup) |
| 2026-09-16 | Unknown publication date = FY end + 183 days, flagged `assumed` | AGM must be held within 6 months of FY end (Companies Act, 2013) | Claude (setup) |
| 2026-09-16 | Standalone auditor's report preferred over consolidated | CARO applies to standalone statements; consolidated reports repeat subsidiaries' matters | Claude (setup) |
| 2026-09-16 | Section extraction is rule-based (regex headings + run logic), accuracy hand-checked on 30 reports | Explainable, fast, no training data needed; accuracy is reportable | Claude (setup) |
| 2026-09-16 | Laptops (VS Code) for Phases 0–3; Colab GPU for Phases 4 and 6; GitHub for code; Google Drive for data | Exchange sites block cloud IPs; no local NVIDIA GPU | Team + Claude |
