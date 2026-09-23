"""Bankruptcy Prediction Project (bpp).

Multimodal prediction of NCLT admission under India's Insolvency and Bankruptcy
Code (IBC, 2016) from annual-report narratives plus accounting ratios.

Sub-packages
------------
scrape    Phase 1-2  IBBI announcements, listed-company lists, annual report download
cohort    Phase 1-2  name matching, healthy-peer matching, firm-year frame, labels
extract   Phase 2    PDF text (+OCR), section segmentation, extraction QA
features  Phase 3    financial ratios                      (not built yet)
nlp       Phase 4    linguistic engine                     (not built yet)
models    Phase 5-6  baselines and the gated fusion network (not built yet)
eval      Phase 7-8  experiments, statistics, explainability (not built yet)
"""

__version__ = "0.1.0"
