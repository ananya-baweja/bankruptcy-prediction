"""Phase 3 - financial features read from the annual reports.

CMIE Prowess is not available to this project, so the figures come from the
standalone financial statements in the PDFs Phase 2 collected. MCA21 is not used.

Modules
-------
``numbers``     Indian number and unit parsing: crore/lakh headers, bracketed
                negatives, Indian and Western digit grouping, OCR damage.
                Everything is stored in ₹ crore.
``statements``  Locating the standalone balance sheet, statement of profit and
                loss and cash flow statement in a report, and mapping their line
                items to the standard fields.
``ratios``      The 12 stream-C ratios of Table 7: liquidity (current, quick,
                cash/TA), profitability (EBITDA margin, ROCE, ROA), leverage
                (D/E, debt/EBITDA), solvency (interest coverage, RE/TA),
                promoter pledge and Altman Z'' (emerging markets).
``financials``  The pipeline: extract, apply the as-first-published rule and the
                gap-filling order, validate, score confidence, compute ratios,
                and write the spot-check sheet.

Run it with ``bpp financials``. See ``docs/06_phase3_financials.md``.

Ratios are stored raw. Winsorising at the 1st/99th percentile and scaling are
fitted inside the cross-validation folds (``ratios.winsorise_bounds``), because
bounds learnt over the whole table leak the test fold's distribution into
training. See ``docs/decisions_log.md``.
"""
