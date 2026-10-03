"""Diagrams for the final report, in the style of the progress report's figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = dict(green=("#2E7D4F", "#E6F2E8"), blue=("#2E5C9E", "#E7EDF8"), orange=("#C0622B", "#FCECE1"),
         purple=("#6A4FA3", "#EFE9F7"), grey=("#888888", "#F1F1F1"), red=("#B53F3F", "#FBEAEA"))
from pathlib import Path
OUT = str(Path(__file__).resolve().parent / "fig")


def canvas(w, h, title, sub):
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.text(0.18, h - 0.3, title, fontsize=17, fontweight="bold", va="top", color="#111")
    ax.text(0.18, h - 0.68, sub, fontsize=12.5, va="top", color="#666")
    return fig, ax


def box(ax, x, y, w, h, title, body="", col="blue", dashed=False, tsize=13, bsize=10.5):
    ec, fc = C[col]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", lw=2.0,
                                ec=ec, fc=fc, ls="--" if dashed else "-"))
    if body:
        ax.text(x + w / 2, y + h / 2 + 0.13, title, ha="center", va="bottom", fontsize=tsize, fontweight="bold",
                color="#151515", wrap=True)
        ax.text(x + w / 2, y + h / 2 + 0.07, body, ha="center", va="top", fontsize=bsize, color="#555",
                linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color="#151515")


def arrow(ax, x1, y1, x2, y2, rad=0.0, col="#666", dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=2.2, color=col,
                                 connectionstyle=f"arc3,rad={rad}", ls="--" if dashed else "-"))


def note(ax, x, y, s, size=10.5, col="#888", ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=col, linespacing=1.35)


# ---------------------------------------------------------------- 1 overview
fig, ax = canvas(11.6, 5.3, "System overview",
                 "Two kinds of evidence about the same company-year, read by two model families and averaged.")
box(ax, 0.2, 1.75, 1.75, 1.45, "Companies", "135 insolvent and\n135 matched healthy\nlisted firms", "green")
box(ax, 2.25, 1.75, 1.8, 1.45, "Annual reports", "763 company-years,\n1-3 years before\nadmission", "green")
box(ax, 4.45, 2.75, 2.35, 1.25, "What they say", "MD&A and auditor's report:\nFinBERT sentences and\n51 engineered features", "blue")
box(ax, 4.45, 0.85, 2.35, 1.25, "What they report", "11 accounting ratios,\nincl. Altman Z''", "blue")
box(ax, 7.2, 2.75, 2.05, 1.25, "Gated network", "FinBERT + BiGRU +\nattention + gate", "orange")
box(ax, 7.2, 0.85, 2.05, 1.25, "Gradient boosting", "ratios + language\nfeatures", "orange")
box(ax, 9.65, 1.75, 1.75, 1.45, "Risk score", "50/50 rank blend,\nfixed in advance", "purple")
arrow(ax, 1.97, 2.47, 2.22, 2.47)
arrow(ax, 4.07, 2.7, 4.42, 3.3, rad=-0.2); arrow(ax, 4.07, 2.25, 4.42, 1.5, rad=0.2)
arrow(ax, 6.82, 3.37, 7.17, 3.37); arrow(ax, 6.82, 1.47, 7.17, 1.47)
arrow(ax, 9.27, 3.37, 9.62, 2.7, rad=-0.2); arrow(ax, 9.27, 1.47, 9.62, 2.2, rad=0.2)
note(ax, 8.22, 2.42, "each model sees both inputs", 10)
note(ax, 5.62, 0.45, "NLP: text to features (Part 1)", 10.5)
note(ax, 8.22, 0.45, "Deep learning and evaluation (Part 2)", 10.5)
fig.savefig(f"{OUT}/fig1_overview.png"); plt.close(fig)

# ---------------------------------------------------------------- 2 sample
fig, ax = canvas(11.6, 5.3, "Building the sample",
                 "Insolvent companies come from public IBBI records; each is matched to a healthy peer.")
box(ax, 0.2, 2.75, 2.55, 1.3, "IBBI announcements", "9,147 insolvency (CIRP)\nannouncements, 2017-2026", "green")
box(ax, 3.1, 2.75, 2.55, 1.3, "Listed companies", "606 debtors with a listed\ncompany CIN", "blue")
box(ax, 6.0, 2.75, 2.55, 1.3, "Eligible and matched", "listed long enough to have\nreports; peer found", "blue")
box(ax, 8.9, 2.75, 2.5, 1.3, "Identity confirmed", "all 135 by the CIN printed\nin their own reports", "orange")
box(ax, 8.9, 0.55, 2.5, 1.45, "Healthy peers", "same BSE industry sub-group,\ntotal assets within 30%,\nnever insolvent", "blue")
box(ax, 5.1, 0.55, 3.3, 1.45, "Company-years", "135 pairs x 4 years = 1,080;\n1,154 reports read\n(148,424 pages, 7,490 by OCR)", "purple")
box(ax, 0.2, 0.55, 4.4, 1.45, "Modelling table", "763 company-years (362 distressed, 401 healthy)\nafter the leakage rules; strict matched subset 609", "green")
arrow(ax, 2.77, 3.4, 3.07, 3.4); arrow(ax, 5.67, 3.4, 5.97, 3.4); arrow(ax, 8.57, 3.4, 8.87, 3.4)
arrow(ax, 10.15, 2.72, 10.15, 2.03); arrow(ax, 8.87, 1.27, 8.43, 1.27); arrow(ax, 5.07, 1.27, 4.63, 1.27)
fig.savefig(f"{OUT}/fig2_sample.png"); plt.close(fig)

# ---------------------------------------------------------------- 4 documents
fig, ax = canvas(11.6, 5.4, "From annual report to model inputs",
                 "Each report is split into sections; two of them feed the models, in two forms.")
box(ax, 0.2, 2.95, 1.75, 1.25, "Annual report", "PDF, ~129 pages\non average", "green")
box(ax, 2.3, 2.95, 1.95, 1.25, "Read the text", "PDF text layer;\nOCR when scanned", "blue")
box(ax, 4.6, 2.95, 2.0, 1.25, "Find sections", "heading rules;\nCARO rule fixed\non real reports", "blue")
box(ax, 7.0, 3.55, 4.4, 0.62, "MD&A - management's account of the year", "", "purple", tsize=11)
box(ax, 7.0, 2.85, 4.4, 0.62, "Auditor's report - opinion, going concern", "", "purple", tsize=11)
box(ax, 7.0, 2.15, 4.4, 0.62, "CARO annexure - loan defaults, unpaid dues", "", "purple", tsize=11)
arrow(ax, 1.97, 3.57, 2.27, 3.57); arrow(ax, 4.27, 3.57, 4.57, 3.57); arrow(ax, 6.62, 3.57, 6.97, 3.3)
box(ax, 0.6, 0.35, 4.9, 1.35, "Raw text -> FinBERT",
    "sentences of 5+ words, first 96 per section,\neach a 768-value vector (encoder frozen)", "orange")
box(ax, 6.1, 0.35, 4.9, 1.35, "Counts -> 51 features + perplexity",
    "tone (Loughran-McDonald), hedging, readability,\nauditor and CARO flags, year-on-year drift", "orange")
arrow(ax, 7.6, 2.12, 3.6, 1.73, rad=0.15); arrow(ax, 9.2, 2.12, 8.6, 1.73)
note(ax, 5.8, 0.12, "Raw text keeps every word: removing \"may\" and \"could\" would delete the hedging being measured.", 10)
fig.savefig(f"{OUT}/fig4_documents.png"); plt.close(fig)

# ---------------------------------------------------------------- 5 architecture as trained
fig, ax = canvas(11.6, 6.4, "The deep model as trained",
                 "Three inputs per company-year; a softmax gate learns how much weight each block receives.")
box(ax, 0.2, 3.85, 2.4, 1.0, "A. Report sentences", "first 96 MD&A sentences", "green", bsize=10)
box(ax, 0.2, 2.15, 2.4, 1.0, "B. Language features", "51 features + perplexity\n+ accounts-missing flag", "green", bsize=10)
box(ax, 0.2, 0.55, 2.4, 1.0, "C. Accounting ratios", "11 values", "green", bsize=10)
box(ax, 3.0, 4.35, 2.3, 0.75, "FinBERT (frozen)", "", "grey", tsize=11.5)
box(ax, 3.0, 3.2, 2.3, 1.0, "BiGRU + attention", "96x768 -> 96x64 -> 64", "orange", bsize=10)
box(ax, 3.0, 2.15, 2.3, 1.0, "Dense network", "53 -> 64 -> 64", "orange", bsize=10)
box(ax, 3.0, 0.55, 2.3, 1.0, "Dense network", "11 -> 64 -> 64, BatchNorm", "orange", bsize=10)
box(ax, 5.75, 1.6, 1.9, 2.3, "Gate", "softmax over the\nthree blocks:\none weight each", "orange", bsize=10)
box(ax, 8.05, 2.15, 1.6, 1.2, "Fused", "3 x 64 = 192", "blue", bsize=10)
box(ax, 10.0, 2.15, 1.4, 1.2, "Risk", "192 -> 64 -> 1", "purple", bsize=10)
arrow(ax, 2.62, 4.4, 2.97, 4.65); arrow(ax, 4.15, 4.33, 4.15, 4.22)
arrow(ax, 2.62, 2.65, 2.97, 2.65); arrow(ax, 2.62, 1.05, 2.97, 1.05)
arrow(ax, 5.32, 3.7, 5.72, 3.5); arrow(ax, 5.32, 2.65, 5.72, 2.65); arrow(ax, 5.32, 1.05, 5.72, 1.8)
arrow(ax, 7.67, 2.75, 8.02, 2.75); arrow(ax, 9.67, 2.75, 9.97, 2.75)
note(ax, 9.0, 1.15, "dropout 0.5, focal loss (gamma 1) + 0.1 x cross-entropy,\nAdamW 1e-3, batch 16, early stopping on an inner\npair-grouped split; 186,000 trainable parameters", 9.5, "#555")
note(ax, 9.0, 4.55, "Not built: the adversarial business-group head\n(deferred; see Limitations)", 9.5, "#9b3b3b")
fig.savefig(f"{OUT}/fig5_architecture.png"); plt.close(fig)
print("diagrams written")
