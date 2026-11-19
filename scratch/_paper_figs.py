"""Figures for the IEEE paper, written to Research paper/latex/figs/ as PDF (vector)
plus PNG previews. Reads only saved predictions and numbers already printed by
scratch/_paper_extra.py and scratch/_paper_spoof.py - no model runs, no text.
    python -m scratch._paper_figs
"""
from pathlib import Path

import data   # before anything that might pull torch (Windows libiomp)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import precision_recall_curve

OUT = Path(data.__file__).resolve().parent / "Research paper" / "latex" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
PRED = data.DATA / "paper" / "preds"

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated pair)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def arch():
    fig, ax = plt.subplots(figsize=(7.16, 2.75))
    ax.set_xlim(0, 100), ax.set_ylim(0, 40), ax.axis("off")

    def box(x, y, w, h, text, fill="#ffffff", edge=INK2, bold=False, size=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                    fc=fill, ec=edge, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
                weight="bold" if bold else "normal", linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK2))

    tint = "#eaf2fc"
    rows = [  # y, input, content models, link / audio layer
        (30, "Email\n(Gmail / paste)", "DistilBERT 3-class\n+ persuasion head",
         "Mail link model (PSL split,\nbrand mismatch) + Safe\nBrowsing + attachments"),
        (17, "SMS\n(Android / paste)", "char 3-5-gram\nTF-IDF + LR",
         "URL XGBoost\n(15 domain-lexical\nfeatures)"),
        (4, "Call audio\n(upload)", "Whisper small ->\nrules + act model\n(EMA ratchet)",
         "wav2vec2 XLS-R\nspoof head\n(one-sided)"),
    ]
    for y, inp, nlp, side in rows:
        box(0.5, y, 13, 8, inp, bold=True)
        box(18, y, 17, 8, nlp)
        box(39, y, 19, 8, side, size=6.5)
        arrow(13.8, y + 4, 17.7, y + 4)
        arrow(35.3, y + 4, 38.7, y + 4)
        arrow(58.3, y + 4, 66.7, 20)
    box(62, 0.5, 13, 38.5, "", fill=tint, edge=BLUE)
    ax.text(68.5, 33.5, "Behaviour\nengine", ha="center", va="center", fontsize=7.5, weight="bold", color=INK)
    ax.text(68.5, 21, "per-sender /\nper-caller\nprofiles\n(Postgres)\n\nSPF / DKIM /\nDMARC, repeat-\nsender trust, crowd\nreports, DLT\nregistry,\ncaller number",
            ha="center", va="center", fontsize=6.3, color=INK)
    box(79, 13, 10, 14, "Fusion\nadaptive\nweighted sum\n+ gates", fill=tint, edge=BLUE, size=6.8)
    arrow(75.3, 20, 78.7, 20)
    box(92, 13, 7.5, 14, "SAFE\n<0.35\n\nSUSP.\n<0.70\n\nPHISH.", size=6.3)
    arrow(89.3, 20, 91.7, 20)
    ax.text(26.5, 39.6, "content models", ha="center", fontsize=6.5, color=INK2, style="italic")
    ax.text(48.5, 39.6, "link / audio layer", ha="center", fontsize=6.5, color=INK2, style="italic")
    ax.text(95.8, 10.5, "+ reason\nstrings", ha="center", va="top", fontsize=6.3, color=INK2)
    ax.text(26.5, 15.6, "SMS path also runs on the phone (Kotlin port)", ha="center", va="top",
            fontsize=5.8, color=INK2, style="italic")
    save(fig, "arch")


def url_pr():
    u = np.load(PRED / "url.npz")
    y = u["y"]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for s, colour, ls, name in [(u["p"], BLUE, "-", "XGBoost (shipped)"),
                                (u["q"], ORANGE, "--", "Logistic regression, same features")]:
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=colour, ls=ls, lw=1.6, label=name)
    ax.axhline(y.mean(), color=INK2, lw=0.8, ls=":")
    ax.text(0.02, y.mean() + 0.025, f"all-benign / chance = base rate {y.mean():.1%}", fontsize=6.5, color=INK2)
    rec, prec = np.mean(u["p"][y == 1] >= .5), np.mean(y[u["p"] >= .5] == 1)
    ax.plot([rec], [prec], "o", ms=5, mfc="white", mec=BLUE, mew=1.4)
    ax.annotate("shipped cut-off 0.5", (rec, prec), (rec - 0.42, prec + 0.22), fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=INK2))
    ax.set_xlabel("Recall (phishing links caught)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01), ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, lw=0.5), ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    save(fig, "url_pr")


def spoof():
    # scratch/_paper_spoof.py, 2026-10-28 (EER %, lower is better)
    conds = ["In-the-Wild, clean", "In-the-Wild, phone", "ASVspoof 21 LA, pooled",
             "ASVspoof 21 LA, PSTN", "ASVspoof 5, eval"]
    shipped = [13.31, 16.27, 5.82, 8.13, 14.23]
    prev = [13.55, 17.94, 5.36, 6.50, 9.48]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    yy = np.arange(len(conds))[::-1]
    h = 0.36
    ax.barh(yy + h / 2 + 0.02, shipped, h, color=BLUE, label="shipped (+ modern TTS, real phone calls)")
    ax.barh(yy - h / 2 - 0.02, prev, h, color="white", edgecolor=ORANGE, hatch="////", lw=0.9,
            label="previous (ASVspoof 2019 + 5)")
    for y0, a, b in zip(yy, shipped, prev):
        ax.text(a + 0.3, y0 + h / 2 + 0.02, f"{a:.1f}", va="center", fontsize=6.3, color=INK)
        ax.text(b + 0.3, y0 - h / 2 - 0.02, f"{b:.1f}", va="center", fontsize=6.3, color=INK2)
    ax.set_yticks(yy, conds)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Equal error rate (%) - lower is better")
    ax.set_xlim(0, 20)
    ax.grid(axis="x", color=GRID, lw=0.5), ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.3, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1)
    save(fig, "spoof_eer")


if __name__ == "__main__":
    arch()
    url_pr()
    spoof()
    print("figures ->", OUT)
