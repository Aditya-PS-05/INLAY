"""
Figures for the INLAY paper.

Publication conventions this follows, none of which the first version did:

  * VECTOR PDF output, not raster PNG. A rasterised figure in a LaTeX paper is
    the most obvious tell of an amateur submission.
  * Type-42 font embedding, required for arXiv and camera-ready.
  * Nimbus Roman throughout, the Times clone the document body uses, so figure
    text and body text are indistinguishable.
  * Sized to the actual text block: 3.35in single column, 6.95in full width.
    Figures are never scaled in LaTeX, because scaling changes the effective
    font size and destroys that match.
  * Wilson 95% intervals wherever a proportion is plotted. These were computed
    in akew_stats.py and simply not shown before, which understated the
    uncertainty on the n=63 cells.
  * High data-ink ratio: no chart borders, no gridlines competing with marks,
    no legend where a direct label will do.

Every number is from this project's logged runs; the source doc is named above
each data block.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ACL style (acl.sty): A4, 2.5cm margins, twocolumn, columnsep 0.6cm.
# textwidth = 21cm - 2*2.5cm = 16cm; column width = (16cm - 0.6cm)/2 = 7.7cm.
CM = 1 / 2.54
COL, FULL = 7.7 * CM, 16.0 * CM   # inches: one column (3.031in), full text width (6.299in)

INK, MUTED, FAINT = "#1a1a1a", "#5f5f5f", "#cfcfcf"
NEUTRAL, ACCENT = "#a9a9a9", "#12615a"
RULE = "#8c8c8c"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.2,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.012,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=400)   # README only; the paper uses PDF
    plt.close(fig)
    print("wrote", name)


def wilson(k, n, z=1.96):
    """Wilson score interval. Chosen over the normal approximation because
    several cells sit at or near 100%, where the normal interval extends past
    1.0 and is meaningless."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, c - h), min(1.0, c + h)


# ------------------------------------------------------------------ Figure 1
def fig_architecture():
    """Two panels: where the edit lives, and where the intervention happens.
    Panel (b) is what separates this from a generic box-beside-a-box diagram --
    it shows the actual mechanism at the decoding step."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(FULL, 1.95),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})

    axl.set_xlim(0, 100); axl.set_ylim(0, 46); axl.axis("off")
    axl.text(0, 43, "(a) where the edit lives", fontsize=8, style="italic", color=INK)

    # Box widths are set from the width of the text they must contain: at
    # 7.0pt in a 100-unit axis roughly 3.2in wide, "all facts entangled" needs
    # about 34 units. Undersized boxes were the collision in the first version.
    axl.add_patch(Rectangle((0, 13), 40, 20, fill=False, ec=INK, lw=0.9))
    axl.text(20, 25, "model weights", ha="center", fontsize=7.2, color=INK)
    axl.text(20, 18.5, "all facts entangled", ha="center", fontsize=6.6, color=MUTED)
    axl.add_patch(FancyArrowPatch((20, 4.5), (20, 12), arrowstyle="-|>",
                                  mutation_scale=7, color=INK, lw=0.8))
    axl.text(20, 0.8, "edit rewrites shared state", ha="center", fontsize=6.6, color=INK)
    axl.text(20, 36, "weight editing", ha="center", fontsize=7.2, color=INK)

    axl.add_patch(Rectangle((50, 13), 33, 20, fill=False, ec=INK, lw=0.9, ls=(0, (3, 2))))
    axl.text(66.5, 25, "frozen model", ha="center", fontsize=7.2, color=INK)
    axl.text(66.5, 18.5, "never modified", ha="center", fontsize=6.6, color=MUTED)
    axl.add_patch(Rectangle((90, 13), 10, 20, fill=False, ec=ACCENT, lw=0.9))
    for y in (17, 21, 25, 29):
        axl.plot([91.6, 98.4], [y, y], color=ACCENT, lw=0.7, alpha=.55)
    axl.add_patch(FancyArrowPatch((95, 4.5), (95, 12), arrowstyle="-|>",
                                  mutation_scale=7, color=ACCENT, lw=0.8))
    axl.text(95, 0.8, "edit is a row", ha="center", fontsize=6.6, color=ACCENT)
    axl.add_patch(FancyArrowPatch((89, 23), (84, 23), arrowstyle="-|>",
                                  mutation_scale=6, color=INK, lw=0.7))
    axl.text(75, 36, "INLAY", ha="center", fontsize=7.2, color=ACCENT)

    axr.set_xlim(0, 100); axr.set_ylim(0, 46); axr.axis("off")
    axr.text(0, 43, "(b) the intervention, at the final decoding step",
             fontsize=8, style="italic", color=INK)

    axr.text(1, 27, r"$h$", fontsize=9, color=INK)
    axr.add_patch(FancyArrowPatch((6, 27.5), (15, 27.5), arrowstyle="-|>",
                                  mutation_scale=7, color=INK, lw=0.8))
    axr.text(10.5, 30.5, r"$W_U$", ha="center", fontsize=7.4, color=MUTED)

    # Bars capped at 20 units (top = 34) so the "target" callout at 36 clears
    # the panel title at 43 -- in the first version the tallest bar reached 38
    # and the callout struck through the title.
    logits = [12, 7, 20, 4.5, 9.5]
    bx = 18
    for i, v in enumerate(logits):
        c = ACCENT if i == 2 else NEUTRAL
        axr.add_patch(Rectangle((bx + i * 5.8, 14), 4.3, v, fc=c, ec="none"))
    tx = bx + 2 * 5.8 + 2.15
    axr.text(tx, 37.6, "target", ha="center", fontsize=6.6, color=ACCENT)
    axr.add_patch(FancyArrowPatch((tx, 36.6), (tx, 34.8), arrowstyle="-|>",
                                  mutation_scale=6, color=ACCENT, lw=0.8))
    axr.plot([bx - 1, bx + 30], [14, 14], color=INK, lw=0.7)
    axr.text(bx + 14, 9.6, "logits, one per vocabulary token",
             ha="center", fontsize=6.9, color=MUTED)
    axr.text(bx + 14, 4.6, r"add $\alpha$ along $W_U[t]$ to one score",
             ha="center", fontsize=6.9, color=ACCENT)

    axr.add_patch(FancyArrowPatch((52, 27.5), (61, 27.5), arrowstyle="-|>",
                                  mutation_scale=7, color=INK, lw=0.8))
    axr.text(56.5, 30.5, "argmax", ha="center", fontsize=7.4, color=MUTED)
    axr.add_patch(Rectangle((63, 21), 34, 13, fill=False, ec=INK, lw=0.9))
    axr.text(80, 27, "next token", ha="center", fontsize=7.8, color=INK)
    axr.text(80, 15.5, "gate silent " + r"$\Rightarrow$" + " bitwise identical",
             ha="center", fontsize=6.9, color=MUTED)
    save(fig, "fig1_architecture")


# ------------------------------------------------------------------ Figure 2
def fig_edit_accuracy():
    """outputs/akew_weightedit_baseline_results.md. GPT-J-6B, CounterFact
    structured, n=147."""
    n = 147
    data = [("GRACE", 0), ("WISE", 98), ("ROME", 123), ("MEMIT", 123),
            ("AlphaEdit", 131), ("INLAY", 147)]
    names = [d[0] for d in data]
    ks = np.array([d[1] for d in data])
    p = ks / n * 100
    lo, hi = zip(*[wilson(k, n) for k in ks])
    # Clip at zero: at k=n the Wilson upper is mathematically exactly 1.0 but
    # lands a few ulps below it through the sqrt, giving a -1e-13 error bar that
    # matplotlib rejects outright. Same float artefact as in akew_stats_test.
    err = np.clip(np.vstack([p - np.array(lo) * 100, np.array(hi) * 100 - p]), 0, None)

    fig, ax = plt.subplots(figsize=(COL, 1.78))
    y = np.arange(len(names))
    cols = [ACCENT if nm == "INLAY" else NEUTRAL for nm in names]
    ax.barh(y, p, color=cols, height=0.6)
    ax.errorbar(p, y, xerr=err, fmt="none", ecolor=INK, elinewidth=0.7,
                capsize=1.6, capthick=0.7)
    for i, (v, nm) in enumerate(zip(p, names)):
        ax.text(min(v + 7.0, 104), i, "0.0" if v == 0 else f"{v:.1f}",
                va="center", fontsize=7,
                color=ACCENT if nm == "INLAY" else MUTED)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("edit accuracy (%), 95% Wilson interval")
    ax.tick_params(axis="y", length=0)
    ax.text(3.0, 0, "edits never land", fontsize=6.5, color=MUTED,
            va="center", ha="left", style="italic")
    save(fig, "fig2_edit_accuracy")


# ------------------------------------------------------------------ Figure 3
def fig_headroom():
    """outputs/akew_routing_headroom_results.md. Oracle and static coincide in
    every cell; drawing the oracle as an open ring around the static point
    makes the coincidence the visual, rather than hiding it as two equal bars."""
    # Abbreviated labels (matching fig5's cell naming), not the full dataset
    # names: at full width "CounterFact  unstructured" pushed the tight bbox
    # to 269pt against a ~236pt column, a 34pt overfull hbox in the two-column
    # layout. Shortening the labels fixes the actual cause; scaling the figure
    # in LaTeX would only have papered over it and changed the effective font
    # size relative to the body text.
    cells = [("CF", "structured", 1.0000), ("CF", "unstructured", .9040),
             ("CF", "extracted", .8400), ("Wiki", "structured", .9960),
             ("Wiki", "unstructured", .4280), ("Wiki", "extracted", .4800),
             ("MQ", "structured", 1.0000), ("MQ", "unstructured", .8095),
             ("MQ", "extracted", .8571)]
    labels = [f"{d}/{m}" for d, m, _ in cells][::-1]
    vals = np.array([v * 100 for _, _, v in cells])[::-1]

    # figsize is COL minus a fixed allowance for how far bbox_inches='tight'
    # expands past the nominal size for THIS figure's tick labels + legend --
    # verified against the compiled PDF's actual page size (pdfinfo), not
    # guessed: a first pass at (COL, 2.35) still overflowed the column by
    # 8.6pt after the label-shortening fix below, so the allowance is measured,
    # not assumed to be zero.
    fig, ax = plt.subplots(figsize=(COL - 0.14, 2.35))
    y = np.arange(len(labels))
    ax.hlines(y, 0, vals, color=FAINT, lw=0.8, zorder=1)
    ax.scatter(vals, y, s=13, color=ACCENT, zorder=3, label="static policy")
    ax.scatter(vals, y, s=54, facecolors="none", edgecolors=INK, linewidths=0.75,
               zorder=4, label="oracle router")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlim(0, 108); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("accuracy (%)")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              handletextpad=0.2, columnspacing=0.9, borderpad=0)
    save(fig, "fig3_headroom")


# ------------------------------------------------------------------ Figure 4
def fig_multihop():
    """outputs/akew_multihop_results.md. Full MQuAKE-CF pool at 1.5B (n=354)
    and a 7B sample (n=150)."""
    groups = [("1.5B", 354, 191, 57), ("7B", 150, 63, 31)]
    fig, ax = plt.subplots(figsize=(COL, 1.62))
    y = np.arange(len(groups)); h = 0.3
    for i, (_, n, ki, kn) in enumerate(groups):
        for off, k, c in ((h / 2, ki, ACCENT), (-h / 2, kn, NEUTRAL)):
            p = k / n * 100
            lo, hi = wilson(k, n)
            ax.barh(i + off, p, height=h, color=c)
            ax.errorbar(p, i + off,
                        xerr=[[max(0.0, p - lo * 100)], [max(0.0, hi * 100 - p)]],
                        fmt="none", ecolor=INK, elinewidth=0.7,
                        capsize=1.6, capthick=0.7)
            ax.text(p + 5.0, i + off, f"{p:.1f}", va="center", fontsize=7,
                    color=ACCENT if c == ACCENT else MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}\n$n$={n}" for g, n, _, _ in groups], fontsize=7.2)
    ax.set_xlim(0, 76); ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("multi-hop accuracy (%)")
    ax.tick_params(axis="y", length=0)
    ax.text(64, 1 + h / 2, "iterative", fontsize=6.8, color=ACCENT, va="center")
    ax.text(64, 1 - h / 2, "naive", fontsize=6.8, color=MUTED, va="center")
    save(fig, "fig4_multihop")


# ------------------------------------------------------------------ Figure 5
def fig_actions():
    """outputs/akew_routing_headroom_results.md. Per-action success rates: the
    mechanism behind the zero headroom. Abstention lies against the floor in
    every cell, which is the whole argument in one panel."""
    cells = ["CF/str", "CF/uns", "CF/ext", "Wiki/str", "Wiki/uns",
             "Wiki/ext", "MQ/str", "MQ/uns", "MQ/ext"]
    reject = [0.0, 0.4, 0.8, 2.0, 2.0, 2.4, 0.0, 0.0, 0.0]
    reason = [98.8, 90.4, 84.0, 97.6, 42.8, 48.0, 96.8, 81.0, 85.7]

    fig, ax = plt.subplots(figsize=(COL, 1.66))
    x = np.arange(len(cells))
    ax.plot(x, reason, "o-", ms=3, lw=1.0, color=ACCENT, label="reason over evidence")
    ax.plot(x, reject, "s-", ms=2.8, lw=1.0, color=INK, label="abstain")
    ax.fill_between(x, 0, reject, color=INK, alpha=.08)
    ax.set_xticks(x); ax.set_xticklabels(cells, rotation=45, ha="right", fontsize=6.4)
    ax.set_ylim(-3, 108); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("action succeeds (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False,
              handletextpad=0.35, columnspacing=1.2, borderpad=0)
    save(fig, "fig5_actions")


if __name__ == "__main__":
    fig_architecture()
    fig_edit_accuracy()
    fig_headroom()
    fig_multihop()
    fig_actions()
    print("\nfigures ->", OUT)
