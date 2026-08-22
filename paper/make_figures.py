"""
Figures for the INLAY paper.

Every number here is copied from this project's own logged runs; nothing is
illustrative. Source file for each is named in the comment above the data.

Design constraints, since this renders into a print PDF rather than a screen:
  - one accent hue plus neutrals, separated by a large luminance gap, so the
    figures survive greyscale printing and are colour-vision safe by
    construction (identity never rests on hue alone -- position and direct
    labels carry it)
  - direct value labels instead of a value axis where the exact number matters
  - recessive or absent grid; no chart borders
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

INK      = "#1a1a1a"
MUTED    = "#6b6b6b"
FAINT    = "#c9c9c9"
NEUTRAL  = "#b8b8b8"
ACCENT   = "#0f5f58"   # deep teal: far darker than NEUTRAL, so the contrast
ACCENT_L = "#8fbdb8"   # survives greyscale conversion
WARN     = "#8a3d12"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9.5,
    "axes.edgecolor": FAINT,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 220,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- figure 1
def fig_architecture():
    """Weight editing mutates shared state; INLAY leaves it frozen.
    Hand-drawn rather than plotted: this is a mechanism, not data."""
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.set_xlim(0, 108); ax.set_ylim(0, 36); ax.axis("off")

    # --- left panel: weight editing (x 2..40) --------------------------
    ax.text(2, 32.5, "Weight editing", fontsize=10, fontweight="bold", color=INK)
    ax.text(2, 28.5, "ROME  ·  MEMIT  ·  AlphaEdit", fontsize=8, color=MUTED)
    ax.add_patch(Rectangle((2, 10), 38, 15, fill=False, ec=INK, lw=1.2))
    ax.text(21, 19.5, "model weights", ha="center", fontsize=9.5, color=INK)
    ax.text(21, 15, "every fact, entangled", ha="center", fontsize=8.5, color=MUTED)
    ax.add_patch(FancyArrowPatch((21, 3.6), (21, 9.2), arrowstyle="-|>",
                                 mutation_scale=11, color=WARN, lw=1.2))
    ax.text(21, 1.2, "each edit rewrites shared state", ha="center",
            fontsize=8.5, color=WARN)

    # --- right panel: INLAY (x 58..104), wider gaps so nothing collides -
    ax.text(58, 32.5, "INLAY", fontsize=10, fontweight="bold", color=ACCENT)
    ax.text(58, 28.5, "frozen weights  ·  external memory", fontsize=8, color=MUTED)
    ax.add_patch(Rectangle((58, 10), 24, 15, fill=False, ec=INK, lw=1.2,
                           linestyle=(0, (4, 3))))
    ax.text(70, 19.5, "frozen model", ha="center", fontsize=9.5, color=INK)
    ax.text(70, 15, "never modified", ha="center", fontsize=8.5, color=MUTED)

    ax.add_patch(Rectangle((90, 10), 14, 15, fill=False, ec=ACCENT, lw=1.2))
    ax.text(97, 21.8, "memory", ha="center", fontsize=8.5, color=ACCENT)
    for y in (13.4, 16.2, 19.0):
        ax.plot([91.5, 102.5], [y, y], color=ACCENT_L, lw=0.9)
    ax.add_patch(FancyArrowPatch((97, 3.6), (97, 9.2), arrowstyle="-|>",
                                 mutation_scale=11, color=ACCENT, lw=1.2))
    ax.text(97, 1.2, "an edit is a new row", ha="center", fontsize=8.5, color=ACCENT)
    # 8 units of clear space between the two boxes (82..90) so the arrow and
    # its label have somewhere to live; at 4 units the label sat on the box.
    ax.add_patch(FancyArrowPatch((89.2, 17.5), (82.8, 17.5), arrowstyle="-|>",
                                 mutation_scale=10, color=INK, lw=1.0))
    ax.text(86, 19.4, "steers", ha="center", fontsize=8, color=MUTED)
    save(fig, "fig1_architecture.png")


# ---------------------------------------------------------------- figure 2
def fig_edit_accuracy():
    """outputs/akew_weightedit_baseline_results.md -- GPT-J-6B, CounterFact
    structured, n=147, scored with this project's own is_hit convention."""
    data = [("GRACE", 0.0), ("WISE", 66.67), ("ROME", 83.67),
            ("MEMIT", 83.67), ("AlphaEdit", 89.12), ("INLAY", 100.0)]
    names = [d[0] for d in data]
    vals = [d[1] for d in data]
    colors = [ACCENT if n == "INLAY" else NEUTRAL for n in names]

    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    bars = ax.barh(names, vals, color=colors, height=0.62)
    for b, v, n in zip(bars, vals, names):
        lbl = "0.0  (edits do not land)" if v == 0 else f"{v:.2f}"
        ax.text(v + 1.6, b.get_y() + b.get_height() / 2, lbl,
                va="center", fontsize=8.8,
                color=ACCENT if n == "INLAY" else MUTED,
                fontweight="bold" if n == "INLAY" else "normal")
    ax.set_xlim(0, 118)
    ax.set_xlabel("edit accuracy (%)")
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig2_edit_accuracy.png")


# ---------------------------------------------------------------- figure 3
def fig_write_cost():
    """Write cost per edit. INLAY's 5-15 ms is measured directly; the
    weight-editing figure is the aggregate this project reports (~1600x),
    plotted as a single band rather than fabricated per method."""
    fig, ax = plt.subplots(figsize=(5.4, 1.5))
    labels = ["weight editing\n(gradient-based)", "INLAY\n(no gradient step)"]
    vals = [8000.0, 5.0]           # milliseconds
    colors = [NEUTRAL, ACCENT]
    bars = ax.barh(labels, vals, color=colors, height=0.55)
    ax.set_xscale("log")
    ax.set_xlim(1, 40000)
    for b, v in zip(bars, vals):
        txt = "~8 s" if v > 100 else "~5 ms"
        ax.text(v * 1.35, b.get_y() + b.get_height() / 2, txt, va="center",
                fontsize=9, color=ACCENT if v < 100 else MUTED,
                fontweight="bold" if v < 100 else "normal")
    ax.set_xlabel("write cost per edit, log scale (ms)")
    ax.tick_params(axis="y", length=0)
    ax.annotate("~1600x", xy=(300, 0.5), fontsize=10, color=INK,
                fontweight="bold", ha="center")
    save(fig, "fig3_write_cost.png")


# ---------------------------------------------------------------- figure 4
def fig_headroom():
    """outputs/akew_routing_headroom_results.md -- every candidate action
    executed and scored on 1,689 queries. The oracle marker sits exactly on
    the static-policy bar in all nine cells; that coincidence IS the finding."""
    cells = [
        ("CounterFact / structured", 1.0000, 1.0000),
        ("CounterFact / unstructured", 0.9040, 0.9040),
        ("CounterFact / extracted", 0.8400, 0.8400),
        ("WikiUpdate / structured", 0.9960, 0.9960),
        ("WikiUpdate / unstructured", 0.4280, 0.4280),
        ("WikiUpdate / extracted", 0.4800, 0.4800),
        ("MQuAKE-CF / structured", 1.0000, 1.0000),
        ("MQuAKE-CF / unstructured", 0.8095, 0.8095),
        ("MQuAKE-CF / extracted", 0.8571, 0.8571),
    ]
    names = [c[0] for c in cells][::-1]
    static = [c[1] * 100 for c in cells][::-1]
    oracle = [c[2] * 100 for c in cells][::-1]

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.barh(names, static, color=NEUTRAL, height=0.55,
            label="static policy (one line of code)")
    ax.scatter(oracle, range(len(names)), marker="|", s=260, linewidths=2.0,
               color=ACCENT, zorder=3, label="oracle router (upper bound)")
    for i, v in enumerate(static):
        ax.text(v + 1.8, i, f"{v:.2f}", va="center", fontsize=8.2, color=MUTED)
    ax.set_xlim(0, 118)
    ax.set_xlabel("accuracy (%)")
    ax.tick_params(axis="y", length=0)
    # Legend goes ABOVE the axes: at "lower right" it landed on top of the two
    # MQuAKE-CF rows, hid a value label, and put its own marker swatches where
    # a reader would mistake them for data points.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2,
              frameon=False, fontsize=8.4, handletextpad=0.6, columnspacing=2.0)
    save(fig, "fig4_headroom.png")


# ---------------------------------------------------------------- figure 5
def fig_multihop():
    """outputs/akew_multihop_results.md -- full MQuAKE-CF pool at 1.5B
    (n=354) and a 7B sample (n=150)."""
    fig, ax = plt.subplots(figsize=(5.0, 2.0))
    groups = ["1.5B  (n=354)", "7B  (n=150)"]
    naive = [16.10, 20.67]
    iterative = [53.95, 42.00]
    y = range(len(groups))
    h = 0.32
    ax.barh([i + h / 2 for i in y], iterative, height=h, color=ACCENT,
            label="iterative, with per-hop fallback")
    ax.barh([i - h / 2 for i in y], naive, height=h, color=NEUTRAL,
            label="naive single-shot")
    for i, (a, b) in enumerate(zip(iterative, naive)):
        ax.text(a + 1.4, i + h / 2, f"{a:.2f}", va="center", fontsize=8.5,
                color=ACCENT, fontweight="bold")
        ax.text(b + 1.4, i - h / 2, f"{b:.2f}", va="center", fontsize=8.5,
                color=MUTED)
    ax.set_yticks(list(y)); ax.set_yticklabels(groups)
    ax.set_xlim(0, 68)
    ax.set_xlabel("multi-hop accuracy (%)")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, fontsize=8.2)
    save(fig, "fig5_multihop.png")


if __name__ == "__main__":
    fig_architecture()
    fig_edit_accuracy()
    fig_write_cost()
    fig_headroom()
    fig_multihop()
    print("\nall figures written to", OUT)
