"""Forest plot of the frozen v4B.1 paired contrasts (far grasp, all-40 handover): point = seed-averaged relative change,
bar = paired recording-bootstrap 95% CI, small markers = per-seed paired values. Single ink colour; rows are labelled.

    env -u PYTHONPATH -u LD_LIBRARY_PATH ../../../.venv/bin/python scripts/figures_v4b1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"

ROWS = [("RpBoth__vs__RpBoth_NoSynthRGB", "H1  Both (gen. RGB) vs Both (no synth RGB)"),
        ("RpBoth_rerun__vs__RpBoth", "      replication: Both rerun vs Both"),
        ("RpBoth_NoSynthRGB__vs__R", "      Both (no synth RGB) vs R"),
        ("RpBoth__vs__R", "      Both vs R (v4B)"),
        ("RpPos_PhaseMask__vs__RpPos", "H2  Position phase-mask vs Position"),
        ("RpPos_MatchedRandMask__vs__RpPos", "      per-row matched random vs Position"),
        ("RpPos_PhaseMask__vs__RpPos_MatchedRandMask", "      phase-mask vs per-row matched random"),
        ("RpPos_rerun__vs__RpPos", "      replication: Position rerun vs Position"),
        ("RpPos_PhaseMask__vs__R", "      Position phase-mask vs R"),
        ("RpPos__vs__R", "      Position vs R (v4B)")]


def main():
    res = json.loads((HERE / "results.json").read_text())
    C = res["contrasts"]
    rows = [(k, lab) for k, lab in ROWS if k in C]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * len(rows) + 1.6), sharey=True)
    for ax, (key, title) in zip(axes, (("far/grasp", "Far-position grasp error"), ("all40/handover", "Handover error (40 recordings)"))):
        for y, (k, _) in enumerate(rows):
            pt, ci = C[k][key]
            ax.plot([100 * ci[0], 100 * ci[1]], [y, y], color=INK, lw=2, solid_capstyle="round")
            ax.plot(100 * pt, y, "o", color=INK, ms=8, mec="white", mew=2, zorder=3)
            for s in C[k]["per_seed"].values():
                ax.plot(100 * s[key], y + 0.22, "|", color=MUTED, ms=7)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(title + "\nrelative change, + = first model better", fontsize=10, color=INK, loc="left")
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED)
        ax.set_xlabel("%", color=MUTED)
    axes[0].set_yticks(range(len(rows)), [lab for _, lab in rows], fontsize=9, color=INK)
    axes[0].invert_yaxis()
    fig.text(0.01, 0.01, "dot: seed-averaged paired change; bar: paired recording-bootstrap 95% CI; ticks: per-seed paired values. "
             "Development recordings 10-49; open-loop.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures/v4b1_contrasts.png", dpi=150)
    print("wrote figures/v4b1_contrasts.png")


if __name__ == "__main__":
    main()
