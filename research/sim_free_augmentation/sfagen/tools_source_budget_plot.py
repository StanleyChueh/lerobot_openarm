"""Coverage vs number of real source demos (mean over all subsets of that size, band = min..max over subsets)."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = Path(__file__).resolve().parent
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"real only": "#8f8e8a", "real + synthetic (standard grid)": "#2a78d6", "real + synthetic (dense grid, interaction ≤ 4.3 cm)": "#eb6834",
       "real + synthetic (dense grid, uncapped)": "#eb6834"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})
for task in ("pringles", "battery"):
    std = json.loads((S / f"logs/source_budget_{task}.json").read_text())
    dp = S / f"logs/source_budget_{task}_dense.json"; den = json.loads(dp.read_text()) if dp.exists() else None
    cp = S / f"logs/source_budget_{task}_dense_capped.json"; cap = json.loads(cp.read_text()) if cp.exists() else None
    keys = [k[: -len("_src_within_2cm")] for k in std["per_n"]["1"] if k.endswith("_src_within_2cm")]
    fig, axs = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.8), constrained_layout=True, squeeze=False)
    for ax, k in zip(axs[0], keys):
        npool = std["n_pool"][k]; ns = sorted(int(n) for n in std["per_n"])
        series = [("real only", std, "src"), ("real + synthetic (standard grid)", std, "aug")] + \
                 ([("real + synthetic (dense grid, interaction ≤ 4.3 cm)", cap, "aug")] if cap else []) + ([("real + synthetic (dense grid, uncapped)", den, "aug")] if den else [])
        for name, src, side in series:
            m = [src["per_n"][str(n)][f"{k}_{side}_within_2cm"]["mean"] for n in ns]
            lo = [src["per_n"][str(n)][f"{k}_{side}_within_2cm"]["min"] for n in ns]; hi = [src["per_n"][str(n)][f"{k}_{side}_within_2cm"]["max"] for n in ns]
            dotted = "uncapped" in name
            if not dotted:
                ax.fill_between(ns, lo, hi, color=COL[name], alpha=0.12, lw=0)
            ax.plot(ns, m, color=COL[name], lw=1.2 if dotted else 2, ls=":" if dotted else "-", marker=None if dotted else "o", ms=4, label=name)
        base = std["per_n"]["10"][f"{k}_src_within_2cm"]["mean"]
        ax.axhline(base, color=INK2, lw=1, ls="--"); ax.text(10, base, " 10 real only", va="bottom", ha="right", color=INK2, fontsize=8)
        ax.set_xticks(ns); ax.set_xlabel("real source demos used"); ax.set_ylim(0, npool + 1)
        ax.set_ylabel(f"other real recordings within 2 cm (of {npool})"); ax.grid(color=GRID, lw=0.5); ax.set_title(f"{task}: {k} position")
    h, l = axs[0][0].get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle(f"{task.capitalize()}: coverage of unseen real recordings vs number of real source demos (mean; band = best/worst subset)", fontweight="bold", fontsize=10)
    fig.savefig(S / f"logs/source_budget_{task}.png", dpi=150); plt.close(fig); print("saved", task)
