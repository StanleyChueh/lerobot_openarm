"""Human-review contact sheets: 5 body_cam key frames (start, 25%, 50%, 75%, last) of every evaluation episode,
labelled with the simulator's success flag, the gripper-derived stage and who holds the can at the end.
Automatic labels are NOT visual judgements; the sheets are for a person to check them."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, FIG

K = np.load(DATA / "eval_keyframes.npz")
E = pd.read_csv(DATA / "eval_episodes.csv")
out = FIG / "contact_sheets"
out.mkdir(parents=True, exist_ok=True)
for name in ["eval_manual", "eval_generated"]:
    g = E[E.set == name].reset_index(drop=True)
    for page in range(4):
        sub = g.iloc[page * 25:(page + 1) * 25]
        fig, ax = plt.subplots(len(sub), 5, figsize=(10, 2.0 * len(sub)))
        for r, (_, e) in enumerate(sub.iterrows()):
            fr = K[f"{name}__{int(e.episode)}"]
            for c in range(5):
                ax[r, c].imshow(fr[c]); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            ax[r, 0].set_ylabel(f"ep {int(e.episode)}\n{'SUCCESS' if e.success else 'fail'}", fontsize=8,
                                color="#1b7a3a" if e.success else "#b03030")
            ax[r, 4].set_title(f"stage: {e.stage_reached} | end holder: {e.end_holder} | "
                               f"x={e.x_est:.3f} y={e.y_est:.3f}", fontsize=7, loc="right")
        for c, t in enumerate(["start", "25%", "50%", "75%", "last frame"]):
            ax[0, c].set_title(t, fontsize=8)
        fig.suptitle(f"{name}: episodes {page * 25}-{page * 25 + len(sub) - 1} (human review sheet)", y=1.0)
        fig.tight_layout()
        fig.savefig(out / f"{name}_{page * 25:03d}.png", dpi=60)
        plt.close(fig)
print("written", sorted(p.name for p in out.iterdir()))
