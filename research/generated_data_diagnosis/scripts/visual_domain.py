"""How far are the generated frames from what the policy sees at evaluation (no domain randomization)?

For each camera, the reference is the per-pixel median of the 200 evaluation first frames (can at different
spots, so the median is the empty scene). Every episode's first frame is scored by the normalized correlation
of its downsampled grayscale with that reference, over the pixels the can never covers at evaluation.
Manual demos were recorded through the same server env as the evaluations, so their scores show the nominal
range; a generated frame inside that range is "evaluation-like".
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DATA, DATASETS, REPO, savefig

sys.path.insert(0, str(REPO))
from tools.dataset_diagnostics.loader import load_dataset  # noqa: E402

CAMS = ["observation.images.body_cam", "observation.images.wrist_cam", "observation.images.right_wrist_cam"]


def gray_small(rgb):
    g = rgb.astype(np.float32).mean(-1)
    return g.reshape(60, 8, 80, 8).mean((1, 3))


frames = {}
for name in ["eval_manual", "eval_generated", "manual", "generated"]:
    v = load_dataset(root=DATASETS[name], offline=True)
    for cam in CAMS:
        frames[(name, cam)] = np.stack([gray_small(v.read_frame(cam, ep, 0)) for ep in range(v.info["total_episodes"])])
    print(name, "done", flush=True)

out, fig_data = {}, {}
for cam in CAMS:
    ev = np.r_[frames[("eval_manual", cam)], frames[("eval_generated", cam)]]
    ref = np.median(ev, 0)
    # pixels the can never darkens/brightens much at evaluation: small spread across eval frames
    spread = ev.std(0)
    mask = spread < np.percentile(spread, 60)

    def score(x):
        a = x[:, mask] - x[:, mask].mean(1, keepdims=True)
        b = ref[mask] - ref[mask].mean()
        return (a @ b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b) + 1e-9)

    s = {n: score(frames[(n, cam)]) for n in ["eval_manual", "eval_generated", "manual", "generated"]}
    lo = min(s["manual"].min(), s["eval_manual"].min(), s["eval_generated"].min())
    out[cam] = {n: {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())} for n, v in s.items()}
    out[cam]["nominal_threshold(min of manual+eval)"] = float(lo)
    out[cam]["generated_frac_within_nominal"] = float((s["generated"] >= lo).mean())
    fig_data[cam] = s
    print(cam, json.dumps(out[cam]))
json.dump(out, open(DATA / "visual_domain.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for a, cam in zip(ax, CAMS):
    s = fig_data[cam]
    bins = np.linspace(min(v.min() for v in s.values()), 1, 50)
    a.hist(s["generated"], bins=bins, color="#eb6834", alpha=0.8, label="generated_500")
    for n, c in [("manual", "#2a78d6"), ("eval_manual", "#1baf7a"), ("eval_generated", "#7a5cd6")]:
        a.hist(s[n], bins=bins, histtype="step", lw=1.8, color=c, label=n, density=False)
    a.axvline(out[cam]["nominal_threshold(min of manual+eval)"], color="k", ls=":")
    a.set_title(f"{cam.split('.')[-1]}: {out[cam]['generated_frac_within_nominal']:.0%} of generated in nominal range", fontsize=10)
    a.set_xlabel("correlation with the evaluation scene (first frame)"); a.set_yscale("log")
ax[0].legend(fontsize=8)
savefig(fig, "visual_domain_gap.png")
