"""Body-camera distortion score from per-frame renderer diagnostics (render.py `tmask_px`, `miss_px`, `spill_px`).

Per rendered frame:  s = (miss + spill) / tmask
  tmask  visible target silhouette of every moved robot part / object (FK + proxy z-buffer)
  miss   silhouette pixels not covered by real pixels (left as table plate, part-filled or inpainted): missing arm/can
  spill  forward-warped pixels landing > 3 px outside the silhouette: smear / ghost carried with the arm
Frames with tmask < MIN_PX and copied frames score 0. Episode score = 95th percentile over frames.
The earlier provenance-only score (borrowed + inpainted fraction) was rejected: it scored a visibly broken 12 cm F1W
episode at 0.002 because missing fingers are neither warped nor counted as filled."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

MIN_PX = 2000


def frame_scores(frames):
    out = []
    for f in frames:
        if f.get("copied"):
            out.append(0.0); continue
        if "tmask_px" not in f:
            raise ValueError("render diagnostics missing (rendered before distortion instrumentation)")
        t = int(f["tmask_px"])
        out.append((int(f["miss_px"]) + int(f["spill_px"])) / t if t >= MIN_PX else 0.0)
    return np.asarray(out, float)


def episode_score(frames):
    s = frame_scores(frames)
    return float(np.percentile(s, 95)) if len(s) else 0.0


def score_file(path):
    return episode_score(json.loads(Path(path).read_text()))
