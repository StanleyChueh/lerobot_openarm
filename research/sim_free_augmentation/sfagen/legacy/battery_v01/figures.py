"""Example frames: real source (top) vs synthetic (bottom), body_cam, for given (episode, key) pairs."""
import json, sys
from pathlib import Path
import numpy as np, cv2
HERE = Path(__file__).resolve().parent
sys.path[:0] = ["/home/stanley/Stanley_ws/sfa_hf_export/battery/bsfa", "/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation"]
import bdata
out = sys.argv[1]; picks = [(int(p.split(":")[0]), p.split(":")[1]) for p in sys.argv[2].split(",")]
rows = []
for E, key in picks:
    F = bdata.decode(bdata.CAMS[0], E); d = bdata.table(); d = d[d.episode_index == E]; ev = bdata.grip_events(np.stack(d["action"].to_numpy()))
    body = np.load(HERE / f"work/body/{E}_{key}.npy", mmap_mode="r")
    tLg, tRg = ev["L"]["closes"][0], ev["R"]["closes"][0]
    ts = [0, tLg - 20, tLg + 50, tRg] if key.startswith("st") else [0, tRg - 20, tRg + 10, tRg + 45]
    cols = []
    for t in ts:
        src = F[t].copy(); syn = np.asarray(body[t]).copy()
        cv2.putText(src, f"real ep {E}  t={t}", (8, 24), 0, 0.7, (255, 255, 0), 2)
        cv2.putText(syn, f"synthetic {key}", (8, 24), 0, 0.7, (255, 160, 60), 2)
        cols.append(np.concatenate([src, syn], 0))
    rows.append(np.concatenate(cols, 1))
cv2.imwrite(out, cv2.cvtColor(cv2.resize(np.concatenate(rows, 0), None, fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
