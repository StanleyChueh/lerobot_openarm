"""Probe: render existing hand-over-location candidates (frozen v3.1 G+/C) with MGW; montage around the hand-over."""
import json, sys
from pathlib import Path
import numpy as np, cv2
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "render_v4"))
import mgw
from sfa3 import recon3
V31 = mgw.SFA / "v3_1/outputs"
E = int(sys.argv[1]); keys = sys.argv[2].split(","); dev = sys.argv[3]; out = sys.argv[4]
man = {(e["E"], e["key"]): e for e in json.loads((V31 / "dataset_export/C_all_manifest.json").read_text())} | \
      {(e["E"], e["key"]): e for e in json.loads((V31 / "dataset_export/Gp_matched_manifest.json").read_text())}
rec = recon3.build(E, with_plate=True, log=lambda *a: None)
R = mgw.MGWForward(rec, device=dev)
ev = rec.ev; print("events", {k: ev[k] for k in ("right_close", "left_close", "right_open")})
rows = []
for key in keys:
    z = np.load(V31 / f"ep{E}/candidates/{key}.npz"); vid = mgw.candidate_video(rec, z)
    ts = [ev["right_close"] + 20, (ev["right_close"] + ev["left_close"]) // 2, ev["left_close"] - 5, ev["right_open"] + 5, ev["right_open"] + 40]
    row = []
    for t in ts:
        j, q, g, can = vid[t]; img, info = R.render(j, q, g, can)
        src = rec.F[j]
        row.append(np.concatenate([src, img], 0))
        print(key, t, info)
    rows.append(np.concatenate(row, 1))
cv2.imwrite(out, cv2.cvtColor(cv2.resize(np.concatenate(rows, 0), None, fx=0.45, fy=0.45), cv2.COLOR_RGB2BGR))
