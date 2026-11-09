"""v3 render (cached raw) vs MGW render for selected synthetic candidates and frames."""
import json, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, '.')
import mgw
from sfa3 import recon3
V31 = mgw.SFA / "v3_1/outputs"
OUT = Path("/tmp/claude-1000/-home-stanley-Stanley-ws-lerobot-openarm/fa7abc16-fc24-4e89-be66-4d1e5716fc93/scratchpad")
man = {(e["E"], e["variation"]): e for e in json.loads((V31 / "dataset_export/Gp_matched_evidence_complete_manifest.json").read_text())}
jobs = [(0, "orient_yaw30", (150, 190, 220)), (1, "grasp_higher", (160, 190)), (7, "pos_right_up", (159, 195)), (0, "init_x-6", (50, 110))]
dev = sys.argv[1]
for E in sorted({j[0] for j in jobs}):
    rec = recon3.build(E, with_plate=True, log=lambda *a: None)
    R = mgw.MGWForward(rec, device=dev)
    for (EE, var, ts) in jobs:
        if EE != E: continue
        e = man[(E, var)]; a, b = e["rows"]
        z = np.load(V31 / f"ep{E}/candidates/{e['key']}.npz")
        raw = np.load(V31 / f"cache/ep{E}/raw/{e['alias']}.npy", mmap_mode="r")
        vid = mgw.candidate_video(rec, z)
        rows = []
        for t in ts:
            i = a + t; j, q, g, can = vid[i]
            img, info = R.render(j, q, g, can)
            rows.append(np.concatenate([rec.F[j], raw[i], img], 1))
            print(E, var, t, info, flush=True)
        cv2.imwrite(str(OUT / f"cmp_{E}_{var}.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
