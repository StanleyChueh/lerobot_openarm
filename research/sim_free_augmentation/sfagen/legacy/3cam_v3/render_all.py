"""Render every selected candidate's body_cam frames with the MGW forward renderer (render_v4/mgw.py)."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "render_v4"))
import mgw
from sfa3 import recon3
V31 = mgw.SFA / "v3_1/outputs"
OUT = HERE / "work/body"; OUT.mkdir(parents=True, exist_ok=True)
assert not (HERE.parent / "render_v4/calib.json").exists(), "renderer must use the pinned inherited camera"
eps = [int(x) for x in sys.argv[1].split(",")]; dev = sys.argv[2]
sel = json.loads((HERE / "selection.json").read_text())["selected"]
for E in eps:
    todo = [e for e in sel if e["E"] == E and not (OUT / f"{E}_{e['key']}.npy").exists()]
    if not todo:
        continue
    rec = recon3.build(E, with_plate=True, log=lambda *a: None)
    R = mgw.MGWForward(rec, device=dev)
    for e in todo:
        t0 = time.time(); a, b = e["rows"]
        z = np.load(V31 / f"ep{E}/candidates/{e['key']}.npz")
        vid = mgw.candidate_video(rec, z)
        frames, infos = [], []
        for i in range(a, b):
            j, q, g, can = vid[i]
            img, info = R.render(j, q, g, can)
            frames.append(img); infos.append({"source_frame": j, **info})
        np.save(OUT / f"{E}_{e['key']}.npy", np.stack(frames))
        (OUT / f"{E}_{e['key']}.json").write_text(json.dumps(infos))
        print(E, e["key"], b - a, round(time.time() - t0, 1), "s", flush=True)
