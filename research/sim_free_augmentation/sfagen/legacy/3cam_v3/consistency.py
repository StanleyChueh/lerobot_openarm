"""Label-image consistency of the synthetic body_cam frames, v3 renderer (published 20-episode revision) vs MGW.
For every 3rd frame, with the parts that the candidate MOVES relative to the source frame (>= 2 px):
  placed  = IoU( image foreground (|img - plate| > 25) , target silhouette of the moved parts ) inside the ROI
  ghost   = share of the moved parts' SOURCE silhouette (outside the target silhouette) that still shows foreground
            (= old arm/can left behind)
ROI = moved source U target silhouettes dilated 31 px. Silhouettes: z-buffered URDF/can mesh render."""
import json, sys
from pathlib import Path
import numpy as np, cv2, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "render_v4"))
import mgw
from sfa3 import recon3
V31 = mgw.SFA / "v3_1/outputs"
eps = [int(x) for x in sys.argv[1].split(",")]; dev = sys.argv[2]
sel = json.loads((HERE / "selection.json").read_text())["selected"]
out = {}
for E in eps:
    rec = recon3.build(E, with_plate=True, log=lambda *a: None)
    R = mgw.MGWForward(rec, device=dev)
    for e in [x for x in sel if x["E"] == E]:
        a, b = e["rows"]
        z = np.load(V31 / f"ep{E}/candidates/{e['key']}.npz"); vid = mgw.candidate_video(rec, z)
        new = np.load(HERE / f"work/body/{E}_{e['key']}.npy", mmap_mode="r"); old = np.load(V31 / f"cache/ep{E}/raw/{e['alias']}.npy", mmap_mode="r")
        res = {"v3": {"placed": [], "ghost": []}, "mgw": {"placed": [], "ghost": []}}
        for k in range(0, b - a, 3):
            j, q, g, can = vid[a + k]
            Mt = R.placements(q, g, can); S = R.source_state(j)
            sub = torch.arange(0, len(R.local), 50, device=R.dev)
            uo, _ = R.project(R._world_sub(S["Ms"], sub)); un, _ = R.project(R._world_sub(Mt, sub))
            disp = torch.zeros(R.CAN + 1, device=R.dev).scatter_reduce(0, R.gid[sub], (un - uo).norm(dim=1), reduce="amax")
            moved = ((disp >= 2.0) & torch.tensor(R.part_side != "B", device=R.dev)).cpu().numpy()
            if not moved.any():
                continue
            _, lt, _ = R.zbuf(Mt); lt = lt.cpu().numpy(); ls = S["lab"].cpu().numpy()
            T = (lt >= 0) & moved[np.clip(lt, 0, None)]; Sm = (ls >= 0) & moved[np.clip(ls, 0, None)]
            if T.sum() < 200:
                continue
            roi = cv2.dilate((T | Sm).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
            vac = Sm & ~cv2.dilate(T.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~((lt >= 0) & ~moved[np.clip(lt, 0, None)])
            for name, img in (("v3", np.asarray(old[a + k])), ("mgw", np.asarray(new[k]))):
                fg = np.abs(img.astype(np.int16) - rec.plate.astype(np.int16)).max(2) > 25
                fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
                f = fg & roi
                res[name]["placed"].append(float((f & T).sum() / max(1, (f | T).sum())))
                if vac.sum() > 200:
                    res[name]["ghost"].append(float((fg & vac).sum() / vac.sum()))
        out[f"{E}_{e['key']}"] = {n: {m: float(np.mean(v)) if v else None for m, v in d.items()} for n, d in res.items()}
        print(E, e["variation"], out[f"{E}_{e['key']}"], flush=True)
(HERE / f"analysis/consistency_{sys.argv[1].replace(',', '-')}.json").write_text(json.dumps(out, indent=1))
