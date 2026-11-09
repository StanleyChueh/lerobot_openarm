import json, sys
from pathlib import Path
import numpy as np, cv2, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "render_v4"))
import mgw
from sfa3 import recon3
V31 = mgw.SFA / "v3_1/outputs"
sel = {(e["E"], e["key"]): e for e in json.loads((HERE / "selection.json").read_text())["selected"]}
rec = recon3.build(0, with_plate=True, log=lambda *a: None)
R = mgw.MGWForward(rec, device=sys.argv[1])
for key in ("init_x-6__G+__s1.0", "init_y+3__G+__s1.0"):
    e = sel[(0, key)]; a, b = e["rows"]
    z = np.load(V31 / f"ep0/candidates/{key}.npz"); vid = mgw.candidate_video(rec, z)
    old = np.load(V31 / f"cache/ep0/raw/{e['alias']}.npy", mmap_mode="r")
    res = {}
    for cc in (0, 9, 15):
        mgw.MGWForward.CLEAR_CAD = cc; P, G = [], []
        for k in range(0, b - a, 6):
            j, q, g, can = vid[a + k]; img, _ = R.render(j, q, g, can)
            Mt = R.placements(q, g, can); S = R.source_state(j)
            sub = torch.arange(0, len(R.local), 50, device=R.dev)
            uo, _ = R.project(R._world_sub(S["Ms"], sub)); un, _ = R.project(R._world_sub(Mt, sub))
            disp = torch.zeros(R.CAN + 1, device=R.dev).scatter_reduce(0, R.gid[sub], (un - uo).norm(dim=1), reduce="amax")
            moved = ((disp >= 2.0) & torch.tensor(R.part_side != "B", device=R.dev)).cpu().numpy()
            if not moved.any(): continue
            _, lt, _ = R.zbuf(Mt); lt = lt.cpu().numpy(); ls = S["lab"].cpu().numpy()
            T = (lt >= 0) & moved[np.clip(lt, 0, None)]; Sm = (ls >= 0) & moved[np.clip(ls, 0, None)]
            roi = cv2.dilate((T | Sm).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
            vac = Sm & ~cv2.dilate(T.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~((lt >= 0) & ~moved[np.clip(lt, 0, None)])
            fg = np.abs(img.astype(np.int16) - rec.plate.astype(np.int16)).max(2) > 25
            fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
            f = fg & roi; P.append((f & T).sum() / max(1, (f | T).sum()))
            if vac.sum() > 200: G.append((fg & vac).sum() / vac.sum())
            if cc == 9 and k == 60: cv2.imwrite(f"/tmp/claude-1000/-home-stanley-Stanley-ws-lerobot-openarm/fa7abc16-fc24-4e89-be66-4d1e5716fc93/scratchpad/clear_{key[:8]}.jpg", cv2.cvtColor(np.concatenate([np.asarray(old[a + k]), img], 1), cv2.COLOR_RGB2BGR))
        res[cc] = (round(float(np.mean(P)), 3), round(float(np.mean(G)), 3))
    print(key, "clear_cad -> (placed, ghost):", res, flush=True)
