"""Pringles hand-over-LOCATION variants (frozen v3.1 condition C: grasp-relative SE(3) coordination of both arms; the
hand-hand relation is preserved at every frame of the transfer, cf. DexMimicGen, Jiang et al. ICRA 2025).
1) replay checks identical to render_v4/traj_check.py; 2) MGW body-cam render; 3) label-image consistency (placed /
ghost, as 3cam_v3/consistency.py) per episode AND inside the hand-over window (left close -30 .. right open +20),
compared with the same metrics of the published v3 episodes (render gate)."""
import json, sys, types
from pathlib import Path
import numpy as np, cv2, torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
src = (ROOT / "render_v4/traj_check.py").read_text().rsplit("\nmain()", 1)[0]
tc = types.ModuleType("tc"); tc.__file__ = str(ROOT / "render_v4/traj_check.py"); exec(compile(src, "traj_check", "exec"), tc.__dict__)
sys.path.insert(0, str(ROOT / "render_v4"))
import mgw
from sfa3 import recon3
from sfa import data
V31 = mgw.SFA / "v3_1/outputs"; OUT = HERE / "pr_handover"; OUT.mkdir(exist_ok=True)
dev = sys.argv[1]
man = [e for e in json.loads((V31 / "dataset_export/C_all_manifest.json").read_text()) if "pos_" in e["key"]]


def metrics(R, rec, vid, frames, a, b, win):
    res = {"all": {"placed": [], "ghost": []}, "handover": {"placed": [], "ghost": []}}
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
        img = frames[k]
        fg = np.abs(img.astype(np.int16) - rec.plate.astype(np.int16)).max(2) > 25
        fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        f = fg & roi; pv = float((f & T).sum() / max(1, (f | T).sum())); gv = float((fg & vac).sum() / vac.sum()) if vac.sum() > 200 else None
        for w in (["all", "handover"] if win[0] <= a + k <= win[1] else ["all"]):
            res[w]["placed"].append(pv)
            if gv is not None:
                res[w]["ghost"].append(gv)
    return {w: {m: (float(np.mean(v)) if v else None) for m, v in d.items()} for w, d in res.items()}


def main():
    traj = json.loads((ROOT / "render_v4/traj_check.json").read_text())["envelope"]
    env_v, env_a = np.array(traj["speed_rad_s"]), np.array(traj["accel_rad_s2"])
    lo = np.r_[[tc.robot.lo[s][i] for i in range(7) for s in "RL"]]; hi = np.r_[[tc.robot.hi[s][i] for i in range(7) for s in "RL"]]
    D_REAL, Z_REAL = traj["real_min_mesh_distance_mm"] / 1000, traj["real_lowest_arm_z_m"]
    out = {}
    for E in sorted({e["E"] for e in man}):
        s = data.load(E, cams=()); idn = np.load(V31 / f"ep{E}/candidates/identity__G+__s1.0.npz"); evs = data.command_events(s.action)
        relR = tc.rel_grasp(s.state, idn["can"], evs["right_close"] + 3, "R"); relL = tc.rel_grasp(s.state, idn["can"], evs["left_close"] + 3, "L")
        rec = recon3.build(E, with_plate=True, log=lambda *a: None); R = mgw.MGWForward(rec, device=dev)
        for e in [x for x in man if x["E"] == E]:
            key = e["key"]; a, b = e["rows"]
            z = np.load(V31 / f"ep{E}/candidates/{key}.npz")
            S, A, can = z["state"][a:b].astype(np.float64), z["action"][a:b].astype(np.float64), z["can"][a:b]
            ev = data.command_events(A); v, ac = tc.kin(A); d = tc.min_dist(S); lz = tc.lowest_z(S)
            gerr = lambda X, Y: 1000 * float(np.linalg.norm(X[:3, 3] - Y[:3, 3]))  # noqa: E731
            c = {"limit_violations": int(((S[:, tc.ARM] < lo - 1e-6) | (S[:, tc.ARM] > hi + 1e-6)).sum() + ((A[:, tc.ARM] < lo - 1e-6) | (A[:, tc.ARM] > hi + 1e-6)).sum()),
                 "speed_ratio_to_real_max": float((v / env_v).max()), "accel_ratio_to_real_max": float((ac / env_a).max()),
                 "speed_ratio_to_urdf": float((v / np.r_[[tc.robot.vel[s_][i] for i in range(7) for s_ in "RL"]]).max()),
                 "min_mesh_distance_mm": 1000 * float(d.min()), "lowest_arm_z_m": float(lz.min()),
                 "grasp_R_err_mm": gerr(tc.rel_grasp(S, can, ev["right_close"] + 3, "R"), relR),
                 "start_state_err_rad": float(np.abs(S[0] - s.state[int(round(z["tau"][a]))]).max())}
            c["pass"] = bool(c["limit_violations"] == 0 and c["speed_ratio_to_real_max"] <= 1.25 and c["accel_ratio_to_real_max"] <= 1.5
                             and c["speed_ratio_to_urdf"] <= 1.0 and c["min_mesh_distance_mm"] >= min(1000 * D_REAL, 5.0)
                             and c["lowest_arm_z_m"] >= Z_REAL - 0.01 and c["grasp_R_err_mm"] <= 2 and c["start_state_err_rad"] <= 0.01)
            vid = mgw.candidate_video(rec, z); frames = []
            for i in range(a, b):
                j, q, g, cn = vid[i]; img, _ = R.render(j, q, g, cn); frames.append(img)
            frames = np.stack(frames); np.save(OUT / f"{E}_{key}.npy", frames)
            win = (evs["left_close"] - 30 + a, evs["right_open"] + 20 + a)
            c["render"] = metrics(R, rec, vid, frames, a, b, win)
            c["handover_tcp_shift_mm"] = 1000 * float(np.linalg.norm(
                tc.rel_grasp(S, np.repeat(np.eye(4)[None], len(S), 0), ev["left_close"], "R")[:3, 3] * 0 +
                (lambda sp: (tc.robot.fk(sp["R"][0], sp["L"][0]), tc.robot.frame(tc.robot.tcp["R"])[:3, 3])[1])(tc.Robot.split(S[ev["left_close"]:ev["left_close"] + 1]))
                - (lambda sp: (tc.robot.fk(sp["R"][0], sp["L"][0]), tc.robot.frame(tc.robot.tcp["R"])[:3, 3])[1])(tc.Robot.split(s.state[evs["left_close"]:evs["left_close"] + 1]))))
            out[f"{E}_{key}"] = c
            print(E, key, "replay", "PASS" if c["pass"] else "FAIL", round(c["speed_ratio_to_real_max"], 2), round(c["accel_ratio_to_real_max"], 2),
                  "shift mm", round(c["handover_tcp_shift_mm"], 1), c["render"], flush=True)
            (HERE / "pr_handover.json").write_text(json.dumps(out, indent=1))
        del R, rec; torch.cuda.empty_cache()


main()
