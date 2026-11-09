"""Generate + check battery-position candidates for the given source episodes. Writes candidates/ep{E}/{key}.npz and
candidates/ep{E}/checks.json. Real-demo envelope (speed/accel/clearance) is computed over all 10 battery demos."""
import json, sys
from pathlib import Path
import numpy as np, cv2
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path[:0] = [str(HERE)]
import brecon, bplan, bdata  # noqa
import pinocchio as pin
from sfa.robot import Robot, URDF, MESH_DIR, NAMES

FPS = 30
SHIFTS = {f"bat_x{sx:+d}": (sx / 100, 0) for sx in (-6, -3, 3, 6)} | {f"bat_y{sy:+d}": (0, sy / 100) for sy in (-6, -3, 3, 6)}
ARM = [NAMES.index(f"{s}J{i}.pos") for i in range(1, 8) for s in "RL"]
robot = Robot()
cm = pin.buildGeomFromUrdf(robot.m, str(URDF), pin.GeometryType.COLLISION, package_dirs=[str(MESH_DIR)])
side = ["R" if g.name.startswith("openarm_right_") else "L" if g.name.startswith("openarm_left_") else "B" for g in cm.geometryObjects]
for i in range(len(side)):
    for j in range(i + 1, len(side)):
        n = cm.geometryObjects[i].name + cm.geometryObjects[j].name
        if {side[i], side[j]} == {"R", "L"} or ({side[i], side[j]} in ({"R", "B"}, {"L", "B"}) and "link0" not in n and "link1" not in n):
            cm.addCollisionPair(pin.CollisionPair(i, j))
cd = pin.GeometryData(cm)


def min_dist(S, step=3):
    sp = Robot.split(S.astype(np.float64)); d = []
    for t in range(0, len(S), step):
        pin.computeDistances(robot.m, robot.d, cm, cd, robot.q_full(sp["R"][t], sp["L"][t], sp["gR"][t], sp["gL"][t]))
        d.append(min(r.min_distance for r in cd.distanceResults))
    return np.array(d)


def lowest_z(S, step=3):
    sp = Robot.split(S.astype(np.float64)); z = []
    for t in range(0, len(S), step):
        robot.fk(sp["R"][t], sp["L"][t], sp["gR"][t], sp["gL"][t])
        z.append(min(P[:, 2].min() for n, s, P in robot.link_points("hull") if s in "RL" and n not in ("link0", "link1", "link2")))
    return np.array(z)


def kin(A):
    return np.abs(np.diff(A[:, ARM], axis=0)).max(0) * FPS, np.abs(np.diff(A[:, ARM], 2, axis=0)).max(0) * FPS ** 2


def envelope():
    p = ROOT / "cache/envelope.json"
    if p.exists():
        return json.loads(p.read_text())
    df = bdata.table(); V, Acc, D, Z = [], [], [], []
    for e in range(10):
        d = df[df.episode_index == e]; A = np.stack(d["action"].to_numpy()).astype(float); S = np.stack(d["observation.state"].to_numpy()).astype(float)
        v, a = kin(A); V.append(v); Acc.append(a); D.append(float(min_dist(S).min())); Z.append(float(lowest_z(S).min()))
        print("envelope", e, round(1000 * D[-1], 1), round(Z[-1], 3), flush=True)
    env = {"speed": np.max(V, 0).tolist(), "accel": np.max(Acc, 0).tolist(), "min_dist_m": min(D), "min_z_m": min(Z), "per_episode_min_dist_m": D}
    p.write_text(json.dumps(env)); return env


def placement_ok(rec, cand):
    Tb = cand["can"][0]; P = (Tb[:3, :3] @ rec._bat_points().T).T + Tb[:3, 3]
    uv, _ = rec.cam.project(P)
    inside = bool((uv[:, 0] > 20).all() and (uv[:, 0] < 620).all() and (uv[:, 1] > 20).all() and (uv[:, 1] < 460).all())
    m = np.zeros((480, 640), bool); ok = (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
    m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = True
    v = cv2.cvtColor(rec.plate, cv2.COLOR_RGB2HSV)[..., 2]
    on_table = float(np.median(v[cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0])) < 110
    r0, _ = rec.silhouettes(0, battery=False)
    no_robot = not (cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & r0).any()
    st = rec.tcp("L", rec.ev["L"]["closes"][0])[:2, 3]
    far_station = float(np.linalg.norm(Tb[:2, 3] + Tb[:2, 2] * brecon.BAT_L / 2 - st))
    return {"in_view": inside, "on_table": on_table, "not_under_robot": no_robot, "dist_to_station_m": far_station,
            "ok": inside and on_table and no_robot and far_station >= 0.10}


def main(eps, dev):
    env = envelope()
    lo = np.r_[[robot.lo[s][i] for i in range(7) for s in "RL"]]; hi = np.r_[[robot.hi[s][i] for i in range(7) for s in "RL"]]
    for E in eps:
        rec = brecon.BRecon(E, device=dev)
        out = ROOT / f"candidates/ep{E}"; out.mkdir(parents=True, exist_ok=True); checks = {}
        for key, dxy in SHIFTS.items():
            c = bplan.plan(rec, dxy)
            S, A = c["state"].astype(float), c["action"].astype(float)
            v, a = kin(A)
            moved = c["w"] > 1e-9
            dmin = min_dist(S[moved]) if moved.any() else np.array([np.inf])
            dsrc = min_dist(rec.state[moved]) if moved.any() else np.array([np.inf])
            spn, sps = Robot.split(S), Robot.split(rec.state)
            hn = np.linalg.norm(robot.tcp_traj({"R": spn["R"], "L": spn["L"]}, "R")[:, :3, 3] - robot.tcp_traj({"R": spn["R"], "L": spn["L"]}, "L")[:, :3, 3], axis=1)
            hs = np.linalg.norm(robot.tcp_traj({"R": sps["R"], "L": sps["L"]}, "R")[:, :3, 3] - robot.tcp_traj({"R": sps["R"], "L": sps["L"]}, "L")[:, :3, 3], axis=1)
            lz = lowest_z(S[moved]) if moved.any() else np.array([np.inf])
            pl = placement_ok(rec, c)
            ck = {"shift_m": list(dxy), "marks": c["marks"], "lead": int(c["lead"]),
                  "ik_pos_err_mm_max": float(1000 * c["ik_err"][:, 0].max()), "ik_rot_err_deg_max": float(c["ik_err"][:, 1].max()),
                  "limit_violations": int(sum(((X[:, ARM] < np.minimum(lo, Y[:, ARM].min(0)) - 1e-6) | (X[:, ARM] > np.maximum(hi, Y[:, ARM].max(0)) + 1e-6)).sum()
                                              for X, Y in ((S, rec.state), (A, rec.action)))),
                  "limit_rule": "URDF limits extended to the recorded demo's own range (the demo was executed)",
                  "speed_ratio_to_real_max": float((v / np.array(env["speed"])).max()), "accel_ratio_to_real_max": float((a / np.array(env["accel"])).max()),
                  "min_mesh_distance_mm": float(1000 * dmin.min()), "source_min_mesh_distance_same_frames_mm": float(1000 * dsrc.min()),
                  "min_hand_distance_mm": float(1000 * hn[moved].min()), "source_min_hand_distance_same_frames_mm": float(1000 * hs[moved].min()),
                  "lowest_arm_z_m": float(lz.min()), "real_lowest_arm_z_m": env["min_z_m"],
                  "left_arm_untouched": bool(np.array_equal(S[:, [NAMES.index(f'LJ{i}.pos') for i in range(1, 9)]], rec.state[:, [NAMES.index(f'LJ{i}.pos') for i in range(1, 9)]])),
                  "start_state_err_rad": float(np.abs(S[0] - rec.state[0]).max()),
                  "after_rejoin_identical": bool(np.array_equal(S[c["marks"]["t_rejoin"] + 1:], rec.state[c["marks"]["t_rejoin"] + 1:].astype(np.float32).astype(float))),
                  "placement": pl, "battery_initial_world_m": (c["can"][0][:3, 3] + c["can"][0][:3, 2] * brecon.BAT_L / 2).tolist()}
            ck["pass"] = bool(ck["ik_pos_err_mm_max"] <= 1.0 and ck["ik_rot_err_deg_max"] <= 0.5 and ck["limit_violations"] == 0
                              and ck["speed_ratio_to_real_max"] <= 1.25 and ck["accel_ratio_to_real_max"] <= 1.5
                              and ck["min_mesh_distance_mm"] >= min(ck["source_min_mesh_distance_same_frames_mm"], 20.0) - 1.0
                              and ck["min_hand_distance_mm"] >= ck["source_min_hand_distance_same_frames_mm"] - 5.0
                              and ck["lowest_arm_z_m"] >= env["min_z_m"] - 0.01 and ck["left_arm_untouched"]
                              and ck["start_state_err_rad"] <= 0.01 and pl["ok"])
            checks[key] = ck
            np.savez_compressed(out / f"{key}.npz", **{k: c[k] for k in ("state", "action", "qR", "qL", "can", "tau", "w", "ik_err")})
            print(E, key, "PASS" if ck["pass"] else "FAIL", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in ck.items()
                  if k in ("ik_pos_err_mm_max", "speed_ratio_to_real_max", "accel_ratio_to_real_max", "min_mesh_distance_mm", "min_hand_distance_mm", "source_min_hand_distance_same_frames_mm", "lowest_arm_z_m")}, pl, flush=True)
        (out / "checks.json").write_text(json.dumps(checks, indent=1, default=float))
        del rec


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1].split(",")], sys.argv[2])
