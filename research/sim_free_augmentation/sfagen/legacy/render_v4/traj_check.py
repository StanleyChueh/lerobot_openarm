"""Replay-readiness of every G+ candidate trajectory (kinematic/geometric checks only; nothing runs on the robot).

Per candidate (exported rows a:b), on the commanded ACTION stream (what a replay sends) and the planned state:
  limits        URDF position limits (state and action)
  speed/accel   per-joint |da|/dt and |d2a|/dt2 against the envelope of the 10 real demos (max over demos) x margin
  collision     coal mesh distance between the two arms' collision meshes, and each arm vs body; compared with the
                minimum seen in the real demos (which were physically executed)
  table         lowest arm collision point above the table plane vs the real demos
  grasp         hand-can relative pose at right grasp and left grasp equals the source's (the attachment model)
  start         frame 0 state equals the source frame 0 (replay starts from a recorded pose)
  gripper       gripper command sequence equals the source's at the time map
"""
import json, sys
from pathlib import Path
import numpy as np
import pinocchio as pin
SFA = Path("/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation"); sys.path.insert(0, str(SFA))
from sfa import data
from sfa.robot import Robot, URDF, MESH_DIR, NAMES
V31 = SFA / "v3_1/outputs"
HERE = Path(__file__).resolve().parent
FPS = 30
robot = Robot()
cm = pin.buildGeomFromUrdf(robot.m, str(URDF), pin.GeometryType.COLLISION, package_dirs=[str(MESH_DIR)])
side = ["R" if g.name.startswith("openarm_right_") else "L" if g.name.startswith("openarm_left_") else "B" for g in cm.geometryObjects]
pairs = [(i, j) for i in range(len(side)) for j in range(i + 1, len(side))
         if {side[i], side[j]} == {"R", "L"} or ({side[i], side[j]} in ({"R", "B"}, {"L", "B"}) and
                                                 not any(k in cm.geometryObjects[i].name + cm.geometryObjects[j].name for k in ("link0", "link1")))]
for i, j in pairs:
    cm.addCollisionPair(pin.CollisionPair(i, j))
cd = pin.GeometryData(cm)
ARM = [NAMES.index(f"{s}J{i}.pos") for i in range(1, 8) for s in "RL"]
GRIP = [NAMES.index("RJ8.pos"), NAMES.index("LJ8.pos")]


def min_dist(S, step=2):
    sp = Robot.split(S.astype(np.float64)); d = []
    for t in range(0, len(S), step):
        q = robot.q_full(sp["R"][t], sp["L"][t], sp["gR"][t], sp["gL"][t])
        pin.computeDistances(robot.m, robot.d, cm, cd, q)
        d.append(min(r.min_distance for r in cd.distanceResults))
    return np.array(d)


def lowest_z(S, step=2):
    sp = Robot.split(S.astype(np.float64)); z = []
    for t in range(0, len(S), step):
        robot.fk(sp["R"][t], sp["L"][t], sp["gR"][t], sp["gL"][t])
        z.append(min(P[:, 2].min() for n, s, P in robot.link_points("hull") if s in "RL" and n not in ("link0", "link1", "link2")))
    return np.array(z)


def rel_grasp(S, can, t, s):
    sp = Robot.split(S[t:t + 1].astype(np.float64))
    robot.fk(sp["R"][0], sp["L"][0])
    return np.linalg.inv(robot.frame(robot.tcp[s])) @ can[t]


def kin(A):
    v = np.abs(np.diff(A[:, ARM], axis=0)) * FPS
    a = np.abs(np.diff(A[:, ARM], 2, axis=0)) * FPS ** 2
    return v.max(0), a.max(0)


def main():
    man = json.loads((V31 / "dataset_export/Gp_matched_evidence_complete_manifest.json").read_text())
    lo = np.r_[[robot.lo[s][i] for i in range(7) for s in "RL"]]; hi = np.r_[[robot.hi[s][i] for i in range(7) for s in "RL"]]
    real = {}
    env_v, env_a, dmin_real, zmin_real = [], [], [], []
    for E in range(10):
        s = data.load(E, cams=())
        A, S = s.action.astype(np.float64), s.state.astype(np.float64)
        v, a = kin(A); env_v.append(v); env_a.append(a)
        d = min_dist(S); z = lowest_z(S)
        dmin_real.append(d.min()); zmin_real.append(z.min())
        idn = np.load(V31 / f"ep{E}/candidates/identity__G+__s1.0.npz")
        ev = data.command_events(A)
        real[E] = {"S": S, "A": A, "can": idn["can"], "ev": ev, "dmin": float(d.min()), "zmin": float(z.min()),
                   "relR": rel_grasp(S, idn["can"], ev["right_close"] + 3, "R"), "relL": rel_grasp(S, idn["can"], ev["left_close"] + 3, "L")}
        print("real", E, "min arm dist mm", round(1000 * d.min(), 1), "lowest z", round(z.min(), 3), flush=True)
    env_v, env_a = np.max(env_v, 0), np.max(env_a, 0)
    D_REAL, Z_REAL = float(min(dmin_real)), float(min(zmin_real))
    out = []
    for e in man:
        E, key = e["E"], e["key"]; a, b = e["rows"]
        z = np.load(V31 / f"ep{E}/candidates/{key}.npz")
        S, A, can = z["state"][a:b].astype(np.float64), z["action"][a:b].astype(np.float64), z["can"][a:b]
        tau = z["tau"][a:b]
        r = real[E]; ev = data.command_events(A)
        v, ac = kin(A)
        d = min_dist(S); lz = lowest_z(S)
        relR = rel_grasp(S, can, ev["right_close"] + 3, "R"); relL = rel_grasp(S, can, ev["left_close"] + 3, "L")
        gerr = lambda X, Y: (1000 * float(np.linalg.norm(X[:3, 3] - Y[:3, 3])), float(np.degrees(np.linalg.norm(pin.log3(X[:3, :3].T @ Y[:3, :3])))))
        ti = np.clip(np.round(tau).astype(int), 0, len(r["A"]) - 1)
        c = {"E": E, "key": key, "variation": e["variation"], "family": e["family"], "frames": int(b - a),
             "limit_violations_state": int(((S[:, ARM] < lo - 1e-6) | (S[:, ARM] > hi + 1e-6)).sum()),
             "limit_violations_action": int(((A[:, ARM] < lo - 1e-6) | (A[:, ARM] > hi + 1e-6)).sum()),
             "speed_ratio_to_real_max": float((v / env_v).max()), "accel_ratio_to_real_max": float((ac / env_a).max()),
             "speed_ratio_to_urdf": float((v / np.r_[[robot.vel[s][i] for i in range(7) for s in "RL"]]).max()),
             "min_mesh_distance_mm": 1000 * float(d.min()), "real_min_mesh_distance_mm": 1000 * D_REAL,
             "lowest_arm_z_m": float(lz.min()), "real_lowest_arm_z_m": Z_REAL,
             "grasp_R_err_mm_deg": gerr(relR, r["relR"]), "grasp_L_err_mm_deg": gerr(relL, r["relL"]),
             "start_state_err_rad": float(np.abs(S[0] - r["S"][int(round(tau[0]))]).max()),
             "gripper_cmd_equal": bool(np.allclose(A[:, GRIP], r["A"][ti][:, GRIP], atol=1e-4)) if np.allclose(tau, np.round(tau)) else None,
             "can_init_xy_m": can[0][:2, 3].tolist(), "source_can_init_xy_m": r["can"][0][:2, 3].tolist()}
        c["pass"] = bool(c["limit_violations_action"] == 0 and c["limit_violations_state"] == 0
                         and c["speed_ratio_to_real_max"] <= 1.25 and c["accel_ratio_to_real_max"] <= 1.5
                         and c["speed_ratio_to_urdf"] <= 1.0
                         and c["min_mesh_distance_mm"] >= min(1000 * D_REAL, 5.0)
                         and c["lowest_arm_z_m"] >= Z_REAL - 0.01
                         and c["grasp_R_err_mm_deg"][0] <= 2 and c["grasp_L_err_mm_deg"][0] <= 2
                         and c["start_state_err_rad"] <= 0.01)
        out.append(c)
        print(E, key, "PASS" if c["pass"] else "FAIL", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()
              if k in ("speed_ratio_to_real_max", "accel_ratio_to_real_max", "min_mesh_distance_mm", "lowest_arm_z_m", "grasp_R_err_mm_deg", "grasp_L_err_mm_deg", "start_state_err_rad")}, flush=True)
    rep = {"envelope": {"speed_rad_s": env_v.tolist(), "accel_rad_s2": env_a.tolist(), "real_min_mesh_distance_mm": 1000 * D_REAL,
                        "real_lowest_arm_z_m": Z_REAL, "real_per_episode": {E: {"dmin_mm": 1000 * real[E]["dmin"], "zmin": real[E]["zmin"]} for E in real}},
           "candidates": out, "n_pass": sum(c["pass"] for c in out), "n": len(out)}
    (HERE / "traj_check.json").write_text(json.dumps(rep, indent=1, default=float))
    print("PASS", rep["n_pass"], "/", rep["n"])

main()
