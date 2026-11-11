"""Station-position variants for the battery task: the charging station's resting pose is translated by d in the table
plane; the LEFT arm's reach, grasp and carry follow it (tplan.py), and rejoin the recording before the right arm comes
within 20 cm of the left hand (or when the left hand settles at the insertion pose, whichever is first). The
insertion itself is therefore the recorded, physically executed bimanual motion. Writes
target/candidates_station/ep{E}/{key}.npz and checks.json. Replay checks mirror battery/bsfa/bgen.py."""
import json, sys
from pathlib import Path
import numpy as np, cv2
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path[:0] = [str(HERE)]
import srecon, tplan  # noqa: E402  (srecon patches the battery reconstruction onto the calibrated camera)
sys.path.insert(0, str(ROOT / "battery/bsfa"))
import bgen  # noqa: E402  (collision model, kinematic envelope of the 10 real demos)
from sfa.robot import Robot, NAMES  # noqa: E402

SHIFTS = {f"st_x{sx:+d}": (sx / 100, 0) for sx in (-6, -3, 3, 6)} | {f"st_y{sy:+d}": (0, sy / 100) for sy in (-6, -3, 3, 6)}
OUT = HERE / "candidates_station"
RJ8 = [NAMES.index(f"RJ{i}.pos") for i in range(1, 9)]


def schedule(rec):
    sp = Robot.split(rec.state)
    TL = rec.robot.tcp_traj({"R": sp["R"], "L": sp["L"]}, "L")[:, :3, 3]; TR = rec.robot.tcp_traj({"R": sp["R"], "L": sp["L"]}, "R")[:, :3, 3]
    v = np.r_[0, np.linalg.norm(np.diff(TL, axis=0), axis=1)] * 30
    tg = rec.tLg; t0 = int(np.flatnonzero(v[:tg] > 0.02)[0])
    settle = next((t for t in range(tg + 20, rec.T - 15) if (v[t:t + 15] < 0.015).all()), rec.T - 1)
    near = next((t for t in range(tg, rec.T) if np.linalg.norm(TR[t] - TL[t]) < 0.20), rec.T - 1)
    return {"t_motion": t0, "t_grasp": tg, "t_settle": settle, "t_right_near": near, "t_rejoin": int(min(settle, near))}


def in_box(rec, S, T0, t_end):
    sp = Robot.split(S.astype(np.float64)); out = []
    for t in range(0, t_end, 3):
        rec.robot.fk(sp["R"][t], sp["L"][t], sp["gR"][t], sp["gL"][t]); h = 0
        for n, s, Q in rec.robot.link_points("hull"):
            if s not in "RL":
                continue
            ql = (Q - T0[:3, 3]) @ T0[:3, :3]
            h += int(((np.abs(ql[:, 0]) < rec.st_ext[0] / 2 + 0.01) & (np.abs(ql[:, 1]) < rec.st_ext[1] / 2 + 0.01) & (ql[:, 2] < rec.st_ext[2] + 0.01)).sum())
        out.append(h)
    return np.array(out)


def placement(rec, T0, bat_T, cand_state, marks):
    """station box at its new resting pose: in view, on the table, clear of the battery, and the robot never closer to
    the (moved) box than the recorded robot is to the recorded box, frame by frame, before the final approach (robot
    hull points within 1 cm of the box: candidate count <= source count)."""
    P = (T0[:3, :3] @ srecon.box_points(rec.st_ext, 0.004).T).T + T0[:3, 3]
    uv, _ = rec.cam.project(P)
    inside = bool((uv[:, 0] > 10).all() and (uv[:, 0] < 630).all() and (uv[:, 1] > 10).all() and (uv[:, 1] < 470).all())
    m = rec._station_sil(T0)
    v = cv2.cvtColor(rec.plate, cv2.COLOR_RGB2HSV)[..., 2]
    on_table = float(np.median(v[m])) < 110
    bat = bat_T[:3, 3] + bat_T[:3, 2] * 0.0325
    loc = np.linalg.inv(T0) @ np.r_[bat, 1.0]
    half = rec.st_ext[:2] / 2 + 0.02
    bat_clear = bool(np.any(np.abs(loc[:2]) > half))
    hc = in_box(rec, cand_state, T0, marks["t_grasp"] - 15); hs = in_box(rec, rec.state, rec.st_table, marks["t_grasp"] - 15)
    extra = int(np.maximum(hc - hs, 0).sum())
    return {"in_view": inside, "on_table": on_table, "clear_of_battery": bat_clear, "robot_points_near_box_extra_vs_source": extra,
            "ok": inside and on_table and bat_clear and extra == 0}


def main(eps, dev):
    env = bgen.envelope()
    lo = np.r_[[bgen.robot.lo[s][i] for i in range(7) for s in "RL"]]; hi = np.r_[[bgen.robot.hi[s][i] for i in range(7) for s in "RL"]]
    for E in eps:
        rec = srecon.StationRecon(E, device=dev)
        mk = schedule(rec)
        out = OUT / f"ep{E}"; out.mkdir(parents=True, exist_ok=True); checks = {}
        for key, dxy in SHIFTS.items():
            d = np.array([dxy[0], dxy[1], 0.0])
            try:
                c = tplan.plan(rec.robot, rec.state, rec.action, "L", d, mk["t_motion"], mk["t_grasp"], mk["t_rejoin"])
            except AssertionError as ex:
                checks[key] = {"pass": False, "error": f"schedule {ex}"}; continue
            S, A = c["state"], c["action"]
            v, a = bgen.kin(A); moved = c["w"] > 1e-9
            dmin = bgen.min_dist(S[moved]); dsrc = bgen.min_dist(rec.state[moved])
            spn, sps = Robot.split(S), Robot.split(rec.state)
            hd = lambda sp: np.linalg.norm(rec.robot.tcp_traj({"R": sp["R"], "L": sp["L"]}, "R")[:, :3, 3] - rec.robot.tcp_traj({"R": sp["R"], "L": sp["L"]}, "L")[:, :3, 3], axis=1)  # noqa: E731
            hn, hs = hd(spn), hd(sps)
            lz = bgen.lowest_z(S[moved])
            # station track: table pose shifted by d until the left grasp, then attached to the NEW left TCP
            st = np.zeros((rec.T, 4, 4))
            for t in range(rec.T):
                if t < rec.tLg + 3:
                    st[t] = rec.st_table.copy(); st[t][:3, 3] += d
                else:
                    rec.robot.fk(spn["R"][t], spn["L"][t]); st[t] = rec.robot.frame(rec.robot.tcp["L"]) @ rec.XS
            pl = placement(rec, st[0], rec.table, S, mk)
            ck = {"shift_m": list(dxy), "marks": mk, "lead": c["lead"],
                  "ik_pos_err_mm_max": float(1000 * c["ik_err"][:, 0].max()), "ik_rot_err_deg_max": float(c["ik_err"][:, 1].max()),
                  "limit_violations": int(sum(((X[:, bgen.ARM] < np.minimum(lo, Y[:, bgen.ARM].min(0)) - 1e-6) | (X[:, bgen.ARM] > np.maximum(hi, Y[:, bgen.ARM].max(0)) + 1e-6)).sum()
                                              for X, Y in ((S, rec.state), (A, rec.action)))),
                  "speed_ratio_to_real_max": float((v / np.array(env["speed"])).max()), "accel_ratio_to_real_max": float((a / np.array(env["accel"])).max()),
                  "min_mesh_distance_mm": float(1000 * dmin.min()), "source_min_mesh_distance_same_frames_mm": float(1000 * dsrc.min()),
                  "min_hand_distance_mm": float(1000 * hn[moved].min()), "source_min_hand_distance_same_frames_mm": float(1000 * hs[moved].min()),
                  "lowest_arm_z_m": float(lz.min()), "real_lowest_arm_z_m": env["min_z_m"],
                  "right_arm_untouched": bool(np.array_equal(S[:, RJ8], rec.state[:, RJ8])),
                  "start_state_err_rad": float(np.abs(S[0] - rec.state[0]).max()),
                  "after_rejoin_identical": bool(np.array_equal(S[mk["t_rejoin"] + 1:], rec.state[mk["t_rejoin"] + 1:])),
                  "placement": pl, "station_centre_initial_world_m": st[0][:3, 3].tolist(),
                  "left_tcp_at_grasp_world_m": c["tcp_new"][mk["t_grasp"]][:3, 3].tolist()}
            ck["pass"] = bool(ck["ik_pos_err_mm_max"] <= 1.0 and ck["ik_rot_err_deg_max"] <= 0.5 and ck["limit_violations"] == 0
                              and ck["speed_ratio_to_real_max"] <= 1.25 and ck["accel_ratio_to_real_max"] <= 1.5
                              and ck["min_mesh_distance_mm"] >= min(ck["source_min_mesh_distance_same_frames_mm"], 20.0) - 1.0
                              and ck["min_hand_distance_mm"] >= ck["source_min_hand_distance_same_frames_mm"] - 5.0
                              and ck["lowest_arm_z_m"] >= env["min_z_m"] - 0.01 and ck["right_arm_untouched"]
                              and ck["start_state_err_rad"] <= 0.01 and ck["after_rejoin_identical"] and pl["ok"])
            checks[key] = ck
            np.savez_compressed(out / f"{key}.npz", state=S.astype(np.float32), action=A.astype(np.float32), qR=c["q"]["R"], qL=c["q"]["L"],
                                can=st, tau=np.arange(rec.T, dtype=float), w=c["w"], ik_err=c["ik_err"])
            print(E, key, "PASS" if ck["pass"] else "FAIL", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in ck.items() if k in (
                "ik_pos_err_mm_max", "speed_ratio_to_real_max", "accel_ratio_to_real_max", "min_mesh_distance_mm",
                "source_min_mesh_distance_same_frames_mm", "min_hand_distance_mm", "source_min_hand_distance_same_frames_mm", "lowest_arm_z_m")}, pl, flush=True)
        (out / "checks.json").write_text(json.dumps(checks, indent=1, default=float))
        del rec


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1].split(",")], sys.argv[2])
