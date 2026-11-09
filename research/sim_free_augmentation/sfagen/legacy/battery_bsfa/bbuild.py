"""Assemble ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00: every battery-position candidate that passes
all replay checks; body_cam = MGW render; wrist cams = the source recording's real wrist frames at the same time index
(time map is the identity), flagged exact where the whole scene (all joints + battery pose) equals the recording."""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import bdata, brecon  # noqa
from sfa.robot import Robot
REPO_ID = "ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00"
CAMS = bdata.CAMS; NAMES = bdata.NAMES
OUT = ROOT / "dataset"


def features():
    v = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    return {"observation.state": {"dtype": "float32", "shape": (16,), "names": NAMES},
            "action": {"dtype": "float32", "shape": (16,), "names": NAMES}, **{c: dict(v) for c in CAMS},
            "wrist_view_valid": {"dtype": "float32", "shape": (2,), "names": ["wrist_cam", "right_wrist_cam"]},
            "wrist_view_pose_offset": {"dtype": "float32", "shape": (4,), "names": ["wrist_cam_trans_m", "wrist_cam_rot_rad",
                                                                                   "right_wrist_cam_trans_m", "right_wrist_cam_rot_rad"]}}


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pandas as pd
    task = pd.read_parquet(bdata.ROOT / "meta/tasks.parquet").index[0]
    robot = Robot()
    tcp = lambda S: {s: robot.tcp_traj({"R": Robot.split(S)["R"], "L": Robot.split(S)["L"]}, s) for s in "LR"}  # noqa: E731
    sel = []
    for E in range(10):
        p = ROOT / f"candidates/ep{E}/checks.json"
        if p.exists():
            sel += [(E, k, v) for k, v in sorted(json.loads(p.read_text()).items()) if v["pass"]]
    if OUT.exists():
        shutil.rmtree(OUT)
    ds = LeRobotDataset.create(repo_id=REPO_ID, fps=30, features=features(), root=OUT, robot_type="openarm_umeow",
                               use_videos=True, streaming_encoding=True)
    eps = []; cur = None
    for n, (E, key, ck) in enumerate(sel):
        if cur is None or cur[0] != E:
            rec = brecon.BRecon(E, device="cuda:0"); ep = rec.ep; cur = (E, rec, ep, tcp(ep.state))
        _, rec, ep, tsrc = cur
        z = np.load(ROOT / f"candidates/ep{E}/{key}.npz")
        st, ac = z["state"].astype(np.float32), z["action"].astype(np.float32)
        body = np.load(ROOT / f"work/body/{E}_{key}.npy", mmap_mode="r"); rinfo = json.loads((ROOT / f"work/body/{E}_{key}.json").read_text())
        assert len(body) == len(st) == rec.T
        src_can = np.stack([rec.source_can(rec.robot, t) for t in range(rec.T)])
        valid = (np.abs(st - ep.state.astype(np.float32)).max(1) < 1e-6) & (np.abs(z["can"] - src_can).reshape(rec.T, -1).max(1) < 1e-6)
        T = tcp(st.astype(np.float64)); off = np.zeros((rec.T, 4), np.float32)
        for c, sd in ((0, "L"), (2, "R")):
            off[:, c] = np.linalg.norm(T[sd][:, :3, 3] - tsrc[sd][:, :3, 3], axis=1)
            off[:, c + 1] = np.arccos(np.clip((np.einsum("nij,nij->n", T[sd][:, :3, :3], tsrc[sd][:, :3, :3]) - 1) / 2, -1, 1))
        for i in range(rec.T):
            ds.add_frame({"observation.state": st[i], "action": ac[i], CAMS[0]: np.ascontiguousarray(body[i]),
                          CAMS[1]: ep.frames[CAMS[1]][i], CAMS[2]: ep.frames[CAMS[2]][i],
                          "wrist_view_valid": np.full(2, float(valid[i]), np.float32), "wrist_view_pose_offset": off[i], "task": task})
        ds.save_episode()
        tg = rec.tg
        eps.append({"export_episode_index": n, "synthetic": True, "source_dataset": f"ethanCSL/openarm_insertbattery_lerobot_real_10_v02@{bdata.REV[:8]}",
                    "source_real_episode": E, "variation": key, "battery_shift_m": ck["shift_m"], "n_frames": rec.T,
                    "state_sha256": hashlib.sha256(st.tobytes()).hexdigest(), "action_sha256": hashlib.sha256(ac.tobytes()).hexdigest(),
                    "battery_initial_world_m": ck["battery_initial_world_m"],
                    "right_tcp_at_grasp_world_m": T["R"][tg][:3, 3].tolist(), "source_right_tcp_at_grasp_world_m": tsrc["R"][tg][:3, 3].tolist(),
                    "schedule": ck["marks"], "action_lead_frames": ck["lead"], "replay_checks": {k: ck[k] for k in (
                        "ik_pos_err_mm_max", "ik_rot_err_deg_max", "limit_violations", "speed_ratio_to_real_max", "accel_ratio_to_real_max",
                        "min_mesh_distance_mm", "source_min_mesh_distance_same_frames_mm", "min_hand_distance_mm",
                        "source_min_hand_distance_same_frames_mm", "lowest_arm_z_m", "left_arm_untouched", "start_state_err_rad",
                        "after_rejoin_identical", "placement")},
                    "render": {"rendered_frames": int(sum(1 for r in rinfo if not r.get("copied"))),
                               "copied_frames_after_rejoin": int(sum(1 for r in rinfo if r.get("copied")))},
                    "wrist_valid_fraction": float(valid.mean()),
                    "wrist_offset_max": {"right_trans_cm": round(100 * float(off[:, 2].max()), 2), "right_rot_deg": round(float(np.degrees(off[:, 3].max())), 2)}})
        print(n, E, key, rec.T, flush=True)
    ds.finalize()
    prov = {"name": REPO_ID, "task": task, "contains_real_episodes": False, "physics_used": False, "force_validity": "unknown",
            "source_dataset": "ethanCSL/openarm_insertbattery_lerobot_real_10_v02", "source_revision": bdata.REV,
            "state_semantics": "planned kinematic candidate (not measured)",
            "method": {"trajectory": "object-centric adaptation of the right-arm reach/grasp/lift to a translated battery; "
                                     "rejoins the recording at the lift apex, before the insertion (bsfa/bplan.py)",
                       "body_cam": "MGW mesh-guided forward warping (render_v4/mgw.py) + part-aware hole filling",
                       "wrist_cams": "real source wrist frames at the same time index"},
            "episodes": eps}
    (OUT / "meta/sfa_provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    print("built", ds.meta.total_episodes, ds.meta.total_frames)


if __name__ == "__main__":
    main()
