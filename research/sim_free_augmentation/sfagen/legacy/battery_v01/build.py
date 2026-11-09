"""Assemble ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00, revision v01:
  * battery-position episodes ('bat_*'): right-arm reach/grasp/lift follows the moved battery (battery/bsfa/bplan.py),
  * station-position episodes ('st_*'): left-arm reach/grasp/carry follows the moved charging station (target/tplan.py),
every one passing all replay checks; body_cam = MGW render on the calibrated battery-rig camera (+ part-aware fill,
+ per-frame relit plate); wrist cams = the source recording's real wrist frames at the same time index, flagged exact
where the whole scene (all joints + battery + station) equals the recording."""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; T = HERE.parent / "target"
sys.path[:0] = [str(T)]
import srecon  # noqa: E402
import bdata  # noqa: E402
from sfa.robot import Robot  # noqa: E402
REPO_ID = "ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00"
CAMS = bdata.CAMS; NAMES = bdata.NAMES
OUT = HERE / "dataset"
FAM = {"bat": T / "bat2/candidates", "st": T / "candidates_station"}
CHECK_KEYS = ("ik_pos_err_mm_max", "ik_rot_err_deg_max", "limit_violations", "speed_ratio_to_real_max", "accel_ratio_to_real_max",
              "min_mesh_distance_mm", "source_min_mesh_distance_same_frames_mm", "min_hand_distance_mm",
              "source_min_hand_distance_same_frames_mm", "lowest_arm_z_m", "start_state_err_rad", "after_rejoin_identical", "placement")


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
        for fam, cdir in FAM.items():
            p = cdir / f"ep{E}/checks.json"
            if p.exists():
                sel += [(E, fam, k, v) for k, v in sorted(json.loads(p.read_text()).items()) if v["pass"]]
    missing = [(E, k) for E, f, k, v in sel if not (HERE / f"work/body/{E}_{k}.npy").exists()]
    assert not missing, missing
    if OUT.exists():
        shutil.rmtree(OUT)
    ds = LeRobotDataset.create(repo_id=REPO_ID, fps=30, features=features(), root=OUT, robot_type="openarm_umeow",
                               use_videos=True, streaming_encoding=True)
    eps = []; cur = None
    for n, (E, fam, key, ck) in enumerate(sel):
        if cur is None or cur[0] != E:
            rec = srecon.StationRecon(E, device="cuda:0"); ep = rec.ep
            bat = np.stack([srecon.brecon.BRecon.source_can(rec, rec.robot, t) for t in range(rec.T)])
            st_src = np.stack([rec.station_pose(t) for t in range(rec.T)])
            cur = (E, rec, ep, tcp(ep.state), bat, st_src)
        _, rec, ep, tsrc, bat, st_src = cur
        z = np.load(FAM[fam] / f"ep{E}/{key}.npz")
        st, ac = z["state"].astype(np.float32), z["action"].astype(np.float32)
        body = np.load(HERE / f"work/body/{E}_{key}.npy", mmap_mode="r"); rinfo = json.loads((HERE / f"work/body/{E}_{key}.json").read_text())
        assert len(body) == len(st) == rec.T
        obj_src = bat if fam == "bat" else st_src
        valid = (np.abs(st - ep.state.astype(np.float32)).max(1) < 1e-6) & (np.abs(z["can"] - obj_src).reshape(rec.T, -1).max(1) < 1e-6)
        Tn = tcp(st.astype(np.float64)); off = np.zeros((rec.T, 4), np.float32)
        for c, sd in ((0, "L"), (2, "R")):
            off[:, c] = np.linalg.norm(Tn[sd][:, :3, 3] - tsrc[sd][:, :3, 3], axis=1)
            off[:, c + 1] = np.arccos(np.clip((np.einsum("nij,nij->n", Tn[sd][:, :3, :3], tsrc[sd][:, :3, :3]) - 1) / 2, -1, 1))
        for i in range(rec.T):
            ds.add_frame({"observation.state": st[i], "action": ac[i], CAMS[0]: np.ascontiguousarray(body[i]),
                          CAMS[1]: ep.frames[CAMS[1]][i], CAMS[2]: ep.frames[CAMS[2]][i],
                          "wrist_view_valid": np.full(2, float(valid[i]), np.float32), "wrist_view_pose_offset": off[i], "task": task})
        ds.save_episode()
        evs = bdata.grip_events(ep.action); tLg = evs["L"]["closes"][0]; tRg = evs["R"]["closes"][0]; tRo = evs["R"]["opens"][0]
        e = {"export_episode_index": n, "synthetic": True, "source_dataset": f"ethanCSL/openarm_insertbattery_lerobot_real_10_v02@{bdata.REV[:8]}",
             "source_real_episode": E, "family": {"bat": "battery_position", "st": "station_position"}[fam], "variation": key,
             "shift_m": ck["shift_m"], "n_frames": rec.T,
             "state_sha256": hashlib.sha256(st.tobytes()).hexdigest(), "action_sha256": hashlib.sha256(ac.tobytes()).hexdigest(),
             "targets_world_m": {"battery_grasp_right_tcp": Tn["R"][tRg][:3, 3].tolist(), "station_pick_left_tcp": Tn["L"][tLg][:3, 3].tolist(),
                                 "insertion_right_tcp": Tn["R"][tRo][:3, 3].tolist()},
             "source_targets_world_m": {"battery_grasp_right_tcp": tsrc["R"][tRg][:3, 3].tolist(), "station_pick_left_tcp": tsrc["L"][tLg][:3, 3].tolist(),
                                        "insertion_right_tcp": tsrc["R"][tRo][:3, 3].tolist()},
             "schedule": ck["marks"], "action_lead_frames": ck["lead"], "replay_checks": {k: ck[k] for k in CHECK_KEYS if k in ck},
             "render": {"rendered_frames": int(sum(1 for r in rinfo if not r.get("copied"))),
                        "copied_frames_after_rejoin": int(sum(1 for r in rinfo if r.get("copied")))},
             "wrist_valid_fraction": float(valid.mean()),
             "wrist_offset_max": {"left_trans_cm": round(100 * float(off[:, 0].max()), 2), "right_trans_cm": round(100 * float(off[:, 2].max()), 2)}}
        if fam == "bat":
            e["battery_initial_world_m"] = ck["battery_initial_world_m"]
        else:
            e["station_centre_initial_world_m"] = ck["station_centre_initial_world_m"]
        eps.append(e)
        print(n, E, key, rec.T, flush=True)
    ds.finalize()
    prov = {"name": REPO_ID, "task": task, "contains_real_episodes": False, "physics_used": False, "force_validity": "unknown",
            "source_dataset": "ethanCSL/openarm_insertbattery_lerobot_real_10_v02", "source_revision": bdata.REV,
            "state_semantics": "planned kinematic candidate (not measured)",
            "camera": json.loads((T / "calib_battery.json").read_text()) | {"method": "render-and-compare fit, target/calib_battery.py"},
            "method": {"trajectory": "object-centric segment transform of ONE arm toward a translated object (battery: right arm; "
                                     "station: left arm), rejoining the recording before the bimanual insertion",
                       "body_cam": "MGW mesh-guided forward warping (render_v4/mgw.py), part-aware fill, PLATE_ADAPT=10, calibrated camera",
                       "wrist_cams": "real source wrist frames at the same time index"},
            "previous_revisions": {"v00-36ep-inherited-cam": "b41a6ef6bb868c18bf42dd2b6cdd497eced7ffbe"},
            "episodes": eps}
    (OUT / "meta/sfa_provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    print("built", ds.meta.total_episodes, ds.meta.total_frames)


if __name__ == "__main__":
    main()
