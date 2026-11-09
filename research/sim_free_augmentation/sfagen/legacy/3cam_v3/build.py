"""v3 three-camera synthetic-only dataset: 55 replay-checked episodes, body_cam re-rendered with the MGW renderer.

body_cam   MGW forward 3-D-flow warp (render_v4/mgw.py), inherited camera, robust robot masks
wrist cams the source recording's real wrist frame at the generator time map round(tau) for every frame (no black);
           wrist_view_valid = 1 where the whole scene equals that source frame; wrist_view_pose_offset per frame
labels     G+ candidate state/action (bit-identical to the v3.1 evidence-complete export; SHA-256 checked)
"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; EXP = HERE.parent
SFA = Path("/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation"); V31 = SFA / "v3_1/outputs"
sys.path.insert(0, str(SFA))
from sfa import data
from sfa.robot import Robot, Camera
REPO_ID = "ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00"
CAMS = ("observation.images.body_cam", "observation.images.wrist_cam", "observation.images.right_wrist_cam")
NAMES = [f"{s}J{j}.pos" for j in range(1, 9) for s in "RL"]
TOL = 1e-6
OUT = HERE / "dataset"


def features():
    v = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    return {"observation.state": {"dtype": "float32", "shape": (16,), "names": NAMES},
            "action": {"dtype": "float32", "shape": (16,), "names": NAMES}, **{c: dict(v) for c in CAMS},
            "wrist_view_valid": {"dtype": "float32", "shape": (2,), "names": ["wrist_cam", "right_wrist_cam"]},
            "wrist_view_pose_offset": {"dtype": "float32", "shape": (4,), "names": ["wrist_cam_trans_m", "wrist_cam_rot_rad",
                                                                                   "right_wrist_cam_trans_m", "right_wrist_cam_rot_rad"]}}


def rot_angle(Ra, Rb):
    return np.arccos(np.clip((np.einsum("nij,nij->n", Ra, Rb) - 1) / 2, -1, 1))


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    assert data.REV == "cebdee98115ab6827a4ec2655335c536b5dd085b"
    robot, cam = Robot(), Camera()
    tcp = lambda S: {s: robot.tcp_traj({"R": Robot.split(S)["R"], "L": Robot.split(S)["L"]}, s) for s in "LR"}  # noqa: E731
    sel = sorted(json.loads((HERE / "selection.json").read_text())["selected"], key=lambda e: (e["E"], e["family"], e["variation"]))
    task = json.loads((EXP / "3cam/dataset/meta/sfa_provenance.json").read_text())["task"]
    if OUT.exists():
        shutil.rmtree(OUT)
    ds = LeRobotDataset.create(repo_id=REPO_ID, fps=30, features=features(), root=OUT, robot_type="openarm_umeow",
                               use_videos=True, streaming_encoding=True)
    src, eps = {}, []
    for k, e in enumerate(sel):
        E = e["E"]
        if E not in src:
            s = data.load(E); src[E] = (s, tcp(s.state.astype(np.float64)))
        s, tsrc = src[E]
        S_src = s.state.astype(np.float32)
        a, b = e["rows"]
        z = np.load(V31 / f"ep{E}/candidates/{e['key']}.npz")
        idn = np.load(V31 / f"ep{E}/candidates/identity__G+__s1.0.npz")
        st, ac = z["state"][a:b].astype(np.float32), z["action"][a:b].astype(np.float32)
        assert hashlib.sha256(st.tobytes()).hexdigest() == e["state_sha256"] and hashlib.sha256(ac.tobytes()).hexdigest() == e["action_sha256"]
        body = np.load(HERE / f"work/body/{E}_{e['key']}.npy")
        rinfo = json.loads((HERE / f"work/body/{E}_{e['key']}.json").read_text())
        assert len(body) == len(st)
        tau = z["tau"][a:b].astype(np.float64)
        ti = np.clip(np.round(tau).astype(int), 0, len(S_src) - 1)
        valid = (np.abs(tau - ti) < 1e-9) & (np.abs(st - S_src[ti]).max(1) < TOL) \
            & (np.abs(z["can"][a:b] - idn["can"][ti]).reshape(b - a, -1).max(1) < TOL)
        T = tcp(st.astype(np.float64)); off = np.zeros((len(st), 4), np.float32)
        for c, sd in ((0, "L"), (2, "R")):
            off[:, c] = np.linalg.norm(T[sd][:, :3, 3] - tsrc[sd][ti][:, :3, 3], axis=1)
            off[:, c + 1] = rot_angle(T[sd][:, :3, :3], tsrc[sd][ti][:, :3, :3])
        for i in range(len(st)):
            ds.add_frame({"observation.state": st[i], "action": ac[i], CAMS[0]: body[i],
                          CAMS[1]: np.ascontiguousarray(s.frames[CAMS[1]][ti[i]]), CAMS[2]: np.ascontiguousarray(s.frames[CAMS[2]][ti[i]]),
                          "wrist_view_valid": np.full(2, float(valid[i]), np.float32), "wrist_view_pose_offset": off[i], "task": task})
        ds.save_episode()
        can0 = z["can"][a][:3, 3]; uv0, _ = cam.project(can0[None])
        robot_px = np.array([r.get("warped_px", 0) for r in rinfo], float)
        holes = np.array([r.get("inpainted_px", 0) for r in rinfo], float)
        eps.append({"export_episode_index": k, "synthetic": True, "source_dataset": "ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98",
                    "source_real_episode": E, "key": e["key"], "variation": e["variation"], "family": e["family"], "scale": e["scale"],
                    "rows": [a, b], "n_frames": int(b - a), "state_sha256": e["state_sha256"], "action_sha256": e["action_sha256"],
                    "can_initial_world_m": can0.tolist(), "can_initial_body_cam_px": uv0[0].tolist(),
                    "source_can_initial_world_m": idn["can"][0][:3, 3].tolist(),
                    "replay_checks": e["replay"], "left_tcp_dev_mm": e["left_tcp_dev_mm"],
                    "render": {"renderer": "MGW forward (render_v4/mgw.py)", "inpainted_px_max": float(holes.max()),
                               "inpainted_px_mean": float(holes.mean()), "warped_px_mean": float(robot_px.mean())},
                    "wrist_source_rows": ti.tolist(), "wrist_valid": valid.astype(int).tolist(),
                    "wrist_offset_max": {"left_trans_cm": round(100 * float(off[:, 0].max()), 2), "left_rot_deg": round(float(np.degrees(off[:, 1].max())), 2),
                                         "right_trans_cm": round(100 * float(off[:, 2].max()), 2), "right_rot_deg": round(float(np.degrees(off[:, 3].max())), 2)}})
        print(k, E, e["variation"], b - a, flush=True)
    ds.finalize()
    prov = {"name": REPO_ID, "task": task, "contains_real_episodes": False,
            "description": "55 synthetic, simulation-free episodes generated from the 10 real source demonstrations of "
                           "ethanCSL/openarm_pringles_lerobot_real_10_v00 (cebdee98): 46 new initial can positions, 9 hand-over timing.",
            "selection_rule": "evidence-complete G+ candidates passing all replay checks with the receiving (left) arm on its recorded trajectory (<= 1 mm)",
            "renderer": {"body_cam": "MGW mesh-guided forward 3-D-flow warp, robust robot masks, multi-frame texture borrowing",
                         "benchmark": "render_v4/bench_heldout.json (7 held-out episodes, 226 real ground-truth pose transfers)"},
            "physics_used": False, "force_validity": "unknown",
            "state_semantics": "planned kinematic candidate (not measured)",
            "previous_revisions": {"v00-35ep-with-real": "0b8d50b6648614bfb047f56959a453a7547c2246",
                                   "v01-25ep-black-wrist": "15d337bf8c192d02a192df94384033f5600c8fff",
                                   "v02-20ep-v3render": "f9692b379c51b2e4ebe091255ebe14aa0d9dd5fe"},
            "episodes": eps}
    (OUT / "meta/sfa_provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    print("built", ds.meta.total_episodes, ds.meta.total_frames)


if __name__ == "__main__":
    main()
