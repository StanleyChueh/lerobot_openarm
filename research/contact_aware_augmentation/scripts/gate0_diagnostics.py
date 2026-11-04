"""Gate 0 read-only diagnostics for "Contact-Aware Generative Augmentation for Few-Shot VLA Learning".

Reads LeRobot dataset folders only. Writes only into --out. No network, no GPU, no robot, no CAN, no training.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 .venv/bin/python \
        research/contact_aware_augmentation/scripts/gate0_diagnostics.py \
        --pringles-root <real_10_v00 snapshot> --pool-root <real_v00 snapshot> --battery-root <insertbattery_v02 copy> \
        --annotations research/dataset_diagnostics/annotations/real_v00.csv \
        --hub-meta research/contact_aware_augmentation/outputs/hub_metadata.json \
        --out research/contact_aware_augmentation/outputs

Reuses tools/dataset_diagnostics (loader, gripper endpoints, openness) and the follower URDF used by its FK
grasp proxy. Every number written here is computed from the files; nothing is estimated force or torque.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from tools.dataset_diagnostics.loader import DatasetView, load_dataset  # noqa: E402
from tools.dataset_diagnostics.openarm import DEFAULT_URDF, gripper_endpoints, to_openness  # noqa: E402

ARM = [f"{s}J{n}.pos" for n in range(1, 8) for s in ("R", "L")]
SIGNAL_KEYWORDS = ("torque", "tau", "effort", "current", "force", "wrench", "vel", "temp", "telemetry", "contact")

# body_cam: extrinsic FITTED by IsaacLab scripts/tools/sysid/calibrate_camera.py (robot-silhouette chamfer on
# real_v00 frames, IoU 0.33 -> 0.71), as copied into stack_ik_abs_visuomotor_env_cfg.py (IsaacLab 4aa5e026):
# parent openarm_body_link0 (= URDF world, identity joint), ROS optical convention, quaternion (w, x, y, z).
# Intrinsics: fx = fy = 607.22 (rs-enumerate-devices), real principal point (326.4, 238.7) (REAL_DATA_AUDIT.md §3).
BODY_CAM = {"parent": "openarm_body_link0", "pos": [0.05621, -0.00305, 0.83062],
            "quat_wxyz": [0.18720, -0.68244, 0.69543, -0.12490], "fx": 607.22, "fy": 607.22, "cx": 326.4, "cy": 238.7,
            "source": "IsaacLab@4aa5e026 stack_ik_abs_visuomotor_env_cfg.py body_cam OffsetCfg (FITTED, unvalidated "
                      "on held-out keypoints)"}
FK_FRAMES = [f"openarm_{s}_{l}" for s in ("right", "left") for l in ("link1", "link2", "link3", "link4", "link5",
                                                                       "link6", "link7", "hand_tcp")]


# ------------------------------------------------------------------------------------------------ helpers
def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_sha(path: Path) -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True).stdout.strip()
    return {"sha": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(points_parent: np.ndarray, cam: dict = BODY_CAM) -> tuple[np.ndarray, np.ndarray]:
    """(N,3) points in the camera's parent frame -> (N,2) pixels and (N,) depth along the optical axis."""
    R, t = quat_wxyz_to_R(cam["quat_wxyz"]), np.asarray(cam["pos"], float)
    c = (np.atleast_2d(points_parent) - t) @ R  # = R^T (p - t), row form
    z = c[:, 2]
    uv = np.stack([cam["fx"] * c[:, 0] / z + cam["cx"], cam["fy"] * c[:, 1] / z + cam["cy"]], axis=1)
    return uv, z


def best_lag(actions: dict[int, np.ndarray], states: dict[int, np.ndarray], cols: list[int], kmax: int = 5) -> dict:
    """Mean |action[t] - state[t+k]| over arm joints for k = 0..kmax (the lead of the command over the state)."""
    err = {}
    for k in range(kmax + 1):
        vals = [np.abs(a[: len(a) - k, cols] - s[k:, cols]).mean() for e in actions
                for a, s in [(actions[e], states[e])] if len(a) > k]
        err[k] = float(np.mean(vals))
    k_best = min(err, key=err.get)
    return {"k_best": k_best, "mean_abs_err_rad": {str(k): round(v, 5) for k, v in err.items()}}


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def gripper_events(a_open: np.ndarray, s_open: np.ndarray, settle: int = 15, empty_thresh: float = 0.12) -> dict:
    """Commanded close/open instants (action openness < 0.5) and the measured stall level while closed.

    `stall` = median measured openness from `settle` frames after the close command to the open command. A stall
    clearly above the empty-close level means the jaws were blocked by something (geometric grasp evidence);
    it is NOT a force measurement."""
    closed = a_open < 0.5
    segs = segments(closed)
    out = {"n_closes": len(segs), "closes": [], "first_close": None, "first_open_after_close": None}
    for s0, s1 in segs:
        w = s_open[min(s0 + settle, s1 - 1): s1] if s1 - s0 > settle else s_open[s0:s1]
        stall = float(np.median(w)) if len(w) else float("nan")
        out["closes"].append({"close": int(s0), "open": int(s1) if s1 < len(a_open) else None, "frames": int(s1 - s0),
                              "stall_openness": round(stall, 4), "blocked": bool(stall > empty_thresh)})
    if segs:
        out["first_close"] = int(segs[0][0])
        out["first_open_after_close"] = int(segs[0][1]) if segs[0][1] < len(a_open) else None
    return out


class FK:
    def __init__(self, view: DatasetView, urdf: Path = DEFAULT_URDF):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        names = view.names("observation.state")
        self.map = [(names.index(f"{p}J{n}.pos"), self.model.idx_qs[self.model.getJointId(f"openarm_{s}_joint{n}")])
                    for n in range(1, 8) for s, p in (("right", "R"), ("left", "L"))]
        self.fid = {f: self.model.getFrameId(f) for f in FK_FRAMES}

    def frames(self, state: np.ndarray) -> dict[str, np.ndarray]:
        """state (D,) -> {frame: 4x4 pose in URDF world}."""
        q = self.pin.neutral(self.model)
        for col, qi in self.map:
            q[qi] = state[col]
        self.pin.framesForwardKinematics(self.model, self.data, q)
        return {f: self.data.oMf[i].homogeneous.copy() for f, i in self.fid.items()}


# ------------------------------------------------------------------------------------------------ sections
def provenance(view: DatasetView, label: str, hash_videos: bool) -> dict:
    files = {}
    for p in sorted(view.root.rglob("*")):
        if p.is_file() and ".cache" not in p.relative_to(view.root).parts and p.suffix in (".json", ".parquet", ".mp4", ".md"):
            rel = str(p.relative_to(view.root))
            if p.suffix == ".mp4" and not hash_videos:
                files[rel] = {"bytes": p.stat().st_size, "sha256": None}
            else:
                files[rel] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    feats = {k: {"dtype": v["dtype"], "shape": v["shape"], "names": v.get("names"),
                 **({"codec": v["info"].get("video.codec"), "fps": v["info"].get("video.fps"),
                     "gop": v["info"].get("video.g"), "crf": v["info"].get("video.crf")}
                    if v["dtype"] == "video" else {})} for k, v in view.features.items()}
    tasks = []
    tp = view.root / "meta" / "tasks.parquet"
    if tp.exists():
        import pandas as pd
        tasks = [str(t) for t in pd.read_parquet(tp).index]
    return {"label": label, "root": str(view.root), "codebase_version": view.version,
            "robot_type": view.info.get("robot_type"), "fps": view.fps, "tasks": tasks,
            "total_episodes_info": view.info.get("total_episodes"), "total_frames_info": view.info.get("total_frames"),
            "episodes_in_metadata": int(len(view.episodes)), "frames_in_data": int(len(view.data)),
            "features": feats, "files": files}


def timing(view: DatasetView, decode_videos: bool) -> dict:
    d, fps = view.data, view.fps
    out = {"fps_nominal": fps}
    out["timestamp_minus_frame_index_over_fps_max_abs_s"] = float(np.abs(d["timestamp"] - d["frame_index"] / fps).max())
    contiguous, lengths = True, {}
    for e, g in d.groupby("episode_index"):
        fi = g["frame_index"].to_numpy()
        contiguous &= bool((np.diff(fi) == 1).all() and fi[0] == 0)
        lengths[int(e)] = len(g)
    out["frame_index_contiguous_from_0"] = contiguous
    out["global_index_contiguous"] = bool((np.diff(d["index"].to_numpy()) == 1).all())
    meta_len = view.episodes["length"].astype(int).to_dict()
    out["metadata_length_equals_rows"] = all(int(meta_len[e]) == n for e, n in lengths.items())
    out["episode_length_frames"] = {"min": min(lengths.values()), "median": float(np.median(list(lengths.values()))),
                                    "max": max(lengths.values()), "total": sum(lengths.values())}
    out["episode_length_s_nominal"] = {k: round(v / fps, 2) for k, v in out["episode_length_frames"].items()
                                       if k != "total"}
    cams = {}
    for cam in view.camera_keys:
        span_err = []
        for e in lengths:
            m = view.episodes.loc[e]
            span = float(m[f"videos/{cam}/to_timestamp"]) - float(m[f"videos/{cam}/from_timestamp"])
            span_err.append(abs(span - lengths[e] / fps))
        c = {"video_span_minus_length_over_fps_max_abs_s": round(float(max(span_err)), 6)}
        files = sorted({view.video_path(cam, e)[0] for e in lengths})
        present = [f for f in files if (view.root / f).exists()]
        c["files"] = files
        c["files_local"] = len(present)
        if decode_videos and len(present) == len(files):
            import av
            n_pkt, w_h = 0, None
            for f in present:
                with av.open(str(view.root / f)) as cont:
                    st = cont.streams.video[0]
                    w_h = (st.codec_context.width, st.codec_context.height)
                    c.setdefault("codec", st.codec_context.name)
                    n_pkt += sum(1 for p in cont.demux(st) if p.size > 0)
            c["video_packets_total"] = n_pkt
            c["video_packets_equal_frames"] = n_pkt == sum(lengths.values())
            c["width_height"] = w_h
        else:
            c["video_packets_total"] = None
            c["note"] = "video files not on disk; only Hub metadata checked" if len(present) < len(files) else ""
        cams[cam] = c
    out["cameras"] = cams
    out["measured_loop_rate"] = ("UNKNOWN: timestamps are nominal frame_index/fps; no DIAG/telemetry log of the "
                                 "recording sessions was found on this machine")
    return out


def signals(view: DatasetView) -> dict:
    cols = list(view.data.columns) + [c for c in view.features if c not in view.data.columns]
    names = {k: v.get("names") for k, v in view.features.items() if v["dtype"] in ("float32", "float64")}
    hits = [c for c in cols if any(w in c.lower() for w in SIGNAL_KEYWORDS)]
    name_hits = {k: [n for n in (v or []) if any(w in n.lower() for w in SIGNAL_KEYWORDS)] for k, v in names.items()}
    vec_names = sorted({n for v in names.values() for n in (v or [])})
    return {"columns": cols, "vector_feature_names": names, "keyword_hits_in_columns": hits,
            "keyword_hits_in_vector_names": {k: v for k, v in name_hits.items() if v},
            "all_vector_names_are_pos": all(n.endswith(".pos") for n in vec_names),
            "verdict": ("joint positions only (16 x .pos state, 16 x .pos action) + RGB video; no torque, current, "
                        "velocity, temperature, force or telemetry field") if not hits and all(
                n.endswith(".pos") for n in vec_names) else "CHECK: non-position signals present"}


def action_semantics(view: DatasetView) -> dict:
    names = view.names("action")
    assert names == view.names("observation.state"), "action and state names differ"
    acts, sts = view.vectors("action"), view.vectors("observation.state")
    cols = [names.index(n) for n in ARM]
    out = {"names_order": names, "action_names_equal_state_names": True, "lag": best_lag(acts, sts, cols)}
    grip = {}
    for g in ("RJ8.pos", "LJ8.pos"):
        o, c = gripper_endpoints(view, g)
        j = names.index(g)
        allv = np.concatenate([a[:, j] for a in acts.values()])
        at_end = np.isclose(allv, o, atol=1e-3) | np.isclose(allv, c, atol=1e-3)
        sv = np.concatenate([s[:, j] for s in sts.values()])
        grip[g] = {"open_cmd": round(o, 4), "closed_cmd": round(c, 4), "frac_frames_at_endpoint": round(float(at_end.mean()), 4),
                   "distinct_action_values": int(len(np.unique(allv.round(4)))),
                   "state_range": [round(float(sv.min()), 4), round(float(sv.max()), 4)]}
    out["grippers"] = grip
    return out


def episode_events(view: DatasetView) -> list[dict]:
    names = view.names("action")
    acts, sts = view.vectors("action"), view.vectors("observation.state")
    ep_ = {g: gripper_endpoints(view, g) for g in ("RJ8.pos", "LJ8.pos")}
    rows = []
    for e in sorted(acts):
        row = {"episode": e, "length": len(acts[e])}
        for g, side in (("RJ8.pos", "right"), ("LJ8.pos", "left")):
            j = names.index(g)
            ev = gripper_events(to_openness(acts[e][:, j], *ep_[g]), to_openness(sts[e][:, j], *ep_[g]))
            row[f"{side}_n_closes"] = ev["n_closes"]
            row[f"{side}_close"] = ev["first_close"]
            row[f"{side}_open"] = ev["first_open_after_close"]
            row[f"{side}_stall_openness"] = ev["closes"][0]["stall_openness"] if ev["closes"] else None
            row[f"{side}_blocked"] = ev["closes"][0]["blocked"] if ev["closes"] else None
            second = ev["closes"][1] if len(ev["closes"]) > 1 else None
            row[f"{side}_close2"] = second["close"] if second else None
            row[f"{side}_open2"] = second["open"] if second else None
        rows.append(row)
    return rows


def tracking_residual(view: DatasetView, events: list[dict], task: str, lag: int) -> list[dict]:
    """Kinematic residual a[t] - s[t+lag] per arm in phase windows, with the joint speed in the same window.

    This is a POSITION tracking residual. Under the MIT PD loop it is related to motor effort only through unknown
    gains on the clamped (unrecorded) command, gravity feed-forward and friction; it is not force."""
    names = view.names("action")
    acts, sts = view.vectors("action"), view.vectors("observation.state")
    fps = view.fps
    rows = []
    for ev in events:
        e = ev["episode"]
        a, s = acts[e], sts[e]
        T = len(a)
        res = a[: T - lag] - s[lag:]
        spd = np.vstack([np.zeros((1, s.shape[1])), np.diff(s, axis=0)])[lag:] * fps
        rc, lc, ro, lo = ev["right_close"], ev["left_close"], ev["right_open"], ev["left_open"]
        if task == "pringles":  # right grasps, hands over to left
            win = {"free_approach": (15, rc), "right_carry": (rc + 15 if rc else None, lc - 10 if lc else None),
                   "handover": (lc - 10 if lc else None, ro + 5 if ro else None)}
        else:  # battery: left lifts charger, right inserts battery, right releases
            first = min(x for x in (rc, lc) if x is not None) if (rc or lc) else None
            win = {"free_approach": (15, first), "bimanual_carry": (max(rc or 0, lc or 0) + 15, ro - 30 if ro else None),
                   "pre_release_1s": (ro - 30 if ro else None, ro)}
        for name, (w0, w1) in win.items():
            if w0 is None or w1 is None or w1 - w0 < 5:
                continue
            w1 = min(w1, T - lag)
            for side, p in (("right", "R"), ("left", "L")):
                cols = [names.index(f"{p}J{n}.pos") for n in range(1, 8)]
                r, v = res[w0:w1][:, cols], spd[w0:w1][:, cols]
                rows.append({"episode": e, "window": name, "side": side, "frames": int(w1 - w0),
                             "res_rms_rad": round(float(np.sqrt((r ** 2).mean())), 5),
                             "speed_rms_rad_s": round(float(np.sqrt((v ** 2).mean())), 4),
                             "res_over_speed_s": round(float(np.sqrt((r ** 2).mean()) / max(np.sqrt((v ** 2).mean()), 1e-6)), 4)})
    return rows


def right_close_segments(view: DatasetView) -> dict[int, list[tuple[int, int]]]:
    names = view.names("action")
    o, c = gripper_endpoints(view, "RJ8.pos")
    j = names.index("RJ8.pos")
    return {e: [(int(a), int(b)) for a, b in segments(to_openness(x[:, j], o, c) < 0.5)]
            for e, x in view.vectors("action").items()}


def event_aligned_residual(view: DatasetView, anchors: dict[int, int], lag: int, half: int = 60) -> dict:
    """Per-arm norm over J1..J7 of a[t] - s[t+lag], and of the joint speed, aligned on an event frame; median and
    IQR across episodes for each offset in [-half, half]. Used to ask: does a contact event leave any kinematic
    trace beyond what the motion itself explains?"""
    names = view.names("action")
    acts, sts = view.vectors("action"), view.vectors("observation.state")
    fps = view.fps
    offs = np.arange(-half, half + 1)
    out = {"offsets": offs.tolist(), "n": 0}
    for side, p in (("right", "R"), ("left", "L")):
        cols = [names.index(f"{p}J{n}.pos") for n in range(1, 8)]
        R, V = [], []
        for e, k in anchors.items():
            a, s = acts[e], sts[e]
            if k - half < 1 or k + half + lag >= len(a):
                continue
            idx = k + offs
            R.append(np.linalg.norm(a[idx][:, cols] - s[idx + lag][:, cols], axis=1))
            V.append(np.linalg.norm((s[idx][:, cols] - s[idx - 1][:, cols]) * fps, axis=1))
        R, V = np.array(R), np.array(V)
        out["n"] = len(R)
        out[side] = {"res_norm_median": np.median(R, 0).round(4).tolist(), "res_norm_q25": np.percentile(R, 25, 0).round(4).tolist(),
                     "res_norm_q75": np.percentile(R, 75, 0).round(4).tolist(), "speed_norm_median": np.median(V, 0).round(4).tolist()}
        p0, p1 = max(half - 60, 0), max(half - 10, 1)
        pre = R[:, p0:p1].mean(1)  # [-2.0, -0.33) s at 30 fps, clipped to the window
        res_s = {}
        for a0, a1 in ((0, 30), (30, 60), (60, half + 1)):  # [0,1) s, [1,2) s, [2, half/fps] s
            a1 = min(a1, half + 1)
            if a1 - a0 < 5:
                continue
            post = R[:, half + a0: half + a1].mean(1)
            vpost = V[:, half + a0: half + a1].mean(1)
            res_s[f"post_{a0}_{a1}"] = {"res_median": round(float(np.median(post)), 4), "frac_eps_gt_pre": round(float((post > pre).mean()), 3),
                                        "speed_median": round(float(np.median(vpost)), 4)}
        out[side]["pre_mean_vs_post_mean"] = {"pre_res_median": round(float(np.median(pre)), 4),
                                              "pre_speed_median": round(float(np.median(V[:, p0:p1].mean(1))), 4), **res_s}
    return out


def fk_geometry(view: DatasetView, events: list[dict], task: str) -> list[dict]:
    """Bimanual geometry from FK only: TCP positions and the right TCP expressed in the left TCP frame at the
    contact-relevant instant (pringles: left close = take-over; battery: right open = release after insertion)."""
    fk = FK(view)
    sts = view.vectors("observation.state")
    rows = []
    for ev in events:
        e = ev["episode"]
        k = ev["left_close"] if task == "pringles" else ev["right_open"]
        if k is None:
            continue
        k = min(k, len(sts[e]) - 1)
        F = fk.frames(sts[e][k])
        Tr, Tl = F["openarm_right_hand_tcp"], F["openarm_left_hand_tcp"]
        rel = np.linalg.inv(Tl) @ Tr
        ang = np.degrees(np.arccos(np.clip((np.trace(rel[:3, :3]) - 1) / 2, -1, 1)))
        rows.append({"episode": e, "frame": int(k), "event": "left_close" if task == "pringles" else "right_open",
                     "right_tcp_xyz": [round(float(x), 4) for x in Tr[:3, 3]],
                     "left_tcp_xyz": [round(float(x), 4) for x in Tl[:3, 3]],
                     "tcp_distance_m": round(float(np.linalg.norm(Tr[:3, 3] - Tl[:3, 3])), 4),
                     "right_in_left_xyz": [round(float(x), 4) for x in rel[:3, 3]],
                     "right_in_left_angle_deg": round(float(ang), 2)})
    return rows


def summarize_geometry(rows: list[dict]) -> dict:
    if not rows:
        return {}
    rel = np.array([r["right_in_left_xyz"] for r in rows])
    d = np.array([r["tcp_distance_m"] for r in rows])
    a = np.array([r["right_in_left_angle_deg"] for r in rows])
    return {"n": len(rows), "tcp_distance_m": {"median": round(float(np.median(d)), 4), "min": round(float(d.min()), 4),
                                                "max": round(float(d.max()), 4)},
            "right_in_left_xyz_std_m": [round(float(x), 4) for x in rel.std(0)],
            "right_in_left_xyz_mean_m": [round(float(x), 4) for x in rel.mean(0)],
            "right_in_left_angle_deg": {"median": round(float(np.median(a)), 2), "min": round(float(a.min()), 2),
                                        "max": round(float(a.max()), 2)}}


def fk_vs_annotation(view: DatasetView, events: list[dict], ann_csv: Path, ann_label: str) -> tuple[list[dict], dict]:
    """Projected right TCP at the first commanded close vs the human-clicked can position on body_cam frame 0.

    The can is pushed ~2 cm before the grasp (sim) and the clicked point is the lid/body (varies), so this is a
    combined error bound (calibration + FK + push + click reference), NOT a calibration validation."""
    ann = {int(r["episode_index"]): (float(r["u"]), float(r["v"])) for r in csv.DictReader(open(ann_csv))
           if r["dataset"] == ann_label and r["verified"] == "1"}
    fk = FK(view)
    sts = view.vectors("observation.state")
    rows = []
    for ev in events:
        e, k = ev["episode"], ev["right_close"]
        if k is None or e not in ann:
            continue
        p = fk.frames(sts[e][k])["openarm_right_hand_tcp"][:3, 3]
        uv, z = project(p[None])
        du, dv = uv[0, 0] - ann[e][0], uv[0, 1] - ann[e][1]
        rows.append({"episode": e, "grasp_frame": k, "proj_u": round(float(uv[0, 0]), 1), "proj_v": round(float(uv[0, 1]), 1),
                     "depth_m": round(float(z[0]), 3), "ann_u": ann[e][0], "ann_v": ann[e][1],
                     "du_px": round(float(du), 1), "dv_px": round(float(dv), 1), "err_px": round(float(np.hypot(du, dv)), 1)})
    err = np.array([r["err_px"] for r in rows])
    du = np.array([r["du_px"] for r in rows])
    dv = np.array([r["dv_px"] for r in rows])
    summ = {"n": len(rows), "err_px": {"median": round(float(np.median(err)), 1), "p90": round(float(np.percentile(err, 90)), 1),
                                       "max": round(float(err.max()), 1)},
            "bias_px": {"du_median": round(float(np.median(du)), 1), "dv_median": round(float(np.median(dv)), 1)},
            "residual_after_median_bias_px": {
                "median": round(float(np.median(np.hypot(du - np.median(du), dv - np.median(dv)))), 1),
                "p90": round(float(np.percentile(np.hypot(du - np.median(du), dv - np.median(dv)), 90)), 1)},
            "corr_proj_vs_ann": {"u": round(float(np.corrcoef([r["proj_u"] for r in rows], [r["ann_u"] for r in rows])[0, 1]), 3),
                                 "v": round(float(np.corrcoef([r["proj_v"] for r in rows], [r["ann_v"] for r in rows])[0, 1]), 3)}} if rows else {}
    return rows, summ


def overlays(view: DatasetView, events: list[dict], episodes: list[int], out_dir: Path) -> list[str]:
    """Contact sheets: body_cam key frames with the FK skeleton projected (red = right arm, blue = left arm)."""
    import cv2
    fk = FK(view)
    sts = view.vectors("observation.state")
    written = []
    for e in episodes:
        ev = next(x for x in events if x["episode"] == e)
        T = len(sts[e])
        keys = [("start", 0), ("pre-grasp", max(ev["right_close"] - 20, 0)), ("R close", ev["right_close"]),
                ("L close", ev["left_close"]), ("R open", ev["right_open"]), ("end", T - 1)]
        tiles = []
        for lab, k in keys:
            if k is None:
                continue
            img = np.ascontiguousarray(view.read_frame("observation.images.body_cam", e, int(k))[..., ::-1])
            F = fk.frames(sts[e][k])
            for side, col in (("right", (0, 0, 255)), ("left", (255, 0, 0))):
                pts = np.array([F[f"openarm_{side}_{l}"][:3, 3] for l in ("link2", "link3", "link4", "link5", "link6",
                                                                           "link7", "hand_tcp")])
                uv, z = project(pts)
                uv = uv[z > 0.05]
                for i in range(len(uv) - 1):
                    cv2.line(img, tuple(map(int, uv[i])), tuple(map(int, uv[i + 1])), col, 2, cv2.LINE_AA)
                if len(uv):
                    cv2.circle(img, tuple(map(int, uv[-1])), 7, col, 2, cv2.LINE_AA)
            cv2.putText(img, f"ep{e} f{k} {lab}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(cv2.resize(img, (320, 240)))
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        p = out_dir / f"fk_overlay_body_cam_ep{e}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        written.append(str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p))
    return written


def battery_sheets(view: DatasetView, events: list[dict], segs: dict, episodes: list[int], out_dir: Path) -> list[str]:
    """Battery body_cam key frames at the gripper events (visual check of the phase interpretation); TCPs projected."""
    import cv2
    fk = FK(view)
    sts = view.vectors("observation.state")
    written = []
    for e in episodes:
        rel, _ = view.video_path("observation.images.body_cam", e)
        if not (view.root / rel).exists() or len(segs.get(e, [])) < 2:
            continue
        ev = next(x for x in events if x["episode"] == e)
        (c1, o1), (c2, o2) = segs[e][:2]
        T = len(sts[e])
        keys = [("L close", ev["left_close"]), ("R close1", c1), ("R open1-10", o1 - 10), ("R open1", o1), ("R close2", c2),
                ("R close2+20", c2 + 20), ("R open2", o2), ("L open", ev["left_open"] or T - 1), ("end", T - 1)]
        tiles = []
        for lab, k in keys:
            k = min(int(k), T - 1)
            img = np.ascontiguousarray(view.read_frame("observation.images.body_cam", e, k)[..., ::-1])
            F = fk.frames(sts[e][k])
            for side, col in (("right", (0, 0, 255)), ("left", (255, 0, 0))):
                uv, _ = project(F[f"openarm_{side}_hand_tcp"][:3, 3][None])
                cv2.circle(img, tuple(map(int, uv[0])), 7, col, 2, cv2.LINE_AA)
            cv2.putText(img, f"ep{e} f{k} {lab}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(cv2.resize(img, (320, 240)))
        sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)])
        p = out_dir / f"battery_phases_body_cam_ep{e}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        written.append(str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p))
    return written


def same_as_pool(real10: DatasetView, pool: DatasetView) -> dict:
    a, b = real10.vectors("observation.state"), pool.vectors("observation.state")
    eq = {e: bool(e in b and a[e].shape == b[e].shape and np.array_equal(a[e], b[e])) for e in a}
    return {"real_10_episode_equals_pool_episode_same_index": eq, "all": all(eq.values())}


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pringles-root", required=True)
    ap.add_argument("--pool-root", default=None)
    ap.add_argument("--battery-root", required=True)
    ap.add_argument("--annotations", default=str(REPO / "research/dataset_diagnostics/annotations/real_v00.csv"))
    ap.add_argument("--hub-meta", default=None, help="JSON written by hub_metadata.py (optional)")
    ap.add_argument("--poc-episodes", default="8,0,4")
    ap.add_argument("--battery-sheet-episodes", default="0,7", help="drawn only if their body_cam video is on disk")
    ap.add_argument("--no-video-decode", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    P = load_dataset(root=args.pringles_root, label="pringles_real_10_v00")
    B = load_dataset(root=args.battery_root, label="insertbattery_real_v02")
    pool = load_dataset(root=args.pool_root, label="pringles_real_v00_pool") if args.pool_root else None

    summary = {"generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
               "script": str(Path(__file__).relative_to(REPO)), "repos": {"lerobot_openarm": git_sha(REPO)},
               "body_cam": BODY_CAM, "urdf": str(DEFAULT_URDF.relative_to(REPO)), "urdf_sha256": sha256(DEFAULT_URDF)}
    for name, path in (("lerobot", REPO.parent / "lerobot"), ("IsaacLab", REPO.parent / "IsaacLab")):
        if (path / ".git").exists():
            summary["repos"][name] = git_sha(path)

    prov = {"pringles_real_10_v00": provenance(P, "pringles_real_10_v00", hash_videos=True),
            "insertbattery_real_v02": provenance(B, "insertbattery_real_v02", hash_videos=False)}
    if pool is not None:
        prov["pringles_real_v00_pool"] = provenance(pool, "pringles_real_v00_pool", hash_videos=False)
    if args.hub_meta:
        hub = json.loads(Path(args.hub_meta).read_text())
        for k, v in prov.items():
            h = hub.get(v.get("hub_repo_id") or {"pringles_real_10_v00": "ethanCSL/openarm_pringles_lerobot_real_10_v00",
                                                  "insertbattery_real_v02": "ethanCSL/openarm_insertbattery_lerobot_real_v02",
                                                  "pringles_real_v00_pool": "ethanCSL/openarm_pringles_lerobot_real_v00"}[k])
            if h:
                v["hub"] = {kk: vv for kk, vv in h.items() if kk != "files"}
                mism = []
                for rel, f in v["files"].items():
                    hf = h["files"].get(rel)
                    if hf is None:
                        mism.append(f"{rel}: not on Hub")
                    elif hf.get("size") != f["bytes"] or (f["sha256"] and hf.get("lfs_sha256") and hf["lfs_sha256"] != f["sha256"]):
                        mism.append(f"{rel}: differs from Hub")
                v["local_vs_hub"] = {"files_compared": len(v["files"]), "mismatches": mism,
                                     "hub_files_not_local": sorted(set(h["files"]) - set(v["files"]))}
    (out / "provenance_local.json").write_text(json.dumps(prov, indent=1))

    summary["timing"] = {"pringles_real_10_v00": timing(P, not args.no_video_decode),
                         "insertbattery_real_v02": timing(B, not args.no_video_decode)}
    summary["signals"] = {"pringles_real_10_v00": signals(P), "insertbattery_real_v02": signals(B)}
    summary["action_semantics"] = {"pringles_real_10_v00": action_semantics(P), "insertbattery_real_v02": action_semantics(B)}
    if pool is not None:
        summary["real_10_vs_pool"] = same_as_pool(P, pool)

    evP, evB = episode_events(P), episode_events(B)
    write_csv(evP, out / "events_pringles_real_10.csv")
    write_csv(evB, out / "events_battery_v02.csv")

    def ev_summary(rows, keys):
        s = {}
        for k in keys:
            v = [r[k] for r in rows if r[k] is not None]
            s[k] = {"n": len(v), "min": min(v), "median": float(np.median(v)), "max": max(v)} if v else None
        s["episodes_with_single_close_both_arms"] = sum(r["right_n_closes"] == 1 and r["left_n_closes"] == 1 for r in rows)
        s["episodes"] = len(rows)
        s["right_blocked"] = sum(bool(r["right_blocked"]) for r in rows)
        s["left_blocked"] = sum(bool(r["left_blocked"]) for r in rows)
        return s
    keys = ["right_close", "right_open", "left_close", "left_open", "right_stall_openness", "left_stall_openness"]
    summary["events"] = {"pringles_real_10_v00": ev_summary(evP, keys), "insertbattery_real_v02": ev_summary(evB, keys)}
    if all(r["left_close"] is not None and r["right_open"] is not None for r in evP):
        ov = [r["right_open"] - r["left_close"] for r in evP]
        summary["events"]["pringles_real_10_v00"]["cmd_overlap_frames(right_open-left_close)"] = {"min": min(ov), "median": float(np.median(ov)), "max": max(ov)}
    order = [("left_close<right_close" if (r["left_close"] or 1e9) < (r["right_close"] or 1e9) else "right_close<=left_close") for r in evB]
    summary["events"]["insertbattery_real_v02"]["close_order"] = {o: order.count(o) for o in set(order)}

    lagP = summary["action_semantics"]["pringles_real_10_v00"]["lag"]["k_best"]
    lagB = summary["action_semantics"]["insertbattery_real_v02"]["lag"]["k_best"]
    tr = tracking_residual(P, evP, "pringles", lagP) + [dict(r, dataset="battery") for r in tracking_residual(B, evB, "battery", lagB)]
    for r in tr:
        r.setdefault("dataset", "pringles")
    write_csv(tr, out / "tracking_residual_windows.csv")
    agg = {}
    for r in tr:
        agg.setdefault((r["dataset"], r["window"], r["side"]), []).append(r)
    summary["tracking_residual"] = {f"{d}/{w}/{s}": {"n": len(v), "res_rms_rad_median": round(float(np.median([x["res_rms_rad"] for x in v])), 5),
                                                     "speed_rms_median": round(float(np.median([x["speed_rms_rad_s"] for x in v])), 4),
                                                     "res_over_speed_median_s": round(float(np.median([x["res_over_speed_s"] for x in v])), 4)}
                                    for (d, w, s), v in sorted(agg.items())}

    segB = right_close_segments(B)
    summary["battery_right_gripper_structure"] = {
        "n_close_segments_distribution": {str(k): sum(len(v) == k for v in segB.values()) for k in sorted({len(v) for v in segB.values()})},
        "interpretation": ("close1 = grasp the AA cell, open1 = release over the bay, close2 = re-close with the cell "
                           "no longer in hand (pressing/seating, SUPPORTED by visual check of episodes 0 and 7 only), "
                           "open2 = release at the end")}
    aligned = {"pringles@left_close": event_aligned_residual(P, {r["episode"]: r["left_close"] for r in evP}, lagP),
               "battery@right_open1": event_aligned_residual(B, {e: v[0][1] for e, v in segB.items() if len(v) >= 2}, lagB, half=120),
               "battery@right_close2": event_aligned_residual(B, {e: v[1][0] for e, v in segB.items() if len(v) >= 2}, lagB, half=120)}
    (out / "event_aligned_residual.json").write_text(json.dumps(aligned))
    summary["event_aligned_residual"] = {k: {s: v[s]["pre_mean_vs_post_mean"] for s in ("right", "left")} | {"n": v["n"]}
                                         for k, v in aligned.items()}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=False)
        for ax, (k, v) in zip(axes, aligned.items()):
            x = np.array(v["offsets"]) / P.fps
            for side, col in (("right", "#c0392b"), ("left", "#2e6da4")):
                ax.plot(x, v[side]["res_norm_median"], color=col, label=f"{side} |a(t)-s(t+lag)|")
                ax.fill_between(x, v[side]["res_norm_q25"], v[side]["res_norm_q75"], color=col, alpha=0.15)
                ax.plot(x, np.array(v[side]["speed_norm_median"]) * 0.05, color=col, ls=":", label=f"{side} speed x0.05")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_title(f"{k} (n={v['n']})")
            ax.set_xlabel("s from event")
        axes[0].set_ylabel("rad (norm over J1..J7)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "figures" / "event_aligned_residual.png", dpi=110)
        plt.close(fig)
    except ImportError:
        pass

    # battery: only episodes with >= 2 right closes, so right_open is the release over the bay (ep 22 has one close:
    # its first open is the end-of-episode release, not the insertion)
    gP, gB = fk_geometry(P, evP, "pringles"), fk_geometry(B, [r for r in evB if r["right_n_closes"] >= 2], "battery")
    write_csv(gP + gB, out / "fk_bimanual_geometry.csv")
    summary["fk_geometry"] = {"pringles_real_10_v00@left_close": summarize_geometry(gP),
                              "insertbattery_real_v02@right_open1(>=2 right closes)": summarize_geometry(gB),
                              "battery_excluded_single_close": [r["episode"] for r in evB if r["right_n_closes"] < 2]}

    src = pool if pool is not None else P
    ev_src = episode_events(pool) if pool is not None else evP
    fa_rows, fa_sum = fk_vs_annotation(src, ev_src, Path(args.annotations), "real_v00_pool")
    write_csv(fa_rows, out / "fk_projection_vs_annotation.csv")
    summary["fk_projection_vs_annotation"] = {"dataset": src.label, **fa_sum}

    summary["poc_episodes"] = [int(x) for x in args.poc_episodes.split(",")]
    summary["figures"] = overlays(P, evP, summary["poc_episodes"], out / "figures")
    summary["figures"] += battery_sheets(B, evB, segB, [int(x) for x in args.battery_sheet_episodes.split(",")], out / "figures")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("events", "fk_geometry", "fk_projection_vs_annotation", "tracking_residual")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
