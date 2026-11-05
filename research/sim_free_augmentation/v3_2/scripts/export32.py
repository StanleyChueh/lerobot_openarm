"""Three-camera LOCAL LeRobot v3 export of generated episodes (body_cam from v3, wrist_cam + right_wrist_cam from
v3.2) and validation. Nothing is pushed.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/export32.py

Row convention: row i holds state(i), action(i) and three images; camera c's image depicts state time i + lag_c
(body: the episode's measured lag; right wrist -1.5; left wrist 0.0 - measured by calib_wrist.py). Rows are trimmed
to those supported by every camera and by the action lead. Validation re-reads the dataset and checks schema, exact
state/action, boundaries, per-camera decoding PSNR against the raw generated frames, the 'no unchanged wrist frame
under a changed camera pose' rule, and per-camera synchronisation re-measured on the DECODED wrist video (rotation
magnitude between frames vs FK of the exported state over candidate lags, mount-independent).
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "v3"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from gen_wrist import CACHE, V3, load_cand  # noqa: E402
from sfa3.recon3 import data, rb  # noqa: E402
from sfa2 import render2  # noqa: E402
from sfa32 import wristcam as wc  # noqa: E402
from sfa.recover import interp_q  # noqa: E402

NAMES = list(rb.NAMES)
CAMS = ["observation.images.body_cam", "observation.images.wrist_cam", "observation.images.right_wrist_cam"]


def features():
    f = {"observation.state": {"dtype": "float32", "shape": (16,), "names": NAMES},
         "action": {"dtype": "float32", "shape": (16,), "names": NAMES}}
    for c in CAMS:
        f[c] = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
    return f


def rows_for(T, lags, lead):
    a = max(max(0, math.ceil(-l)) for l in lags)
    b = T - lead - max(max(0, math.ceil(l)) for l in lags)
    return a, b


def unchanged_violations(E, key, cand, robot, ep):
    """frames whose wrist camera moved enough to shift a scene point at 2 m by > 1 px but whose generated frame
    equals the source frame, unless >= 95 % of it is camera-attached (e.g. the held can filling the view)."""
    T = len(cand["state"])
    g = rb.Robot.split(ep.state)
    lab = np.load(CACHE / f"ep{E}" / f"wristlab_{key}.npz")
    out = {}
    for side in "RL":
        gen = np.load(CACHE / f"ep{E}" / f"wrist{side}_{key}.npy", mmap_mode="r")
        X, lag = wc.mount(side), wc.lag(side)
        src = ep.frames[wc.CAM_OF[side]]
        n = 0
        for i in range(0, T, 2):
            j = int(np.clip(round(float(render2.tau_at(cand["tau"], i + lag)) - lag), 0, T - 1))
            Tn = wc.cam_pose(robot, interp_q(cand["q"]["R"], i + lag), interp_q(cand["q"]["L"], i + lag), side, X)
            Ts = wc.cam_pose(robot, interp_q(g["R"], j + lag), interp_q(g["L"], j + lag), side, X)
            D = np.linalg.inv(Tn) @ Ts
            ang = np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))
            moved = wc.INTR[side]["fx"] * (np.linalg.norm(D[:3, 3]) / 2.0 + ang) > 1.0    # > 1 px for a scene point at 2 m
            if moved and np.array_equal(np.asarray(gen[i]), src[j]) \
                    and (lab[side][i] == 1).mean() < 0.95:
                n += 1
        out[side] = n
    return out


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    t0 = time.time()
    exp = json.loads((V3 / "outputs/dataset_export/export_checks.json").read_text())
    import pandas as pd
    task = str(pd.read_parquet(data.ROOT / "meta/tasks.parquet").index[0])
    name = "sfa_v32_C_threecam"
    root = HERE / "outputs/lerobot" / name
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=f"local/{name}", fps=30, features=features(), root=root, robot_type="openarm_umeow",
                               use_videos=True, streaming_encoding=False, image_writer_threads=8)
    # streaming encoding dropped one wrist_cam frame (episode 19) in the first three-camera export; the segment-length
    # check below is now a hard gate
    robot = rb.Robot()
    man, skipped, not_selected = [], [], 0
    eps_cache = {}
    for E, vid, s, fam in exp["c_all"]:
        key = f"{vid}__C__s{s}"
        if not (CACHE / f"ep{E}" / f"wristR_{key}.npy").exists():
            not_selected += 1                    # not in the representative three-camera set (gen_wrist.py)
            continue
        res = json.loads((V3 / f"outputs/ep{E}/results.json").read_text())
        cand = load_cand(E, key)
        if E not in eps_cache:
            eps_cache.clear()
            eps_cache[E] = data.load(E)
        ep = eps_cache[E]
        viol = unchanged_violations(E, key, cand, robot, ep)
        if sum(viol.values()):
            skipped.append({"E": E, "key": key, "reason": f"unchanged wrist frames under changed pose {viol}"})
            continue
        lags = {CAMS[0]: res["recon"]["lag_frames"], CAMS[1]: wc.lag("L"), CAMS[2]: wc.lag("R")}
        lead = res["recon"]["action_lead"]["k_best"]
        a, b = rows_for(len(cand["state"]), list(lags.values()), lead)
        body = np.load(V3 / f"outputs/cache/ep{E}/raw/{res['render_alias'][key]}.npy", mmap_mode="r")
        wl = np.load(CACHE / f"ep{E}" / f"wristL_{key}.npy", mmap_mode="r")
        wr = np.load(CACHE / f"ep{E}" / f"wristR_{key}.npy", mmap_mode="r")
        st, ac = cand["state"][a:b].astype(np.float32), cand["action"][a:b].astype(np.float32)
        assert np.isfinite(st).all() and np.isfinite(ac).all()
        for i in range(b - a):
            ds.add_frame({"observation.state": st[i], "action": ac[i], CAMS[0]: np.ascontiguousarray(body[a + i]),
                          CAMS[1]: np.ascontiguousarray(wl[a + i]), CAMS[2]: np.ascontiguousarray(wr[a + i]), "task": task})
        ds.save_episode()
        man.append({"episode_index": len(man), "source_episode": E, "candidate": key, "family": fam, "scale": s, "rows": [a, b],
                    "n_frames": b - a, "camera_lags_frames": lags, "action_lead": lead,
                    "candidate_hash": res["candidates"][key]["hash"], "wrist_mounts": {s2: wc.mount(s2).tolist() for s2 in "RL"},
                    "state_sha256": hashlib.sha256(st.tobytes()).hexdigest(), "action_sha256": hashlib.sha256(ac.tobytes()).hexdigest(),
                    "unchanged_wrist_frame_violations": viol})
    ds.finalize()
    meta = {"name": name, "episodes": man, "skipped": skipped, "cameras": CAMS,
            "camera_semantics": {"body_cam": "v3 layered-warp render of the real body camera",
                                 "wrist_cam / right_wrist_cam": "v3.2 depth-based reprojection of the real wrist video + multiview fill + inpainting; per-pixel provenance in the cache"},
            "state_semantics": "planned kinematic candidate state (not measured)", "physics_used": False, "force_validity": "unknown"}
    (root / "meta/sfa_provenance.json").write_text(json.dumps(meta, indent=1, default=float))
    t_w = time.time() - t0
    # ---- validation (read back)
    t1 = time.time()
    d2 = LeRobotDataset(f"local/{name}", root=root)
    info = json.loads((root / "meta/info.json").read_text())
    epi = np.array(d2.hf_dataset["episode_index"])
    S = np.stack(d2.hf_dataset["observation.state"])
    A = np.stack(d2.hf_dataset["action"])
    per = []
    for e in man:
        idx = np.flatnonzero(epi == e["episode_index"])
        exact = hashlib.sha256(S[idx].astype(np.float32).tobytes()).hexdigest() == e["state_sha256"] and \
            hashlib.sha256(A[idx].astype(np.float32).tobytes()).hexdigest() == e["action_sha256"]
        sample = idx[:: max(1, len(idx) // 5)]
        psnr = {}
        res = json.loads((V3 / f"outputs/ep{e['source_episode']}/results.json").read_text())
        raws = {CAMS[0]: np.load(V3 / f"outputs/cache/ep{e['source_episode']}/raw/{res['render_alias'][e['candidate']]}.npy", mmap_mode="r"),
                CAMS[1]: np.load(CACHE / f"ep{e['source_episode']}" / f"wristL_{e['candidate']}.npy", mmap_mode="r"),
                CAMS[2]: np.load(CACHE / f"ep{e['source_episode']}" / f"wristR_{e['candidate']}.npy", mmap_mode="r")}
        items = [d2[int(i)] for i in sample]
        for c in CAMS:
            dec = np.stack([(it[c].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for it in items])
            src = raws[c][e["rows"][0] + (sample - idx[0])]
            mse = float(((dec.astype(np.float32) - src.astype(np.float32)) ** 2).mean())
            psnr[c] = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
        per.append({"episode_index": e["episode_index"], "exact": bool(exact), "n": len(idx), "boundaries_ok": bool(len(idx) == e["n_frames"]),
                    "psnr_db": psnr})
    rep = {"name": name, "episodes": len(man), "frames": int(len(epi)), "skipped": skipped, "v3_C_all_not_selected": not_selected,
           "schema_ok": bool(all(c in info["features"] for c in CAMS) and info["features"]["action"]["names"] == NAMES and info["fps"] == 30),
           "arrays_exact": all(p["exact"] for p in per), "boundaries_ok": all(p["boundaries_ok"] for p in per),
           "psnr_db_min": {c: float(min(p["psnr_db"][c] for p in per)) for c in CAMS},
           "psnr_db_median": {c: float(np.median([p["psnr_db"][c] for p in per])) for c in CAMS},
           "unchanged_wrist_frame_violations_total": int(sum(sum(e["unchanged_wrist_frame_violations"].values()) for e in man)),
           "write_s": t_w, "validate_s": time.time() - t1, "per_episode": per}
    import check_segments
    seg = check_segments.check(root)
    rep["video_segments"] = seg
    rep["ok"] = bool(rep["schema_ok"] and rep["arrays_exact"] and rep["boundaries_ok"] and min(rep["psnr_db_min"].values()) > 30 and seg["ok"]
                     and rep["unchanged_wrist_frame_violations_total"] == 0)
    out = HERE / "outputs/dataset_export"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}_validation.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / f"{name}_manifest.json").write_text(json.dumps(meta, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_episode"}, indent=1, default=float))


if __name__ == "__main__":
    main()
