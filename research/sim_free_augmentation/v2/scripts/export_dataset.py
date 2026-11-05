"""Export accepted candidates as a LOCAL single-camera LeRobot v3 research dataset, then validate it.

    cd research/sim_free_augmentation/v2
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/export_dataset.py \
        --episodes 0 4 8 --condition C

* Only observation.images.body_cam is exported: it is the only generated camera. The source wrist / right-wrist views
  are NOT included (they would show the source arm motion, unsynchronised with the changed trajectories).
* observation.state = planned candidate state (kinematic, not measured); action = candidate command (16-D, source
  joint order, motor radians; grippers = source commands under the time warp); fps 30; task string of the source.
* Rows follow the source row convention (see sfa2/sync.py); unsupported boundary rows are trimmed, never padded.
* Only candidates accepted by the independent verifier AND whose synchronisation detectors pass are exported.
  The identity control is not exported (it equals the source).
* Nothing is pushed: the dataset lives under outputs/lerobot/ (git-ignored); a manifest + validation report is
  committed instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sfa2 import metrics2 as M2, recon, sync  # noqa: E402
from sfa2.recon import data, rb  # noqa: E402

TASK_FALLBACK = "Pick up the Pringles can with the right arm, hand it over to the left arm"


def source_task() -> str:
    import pandas as pd
    t = pd.read_parquet(data.ROOT / "meta/tasks.parquet")
    return str(t.index[0])


def features():
    names = list(rb.NAMES)
    return {"observation.state": {"dtype": "float32", "shape": (16,), "names": names},
            "action": {"dtype": "float32", "shape": (16,), "names": names},
            "observation.images.body_cam": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}}


def selected(E: int, cond: str):
    res = json.loads((HERE / f"outputs/ep{E}/results.json").read_text())
    out = []
    for k, c in sorted(res["candidates"].items()):
        if c["cond"] != cond or c["var"] == "identity" or not c["accepted"]:
            continue
        m = res["clips"][c["render_alias"]]["metrics"]
        sync_ok = m["action_lead"]["lead"] == res["recon"]["action_lead"]["k_best"] and \
            abs(m["lag_scan"]["lag"] - res["recon"]["lag_frames"]) <= 0.5
        out.append((k, c, sync_ok, res))
    return out


def export(episodes, cond, root: Path, name: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if root.exists():
        shutil.rmtree(root)
    task = source_task()
    ds = LeRobotDataset.create(repo_id=f"local/{name}", fps=30, features=features(), root=root, robot_type="openarm_umeow",
                               use_videos=True, streaming_encoding=True)
    manifest = {"name": name, "condition": cond, "task": task, "episodes": [], "skipped": [],
                "state_semantics": "planned kinematic candidate state (IK solution), NOT measured robot feedback",
                "action_semantics": "a_new(t) = q_new(t+lead) + r_src(tau(t)) on arm joints (source command-state residual "
                                    "transported); gripper channels = source commands under zero-order hold of tau",
                "row_convention": "row i: state(i), action(i), body_cam image depicting state time i + lag (as in the source)",
                "cameras": {"generated": ["observation.images.body_cam"],
                            "omitted": ["observation.images.wrist_cam", "observation.images.right_wrist_cam"],
                            "why_omitted": "wrist views were not generated; source wrist frames would contradict the edited arm motion"},
                "physics_used": False, "simulation_used": False, "force_validity": "unknown (no force telemetry)"}
    ep_out = 0
    for E in episodes:
        for key, c, sync_ok, res in selected(E, cond):
            if not sync_ok:
                manifest["skipped"].append({"source_episode": E, "candidate": key, "reason": "sync detector failed"})
                continue
            z = np.load(HERE / f"outputs/ep{E}/candidates/{key}.npz")
            frames = np.load(HERE / f"outputs/cache/ep{E}/raw/{c['render_alias']}.npy", mmap_mode="r")
            lag, lead = res["recon"]["lag_frames"], res["recon"]["action_lead"]["k_best"]
            a, b = max(0, int(np.ceil(-lag))), len(z["state"]) - lead - max(0, int(np.ceil(lag)))
            st, ac = z["state"][a:b].astype(np.float32), z["action"][a:b].astype(np.float32)
            assert np.isfinite(st).all() and np.isfinite(ac).all(), key
            for i in range(b - a):
                ds.add_frame({"observation.state": st[i], "action": ac[i],
                              "observation.images.body_cam": np.ascontiguousarray(frames[a + i]), "task": task})
            ds.save_episode()
            j_of = np.load(HERE / f"outputs/cache/ep{E}/render_maps.npz")[c["render_alias"]]
            tm = sync.time_map(type("R", (), {"lag": lag, "lead": lead, "T": len(z["state"]), "fps": 30.0, "e": E})(),
                               {"tau": z["tau"]}, j_of)
            manifest["episodes"].append({
                "episode_index": ep_out, "source_episode": E, "candidate": key, "variation": c["var"], "condition": cond,
                "scale": c["scale"], "rows": [a, b], "n_frames": b - a, "lag_frames": lag, "action_lead_frames": lead,
                "candidate_hash": c["hash"], "config_sha256": res["config_sha256"], "source_tree_sha256": res["source_tree_sha256"],
                "verifier_failed": c["verify"]["failed"], "targets": c["targets"],
                "time_map": {k: np.asarray(v).tolist() for k, v in tm.items()},
                "state_sha256": hashlib.sha256(st.tobytes()).hexdigest(), "action_sha256": hashlib.sha256(ac.tobytes()).hexdigest()})
            ep_out += 1
    ds.finalize()
    (root / "meta/sfa_provenance.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def validate(root: Path, name: str, manifest: dict) -> dict:
    """schema, round-trip and replay/decoding checks on the written dataset (read back through LeRobot)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    rep = {"checks": {}}
    info = json.loads((root / "meta/info.json").read_text())
    f = info["features"]
    want = features()
    schema_ok = (info["codebase_version"].startswith("v3") and info["fps"] == 30 and info["robot_type"] == "openarm_umeow"
                 and set(k for k in f if not k in ("timestamp", "frame_index", "episode_index", "index", "task_index")) == set(want)
                 and f["observation.state"]["names"] == rb.NAMES and f["action"]["names"] == rb.NAMES
                 and f["observation.state"]["dtype"] == "float32" and list(f["observation.images.body_cam"]["shape"]) == [480, 640, 3]
                 and not any("wrist" in k for k in f))
    rep["checks"]["schema"] = {"ok": bool(schema_ok), "features": sorted(f), "codec": f["observation.images.body_cam"].get("info", {}).get("video.codec")}
    ds = LeRobotDataset(f"local/{name}", root=root)
    rep["n_frames"], rep["n_episodes"] = len(ds), ds.num_episodes
    count_ok = ds.num_episodes == len(manifest["episodes"]) and len(ds) == sum(e["n_frames"] for e in manifest["episodes"])
    rep["checks"]["counts"] = {"ok": bool(count_ok)}
    robot, cam = rb.Robot(), rb.Camera()
    recs = {}
    per = []
    t0 = time.time()
    hf = ds.hf_dataset
    epi = np.array(hf["episode_index"])
    for e in manifest["episodes"]:
        idx = np.flatnonzero(epi == e["episode_index"])
        items = [ds[int(i)] for i in idx]
        st = np.stack([it["observation.state"].numpy() for it in items])
        ac = np.stack([it["action"].numpy() for it in items])
        ts = np.array([float(it["timestamp"]) for it in items])
        fi = np.array([int(it["frame_index"]) for it in items])
        img = np.stack([(it["observation.images.body_cam"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for it in items])
        E = e["source_episode"]
        z = np.load(HERE / f"outputs/ep{E}/candidates/{e['candidate']}.npz")
        a, b = e["rows"]
        raw = np.load(HERE / f"outputs/cache/ep{E}/raw/{json.loads((HERE / f'outputs/ep{E}/results.json').read_text())['candidates'][e['candidate']]['render_alias']}.npy",
                      mmap_mode="r")[a:b]
        mae = float(np.abs(img.astype(np.int16) - raw.astype(np.int16)).mean())
        mse = float(((img.astype(np.float32) - raw.astype(np.float32)) ** 2).mean())
        psnr = float(10 * np.log10(255 ** 2 / max(mse, 1e-9)))
        # replay / decoding: re-measure the synchronisation on the DECODED dataset
        if E not in recs:
            recs[E] = recon.build(E, HERE / f"outputs/cache/ep{E}", with_plate=False, log=lambda *x: None)
        rec = recs[E]
        lead = sync.action_lead(st, ac)["lead"]
        can_rows = z["can"][a:b]
        uv, ok = M2.lid_track(img, cam, list(can_rows))
        marks = json.loads((HERE / f"outputs/ep{E}/results.json").read_text())["candidates"][e["candidate"]]["marks"]
        win = range(max(0, marks["grasp"] + 10 - a), max(0, marks["capture"] - 3 - a))
        ls = M2.lag_scan(robot, cam, st.astype(float), rec.X["R"], uv, ok, win, np.arange(-6.0, 3.01, 0.25))
        r = {"episode_index": e["episode_index"], "candidate": e["candidate"], "n": len(idx),
             "state_exact": bool(np.array_equal(st, z["state"][a:b].astype(np.float32))),
             "action_exact": bool(np.array_equal(ac, z["action"][a:b].astype(np.float32))),
             "frame_index_contiguous": bool(np.array_equal(fi, np.arange(len(idx)))),
             "timestamps_ok": bool(np.abs(ts - np.arange(len(idx)) / 30.0).max() < 1e-4),
             "decoded_vs_raw_mae": mae, "decoded_vs_raw_psnr_db": psnr,
             "replay_action_lead": lead, "replay_lag": ls["lag"], "expected_lag": rec.lag, "expected_lead": rec.lead,
             "replay_sync_ok": bool(lead == rec.lead and abs(ls["lag"] - rec.lag) <= 0.5)}
        r["ok"] = bool(r["state_exact"] and r["action_exact"] and r["frame_index_contiguous"] and r["timestamps_ok"]
                       and r["replay_sync_ok"] and psnr > 30)
        per.append(r)
    rep["episodes"] = per
    rep["checks"]["round_trip_and_replay"] = {"ok": all(r["ok"] for r in per), "n_ok": sum(r["ok"] for r in per)}
    rep["validation_runtime_s"] = time.time() - t0
    rep["ok"] = all(v["ok"] for v in rep["checks"].values())
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 4, 8])
    ap.add_argument("--condition", default="C")
    args = ap.parse_args()
    name = f"openarm_pringles_sfa_v2_{args.condition}_bodycam"
    root = HERE / "outputs/lerobot" / name
    t0 = time.time()
    man = export(args.episodes, args.condition, root, name)
    t_exp = time.time() - t0
    rep = validate(root, name, man)
    rep["export_runtime_s"] = t_exp
    out = HERE / "outputs/dataset_export"
    out.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in man.items() if k != "episodes"} | {"episodes": [{k: v for k, v in e.items() if k != "time_map"} for e in man["episodes"]]}
    (out / f"{name}_manifest.json").write_text(json.dumps(slim, indent=1))
    (out / f"{name}_validation.json").write_text(json.dumps(rep, indent=1))
    shutil.copy(root / "meta/info.json", out / f"{name}_info.json")
    print(json.dumps({"episodes": len(man["episodes"]), "skipped": man["skipped"], "ok": rep["ok"], "checks": rep["checks"],
                      "export_s": round(t_exp, 1), "validate_s": round(rep["validation_runtime_s"], 1)}, indent=1))


if __name__ == "__main__":
    main()
