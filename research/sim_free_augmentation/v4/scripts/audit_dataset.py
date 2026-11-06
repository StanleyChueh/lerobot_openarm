"""v4B Task B0: validity audit of every synthetic episode a v4B arm can sample, plus the real dataset. Read-only.

    cd research/sim_free_augmentation/v4
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/audit_dataset.py

Checks (per episode):
  readback      frozen LeRobot rows == manifest sha256 == candidate arrays[rows]  (state and action, float32)
  labels        v3.1 hard gates (construction/unsupported/gripper chronology/finite/hash), IK, targets, verifier;
                residual-uncertain flag; receiver alignment status
  chronology    on the EXPORTED actions: right close < left close < right open, same number of gripper transitions
                as the source
  duplication   synthetic state differs from its source; no two selected episodes share a state hash; leakage:
                nearest pool-10..49 recording distance (state L-inf, first 60 frames) for every training episode
  alignment     standing-can frames before the grasp (every 10th frame of [0, rc-5)): red-can centroid in the
                rendered body frame (searched within 90 px of the projection) minus the projected candidate can centre,
                compared with the SAME quantity on the real source frames (cancels the shared calibration offset)
  artifacts     frozen v3.1 image QA verdicts + contact sheets for visual inspection
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
SFA = HERE.parent
V31 = SFA / "v3_1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SFA / "v3"))
sys.path.insert(0, str(V31 / "scripts"))
from sfa4b import mixture as mx  # noqa: E402
from sfa3.recon3 import data, rb  # noqa: E402

OUT = HERE / "outputs/v4b"
ROOT = V31 / "outputs/lerobot"


def h32(x):
    return hashlib.sha256(np.ascontiguousarray(x, np.float32).tobytes()).hexdigest()


def table(name):
    import pandas as pd
    df = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / name / "data").rglob("*.parquet"))]).sort_values("index")
    return df


def n_trans(g):
    c = g < 0.5
    return int((c[1:] != c[:-1]).sum())


def can_centre(can4):
    return can4[:3, 3] + 0.1 * can4[:3, 2]


def red_centroid_near(rgb, uv, r=90):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = ((hsv[..., 0] < 8) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)
    yy, xx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    m = red & ((xx - uv[0]) ** 2 + (yy - uv[1]) ** 2 <= r * r)
    if m.sum() < 150:
        return None
    return np.array([xx[m].mean(), yy[m].mean()])


def main():
    sel = mx.selection()
    used = {}
    for arm, d in sel["arms"].items():
        for E, eps in d.items():
            for i in eps:
                used.setdefault(i, set()).add(arm)
    man = {e["episode_index"]: e for e in json.loads((V31 / "outputs/dataset_export/Gp_matched_evidence_complete_manifest.json").read_text())}
    audit = {r["ec_dataset_episode_index"]: r for r in json.loads((OUT / "family_audit.json").read_text())["candidates"]
             if r["in_frozen_Gp_matched_evidence_complete_dataset"]}
    dfS, dfR = table("sfa_v31_Gp_matched_evidence_complete"), table("sfa_v31_R")
    cam = rb.Camera()
    # held-out pool states for leakage
    from tools.dataset_diagnostics.loader import load_dataset
    import vision_ablation as va
    pool = load_dataset(root=va.POOL, label="pool", episodes=list(range(10, 50))).vectors("observation.state")
    def nearest_pool(S):
        return float(min(np.abs(pool[e][:60] - S[:60]).max() for e in pool if len(pool[e]) >= 60))
    rows, hashes = [], {}
    res_cache, src_cache = {}, {}
    sheet_pos, sheet_ho = [], []
    for i in sorted(used):
        m, a = man[i], audit[i]
        E, key = m["E"], m["key"]
        if E not in res_cache:
            res_cache[E] = json.loads((V31 / f"outputs/ep{E}/results.json").read_text())
            src = data.load(E)
            src_cache[E] = src
        res, src = res_cache[E], src_cache[E]
        c = res["candidates"][key]
        z = np.load(V31 / f"outputs/ep{E}/candidates/{key}.npz")
        r0, r1 = m["rows"]
        d = dfS[dfS["episode_index"] == i]
        S = np.stack(d["observation.state"].to_numpy()).astype(np.float32)
        A = np.stack(d["action"].to_numpy()).astype(np.float32)
        readback = h32(S) == m["state_sha256"] == h32(z["state"][r0:r1]) and h32(A) == m["action_sha256"] == h32(z["action"][r0:r1])
        ev_c, ev_s = data.command_events(A), data.command_events(src.action)
        chron = all(ev_c[k] is not None for k in ev_c) and ev_c["right_close"] < ev_c["left_close"] < ev_c["right_open"]
        gs, gc = rb.Robot.split(src.action), rb.Robot.split(A)
        same_n = n_trans(gs["gR"]) == n_trans(gc["gR"]) and n_trans(gs["gL"]) == n_trans(gc["gL"])
        hashes.setdefault(m["state_sha256"], []).append(i)
        # alignment on standing-can frames
        rc = res["recon"]["events"]["right_close"]
        raw = np.load(V31 / f"outputs/cache/ep{E}/raw/{res['render_alias'][key]}.npy", mmap_mode="r")
        ident = np.load(V31 / f"outputs/ep{E}/candidates/identity__G+__s1.0.npz")
        Fsrc = src.frames["observation.images.body_cam"]
        dev = []
        for t in range(0, max(1, rc - 5), 10):
            ug = cam.project(can_centre(z["can"][t])[None])[0][0]
            us = cam.project(can_centre(ident["can"][t])[None])[0][0]
            cg, cs = red_centroid_near(np.asarray(raw[t]), ug), red_centroid_near(Fsrc[t], us)
            if cg is not None and cs is not None:
                dev.append(float(np.linalg.norm((cg - ug) - (cs - us))))
        same_pre = float(np.mean([np.array_equal(np.asarray(raw[t]), Fsrc[t]) for t in range(0, max(1, rc - 5), 10)]))
        qa = next((x for x in json.loads((V31 / "outputs/dataset_export/export_checks.json").read_text())["checks"]
                   if x["episode"] == E and x["key"] == key), {})
        rows.append({"ec_episode_index": i, "E": E, "key": key, "class": a["class"], "arms": sorted(used[i]),
                     "readback_exact": bool(readback), "labels_ok": c["labels_ok"], "verify_failed": c["verify_failed"],
                     "ik_ok": c["ik"]["ok"], "targets_ok": c["targets_ok"], "residual_uncertain": c["residual_uncertain"],
                     "receiver_lag_status": m["receiver_lag_status"], "chronology_ok": bool(chron), "same_gripper_transition_count": bool(same_n),
                     "state_differs_from_source": float(np.abs(z["state"] - ident["state"]).max()) > 1e-3,
                     "nearest_pool_recording_linf_rad": nearest_pool(S), "image_qa_ok": qa.get("image_qa_ok"), "image_qa_why": qa.get("image_qa_why"),
                     "pregrasp_alignment_dev_px_median": float(np.median(dev)) if dev else None, "pregrasp_alignment_n": len(dev),
                     "pregrasp_frames_identical_to_source_frac": same_pre, "init_shift_mm": a["init_shift_mm"]})
        if a["class"] == "position_only" and "RpPos" in used[i]:
            t2 = min(rc, len(raw) - 1)
            sheet_pos.append([Fsrc[0], np.asarray(raw[0]), np.asarray(raw[t2]), f"ep{E} {key.split('__')[0]}"])
        if a["class"] == "handover_only":
            lc = res["recon"]["events"]["left_close"]
            sheet_ho.append([Fsrc[lc], np.asarray(raw[min(lc, len(raw) - 1)]), None, f"ep{E} {key.split('__')[0]}"])
    # real dataset readback + leakage
    real = []
    for e in range(10):
        d = dfR[dfR["episode_index"] == e]
        S = np.stack(d["observation.state"].to_numpy()).astype(np.float32)
        real.append({"episode": e, "equals_source_state": bool(np.abs(S - src_cache.get(e, data.load(e, cams=())).state.astype(np.float32)).max() < 1e-6),
                     "nearest_pool_recording_linf_rad": nearest_pool(S)})
    dup = {h: v for h, v in hashes.items() if len(v) > 1}
    by_class = {}
    for cls in ("position_only", "handover_only"):
        R = [r for r in rows if r["class"] == cls]
        dv = [r["pregrasp_alignment_dev_px_median"] for r in R if r["pregrasp_alignment_dev_px_median"] is not None]
        by_class[cls] = {"n": len(R), "readback_exact": sum(r["readback_exact"] for r in R), "labels_ok": sum(r["labels_ok"] for r in R),
                         "chronology_ok": sum(r["chronology_ok"] for r in R), "same_transition_count": sum(r["same_gripper_transition_count"] for r in R),
                         "residual_uncertain": sum(r["residual_uncertain"] for r in R), "receiver_ok": sum(r["receiver_lag_status"] == "ok" for r in R),
                         "image_qa_ok": sum(bool(r["image_qa_ok"]) for r in R), "differs_from_source": sum(r["state_differs_from_source"] for r in R),
                         "alignment_dev_px_median_of_medians": float(np.median(dv)) if dv else None,
                         "alignment_dev_px_max": float(np.max(dv)) if dv else None, "alignment_dev_gt_10px": int(sum(x > 10 for x in dv)),
                         "pregrasp_identical_to_source_frac_mean": float(np.mean([r["pregrasp_frames_identical_to_source_frac"] for r in R])),
                         "min_nearest_pool_linf_rad": float(min(r["nearest_pool_recording_linf_rad"] for r in R))}
    out = {"episodes": rows, "real": real, "duplicate_state_hashes": dup, "summary": by_class,
           "selection": {"sources": sel["sources"], "k": sel["k"], "keys": sel["keys"]}}
    (OUT / "dataset_audit.json").write_text(json.dumps(out, indent=1, default=float))
    for nm, sh, cols in (("contact_sheet_position.jpg", sheet_pos, 3), ("contact_sheet_handover.jpg", sheet_ho, 2)):
        tiles = []
        for t in sh:
            ims = [cv2.resize(x, (213, 160)) for x in t[:cols] if x is not None]
            row = np.concatenate(ims, 1).copy()
            cv2.putText(row, t[3], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(row)
        per = 2
        while len(tiles) % per:
            tiles.append(np.zeros_like(tiles[0]))
        img = np.concatenate([np.concatenate(tiles[k:k + per], 1) for k in range(0, len(tiles), per)], 0)
        cv2.imwrite(str(OUT / nm), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps({"summary": by_class, "duplicates": dup, "real": real}, indent=1, default=float))


if __name__ == "__main__":
    main()
