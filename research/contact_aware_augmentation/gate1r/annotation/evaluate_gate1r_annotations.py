"""Evaluate Gate 1R human annotations. Exit 3 and status PENDING when no human file exists (GT is never inferred).

    env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python research/contact_aware_augmentation/gate1r/annotation/evaluate_gate1r_annotations.py \
        --keypoints "research/contact_aware_augmentation/outputs/gate1r/annotation/keypoints_*.csv" \
        --events "research/contact_aware_augmentation/outputs/gate1r/annotation/events_*.csv" \
        --gate1 research/contact_aware_augmentation/outputs/gate1 --out research/contact_aware_augmentation/outputs/gate1r/annotation_eval.json

Ground-truth rules (fixed before any annotation exists):
  keypoint GT : visibility == visible AND confidence >= 2 for every annotator; with >= 2 annotators all pairwise
                distances <= 6 px; GT = mean. Repeats measure repeatability only.
  event GT    : observable AND confidence >= 2 for every annotator; with >= 2 annotators max spread <= 2 frames.
Gates:
  G1 robot reprojection (jaw midpoint = mean of the two pad tips; wrist tags), LOEO: median <= 8 px, p90 <= 15 px,
     and >= 20 GT points per gripper over the three episodes
  G2 tracker vs human lid centre: median <= 6 px AND tracker-ok on >= 90 % of human-visible lid frames, separately
     in the grasp/lift and transition windows (coverage is a gate, not a footnote)
  EV events: every event GT or explicitly not-observable for all three episodes
  sealed_ready = G1 and G2 and EV pass AND human annotations exist for the sealed episodes' required windows.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent / "gate1"), str(HERE.parent.parent / "gate1" / "annotation")]
import gate1_core as gc  # noqa: E402
import spec as S  # noqa: E402
from gate1_core import g0  # noqa: E402

AGREE_PX, AGREE_FRAMES, MIN_CONF = 6.0, 2, 2


def read(patterns):
    rows = []
    for p in [x for pat in patterns for x in glob.glob(pat)]:
        rows += [r for r in csv.DictReader(open(p)) if r.get("annotator")]
    assert not {int(r["episode"]) for r in rows} - set(S.PILOT) - set(S.SEALED), "unknown episode"
    return rows


def keypoint_gt(rows):
    ann = sorted({r["annotator"] for r in rows})
    by = {}
    for r in rows:
        if int(r["repeat"]) == 0:
            by.setdefault((int(r["episode"]), int(r["frame"]), r["keypoint"]), []).append(r)
    gt, st = {}, {"points": 0, "gt": 0, "excluded_not_visible_or_low_conf": 0, "excluded_disagreement": 0, "annotators": ann}
    for k, rs in by.items():
        st["points"] += 1
        if len({r["annotator"] for r in rs}) < len(ann) or any(r["visibility"] != "visible" or int(r["confidence"] or 0) < MIN_CONF for r in rs):
            st["excluded_not_visible_or_low_conf"] += 1
            continue
        P = np.array([[float(r["u"]), float(r["v"])] for r in rs])
        if len(P) > 1 and max(np.linalg.norm(a - b) for a in P for b in P) > AGREE_PX:
            st["excluded_disagreement"] += 1
            continue
        gt[k] = P.mean(0)
        st["gt"] += 1
    return gt, st


def event_gt(rows):
    ann = sorted({r["annotator"] for r in rows})
    by = {}
    for r in rows:
        by.setdefault((int(r["episode"]), r["event"]), []).append(r)
    gt, notobs = {}, []
    for k, rs in by.items():
        if len({r["annotator"] for r in rs}) < len(ann):
            continue
        if all(r["observable"] == "0" for r in rs):
            notobs.append(k)
            continue
        if any(r["observable"] != "1" or int(r["confidence"] or 0) < MIN_CONF for r in rs):
            continue
        f = [int(r["frame"]) for r in rs]
        if max(f) - min(f) <= AGREE_FRAMES:
            gt[k] = float(np.mean(f))
    return gt, notobs


def g2(gt, gate1_dir: Path, manifest: list[dict]) -> dict:
    purpose = {(int(r["episode"]), int(r["frame"])): r["purpose"] for r in manifest}
    out = {}
    for win in ("dense_grasp_lift", "dense_transition"):
        err, n_vis, n_ok = [], 0, 0
        for (e, f, k), uv in gt.items():
            if k != "obj_lid_center" or purpose.get((e, f)) != win:
                continue
            d = np.load(gate1_dir / "tracks" / f"lid_track_ep{e}.npz")
            n_vis += 1
            if d["status"][f] == "ok":
                n_ok += 1
                err.append(float(np.linalg.norm(d["uv"][f] - uv)))
        res = {"human_visible": n_vis, "tracker_ok": n_ok, "coverage": n_ok / n_vis if n_vis else None}
        if err:
            res.update({"median_px": float(np.median(err)), "p90_px": float(np.percentile(err, 90))})
        res["status"] = ("PENDING" if not n_vis else
                         "PASS" if (res["coverage"] >= 0.9 and res.get("median_px", 1e9) <= 6.0) else "FAIL")
        out[win] = res
    out["status"] = "PASS" if all(v["status"] == "PASS" for v in out.values() if isinstance(v, dict)) else \
        ("PENDING" if any(v["status"] == "PENDING" for v in out.values() if isinstance(v, dict)) else "FAIL")
    return out


def g1(gt, Q, fk, lags) -> dict:
    sys.path.insert(0, str(HERE.parent.parent / "gate1" / "annotation"))
    from evaluate_annotations import frame_poses
    pts = {}
    for side, pre in (("right", "R"), ("left", "L")):
        for (e, f, k), uv in gt.items():
            if k == f"{pre}_pad_tip_1" and (e, f, f"{pre}_pad_tip_2") in gt:
                pts.setdefault((side, "jaw_mid"), []).append((e, f, (uv + gt[(e, f, f"{pre}_pad_tip_2")]) / 2))
            if k == f"{pre}_wrist_tag":
                pts.setdefault((side, "wrist_tag"), []).append((e, f, uv))
    errs, per = [], {}
    links = {"jaw_mid": ("hand_tcp", "link7"), "wrist_tag": ("link5", "link6", "link7")}
    for (side, kind), P in pts.items():
        for test in S.PILOT:
            dev = [x for x in P if x[0] != test]
            tst = [x for x in P if x[0] == test]
            if len(dev) < 6 or not tst:
                continue
            best = None
            for link in links[kind]:
                name = f"openarm_{side}_{link}"
                Ts = np.concatenate([frame_poses(fk, Q[e], lags[test], [name], [f])[name] for e, f, _ in dev])
                fit = gc.fit_point(Ts, np.array([u for *_, u in dev]), g0.BODY_CAM)
                if best is None or fit["rms"] < best[1]["rms"]:
                    best = (name, fit)
            Tt = np.concatenate([frame_poses(fk, Q[e], lags[test], [best[0]], [f])[best[0]] for e, f, _ in tst])
            e_ = np.linalg.norm(gc.project_points(Tt, best[1]["p"], g0.BODY_CAM) - np.array([u for *_, u in tst]), axis=1)
            errs += e_.tolist()
            per[f"{side}_{kind}_test{test}"] = {"link": best[0], "n": len(e_), "median_px": float(np.median(e_))}
    n_side = {s: sum(len(v) for (sd, _), v in pts.items() if sd == s) for s in ("right", "left")}
    if not errs or min(n_side.values()) < 20:
        return {"status": "PENDING", "n_points": n_side, "per": per}
    e = np.array(errs)
    return {"status": "PASS" if np.median(e) <= 8 and np.percentile(e, 90) <= 15 else "FAIL", "n_points": n_side,
            "median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90)), "per": per}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keypoints", nargs="*", default=[])
    ap.add_argument("--events", nargs="*", default=[])
    ap.add_argument("--manifest", default=str(HERE.parents[1] / "outputs/gate1r/annotation/manifest_keypoints.csv"))
    ap.add_argument("--gate1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    kp, evr = read(a.keypoints), read(a.events)
    if not kp and not evr:
        res = {"status": "PENDING", "G1": "PENDING", "G2": "PENDING", "EV": "PENDING", "sealed_ready": False,
               "reason": "no human annotation file; ground truth is never inferred from tracker, FK or verifier output"}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        sys.exit(3)
    from tools.dataset_diagnostics.loader import load_dataset
    sys.path.insert(0, str(HERE.parent))
    import run_gate1r as rr
    P = load_dataset(root=rr.PRINGLES, label="p", episodes=list(S.PILOT))
    Q = P.vectors("observation.state")
    fk = g0.FK(P)
    res_r = json.loads((HERE.parents[1] / "outputs/gate1r/results_full.json").read_text())
    lags = {f["test"]: f["lag"] for f in res_r["folds"]}
    gt, st = keypoint_gt(kp)
    egt, notobs = event_gt(evr)
    ev_ok = all(((e, n) in egt) or ((e, n) in notobs) for e in S.PILOT for n in ("right_first_contact", "lift_off", "left_first_contact", "right_release"))
    res = {"status": "EVALUATED", "keypoint_gt": st, "G1": g1(gt, Q, fk, lags), "G2": g2(gt, Path(a.gate1), list(csv.DictReader(open(a.manifest)))),
           "EV": {"status": "PASS" if ev_ok else "INCOMPLETE", "gt": {f"{k[0]}:{k[1]}": v for k, v in egt.items()},
                  "not_observable": [f"{e}:{n}" for e, n in notobs]}}
    sealed_annotated = any(int(r["episode"]) in S.SEALED for r in kp)
    res["sealed_ready"] = bool(res["G1"]["status"] == "PASS" and res["G2"]["status"] == "PASS" and ev_ok and sealed_annotated)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
