"""Score human annotations (when they exist) for the Gate 1 pilot. With no human file it reports PENDING and exits 3.

    env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python research/contact_aware_augmentation/gate1/annotation/evaluate_annotations.py \
        --annotations research/contact_aware_augmentation/outputs/gate1/annotation/annotations_*.csv \
        --gate1 research/contact_aware_augmentation/outputs/gate1 --out research/contact_aware_augmentation/outputs/gate1/annotation_eval.json

Ground-truth rule (fixed before any annotation exists):
  * only `visible` answers enter ground truth; `uncertain` and `occluded` never do;
  * with >= 2 annotators, a point is GT only if every annotator marked it visible and all pairwise distances are
    <= AGREE_PX; GT = their mean. Disagreeing points are excluded and counted;
  * with 1 annotator, GT = that annotator's visible clicks, reported as SINGLE-ANNOTATOR (weaker);
  * repeat (duplicate) frames measure intra-annotator repeatability and are not double-counted in GT.

Outputs (each PASS / FAIL / PENDING against the Gate-1 criteria):
  G1  robot keypoint reprojection, LOEO over the pilot episodes: median <= 8 px and p90 <= 15 px
  G2  tracker vs human lid centre on tracker-`ok` frames: median <= 6 px; plus tracker coverage of GT-visible frames
  R   human click repeatability (intra, inter)
  H   rigid-attachment residual computed on HUMAN lid points (separates score-model error from tracker error)
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
sys.path[:0] = [str(HERE.parent), str(HERE)]
import gate1_core as gc  # noqa: E402
from annotate_gate1 import KEYPOINTS, PILOT, PRINGLES, SEALED  # noqa: E402
from gate1_core import g0  # noqa: E402

AGREE_PX = 6.0
ROBOT_KP = {"right_wrist_tag": ("right", ("link5", "link6", "link7")), "left_wrist_tag": ("left", ("link5", "link6", "link7")),
            "right_jaw_mid": ("right", ("hand_tcp", "link7")), "left_jaw_mid": ("left", ("hand_tcp", "link7"))}


def load_answers(paths) -> list[dict]:
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    assert not {int(r["episode"]) for r in rows} & SEALED, "annotations contain sealed episodes"
    return rows


def repeatability(rows) -> dict:
    out = {}
    by = {}
    for r in rows:
        if r["visibility"] == "visible":
            by.setdefault((r["annotator"], int(r["episode"]), int(r["frame"]), r["keypoint"]), {})[int(r["repeat"])] = \
                np.array([float(r["u"]), float(r["v"])])
    intra = [np.linalg.norm(d[0] - d[1]) for d in by.values() if 0 in d and 1 in d]
    out["intra_px"] = {"n": len(intra), **({"median": round(float(np.median(intra)), 2), "p90": round(float(np.percentile(intra, 90)), 2)} if intra else {})}
    ann = sorted({r["annotator"] for r in rows})
    inter = []
    for (a, e, f, k), d in by.items():
        for b in ann:
            if b > a and (b, e, f, k) in by and 0 in d and 0 in by[(b, e, f, k)]:
                inter.append(np.linalg.norm(d[0] - by[(b, e, f, k)][0]))
    out["inter_px"] = {"n": len(inter), **({"median": round(float(np.median(inter)), 2), "p90": round(float(np.percentile(inter, 90)), 2)} if inter else {})}
    out["annotators"] = ann
    return out


def ground_truth(rows) -> tuple[dict, dict]:
    """(episode, frame, keypoint) -> (u, v) from non-repeat answers, by the rule in the module docstring."""
    ann = sorted({r["annotator"] for r in rows})
    by = {}
    for r in rows:
        if int(r["repeat"]) == 0:
            by.setdefault((int(r["episode"]), int(r["frame"]), r["keypoint"]), []).append(r)
    gt, stats = {}, {"points": 0, "gt": 0, "excluded_uncertain_or_occluded": 0, "excluded_disagreement": 0,
                     "single_annotator": len(ann) < 2}
    for k, rs in by.items():
        stats["points"] += 1
        if len({r["annotator"] for r in rs}) < len(ann) or any(r["visibility"] != "visible" for r in rs):
            stats["excluded_uncertain_or_occluded"] += 1
            continue
        P = np.array([[float(r["u"]), float(r["v"])] for r in rs])
        if len(P) > 1 and max(np.linalg.norm(a - b) for a in P for b in P) > AGREE_PX:
            stats["excluded_disagreement"] += 1
            continue
        gt[k] = P.mean(0)
        stats["gt"] += 1
    return gt, stats


def frame_poses(fk, q, lag, frame_names, frames_idx):
    T = len(q)
    out = {n: [] for n in frame_names}
    for t in frames_idx:
        tq = float(np.clip(t + lag, 0, T - 1))
        i0 = int(np.floor(tq)); i1 = min(i0 + 1, T - 1); w = tq - i0
        F = fk.frames((1 - w) * q[i0] + w * q[i1])
        for n in frame_names:
            out[n].append(F[n])
    return {n: np.array(v) for n, v in out.items()}


def robot_reprojection(gt, Q, fk, lags) -> dict:
    """G1: per robot keypoint, a constant 3-D offset in a link frame; link chosen and offset fitted on the 2 dev
    episodes, error measured on the held-out episode (LOEO)."""
    res = {"per_keypoint": {}, "errors_px": []}
    for kp, (side, links) in ROBOT_KP.items():
        pts = {e: [(f, uv) for (ee, f, k), uv in gt.items() if ee == e and k == kp] for e in PILOT}
        res["per_keypoint"][kp] = {}
        for test in PILOT:
            dev = [e for e in PILOT if e != test]
            if sum(len(pts[e]) for e in dev) < 6 or not pts[test]:
                res["per_keypoint"][kp][test] = "insufficient GT"
                continue
            best = None
            for link in links:
                name = f"openarm_{side}_{link}"
                Ts = np.concatenate([frame_poses(fk, Q[e], lags[test], [name], [f for f, _ in pts[e]])[name] for e in dev])
                uv = np.concatenate([[u for _, u in pts[e]] for e in dev])
                fit = gc.fit_point(Ts, uv, g0.BODY_CAM)
                if best is None or fit["rms"] < best[1]["rms"]:
                    best = (name, fit)
            name, fit = best
            Tt = frame_poses(fk, Q[test], lags[test], [name], [f for f, _ in pts[test]])[name]
            err = np.linalg.norm(gc.project_points(Tt, fit["p"], g0.BODY_CAM) - np.array([u for _, u in pts[test]]), axis=1)
            res["errors_px"] += err.tolist()
            res["per_keypoint"][kp][test] = {"link": name, "dev_rms_px": round(fit["rms"], 2), "n_test": len(err),
                                             "test_median_px": round(float(np.median(err)), 2), "test_p90_px": round(float(np.percentile(err, 90)), 2)}
    e = np.array(res.pop("errors_px"))
    if len(e):
        res["all"] = {"n": len(e), "median_px": round(float(np.median(e)), 2), "p90_px": round(float(np.percentile(e, 90)), 2)}
        res["G1"] = "PASS" if np.median(e) <= 8 and np.percentile(e, 90) <= 15 else "FAIL"
    else:
        res["G1"] = "PENDING"
    return res


def tracker_vs_gt(gt, gate1_dir: Path) -> dict:
    err, cover_ok, cover_all = [], 0, 0
    for e in PILOT:
        d = np.load(gate1_dir / "tracks" / f"lid_track_ep{e}.npz")
        for (ee, f, k), uv in gt.items():
            if ee != e or k != "lid_center":
                continue
            cover_all += 1
            if d["status"][f] == "ok":
                cover_ok += 1
                err.append(np.linalg.norm(d["uv"][f] - uv))
    out = {"gt_lid_points": cover_all, "tracker_ok_on_gt_points": cover_ok}
    if err:
        out.update({"median_px": round(float(np.median(err)), 2), "p90_px": round(float(np.percentile(err, 90)), 2),
                    "G2": "PASS" if np.median(err) <= 6 else "FAIL"})
    else:
        out["G2"] = "PENDING"
    return out


def human_rigid_residual(gt, Q, fk, lags, events) -> dict:
    out = {}
    for e in PILOT:
        rc, lc, ro = events[e]["right_close"], events[e]["left_close"], events[e]["right_open"]
        lid = {f: uv for (ee, f, k), uv in gt.items() if ee == e and k == "lid_center"}
        for role, side, lo, hi in (("giver", "right", rc + gc.WINDOW["giver_start"], lc + gc.WINDOW["giver_end"]),
                                   ("receiver", "left", ro + gc.WINDOW["recv_start"], 10 ** 6)):
            fr = sorted(f for f in lid if lo <= f <= hi)
            if len(fr) < 5:
                out[f"ep{e}_{role}"] = "insufficient GT"
                continue
            Ts = frame_poses(fk, Q[e], lags[e], [f"openarm_{side}_hand_tcp"], fr)[f"openarm_{side}_hand_tcp"]
            fit = gc.fit_point(Ts, np.array([lid[f] for f in fr]), g0.BODY_CAM)
            out[f"ep{e}_{role}"] = {"n": len(fr), "rms_px": round(fit["rms"], 2)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", nargs="*", default=[])
    ap.add_argument("--gate1", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    paths = [p for a in args.annotations for p in glob.glob(a)]
    rows = load_answers(paths) if paths else []
    rows = [r for r in rows if r.get("annotator")]
    if not rows:
        res = {"status": "PENDING", "reason": "no human annotation file found; GT is never inferred from the tracker or FK",
               "G1": "PENDING", "G2": "PENDING"}
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        sys.exit(3)
    from tools.dataset_diagnostics.loader import load_dataset
    P = load_dataset(root=PRINGLES, label="p", episodes=list(PILOT))
    Q = P.vectors("observation.state")
    fk = g0.FK(P)
    m = json.loads((Path(args.gate1) / "metrics_full.json").read_text())
    lags = {f["test"]: f["lag"] for f in m["folds"]}  # lag frozen on each fold's dev episodes
    events = {e: m["tracker"]["per_episode"][str(e)]["events_action"] for e in PILOT}
    gt, gst = ground_truth(rows)
    res = {"status": "EVALUATED", "files": paths, "ground_truth": gst, "repeatability": repeatability(rows),
           "robot_reprojection": robot_reprojection(gt, Q, fk, lags), "tracker_vs_gt": tracker_vs_gt(gt, Path(args.gate1)),
           "human_lid_rigid_residual": human_rigid_residual(gt, Q, fk, lags, events)}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
