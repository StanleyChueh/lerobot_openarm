"""Pinned Pringles rollout diagnosis. CPU only; see results/eval_v00/PROTOCOL.md.

Run from repository root: .venv/bin/python -B research/sim_free_augmentation/sfagen/tools/rollout_report.py
Pass --fetch to download the three pinned Hub datasets into --inputs first.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import Delaunay
from scipy.stats import binomtest, fisher_exact

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from sfagen.camera import Camera
from sfagen.data import Source
from sfagen.robot import Robot

PINS = {
    "real": ("eval_openarm_pringles_lerobot_real_smolvla_10_v00_async_rtc_smolvla_v00", "de1bbf51fdac10218dd5602260ad65dd9da2aca5"),
    "mixed": ("eval_openarm_pringles_lerobot_real_and_sim_10_3cam_async_rtc_smolvla_v00", "af877eb5294e1e747a0eee113e824bf888bd625e"),
    "training_v00": ("openarm_pringles_lerobot_real_and_sim_10_3cam_v00", "13d7231a48866069001f4a0635259dd7d80b1e17"),
    "training_v01": ("openarm_pringles_lerobot_real_to_sim_10_3cam_v01", "1d64b8e7aeed7585b54c776f6b60e26610b37947"),
}
GROUPS = [("front-right", range(4)), ("rear-right", range(4, 8)), ("rear-left", range(8, 11)),
          ("front-left", range(11, 16)), ("middle", range(16, 20))]
BODY = "observation.images.body_cam"


def lid(frame, near=None, area0=None):
    """Known yellow lid: largest connected colour component, or nearest tracked one.

    shortcut: colour/area tracking flags occlusion as missing; use segmentation for other objects.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (12, 55, 100), (45, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500 or len(c) < 5:
            continue
        (u, v), axes, _ = cv2.fitEllipse(cv2.convexHull(c))
        if min(axes) < 20 or max(axes) / min(axes) > 2:
            continue
        if near is not None and (np.linalg.norm(np.array([u, v]) - near) > 45 or not 0.55 < area / area0 < 1.8):
            continue
        candidates.append((np.array([u, v]), area))
    if not candidates:
        return None
    return min(candidates, key=lambda c: np.linalg.norm(c[0] - near)) if near is not None else max(candidates, key=lambda c: c[1])


def first_run(mask, n=3):
    hits = np.flatnonzero(np.convolve(np.asarray(mask, int), np.ones(n, int), mode="valid") == n) if len(mask) >= n else []
    return int(hits[0]) if len(hits) else None


def first_close(robot, values, side="R"):
    events = robot.gripper_events(values)[side]["closes"]
    return events[0] if events else None


def estimate_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    rng = np.random.default_rng(0)
    return {"n": len(x), "median": float(np.median(x)),
            "ci95": np.percentile(np.median(rng.choice(x, (2000, len(x))), axis=1), [2.5, 97.5]).tolist()}


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "sheets").mkdir(exist_ok=True)
    if args.fetch:
        for name, (repo, rev) in PINS.items():
            files = ["meta/sfa_provenance.json", "meta/info.json"] if name == "training_v01" else []
            subprocess.run(["hf", "download", "ethanCSL/" + repo, *files, "--repo-type", "dataset", "--revision", rev,
                            "--local-dir", str(args.inputs / name)], check=True)
    args.inputs = args.inputs.resolve()
    for name, (_, rev) in PINS.items():
        metadata = args.inputs / name / ".cache/huggingface/download"
        for file in (args.inputs / name).rglob("*"):
            if file.is_file() and ".cache" not in file.parts:
                downloaded = metadata / (str(file.relative_to(args.inputs / name)) + ".metadata")
                assert downloaded.read_text().splitlines()[0] == rev, f"Unverified revision: {file}"
    cfg = yaml.safe_load((HERE / "configs/openarm_pringles_handover.yaml").read_text())
    robot = Robot(cfg["robot"], HERE / "configs")
    cal = json.loads((HERE / "results/pringles/calibration.json").read_text())
    cam = Camera(cfg["camera"]).perturbed(cal["params7"])
    zlid = json.loads((HERE / "results/pringles/table_height.json").read_text())["z"] + 0.23
    world = lambda uv, z=zlid: cam.backproject_plane(np.atleast_2d(uv), z)
    training = Source(dict(root=str(args.inputs / "training_v00"), body_camera=BODY))
    prov = json.loads((training.root / "meta/sfa_provenance.json").read_text())["episodes"]
    label_checks = 0
    for p in prov[10:]:
        s, a = training.arrays(p["episode_index"])
        for array, key in ((s, "state_sha256"), (a, "action_sha256")):
            assert hashlib.sha256(array.astype(np.float32).tobytes()).hexdigest() == p[key], (p["episode_index"], key)
            label_checks += 1
    assert label_checks == 130
    (out / "label_checks.json").write_text(json.dumps(dict(checks=label_checks, matched=label_checks, mismatches=[]), indent=2))
    sources, offsets, heights, source_uv, local_offsets = [], [], [], [], []
    for e in range(10):
        s, a = training.arrays(e)
        det = lid(training.decode(BODY, e)[0])
        if det is None:
            raise ValueError(f"source {e}: no lid")
        uv = det[0]; p = world(uv)[0]; source_uv.append(uv); sources.append(p[:2])
        close = first_close(robot, a)
        T = robot.tcp_from_state(s, "R")[close]; tcp = T[:3, 3]
        offsets.append(tcp[:2] - p[:2]); heights.append(tcp[2])
        local_offsets.append(np.linalg.solve(T, np.r_[p, 1])[:3])
    offset = np.median(offsets, axis=0); zgrasp = float(np.median(heights))
    local_offset = np.median(local_offsets, axis=0)
    rows, strips, traces = [], {}, {}
    for model in ("real", "mixed"):
        src = Source(dict(root=str(args.inputs / model), body_camera=BODY))
        for e in src.episode_ids:
            s, a = src.arrays(e); frames = src.decode(BODY, e); n = len(s)
            assert len(frames) == n and np.isfinite(s).all() and np.isfinite(a).all()
            close = first_close(robot, a); state_close = first_close(robot, s)
            if close is None:
                raise ValueError(f"{model} {e}: no close")
            initial = lid(frames[0])
            if initial is None:
                raise ValueError(f"{model} {e}: no initial lid")
            uv0, area0 = initial; p = world(uv0)[0]
            poses = robot.tcp_from_state(s, "R"); tcp = poses[:, :3, 3]
            dist = np.linalg.norm(tcp[:, :2] - p[:2] - offset, axis=1)
            near = (dist <= 0.05) & (np.abs(tcp[:, 2] - zgrasp) <= 0.05)
            approach = first_run(near[:close + 1])
            uv = np.full((close + 1, 2), np.nan); prev = uv0
            for t in range(close + 1):
                det = lid(frames[t], prev, area0)
                if det is not None:
                    uv[t] = det[0]; prev = det[0]
            delta = np.linalg.norm(uv - uv0, axis=1)
            move = first_run(delta > 5)
            group = next(g for g, ids in GROUPS if e in ids)
            sensitivity = max(np.linalg.norm(world(uv0 + d, z)[0][:2] - p[:2]) * 100
                              for d in ([5, 0], [-5, 0], [0, 5], [0, -5]) for z in (zlid - 0.01, zlid + 0.01))
            events = robot.gripper_events(a)
            lc = next((t for t in events["L"]["closes"] if t > close), None)
            ro = next((t for t in events["R"]["opens"] if t > close), None)
            row = dict(model=model, episode=e, group=group, frames=n, can_uv=uv0.tolist(), can_xy_cm=(100 * p[:2]).tolist(),
                       sensitivity_cm=float(sensitivity), action_close=close, state_close=state_close, close_s=close / src.fps,
                       state_close_minus_action_s=(state_close - close) / src.fps if state_close is not None else None,
                       approach=approach, approach_to_close_s=(close - approach) / src.fps if approach is not None else None,
                       tcp_close_xy_cm=(100 * tcp[close, :2]).tolist(), raw_error_cm=float(np.linalg.norm(tcp[close, :2] - p[:2]) * 100),
                       adjusted_error_cm=float(dist[close] * 100), lid_motion_frame=move,
                       orientation_adjusted_error_cm=float(np.linalg.norm((poses[close] @ np.r_[local_offset, 1])[:2] - p[:2]) * 100),
                       motion_before_close_s=(close - move) / src.fps if move is not None else None,
                       preclose_lid_missing_frac=float(np.isnan(uv).any(1).mean()), left_close=lc, right_release=ro,
                       handover_command_overlap_s=(ro - lc) / src.fps if lc is not None and ro is not None else None)
            rows.append(row); traces[f"{model}_{e}"] = dict(lid_uv=uv.tolist(), tcp_xyz_cm=(100 * tcp).tolist())
            times = [0, approach if approach is not None else max(0, close - 30), max(0, close - 10), close,
                     min(n - 1, close + 10), min(n - 1, lc if lc is not None else n - 1)]
            tiles = []
            for label, t in zip(("start", "approach", "pre-close", "close", "+10", "L-close"), times):
                im = frames[t].copy(); cv2.circle(im, tuple(uv0.astype(int)), 7, (0, 255, 255), 2)
                cv2.putText(im, f'{model} e{e} {label} {t/30:.2f}s', (8, 25), 0, 0.63, (255, 255, 255), 2)
                tiles.append(cv2.resize(im, (256, 192)))
            strips[model, e] = np.concatenate(tiles, 1)
            print(model, e, "close", close, "error cm", round(row["adjusted_error_cm"], 2), flush=True)
    for model in ("real", "mixed"):
        for group, ids in GROUPS:
            cv2.imwrite(str(out / f"sheets/{model}_{group}.jpg"), cv2.cvtColor(np.concatenate([strips[model, e] for e in ids], 0), cv2.COLOR_RGB2BGR))
    # All world coordinates use one camera and lid plane; native legacy poses have a different fitted height.
    synth_uv = np.array([p["can_initial_body_cam_px"] for p in prov if p["synthetic"]])
    synth = world(synth_uv)[:, :2]; realxy = np.array(sources)
    v01root = args.inputs / "training_v01"
    v01p = json.loads((v01root / "meta/sfa_provenance.json").read_text())["episodes"]
    v01 = []
    for p in v01p:
        z = np.load(args.work / f"openarm_pringles_handover/cache/recon_ep{p['source_real_episode']}.npz", allow_pickle=True)
        obj = json.loads(str(z["objects"]))[0]
        v01.append((np.array(obj["table"])[:3, 3] + np.array(p["shift_m"]))[:2])
    v01 = np.array(v01)
    coverage = []
    sets = {"real": realxy, "real_plus_v00": np.r_[realxy, synth], "real_plus_v01": np.r_[realxy, v01]}
    for r in rows:
        pt = np.array(r["can_xy_cm"]) / 100
        coverage.append({"model": r["model"], "episode": r["episode"], "group": r["group"],
                         **{k: {"nn_cm": float(np.linalg.norm(pts - pt, axis=1).min() * 100),
                                "inside_hull": bool(Delaunay(pts).find_simplex(pt) >= 0)} for k, pts in sets.items()}})
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    ax.scatter(100 * synth[:, 1], 100 * synth[:, 0], s=28, c="#ee9a39", label="v00: 65 synthetic (46 can variants)")
    ax.scatter(100 * v01[:, 1], 100 * v01[:, 0], s=24, marker="+", c="#a0a0a0", label=f"v01: {len(v01)} targets")
    ax.scatter(100 * realxy[:, 1], 100 * realxy[:, 0], s=60, c="#2465ad", label="10 real source starts")
    for i, (g, _) in enumerate(GROUPS):
        xy = np.array([r["can_xy_cm"] for r in rows if r["group"] == g])
        ax.scatter(xy[:, 1], xy[:, 0], s=28, edgecolors="black", c=[plt.cm.Set1(i)], marker="s", label=f"{i+1} {g}")
        m = xy.mean(0); ax.text(m[1] + 0.5, m[0], str(i+1), fontsize=15)
    ax.set(xlabel="World y (cm; robot left)", ylabel="World x (cm; forward)", title="Measured rollout starts and exact training coverage")
    ax.invert_xaxis(); ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.savefig(out / "coverage.png", dpi=150); plt.close(fig)
    summary = {}
    for model in ("real", "mixed"):
        summary[model] = {g: {key: estimate_ci([r[key] for r in rows if r["model"] == model and r["group"] == g and r[key] is not None])
                              for key in ("adjusted_error_cm", "orientation_adjusted_error_cm", "approach_to_close_s", "motion_before_close_s", "state_close_minus_action_s")}
                          for g, _ in GROUPS}
    outcomes = {"provenance": "User-reported group totals; Hub labels all null", "real": [1, 0, 0, 5, 4], "mixed": [2, 2, 2, 5, 4],
                "n": [4, 4, 3, 5, 4], "fisher_two_sided_p": float(fisher_exact([[10, 10], [15, 5]]).pvalue),
                "ci95": {m: list(binomtest(k, 20).proportion_ci()) for m, k in (("real", 10), ("mixed", 15))}}
    (out / "rollouts.json").write_text(json.dumps(dict(rows=rows, summary=summary, outcomes=outcomes, source_tcp_minus_can_xy_cm=(100 * offset).tolist(),
                                                     source_offsets_cm=(100 * np.array(offsets)).tolist(), source_local_can_points_m=np.array(local_offsets).tolist(),
                                                     grasp_height_m=zgrasp, lid_height_m=zlid), indent=2))
    pd.DataFrame(rows).to_csv(out / "rollouts.csv", index=False)
    (out / "traces.json").write_text(json.dumps(traces))
    (out / "coverage.json").write_text(json.dumps(dict(rows=coverage, real_xy_cm=(100 * realxy).tolist(), v00_xy_cm=(100 * synth).tolist(),
                                                     v01_xy_cm=(100 * v01).tolist(), source_uv=np.array(source_uv).tolist()), indent=2))
    inputs = [p for name in PINS for p in (args.inputs / name).rglob("*") if p.is_file() and ".cache" not in p.parts]
    hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (args.work / "openarm_pringles_handover/cache").glob("recon_ep*.npz")})
    hashes.update({str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   (Path(__file__), HERE / "results/eval_v00/PROTOCOL.md", HERE / "configs/openarm_pringles_handover.yaml",
                    HERE / "results/pringles/calibration.json", HERE / "results/pringles/table_height.json")})
    (out / "run_meta.json").write_text(json.dumps(dict(pins=PINS, git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                                     hashes=hashes, command=sys.argv), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, default=REPO / "artifacts/eval_v00_inputs")
    p.add_argument("--work", type=Path, default=Path.home() / "Stanley_ws/sfa_hf_export/sfagen/configs/work")
    p.add_argument("--out", type=Path, default=HERE / "results/eval_v00")
    p.add_argument("--fetch", action="store_true")
    run(p.parse_args())
