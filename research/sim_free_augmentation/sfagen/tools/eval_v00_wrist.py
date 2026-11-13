"""CPU audit of exact v00 wrist labels and camera-relative can geometry; no rendering."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

from rollout_report import HERE, REPO, Source, Robot, first_close
sys.path.insert(0, str(HERE))
from sfagen.recon import interp_rows
from sfagen.wrist import WristCam


def relative_pixel(C, P, K):
    p = (np.atleast_2d(P) - C[:3, 3]) @ C[:3, :3]
    return (p @ K.T)[:, :2] / p[:, 2, None], p[:, 2]


def cluster_ci(episodes, ph, key):
    groups = {}
    for r in episodes:
        if r[ph][key] is not None:
            groups.setdefault(r["source"], []).append(r[ph][key])
    if not groups:
        return None
    keys = list(groups); rng = np.random.default_rng(0)
    bs = [np.median(np.concatenate([groups[e] for e in rng.choice(keys, len(keys))])) for _ in range(2000)]
    return dict(n_sources=len(keys), n_descendants=sum(map(len, groups.values())),
                median=float(np.median(np.concatenate(list(groups.values())))), ci95=np.percentile(bs, [2.5, 97.5]).tolist())


def phase(t, tg, tl):
    if abs(t - tg) <= 5:
        return "grasp"
    if tl is not None and abs(t - tl) <= 5:
        return "handover"
    if 0.3 * tg <= t < 0.6 * tg:
        return "approach"
    if 0.6 * tg <= t < tg - 5:
        return "final_approach"
    return None


def main():
    inputs = REPO / "artifacts/eval_v00_inputs"
    out = HERE / "results/eval_v00"
    src = Source(dict(root=str(inputs / "training_v00"), body_camera="observation.images.body_cam"))
    prov = json.loads((src.root / "meta/sfa_provenance.json").read_text())["episodes"]
    cfg = yaml.safe_load((HERE / "configs/openarm_pringles_handover.yaml").read_text())
    robot = Robot(cfg["robot"], HERE / "configs")
    spec = yaml.safe_load((HERE / "configs/_openarm_wrist_cameras_refined.yaml").read_text())
    wc = WristCam("observation.images.right_wrist_cam", spec["observation.images.right_wrist_cam"], robot)
    arrays = {e: src.arrays(e) for e in range(10)}
    rows, episode_rows, checks, sheets = [], [], [], []
    for p in prov[10:]:
        e, se = p["episode_index"], p["source_real_episode"]
        S, A = src.arrays(e); SS, AA = arrays[se]
        mapping = np.array(p["wrist_source_rows"], int)
        assert len(mapping) == len(S) and mapping.min() >= 0 and mapping.max() < len(SS)
        tg, sg = first_close(robot, A), first_close(robot, AA)
        tl = next((t for t in robot.gripper_events(A)["L"]["closes"] if t > tg), None)
        q0 = robot.split(SS)
        at = min(sg + 8, len(SS) - 1)
        robot.fk({s: q0[s][at] for s in robot.arms})
        Tattach = robot.frame(robot.tcp["R"])
        psrc = np.array(p["source_can_initial_world_m"]); ptgt = np.array(p["can_initial_world_m"])
        # The lid centre is outside the wrist FoV near grasp; score visible can-surface points instead.
        th, zz = np.meshgrid(np.linspace(0, 2 * np.pi, 32, endpoint=False), np.linspace(-0.23, 0, 16))
        surface = np.stack([0.0375 * np.cos(th).ravel(), 0.0375 * np.sin(th).ravel(), zz.ravel()], 1)
        surface_src, surface_tgt = surface + psrc, surface + ptgt
        local = np.linalg.solve(Tattach, np.c_[surface_src, np.ones(len(surface_src))].T).T
        values = src.df[src.df.episode_index == e].sort_values("frame_index")
        valid = np.stack(values.wrist_view_valid.to_numpy())
        for t in range(len(S)):
            ph = phase(t, tg, tl)
            if ph is None:
                continue
            j = int(mapping[t])
            qs = robot.split(interp_rows(SS, [j + wc.lag])); qt = robot.split(interp_rows(S, [t + wc.lag]))
            transforms = []
            for q in (qs, qt):
                robot.fk({s: q[s][0] for s in robot.arms})
                transforms.append((robot.frame(wc.fid) @ wc.X, robot.frame(robot.tcp["R"])))
            (Cs, Ts), (Ct, Tt) = transforms
            # Rigid-attachment proxy after giver close+8; does not model slip/tilt during contact.
            Ps = surface_src if j + wc.lag < sg + 8 else (local @ Ts.T)[:, :3]
            Pt = surface_tgt if t + wc.lag < tg + 8 else (local @ Tt.T)[:, :3]
            us, zs = relative_pixel(Cs, Ps, wc.K); ut, zt = relative_pixel(Ct, Pt, wc.K)
            visible = ((zs > 0) & (zt > 0) & (us[:, 0] >= 0) & (us[:, 0] < wc.W) & (us[:, 1] >= 0) & (us[:, 1] < wc.H)
                       & (ut[:, 0] >= 0) & (ut[:, 0] < wc.W) & (ut[:, 1] >= 0) & (ut[:, 1] < wc.H))
            rows.append(dict(episode=e, source=se, variation=p["variation"], phase=ph, frame=t, source_frame=j,
                             right_invalid=int(valid[t, 1] == 0), camera_translation_cm=float(np.linalg.norm(Cs[:3, 3] - Ct[:3, 3]) * 100),
                             camera_rotation_deg=float(np.degrees(Rotation.from_matrix(Cs[:3, :3].T @ Ct[:3, :3]).magnitude())),
                             can_delta_px=float(np.median(np.linalg.norm(us[visible] - ut[visible], axis=1))) if visible.any() else None,
                             both_in_view=bool(visible.any()), visible_surface_points=int(visible.sum())))
        er = [r for r in rows if r["episode"] == e]
        episode_rows.append(dict(episode=e, source=se, variation=p["variation"], frames=len(S),
                                 right_invalid_frac=float((valid[:, 1] == 0).mean()), mapping_nonunit_steps=int((np.diff(mapping) != 1).sum()),
                                 **{ph: {k: float(np.median([r[k] for r in er if r["phase"] == ph and r[k] is not None]))
                                         if any(r["phase"] == ph and r[k] is not None for r in er) else None
                                         for k in ("camera_translation_cm", "camera_rotation_deg", "can_delta_px")}
                                    for ph in ("approach", "final_approach", "grasp", "handover")}))
        # A deterministic sample of actual copied RGB at the grasp, including both shift directions and handover edits.
        if e in (12, 13, 15, 27, 29, 35, 55, 65, 74):
            fs = src.decode(wc.name, se); ft = src.decode(wc.name, e)
            j = int(mapping[tg]); a, b = fs[j], ft[tg]
            mse = float(((a.astype(float) - b.astype(float)) ** 2).mean())
            checks.append(dict(episode=e, source=se, source_frame=j, target_frame=tg, copy_psnr_db=float(10 * np.log10(255**2 / max(mse, 1e-6)))))
            tiles = []
            for im, title in ((a, f"source {se} frame{j}"), (b, f"v00 e{e} {p['variation']} frame{tg}")):
                im = im.copy(); cv2.putText(im, title, (8, 25), 0, 0.65, (255, 255, 255), 2); tiles.append(cv2.resize(im, (320, 240)))
            sheets.append(np.concatenate(tiles, 1))
        print("wrist", e, flush=True)
    summary = {}
    for family, eps in (("all", episode_rows), ("object_position", [r for r in episode_rows if r["variation"].startswith("init_")]),
                        ("timing", [r for r in episode_rows if r["variation"].startswith("timing_")]),
                        ("handover_location", [r for r in episode_rows if not r["variation"].startswith(("init_", "timing_"))])):
        summary[family] = {ph: {k: cluster_ci(eps, ph, k)
                               for k in ("camera_translation_cm", "camera_rotation_deg", "can_delta_px")}
                           for ph in ("approach", "final_approach", "grasp", "handover")}
    pd.DataFrame(rows).to_csv(out / "wrist_frames.csv", index=False)
    (out / "wrist.json").write_text(json.dumps(dict(episodes=episode_rows, summary=summary, rgb_copy_checks=checks,
                                                   assumptions="Calibrated FK; provenance source-row mapping; median displacement of visible can-surface points from native legacy lid origin; rigid attachment after close+8; no slip model; CIs cluster source episodes."), indent=2))
    cv2.imwrite(str(out / "sheets/wrist_copied_rgb.jpg"), cv2.cvtColor(np.concatenate(sheets), cv2.COLOR_RGB2BGR))
    assert len(episode_rows) == 65 and len([r for r in episode_rows if r["variation"].startswith("init_")]) == 46
    assert all(np.isfinite(r["camera_translation_cm"]) for r in rows)


if __name__ == "__main__":
    main()
