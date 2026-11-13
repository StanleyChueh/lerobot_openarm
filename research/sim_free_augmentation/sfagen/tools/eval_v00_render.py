"""Bounded three-way renderer diagnosis, with the central frame of every R1 phase/pair.

360 cross-episode scenes (90 ordered pairs x 4 phases), plus existing +/-6 cm
variants. This exploratory subset does not rerun or change the registered R1 gate.
"""
import copy
import json
import shutil
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from _provenance import HERE, SFA, LEGACY, baseline_modules, git_state, sha256
import xfer_bench as xb
from canfix_bench import boot
sys.path.insert(0, str(HERE))
from sfagen.context import Context
from sfagen.recon import Recon, interp_rows
from sfagen.render import Renderer
from sfagen.data import Source
sys.path[:0] = [str(LEGACY / "render_v4"), str(SFA / "v3"), str(SFA / "v2"), str(SFA)]
import mgw
from sfa3 import recon3


def legacy_base(pose):
    """Legacy origin is the lid, local +z goes down; sfagen origin is the base, +z up."""
    X = np.diag([1., -1., -1., 1.]); X[2, 3] = 0.23
    return pose @ X


def main(device):
    out = HERE / "results/eval_v00/render"
    out.mkdir(parents=True, exist_ok=True)
    work = SFA.parents[1] / "artifacts/eval_v00_inputs/render_work"
    cache = work / "cache"; cache.mkdir(parents=True, exist_ok=True)
    oldwork = Path.home() / "Stanley_ws/sfa_hf_export/sfagen/configs/work/openarm_pringles_handover"
    for p in oldwork.glob("*.json"):
        shutil.copyfile(p, work / p.name)
    for e in range(10):
        for name in (f"recon_ep{e}.npz", f"sam2_ep{e}.npz"):
            shutil.copyfile(oldwork / "cache" / name, cache / name)
        assert recon3.lid_track_path(e).exists(), f"Missing legacy lid cache {e}"
    # A cache miss must fail; this diagnosis cannot write frozen caches or regenerate segmentation.
    def no_segment(*a, **kw):
        raise RuntimeError("Legacy SAM cache miss; stopped before generating or writing frozen evidence")
    recon3.recon2.sg.segment = no_segment
    ctx = Context(HERE / "configs/openarm_pringles_handover.yaml", work)
    recs, old = {}, {}
    for e in range(10):
        recs[e] = Recon(ctx, e, device)
        old[e] = recon3.build(e, device=device, with_plate=True, log=lambda *a: None)
        print("loaded", e, "lags", recs[e].lag, old[e].lag, flush=True)
    code = baseline_modules()
    models = {str(e): dict(table=old[e].table.tolist(), X={k: v.tolist() for k, v in old[e].X.items()},
                          events=old[e].ev, lag=old[e].lag, sfagen_table=recs[e].objects[0]["table"].tolist()) for e in range(10)}
    (out / "legacy_models.json").write_text(json.dumps(models, indent=2))
    frames = {}
    for e, r in recs.items():
        tg = int(r.objects[0]["grasp_frame"])
        tl = min([t for t in r.ev["L"]["closes"] if t > tg], default=r.T - 1)
        frames[e] = {p: js[1] for p, js in xb.video_frames(tg, tl, r.lag, r.T).items()}
    med = {e: np.median(r.F[::3], 0).astype(np.int16) for e, r in recs.items()}
    def fg(e, j):
        m = (np.abs(recs[e].F[j].astype(np.int16) - med[e]).max(2) > 30).astype(np.uint8)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    def scene(r, j):
        return {s: r.gv[s][j] for s in r.arms}, {s: float(r.grip_v[s][j]) for s in r.grip_v}
    def changed_rec(e):
        r = copy.copy(recs[e]); o = old[e]
        def obj_pose(self, k, t, Q=None):
            return legacy_base(o.source_can(o_robot, t))
        r.obj_pose = types.MethodType(obj_pose, r)
        return r
    # source_can expects the legacy Robot API; keep it independent of sfagen's mutable FK object.
    from sfa.robot import Robot as OldRobot
    o_robot = OldRobot()
    rows = []
    training = Source(dict(root=str(SFA.parents[1] / "artifacts/eval_v00_inputs/training_v00"), body_camera=ctx.src.body))
    prov = json.loads((training.root / "meta/sfa_provenance.json").read_text())["episodes"]
    for ea in range(10):
        ra = recs[ea]
        U = Renderer(ra, device, ctx.cfg["render"])
        P = Renderer(changed_rec(ea), device, ctx.cfg["render"])
        L = mgw.MGWForward(old[ea], device=device)
        for eb in range(10):
            if ea == eb:
                continue
            rb = recs[eb]
            for ph in xb.PHASES:
                ja, tb = frames[ea][ph], frames[eb][ph]
                q, g = scene(rb, tb)
                can_old = old[eb].source_can(o_robot, tb + rb.lag)
                imgs = {"legacy": L.render(ja, q, g, can_old)[0],
                        "sfagen": U.render(ja, q, g, [rb.obj_pose(0, tb + rb.lag)])[0],
                        "sfagen_lid": P.render(ja, q, g, [legacy_base(can_old)])[0]}
                mask = cv2.dilate((fg(ea, ja) | fg(eb, tb)).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
                canmask = cv2.dilate(rb.obj_sil(0, rb.obj_pose(0, tb + rb.lag)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                row = dict(a=ea, b=eb, phase=ph, ja=ja, tb=tb)
                for name, im in imgs.items():
                    row[name] = dict(all=xb.psnr(im, rb.F[tb], mask), can=xb.psnr(im, rb.F[tb], canmask & mask))
                rows.append(row)
                if ph == "grasp" and eb == (ea + 1) % 10:
                    tiles = []
                    for name, im in [("target real", rb.F[tb]), *imgs.items()]:
                        im = im.copy(); cv2.putText(im, name, (10, 25), 0, 0.8, (255, 255, 255), 2); tiles.append(cv2.resize(im, (320, 240)))
                    cv2.imwrite(str(out / f"cross_A{ea}_B{eb}.jpg"), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
        # Existing v00 labels, same target joints/source frame and physical 6 cm shift in all three pipelines.
        for p in prov[10:]:
            if p["source_real_episode"] != ea or not p["variation"].startswith("init_"):
                continue
            shift = np.array(p["can_initial_world_m"]) - p["source_can_initial_world_m"]
            if not np.isclose(np.linalg.norm(shift[:2]), 0.06, atol=0.002):
                continue
            S, A = training.arrays(p["episode_index"])
            sp = ctx.robot.split(S); mapping = np.array(p["wrist_source_rows"])
            tg = ctx.robot.gripper_events(A)["R"]["closes"][0]
            ti = max(0, tg - 5); ja = int(mapping[ti])
            qt = {s: interp_rows(sp[s], [ti + ra.lag])[0] for s in ra.arms}
            gt = {s: float(interp_rows(sp["g" + s][:, None], [ti + ra.lag])[0, 0]) for s in ra.arms}
            can_old = old[ea].source_can(o_robot, ja + ra.lag).copy(); can_old[:3, 3] += shift
            can_new = ra.obj_pose(0, ja + ra.lag).copy(); can_new[:3, 3] += shift
            imgs = [L.render(ja, qt, gt, can_old)[0], U.render(ja, qt, gt, [can_new])[0], P.render(ja, qt, gt, [legacy_base(can_old)])[0]]
            tiles = []
            for name, im in zip(("legacy", "sfagen", "sfagen + lid pose"), imgs):
                im = im.copy(); cv2.putText(im, f"{name} e{ea} {p['variation']}", (10, 25), 0, 0.7, (255, 255, 255), 2); tiles.append(cv2.resize(im, (320, 240)))
            cv2.imwrite(str(out / f"shift_e{ea}_{p['variation']}.jpg"), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
        (out / "frames.json").write_text(json.dumps(rows, indent=2))
        print("render source", ea, "scenes", len(rows), flush=True)
        del U, P, L; torch.cuda.empty_cache()
    summary = {}
    for region in ("all", "can"):
        summary[region] = {}
        for left, right in (("sfagen", "legacy"), ("sfagen_lid", "sfagen"), ("sfagen_lid", "legacy")):
            gains = {(a, b): float(np.mean([r[left][region] - r[right][region] for r in rows if r["a"] == a and r["b"] == b]))
                     for a in range(10) for b in range(10) if a != b}
            mean, ci = boot(gains, lambda v, w: float(np.average(v, weights=w)), list(range(10)))
            summary[region][f"{left}_minus_{right}"] = dict(mean_db=mean, ci95=ci)
        summary[region]["mean_psnr_db"] = {n: float(np.mean([r[n][region] for r in rows])) for n in ("legacy", "sfagen", "sfagen_lid")}
    (out / "summary.json").write_text(json.dumps(dict(scenes=len(rows), pairs=90, central_offset_only=True, metrics=summary), indent=2))
    (out / "run_meta.json").write_text(json.dumps(dict(git=git_state(), code=code, device=torch.cuda.get_device_name(device),
                                                      protocol_sha256=sha256(out.parent / "PROTOCOL.md"),
                                                      models_sha256=sha256(out / "legacy_models.json")), indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
