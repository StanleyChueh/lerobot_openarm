"""Three-camera generation: body_cam from the v3 renderer (cached raw frames) + both wrist cameras synthesised by
sfa32.render_wrist, for v3 candidates. Short clips first (--clip), then complete episodes.

    cd research/sim_free_augmentation/v3_2
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/gen_wrist.py --episodes 0 3 8 --clip
    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/gen_wrist.py --episodes 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
V3 = HERE.parent / "v3"
sys.path.insert(0, str(V3))
sys.path.insert(0, str(HERE))
from sfa3 import recon3, render3  # noqa: E402
from sfa3.recon3 import data, rb  # noqa: E402
from sfa2 import render2  # noqa: E402
from sfa32 import render_wrist as rw, wristcam as wc  # noqa: E402
from sfa.recover import interp_q  # noqa: E402
from sfa.render import label, write_mp4  # noqa: E402

OUT = HERE / "outputs"
CACHE = OUT / "cache"
COLORS = np.array([[60, 160, 60], [40, 110, 230], [230, 150, 30], [150, 90, 220], [230, 60, 60], [40, 40, 40]], np.uint8)


def source_geometry(E, ep, robot, device):
    """WristSource for both wrist cameras (flows cached by episode)."""
    p = CACHE / f"ep{E}" / "wrist_flows.npz"
    from sfa import track as trk
    t0 = time.time()
    srcs = {}
    for side in "RL":
        F = ep.frames[wc.CAM_OF[side]]
        T = len(F)
        pairs = [(j, j + 3 if j + 3 < T else j - 3) for j in range(T)]
        fk = CACHE / f"ep{E}" / f"wrist_flow_{side}.npy"
        if fk.exists():
            fl = np.load(fk, mmap_mode="r")
        else:
            fk.parent.mkdir(parents=True, exist_ok=True)
            fl = trk.flow_pairs(F, pairs, device).astype(np.float16)
            np.save(fk, fl)
        flows = {pr: np.asarray(fl[i], np.float32) for i, pr in enumerate(pairs)}
        srcs[side] = rw.WristSource(ep, side, robot, flows)
    return srcs, time.time() - t0


def render_candidate(rec, robot, src, cand, frames=None):
    """generate both wrist videos for a candidate (dict with q, tau, can, state). Returns {side: (frames, labels)}."""
    T = rec.T
    frames = range(T) if frames is None else frames
    out = {}
    for side in "RL":
        S = src[side]
        lag = S.lag
        imgs, labs = [], []
        for i in frames:
            ts = i + lag
            j = int(np.clip(round(float(render2.tau_at(cand["tau"], ts)) - lag), 0, T - 1))
            Tn = wc.cam_pose(robot, interp_q(cand["q"]["R"], ts), interp_q(cand["q"]["L"], ts), side, S.X)
            can_src = rec.source_can(robot, j + lag)
            can_new = render2.can_at(cand["can"], ts)
            g_s = rb.Robot.split(rec.state)["g" + side][int(np.clip(round(j + lag), 0, T - 1))]
            g_n = rb.Robot.split(cand["state"])["g" + side][int(np.clip(round(ts), 0, T - 1))]
            holds = rec.source_holder(int(np.clip(round(j + lag), 0, T - 1))) != "table"
            im, lab = rw.render_frame(S, j, Tn, can_src, can_new, can_attached=bool(holds and g_s < rw.CLOSED and g_n < rw.CLOSED))
            imgs.append(im)
            labs.append(lab)
        out[side] = (np.stack(imgs), np.stack(labs))
    return out


def load_cand(E, key):
    z = np.load(V3 / f"outputs/ep{E}/candidates/{key}.npz")
    return {"state": z["state"], "action": z["action"], "valid": z["valid"], "can": z["can"], "tau": z["tau"],
            "q": {"R": z["qR"], "L": z["qL"]}, "table_new": z["table_new"]}


def body_frames(E, key, res):
    return np.load(V3 / f"outputs/cache/ep{E}/raw/{res['render_alias'][key]}.npy", mmap_mode="r")


def provenance_rgb(lab):
    return COLORS[lab]


def panel(ep_frames, gen, labs, body, t, idx, E, key):
    H, W = 240, 320
    sm = lambda im: cv2.resize(np.asarray(im), (W, H), interpolation=cv2.INTER_AREA)  # noqa: E731
    top = [label(sm(ep_frames["observation.images.body_cam"][t]), f"orig body t={t}"),
           label(sm(ep_frames[wc.CAM_OF["L"]][t]), "orig wrist_cam (L)"), label(sm(ep_frames[wc.CAM_OF["R"]][t]), "orig right_wrist_cam (R)")]
    mid = [label(sm(body[t]), f"gen body {key[:22]}"), label(sm(gen["L"][0][idx]), "gen wrist_cam (L)"),
           label(sm(gen["R"][0][idx]), "gen right_wrist_cam (R)")]
    bot = [label(sm(body[t] // 2), "provenance:"), label(sm(provenance_rgb(gen["L"][1][idx])), "L: grn reproj blu attach"),
           label(sm(provenance_rgb(gen["R"][1][idx])), "org can pur multiview red inpaint")]
    return np.concatenate([np.concatenate(r, 1) for r in (top, mid, bot)], 0)


G_ = {}


def full_job(key):
    """full-episode wrist synthesis for one candidate (fork-shared rec/robot/src)."""
    E, rec, src, d = G_["E"], G_["rec"], G_["src"], G_["dir"]
    robot = rb.Robot()
    t1 = time.time()
    cand = load_cand(E, key)
    gen = render_candidate(rec, robot, src, cand)
    np.save(CACHE / f"ep{E}" / f"wristL_{key}.npy", gen["L"][0])
    np.save(CACHE / f"ep{E}" / f"wristR_{key}.npy", gen["R"][0])
    np.savez_compressed(CACHE / f"ep{E}" / f"wristlab_{key}.npz", L=gen["L"][1], R=gen["R"][1])
    frac = {s: {k: float((gen[s][1] == v).mean()) for k, v in rw.LAB.items()} for s in "RL"}
    return key, {"render_s": time.time() - t1, "provenance_frac": frac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 3, 8])
    ap.add_argument("--clip", action="store_true", help="short 32-frame approach/handover clips only")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--keys", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    robot = rb.Robot()
    summary = {}
    for E in args.episodes:
        t0 = time.time()
        ep = data.load(E)
        rec = recon3.build(E, with_plate=False, log=lambda *a: None)
        src, t_src = source_geometry(E, ep, robot, args.device)
        res = json.loads((V3 / f"outputs/ep{E}/results.json").read_text())
        exp = json.loads((V3 / "outputs/dataset_export/export_checks.json").read_text())
        ok = [f"{v}__C__s{s}" for e2, v, s, f in exp["c_all"] if e2 == E]
        if args.clip:
            keys = args.keys or (["identity__C__s1.0"] + [k for k in ok if k.startswith(("init_x-6", "init_y-3", "pos_right_up", "init_x+5"))][:3])
        else:
            # representative full-episode set per source episode: identity control, up to two new-initial-position
            # variants, one new-position+hand-over variant and one original-position variant (all exported in v3)
            pick = [k for k in ok if k.startswith("init_") and "_pos_" not in k and "_grasp_" not in k and "_orient_" not in k][:2]
            pick += [k for k in ok if k.startswith("init_") and k not in pick][:1]
            pick += [k for k in ok if not k.startswith("init_")][:1]
            keys = args.keys or ["identity__C__s1.0"] + pick
        d = OUT / ("clips" if args.clip else "episodes") / f"ep{E}"
        d.mkdir(parents=True, exist_ok=True)
        ev = rec.ev
        if not args.clip:
            from multiprocessing import get_context
            G_.update(E=E, rec=rec, src=src, dir=d)
            t1 = time.time()
            with get_context("fork").Pool(args.workers) as P:
                for key, info in P.imap_unordered(full_job, keys):
                    summary.setdefault(E, {})[key] = info
            summary[E]["wall_render_s"] = time.time() - t1
            summary[E]["n_candidates"] = len(keys)
            summary[E]["source_geometry_s"] = t_src
            summary[E]["episode_s"] = time.time() - t0
            print(f"ep{E}: {len(keys)} candidates x 2 wrist cams in {time.time() - t1:.0f} s (source geometry {t_src:.0f} s)", flush=True)
            (OUT / "episode_summary.json").write_text(json.dumps(summary, indent=1))
            continue
        for key in keys:
            cand = load_cand(E, key)
            windows = {"approach": range(max(0, ev["right_close"] - 40), ev["right_close"] - 8),
                       "handover": range(ev["left_close"] - 20, min(rec.T, ev["left_close"] + 12))} if args.clip else {"full": range(rec.T)}
            body = body_frames(E, key, res)
            for wn, fr in windows.items():
                t1 = time.time()
                gen = render_candidate(rec, robot, src, cand, fr)
                t_r = time.time() - t1
                vid = [panel(ep.frames, gen, None, body, t, n, E, key) for n, t in enumerate(fr)]
                write_mp4(d / f"{key}_{wn}_3cam.mp4", vid, crf=22)
                frac = {s: {k: float((gen[s][1] == v).mean()) for k, v in rw.LAB.items()} for s in "RL"}
                summary.setdefault(E, {})[f"{key}_{wn}"] = {"frames": len(fr), "render_s": t_r, "provenance_frac": frac}
                if not args.clip:
                    np.savez_compressed(CACHE / f"ep{E}" / f"wrist_{key}.npz", L=gen["L"][0], R=gen["R"][0],
                                        labL=gen["L"][1], labR=gen["R"][1])
                print(f"ep{E} {key} {wn}: {len(fr)} frames, wrist render {t_r:.1f} s, provenance R {frac['R']}", flush=True)
        summary[E]["source_geometry_s"] = t_src
        summary[E]["episode_s"] = time.time() - t0
    (OUT / ("clip_summary.json" if args.clip else "episode_summary.json")).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
