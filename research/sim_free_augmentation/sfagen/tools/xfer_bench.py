"""R1: how far can the object move before body-camera RGB stops being valid? Protocol: results/r1_relocation/PROTOCOL.md.

Cross-episode pose transfer: re-render source A's frame with episode B's real joints and can pose; compare with B's real
frame. usage: xfer_bench.py run DEVICE OUTDIR      (GPU; renders and scores, writes frames.jsonl + summary.json + sheets)
              xfer_bench.py summarize OUTDIR       (CPU; recompute summary.json from frames.jsonl)
              xfer_bench.py wrist OUTDIR           (CPU; wrist_view_valid fraction vs shift from existing exports)"""
import itertools, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

EDGES = [0, 3, 5, 7, 9, 11, 15, 21]            # cm, frozen in PROTOCOL.md
PHASES = ("start", "approach", "grasp", "lift")
OFFSETS = (-2, 0, 2)
REF_K = 8                                       # same-episode ceiling: render frame j with the state at j + 8 (xbench k=8)
DELTA_DB = -0.5
N_BOOT, SEED = 2000, 0
REGIONS = ("all", "can", "robot", "revealed", "other")


# ---------------------------------------------------------------- pure helpers (CPU-tested)
def phase_times(t_grasp, t_left_close):
    """State times of the four phases (PROTOCOL.md table)."""
    return {"start": 5, "approach": int(round(0.6 * t_grasp)), "grasp": int(t_grasp),
            "lift": int(t_grasp + round(0.5 * (t_left_close - t_grasp)))}


def video_frames(t_grasp, t_left_close, lag, T):
    """{phase: [video frame for each offset]}; video frame = round(state time - lag), clipped to the episode."""
    return {p: [int(np.clip(round(t + o - lag), 0, T - 1)) for o in OFFSETS] for p, t in phase_times(t_grasp, t_left_close).items()}


def bin_of(d_cm):
    for i, (a, b) in enumerate(zip(EDGES, EDGES[1:])):
        if a <= d_cm < b or (b == EDGES[-1] and d_cm <= b):
            return i
    return None


def psnr(a, b, m):
    if m.sum() == 0:
        return None
    return float(10 * np.log10(255 ** 2 / max(float(((a.astype(float) - b.astype(float)) ** 2)[m].mean()), 1e-6)))


def _weighted_median(v, w):
    v, w = np.asarray(v, float), np.asarray(w, float); keep = w > 0; v, w = v[keep], w[keep]
    if not len(v):
        return np.nan
    o = np.argsort(v); v, c = v[o], np.cumsum(w[o])
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def bin_stats(pairs, n_boot=N_BOOT, seed=SEED, episodes=range(10)):
    """pairs: [{a, b, bin, render, copy}] (pair-level mean PSNR). Episode-cluster bootstrap (resample episodes, keep pairs
    with both ends drawn, weight = multiplicity product). Returns per-bin stats and r*."""
    eps = list(episodes); nb = len(EDGES) - 1
    A = np.array([p["a"] for p in pairs]); B = np.array([p["b"] for p in pairs]); K = np.array([p["bin"] for p in pairs])
    R = np.array([p["render"] for p in pairs]); C = np.array([p["copy"] for p in pairs])

    def stats(w):
        med = [_weighted_median(R[K == k], w[K == k]) for k in range(nb)]
        rc = [float(np.average(R[K == k] - C[K == k], weights=w[K == k])) if w[K == k].sum() > 0 else np.nan for k in range(nb)]
        return np.array(med), np.array(med) - med[0], np.array(rc)

    med, delta, rc = stats(np.ones(len(pairs)))
    rng = np.random.default_rng(seed); bd, brc = [], []
    for _ in range(n_boot):
        cnt = Counter(rng.choice(eps, len(eps))); w = np.array([cnt[a] * cnt[b] for a, b in zip(A, B)], float)
        _, d_, r_ = stats(w); bd.append(d_); brc.append(r_)
    bd, brc = np.array(bd), np.array(brc)
    ci = lambda x, k: [float(np.nanpercentile(x[:, k], 2.5)), float(np.nanpercentile(x[:, k], 97.5))]  # noqa: E731
    rows = []
    for k in range(nb):
        n = int((K == k).sum())
        ok = bool(n and delta[k] >= DELTA_DB and ci(brc, k)[0] > 0) if k else True
        rows.append({"bin_cm": [EDGES[k], EDGES[k + 1]], "pairs": n, "median_render_psnr": float(med[k]), "delta_vs_ref_db": float(delta[k]),
                     "delta_ci95": ci(bd, k) if k else [0.0, 0.0], "render_minus_copy_db": float(rc[k]), "render_minus_copy_ci95": ci(brc, k), "passes": ok})
    return rows, r_star(rows)


def r_star(rows):
    """Upper edge of the last bin in the unbroken run of passing bins starting at [3,5); 3 cm if [3,5) fails."""
    r = EDGES[1]
    for row in rows[1:]:
        if not (row["pairs"] and row["passes"]):
            break
        r = row["bin_cm"][1]
    return r


def decision(r):
    return "A: keep wide F1W shifts, fix the worst sub-region" if r >= 11 else ("B: anchor shifts within r* of recorded positions, spread sources" if r <= 7
                                                                              else "between: F1W planner capped at r*")


def summarize(out):
    rows = [json.loads(l) for l in (out / "frames.jsonl").read_text().splitlines()]
    cross = [r for r in rows if r["kind"] == "cross"]; ref = [r for r in rows if r["kind"] == "same_episode"]
    def pair_level(rs, region="all", phase=None):
        g = {}
        for r in rs:
            if phase and r["phase"] != phase:
                continue
            v, c = r["render"].get(region), r["copy"].get(region)
            if v is not None and c is not None:
                g.setdefault((r["a"], r["b"]), []).append((v, c, r["disp_cm"], r["dq_max"]))
        return [{"a": a, "b": b, "bin": bin_of(x[0][2]), "render": float(np.mean([y[0] for y in x])), "copy": float(np.mean([y[1] for y in x])),
                 "disp_cm": x[0][2], "dq": float(np.mean([y[3] for y in x]))} for (a, b), x in g.items()]
    prim = pair_level(cross); table, rs = bin_stats(prim)
    S = {"protocol_sha256": __import__("hashlib").sha256((Path(__file__).resolve().parents[1] / "results/r1_relocation/PROTOCOL.md").read_bytes()).hexdigest(),
         "frames": len(cross), "pairs": len(prim), "bins": table, "r_star_cm": rs, "decision": decision(rs),
         "same_episode_ceiling_psnr": float(np.mean([r["render"]["all"] for r in ref])) if ref else None,
         "same_episode_copy_psnr": float(np.mean([r["copy"]["all"] for r in ref])) if ref else None,
         "by_region": {g: [{"bin_cm": t["bin_cm"], "median_render_psnr": t["median_render_psnr"], "delta_vs_ref_db": t["delta_vs_ref_db"], "pairs": t["pairs"]}
                           for t in bin_stats(pair_level(cross, region=g), n_boot=200)[0]] for g in REGIONS[1:]},
         "by_phase": {p: [{"bin_cm": t["bin_cm"], "median_render_psnr": t["median_render_psnr"], "delta_vs_ref_db": t["delta_vs_ref_db"]}
                          for t in bin_stats(pair_level(cross, phase=p), n_boot=200)[0]] for p in PHASES},
         "joint_change": [{"dq_rad": p["dq"], "disp_cm": p["disp_cm"], "render": p["render"], "copy": p["copy"]} for p in prim]}
    meta = out / "run_meta.json"
    if meta.exists():
        S["run"] = json.loads(meta.read_text())
    (out / "summary.json").write_text(json.dumps(S, indent=1)); return S


# ---------------------------------------------------------------- GPU run
def run(dev, out):
    import cv2, torch
    from _provenance import HERE, baseline_modules, git_state
    sys.path.insert(0, str(HERE))
    from sfagen.context import Context
    from sfagen.recon import Recon
    from sfagen.render import Renderer
    from sfagen.bench import ssim_masked
    out.mkdir(parents=True, exist_ok=True); (out / "sheets").mkdir(exist_ok=True)
    ctx = Context(HERE / "configs/openarm_pringles_handover.yaml"); eps = list(range(10))
    recs = {e: Recon(ctx, e, dev) for e in eps}
    meta = {"git": git_state(), "code": baseline_modules(), "device": torch.cuda.get_device_name(dev), "config": str(ctx.config_path), "work": str(ctx.work)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    med = {e: np.median(r.F[::3], 0).astype(np.int16) for e, r in recs.items()}
    fgm = lambda e, j: cv2.morphologyEx((np.abs(recs[e].F[j].astype(np.int16) - med[e]).max(2) > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0  # noqa: E731
    d5 = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # noqa: E731

    def frames_of(r):
        o = r.objects[0]; other = [s for s in r.arms if s != o["arm"]][0]; tg = int(o["grasp_frame"])
        tl = min([c for c in r.ev[other]["closes"] if c > tg], default=r.T - 1)
        return video_frames(tg, tl, r.lag, r.T)
    F = {e: frames_of(r) for e, r in recs.items()}
    pos = {e: np.asarray(r.objects[0]["table"])[:3, 3] for e, r in recs.items()}

    def scene(r, t):
        return {s: r.gv[s][t] for s in r.arms}, {s: float(r.grip_v[s][t]) for s in r.grip_v}, [r.obj_pose(0, t + r.lag)]

    def score(img, gt, copy, ea, ja, eb, tb):
        m = cv2.dilate((fgm(ea, ja) | fgm(eb, tb)).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        rb = recs[eb]; can = d5(rb.obj_sil(0, rb.obj_pose(0, tb + rb.lag))) & m
        robot = d5(rb.robot_sil(tb)) & m & ~can
        rev = d5(recs[ea].obj_sil(0, recs[ea].obj_pose(0, ja + recs[ea].lag))) & m & ~can & ~robot
        reg = {"all": m, "can": can, "robot": robot, "revealed": rev, "other": m & ~(can | robot | rev)}
        f = lambda x: {g: (psnr(x, gt, mm) if mm.sum() >= 200 else None) for g, mm in reg.items()}  # noqa: E731
        return f(img), f(copy), {"render": ssim_masked(img, gt, m), "copy": ssim_masked(copy, gt, m)}, m

    fo = open(out / "frames.jsonl", "w"); sheet = {}
    for ea in eps:
        ra = recs[ea]; U = Renderer(ra, dev, ctx.cfg.get("render", {}))
        for p in PHASES:                                     # same-episode ceiling (xbench k = 8 style)
            for j in F[ea][p]:
                t = min(j + REF_K, ra.T - 1); q, g, objs = scene(ra, t)
                img, _ = U.render(j, q, g, objs, exclude=(t - 5, t + 5)); rr, cc, ss, _ = score(img, ra.F[t], ra.F[j], ea, j, ea, t)
                fo.write(json.dumps({"kind": "same_episode", "a": ea, "b": ea, "phase": p, "ja": j, "tb": t, "render": rr, "copy": cc, "ssim": ss}) + "\n")
        for eb in eps:
            if eb == ea:
                continue
            rb = recs[eb]; disp = float(np.linalg.norm(pos[ea][:2] - pos[eb][:2]) * 100)
            for p in PHASES:
                for ja, tb in zip(F[ea][p], F[eb][p]):
                    q, g, objs = scene(rb, tb)
                    img, info = U.render(ja, q, g, objs); gt, copy = rb.F[tb], ra.F[ja]
                    rr, cc, ss, m = score(img, gt, copy, ea, ja, eb, tb)
                    dq = float(max(np.abs(ra.gv[s][ja] - rb.gv[s][tb]).max() for s in ra.arms))
                    fo.write(json.dumps({"kind": "cross", "a": ea, "b": eb, "phase": p, "ja": ja, "tb": tb, "disp_cm": disp, "dq_max": dq,
                                         "render": rr, "copy": cc, "ssim": ss, "info": info}) + "\n")
                    k = bin_of(disp)
                    if p == "grasp" and ja == F[ea][p][1] and k not in sheet:         # one contact sheet per bin
                        edge = (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) - m.astype(np.uint8)) > 0
                        reg = gt.copy(); reg[edge] = (255, 0, 255)
                        sheet[k] = np.concatenate([gt, img, copy, reg], 1)
                        cv2.imwrite(str(out / f"sheets/bin{k}_{EDGES[k]}-{EDGES[k+1]}cm_A{ea}_B{eb}.jpg"), cv2.cvtColor(sheet[k], cv2.COLOR_RGB2BGR))
            fo.flush()
        print("source", ea, "done", flush=True); del U; torch.cuda.empty_cache()
    fo.close()
    return summarize(out)


def wrist(out):
    """Fraction of frames with wrist_view_valid == 0 per episode vs |shift| (published Pringles + F1W previews)."""
    import pandas as pd
    from _provenance import HERE
    rows = []
    for name in ("openarm_pringles_handover", "openarm_pringles_handover_workspace"):
        root = HERE / "configs/work" / name / "dataset"; prov = json.loads((root / "meta/sfa_provenance.json").read_text())["episodes"]
        df = pd.concat([pd.read_parquet(f, columns=["episode_index", "wrist_view_valid"]) for f in sorted((root / "data").rglob("*.parquet"))])
        for e, g in df.groupby("episode_index"):
            v = np.stack(g["wrist_view_valid"].to_numpy()); p = prov[int(e)]
            rows.append({"dataset": name, "episode": int(e), "shift_cm": float(np.linalg.norm(p["shift_m"][:2]) * 100), "family": p["family"],
                         "invalid_frac_left": float((v[:, 0] == 0).mean()), "invalid_frac_right": float((v[:, 1] == 0).mean()), "frames": len(v)})
    (out / "wrist_validity.json").write_text(json.dumps(rows, indent=1)); return rows


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        s = run(sys.argv[2], Path(sys.argv[3]))
    elif cmd == "summarize":
        s = summarize(Path(sys.argv[2]))
    else:
        s = wrist(Path(sys.argv[2])); print(len(s), "episodes"); sys.exit()
    print(json.dumps({k: s[k] for k in ("frames", "pairs", "r_star_cm", "decision", "same_episode_ceiling_psnr")}, indent=1))
    for b in s["bins"]:
        print(b["bin_cm"], b["pairs"], f'{b["median_render_psnr"]:.2f} dB', f'Δ {b["delta_vs_ref_db"]:+.2f} {np.round(b["delta_ci95"], 2).tolist()}',
              f'render-copy {b["render_minus_copy_db"]:+.2f} {np.round(b["render_minus_copy_ci95"], 2).tolist()}', "PASS" if b["passes"] else "fail")
