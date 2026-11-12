"""A: can-appearance fix benchmark. Protocol: results/canfix/PROTOCOL.md (frozen before rendering).

Same frames, pairs and regions as R1 (xfer_bench.py), restricted to one split, rendered with a render-config override.
usage: canfix_bench.py run DEVICE OUTDIR SPLIT '{"object_erode": 3}'   SPLIT = dev | heldout   (GPU)
       canfix_bench.py compare BASE_DIR CAND_DIR [--rule]               (CPU; paired gains, bootstrap, adoption rule)"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import xfer_bench as xb

SPLITS = {"dev": [0, 1, 2, 3, 4], "heldout": [5, 6, 7, 8, 9]}
PROV = ("source", "plate", "warped", "borrowed", "part_fill", "inpainted", "atlas")
N_BOOT, SEED = 2000, 0


def pair_means(rows, region, kind="cross"):
    g = {}
    for r in rows:
        if r["kind"] == kind and r["render"].get(region) is not None:
            g.setdefault((r["a"], r["b"]), []).append(r["render"][region])
    return {k: float(np.mean(v)) for k, v in g.items()}


def gains(base, cand, region, kind="cross"):
    """Paired candidate - baseline pair-level gains on identical frames."""
    b, c = pair_means(base, region, kind), pair_means(cand, region, kind)
    return {k: c[k] - b[k] for k in b if k in c}


def boot(g, stat, eps, n=N_BOOT, seed=SEED):
    """Episode-cluster bootstrap of stat(values, weights); weight = product of the two episodes' multiplicities."""
    keys = list(g); v = np.array([g[k] for k in keys]); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        cnt = Counter(rng.choice(eps, len(eps))); w = np.array([cnt[a] * cnt[b] if a != b else cnt[a] for a, b in keys], float)
        if w.sum() > 0:
            out.append(stat(v, w))
    return stat(v, np.ones(len(v))), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def rule_lift(base, cand, eps):
    """Amendment 1 (PROTOCOL.md): same-can relocation test; adoption also needs the lift criteria."""
    med = lambda v, w: xb._weighted_median(v, w)  # noqa: E731
    mean = lambda v, w: float(np.average(v, weights=w))  # noqa: E731
    can, can_ci = boot(gains(base, cand, "can", "lift"), med, eps)
    allg, all_ci = boot(gains(base, cand, "all", "lift"), mean, eps)
    return {"lift_can_median_gain_db": can, "lift_can_ci95": can_ci, "lift_all_mean_gain_db": allg, "lift_all_ci95": all_ci}


def rule(base, cand, eps):
    med = lambda v, w: xb._weighted_median(v, w)  # noqa: E731
    mean = lambda v, w: float(np.average(v, weights=w))  # noqa: E731
    can, can_ci = boot(gains(base, cand, "can"), med, eps)
    allg, all_ci = boot(gains(base, cand, "all"), mean, eps)
    ident = gains(base, cand, "all", "same_episode"); idm = float(np.mean(list(ident.values()))) if ident else None
    ok = [can >= 0.5 and can_ci[0] > 0, all_ci[0] >= -0.1, idm is None or idm >= -0.1]
    return {"can_median_gain_db": can, "can_ci95": can_ci, "all_mean_gain_db": allg, "all_ci95": all_ci,
            "identity_mean_gain_db": idm, "criteria": ok, "adopt": all(ok)}


def by_class(rows):
    """Can-region pixel provenance: fraction and mean squared error per class, per shift bin."""
    out = {}
    for r in rows:
        if r["kind"] != "cross" or "can_prov" not in r:
            continue
        k = xb.bin_of(r["disp_cm"]); d = out.setdefault(k, {c: [0, 0.0] for c in PROV})
        for c, (n, se) in r["can_prov"].items():
            d[c][0] += n; d[c][1] += se
    res = {}
    for k, d in sorted(out.items()):
        tot = sum(n for n, _ in d.values())
        res[f"{xb.EDGES[k]}-{xb.EDGES[k+1]}cm"] = {c: {"frac": n / tot, "psnr": (10 * np.log10(255 ** 2 / (se / n)) if n else None)} for c, (n, se) in d.items() if n}
    return res


def run(dev, out, split, override):
    import cv2, torch
    from _provenance import HERE, baseline_modules, git_state
    sys.path.insert(0, str(HERE))
    from sfagen.context import Context
    from sfagen.recon import Recon
    from sfagen.render import Renderer
    out.mkdir(parents=True, exist_ok=True)
    ctx = Context(HERE / "configs/openarm_pringles_handover.yaml"); eps = SPLITS[split]
    cfg = {**ctx.cfg.get("render", {}), **override}
    recs = {e: Recon(ctx, e, dev) for e in eps}
    (out / "run_meta.json").write_text(json.dumps({"git": git_state(), "code": baseline_modules(), "split": split, "render_cfg": cfg,
                                                   "protocol_sha256": __import__("hashlib").sha256((HERE / "results/canfix/PROTOCOL.md").read_bytes()).hexdigest()}, indent=1))
    med = {e: np.median(r.F[::3], 0).astype(np.int16) for e, r in recs.items()}
    fgm = lambda e, j: cv2.morphologyEx((np.abs(recs[e].F[j].astype(np.int16) - med[e]).max(2) > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0  # noqa: E731
    d5 = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # noqa: E731

    def frames_of(r):
        o = r.objects[0]; other = [s for s in r.arms if s != o["arm"]][0]; tg = int(o["grasp_frame"])
        tl = min([c for c in r.ev[other]["closes"] if c > tg], default=r.T - 1)
        return xb.video_frames(tg, tl, r.lag, r.T)
    F = {e: frames_of(r) for e, r in recs.items()}; pos = {e: np.asarray(r.objects[0]["table"])[:3, 3] for e, r in recs.items()}
    scene = lambda r, t: ({s: r.gv[s][t] for s in r.arms}, {s: float(r.grip_v[s][t]) for s in r.grip_v}, [r.obj_pose(0, t + r.lag)])  # noqa: E731

    def score(img, gt, ea, ja, eb, tb, prov):                 # identical regions to xfer_bench.run
        m = cv2.dilate((fgm(ea, ja) | fgm(eb, tb)).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        rb = recs[eb]; can = d5(rb.obj_sil(0, rb.obj_pose(0, tb + rb.lag))) & m
        robot = d5(rb.robot_sil(tb)) & m & ~can
        rev = d5(recs[ea].obj_sil(0, recs[ea].obj_pose(0, ja + recs[ea].lag))) & m & ~can & ~robot
        reg = {"all": m, "can": can, "robot": robot, "revealed": rev, "other": m & ~(can | robot | rev)}
        se = ((img.astype(float) - gt.astype(float)) ** 2).mean(2)
        cp = {PROV[c]: [int((can & (prov == c)).sum()), float(se[can & (prov == c)].sum())] for c in range(len(PROV))}
        res = {g: (xb.psnr(img, gt, mm) if mm.sum() >= 200 else None) for g, mm in reg.items()}
        if can.sum() >= 200:     # diagnostic: can PSNR after the best 2-D shift of the render (pose misalignment, not appearance)
            ys, xs = np.nonzero(can); best = (res["can"], 0, 0)
            for dy in range(-16, 17, 2):
                for dx in range(-16, 17, 2):
                    yy, xx = np.clip(ys - dy, 0, img.shape[0] - 1), np.clip(xs - dx, 0, img.shape[1] - 1)
                    v = 10 * np.log10(255 ** 2 / max(float(((img[yy, xx].astype(float) - gt[ys, xs]) ** 2).mean()), 1e-6))
                    if v > best[0]:
                        best = (v, dx, dy)
            res["can_aligned"], res["can_align_px"] = best[0], float(np.hypot(best[1], best[2]))
        return res, cp

    fo = open(out / "frames.jsonl", "w")
    for ea in eps:
        ra = recs[ea]; U = Renderer(ra, dev, cfg)
        for p in xb.PHASES:
            for j in F[ea][p]:
                t = min(j + xb.REF_K, ra.T - 1); q, g, objs = scene(ra, t)
                img, _ = U.render(j, q, g, objs, exclude=(t - 5, t + 5)); rr, _ = score(img, ra.F[t], ea, j, ea, t, U.last_prov)
                fo.write(json.dumps({"kind": "same_episode", "a": ea, "b": ea, "phase": p, "ja": j, "tb": t, "render": rr}) + "\n")
        tg = int(ra.objects[0]["grasp_frame"])               # amendment 1: same-can relocation (pre-grasp frame -> lifted scene)
        for jt in (tg - 15, tg - 10, tg - 5):
            for kt in (tg + 15, tg + 25, tg + 35):
                j, k = int(round(jt - ra.lag)), int(round(kt - ra.lag))
                if k >= ra.T or ra.holder_at(0, k + ra.lag) not in ra.arms:
                    continue
                q, g, objs = scene(ra, k); img, info = U.render(j, q, g, objs, exclude=(k - 20, k + 20))
                rr, cp = score(img, ra.F[k], ea, j, ea, k, U.last_prov)
                mv = float(np.linalg.norm(ra.obj_pose(0, k + ra.lag)[:3, 3] - ra.obj_pose(0, j + ra.lag)[:3, 3]) * 100)
                fo.write(json.dumps({"kind": "lift", "a": ea, "b": ea, "ja": j, "tb": k, "disp_cm": mv, "render": rr, "can_prov": cp, "info": info}) + "\n")
                if jt == tg - 10 and kt == tg + 25:
                    cv2.imwrite(str(out / f"lift_{ea}.jpg"), cv2.cvtColor(np.concatenate([ra.F[j], ra.F[k], img], 1), cv2.COLOR_RGB2BGR))
        for eb in eps:
            if eb == ea:
                continue
            rb = recs[eb]; disp = float(np.linalg.norm(pos[ea][:2] - pos[eb][:2]) * 100)
            for p in xb.PHASES:
                for i, (ja, tb) in enumerate(zip(F[ea][p], F[eb][p])):
                    q, g, objs = scene(rb, tb); img, info = U.render(ja, q, g, objs)
                    rr, cp = score(img, rb.F[tb], ea, ja, eb, tb, U.last_prov)
                    fo.write(json.dumps({"kind": "cross", "a": ea, "b": eb, "phase": p, "ja": ja, "tb": tb, "disp_cm": disp, "render": rr, "can_prov": cp, "info": info}) + "\n")
                    if p == "grasp" and i == 1:
                        cv2.imwrite(str(out / f"sheet_A{ea}_B{eb}.jpg"), cv2.cvtColor(np.concatenate([rb.F[tb], img], 1), cv2.COLOR_RGB2BGR))
            fo.flush()
        print("source", ea, "done", flush=True); del U; torch.cuda.empty_cache()
    fo.close()


def load(d):
    return [json.loads(l) for l in (Path(d) / "frames.jsonl").read_text().splitlines()]


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], Path(sys.argv[3]), sys.argv[4], json.loads(sys.argv[5]) if len(sys.argv) > 5 else {})
    else:
        base, cand = load(sys.argv[2]), load(sys.argv[3]); split = json.loads((Path(sys.argv[3]) / "run_meta.json").read_text())["split"]
        res = {"split": split, "base": sys.argv[2], "cand": sys.argv[3], **rule(base, cand, SPLITS[split]), **rule_lift(base, cand, SPLITS[split]),
               "can_provenance_base": by_class(base), "can_provenance_cand": by_class(cand)}
        if "--rule" not in sys.argv:
            res.pop("adopt"); res.pop("criteria")
        print(json.dumps(res, indent=1)); (Path(sys.argv[3]) / f"compare_vs_{Path(sys.argv[2]).name}.json").write_text(json.dumps(res, indent=1))
