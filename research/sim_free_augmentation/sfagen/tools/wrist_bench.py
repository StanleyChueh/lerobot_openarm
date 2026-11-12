"""B: wrist-camera rendering benchmark. Protocol: results/wrist/PROTOCOL.md (frozen before any wrist rendering).

usage: wrist_bench.py run DEVICE OUTDIR TASK SPLIT ['{"depth": "da2"}'] [--spec configs/_openarm_wrist_cameras.yaml] [--t1]
           TASK = pringles | battery; SPLIT = dev (0-4) | heldout (5-9) | all (0-9); --t1 = skip T2      (GPU)
       wrist_bench.py summarize OUTDIR                                                                  (CPU; gate)"""
import itertools, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

CONFIGS = {"pringles": "configs/openarm_pringles_handover.yaml", "battery": "configs/openarm_battery_insertion.yaml"}
SPLITS = {"dev": [0, 1, 2, 3, 4], "heldout": [5, 6, 7, 8, 9], "all": list(range(10))}
GRID, DS, OFFSETS, PHASES = 10, (4, 8, 15), (-2, 0, 2), ("start", "approach", "grasp", "lift")
EDGES = [0, 3, 6, 11, 30]; CAP_CM, MAX_ROT, N_BOOT, SEED = 11, 5.0, 2000, 0
REGIONS = ("all", "own_arm", "other_arm", "objects", "world")


def psnr(a, b, m=None):
    e = (a.astype(float) - b.astype(float)) ** 2
    e = e.mean(2)[m] if m is not None else e
    if e.size == 0:
        return None
    return float(10 * np.log10(255 ** 2 / max(float(e.mean()), 1e-6)))


def rot_deg(A, B):
    return float(np.degrees(np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1))))


def phase_frames(t_g, t_l, lag, T):
    t = {"start": 5, "approach": int(round(0.6 * t_g)), "grasp": int(t_g), "lift": int(t_g + round(0.5 * (t_l - t_g)))}
    return {p: [int(np.clip(round(v + o - lag), 0, T - 1)) for o in OFFSETS] for p, v in t.items()}


def bin_of(cm):
    for i, (a, b) in enumerate(zip(EDGES, EDGES[1:])):
        if a <= cm < b or (b == EDGES[-1] and cm <= b):
            return i
    return None


def boot_mean(keys, vals, eps, n=N_BOOT, seed=SEED):
    """Episode-cluster bootstrap of the mean; key = (a, b); weight = multiplicity product (a == b: multiplicity)."""
    v = np.asarray(vals, float); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        c = Counter(rng.choice(eps, len(eps))); w = np.array([c[a] * c[b] if a != b else c[a] for a, b in keys], float)
        if w.sum() > 0:
            out.append(float(np.average(v, weights=w)))
    return float(v.mean()), [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else [np.nan, np.nan]


def summarize(out):
    rows = [json.loads(l) for l in (out / "frames.jsonl").read_text().splitlines()]
    meta = json.loads((out / "run_meta.json").read_text()); eps = SPLITS[meta["split"]]; S = {"meta": meta, "cameras": {}}
    for cam in sorted({r["cam"] for r in rows}):
        R = [r for r in rows if r["cam"] == cam]; res = {}
        t1 = {}
        for r in (x for x in R if x["kind"] == "T1"):
            t1.setdefault(r["a"], []).append((r["render"]["all"], r["copy"]["all"]))
        k1 = [(e, e) for e in t1]; g1 = [np.mean([a - b for a, b in v]) for v in t1.values()]
        m, ci = boot_mean(k1, g1, eps) if k1 else (None, [None, None])
        res["T1"] = {"frames": sum(len(v) for v in t1.values()), "render_psnr": float(np.mean([a for v in t1.values() for a, _ in v])) if t1 else None,
                     "copy_psnr": float(np.mean([b for v in t1.values() for _, b in v])) if t1 else None, "render_minus_copy_db": m, "ci95": ci,
                     "by_region": {g: float(np.nanmean([(x["render"][g] - x["copy"][g]) for x in R if x["kind"] == "T1" and x["render"].get(g) is not None and x["copy"].get(g) is not None] or [np.nan])) for g in REGIONS}}
        pairs = {}
        for r in (x for x in R if x["kind"] == "T2" and x["rot_deg"] <= MAX_ROT):
            pairs.setdefault((r["a"], r["b"]), []).append(r)
        bins = []
        for k in range(len(EDGES) - 1):
            P = {key: v for key, v in pairs.items() if bin_of(float(np.mean([x["trans_cm"] for x in v]))) == k}
            g = [float(np.mean([x["render"]["all"] - x["copy"]["all"] for x in v])) for v in P.values()]
            m2, ci2 = boot_mean(list(P), g, eps) if P else (None, [None, None])
            bins.append({"bin_cm": EDGES[k:k + 2], "pairs": len(P), "frames": sum(len(v) for v in P.values()),
                         "render_psnr": float(np.mean([x["render"]["all"] for v in P.values() for x in v])) if P else None,
                         "copy_psnr": float(np.mean([x["copy"]["all"] for v in P.values() for x in v])) if P else None,
                         "render_minus_copy_db": m2, "ci95": ci2})
        res["T2"] = {"bins": bins, "frames_rot_ok": sum(len(v) for v in pairs.values()), "frames_total": sum(1 for x in R if x["kind"] == "T2")}
        tested = [b for b in bins if b["bin_cm"][1] <= CAP_CM and b["pairs"] >= 3]
        res["gate"] = {"T1_ok": ci[0] is not None and ci[0] > 0, "T2_ok": bool(tested) and all(b["ci95"][0] > 0 for b in tested),
                       "T2_bins_tested": [b["bin_cm"] for b in tested]}
        res["gate"]["pass"] = res["gate"]["T1_ok"] and res["gate"]["T2_ok"]
        inp = [x["info"].get("inpainted_frac", 0) for x in R if "info" in x]; res["inpainted_frac_median"] = float(np.median(inp)) if inp else None
        S["cameras"][cam] = res
    (out / "summary.json").write_text(json.dumps(S, indent=1)); return S


def run(dev, out, task, split, rcfg, spec_path, t1_only):
    import cv2, torch, yaml
    from _provenance import HERE, baseline_modules, git_state
    sys.path.insert(0, str(HERE))
    from sfagen.context import Context
    from sfagen.recon import Recon
    from sfagen.wrist import WristCam, WristRenderer
    out.mkdir(parents=True, exist_ok=True); (out / "sheets").mkdir(exist_ok=True)
    ctx = Context(HERE / CONFIGS[task]); spec = yaml.safe_load((HERE / spec_path).read_text()); eps = SPLITS[split]
    rcfg = {"cache": 40, **rcfg}
    (out / "run_meta.json").write_text(json.dumps({"git": git_state(), "code": baseline_modules(), "task": task, "split": split, "render_cfg": rcfg, "spec": spec,
                                                   "protocol_sha256": __import__("hashlib").sha256((HERE / "results/wrist/PROTOCOL.md").read_bytes()).hexdigest()}, indent=1))
    recs = {e: Recon(ctx, e, dev) for e in eps}; fo = open(out / "frames.jsonl", "w")
    pal = np.array([[0, 0, 0], [0, 0, 0], [0, 255, 0], [255, 255, 0], [0, 0, 0], [255, 0, 0]], np.uint8)
    for cam, sp in spec.items():
        wc = WristCam(cam, sp, ctx.robot); R = {}
        for e in eps:
            R[e] = WristRenderer(recs[e], ctx.src.load(e, cams=[cam]).frames[cam], wc, rcfg, dev)

        def regions(r, k):
            lab = r.source(k)["lab"].cpu().numpy(); side = np.array(r.G.side + ["X"] * (r.K + 1))
            own = (lab >= 0) & (lab < r.OBJ0) & (side[np.clip(lab, 0, None)] == wc.arm); robot = (lab >= 0) & (lab < r.OBJ0)
            return {"all": None, "own_arm": own, "other_arm": robot & ~own, "objects": lab >= r.OBJ0, "world": lab < 0}

        def score(img, gt, cp, reg):
            f = lambda x: {g: (psnr(x, gt, m) if m is None or m.sum() >= 500 else None) for g, m in reg.items()}  # noqa: E731
            return f(img), f(cp)

        def sheet(name, gt, cp, img, prov):
            cv2.imwrite(str(out / "sheets" / name), cv2.cvtColor(np.concatenate([gt, cp, img, (0.5 * img + 0.5 * pal[prov]).astype(np.uint8)], 1)[::2, ::2], cv2.COLOR_RGB2BGR))

        for e in eps:                                          # T1: same-episode novel view
            r = R[e]
            for j in range(GRID, r.T - 20, GRID):
                for d in DS:
                    k = j + d
                    if k >= r.T:
                        continue
                    Cj, Ck = r.source(j)["C"], r.source(k)["C"]; tr = float(np.linalg.norm(Cj[:3, 3] - Ck[:3, 3]) * 100); ro = rot_deg(Cj, Ck)
                    if tr < 1 and ro < 2:
                        continue
                    q, g, objs = r.scene(k + wc.lag); img, info = r.render(j, q, g, objs, exclude=(k - 5, k + 5))
                    rr, cc = score(img, r.F[k], r.F[j], regions(r, k))
                    fo.write(json.dumps({"kind": "T1", "cam": cam, "a": e, "b": e, "j": j, "k": k, "d": d, "trans_cm": tr, "rot_deg": ro, "render": rr, "copy": cc, "info": info}) + "\n")
                    if j == 100 and d == 15:
                        sheet(f"T1_{cam.split('.')[-1]}_ep{e}_j{j}.jpg", r.F[k], r.F[j], img, r.last_prov)
            fo.flush(); print(cam, "T1 ep", e, flush=True)
        if t1_only:
            continue
        F = {}
        for e in eps:
            o = recs[e].objects[0]; other = [s for s in recs[e].arms if s != o["arm"]][0]; tg = int(o["grasp_frame"])
            tl = min([c for c in recs[e].ev[other]["closes"] if c > tg], default=recs[e].T - 1)
            F[e] = phase_frames(tg, tl, wc.lag, R[e].T)
        for a, b in itertools.permutations(eps, 2):            # T2: cross-episode, generation-like
            ra, rb = R[a], R[b]
            for p in PHASES:
                for i, (ja, tb) in enumerate(zip(F[a][p], F[b][p])):
                    Ca, Cb = ra.source(ja)["C"], rb.source(tb)["C"]; tr = float(np.linalg.norm(Ca[:3, 3] - Cb[:3, 3]) * 100); ro = rot_deg(Ca, Cb)
                    rec_ = {"kind": "T2", "cam": cam, "a": a, "b": b, "phase": p, "j": ja, "k": tb, "trans_cm": tr, "rot_deg": ro}
                    if ro > MAX_ROT:
                        fo.write(json.dumps(rec_) + "\n"); continue
                    q, g, objs = rb.scene(tb + wc.lag); img, info = ra.render(ja, q, g, objs)
                    rr, cc = score(img, rb.F[tb], ra.F[ja], regions(rb, tb))
                    fo.write(json.dumps({**rec_, "render": rr, "copy": cc, "info": info}) + "\n")
                    if i == 1:
                        sheet(f"T2_{cam.split('.')[-1]}_A{a}_B{b}_{p}.jpg", rb.F[tb], ra.F[ja], img, ra.last_prov)
            fo.flush()
        print(cam, "T2 done", flush=True); del R; torch.cuda.empty_cache()
    fo.close()
    return summarize(out)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "run":
        spec = a[a.index("--spec") + 1] if "--spec" in a else "configs/_openarm_wrist_cameras.yaml"
        pos = [x for i, x in enumerate(a[2:]) if not x.startswith("--") and a[1 + i] != "--spec"]
        s = run(pos[0], Path(pos[1]), pos[2], pos[3], json.loads(pos[4]) if len(pos) > 4 else {}, spec, "--t1" in a)
    else:
        s = summarize(Path(a[2]))
    for cam, r in s["cameras"].items():
        print(cam, "T1", {k: r["T1"][k] for k in ("frames", "render_psnr", "copy_psnr", "render_minus_copy_db", "ci95")}, "gate", r["gate"])
        for b in r["T2"]["bins"]:
            print("   T2", b)
