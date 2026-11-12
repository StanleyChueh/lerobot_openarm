"""B2: photometric refinement of the wrist-camera mounts on dev T1 (protocol results/wrist/PROTOCOL.md).

Coordinate search around the initial mount: lag +-1 frame (0.5 steps), translation +-5 cm per axis (2.5 cm steps),
rotation +-3 deg per axis (1.5 deg steps), two sweeps. Objective: mean render PSNR on a T1 subsample (j every 20 frames,
k = j + 8, camera moved), 'plane' background depth for speed. Writes OUTDIR/refined_spec.yaml + search.json.
usage: wrist_calib.py DEVICE OUTDIR [SPLIT=dev]"""
import json, sys
from pathlib import Path
import numpy as np

BOUND = [1.0, 0.05, 0.05, 0.05, 3.0, 3.0, 3.0]   # lag frames, translation m, rotation deg (PROTOCOL.md)


def main(dev, out, split="dev"):
    import yaml, pinocchio as pin
    from _provenance import HERE, git_state
    sys.path.insert(0, str(HERE))
    from sfagen.context import Context
    from sfagen.recon import Recon
    from sfagen.wrist import WristCam, WristRenderer
    from wrist_bench import SPLITS, psnr, rot_deg
    out.mkdir(parents=True, exist_ok=True)
    ctx = Context(HERE / "configs/openarm_pringles_handover.yaml"); spec0 = yaml.safe_load((HERE / "configs/_openarm_wrist_cameras.yaml").read_text())
    eps = SPLITS[split]; recs = {e: Recon(ctx, e, dev) for e in eps}; log = {"git": git_state(), "split": split, "cameras": {}}; spec = {}
    for cam, sp in spec0.items():
        frames = {e: ctx.src.load(e, cams=[cam]).frames[cam] for e in eps}

        def score(p):                                    # p = (lag, tx, ty, tz, rx, ry, rz)
            s = dict(sp); s["lag"] = sp["lag"] + p[0]; s["pos"] = (np.array(sp["pos"]) + np.array(p[1:4])).tolist()
            s["R"] = (np.array(sp["R"]) @ pin.exp3(np.radians(np.array(p[4:7])))).tolist()
            wc = WristCam(cam, s, ctx.robot); v = []
            for e in eps:
                r = WristRenderer(recs[e], frames[e], wc, {"depth": "plane", "cache": 30}, dev)
                for j in range(20, r.T - 20, 20):
                    k = j + 8; Cj, Ck = r.source(j)["C"], r.source(k)["C"]
                    if np.linalg.norm(Cj[:3, 3] - Ck[:3, 3]) < 0.01 and rot_deg(Cj, Ck) < 2:
                        continue
                    q, g, o = r.scene(k + wc.lag); im, _ = r.render(j, q, g, o, exclude=(k - 5, k + 5)); v.append(psnr(im, r.F[k]))
            return float(np.mean(v)), s

        p = [0.0] * 7; best, _ = score(p); hist = [{"p": list(p), "psnr": best}]; base = best
        grid = [(0, [-1, -0.5, 0.5, 1])] + [(i, [-0.05, -0.025, 0.025, 0.05]) for i in (1, 2, 3)] + [(i, [-3, -1.5, 1.5, 3]) for i in (4, 5, 6)]
        for sweep in range(2):
            for i, steps in grid:
                for d in steps:
                    q = list(p); q[i] = float(np.clip(p[i] + (d if sweep == 0 else d / 2), -BOUND[i], BOUND[i]))   # frozen search box
                    if q[i] == p[i]:
                        continue
                    v, _ = score(q); hist.append({"p": q, "psnr": v})
                    if v > best + 1e-3:
                        best, p = v, q
            print(cam, "sweep", sweep, round(base, 3), "->", round(best, 3), p, flush=True)
        _, s = score(p); spec[cam] = s
        log["cameras"][cam] = {"initial_psnr": base, "refined_psnr": best, "gain_db": best - base, "offset_lag_t_m_r_deg": p, "evaluations": hist}
    (out / "refined_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False)); (out / "search.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), *(sys.argv[3:4]))
