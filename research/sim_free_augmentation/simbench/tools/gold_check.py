"""Evaluation only: score the masks of a work dir against the hand-checked gold set (R06 protocol, gate G1-G3).
Gated masks = the renderer's masks (they decide which pixels move); the reconstruction masks, the raw SAM2 masks and the
projected URDF silhouette are reported next to them to show where an error comes from. Never writes into a work dir.
Usage: gold_check.py GOLD_DIR OUT_PREFIX [--device cuda:0] NAME=CONFIG[@WORK_DIR] ...   (NAME as in GOLD_DIR/frames.json)"""
import json, sys
import numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.context import Context
from sfagen.recon import Recon
from sfagen.render import Renderer
from sfagen.segment import cache_name
from sfagen import mesh

a = sys.argv[1:]; gd = Path(a.pop(0)); out = a.pop(0); dev = "cuda:0"
if a[0] == "--device":
    a.pop(0); dev = a.pop(0)
frames = json.loads((gd / "frames.json").read_text()); rows, tiles = [], []
frac = lambda x, y: float((x & y).sum() / max(1, y.sum()))  # noqa: E731
for job in a:
    name, spec = job.split("=", 1); cfg, work = (spec.split("@") + [None])[:2]; ctx = Context(cfg, work)
    rc = ctx.cfg.get("reconstruction", {}); cut = rc.get("prompt_object_cut", "neighbourhood")
    for e in sorted({f["episode"] for f in frames.values() if f["name"] == name}):
        sam_p = ctx.cache_dir / cache_name(rc, e, bool(ctx.cfg.get("objects")))
        if not ((ctx.cache_dir / f"recon_ep{e}.npz").exists() and sam_p.exists()):
            print(name, "ep", e, "cache missing, skipped"); continue
        rec = Recon(ctx, e, dev); G, rob = rec.G, rec.robot; R = Renderer(rec, dev, ctx.cfg.get("render", {})); raw = np.load(sam_p)
        sc = rec.W / 640; side_of = np.array(G.side + ["X"]); grip = {}
        for s in rec.arms:                                   # gripper = geometries at or below the arm's last listed joint
            last = rob.m.getJointId(rob.cfg["arms"][s]["joints"][-1]); grip[s] = []
            for i, g in enumerate(rob.vis.geometryObjects):
                jt = g.parentJoint
                while jt > 0 and jt != last:
                    jt = rob.m.parents[jt]
                if jt == last:
                    grip[s].append(i)
        for fid, f in sorted(frames.items()):
            if f["name"] != name or f["episode"] != e or not (gd / f"{fid}.npz").exists():
                continue
            j = f["frame"]; z = np.load(gd / f"{fid}.npz"); gold = {l: z[l] for l in z.files if l[0] != "_"}; r = max(1, round(float(z["_band"]) * sc))
            k = np.ones((2 * r + 1, 2 * r + 1), np.uint8); valid = ~gold.get("ignore", np.zeros((rec.H, rec.W), bool))
            for l, m in gold.items():                        # the band around every gold edge is not scored
                if l != "ignore" and m.any():
                    valid &= ~((cv2.dilate(m.astype(np.uint8), k) > 0) & ~(cv2.erode(m.astype(np.uint8), k) > 0))
            cat = R.source_state(j)["cat"].cpu().numpy()
            lab = mesh.splat(G.world(G.placements({s: rec.gv[s][j] for s in rec.arms}, {s: float(rec.grip_v[s][j]) for s in rec.grip_v})), G.gid, rec.cam, dev=dev)[1].cpu().numpy()
            grobot = np.any([m for l, m in gold.items() if l.startswith("arm_")], 0) if any(l.startswith("arm_") for l in gold) else np.zeros_like(valid)
            row = {"frame": fid, "kind": f["kind"], "band_px": r, "checks": {}, "fail": []}
            for s in rec.arms:
                g = gold.get("arm_" + s, np.zeros_like(valid)); gg = gold.get("grip_" + s, np.zeros_like(valid))
                pr = {"render": cat == R.cats.index(s), "recon": rec.masks[s][j] > 0, "sam2": raw[s][j] > 0, "projection": side_of[lab] == s}
                c = {"gold_px": int(g.sum()), "grip_px": int(gg.sum())}
                for n, p in pr.items():
                    c[n] = {"recall": frac(p, g & valid), "precision": frac(g, p & valid), "grip_recall": frac(p, gg & valid), "px": int(p.sum())}
                pg = np.isin(lab, grip[s])
                if gg.sum() >= 80 * sc * sc and pg.sum() >= 40 * sc * sc:      # how far the projected gripper sits from the gold gripper
                    d = np.array(np.nonzero(gg)).mean(1) - np.array(np.nonzero(pg)).mean(1); c["proj_grip_shift_px"] = [float(d[1]), float(d[0]), float(np.hypot(*d))]
                c["object_px_on_gold_arm"] = int(sum(((cat == R.cats.index(f"obj{o}")) & g & valid).sum() for o in range(len(rec.objects))))
                if g.sum() >= 400 * sc * sc:
                    if c["render"]["recall"] < 0.95: row["fail"].append(f"G1 arm {s} recall {c['render']['recall']:.2f}")
                    if c["render"]["precision"] < 0.95: row["fail"].append(f"G1 arm {s} precision {c['render']['precision']:.2f}")
                elif (pr["render"] & valid).sum() > 100 * sc * sc:
                    row["fail"].append(f"G1 arm {s} not in gold, {int((pr['render'] & valid).sum())} px predicted")
                if gg.sum() >= 80 * sc * sc and c["render"]["grip_recall"] < 0.95:
                    row["fail"].append(f"G2 gripper {s} recall {c['render']['grip_recall']:.2f}")
                row["checks"]["arm_" + s] = c
            po = [cat == R.cats.index(f"obj{o}") for o in range(len(rec.objects))]
            for l in sorted(l for l in gold if l.startswith("obj_") and gold[l].sum() >= 80 * sc * sc):     # gold object -> best predicted object
                g = gold[l]; iou = [float((p & g & valid).sum() / max(1, ((p | g) & valid).sum())) for p in po] or [0.0]; b = int(np.argmax(iou))
                on = frac(grobot, po[b] & valid) if po else 0.0
                row["checks"][l] = {"gold_px": int(g.sum()), "iou": iou[b], "pred": b, "on_gold_robot": on, "px": int(po[b].sum()) if po else 0}
                if iou[b] < 0.90: row["fail"].append(f"G3 {l} IoU {iou[b]:.2f}")
                if on > 0.01: row["fail"].append(f"G3 {l} {100 * on:.0f} % on robot")
            rows.append(row); print(("FAIL " if row["fail"] else "pass ") + fid, "; ".join(row["fail"]))
            im = rec.F[j].copy(); v = im.copy()              # overlay: renderer masks filled, gold edges drawn
            for i_, col in enumerate([(40, 90, 255), (0, 200, 90), (255, 140, 0), (255, 0, 200), (0, 220, 220)][:len(R.cats)]):
                v[cat == i_] = (0.5 * v[cat == i_] + 0.5 * np.array(col)).astype(np.uint8)
            for l, m in gold.items():
                cv2.drawContours(v, cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0], -1,
                                 (255, 0, 255) if l == "ignore" else (255, 255, 0) if l.startswith("grip") else (255, 255, 255), 1)
            v = cv2.resize(v, (640, round(640 * rec.H / rec.W)))
            cv2.putText(v, fid + ("  FAIL" if row["fail"] else "  pass"), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0) if row["fail"] else (0, 160, 0), 1, cv2.LINE_AA)
            for n_, t in enumerate(row["fail"][:6]):
                cv2.putText(v, t, (6, 34 + 15 * n_), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
            tiles.append(v)
        del R, rec
nf = sum(bool(r["fail"]) for r in rows); print(f"\n{nf} of {len(rows)} gold frames fail")
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out + ".json").write_text(json.dumps({"frames": len(rows), "failing": nf, "rows": rows}, indent=1))
if tiles:
    t = [cv2.resize(x, (640, 480)) for x in tiles]
    cv2.imwrite(out + ".jpg", cv2.cvtColor(np.concatenate([np.concatenate(t[k:k + 3] + [np.zeros_like(t[0])] * (3 - len(t[k:k + 3])), 1) for k in range(0, len(t), 3)], 0), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
print("wrote", out + ".json")
