"""Evaluation only: how well do the arm masks cover the robot, and the gripper in particular?
Reference silhouette = URDF mesh z-buffer through the TRUE camera (--truth JSON, simulator, lag 0) or, without truth,
through the fitted camera (then only a consistency check, not accuracy). Gripper = every geometry at or below the arm's
last listed joint in the URDF tree. Bins: far / near (TCP < 10 cm from an object) / held; without configured objects
"held" = gripper commanded closed. Reads existing reconstruction caches; never writes into a work dir.
Usage: mask_check.py OUT_PREFIX [--truth TRUTH.json] [--device cuda:0] NAME=CONFIG[@WORK_DIR][:e1,e2] ..."""
import json, sys
import numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.context import Context
from sfagen.camera import Camera
from sfagen.recon import Recon
from sfagen.render import Renderer
from sfagen import mesh

a = sys.argv[1:]; out = a.pop(0); truth = None; dev = "cuda:0"
while a and a[0].startswith("--"):
    k = a.pop(0); v = a.pop(0)
    if k == "--truth":
        truth = json.load(open(v))
    else:
        dev = v
iou = lambda x, y: float((x & y).sum() / max(1, (x | y).sum()))  # noqa: E731
COLS = ("arm_iou_raw", "arm_iou_final", "arm_recall_final", "arm_precision_final", "gripper_raw", "gripper_final", "gripper_renderer", "leak_final")
panels, report = [], {}
for job in a:
    name, spec = job.split("=", 1); eps = None
    if ":" in spec:
        spec, e_ = spec.rsplit(":", 1); eps = [int(x) for x in e_.split(",")]
    cfg, work = (spec.split("@") + [None])[:2]
    ctx = Context(cfg, work); cam_t = Camera({"intrinsics": truth["intrinsics"], "extrinsics": truth}) if truth else None
    rc = ctx.cfg.get("reconstruction", {}); cut = rc.get("prompt_object_cut", "neighbourhood"); rows, fit_iou, added = [], [], 0
    for e in eps or ctx.episodes():
        sam_p = ctx.cache_dir / (f"sam2_ep{e}.npz" if cut == "neighbourhood" else f"sam2_{cut}_ep{e}.npz")
        if not ((ctx.cache_dir / f"recon_ep{e}.npz").exists() and sam_p.exists()):
            print(name, "ep", e, "cache missing, skipped"); continue
        rec = Recon(ctx, e, dev); G, rob = rec.G, rec.robot; R = Renderer(rec, dev, ctx.cfg.get("render", {})); raw = np.load(sam_p)
        side_of = np.array(G.side + ["X"]); added += getattr(rec, "repaired_links", 0); grip = {}
        for s in rec.arms:
            last = rob.m.getJointId(rob.cfg["arms"][s]["joints"][-1])
            def below(jt):
                while jt > 0:
                    if jt == last:
                        return True
                    jt = rob.m.parents[jt]
                return False
            grip[s] = [i for i, g in enumerate(rob.vis.geometryObjects) if below(g.parentJoint)]
        show = rec.objects[0]["grasp_frame"] if rec.objects else rec.T // 2
        for j in sorted(set(range(5, rec.T - 5, 6)) | {show}):
            if cam_t is not None:
                q = {s: rec.sp[s][j] for s in rec.arms}; g = {s: float(rec.sp["g" + s][j]) for s in rec.arms if "g" + s in rec.sp}
            else:
                q = {s: rec.gv[s][j] for s in rec.arms}; g = {s: float(rec.grip_v[s][j]) for s in rec.grip_v}
            lab = mesh.splat(G.world(G.placements(q, g)), G.gid, cam_t or rec.cam, dev=dev)[1].cpu().numpy(); ref_side = side_of[lab]
            fit_side = rec.robot_sil(j, sides=True); cat = R.source_state(j)["cat"].cpu().numpy()
            for s in rec.arms:
                ref = ref_side == s
                if ref.sum() < 400 * (rec.W / 640) ** 2:
                    continue
                gref = np.isin(lab, grip[s]) & ref; ok = gref.sum() >= 80 * (rec.W / 640) ** 2
                sam = raw[s][j] > 0; fin = rec.masks[s][j] > 0; rb = cat == R.cats.index(s)
                if rec.objects:
                    tcp = rec.tcp(s, j)[:3, 3]; held = any(rec.holder_at(k, j) == s for k in range(len(rec.objects)))
                    d = min(np.linalg.norm(tcp - (rec.obj_pose(k, j) @ np.r_[rec.objects[k]["points"].mean(0), 1])[:3]) for k in range(len(rec.objects)))
                    b = "held" if held else ("near" if d < 0.10 else "far")
                else:
                    b = "held" if rob.grip[s] and rec.sp["g" + s][j] < rob.grip[s].get("close_threshold", 0.5) else "far"
                cov = lambda m: float((m & gref).sum() / gref.sum()) if ok else np.nan  # noqa: E731
                r10 = max(1, round(10 * rec.W / 640)); near_ref = cv2.dilate(ref.astype(np.uint8), np.ones((2 * r10 + 1,) * 2, np.uint8)) > 0
                rows.append((b, iou(sam, ref), iou(fin, ref), (fin & ref).sum() / ref.sum(), (fin & ref).sum() / max(1, fin.sum()),
                             cov(sam), cov(fin), cov(rb), (fin & ~near_ref).sum() / max(1, fin.sum())))
                if cam_t is not None:
                    fit_iou.append(iou(fit_side == s, ref))
            if j == show and len(panels) < 12 and e == (eps or ctx.episodes())[0]:
                im = rec.F[j]; p1, p2, p3 = im.copy(), im.copy(), im.copy()
                cs, _ = cv2.findContours(np.isin(ref_side, rec.arms).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for p, m in ((p1, np.any([raw[s][j] > 0 for s in rec.arms], 0)), (p2, np.any([rec.masks[s][j] > 0 for s in rec.arms], 0))):
                    p[m] = (0.5 * p[m] + 0.5 * np.array([40, 90, 255])).astype(np.uint8); cv2.drawContours(p, cs, -1, (0, 255, 0), 1)
                for i, col in enumerate([(40, 90, 255), (60, 200, 60), (255, 160, 0), (255, 0, 200), (0, 220, 220)][:len(R.cats)]):
                    p3[cat == i] = (0.45 * p3[cat == i] + 0.55 * np.array(col)).astype(np.uint8)
                for p, t in ((p1, f"{name} ep{e} f{j}: raw SAM2 (blue), reference (green)"), (p2, "final arm masks"), (p3, "renderer masks: arms / objects")):
                    cv2.putText(p, t, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
                panels.append(np.concatenate([p1, p2, p3], 1))
        del R, rec
    if not rows:
        continue
    X = np.array([r[1:] for r in rows], float); B = np.array([r[0] for r in rows]); rep = {"reference": "true camera" if truth else "fitted camera", "links_repaired": int(added)}
    if fit_iou:
        rep["fitted_vs_true_silhouette_iou_median"] = float(np.median(fit_iou))
    print(f"\n== {name} ({rep['reference']}; arm-frames {len(rows)}; links repaired {added}" + (f"; fitted-vs-true IoU {rep['fitted_vs_true_silhouette_iou_median']:.2f}" if fit_iou else "") + ")")
    print("  bin      n  " + "  ".join(f"{c:>19s}" for c in COLS) + "   gripper_final<0.2")
    for b in ("far", "near", "held", "all"):
        m = np.ones(len(B), bool) if b == "all" else B == b
        if m.any():
            med = np.nanmedian(X[m], 0); low = float(np.nanmean(X[m][:, 5] < 0.2))
            rep[b] = {"n": int(m.sum()), **{c: float(v) for c, v in zip(COLS, med)}, "gripper_final_below_0.2_frac": low}
            print(f"  {b:5s} {m.sum():4d}  " + "  ".join(f"{v:19.3f}" for v in med) + f"   {low:.2f}")
    report[name] = rep
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out + ".json").write_text(json.dumps(report, indent=1))
if panels:
    h = max(p.shape[0] for p in panels)
    cv2.imwrite(out + ".jpg", cv2.cvtColor(np.concatenate([p for p in panels if p.shape[0] == h], 0), cv2.COLOR_RGB2BGR))
print("wrote", out + ".json")
