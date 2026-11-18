"""Evaluation only: build a hand-checked gold mask set that is independent of the pipeline (R06 protocol).
  gold_make.py frames GOLD_DIR NAME=CONFIG[:e1,e2] ...   pick frames by rule from the recorded state, write PNG + grid view
  gold_make.py seg GOLD_DIR [ID ...]                      clicks/ID.json -> ID.npz + view/ID.jpg (2x) + zooms (4x)
  gold_make.py sheet GOLD_DIR OUT.jpg                     contact sheet of all labelled frames
clicks/ID.json: {"order": [class labels, later = in front], "labels": {LABEL: {"pos": [[x,y]], "neg": [[x,y]],
  "box": [x0,y0,x1,y1], "crop": [x0,y0,x1,y1], "pick": i, "wand": tol (colour pick instead of SAM), "add": [polygon], "sub": [polygon]}}, "zoom": [[x0,y0,x1,y1]]}
Labels: arm_<side> (whole arm incl. gripper), grip_<side> (below the last arm joint), obj_<k>, ignore."""
import json, sys
from pathlib import Path
import cv2, numpy as np

COL = {"arm": (40, 90, 255), "grip": (255, 230, 0), "obj": (255, 120, 0), "ignore": (200, 200, 200)}
col = lambda l: COL[l.split("_")[0]] if not l.startswith("arm_") or l.endswith(("L", "A")) else (0, 220, 120)  # noqa: E731


def pick_frames(sp, arms, grip, fps, T):
    c = min((int(np.flatnonzero(sp["g" + s] < grip[s].get("close_threshold", 0.5))[0]) for s in arms
             if grip[s] and (sp["g" + s] < grip[s].get("close_threshold", 0.5)).any()), default=None)
    if c is None or c < 4:
        return {"q25": T // 4, "q50": T // 2, "q75": 3 * T // 4}
    return {"far": c // 2, "near": max(0, c - round(0.5 * fps)), "grasp": min(T - 1, c + round(0.3 * fps))}


def grid(im, s=2):
    v = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    for x in range(0, im.shape[1], 40):
        cv2.line(v, (x * s, 0), (x * s, v.shape[0]), (255, 255, 0) if x % 80 == 0 else (120, 120, 0), 1)
        if x % 80 == 0:
            cv2.putText(v, str(x), (x * s + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    for y in range(0, im.shape[0], 40):
        cv2.line(v, (0, y * s), (v.shape[1], y * s), (255, 255, 0) if y % 80 == 0 else (120, 120, 0), 1)
        if y % 80 == 0:
            cv2.putText(v, str(y), (2, y * s + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return v


def view(im, masks, clicks=None, box=None, s=2):
    x0, y0, x1, y1 = box or (0, 0, im.shape[1], im.shape[0]); v = cv2.resize(im[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    for l, m in masks.items():
        if l.startswith("grip") or not m.any():
            continue
        if l == "ignore":
            mk = cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST) > 0; v[mk] = (0.6 * v[mk] + 0.4 * np.array([255, 0, 255])).astype(np.uint8); continue
        mm = cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        cv2.drawContours(v, cv2.findContours(mm, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0], -1, col(l), 1)
    for l, m in masks.items():                              # gripper outline on top, dotted look by thin yellow line
        if l.startswith("grip") and m.any():
            mm = cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            cv2.drawContours(v, cv2.findContours(cv2.erode(mm, np.ones((3, 3), np.uint8)), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0], -1, col(l), 1)
    for l, c in (clicks or {}).items():
        for k, mark in (("pos", cv2.MARKER_CROSS), ("neg", cv2.MARKER_TILTED_CROSS)):
            for x, y in c.get(k, []):
                cv2.drawMarker(v, (int((x - x0) * s), int((y - y0) * s)), (0, 255, 0) if k == "pos" else (255, 0, 0), mark, 8, 1)
    return v


def main():
    cmd, gd = sys.argv[1], Path(sys.argv[2]); gd.mkdir(parents=True, exist_ok=True); fj = gd / "frames.json"
    frames = json.loads(fj.read_text()) if fj.exists() else {}
    if cmd == "frames":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen")); from sfagen.context import Context
        (gd / "grid").mkdir(exist_ok=True); (gd / "png").mkdir(exist_ok=True)
        for job in sys.argv[3:]:
            name, spec = job.split("=", 1); spec, e_ = (spec.rsplit(":", 1) + [None])[:2] if ":" in spec else (spec, None)
            ctx = Context(spec, Path(__file__).resolve().parents[4] / "artifacts/simbench_runs/r06/gold_work" / name); rob = ctx.robot
            for e in ([int(x) for x in e_.split(",")] if e_ else ctx.episodes()):
                S, _ = ctx.src.arrays(e); F = ctx.src.decode(ctx.src.body, e)
                for kind, j in pick_frames(rob.split(S), rob.arms, rob.grip, ctx.src.fps, len(S)).items():
                    i = f"{name}_e{e}_f{j}"; frames[i] = {"name": name, "config": str(Path(spec).resolve()), "episode": e, "frame": int(j), "kind": kind,
                                                         "arms": list(rob.arms), "W": int(F.shape[2])}
                    cv2.imwrite(str(gd / "png" / f"{i}.png"), cv2.cvtColor(F[j], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(gd / "grid" / f"{i}.jpg"), cv2.cvtColor(grid(F[j]), cv2.COLOR_RGB2BGR)); print(i, kind)
        fj.write_text(json.dumps(frames, indent=1)); return
    if cmd == "sheet":
        tiles = []
        for i in sorted(frames):
            if (gd / f"{i}.npz").exists():
                im = cv2.cvtColor(cv2.imread(str(gd / "png" / f"{i}.png")), cv2.COLOR_BGR2RGB); z = np.load(gd / f"{i}.npz"); s = 640 / im.shape[1]
                v = view(im, {l: z[l] for l in z.files if l[0] != "_"}, s=s); cv2.putText(v, f"{i} ({frames[i]['kind']})", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
                tiles.append(v)
        rows = [np.concatenate(tiles[k:k + 3] + [np.zeros_like(tiles[0])] * (3 - len(tiles[k:k + 3])), 1) for k in range(0, len(tiles), 3)]
        cv2.imwrite(sys.argv[3], cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]); print("wrote", sys.argv[3], len(tiles)); return
    import torch
    from transformers import Sam2Model, Sam2Processor
    dev = "cuda:0"; name = "facebook/sam2.1-hiera-large"; model = Sam2Model.from_pretrained(name).to(dev).eval(); proc = Sam2Processor.from_pretrained(name)
    (gd / "view").mkdir(exist_ok=True)
    for i in sys.argv[3:] or [p.stem for p in sorted((gd / "clicks").glob("*.json"))]:
        ck = json.loads((gd / "clicks" / f"{i}.json").read_text()); im = cv2.cvtColor(cv2.imread(str(gd / "png" / f"{i}.png")), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]; raw = {}
        for l, c in ck["labels"].items():
            m = np.zeros((H, W), bool)
            if c.get("wand"):                              # colour pick: pixels in the box within `wand` of a clicked colour
                x0, y0, x1, y1 = c.get("box", (0, 0, W, H)); sub = im[y0:y1, x0:x1].astype(np.int16)
                m[y0:y1, x0:x1] = np.any([np.abs(sub - np.array(r_, np.int16)).max(2) < c["wand"] for r_ in c.get("ref") or [im[y, x] for x, y in c["pos"]]], 0)
                m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
            elif c.get("pos"):
                x0, y0, x1, y1 = c.get("crop", (0, 0, W, H)); sub = im[y0:y1, x0:x1]; k = 1024 / max(sub.shape[:2]); sub = cv2.resize(sub, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
                pts = [[(x - x0) * k, (y - y0) * k] for x, y in c["pos"] + c.get("neg", [])]; lab = [1] * len(c["pos"]) + [0] * len(c.get("neg", []))
                kw = {"input_boxes": [[[(c["box"][0] - x0) * k, (c["box"][1] - y0) * k, (c["box"][2] - x0) * k, (c["box"][3] - y0) * k]]]} if c.get("box") else {}
                inp = proc(images=sub, input_points=[[pts]], input_labels=[[lab]], return_tensors="pt", **kw).to(dev)
                with torch.inference_mode():
                    o = model(**inp, multimask_output=True)
                pm = proc.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"])[0][0].numpy(); sc = o.iou_scores[0, 0].float().cpu().numpy()
                p = c.get("pick", int(sc.argmax())); print(i, l, "scores", np.round(sc, 2), "areas", pm.reshape(len(pm), -1).sum(1), "pick", p)
                m[y0:y1, x0:x1] = cv2.resize(pm[p].astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0
                n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))        # drop specks, fill pinholes (< 60 px)
                m = np.isin(cc, [k_ for k_ in range(1, n) if st[k_, 4] >= 60])
                n, cc, st, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8)); m |= np.isin(cc, [k_ for k_ in range(1, n) if st[k_, 4] < 60])
            if c.get("dark"):                              # plus every pixel darker than `dark` inside dark_box (black links on a light scene)
                x0, y0, x1, y1 = c.get("dark_box", (0, 0, W, H)); m[y0:y1, x0:x1] |= cv2.morphologyEx((im[y0:y1, x0:x1].max(2) < c["dark"]).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
            for key, val in (("add", True), ("sub", False)):
                for poly in c.get(key, []):
                    t = np.zeros((H, W), np.uint8); cv2.fillPoly(t, [np.array(poly, np.int32)], 1); m[t > 0] = val
            for line in c.get("lines", []):                 # thick polylines (loose cables -> ignore)
                t = np.zeros((H, W), np.uint8); cv2.polylines(t, [np.array(line, np.int32)], False, 1, c.get("thick", 7)); m[t > 0] = True
            raw[l] = m
        for l in list(raw):                                  # arm = arm clicks + its gripper
            if l.startswith("grip_") and "arm_" + l[5:] in raw:
                raw["arm_" + l[5:]] |= raw[l]
        order = [l for l in (ck.get("order") or [l for l in raw if not l.startswith("grip_")]) if l in raw]
        for a, l in enumerate(order):                        # later in the order = in front
            for l2 in order[a + 1:]:
                raw[l] &= ~raw[l2]
        for l in raw:
            if l.startswith("grip_"):
                raw[l] &= raw.get("arm_" + l[5:], raw[l])
        np.savez_compressed(gd / f"{i}.npz", _band=np.array(ck.get("band", 2)), **raw)
        cv2.imwrite(str(gd / "view" / f"{i}.jpg"), cv2.cvtColor(view(im, raw, ck["labels"]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        for n, b in enumerate(ck.get("zoom", [])):
            cv2.imwrite(str(gd / "view" / f"{i}_z{n}.jpg"), cv2.cvtColor(view(im, raw, None, b, 4), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(i, {l: int(m.sum()) for l, m in raw.items()})


if __name__ == "__main__":
    main()
