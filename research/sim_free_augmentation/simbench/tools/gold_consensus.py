"""Evaluation only (RoboTwin): do not score gold pixels where my hand label and the simulator's true robot silhouette
disagree (white links on the white table, parts hidden by an object). Adds them to `ignore` in GOLD_DIR/ID.npz; the hand
label itself is kept (`_hand_ignore` stores the ignore mask from before). Needs a work dir built with the TRUE camera.
Usage: gold_consensus.py GOLD_DIR NAME=TRUE_CAMERA_CONFIG[@WORK_DIR] ..."""
import json, sys
import numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfagen"))
from sfagen.context import Context
from sfagen.recon import Recon

gd = Path(sys.argv[1]); frames = json.loads((gd / "frames.json").read_text())
for job in sys.argv[2:]:
    name, spec = job.split("=", 1); cfg, work = (spec.split("@") + [None])[:2]; ctx = Context(cfg, work)
    for e in sorted({f["episode"] for f in frames.values() if f["name"] == name}):
        rec = Recon(ctx, e, "cuda:0")
        for fid, f in sorted(frames.items()):
            if f["name"] == name and f["episode"] == e and (gd / f"{fid}.npz").exists():
                z = dict(np.load(gd / f"{fid}.npz")); sd = rec.robot_sil(f["frame"], sides=True)
                kc = np.ones((2 * round(4 * rec.W / 640) + 1,) * 2, np.uint8)      # close the point-splat holes of the silhouette
                hand = z.get("_hand_ignore", z.get("ignore", np.zeros(sd.shape, bool))); ign = hand.copy()
                for s in rec.arms:
                    ign |= (cv2.morphologyEx((sd == s).astype(np.uint8), cv2.MORPH_CLOSE, kc) > 0) ^ z.get("arm_" + s, np.zeros(sd.shape, bool))
                z["_hand_ignore"] = hand; z["ignore"] = ign; np.savez_compressed(gd / f"{fid}.npz", **z)
                print(fid, "ignore px", int(hand.sum()), "->", int(ign.sum()), "| gold arm px still scored", int(sum((z.get("arm_" + s, ign) & ~ign).sum() for s in rec.arms)))
        del rec
