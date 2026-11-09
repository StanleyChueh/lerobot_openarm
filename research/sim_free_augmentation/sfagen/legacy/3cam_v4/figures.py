"""Example frames of hand-over-location episodes: real source (top) vs synthetic (bottom), body_cam."""
import json, sys
from pathlib import Path
import numpy as np, cv2
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation")
from sfa import data
sel = [e for e in json.loads((HERE / "selection.json").read_text())["selected"] if e.get("condition") == "C"]
picks = [e for e in sel if (e["E"], e["key"]) in ((6, "pos_left_down__C__s1.0"), (4, "pos_right_up__C__s0.7"), (8, "pos_left_down__C__s1.0"))]
rows = []
for e in picks:
    E = e["E"]; a, b = e["rows"]; s = data.load(E, cams=("observation.images.body_cam",)); ev = data.command_events(s.action)
    body = np.load(e["body_src"], mmap_mode="r")
    ts = [ev["right_close"] + 25, ev["left_close"] - 25, ev["left_close"], ev["right_open"] + 15]
    cols = []
    for t in ts:
        src = s.frames["observation.images.body_cam"][t].copy(); syn = np.asarray(body[t - a]).copy()
        cv2.putText(src, f"real ep {E}  t={t}", (8, 24), 0, 0.7, (255, 255, 0), 2)
        cv2.putText(syn, f"synthetic {e['variation']} ({e['handover_shift_mm']:.0f} mm)", (8, 24), 0, 0.7, (255, 160, 60), 2)
        cols.append(np.concatenate([src, syn], 0))
    rows.append(np.concatenate(cols, 1))
im = np.concatenate(rows, 0)
cv2.imwrite(str(HERE / "dataset_assets/handover_location_examples.jpg"), cv2.cvtColor(cv2.resize(im, None, fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
