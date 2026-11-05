"""Per-camera synchronisation re-measured on the DECODED three-camera dataset (mount-independent): image rotation
magnitude between frames (RAFT + essential matrix) vs FK rotation magnitude of the EXPORTED state over candidate lags.
Expected: the per-camera lag recorded in the manifest (wrist_cam 0.0, right_wrist_cam -1.5 frames)."""
import json, sys, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "v3")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "scripts"))
import calib_wrist as cw
from sfa3.recon3 import rb
from sfa32 import wristcam as wc
from lerobot.datasets.lerobot_dataset import LeRobotDataset
name = "sfa_v32_C_threecam"
d = LeRobotDataset(f"local/{name}", root=HERE / "outputs/lerobot" / name)
man = json.loads((HERE / "outputs/dataset_export" / f"{name}_manifest.json").read_text())["episodes"]
epi = np.array(d.hf_dataset["episode_index"])
robot = rb.Robot(); lags = np.arange(-4.0, 2.01, 0.5); out = {}
pooled = {s: [] for s in "RL"}
for e in man[::3]:
    idx = np.flatnonzero(epi == e["episode_index"])
    items = [d[int(i)] for i in idx]
    S = np.stack([it["observation.state"].numpy() for it in items]).astype(float)
    frames = {wc.CAM_OF[s]: np.stack([(it[wc.CAM_OF[s]].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for it in items]) for s in "RL"}
    ep = types.SimpleNamespace(frames=frames, state=S, index=e["episode_index"])
    for s in "RL":
        P = cw.pairs_for(ep, s, robot, "cuda:1")
        g = rb.Robot.split(S)
        for p in P:
            pooled[s].append([abs(cw.angle(p["A"][:3, :3]) - cw.angle(cw.hand_rel(robot, g, s, p["j"] + l, p["j"] + cw.GAP + l)[:3, :3])) for l in lags])
for s in "RL":
    curve = np.median(np.array(pooled[s]), axis=0)
    out[wc.CAM_OF[s]] = {"n_pairs": len(pooled[s]), "replay_lag": float(lags[int(np.argmin(curve))]), "expected_lag": wc.lag(s),
                         "curve": dict(zip(map(float, lags), map(float, curve)))}
    print(wc.CAM_OF[s], out[wc.CAM_OF[s]]["replay_lag"], "expected", wc.lag(s), "pairs", len(pooled[s]), flush=True)
(HERE / "outputs/dataset_export/sync_replay_threecam.json").write_text(json.dumps(out, indent=1))
