"""Which conversion produced the published datasets? Compare every frame with the HDF5 candidates."""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from common import load, load_hdf5, to_motor, isaac_names, ARM, NAMES, DATA

h = load_hdf5()
gen = load("generated"); man = load("manual")
print("isaac articulation order:", isaac_names(h))
res = {}

# --- generated_500 vs generated.hdf5 ---
hg = h["generated"]["episodes"]
print("generated: dataset episodes", len(gen["episodes"]), "hdf5 demos", len(hg), "hdf5 success", sum(bool(e["attrs"]["success"]) for e in hg))
lens_ds = [b - a for a, b in gen["episodes"]]; lens_h = [len(e["actions"]) for e in hg]
print("lengths equal:", lens_ds == lens_h)
err = {"state=joint_pos": [], "action=joint_pos_target": [], "action=state[t+1]": [], "action=smooth(state[t+1])": []}
for i, (a, b) in enumerate(gen["episodes"]):
    S, A = gen["state"][a:b], gen["action"][a:b]
    P, T = to_motor(hg[i]["lerobot/joint_pos"]), to_motor(hg[i]["lerobot/joint_pos_target"])
    err["state=joint_pos"].append(np.abs(S - P).max())
    err["action=joint_pos_target"].append(np.abs(A - T).max())
    nxt = np.r_[S[1:], S[-1:]]
    err["action=state[t+1]"].append(np.abs(A[:, ARM] - nxt[:, ARM]).max())
    sm = nxt.copy(); sm[:, ARM] = gaussian_filter1d(nxt[:, ARM], 2, axis=0)
    err["action=smooth(state[t+1])"].append(np.abs(A[:, ARM] - sm[:, ARM]).max())
    if i == 0:
        print("ep0 first action/target", np.round(A[0, :4], 5), np.round(T[0, :4], 5))
for k, v in err.items():
    v = np.array(v); print(f"  {k:28s} max|diff| over 500 eps: median {np.median(v):.2e}  max {v.max():.2e}")
res["generated"] = {k: [float(np.median(v)), float(np.max(v))] for k, v in err.items()}

# --- manual sim_10 vs the source HDF5 (server-recorded) ---
hs = h["source"]["episodes"]; names = isaac_names(h)
print("manual: dataset eps", len(man["episodes"]), "lengths", [b - a for a, b in man["episodes"]], "hdf5", [len(e["actions"]) for e in hs])
errs = {"state=hdf5 state[t-1]": [], "action=hdf5 actions[t]": []}
# source actions are 16D in the sim server's action-slot order; map via the slot names
for i, (a, b) in enumerate(man["episodes"]):
    S, A = man["state"][a:b], man["action"][a:b]
    st = hs[i]["states/articulation/robot/joint_position"]
    named = {n: st[:, c] for c, n in enumerate(names)}
    post = to_motor(named)                       # post-step state of step t
    n = min(len(S), len(post))
    errs["state=hdf5 state[t-1]"].append(np.abs(S[1:n, ARM] - post[:n - 1, ARM]).max())
    res.setdefault("manual_lengths", []).append([int(b - a), len(hs[i]["actions"])])
print({k: [float(np.median(v)), float(np.max(v))] for k, v in errs.items() if v})
json.dump(res, open(DATA / "provenance_check.json", "w"), indent=1)
