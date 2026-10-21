"""M1 audit: schema / semantics diff between the real and sim LeRobot datasets (parquet only)."""
import glob, hashlib, json, sys
import numpy as np, pandas as pd

ROOT = "/home/csl/.cache/huggingface/lerobot/ethanCSL/"
SETS = {"real": "openarm_pringles_lerobot_real_v00", "sim": "openarm_pringles_lerobot_sim_v01_reallook_physfit"}
out = {}
for tag, name in SETS.items():
    d = ROOT + name
    info = json.load(open(d + "/meta/info.json"))
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(d + "/data/*/*.parquet")))
    md5 = {p.split(name + "/")[1]: hashlib.md5(open(p, "rb").read()).hexdigest()[:10]
           for p in sorted(glob.glob(d + "/data/*/*.parquet")) + [d + "/meta/info.json"]}
    sn, an = info["features"]["observation.state"]["names"], info["features"]["action"]["names"]
    S, A = np.stack(df["observation.state"].values), np.stack(df["action"].values)
    ep = df["episode_index"].values
    r = {"md5": md5, "fps": info["fps"], "episodes": int(len(np.unique(ep))), "frames": len(df),
         "state_names": sn, "action_names_equal_state": an == sn}
    lens = np.array([np.sum(ep == e) for e in np.unique(ep)])
    r["ep_len"] = [int(lens.min()), float(np.median(lens)), int(lens.max())]
    dts = np.concatenate([np.diff(df["timestamp"].values[ep == e]) for e in np.unique(ep)])
    r["dt_median_max"] = [round(float(np.median(dts)), 4), round(float(dts.max()), 4)]
    # per-joint ranges, grippers in particular
    r["ranges"] = {n: [round(float(S[:, i].min()), 3), round(float(S[:, i].max()), 3),
                       round(float(A[:, i].min()), 3), round(float(A[:, i].max()), 3)] for i, n in enumerate(sn)}
    gi = [i for i, n in enumerate(sn) if "J8" in n]
    r["gripper_action_unique"] = {sn[i]: int(len(np.unique(np.round(A[:, i], 3)))) for i in gi}
    # start / end pose vs first episode's first frame (home), arm joints only
    arm = [i for i in range(len(sn)) if i not in gi]
    home = S[ep == ep[0]][0, arm]
    st = np.array([np.abs(S[ep == e][0, arm] - home).max() for e in np.unique(ep)])
    en = np.array([np.abs(S[ep == e][-1, arm] - home).max() for e in np.unique(ep)])
    r["start_dev_from_home_max_rad"] = round(float(st.max()), 3)
    r["end_dev_from_home_median_max_rad"] = [round(float(np.median(en)), 3), round(float(en.max()), 3)]
    # action -> state lag: shift k (frames) minimising mean |A[t] - S[t+k]| over arm joints
    errs = {}
    for k in range(0, 13):
        e_all = [np.abs(A[ep == e][: len(A[ep == e]) - k, arm] - S[ep == e][k:, arm]).mean() for e in np.unique(ep)]
        errs[k] = float(np.mean(e_all))
    kbest = min(errs, key=errs.get)
    r["action_state_lag_frames"] = kbest
    r["action_state_err_rad_k0_kbest"] = [round(errs[0], 4), round(errs[kbest], 4)]
    gap = np.abs(A[:, arm] - S[:, arm])
    r["abs_action_minus_state_p95_max"] = [round(float(np.percentile(gap, 95)), 3), round(float(gap.max()), 3)]
    sp = np.concatenate([np.abs(np.diff(S[ep == e][:, arm], axis=0)) * info["fps"] for e in np.unique(ep)])
    r["state_speed_p99_rad_s"] = round(float(np.percentile(sp, 99)), 3)
    out[tag] = r

json.dump(out, open(sys.argv[1], "w"), indent=1)
for tag, r in out.items():
    print("=====", tag)
    for k, v in r.items():
        if k not in ("ranges", "state_names"):
            print(f"  {k}: {v}")
    print("  names:", r["state_names"])
    print("  grip ranges [s_min s_max a_min a_max]:", {n: v for n, v in r["ranges"].items() if "J8" in n})
