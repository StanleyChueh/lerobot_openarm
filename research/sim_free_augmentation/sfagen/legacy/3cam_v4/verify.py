import hashlib, json, sys
from pathlib import Path
import av, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
SFA = Path("/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation"); sys.path.insert(0, str(SFA))
from sfa import data
CAMS = ("observation.images.body_cam", "observation.images.wrist_cam", "observation.images.right_wrist_cam")


def frames(cam):
    for p in sorted((HERE / f"dataset/videos/{cam}").rglob("*.mp4")):
        with av.open(str(p)) as f:
            for fr in f.decode(video=0):
                yield fr.to_ndarray(format="rgb24")


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    prov = json.loads((HERE / "dataset/meta/sfa_provenance.json").read_text())
    LeRobotDataset(prov["name"], root=HERE / "dataset")
    df = pd.read_parquet(sorted((HERE / "dataset/data").rglob("*.parquet")))
    epi = df.episode_index.to_numpy(); S = np.stack(df["observation.state"].to_numpy()).astype(np.float32); A = np.stack(df["action"].to_numpy()).astype(np.float32)
    V = np.stack(df["wrist_view_valid"].to_numpy())
    eps = prov["episodes"]
    rep = {"episodes": len(eps), "frames": int(len(df)), "all_synthetic": all(e["synthetic"] for e in eps),
           "labels_sha_ok": all(hashlib.sha256(S[epi == e["export_episode_index"]].tobytes()).hexdigest() == e["state_sha256"] and
                                hashlib.sha256(A[epi == e["export_episode_index"]].tobytes()).hexdigest() == e["action_sha256"] for e in eps)}
    real = {E: data.load(E) for E in sorted({e["source_real_episode"] for e in eps})}
    rep["no_episode_identical_to_a_real_episode"] = not any(np.array_equal(S[epi == e["export_episode_index"]], real[E].state.astype(np.float32))
                                                            for e in eps for E in real if (epi == e["export_episode_index"]).sum() == len(real[E].state))
    srcep = np.concatenate([np.full(e["n_frames"], e["source_real_episode"]) for e in eps])
    srcrow = np.concatenate([np.array(e["wrist_source_rows"]) for e in eps])
    sel = {(e["E"], e["key"]): e["body_src"] for e in json.loads((HERE / "selection.json").read_text())["selected"]}
    for c in CAMS:
        se, mx, n = [], 255, 0
        if c == CAMS[0]:
            refs = (np.load(sel[(e['source_real_episode'], e['key'])], mmap_mode="r") for e in eps)
            ref_iter = (fr for arr in refs for fr in arr)
        for i, f in enumerate(frames(c)):
            ref = next(ref_iter) if c == CAMS[0] else real[int(srcep[i])].frames[c][int(srcrow[i])]
            if i % 3 == 0:
                se.append(float(((f.astype(np.float32) - ref) ** 2).mean()))
            mx = min(mx, int(f.max())); n += 1
        rep[c] = {"decoded": n, "reference": "rendered frames" if c == CAMS[0] else "real_10 source frame at round(tau)",
                  "mean_psnr_db": round(float(10 * np.log10(255 ** 2 / np.mean(se))), 2), "any_all_black_frame": mx == 0}
    rep["wrist_exact_fraction"] = round(float((V[:, 0] > 0.5).mean()), 4)
    rep["all_replay_checks_pass"] = all(e["replay_checks"]["speed_ratio_to_real_max"] <= 1.25 and e["replay_checks"]["accel_ratio_to_real_max"] <= 1.5 for e in eps)
    rep["episodes_by_condition"] = {c: sum(e["condition"] == c for e in eps) for c in ("G+", "C")}
    (HERE / "verification.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


main()
