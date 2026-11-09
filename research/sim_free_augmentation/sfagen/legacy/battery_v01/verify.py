import hashlib, json, sys
from pathlib import Path
import av, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "battery/bsfa"))
import bdata
CAMS = bdata.CAMS


def frames(cam):
    for p in sorted((HERE / f"dataset/videos/{cam}").rglob("*.mp4")):
        with av.open(str(p)) as f:
            for fr in f.decode(video=0):
                yield fr.to_ndarray(format="rgb24")


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    prov = json.loads((HERE / "dataset/meta/sfa_provenance.json").read_text()); eps = prov["episodes"]
    LeRobotDataset(prov["name"], root=HERE / "dataset")
    df = pd.read_parquet(sorted((HERE / "dataset/data").rglob("*.parquet")))
    epi = df.episode_index.to_numpy(); S = np.stack(df["observation.state"].to_numpy()).astype(np.float32); A = np.stack(df["action"].to_numpy()).astype(np.float32)
    rep = {"episodes": len(eps), "frames": int(len(df)), "all_synthetic": all(e["synthetic"] for e in eps),
           "labels_sha_ok": all(hashlib.sha256(S[epi == e["export_episode_index"]].tobytes()).hexdigest() == e["state_sha256"] and
                                hashlib.sha256(A[epi == e["export_episode_index"]].tobytes()).hexdigest() == e["action_sha256"] for e in eps),
           "sources_used": sorted({e["source_real_episode"] for e in eps})}
    src = bdata.table()
    rep["no_episode_identical_to_a_real_episode"] = not any(
        np.array_equal(S[epi == e["export_episode_index"]], np.stack(src[src.episode_index == E]["observation.state"].to_numpy()).astype(np.float32))
        for e in eps for E in range(10) if (epi == e["export_episode_index"]).sum() == (src.episode_index == E).sum())
    order = [(e["source_real_episode"], e["variation"]) for e in eps]
    for c in CAMS:
        n, se, mn = 0, [], 255
        if c == CAMS[0]:
            it = (fr for E, k in order for fr in np.load(HERE / f"work/body/{E}_{k}.npy", mmap_mode="r"))
        for i, f in enumerate(frames(c)):
            if c == CAMS[0]:
                ref = next(it)
                if i % 5 == 0: se.append(float(((f.astype(np.float32) - ref) ** 2).mean()))
            mn = min(mn, int(f.max())); n += 1
        rep[c] = {"decoded": n, "any_all_black_frame": mn == 0}
        if se: rep[c]["psnr_vs_rendered_db"] = round(float(10 * np.log10(255 ** 2 / np.mean(se))), 2)
    rep["all_replay_checks_pass"] = all(e["replay_checks"]["speed_ratio_to_real_max"] <= 1.25 and e["replay_checks"]["accel_ratio_to_real_max"] <= 1.5 and e["replay_checks"]["placement"]["ok"] for e in eps)
    rep["episodes_by_family"] = {f: sum(e["family"] == f for e in eps) for f in ("battery_position", "station_position")}
    (HERE / "verification.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))


main()
