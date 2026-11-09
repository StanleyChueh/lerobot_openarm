import sys, numpy as np, pandas as pd, av, cv2
from pathlib import Path
ROOT = Path("/home/stanley/Stanley_ws/sfa_hf_export/battery/src")
E = int(sys.argv[1]); OUT = sys.argv[2]
ep = pd.read_parquet(ROOT / "meta/episodes/chunk-000/file-000.parquet")
r = ep[ep.episode_index == E].iloc[0]
tiles = []
for cam in ("observation.images.body_cam", "observation.images.wrist_cam", "observation.images.right_wrist_cam"):
    ci, fi = int(r[f"videos/{cam}/chunk_index"]), int(r[f"videos/{cam}/file_index"]); t0 = float(r[f"videos/{cam}/from_timestamp"])
    n = int(r["length"]); want = set(np.linspace(0, n - 1, 8).astype(int))
    got = {}
    with av.open(str(ROOT / f"videos/{cam}/chunk-{ci:03d}/file-{fi:03d}.mp4")) as f:
        s = f.streams.video[0]
        for fr in f.decode(s):
            k = int(round((fr.time - t0) * 30))
            if k in want: got[k] = fr.to_ndarray(format="rgb24")
            if k > max(want): break
    row = [cv2.putText(cv2.resize(got[k], (320, 240)), f"t={k}", (5, 20), 0, 0.6, (255, 255, 0), 2) for k in sorted(got)]
    tiles.append(np.concatenate(row, 1))
cv2.imwrite(OUT, cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR))
