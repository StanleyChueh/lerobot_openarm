"""Every camera's video segment of every episode must last exactly n_frames / fps (catches dropped frames)."""
import glob, json, sys
from pathlib import Path
import pandas as pd
def check(root):
    info = json.loads((Path(root) / "meta/info.json").read_text())
    cams = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    ep = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))])
    bad = []
    for _, r in ep.iterrows():
        for c in cams:
            dur = r[f"videos/{c}/to_timestamp"] - r[f"videos/{c}/from_timestamp"]
            if abs(dur - r["length"] / info["fps"]) > 0.5 / info["fps"]:
                bad.append({"episode": int(r["episode_index"]), "camera": c, "frames_in_video": round(dur * info["fps"], 2), "length": int(r["length"])})
    return {"episodes": len(ep), "cameras": cams, "bad_segments": bad, "ok": not bad}
if __name__ == "__main__":
    out = {r: check(r) for r in sys.argv[1:]}
    for r, v in out.items():
        print(Path(r).name, v["episodes"], "ok" if v["ok"] else f"BAD {v['bad_segments']}")
