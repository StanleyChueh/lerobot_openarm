"""Replay one real LeRobot episode through the live Isaac server and measure physical outcome.

This is a diagnostic, not a dataset exporter: it writes JSON and sparse unmodified camera frames.
Run the server with the matching task/profile first. Source parquet files are downloaded from HF;
no real robot is commanded.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from PIL import Image
from scipy.spatial.transform import Rotation

from lerobot_robot_openarm_umeow.sim_link import SimClient
from sim_bridge_common import load_calibration, motor_action_to_sim_joints

REAL = {
    "pringles": "ethanCSL/openarm_pringles_lerobot_real_v00",
    "battery": "ethanCSL/openarm_insertbattery_lerobot_real_v01",
}
STARTS = {
    "pringles": {"can": [0.32, 0.0, 0.0]},
    "battery": {"charger": [0.325, 0.070, 0.0], "battery": [0.320, -0.092, 0.0]},
}


def load_episode(repo: str, episode: int):
    api = HfApi()
    names = json.loads(Path(hf_hub_download(repo, "meta/info.json", repo_type="dataset")).read_text())[
        "features"]["action"]["names"]
    paths = [entry.path for entry in api.list_repo_tree(repo, repo_type="dataset", recursive=True)
             if entry.path.startswith("data/") and entry.path.endswith(".parquet")]
    rows = pd.concat(pd.read_parquet(hf_hub_download(repo, p, repo_type="dataset")) for p in paths)
    rows = rows[rows.episode_index == episode].sort_values("frame_index")
    if rows.empty:
        raise ValueError(f"Episode {episode} not found in {repo}")
    return names, np.stack(rows["observation.state"].to_numpy()), np.stack(rows["action"].to_numpy())


def local_battery_pose(objects):
    charger, battery = objects["charger"], objects["battery"]
    q_c = Rotation.from_quat([*charger[4:7], charger[3]])
    q_b = Rotation.from_quat([*battery[4:7], battery[3]])
    xyz = q_c.inv().apply(np.array(battery[:3]) - charger[:3])
    parallel = abs(float(np.dot(q_c.apply([1, 0, 0]), q_b.apply([1, 0, 0]))))
    return [float(v) for v in xyz], parallel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=REAL, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--port", type=int, default=5710)
    parser.add_argument("--starts", help="JSON file of episode-number -> asset -> [x,y,yaw_deg] start poses")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--sample_every", type=int, default=100)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = REAL[args.task]
    names, states, actions = load_episode(repo, args.episode)
    calib = load_calibration(str(Path(__file__).with_name("calibration.json")))
    starts = STARTS[args.task]
    if args.starts:
        starts = json.loads(Path(args.starts).read_text())[str(args.episode)]
    client = SimClient(port=args.port)
    hello = client.connect()
    if hello.get("real_dataset") != repo or hello.get("control_hz") != 30:
        raise SystemExit(f"Wrong server profile/rate: {hello}")
    first = motor_action_to_sim_joints(dict(zip(names, states[0])), calib)
    obs = client.request("reset", joints=first, objects=starts)
    trace = []
    limit = min(len(actions), args.max_steps or len(actions))
    for k in range(limit):
        if k % args.sample_every == 0 or k == limit - 1:
            Image.fromarray(obs["images"]["body_cam"]).save(out / f"body_{k:04d}.png")
        cmd = motor_action_to_sim_joints(dict(zip(names, actions[k])), calib)
        obs = client.request("step", joints=cmd)
        point = {"step": k, "objects": obs["objects"], "success": bool(obs["success"])}
        if args.task == "battery":
            point["battery_local_xyz"], point["battery_axis_parallel"] = local_battery_pose(obs["objects"])
        trace.append(point)
        if k % 100 == 0:
            print(f"step {k}/{limit}: {point}", flush=True)
    client.close()
    summary = {"source": repo, "episode": args.episode, "steps": limit, "starts": starts,
               "any_sim_success": any(p["success"] for p in trace)}
    for object_name in starts:
        zs = [p["objects"][object_name][2] for p in trace]
        summary[f"{object_name}_lift_m"] = max(zs) - zs[0]
        summary[f"{object_name}_end_z"] = zs[-1]
    if args.task == "battery":
        xyz, parallel = trace[-1]["battery_local_xyz"], trace[-1]["battery_axis_parallel"]
        summary["battery_final_local_xyz"] = xyz
        summary["battery_final_axis_parallel"] = parallel
        summary["battery_inserted"] = (
            summary["charger_lift_m"] > 0.035 and summary["battery_lift_m"] > 0.020
            and abs(xyz[0] - (-0.011)) < 0.010 and abs(xyz[1] - 0.0215) < 0.004
            and abs(xyz[2] - 0.0170) < 0.005 and parallel > 0.9
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "trace.json").write_text(json.dumps(trace))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
