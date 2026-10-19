"""Build a local LeRobot sim dataset from real action replays that pass physical task checks.

Start the matching Isaac server first. The script never sends commands to a physical robot and
never uploads a dataset. It records the simulator's own state and three camera images at 30 Hz,
paired with the original real action command for each step. Failed replays are discarded.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot_robot_openarm_umeow import OpenArmIsaac, OpenArmIsaacConfig
from lerobot_robot_openarm_umeow.sim_link import SimClient

from mimic_to_lerobot import _dataset_features
from replay_real_trajectory_in_sim import REAL, STARTS, load_episode, local_battery_pose
from sim_bridge_common import load_calibration, motor_action_to_sim_joints, sim_joints_to_motor_action

TASK_TEXT = {
    "pringles": "Pick up the Pringles can with the right arm, hand it to the left arm",
    "battery": "Pick up the charging station with the left arm, and insert battery in it with the right arm",
}


def parse_episodes(value):
    out = []
    for part in value.split(","):
        lo, sep, hi = part.partition("-")
        out.extend(range(int(lo), int(hi) + 1) if sep else [int(lo)])
    return out


def battery_seated(points):
    if len(points) < 15:
        return False
    for point in points[-15:]:
        xyz, axis = point["battery_local_xyz"], point["battery_axis_parallel"]
        if not (abs(xyz[0] + 0.011) < 0.010 and abs(xyz[1] - 0.0215) < 0.004
                and abs(xyz[2] - 0.0170) < 0.005 and axis > 0.9):
            return False
    return True


def validate(task, trace):
    if task == "pringles":
        heights = [p["objects"]["can"][2] for p in trace]
        consecutive = all(p["success"] for p in trace[-15:])
        return consecutive and max(heights) - heights[0] > 0.04 and heights[-1] > 0.385
    charger_heights = [p["objects"]["charger"][2] for p in trace]
    battery_heights = [p["objects"]["battery"][2] for p in trace]
    return (max(charger_heights) - charger_heights[0] > 0.035
            and max(battery_heights) - battery_heights[0] > 0.020
            and charger_heights[-1] - charger_heights[0] > 0.025
            and battery_seated(trace)
            and all(p["success"] for p in trace[-15:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=REAL, required=True)
    parser.add_argument("--episodes", required=True, help="Comma-separated episode numbers/ranges, e.g. 0,4,5")
    parser.add_argument("--starts", help="JSON map episode -> asset -> [x,y,yaw_deg]")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", required=True, help="New local dataset directory")
    parser.add_argument("--port", type=int, default=5710)
    parser.add_argument("--calibration", default=str(Path(__file__).with_name("calibration.json")))
    args = parser.parse_args()
    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"Dataset directory already exists: {root}")
    starts_by_episode = json.loads(Path(args.starts).read_text()) if args.starts else {}
    calib = load_calibration(args.calibration)
    client = SimClient(port=args.port)
    hello = client.connect()
    if hello.get("real_dataset") != REAL[args.task] or hello.get("control_hz") != 30:
        raise SystemExit(f"Server profile/rate mismatch: {hello}")
    cameras = list(hello["cameras"])
    robot = OpenArmIsaac(OpenArmIsaacConfig(sim_cameras=cameras, calibration=args.calibration))
    dataset = LeRobotDataset.create(
        args.repo_id, 30, root=root, robot_type=robot.name, features=_dataset_features(robot),
        use_videos=True, streaming_encoding=True, encoder_threads=2,
    )
    report = {"task": args.task, "source": REAL[args.task], "sim_task": hello["task"],
              "repo_id": args.repo_id, "episodes": []}
    try:
        for ep in parse_episodes(args.episodes):
            names, states, actions = load_episode(REAL[args.task], ep)
            starts = starts_by_episode.get(str(ep), STARTS[args.task])
            initial = motor_action_to_sim_joints(dict(zip(names, states[0])), calib)
            obs = client.request("reset", joints=initial, objects=starts)
            trace = []
            for k, values in enumerate(actions):
                motor_action = dict(zip(names, map(float, values)))
                motor_obs = sim_joints_to_motor_action(obs["joints"], calib)
                motor_obs.update(obs["images"])
                frame = {**build_dataset_frame(dataset.features, motor_obs, prefix=OBS_STR),
                         **build_dataset_frame(dataset.features, motor_action, prefix=ACTION),
                         "task": TASK_TEXT[args.task]}
                dataset.add_frame(frame)
                obs = client.request("step", joints=motor_action_to_sim_joints(motor_action, calib))
                point = {"step": k, "objects": obs["objects"], "success": bool(obs["success"])}
                if args.task == "battery":
                    point["battery_local_xyz"], point["battery_axis_parallel"] = local_battery_pose(obs["objects"])
                trace.append(point)
                if k % 100 == 0:
                    print(f"episode {ep}: {k}/{len(actions)} steps", flush=True)
            accepted = validate(args.task, trace)
            info = {"real_episode": ep, "frames": len(trace), "accepted": accepted, "start_poses": starts,
                    "final_success_term": bool(trace[-1]["success"]),
                    "final_object_poses": trace[-1]["objects"]}
            if args.task == "battery":
                info["final_battery_local_xyz"] = trace[-1]["battery_local_xyz"]
                info["final_battery_axis_parallel"] = trace[-1]["battery_axis_parallel"]
            if accepted:
                dataset.save_episode()
                info["sim_episode"] = dataset.num_episodes - 1
            else:
                dataset.clear_episode_buffer()
            report["episodes"].append(info)
            print(f"episode {ep}: {'SAVED' if accepted else 'rejected'}", flush=True)
            root.mkdir(parents=True, exist_ok=True)
            (root / "validation.json").write_text(json.dumps(report, indent=2))
    finally:
        client.close()
        dataset.finalize()
    print(f"Saved {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames at {root}", flush=True)


if __name__ == "__main__":
    main()
