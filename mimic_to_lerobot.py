"""Isaac Lab Mimic output (HDF5) -> a lerobot dataset in exactly the format lerobot-record writes on the real arm.

    python mimic_to_lerobot.py --hdf5 ~/Stanley_ws/IsaacLab/logs/demos/pringles_sim_generated.hdf5 \
        --repo_id ethanCSL/openarm_pringles_lerobot_mimic_v00

The input must come from generate_dataset.py with the OpenArm recorders (IsaacLab
openarm_recorders.OpenArmLeRobotRecorderManagerCfg, on for OpenArm tasks), which store per step, by joint name:
the measured joints before the step (lerobot/joint_pos) and the joint targets applied during it
(lerobot/joint_pos_target). Each step becomes one frame:

  observation.state           lerobot/joint_pos[i]          measured joints, before the step
  observation.images.<cam>    obs/<cam>[i]                  rendered before the step
  action                      lerobot/joint_pos_target[i]   the joint command of the step -- what the real
                                                            datasets store (the teleop's IK target), not the
                                                            next measured pose

Joints are mapped to motor radians through calibration.json (sim_bridge_common.sim_joints_to_motor_action, the
same mapping openarm_isaac and the real-arm bridge use), so grippers are raw motor angles and the keys are the
follower's RJ1, LJ1, ... order. Features, robot_type (openarm_umeow), fps and video encoding are built the way
lerobot-record builds them for --robot.type=openarm_isaac, so the result merges with the real and
lerobot-recorded sim datasets (lerobot-edit-dataset --operation.type=merge).

Refuses an HDF5 generated at another control rate than --fps (default 30, the real datasets' rate).
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.append(str(REPO))

from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features  # noqa: E402
from lerobot.processor import make_default_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STR  # noqa: E402
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts  # noqa: E402
from lerobot_robot_openarm_umeow import OpenArmIsaac, OpenArmIsaacConfig  # noqa: E402

from sim_bridge_common import load_calibration, sim_joints_to_motor_action  # noqa: E402

TASK = "Pick up the Pringles can with the right arm, hand it to the left arm"


def _rate_hz(f: h5py.File) -> float | None:
    sim = json.loads(f["data"].attrs.get("env_args", "{}")).get("sim_args")
    return None if not sim else 1.0 / (float(sim["dt"]) * int(sim["decimation"]))


def _dataset_features(robot) -> dict:
    """lerobot-record's own construction (lerobot_record.py, identity processors)."""
    teleop_proc, _, obs_proc = make_default_processors()
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc, initial_features=create_initial_features(action=robot.action_features), use_videos=True
        ),
        aggregate_pipeline_dataset_features(
            pipeline=obs_proc, initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )


def _named(group: h5py.Group) -> dict[str, np.ndarray]:
    return {name: group[name][:].reshape(-1) for name in group}


def _episode_frames(ep: h5py.Group, cameras: list[str], calib: dict):
    """Yield (observation, action) per step, as the robot's observation dict and action dict."""
    if "lerobot" not in ep:
        raise ValueError(
            "no lerobot/joint_pos(_target) in this episode: generate it with the current generate_dataset.py,"
            " which records them for OpenArm tasks (IsaacLab openarm_recorders.py)."
        )
    pos, tgt = _named(ep["lerobot/joint_pos"]), _named(ep["lerobot/joint_pos_target"])
    images = {}
    for cam in cameras:
        if f"obs/{cam}" not in ep:
            raise ValueError(f"camera '{cam}' not in the episode (it has obs/{sorted(ep['obs'])}).")
        img = ep[f"obs/{cam}"][:]
        images[cam] = img[:, 0] if img.ndim == 5 else img
    steps = len(next(iter(tgt.values())))
    for i in range(steps):
        obs = sim_joints_to_motor_action({k: float(v[i]) for k, v in pos.items()}, calib)
        for cam in cameras:
            obs[cam] = np.ascontiguousarray(images[cam][i, ..., :3], dtype=np.uint8)
        action = sim_joints_to_motor_action({k: float(v[i]) for k, v in tgt.items()}, calib)
        yield obs, action


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hdf5", nargs="+", required=True, help="generate_dataset.py output file(s)")
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--root", default=None, help="Local dataset directory (default: lerobot's cache for repo_id).")
    ap.add_argument("--task", default=TASK, help="The task string: use the real datasets' exactly.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cameras", nargs="+", default=["body_cam", "wrist_cam", "right_wrist_cam"])
    ap.add_argument("--calibration", default=str(REPO / "calibration.json"))
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--include_failed", action="store_true", help="Also convert episodes not marked successful.")
    ap.add_argument("--push_to_hub", action="store_true")
    args = ap.parse_args()

    calib = load_calibration(args.calibration)
    robot = OpenArmIsaac(OpenArmIsaacConfig(sim_cameras=args.cameras, calibration=args.calibration))  # features only
    for path in args.hdf5:
        with h5py.File(path, "r") as f:
            hz = _rate_hz(f)
            if hz is None or abs(hz - args.fps) > 1e-6:
                raise SystemExit(f"{path} was generated at {hz} Hz, not {args.fps}: regenerate at {args.fps} Hz.")

    dataset = LeRobotDataset.create(
        args.repo_id, args.fps, root=args.root, robot_type=robot.name, features=_dataset_features(robot),
        use_videos=True, streaming_encoding=True, encoder_threads=2,
    )
    converted = skipped = 0
    for path in args.hdf5:
        with h5py.File(path, "r") as f:
            names = sorted(f["data"], key=lambda n: int(n.split("_")[-1]))
            for name in names:
                if args.max_episodes is not None and converted >= args.max_episodes:
                    break
                ep = f["data"][name]
                if not args.include_failed and not bool(ep.attrs.get("success", True)):
                    skipped += 1
                    continue
                n = 0
                for obs, action in _episode_frames(ep, args.cameras, calib):
                    frame = {**build_dataset_frame(dataset.features, obs, prefix=OBS_STR),
                             **build_dataset_frame(dataset.features, action, prefix=ACTION), "task": args.task}
                    dataset.add_frame(frame)
                    n += 1
                dataset.save_episode()
                converted += 1
                print(f"[mimic_to_lerobot] {Path(path).name}:{name} -> episode {dataset.num_episodes - 1} ({n} frames)",
                      flush=True)
    dataset.finalize()
    print(f"[mimic_to_lerobot] {converted} episodes ({dataset.meta.total_frames} frames) -> {dataset.root}"
          + (f"; {skipped} unsuccessful skipped" if skipped else ""), flush=True)
    if args.push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
