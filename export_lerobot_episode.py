#!/usr/bin/env python
"""Export a single episode's action sequence from a joint-space LeRobot dataset to a plain
.npy file, for replay inside Isaac Sim.

Isaac Sim's own Python environment can't directly import the `lerobot` package -- that's exactly
why eval_smolvla.py/smolvla_server.py are split into two processes talking over TCP (see those
files' docstrings: numpy 1.x in the isaaclab env vs numpy 2.x in the lerobot env). For replaying a
*live policy* that split is necessary every step. For replaying an already-recorded, fixed
trajectory it isn't -- the whole sequence is known upfront, so it's simpler to extract it once
here (in this venv, which has `lerobot` installed) to a plain float32 .npy array, and let the
Isaac-Sim-side script (replay_lerobot_dataset_sim.py) load that file directly with no `lerobot`
import and no live connection at all.

Only supports joint-space-action datasets (action names LJ1.pos..LJ8.pos, see IsaacLab's
convert_hdf5_to_lerobot.py) -- aborts with a clear error on a dataset with a different schema
(e.g. an older EE-delta-action dataset) rather than silently exporting the wrong thing.

Usage:
  python export_lerobot_episode.py \\
      --repo-id ethanCSL/openarm_visuomotor_no_domain_randomization_1000_joints \\
      --episode 0 --output /tmp/episode0_actions.npy
"""

import argparse

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION

LJ_NAMES = [f"LJ{i}.pos" for i in range(1, 8)] + ["LJ8.pos"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", type=str, required=True, help="HF dataset repo id")
    ap.add_argument("--episode", type=int, default=0, help="Episode index inside the dataset")
    ap.add_argument("--output", type=str, required=True, help="Path to write the .npy trajectory to")
    args = ap.parse_args()

    print(f"Loading {args.repo_id}, episode {args.episode}...")
    # revision="main": same reasoning as replay_hf_sim_episode.py -- this dataset has no version
    # tag on the Hub, and LeRobotDataset's default version-tag resolution throws on untagged
    # repos in this environment's huggingface_hub version. Pinning to "main" skips that entirely.
    dataset = LeRobotDataset(args.repo_id, episodes=[args.episode], revision="main", force_cache_sync=True)

    action_names = dataset.features[ACTION]["names"]
    if action_names != LJ_NAMES:
        raise ValueError(
            f"Expected joint-space action names {LJ_NAMES}, got {action_names} -- this export "
            "only supports joint-space-action datasets (see replay_hf_sim_episode.py if you need "
            "to replay an older EE-delta-action dataset instead)."
        )

    actions = dataset.select_columns(ACTION)
    traj = np.stack([actions[i][ACTION].numpy() for i in range(dataset.num_frames)]).astype(np.float32)

    np.save(args.output, traj)
    print(f"Exported {len(traj)} steps, shape {traj.shape} (columns: {LJ_NAMES}), to {args.output}")


if __name__ == "__main__":
    main()
