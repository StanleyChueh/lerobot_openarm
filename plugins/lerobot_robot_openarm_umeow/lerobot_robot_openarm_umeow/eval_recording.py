"""--record_repo_id=<user>/eval_<name>: the evaluation episodes of robot_client.py, saved as a LeRobot dataset.

Every executed action is one frame, built as lerobot-record builds it: the observation the action was sent
from (read right before it) and the action itself (the policy's command, the dataset's action meaning). In
simulation that read costs nothing: openarm_isaac's observation is the reply of the last step. In lockstep
each frame is one control period of sim time, so the dataset's timestamps are the simulator's. On the real arm
the read takes 36-53 ms every tick and slows the control loop.

The name must start with "eval_" (lerobot-record refuses that prefix for collected data), and the run is
refused before connecting if the dataset already exists, locally or on the Hub: an evaluation never writes over
a dataset. A rerun episode (Left arrow) is discarded. The dataset is pushed once every episode has been run;
a run stopped early (Esc, Ctrl-C, an error) keeps it local and prints how to push it.

meta/openarm_eval.json, pushed with it, lists each saved episode's outcome: the simulator's task success
condition (null on the real arm), seconds, executed actions and why it ended.
"""

import json
import shutil
import time
from pathlib import Path

PREFIX = "eval_"


def check_repo_id(repo_id: str, root: str | None = None) -> Path:
    """Refuse a name without the eval_ prefix, or a dataset that already exists. Returns the local root."""
    owner, _, name = repo_id.partition("/")
    if not owner or not name or "/" in name:
        raise SystemExit(f"[openarm] --record_repo_id must be <user>/{PREFIX}<name>, got {repo_id!r}.")
    if not name.startswith(PREFIX):
        raise SystemExit(f"[openarm] --record_repo_id {repo_id!r}: the dataset name must start with {PREFIX!r}"
                         f" (e.g. {owner}/{PREFIX}{name}), so an evaluation can never overwrite a training dataset.")
    from lerobot.utils.constants import HF_LEROBOT_HOME

    local = Path(root).expanduser() if root else HF_LEROBOT_HOME / repo_id
    if local.exists():
        raise SystemExit(f"[openarm] {local} already exists: choose another --record_repo_id (or move it away).")
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()["name"]
        exists = api.repo_exists(repo_id, repo_type="dataset")
    except Exception as e:  # no token, or offline: the push at the end would fail after the whole evaluation
        raise SystemExit(f"[openarm] cannot check {repo_id} on the Hugging Face Hub ({e!r}); log in with"
                         " `hf auth login` and check the network before recording.") from e
    if exists:
        raise SystemExit(f"[openarm] the Hub dataset {repo_id} already exists: choose another --record_repo_id.")
    print(f"[openarm] recording the evaluation to {local}; pushed to the Hub as {repo_id} (user {user})"
          " once every episode has been run.", flush=True)
    return local


class EvalRecorder:
    def __init__(self, repo_id: str, root: Path, robot, fps: float, task: str, policy: str | None):
        from lerobot.datasets import LeRobotDataset
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

        if int(fps) != fps:
            raise SystemExit(f"[openarm] --fps={fps:g}: a dataset needs a whole number of frames per second.")
        features = {**hw_to_dataset_features(robot.action_features, ACTION, True),
                    **hw_to_dataset_features(robot.observation_features, OBS_STR, True)}
        cameras = sum(1 for f in features.values() if f["dtype"] == "video")
        self.dataset = LeRobotDataset.create(repo_id, int(fps), root=root, robot_type=robot.name, features=features,
                                             use_videos=True, image_writer_threads=4 * cameras)  # lerobot-record's 4/camera
        self._frame = build_dataset_frame
        self._obs_prefix, self._action_prefix = OBS_STR, ACTION
        self.task = task
        self.policy = policy
        self.episodes: list[dict] = []
        self.complete = False

    def add(self, observation: dict, action: dict) -> None:
        features = self.dataset.features
        self.dataset.add_frame({**self._frame(features, observation, prefix=self._obs_prefix),
                                **self._frame(features, action, prefix=self._action_prefix), "task": self.task})

    def save(self, outcome: dict) -> None:
        """Save the episode just run, with its outcome."""
        if not self.dataset.has_pending_frames():
            print("[openarm] the episode executed no action: nothing to save.", flush=True)
            return
        t0 = time.perf_counter()
        index = self.dataset.num_episodes
        self.dataset.save_episode()
        self.episodes.append({"episode_index": index, **outcome})
        print(f"[openarm] saved as episode {index} of {self.dataset.repo_id}"
              f" ({time.perf_counter() - t0:.1f} s).", flush=True)

    def discard(self) -> None:
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()

    def finish(self, push: bool) -> None:
        """Close the dataset, write the outcomes, and push it if every episode was run."""
        self.discard()  # an episode interrupted by an error or Ctrl-C
        self.dataset.finalize()
        root, repo_id = Path(self.dataset.root), self.dataset.repo_id
        if not self.episodes:
            print(f"[openarm] no episode saved: {repo_id} not pushed; {root} removed.", flush=True)
            shutil.rmtree(root, ignore_errors=True)
            return
        sims = [e["success"] for e in self.episodes if e["success"] is not None]
        (root / "meta" / "openarm_eval.json").write_text(json.dumps({
            "policy": self.policy, "task": self.task, "complete": push,
            "success_rate": sum(sims) / len(sims) if sims else None,
            "episodes": self.episodes}, indent=2) + "\n")
        by_hand = (f"python -c \"from lerobot.datasets import LeRobotDataset;"
                   f" LeRobotDataset('{repo_id}', root='{root}').push_to_hub()\"")
        if not push:
            print(f"[openarm] the evaluation stopped early: {len(self.episodes)} episode(s) kept in {root}, NOT pushed."
                  f" To push them: {by_hand}", flush=True)
            return
        print(f"[openarm] pushing {repo_id} ({len(self.episodes)} episodes) to the Hugging Face Hub ...", flush=True)
        try:
            self.dataset.push_to_hub()
        except Exception as e:
            print(f"[openarm] push FAILED ({e!r}); the dataset is kept in {root}. To push it: {by_hand}", flush=True)
            return
        print(f"[openarm] pushed: https://huggingface.co/datasets/{repo_id}", flush=True)


def client_class(base):
    """`base` that records a frame for every action it executes, once its `recorder` is set."""

    class RecordingClient(base):
        recorder: EvalRecorder | None = None
        _policy_action = None

        def _action_tensor_to_action_dict(self, action_tensor):
            self._policy_action = super()._action_tensor_to_action_dict(action_tensor)
            return self._policy_action

        def control_loop_action(self, verbose: bool = False):
            if self.recorder is None:
                return super().control_loop_action(verbose)
            observation = self.robot.get_observation()  # the state the action is sent from, as lerobot-record
            performed = super().control_loop_action(verbose)
            self.recorder.add(observation, self._policy_action)
            return performed

    return RecordingClient
