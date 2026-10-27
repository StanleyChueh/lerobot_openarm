"""--record_repo_id=<user>/eval_<name>: the evaluation episodes of robot_client.py, saved as a LeRobot dataset.

Every executed action is one frame, built as lerobot-record builds it: the observation the action was sent
from (read right before it) and the action itself (the policy's command, the dataset's action meaning). In
simulation that read costs nothing: openarm_isaac's observation is the reply of the last step. In lockstep
each frame is one control period of sim time, so the dataset's timestamps are the simulator's. On the real arm
the read takes 36-53 ms every tick and slows the control loop.

The name must start with "eval_" (lerobot-record refuses that prefix for collected data), and so must the
directory of --record_root: only an evaluation dataset is ever written over. If it already exists:

  --resume=true   the new episodes are added to it (--num_episodes more). The local copy is used, or the Hub's
                  is downloaded when there is none (all of it: new episodes are appended to the last video
                  file); a local copy with fewer episodes than the Hub's is refused (it is stale: appending to
                  it would overwrite the Hub's newer files).
  (default)       a fresh evaluation replaces it: the local copy is deleted now; the Hub copy is left as it is
                  until this evaluation is pushed, then every file of the old one is removed from it.

A rerun episode (Left arrow) is discarded. The dataset is pushed once every episode has been run; a run stopped
early (Esc, Ctrl-C, an error) keeps it local, not pushed (--resume=true continues it). --push_to_hub=false keeps it
local only: the Hub is neither checked nor written, and the rules above apply to the local copy alone.

meta/openarm_eval.json, pushed with it, lists each saved episode's outcome: the simulator's task success
condition (null on the real arm), seconds, executed actions and why it ended.
"""

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

PREFIX = "eval_"
EVAL_JSON = "meta/openarm_eval.json"
_DATASET_ENTRIES = {"meta", "data", "videos", "images"}  # all a LeRobot dataset directory may hold


@dataclass
class Target:
    repo_id: str
    root: Path
    mode: str  # "new", "overwrite" or "resume"
    on_hub: bool
    push: bool = True  # False: local only (--push_to_hub=false)


def _episodes_in(info_json: Path) -> int:
    try:
        return int(json.loads(info_json.read_text())["total_episodes"])
    except (OSError, ValueError, KeyError):
        return 0


def _delete_local(root: Path) -> None:
    """Delete an evaluation dataset directory, but only one that holds nothing else."""
    other = {p.name for p in root.iterdir()} - _DATASET_ENTRIES
    if other:
        raise SystemExit(f"[openarm] {root} holds {sorted(other)}, which no LeRobot dataset holds: not deleting it."
                         " Choose another --record_root.")
    shutil.rmtree(root)


def check_repo_id(repo_id: str, root: str | None = None, resume: bool = False, push: bool = True) -> Target:
    """Check the name and what already exists, before anything connects. Deletes the local copy to overwrite.
    push=False: local only, the Hub is not contacted."""
    owner, _, name = repo_id.partition("/")
    if not owner or not name or "/" in name:
        raise SystemExit(f"[openarm] --record_repo_id must be <user>/{PREFIX}<name>, got {repo_id!r}.")
    if not name.startswith(PREFIX):
        raise SystemExit(f"[openarm] --record_repo_id {repo_id!r}: the dataset name must start with {PREFIX!r}"
                         f" (e.g. {owner}/{PREFIX}{name}), so an evaluation can never overwrite a training dataset.")
    from lerobot.utils.constants import HF_LEROBOT_HOME

    local = Path(root).expanduser().resolve() if root else HF_LEROBOT_HOME / repo_id
    if root and not local.name.startswith(PREFIX):
        raise SystemExit(f"[openarm] --record_root {local}: the directory name must start with {PREFIX!r} too.")
    if not push:
        return _check_local_only(repo_id, local, resume)
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    try:
        user = api.whoami()["name"]
        on_hub = api.repo_exists(repo_id, repo_type="dataset")
        hub_n = _episodes_in(Path(hf_hub_download(repo_id, "meta/info.json", repo_type="dataset",
                                                  revision="main"))) if on_hub else 0
    except Exception as e:  # no token, or offline: the push at the end would fail after the whole evaluation
        raise SystemExit(f"[openarm] cannot check {repo_id} on the Hugging Face Hub ({e!r}); log in with"
                         " `hf auth login` and check the network before recording.") from e
    local_n = _episodes_in(local / "meta" / "info.json") if local.exists() else 0
    if local.exists() and local_n == 0:  # never saved an episode (stopped before the first): nothing to keep
        _delete_local(local)

    where = f"{local}; pushed to the Hub as {repo_id} (user {user}) once every episode has been run"
    if not local_n and not hub_n:
        if resume:
            print(f"[openarm] --resume=true: {repo_id} has no episodes yet, starting it.", flush=True)
        mode = "new"
        print(f"[openarm] recording the evaluation to {where}.", flush=True)
    elif resume:
        if local_n and local_n < hub_n:
            raise SystemExit(f"[openarm] {local} has {local_n} episodes, the Hub's {repo_id} has {hub_n}: the local"
                             " copy is stale. Move it away to resume from the Hub's.")
        if not local_n:
            from huggingface_hub import snapshot_download

            print(f"[openarm] downloading {repo_id} from the Hub to resume it ...", flush=True)
            snapshot_download(repo_id, repo_type="dataset", revision="main", local_dir=local,
                              allow_patterns=["meta/*", "data/*", "videos/*"])
        mode = "resume"
        print(f"[openarm] --resume=true: adding to {repo_id} ({max(local_n, hub_n)} episodes"
              + (f", {local_n - hub_n} of them not pushed yet" if local_n > hub_n else "")
              + f") in {where}.", flush=True)
    else:
        mode = "overwrite"
        if local_n:
            _delete_local(local)
            print(f"[openarm] OVERWRITING: deleted the local {repo_id} ({local_n} episodes).", flush=True)
        if hub_n:
            print(f"[openarm] OVERWRITING: the Hub's {repo_id} ({hub_n} episodes) is replaced when this evaluation"
                  " is pushed, and kept if it stops early. --resume=true would add to it instead.", flush=True)
        print(f"[openarm] recording a fresh evaluation to {where}.", flush=True)
    return Target(repo_id, local, mode, on_hub)


def _check_local_only(repo_id: str, local: Path, resume: bool) -> Target:
    local_n = _episodes_in(local / "meta" / "info.json") if local.exists() else 0
    if local.exists() and local_n == 0:
        _delete_local(local)
    where = f"{local} (local only: --push_to_hub=false, never pushed)"
    if not local_n:
        mode = "new"
        print(f"[openarm] recording the evaluation to {where}.", flush=True)
    elif resume:
        mode = "resume"
        print(f"[openarm] --resume=true: adding to {repo_id} ({local_n} episodes) in {where}.", flush=True)
    else:
        mode = "overwrite"
        _delete_local(local)
        print(f"[openarm] OVERWRITING: deleted the local {repo_id} ({local_n} episodes); recording a fresh evaluation"
              f" to {where}.", flush=True)
    return Target(repo_id, local, mode, on_hub=False, push=False)


def _check_compatible(meta, features: dict, fps: int, robot_type: str) -> None:
    """The episodes to add must have the dataset's fps, robot and features."""
    wrong = []
    if meta.fps != fps:
        wrong.append(f"fps {meta.fps} (now {fps})")
    if meta.robot_type != robot_type:
        wrong.append(f"robot_type {meta.robot_type} (now {robot_type})")
    for key, ft in features.items():
        have = meta.features.get(key)
        if have is None or [have["dtype"], list(have["shape"]), have.get("names")] != \
                [ft["dtype"], list(ft["shape"]), ft.get("names")]:
            wrong.append(f"{key}: {have and {k: have.get(k) for k in ('dtype', 'shape', 'names')}} (now {ft})")
    extra = [k for k in meta.features if k not in features and k not in
             ("timestamp", "frame_index", "episode_index", "index", "task_index")]
    wrong += [f"{k}: not recorded now" for k in extra]
    if wrong:
        raise SystemExit("[openarm] --resume=true: this run does not match the dataset:\n  " + "\n  ".join(wrong))


class EvalRecorder:
    def __init__(self, target: Target, robot, fps: float, task: str, policy: str | None):
        from lerobot.datasets import LeRobotDataset
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

        if int(fps) != fps:
            raise SystemExit(f"[openarm] --fps={fps:g}: a dataset needs a whole number of frames per second.")
        features = {**hw_to_dataset_features(robot.action_features, ACTION, True),
                    **hw_to_dataset_features(robot.observation_features, OBS_STR, True)}
        threads = 4 * sum(1 for f in features.values() if f["dtype"] == "video")  # lerobot-record's 4 per camera
        self.target = target
        self.episodes: list[dict] = []
        self.saved = 0  # this run's
        if target.mode == "resume":
            self.dataset = LeRobotDataset.resume(target.repo_id, root=target.root, revision="main",
                                                 image_writer_threads=threads)
            _check_compatible(self.dataset.meta, features, int(fps), robot.name)
            eval_json = target.root / EVAL_JSON
            if eval_json.exists():
                self.episodes = json.loads(eval_json.read_text())["episodes"]
        else:
            self.dataset = LeRobotDataset.create(target.repo_id, int(fps), root=target.root, robot_type=robot.name,
                                                 features=features, use_videos=True, image_writer_threads=threads)
        self._frame = build_dataset_frame
        self._obs_prefix, self._action_prefix = OBS_STR, ACTION
        self.task = task
        self.policy = policy

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
        self.episodes.append({"episode_index": index, "policy": self.policy, **outcome})
        self.saved += 1
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
        if not self.saved:
            if self.target.mode == "resume":
                print(f"[openarm] no episode added: {repo_id} left as it was.", flush=True)
            else:
                print(f"[openarm] no episode saved: {repo_id} not pushed; {root} removed.", flush=True)
                shutil.rmtree(root, ignore_errors=True)
            return
        sims = [e["success"] for e in self.episodes if e.get("success") is not None]
        (root / EVAL_JSON).write_text(json.dumps({
            "task": self.task, "success_rate": sum(sims) / len(sims) if sims else None,
            "episodes": self.episodes}, indent=2) + "\n")
        if sims:
            print(f"[openarm] {repo_id}: {sum(sims)}/{len(sims)} episodes met the task success condition.", flush=True)
        if not self.target.push:
            print(f"[openarm] {self.dataset.num_episodes} episode(s) kept in {root} (local only, not pushed).", flush=True)
            return
        if not push:
            print(f"[openarm] the evaluation stopped early: {self.dataset.num_episodes} episode(s) kept in {root},"
                  " NOT pushed. Run again with --resume=true to continue it.", flush=True)
            return
        print(f"[openarm] pushing {repo_id} ({self.dataset.num_episodes} episodes) to the Hugging Face Hub ...",
              flush=True)
        try:
            self.dataset.push_to_hub()
            if self.target.mode == "overwrite" and self.target.on_hub:
                self._remove_old_hub_files(root)
        except Exception as e:
            print(f"[openarm] push FAILED ({e!r}); the dataset is kept in {root}. To push it: python -c \"from"
                  f" lerobot.datasets import LeRobotDataset; LeRobotDataset('{repo_id}', root='{root}').push_to_hub()\"",
                  flush=True)
            return
        print(f"[openarm] pushed: https://huggingface.co/datasets/{repo_id}", flush=True)

    def _remove_old_hub_files(self, root: Path) -> None:
        """After the fresh evaluation is uploaded: delete the files only the old one had."""
        from huggingface_hub import CommitOperationDelete, HfApi

        api = HfApi()
        mine = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        old = [f for f in api.list_repo_files(self.target.repo_id, repo_type="dataset")
               if f not in mine and f not in (".gitattributes", "README.md")]
        if old:
            api.create_commit(self.target.repo_id, repo_type="dataset",
                              operations=[CommitOperationDelete(path_in_repo=f) for f in old],
                              commit_message="Remove the files of the overwritten evaluation")
            print(f"[openarm] removed {len(old)} file(s) of the overwritten evaluation from the Hub.", flush=True)


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
