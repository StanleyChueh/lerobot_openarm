"""lerobot-record: continue an existing LOCAL dataset automatically instead of failing.

lerobot-record needs `--resume=true` AND `--dataset.root=<local folder>` to add episodes to a dataset it
already recorded; without them it fails (the folder exists, or resume() refuses the shared Hub cache). Its
plugins are imported before it parses the command line, so this looks first: if
$HF_LEROBOT_HOME/<repo_id> (by default ~/.cache/huggingface/lerobot/<repo_id>) already holds episodes, it
adds those two flags and says so. Given flags win: `--resume=false` keeps lerobot's own behaviour, and an
explicit `--dataset.root` is kept. `--dataset.num_episodes` is the TOTAL the dataset should end with
(lerobot's meaning, episodes to add, is what it is turned into); if the dataset already has that many, the
run stops before connecting to the robot.

Only the local copy is continued, never the Hub's. Without local episodes a new dataset is recorded, even
with `--resume=true` (dropped: lerobot's resume would download the Hub's copy). An empty folder left by a
run stopped before its first save is removed first.

If the Hub already has episodes under the same repo_id, this run's push at the end replaces that dataset:
the upload also deletes the Hub's data/, meta/ and videos/ files this recording does not have, in the same
commit. lerobot's upload only adds and overwrites, and lerobot loads every parquet file under data/ and
meta/episodes/, so leftover files of the old dataset would mix into the new one. The old dataset stays in
the Hub repo's commit history.
"""

import json
import shutil
import sys
from pathlib import Path


def _value(argv: list[str], flag: str) -> str | None:
    for i, a in enumerate(argv):
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _drop(argv: list[str], flag: str) -> None:
    """Remove `flag=value` or `flag value` from argv."""
    i = 0
    while i < len(argv):
        if argv[i].startswith(flag + "="):
            del argv[i]
        elif argv[i] == flag:
            del argv[i:i + 2]
        else:
            i += 1


def _local_episodes(folder: Path) -> int:
    try:
        return int(json.loads((folder / "meta" / "info.json").read_text())["total_episodes"])
    except (OSError, ValueError, KeyError):
        return 0


def _is_empty_stub(folder: Path) -> bool:
    """A folder lerobot created for a run stopped before its first save: metadata only, 0 episodes."""
    return (folder.is_dir() and _local_episodes(folder) == 0
            and not any((folder / d).exists() for d in ("data", "videos", "images")))


def _hub_episodes(repo_id: str) -> int:
    """Episodes the Hub has under `repo_id`; 0 if none, not there or offline. Only for the warning below."""
    try:
        from huggingface_hub import hf_hub_download

        info = json.loads(Path(hf_hub_download(repo_id, "meta/info.json", repo_type="dataset",
                                               force_download=True)).read_text())
        return int(info["total_episodes"])
    except Exception:
        return 0


_REPLACED = ["data/**", "meta/**", "videos/**"]


def _replace_on_push(repo_id: str) -> None:
    """Make lerobot's push of `repo_id` delete the Hub's dataset files that the new recording does not have."""
    import lerobot.datasets.lerobot_dataset as ds

    class ReplacingHfApi(ds.HfApi):
        def upload_folder(self, *args, **kwargs):
            if kwargs.get("repo_id") == repo_id and kwargs.get("delete_patterns") is None:
                kwargs["delete_patterns"] = _REPLACED  # files uploaded in the same commit are kept
            return super().upload_folder(*args, **kwargs)

    ds.HfApi = ReplacingHfApi


_DEFAULT_EPISODES = 50  # lerobot's DatasetRecordConfig.num_episodes
_INSTALLED = [False]


def install() -> None:
    if _INSTALLED[0]:
        return
    _INSTALLED[0] = True
    argv = sys.argv
    command = argv[0].rsplit("/", 1)[-1] if argv else ""
    if "lerobot-record" not in command and not command.endswith("lerobot_record.py"):
        return
    repo_id = _value(argv, "--dataset.repo_id")
    resume = (_value(argv, "--resume") or "").lower()
    if not repo_id or resume in ("false", "0", "no"):
        return
    root = _value(argv, "--dataset.root")
    if root is None:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        folder = Path(HF_LEROBOT_HOME) / repo_id
    else:
        folder = Path(root).expanduser()
    episodes = _local_episodes(folder)
    if episodes <= 0:  # nothing local: lerobot creates a new dataset
        if resume in ("true", "1", "yes"):
            argv[:] = [a for a in argv if not a.startswith("--resume")]
            print(f"[openarm] {repo_id}: no local episodes in {folder}: recording a new dataset"
                  " (--resume dropped: only a local dataset is continued, never the Hub's).", flush=True)
        if root is None and _is_empty_stub(folder):
            shutil.rmtree(folder)  # metadata only, 0 episodes: left by a run stopped before its first save
        push = (_value(argv, "--dataset.push_to_hub") or "true").lower() not in ("false", "0", "no")
        hub = _hub_episodes(repo_id) if push else 0
        if hub > 0:
            _replace_on_push(repo_id)
            print(f"[openarm] WARNING: the Hub already has {hub} episode(s) under {repo_id}. This run records a new"
                  " dataset, and its push at the end REPLACES the Hub's (its old files are removed; they stay in the"
                  " repo's history). To keep them: Ctrl-C now and use another --dataset.repo_id, or add"
                  " --dataset.push_to_hub=false.", flush=True)
        return
    added = []
    if resume not in ("true", "1", "yes"):
        argv[:] = [a for a in argv if not a.startswith("--resume")] + ["--resume=true"]
        added.append("--resume=true")
    if root is None:
        argv.append(f"--dataset.root={folder}")
        added.append(f"--dataset.root={folder}")
    given = _value(argv, "--dataset.num_episodes")
    target = int(given) if given and given.isdigit() else _DEFAULT_EPISODES
    if target <= episodes:
        print(f"[openarm] {repo_id} already has {episodes} episode(s) in {folder}, and --dataset.num_episodes"
              f"={target} is the total to reach: nothing to record. Raise --dataset.num_episodes to record more.",
              flush=True)
        sys.exit(0)
    _drop(argv, "--dataset.num_episodes")
    argv.append(f"--dataset.num_episodes={target - episodes}")  # lerobot counts episodes to add
    print(f"[openarm] {repo_id} already has {episodes} episode(s) in {folder}: continuing it"
          + (f" (added {' '.join(added)})" if added else "")
          + f". Recording {target - episodes} more, episodes {episodes}-{target - 1}, to reach {target} in total."
          " To start a separate dataset instead, use another --dataset.repo_id.", flush=True)
