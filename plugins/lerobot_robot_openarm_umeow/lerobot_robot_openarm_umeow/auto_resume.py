"""lerobot-record: continue an existing local dataset automatically instead of failing.

lerobot-record needs `--resume=true` AND `--dataset.root=<local folder>` to add episodes to a dataset it
already recorded; without them it fails (the folder exists, or resume() refuses the shared Hub cache). Its
plugins are imported before it parses the command line, so this looks first: if
$HF_LEROBOT_HOME/<repo_id> (by default ~/.cache/huggingface/lerobot/<repo_id>) already holds episodes, it
adds those two flags and says so. Given flags win: `--resume=false` keeps lerobot's own behaviour, and an
explicit `--dataset.root` is kept. `--dataset.num_episodes` keeps lerobot's meaning: episodes to ADD.

If the local folder is missing (or is an empty stub left by a run stopped before its first save) but the
Hub already has the dataset, it is downloaded first and then continued. Otherwise a new, local dataset
would be created under the same repo_id and its push would OVERWRITE the episodes on the Hub. Offline,
or with --dataset.push_to_hub=false, the Hub is not consulted.
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


def _local_episodes(folder: Path) -> int:
    try:
        return int(json.loads((folder / "meta" / "info.json").read_text())["total_episodes"])
    except (OSError, ValueError, KeyError):
        return 0


def _is_empty_stub(folder: Path) -> bool:
    """A folder lerobot created for a run stopped before its first save: metadata only, 0 episodes."""
    return (folder.is_dir() and _local_episodes(folder) == 0
            and not any((folder / d).exists() for d in ("data", "videos", "images")))


def _fetch_from_hub(repo_id: str, folder: Path) -> int:
    """Download the Hub's copy into `folder` if it has episodes; their count, or 0."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        info = json.loads(Path(hf_hub_download(repo_id, "meta/info.json", repo_type="dataset",
                                               force_download=True)).read_text())
        hub_episodes = int(info["total_episodes"])
    except Exception:
        return 0  # not on the Hub (or offline): a new dataset
    if hub_episodes <= 0:
        return 0
    if folder.exists():
        if not _is_empty_stub(folder):
            raise SystemExit(f"[openarm] {folder} exists but holds no saved episodes, while the Hub has {hub_episodes}"
                             f" for {repo_id}. Move that folder away (or use another --dataset.repo_id) and rerun.")
        shutil.rmtree(folder)  # metadata only, 0 episodes: left by a run stopped before its first save
    print(f"[openarm] {repo_id}: no local copy, the Hub has {hub_episodes} episode(s) -- downloading it to {folder}"
          " to continue it (recording a new dataset under this name would overwrite them on the Hub).", flush=True)
    snapshot_download(repo_id, repo_type="dataset", local_dir=str(folder))
    return _local_episodes(folder)


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
    if episodes <= 0 and root is None and (_value(argv, "--dataset.push_to_hub") or "true").lower() not in ("false", "0", "no"):
        episodes = _fetch_from_hub(repo_id, folder)
    if episodes <= 0:
        return  # nothing anywhere yet: lerobot creates a new dataset as usual
    added = []
    if resume not in ("true", "1", "yes"):
        argv[:] = [a for a in argv if not a.startswith("--resume")] + ["--resume=true"]
        added.append("--resume=true")
    if root is None:
        argv.append(f"--dataset.root={folder}")
        added.append(f"--dataset.root={folder}")
    more = _value(argv, "--dataset.num_episodes")
    total = f" -> {episodes + int(more)} in total" if more and more.isdigit() else ""
    print(f"[openarm] {repo_id} already has {episodes} episode(s) in {folder}: continuing it"
          + (f" (added {' '.join(added)})" if added else "")
          + f". Recording {more or 'the requested number of'} more{total}."
          " To start a separate dataset instead, use another --dataset.repo_id.", flush=True)
