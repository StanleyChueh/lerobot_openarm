"""lerobot-record: continue an existing local dataset automatically instead of failing.

lerobot-record needs `--resume=true` AND `--dataset.root=<local folder>` to add episodes to a dataset it
already recorded; without them it fails (the folder exists, or resume() refuses the shared Hub cache). Its
plugins are imported before it parses the command line, so this looks first: if
$HF_LEROBOT_HOME/<repo_id> (by default ~/.cache/huggingface/lerobot/<repo_id>) already holds episodes, it
adds those two flags and says so. Given flags win: `--resume=false` keeps lerobot's own behaviour, and an
explicit `--dataset.root` is kept. `--dataset.num_episodes` keeps lerobot's meaning: episodes to ADD.
"""

import json
import sys
from pathlib import Path


def _value(argv: list[str], flag: str) -> str | None:
    for i, a in enumerate(argv):
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


def install() -> None:
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
    try:
        episodes = int(json.loads((folder / "meta" / "info.json").read_text())["total_episodes"])
    except (OSError, ValueError, KeyError):
        return  # nothing there yet: lerobot creates a new dataset as usual
    if episodes <= 0:
        return
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
