"""Code provenance shared by the tools: which exact files ran (baseline modules) and which code made an upload."""
import hashlib, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]          # .../research/sim_free_augmentation/sfagen (tracked)
SFA = HERE.parent                                   # .../research/sim_free_augmentation (frozen studies)
LEGACY = HERE / "legacy"
SAM2 = "facebook/sam2.1-hiera-small"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


EXTERNAL = Path.home() / "Stanley_ws/sfa_hf_export"  # the old untracked working copies


def _project_file(f):
    """True for a loaded research .py: from this repo's research tree or from the old untracked working copies."""
    if not Path(f).is_absolute():  # e.g. torch._classes reports a bare "_classes.py"
        return False
    p = Path(f).resolve()
    return p.suffix == ".py" and (p.is_relative_to(SFA) or p.is_relative_to(EXTERNAL.resolve()))


def baseline_modules():
    """Every project module loaded so far must be tracked: inside the repo, or byte-identical to a tracked legacy file of the
    same name (some frozen baseline modules prepend absolute paths to the old untracked copies). Raises otherwise."""
    tracked = {}
    for p in LEGACY.rglob("*.py"):
        tracked.setdefault(p.name, set()).add(sha256(p))
    rows, bad = [], []
    for name, m in sorted(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f or not _project_file(f):
            continue
        p = Path(f).resolve(); h = sha256(p); inside = p.is_relative_to(SFA)
        rows.append({"module": name, "file": str(p), "sha256": h, "in_repo": inside})
        if not inside and h not in tracked.get(p.name, ()):
            bad.append(str(p))
    if bad:
        raise RuntimeError(f"untracked code was imported (not in git and not identical to a tracked legacy copy): {bad}")
    return rows


def git_state():
    run = lambda *a: subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain", "--", ".").splitlines()}


def sam2_revision():
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(SAM2, "config.json")
    return Path(p).parent.name if isinstance(p, str) else None   # .../snapshots/<commit>/config.json


def upload_provenance(code_files):
    """Exact code identity for a dataset upload: git commit, uncommitted changes, per-file sha256, lockfile, SAM2 revision."""
    lock = SFA.parents[1] / "uv.lock"
    return {**git_state(), "files": {str(Path(p).resolve().relative_to(HERE)): sha256(p) for p in code_files},
            "uv_lock_sha256": sha256(lock) if lock.exists() else None, "sam2_model": SAM2, "sam2_revision": sam2_revision(),
            "python": sys.version.split()[0]}
