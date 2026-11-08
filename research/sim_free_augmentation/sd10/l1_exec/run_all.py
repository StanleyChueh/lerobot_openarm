"""SD10-L1 launcher: pre-launch checks, then the nine registered fits SEQUENTIALLY on one GPU in the frozen run order,
then predictions and frozen scoring. Pre-launch provenance is written before the first optimizer update.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 ../../../../.venv/bin/python -B run_all.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import l1lib as L

PY = str(L.REPO / ".venv/bin/python")


def git(*c, cwd=L.REPO):
    return subprocess.run(["git", *c], cwd=cwd, capture_output=True, text=True).stdout.strip()


def prelaunch():
    L.check_seal()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2")
    v = subprocess.run([PY, "-B", str(L.L1 / "verify_preregistration.py")], cwd=L.REPO, env=env, capture_output=True, text=True)
    assert v.returncode == 0, v.stderr[-2000:]
    ver = json.loads(v.stdout[v.stdout.index("{"):])
    assert ver["PASS"] and ver["locked_artifacts_manifest_sha256"] == L.SEAL_SHA
    assert subprocess.run(["git", "merge-base", "--is-ancestor", L.PT.HEAD, "HEAD"], cwd=L.REPO).returncode == 0
    rel = str(L.EXEC.relative_to(L.REPO))
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", rel).splitlines()
    dirty = [d for d in dirty if "/outputs/" not in d and "__pycache__" not in d]
    assert not dirty, f"commit the execution code before launch: {dirty}"
    lib = L.REPO.parent / "lerobot"
    assert git("rev-parse", "HEAD", cwd=lib) == L.PT.LEROBOT_HEAD and not git("status", "--porcelain", cwd=lib)
    import lerobot
    assert lerobot.__file__.startswith(str(L.LEROBOT_SRC)), lerobot.__file__
    rep = json.loads((L.OUT / "frame_cache_report.json").read_text())
    assert rep["eval_rgb_pts_timebase_all_match_manifest"]
    pre = {"time": time.time(), "git_head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
           "exec_code_committed_clean": True, "exec_code_sha256": L.exec_code_hashes(),
           "seal_sha256": L.SEAL_SHA, "verify_preregistration": ver, "lerobot_src_head": L.PT.LEROBOT_HEAD,
           "lerobot_imported_from": lerobot.__file__, "frame_cache": rep | {"train": len(rep["train"])},
           "run_order": L.CFG["run_order"], "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "synthetic_data": "none", "image_augmentation": "none"}
    L.OUT.mkdir(parents=True, exist_ok=True)
    (L.OUT / "prelaunch.json").write_text(json.dumps(pre, indent=1))
    return pre


def main():
    pre = prelaunch()
    runs = []
    t0 = time.time()
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None); env.pop("LD_LIBRARY_PATH", None)
    (L.OUT / "logs").mkdir(parents=True, exist_ok=True)
    for arm, seed in L.CFG["run_order"]:
        name = f"{arm}_s{seed}"
        log = L.OUT / "logs" / f"train_{name}.log"
        if (L.TRAIN / name / "terminal.json").exists():
            runs.append({"run": name, "skipped": "terminal.json exists"}); continue
        t = time.time()
        rc = subprocess.run([PY, "-B", "train_l1.py", "--arm", arm, "--seed", str(seed)], cwd=L.EXEC, env=env,
                            stdout=open(log, "w"), stderr=subprocess.STDOUT).returncode
        txt = log.read_text(errors="ignore")
        runs.append({"run": name, "rc": rc, "wall_s": time.time() - t,
                     "logged_losses": [float(x) for x in re.findall(r"loss ([0-9.]+) gn", txt)]})
        print(runs[-1]["run"], rc, round(runs[-1]["wall_s"]), flush=True)
        if rc != 0:
            (L.OUT / "train_runs.json").write_text(json.dumps({"prelaunch_git_head": pre["git_head"], "runs": runs, "failed": name}, indent=1))
            sys.exit(f"run {name} failed; study INCOMPLETE")
    (L.OUT / "train_runs.json").write_text(json.dumps({"prelaunch_git_head": pre["git_head"], "runs": runs,
                                                       "wall_s": time.time() - t0, "terminal_git_head": git("rev-parse", "HEAD")}, indent=1))
    rc = subprocess.run([PY, "-B", "eval_l1.py"], cwd=L.EXEC, env=env, stdout=open(L.OUT / "logs/eval.log", "w"), stderr=subprocess.STDOUT).returncode
    assert rc == 0, "evaluation failed"
    subprocess.run([PY, "-B", "score_l1.py"], cwd=L.EXEC, env=env, check=True)


if __name__ == "__main__":
    main()
