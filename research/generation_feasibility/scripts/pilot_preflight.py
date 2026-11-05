"""Pilot preflight: checks every prerequisite of the simulated pilot and REFUSES to proceed without a recorded review
approval. It never launches Isaac Sim, never touches a GPU, never opens a network connection.

    env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python research/generation_feasibility/scripts/pilot_preflight.py

Exit codes: 0 all prerequisites met AND approved (the runner may then be started by a person); 2 not approved;
3 approved but a prerequisite is missing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from gf import config as C  # noqa: E402

APPROVAL = HERE / "outputs/PILOT_APPROVED.json"     # written by a reviewer, never by this code
MANIFEST = HERE / "outputs/dry_run/pilot_manifest.json"
FROZEN = [HERE / "gf/config.py", HERE / "gf/plans.py", HERE / "gf/checks.py", HERE / "gf/sim_metrics.py",
          HERE / "gf/descriptors.py"]
# the server-recorded sources with per-step simulator truth for the 10 sim_10 demos (not on this machine)
SERVER_SOURCE_HDF5 = C.ISAACLAB / "logs/demos/openarm_pringles_lerobot_sim_10_v00_annotated.hdf5"


def spec_hash() -> str:
    h = hashlib.sha256()
    for p in FROZEN:
        h.update(p.read_bytes())
    return h.hexdigest()


def git(*a, cwd=C.REPO):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()


def prerequisites() -> list[dict]:
    out = []
    add = lambda name, ok, detail: out.append({"check": name, "ok": bool(ok), "detail": detail})  # noqa: E731
    add("dry-run manifest present", MANIFEST.exists(), str(MANIFEST))
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        add("manifest version matches config", m["version"] == C.VERSION, m["version"])
        n = {c: sum(1 for x in m["candidates"] if x["stage"] == "pilot" and x["cond"] == c) for c in "ABC"}
        add("24 pilot candidates per condition", all(v == C.PILOT_N for v in n.values()), n)
    add("IsaacLab at the audited commit", git("rev-parse", "--short", "HEAD", cwd=C.ISAACLAB) == C.ISAACLAB_COMMIT_AUDITED,
        git("rev-parse", "--short", "HEAD", cwd=C.ISAACLAB))
    add("IsaacLab tree clean", git("status", "--porcelain", cwd=C.ISAACLAB) == "", "uncommitted changes would change the env")
    add("sim_10 LeRobot sources cached", (C.SIM10_ROOT / "meta/info.json").exists(), str(C.SIM10_ROOT))
    add("server source HDF5 with per-step can pose (needed for P0 source replays and GT attachments)",
        SERVER_SOURCE_HDF5.exists(), str(SERVER_SOURCE_HDF5))
    add("pilot runner implemented (scripts/run_pilot_isaac.py)", (HERE / "scripts/run_pilot_isaac.py").exists(),
        "specified in GENERATION_FEASIBILITY_PLAN.md §5; deliberately NOT written before review")
    add("Isaac Lab python env", (Path.home() / "miniforge3/envs/env_isaaclab/bin/python").exists(), "env_isaaclab")
    add("GPU driver present (not queried)", shutil.which("nvidia-smi") is not None, "presence only")
    free = shutil.disk_usage(HERE).free / 1e9
    add("disk >= 20 GB free (no cameras: ~1-2 MB per rollout)", free >= 20, f"{free:.0f} GB")
    return out


def main():
    pre = prerequisites()
    approved = APPROVAL.exists()
    report = {"version": C.VERSION, "spec_sha256": spec_hash(), "approved": approved, "prerequisites": pre}
    if approved:
        a = json.loads(APPROVAL.read_text())
        report["approval"] = a
        report["approval_matches_spec"] = a.get("spec_sha256") == report["spec_sha256"]
    print(json.dumps(report, indent=1, default=str))
    if not approved or not report.get("approval_matches_spec", False):
        print("\nREFUSED: no review approval for this exact spec (outputs/PILOT_APPROVED.json with spec_sha256). "
              "No simulation is started by this script in any case.", file=sys.stderr)
        sys.exit(2)
    if not all(p["ok"] for p in pre):
        print("\nREFUSED: approved, but prerequisites are missing (see above).", file=sys.stderr)
        sys.exit(3)
    print("\nAll prerequisites met. Start the runner manually as described in GENERATION_FEASIBILITY_PLAN.md §5.")


if __name__ == "__main__":
    main()
