"""SD10-G1 geometry stage: 300 primary attempts (T000..T099, condition order rotated by k mod 3), 30 fixed-source-6
controls (T000,T011,...,T099 under each condition label, independent re-execution), 23 source identity diagnostics.

Each job runs in its own forked process with the registered 120 s geometry+checks cap and 4 CPU threads; a timeout or
exception is a failed attempt (Y_A=0), never a missing row. No retries, no fallback sources, no target shrink.
Arrays -> STORE/attempts/*.npz (outside Git, hashed); records -> outputs/geometry_attempts.jsonl.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "4"

import attempt_g1 as AT  # noqa: E402
import g1lib as G  # noqa: E402


def git(*a, cwd=G.REPO):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True).stdout.strip()


def job_list():
    tg = G.TGT["targets"]
    jobs = []
    for k, t in enumerate(tg):
        order = [G.CONDS[(i + k) % 3] for i in range(3)]
        jobs += [("primary", t, c, None) for c in order]
    ctl = G.TGT["fixed_source_sensitivity"]
    for tid in ctl["target_ids"]:
        t = next(x for x in tg if x["target_id"] == tid)
        jobs += [("control_source6", t, c, ctl["source_episode"]) for c in G.CONDS]
    jobs += [("identity", None, None, e) for e in G.UNION]
    return jobs


def identity(e):
    """source-only identity diagnostic: init = I at the source's own (reconstructed) standing-can pose."""
    rec = G.load_light(e)
    robot, geom = AT.robot_geom()
    t0 = time.time()
    spec = G.make_spec(rec, robot, np.eye(4))
    p = G.plan(rec, robot, spec)
    ex = G.export(rec, p)
    from sfa.robot import Camera
    uv, _ = Camera().project(rec.table[:3, 3][None])
    tgt = {"target_id": f"IDENTITY_{e}", "candidate_lid_xyz_robot_base_m": rec.table[:3, 3].tolist(),
           "requested_lid_uv_native_px": uv[0].tolist()}
    r = AT.GG.evaluate(rec, robot, geom, spec, p, ex, tgt)
    rows = np.arange(3, rec.T - 6)
    dev = float(np.abs(ex["action"].astype(float) - rec.action[rows]).max())
    return {"kind": "identity", "source_episode": e, "Y_A_diagnostic": r["Y_A"], "failed_gates": r["failed_gates"],
            "gates": r["gates"], "action_dev_vs_source": dev, "timing_s": {"total_s": time.time() - t0},
            "note": "init = I; source axis tilt makes gate 2 (axis to world-down) a diagnostic, not a pass criterion"}, None


def child(job, path, q):
    kind, t, c, src = job
    try:
        if kind == "identity":
            rec, arrays = identity(src)
        else:
            rec, arrays = AT.run(t, c, source=src, kind=kind)
        if arrays is not None:
            np.savez_compressed(path, **arrays)
            rec["arrays_file"] = path.name
            rec["arrays_sha256"] = G.sha(path)
        q.put(G.jsonable(rec))
    except Exception:  # noqa: BLE001
        import traceback
        q.put({"kind": kind, "status": "exception", "Y_A": False, "error": traceback.format_exc()[-3000:]})


def main():
    G.check_seals()
    rel = str(G.EXEC.relative_to(G.REPO))
    dirty = [l for l in git("status", "--porcelain", "--untracked-files=all", "--", rel).splitlines() if "/outputs/" not in l]
    assert not dirty, f"commit the execution code before the first attempt: {dirty}"
    pre = json.loads((G.OUT / "preprocess.json").read_text())
    assert pre["cold_cache"] and len(pre["sources"]) == 23
    import importlib.metadata as im
    reg = {"git_head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "exec_code_sha256":
           {p.name: G.sha(p) for p in sorted(G.EXEC.glob("*.py"))}, "seal_sha256": G.SEAL_SHA,
           "packages": {k: im.version(k) for k in ("numpy", "scipy", "pin", "opencv-python", "torch", "transformers", "av")},
           "python": sys.version, "cpu_threads": 4, "geometry_device": "CPU (pinocchio float64)",
           "preprocess_device": pre["sources"][0].get("gpu"), "sam_dtype": "bfloat16",
           "store": str(G.STORE), "time": time.time()}
    (G.OUT / "geometry_registration.json").write_text(json.dumps(reg, indent=1))
    robot, geom = AT.robot_geom()                       # loaded once before forking (shared read-only model)
    adir = G.STORE / "attempts"
    adir.mkdir(parents=True, exist_ok=True)
    out = open(G.OUT / "geometry_attempts.jsonl", "w")
    ctx = mp.get_context("fork")
    cap = G.CFG["compute"]["geometry_cap_s"]
    t_all = time.time()
    jobs = job_list()
    for n, job in enumerate(jobs):
        kind, t, c, src = job
        name = f"{kind}_{t['target_id'] if t else 'src'}_{c or 'none'}_{src if src is not None else 'locked'}"
        q = ctx.Queue()
        p = ctx.Process(target=child, args=(job, adir / f"{name}.npz", q))
        t0 = time.time()
        p.start()
        rec = None
        try:
            rec = q.get(timeout=cap)
        except Exception:  # noqa: BLE001
            rec = None
        p.join(5)
        if p.is_alive():
            p.kill(); p.join()
        wall = time.time() - t0
        if rec is None:
            rec = {"kind": kind, "target_id": t["target_id"] if t else None, "condition": c, "source_episode": src,
                   "status": "timeout", "Y_A": False, "failed_gates": ["timeout_120s"]}
        rec.update(job_name=name, job_index=n, wall_s=wall)
        out.write(json.dumps(rec) + "\n"); out.flush()
        if n % 30 == 0 or kind != "primary":
            print(n, name, rec.get("Y_A", rec.get("Y_A_diagnostic")), rec.get("failed_gates"), round(wall, 1), flush=True)
    out.close()
    (G.OUT / "geometry_run.json").write_text(json.dumps({"jobs": len(jobs), "wall_s": time.time() - t_all,
                                                         "terminal_git_head": git("rev-parse", "HEAD")}, indent=1))


if __name__ == "__main__":
    main()
