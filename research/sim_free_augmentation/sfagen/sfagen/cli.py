"""sfagen command line. Every stage records wall time and peak GPU memory to <work>/profile.jsonl.

  python -m sfagen calibrate  CONFIG [--search]          fit the body camera (once per rig)
  python -m sfagen table      CONFIG                     table height of the rig from all source episodes (once per rig)
  python -m sfagen generate   CONFIG [--episodes 0,1]    reconstruct + plan + replay-check all variations
  python -m sfagen render     CONFIG [--episodes 0,1]    render every passing candidate (body camera)
  python -m sfagen build      CONFIG --repo-id ID [--mixed-repo-id ID2]   export LeRobot v3.0 dataset(s)
  python -m sfagen verify     CONFIG                     check the built datasets
  python -m sfagen video      CONFIG [--episodes 0,5] [--speed 2]   real vs synthetic side-by-side video
  python -m sfagen bench      CONFIG --episodes 3,5 [--compare camera|render]   real-ground-truth renderer benchmark
  python -m sfagen run        CONFIG --repo-id ID [...]  generate + render + build
"""
from __future__ import annotations
import argparse, json, os, subprocess, threading, time
import torch
from .context import Context


class Profiler:
    def __init__(self, path):
        self.path, self.peak, self.pid = path, 0, str(os.getpid())
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
                for l in o.splitlines():
                    p, m = [x.strip() for x in l.split(",")]
                    if p == self.pid:
                        self.peak = max(self.peak, int(m))
            except Exception:
                pass
            time.sleep(0.25)

    def stage(self, name, fn, **extra):
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        self.peak = 0; t0 = time.time(); out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time.sleep(0.3)
        r = {"stage": name, "seconds": round(time.time() - t0, 2), "gpu_peak_torch_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3) if torch.cuda.is_available() else 0,
             "gpu_peak_process_gib": round(self.peak / 1024, 3), "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
        with open(self.path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print("[profile]", r, flush=True)
        return out


def estimate_table(ctx, device):
    """table height of the rig from all source episodes: per episode, the free (x, y, z) silhouette fit of every known-size
    object; if the median IoU >= 0.5, z = median of the fits with IoU >= 0.5, else the lowest arm point - 5 mm.
    Writes table_height.json; reconstructions made afterwards use this fixed height."""
    import numpy as np, shutil
    from .recon import Recon
    p = ctx.work / "table_height.json"
    if p.exists():
        p.unlink()
    keep = ctx.cache_dir; ctx.cache_dir = ctx.work / "cache_table"; ctx.cache_dir.mkdir(exist_ok=True); fits = []
    for e in ctx.episodes():
        r = Recon(ctx, e, device)
        for name, o in r.info.get("objects", {}).items():
            if "silhouette_iou" in o:
                fits.append({"episode": e, "object": name, "z": o["table_z"], "iou": o["silhouette_iou"]})
        print("table fit", e, [f for f in fits if f["episode"] == e], flush=True)
    ctx.cache_dir = keep; shutil.rmtree(ctx.work / "cache_table", ignore_errors=True)
    good = [f for f in fits if f["iou"] >= 0.5]
    if fits and np.median([f["iou"] for f in fits]) >= 0.5:
        z, rule = float(np.median([f["z"] for f in good])), "median of known-size object silhouette fits (median IoU >= 0.5)"
    else:
        z, rule = ctx.lowest_arm_z - 0.005, "lowest arm point over the source demos - 5 mm (object silhouette fits not confident)"
    out = {"z": z, "rule": rule, "fits": fits, "lowest_arm_z": ctx.lowest_arm_z}
    p.write_text(json.dumps(out, indent=1)); ctx._table = None; print("table", z, rule, flush=True)
    for d in ("cache", "candidates"):
        shutil.rmtree(ctx.work / d, ignore_errors=True)
    ctx.cache_dir.mkdir(exist_ok=True)
    return out


def main():
    ap = argparse.ArgumentParser(prog="sfagen"); ap.add_argument("cmd", choices=["calibrate", "table", "generate", "render", "build", "verify", "video", "bench", "run"])
    ap.add_argument("config"); ap.add_argument("--episodes"); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--work-dir")
    ap.add_argument("--search", action="store_true"); ap.add_argument("--repo-id"); ap.add_argument("--mixed-repo-id")
    ap.add_argument("--compare", default="camera"); ap.add_argument("--speed", type=int, default=1); ap.add_argument("--keys")
    a = ap.parse_args(); ctx = Context(a.config, a.work_dir); prof = Profiler(ctx.work / "profile.jsonl")
    eps = [int(x) for x in a.episodes.split(",")] if a.episodes else ctx.episodes()
    if a.cmd == "calibrate":
        from . import calibrate
        prof.stage("calibrate", lambda: calibrate.run(ctx, a.device, a.search)); return
    if a.cmd == "table":
        prof.stage("table height (all source episodes)", lambda: estimate_table(ctx, a.device)); return
    if a.cmd in ("generate", "run"):
        from .checks import Checker
        from .recon import Recon
        from . import generate
        ck = prof.stage("replay-check envelope (all source episodes)", lambda: (lambda c: (c.envelope(ctx.episodes()), c)[1])(Checker(ctx)))
        for e in eps:
            rec = prof.stage(f"reconstruct ep{e}", lambda: Recon(ctx, e, a.device), episode=e)
            res = prof.stage(f"plan+check ep{e}", lambda: generate.run_episode(ctx, rec, ck), episode=e)
            print("episode", e, "passing", sum(v["pass"] for v in res.values()), "/", len(res), flush=True)
    if a.cmd in ("render", "run"):
        from .recon import Recon
        from . import render_stage
        for e in eps:
            if not (ctx.work / f"candidates/ep{e}/checks.json").exists():
                continue
            rec = Recon(ctx, e, a.device)
            prof.stage(f"render ep{e}", lambda: render_stage.run_episode(ctx, rec, a.device, keys=a.keys.split(",") if a.keys else None), episode=e)
            prof.stage(f"render wrist ep{e}", lambda: render_stage.run_wrist_episode(ctx, rec, a.device, keys=a.keys.split(",") if a.keys else None), episode=e)
    if a.cmd in ("build", "run"):
        from . import build
        assert a.repo_id, "--repo-id required"
        prof.stage("build", lambda: build.run(ctx, a.repo_id, a.device, a.mixed_repo_id))
    if a.cmd in ("verify", "run"):
        from . import verify
        prof.stage("verify", lambda: verify.run(ctx))
    if a.cmd == "video":
        from . import compare_video
        out = ctx.work / f"compare_{'all' if not a.episodes else 'eps_' + a.episodes.replace(',', '-')}_{a.speed}x.mp4"
        prof.stage("video", lambda: compare_video.run(ctx, out, [int(x) for x in a.episodes.split(",")] if a.episodes else None, a.speed)); return
    if a.cmd == "bench":
        from . import bench
        if a.compare == "camera":
            variants = {"initial_camera": {"camera": "initial"}, "fitted_camera": {"camera": "fitted"}}
        else:
            variants = json.loads(a.compare)
        rows, summ = prof.stage("bench", lambda: bench.run(ctx, eps, variants, a.device))
        tag = "camera" if a.compare == "camera" else "render"
        (ctx.work / f"bench_{tag}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=float)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
