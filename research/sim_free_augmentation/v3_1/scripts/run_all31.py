"""Run v3 generation on all source episodes, two GPU lanes in parallel, with a 1 Hz nvidia-smi log and wall-clock.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/run_all.py [--tag _x] [--families ...] [--episodes ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def lane(dev, eps, args, logdir):
    out = []
    for e in eps:
        t = time.time()
        cmd = [sys.executable, str(HERE / "scripts/run_episode31.py"), "--episode", str(e), "--device", dev,
               "--workers", str(args.workers), "--tag", args.tag, "--config", args.config]
        if args.families:
            cmd += ["--families", *args.families]
        with open(logdir / f"ep{e}{args.tag}.log", "w") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=dict(__import__("os").environ, OMP_NUM_THREADS="1")).returncode
        out.append({"episode": e, "device": dev, "rc": rc, "wall_s": time.time() - t})
        print(f"ep{e} on {dev}: rc={rc} {time.time() - t:.0f} s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--config", default=str(HERE / "configs/v3_1.json"))
    args = ap.parse_args()
    logdir = HERE / "outputs/logs"
    logdir.mkdir(parents=True, exist_ok=True)
    bench = HERE / "outputs/bench"
    bench.mkdir(parents=True, exist_ok=True)
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                            "-l", "1"], stdout=open(bench / f"nvidia_smi_generate{args.tag}.csv", "w"))
    lanes = {"cuda:0": args.episodes[0::2], "cuda:1": args.episodes[1::2]}
    t0 = time.time()
    with ThreadPoolExecutor(2) as ex:
        futs = {d: ex.submit(lane, d, eps, args, logdir) for d, eps in lanes.items()}
        res = [r for d in lanes for r in futs[d].result()]
    smi.terminate()
    wall = time.time() - t0
    (bench / f"generate{args.tag}.json").write_text(json.dumps({"wall_s": wall, "lanes": lanes, "episodes": res,
                                                                "workers_per_lane": args.workers}, indent=1))
    print(f"done in {wall:.0f} s")


if __name__ == "__main__":
    main()
