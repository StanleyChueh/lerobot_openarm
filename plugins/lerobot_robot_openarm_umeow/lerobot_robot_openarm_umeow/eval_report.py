"""Compare evaluation runs side by side: their timing summaries in one table and one plot.

    python -m lerobot_robot_openarm_umeow.eval_report                 # the latest run of each mode and policy
                                                                      # (with and without rerun)
    python -m lerobot_robot_openarm_umeow.eval_report RUN_DIR ...     # these runs (folders in outputs/eval_logs/)
    python -m lerobot_robot_openarm_umeow.eval_report --all           # every run in outputs/eval_logs/
    python -m lerobot_robot_openarm_umeow.eval_report --root=DIR      # runs in another folder

Every run of lerobot-rollout (normal, rtc) and of our robot_client (async, async_rtc) writes a run folder
(eval_metrics.py). This prints their summaries as columns, saves compare_<date>.png next to them and opens it.
"""

import json
import sys
import time
from pathlib import Path

from .eval_metrics import _COLORS, METRICS, POLICY_NAMES, _figure, format_table, load_run, open_file, summarize

_ORDER = {"normal": 0, "rtc": 1, "async": 2, "async_rtc": 3}


def _rerun(meta: dict) -> bool:
    """Whether the run showed the rerun window (on unless --display_data=false)."""
    if "rerun" in meta:
        return bool(meta["rerun"])
    flags = [a.split("=", 1)[1].lower() for a in meta.get("argv", []) if a.startswith("--display_data=")]
    return not flags or flags[-1] not in ("false", "0", "no")


def _runs(args: list[str]) -> tuple[Path, list[Path]]:
    root = Path(next((a.split("=", 1)[1] for a in args if a.startswith("--root=")), "outputs/eval_logs"))
    dirs = [Path(a) for a in args if not a.startswith("--")]
    if dirs:
        return root, dirs
    found = sorted(p.parent for p in root.glob("*/summary.json"))
    if "--all" in args:
        return root, found
    latest = {}
    for d in found:  # folder names start with the date: the last one of each mode and policy wins
        meta = json.loads((d / "summary.json").read_text())
        latest[(meta["policy"], meta["mode"], _rerun(meta))] = d
    return root, list(latest.values())


def _short_names(metas: list[dict]) -> list[str]:
    """SmolVLA / GR00T N1.7 / pi0.5 when known, else the model names without their common prefix."""
    import os

    names = [m["policy"] for m in metas]
    prefix = os.path.commonprefix(names) if len(set(names)) > 1 else ""
    return [POLICY_NAMES.get(m.get("policy_type") or "", "") or (n[len(prefix):] or n) for m, n in zip(metas, names)]


def main() -> None:
    import numpy as np

    root, dirs = _runs(sys.argv[1:])
    if not dirs:
        raise SystemExit(f"[openarm eval] no runs in {root}/ (each evaluation writes one there).")
    runs = []
    for d in dirs:
        meta, ticks, chunks = load_run(d)
        meta["metrics"] = summarize(ticks, chunks, meta["fps"])  # recomputed: the same definitions for every run
        label = f"{meta['mode']}{'' if _rerun(meta) else ' (no rerun)'} | {meta['policy']}"
        if any(r[0] == label for r in runs):
            label += f" | {d.name[:15]}"
        runs.append((label, meta, ticks, chunks, d))
    runs.sort(key=lambda r: (r[1]["policy"], _ORDER.get(r[1]["mode"], 9), not _rerun(r[1]), r[4].name))
    print(format_table([(label, meta["metrics"]) for label, meta, *_ in runs]))
    print("\n  runs: " + "\n        ".join(str(r[4]) for r in runs))

    keys = ["cmd_hz", "idle_pct", "latency_mean_ms", "gap_p95_ms", "dropped_mean", "jump_switch_mean"]
    labels = {k: label for k, label, _ in METRICS}
    fig = _figure(18 + max(0, len(runs) - 8) * 0.7, 11 + (1.5 if len(runs) > 8 else 0))
    ax = fig.subplots(2, 4)
    short = _short_names([meta for _, meta, *_ in runs])
    names = [f"{p}\n{meta['mode']}" + ("" if _rerun(meta) else "\nno rerun") for p, (_, meta, *_) in zip(short, runs)]
    colors = [_COLORS.get(meta["mode"], "0.5") for _, meta, *_ in runs]
    hatches = ["" if _rerun(meta) else "//" for _, meta, *_ in runs]
    for p, key in zip(ax.flat, keys):
        values = [meta["metrics"].get(key) for _, meta, *_ in runs]
        values = [np.nan if v is None else v for v in values]
        p.bar(range(len(runs)), values, color=colors, hatch=hatches, edgecolor="white")
        p.set_title(labels[key], fontsize=10)
        if len(runs) > 8:  # one line per run, turned
            p.set_xticks(range(len(runs)), [n.replace("\n", " ") for n in names], rotation=90, fontsize=7)
        else:
            p.set_xticks(range(len(runs)), names, fontsize=7)
    # gap between commands, as a distribution
    p = ax.flat[6]
    for (_, meta, ticks, _, _), color, label in zip(runs, colors, names):
        label = label.replace("\n", " ")
        gaps = []
        for ep in sorted({x[1] for x in ticks}):
            ts = np.array([x[0] for x in ticks if x[1] == ep and x[2]])
            gaps += list(np.diff(ts) * 1000)
        if gaps:
            g = np.sort(gaps)
            p.plot(g, np.arange(1, len(g) + 1) / len(g), color=color, label=label, lw=1,
                   ls="-" if "no rerun" not in label else "--")
    p.set_xscale("log")
    p.set_title("gap between commands (cumulative share)", fontsize=10)
    p.set_xlabel("ms")
    # latency per chunk, as a distribution
    p = ax.flat[7]
    for (_, meta, _, chunks, _), color, label in zip(runs, colors, names):
        label = label.replace("\n", " ")
        lat = np.sort([c["latency_s"] * 1000 for c in chunks if c.get("latency_s") is not None])
        if len(lat):
            p.plot(lat, np.arange(1, len(lat) + 1) / len(lat), color=color, label=label, lw=1,
                   ls="-" if "no rerun" not in label else "--")
    p.set_xscale("log")
    p.set_title("observation -> actions ready (cumulative share)", fontsize=10)
    p.set_xlabel("ms")
    p.legend(fontsize=6, loc="lower right")
    fig.suptitle("evaluation timing -- normal (blue), rtc (red), async (green), async_rtc (purple);"
                 " hatched / dashed: without rerun")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    root.mkdir(parents=True, exist_ok=True)
    png = root / f"compare_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(png)
    print(f"\n[openarm eval] comparison plot: {png.resolve()}")
    open_file(png)


if __name__ == "__main__":
    main()
