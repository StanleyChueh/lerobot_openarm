"""Timing diagnostics for every evaluation mode, in one format, to compare them.

  normal     lerobot-rollout --strategy.type=episodic                 (patched here by install())
  rtc        lerobot-rollout ... --inference.type=rtc                 (patched here by install())
  async      python -m lerobot_robot_openarm_umeow.robot_client ...   (client_class(), used by robot_client.py)
  async_rtc  the same, with the policy server run with --rtc=true     (the server tags its chunks)

Each run logs every control tick (the action sent, or none; the measured joints; the actions queued) and
every action chunk (when it became available, how long after the observation it was computed from, how
many of its actions were already too late to execute). At the end of each episode one line is printed;
at the end of the run a summary table, and a plot (report.png) is saved and opened. The files are in
outputs/eval_logs/<date>_<mode>_<policy>/ (OPENARM_EVAL_LOG_DIR changes the folder, OPENARM_EVAL_LOG=0
turns it off, OPENARM_EVAL_OPEN=0 only saves the plot). Compare runs with
    python -m lerobot_robot_openarm_umeow.eval_report            (the latest run of each mode and policy)
With the rerun window on, the same numbers are plotted live in its "eval timing" view.

What differs between the modes, and where it shows:
  normal     the robot waits while the policy computes: long gaps between commands at every chunk.
  rtc        a background thread computes the next chunk while the robot executes: no gaps, but the
             actions for the steps that passed during inference are skipped. A chunk that takes longer
             than its own length is skipped whole -- the robot then gets nothing ("ticks with no action").
  async      a separate server computes; chunks overlap and are aggregated in the client's queue.
  async_rtc  the same, with each chunk guided to continue the one being executed (smaller chunk switches).
"""

import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .common import MOTOR_KEYS

ARM_KEYS = [k for k in MOTOR_KEYS if not k.endswith("J8.pos")]
LIVE = "openarm/eval"  # rerun entity of the live plots
LOG = None  # the current run's EvalLog (lerobot-rollout); robot_client.py keeps its own
MODE_NAMES = {"normal": "normal", "rtc": "RTC", "async": "async", "async_rtc": "async + RTC"}
POLICY_NAMES = {"smolvla": "SmolVLA", "groot": "GR00T N1.7", "pi05": "pi0.5", "pi0": "pi0", "act": "ACT"}

# (key, label, format) of the summary, in display order
METRICS = [
    ("episodes", "episodes", "{:.0f}"),
    ("duration_s", "policy time (s)", "{:.0f}"),
    ("loop_hz", "control loop (Hz)", "{:.1f}"),
    ("cmd_hz", "new commands per second", "{:.1f}"),
    ("starved_pct", "ticks with no action (%)", "{:.1f}"),
    ("idle_pct", "time waiting for actions (%)", "{:.1f}"),
    ("gap_p50_ms", "gap between commands p50 (ms)", "{:.0f}"),
    ("gap_p95_ms", "gap between commands p95 (ms)", "{:.0f}"),
    ("gap_max_ms", "gap between commands max (ms)", "{:.0f}"),
    ("chunks", "chunks received", "{:.0f}"),
    ("latency_mean_ms", "observation -> actions ready, mean (ms)", "{:.0f}"),
    ("latency_p95_ms", "observation -> actions ready, p95 (ms)", "{:.0f}"),
    ("compute_mean_ms", "policy compute, mean (ms)", "{:.0f}"),
    ("delay_mean_steps", "control steps passed meanwhile", "{:.1f}"),
    ("dropped_mean", "too-late actions skipped per chunk", "{:.1f}"),
    ("dropped_whole", "chunks skipped whole (too late)", "{:.0f}"),
    ("guided_mean", "RTC: actions each chunk continues", "{:.1f}"),
    ("jump_switch_mean", "chunk switch: arm step, mean (rad)", "{:.4f}"),
    ("jump_switch_p95", "chunk switch: arm step, p95 (rad)", "{:.4f}"),
    ("step_within_median", "within a chunk: arm step, median (rad)", "{:.4f}"),
    ("jerk_mean", "arm |second difference|, mean (rad)", "{:.5f}"),
    ("track_mean", "command - measured, mean (rad)", "{:.3f}"),
]


def _enabled() -> bool:
    return os.environ.get("OPENARM_EVAL_LOG", "1").lower() not in ("0", "false", "no")


def _argv_value(flag: str) -> str | None:
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _vector(values) -> list[float] | None:
    if not values:
        return None
    try:
        return [float(values[k]) for k in MOTOR_KEYS]
    except (KeyError, TypeError, ValueError):
        return None


def _rerun_active() -> bool:
    try:
        import rerun as rr

        return rr.get_global_data_recording() is not None
    except Exception:
        return False


class EvalLog:
    """Ticks and chunks of one evaluation run. tick() from the control loop, chunk() from any thread."""

    def __init__(self, mode: str, model: str, fps: float, policy_type: str | None = None):
        """model: the checkpoint (Hub repo or folder); policy_type: smolvla, groot, pi05, ..."""
        self.mode, self.model, self.policy_type, self.fps = mode, model, policy_type, float(fps)
        self.policy = model.rstrip("/").rsplit("/", 1)[-1] or "policy"  # in the run folder's name
        self.ticks: list[tuple] = []  # (t, episode, sent, queue, new_chunk, action[16] | None, state[16] | None)
        self.chunks: list[dict] = []
        self.episode = 0
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._episode_t0 = None
        self._new_chunk = False  # a chunk arrived since the last action sent
        self._pending_live: list[dict] = []
        self._live = None
        self._finished = False
        from . import rerun_status

        rerun_status.EVAL_VIEW = True  # the rerun layout gets the live timing views
        self._show_run()

    def _show_run(self) -> None:
        """Policy, model and mode on the rerun status panel."""
        from .rerun_status import BOARD

        kind = POLICY_NAMES.get(self.policy_type or "", self.policy_type or "?")
        BOARD.run_info = (f"**policy:** {kind} · **model:** `{self.model}` · "
                          f"**mode:** {MODE_NAMES.get(self.mode, self.mode)}")

    # -- recording ---------------------------------------------------------------------------------------
    def now(self) -> float:
        return time.perf_counter() - self._t0

    def begin_episode(self) -> None:
        with self._lock:
            self.episode += 1
            self._episode_t0 = self.now()
            self._new_chunk = False

    def end_episode(self) -> None:
        with self._lock:
            episode = self.episode
            ticks = [x for x in self.ticks if x[1] == episode]
            chunks = [c for c in self.chunks if c["episode"] == episode]
        if not ticks:
            return
        m = summarize(ticks, chunks, self.fps)
        line = (f"[openarm eval] episode {episode} ({self.mode}): {m['cmd_hz']:.1f} new commands/s"
                f" (loop {m['loop_hz']:.1f} Hz), waiting for actions {m['idle_pct']:.0f}% of the time,"
                f" max gap {m['gap_max_ms']:.0f} ms")
        if m["chunks"]:
            line += (f"; {m['chunks']:.0f} chunks, observation -> actions {m['latency_mean_ms']:.0f} ms"
                     f" (p95 {m['latency_p95_ms']:.0f}), {m['dropped_mean']:.1f} too-late actions skipped per chunk"
                     f"; chunk switch {m['jump_switch_mean']:.3f} rad vs {m['step_within_median']:.3f} within")
        print(line + ".", flush=True)
        if m.get("dropped_whole"):
            print(f"[openarm eval] WARNING: {m['dropped_whole']:.0f} chunk(s) arrived after all their actions were"
                  " due and were skipped whole: inference is slower than one chunk here.", flush=True)

    def tick(self, action: dict | None, state: dict | None, queue: int | None) -> None:
        a, s = _vector(action), _vector(state)
        with self._lock:
            if self._episode_t0 is None:
                self.episode, self._episode_t0 = 1, self.now()
            new = a is not None and self._new_chunk
            if a is not None:
                self._new_chunk = False
            t = self.now()
            last = self.ticks[-1][0] if self.ticks and self.ticks[-1][1] == self.episode else None
            self.ticks.append((t, self.episode, a is not None, queue, new, a, s))
            pending, self._pending_live = self._pending_live, []
        self._log_live(t, last, a is not None, queue, pending)

    def chunk(self, *, latency_s: float | None, n_actions: int, dropped: int = 0, compute_s: float | None = None,
              delay_steps: float | None = None, guided: int | None = None, mode: str | None = None) -> None:
        with self._lock:
            changed = bool(mode) and mode != self.mode
            if changed:
                self.mode = mode
            if delay_steps is None and latency_s is not None:
                delay_steps = latency_s * self.fps
            c = {"t": self.now(), "episode": self.episode, "latency_s": latency_s, "compute_s": compute_s,
                 "n_actions": int(n_actions), "dropped": int(dropped), "delay_steps": delay_steps, "guided": guided}
            self.chunks.append(c)
            self._new_chunk = True
            self._pending_live.append(c)
        if changed:
            self._show_run()

    # -- live view (rerun, control-loop thread only) -----------------------------------------------------
    def _log_live(self, t, last, sent, queue, chunks) -> None:
        if self._live is None:
            self._live = _rerun_active()
        if not self._live:
            return
        try:
            import rerun as rr

            if last is not None:
                rr.log(f"{LIVE}/loop_period_ms", rr.Scalars((t - last) * 1000))
            rr.log(f"{LIVE}/no_action", rr.Scalars(0.0 if sent else 1.0))
            if queue is not None:
                rr.log(f"{LIVE}/queued_actions", rr.Scalars(float(queue)))
            for c in chunks:
                if c["latency_s"] is not None:
                    rr.log(f"{LIVE}/latency_ms", rr.Scalars(c["latency_s"] * 1000))
                rr.log(f"{LIVE}/skipped_actions", rr.Scalars(float(c["dropped"])))
        except Exception:
            self._live = False  # display only: never disturb the control loop

    # -- end of run --------------------------------------------------------------------------------------
    def finish(self) -> Path | None:
        if self._finished:
            return None
        self._finished = True
        with self._lock:
            ticks, chunks = list(self.ticks), list(self.chunks)
        if not ticks:
            return None
        try:
            root = Path(os.environ.get("OPENARM_EVAL_LOG_DIR", "outputs/eval_logs"))
            name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in self.policy)
            out = root / f"{time.strftime('%Y%m%d_%H%M%S')}_{self.mode}_{name}"
            out.mkdir(parents=True, exist_ok=True)
            _write(out, self, ticks, chunks)
            summary = summarize(ticks, chunks, self.fps)
            meta = {"mode": self.mode, "policy": self.policy, "model": self.model, "policy_type": self.policy_type,
                    "fps": self.fps, "rerun": bool(self._live), "argv": sys.argv}
            (out / "summary.json").write_text(json.dumps({**meta, "metrics": summary}, indent=1))
            print(format_table([(f"{self.mode} | {self.policy}", summary)]), flush=True)
            png = out / "report.png"
            plot_run(png, self.mode, self.policy, self.fps, ticks, chunks, summary)
            print(f"[openarm eval] timing log and plot: {out.resolve()}/ (report.png). Compare runs with:"
                  " python -m lerobot_robot_openarm_umeow.eval_report", flush=True)
            open_file(png)
            return out
        except Exception as e:  # diagnostics only: never fail the evaluation for them
            print(f"[openarm eval] could not write the timing report: {e!r}", flush=True)
            return None


# -- files -----------------------------------------------------------------------------------------------
def _write(out: Path, log: EvalLog, ticks, chunks) -> None:
    with open(out / "ticks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "episode", "sent", "queue", "new_chunk"] + [f"a_{k}" for k in MOTOR_KEYS]
                   + [f"s_{k}" for k in MOTOR_KEYS])
        blank = [""] * len(MOTOR_KEYS)
        for t, ep, sent, queue, new, a, s in ticks:
            w.writerow([f"{t:.4f}", ep, int(sent), "" if queue is None else queue, int(new)]
                       + ([f"{x:.5f}" for x in a] if a else blank) + ([f"{x:.5f}" for x in s] if s else blank))
    with open(out / "chunks.csv", "w", newline="") as f:
        keys = ["t", "episode", "latency_s", "compute_s", "n_actions", "dropped", "delay_steps", "guided"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for c in chunks:
            w.writerow({k: ("" if c.get(k) is None else c[k]) for k in keys})


def load_run(path: Path) -> tuple[dict, list, list]:
    """(summary.json, ticks, chunks) of a run folder, in the in-memory format."""
    meta = json.loads((path / "summary.json").read_text())

    def num(x):
        return None if x == "" else float(x)

    ticks = []
    with open(path / "ticks.csv") as f:
        for row in csv.DictReader(f):
            a = [num(row[f"a_{k}"]) for k in MOTOR_KEYS]
            s = [num(row[f"s_{k}"]) for k in MOTOR_KEYS]
            ticks.append((float(row["t"]), int(row["episode"]), row["sent"] == "1", num(row["queue"]),
                          row["new_chunk"] == "1", None if a[0] is None else a, None if s[0] is None else s))
    chunks = []
    with open(path / "chunks.csv") as f:
        for row in csv.DictReader(f):
            c = {k: num(v) for k, v in row.items()}
            c["episode"] = int(c["episode"])
            chunks.append(c)
    return meta, ticks, chunks


# -- metrics ---------------------------------------------------------------------------------------------
def _pct(values, q):
    import numpy as np

    return float(np.percentile(values, q)) if len(values) else float("nan")


def _mean(values):
    return float(sum(values) / len(values)) if len(values) else float("nan")


def summarize(ticks, chunks, fps: float) -> dict:
    import numpy as np

    arm = [MOTOR_KEYS.index(k) for k in ARM_KEYS]
    period = 1.0 / fps
    episodes = sorted({x[1] for x in ticks})
    duration, gaps, jumps_switch, steps_within, jerks, track = 0.0, [], [], [], [], []
    n_ticks = n_sent = 0
    for ep in episodes:
        rows = [x for x in ticks if x[1] == ep]
        t = np.array([x[0] for x in rows])
        duration += float(t[-1] - t[0]) + period
        n_ticks += len(rows)
        sent = [x for x in rows if x[2]]
        n_sent += len(sent)
        if len(sent) < 2:
            continue
        ts = np.array([x[0] for x in sent])
        gaps += list(np.diff(ts) * 1000)
        a = np.array([x[5] for x in sent])[:, arm]
        step = np.abs(np.diff(a, axis=0)).max(axis=1)
        switch = np.array([x[4] for x in sent][1:])
        jumps_switch += list(step[switch])
        steps_within += list(step[~switch])
        if len(a) >= 3:
            jerks += list(np.abs(np.diff(a, n=2, axis=0)).mean(axis=1))
        track += [float(np.abs(np.array(x[5])[arm] - np.array(x[6])[arm]).max()) for x in sent if x[6]]
    gaps_a = np.array(gaps)
    lat = [c["latency_s"] * 1000 for c in chunks if c.get("latency_s") is not None]
    comp = [c["compute_s"] * 1000 for c in chunks if c.get("compute_s") is not None]
    delay = [c["delay_steps"] for c in chunks if c.get("delay_steps") is not None]
    guided = [c["guided"] for c in chunks if c.get("guided") is not None]
    return {
        "episodes": len(episodes),
        "duration_s": duration,
        "loop_hz": n_ticks / duration if duration else float("nan"),
        "cmd_hz": n_sent / duration if duration else float("nan"),
        "starved_pct": 100.0 * (n_ticks - n_sent) / n_ticks if n_ticks else float("nan"),
        # time the arm waited for actions: every gap beyond the loop's usual one (one control period, or more
        # when the loop itself is slower, e.g. in simulation) -- an inference stall, or ticks with nothing to send
        "idle_pct": (100.0 * float(np.clip(gaps_a - max(period * 1000, _pct(gaps_a, 50)), 0, None).sum()) / 1000
                     / duration) if duration and len(gaps_a) else float("nan"),
        "gap_p50_ms": _pct(gaps_a, 50),
        "gap_p95_ms": _pct(gaps_a, 95),
        "gap_max_ms": float(gaps_a.max()) if len(gaps_a) else float("nan"),
        "chunks": len(chunks),
        "latency_mean_ms": _mean(lat),
        "latency_p95_ms": _pct(lat, 95),
        "compute_mean_ms": _mean(comp),
        "delay_mean_steps": _mean(delay),
        "dropped_mean": _mean([c["dropped"] for c in chunks]),
        "dropped_whole": sum(1 for c in chunks if c["n_actions"] and c["dropped"] >= c["n_actions"]),
        "guided_mean": _mean(guided),
        "jump_switch_mean": _mean(jumps_switch),
        "jump_switch_p95": _pct(jumps_switch, 95),
        "step_within_median": _pct(steps_within, 50),
        "jerk_mean": _mean(jerks),
        "track_mean": _mean(track),
    }


def format_table(runs: list[tuple[str, dict]]) -> str:
    """runs: (\"mode | policy\", summary) each; one column per run."""
    width = max(len(label) for _, label, _ in METRICS) + 2
    heads = [(name.split(" | ") + [""])[:2] for name, _ in runs]
    cols = [max(10, *(len(h) for h in head)) for head in heads]
    lines = ["", "[openarm eval] timing summary".ljust(width)
             + "".join(h[0].rjust(c + 2) for h, c in zip(heads, cols)),
             "".ljust(width) + "".join(h[1].rjust(c + 2) for h, c in zip(heads, cols))]
    for key, label, fmt in METRICS:
        cells = []
        for (_, m), c in zip(runs, cols):
            v = m.get(key)
            cells.append(("-" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)).rjust(c + 2))
        lines.append("  " + label.ljust(width - 2) + "".join(cells))
    return "\n".join(lines)


# -- plots -----------------------------------------------------------------------------------------------
_COLORS = {"normal": "#1f77b4", "rtc": "#d62728", "async": "#2ca02c", "async_rtc": "#9467bd"}


def _figure(w, h):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(w, h), dpi=100)
    FigureCanvasAgg(fig)
    return fig


def plot_run(png: Path, mode, policy, fps, ticks, chunks, summary) -> None:
    import numpy as np

    episodes = sorted({x[1] for x in ticks})
    ep = max(episodes, key=lambda e: sum(1 for x in ticks if x[1] == e and x[2]))  # the longest episode
    rows = [x for x in ticks if x[1] == ep]
    t0 = rows[0][0]
    t = np.array([x[0] - t0 for x in rows])
    sent = np.array([x[2] for x in rows])
    cks = [c for c in chunks if c["episode"] == ep]
    ct = np.array([c["t"] - t0 for c in cks])
    fig = _figure(17, 12)
    fig.suptitle(f"{policy} -- {mode} -- episode {ep} of {len(episodes)} (the longest); all episodes in the table",
                 fontsize=13)
    ax = fig.subplots(3, 2)

    # 1. commanded vs measured, the two arm joints that moved most
    a = np.array([x[5] if x[5] else [np.nan] * len(MOTOR_KEYS) for x in rows])
    s = np.array([x[6] if x[6] else [np.nan] * len(MOTOR_KEYS) for x in rows])
    arm = [MOTOR_KEYS.index(k) for k in ARM_KEYS]
    spread = np.nan_to_num(np.nanmax(a[:, arm], axis=0) - np.nanmin(a[:, arm], axis=0)) if sent.any() else np.zeros(len(arm))
    p = ax[0, 0]
    for j, color in zip([arm[i] for i in np.argsort(spread)[-2:]], ["#1f77b4", "#ff7f0e"]):
        p.plot(t, a[:, j], "-", color=color, lw=1.2, label=f"{MOTOR_KEYS[j]} command")
        read = ~np.isnan(s[:, j])  # async: the client reads the arm only on ticks that send an observation
        p.plot(t[read], s[read, j], "--", color=color, lw=1, alpha=0.7, label=f"{MOTOR_KEYS[j]} measured")
    for x in ct:
        p.axvline(x, color="0.6", lw=0.5, alpha=0.5)
    if (~sent).any():
        lo = np.nanmin(a[:, arm]) if sent.any() else 0.0
        p.plot(t[~sent], np.full((~sent).sum(), lo), "|", color="red", ms=8, label="no action this tick")
    p.set_title("commands vs measured joints (grey lines: a new chunk arrived)")
    p.set_xlabel("s")
    p.set_ylabel("rad")
    p.legend(fontsize=8, loc="best")

    # 2. gap between commands
    p = ax[0, 1]
    ts = t[sent]
    if len(ts) > 1:
        p.plot(ts[1:], np.diff(ts) * 1000, ".", ms=3)
    p.axhline(1000 / fps, color="k", lw=0.8, ls="--", label=f"1 control period ({1000 / fps:.0f} ms)")
    p.set_yscale("log")
    p.set_title("time between new commands (spikes: the robot waited for actions)")
    p.set_xlabel("s")
    p.set_ylabel("ms")
    p.legend(fontsize=8)

    # 3. latency per chunk
    p = ax[1, 0]
    lat = np.array([np.nan if c["latency_s"] is None else c["latency_s"] * 1000 for c in cks])
    comp = np.array([np.nan if c.get("compute_s") is None else c["compute_s"] * 1000 for c in cks])
    if len(cks):
        p.plot(ct, lat, "o-", ms=3, lw=0.8, label="observation -> actions ready")
        if np.isfinite(comp).any():
            p.plot(ct, comp, "x", ms=4, label="policy compute")
    p.axhline(1000 / fps, color="k", lw=0.8, ls="--", label="1 control period")
    p.set_title("inference latency per chunk")
    p.set_xlabel("s")
    p.set_ylabel("ms")
    p.legend(fontsize=8)

    # 4. queue
    p = ax[1, 1]
    q = np.array([np.nan if x[3] is None else x[3] for x in rows], dtype=float)
    if np.isfinite(q).any():
        p.plot(t, q, lw=0.8)
    p.set_title("actions queued (0: nothing left to execute)")
    p.set_xlabel("s")
    if len(cks):
        p2 = p.twinx()
        p2.plot(ct, [c["dropped"] for c in cks], "r.", ms=4)
        p2.set_ylabel("too-late actions skipped per chunk", color="red", fontsize=8)

    # 5. step size, chunk switches marked
    p = ax[2, 0]
    idx = np.where(sent)[0]
    if len(idx) > 1:
        step = np.abs(np.diff(a[idx][:, arm], axis=0)).max(axis=1)
        switch = np.array([rows[i][4] for i in idx[1:]])
        p.plot(t[idx[1:]], step, "-", lw=0.6, color="0.4", label="arm step per command")
        p.plot(t[idx[1:]][switch], step[switch], "o", ms=4, color="#ff7f0e", label="first action of a new chunk")
    p.set_title("largest arm joint step between consecutive commands")
    p.set_xlabel("s")
    p.set_ylabel("rad")
    p.legend(fontsize=8)

    # 6. summary table
    p = ax[2, 1]
    p.axis("off")
    text = "\n".join(f"{label:<42s}{'-' if math.isnan(summary[key]) else fmt.format(summary[key]):>10s}"
                     for key, label, fmt in METRICS if isinstance(summary.get(key), (int, float)))
    p.text(0, 1, text, family="monospace", fontsize=9, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png)


def open_file(path: Path) -> None:
    if os.environ.get("OPENARM_EVAL_OPEN", "1").lower() in ("0", "false", "no"):
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return
    try:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    except Exception:
        pass


# -- lerobot-rollout (normal, rtc) -----------------------------------------------------------------------
def _time_policy(policy) -> None:
    """Time the policy's chunk computation (outermost call per thread), wherever it is called from."""
    if policy is None or getattr(policy, "_openarm_timed", False):
        return
    depth = threading.local()
    policy._openarm_calls, policy._openarm_last = 0, None

    def wrap(name):
        fn = getattr(policy, name, None)
        if fn is None:
            return

        def timed(*args, **kwargs):
            level = getattr(depth, "n", 0)
            depth.n = level + 1
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                depth.n = level
                if level == 0:
                    prev = kwargs.get("prev_chunk_left_over")
                    policy._openarm_last = {"start": start, "compute_s": time.perf_counter() - start,
                                            "guided": None if prev is None else int(prev.shape[0])}
                    policy._openarm_calls += 1

        object.__setattr__(policy, name, timed)

    wrap("predict_action_chunk")
    wrap("_get_action_chunk")  # SmolVLA's select_action computes its chunks here
    object.__setattr__(policy, "_openarm_timed", True)


def _queue_size(engine) -> int | None:
    queue = getattr(engine, "action_queue", None)  # RTC
    if queue is not None and hasattr(queue, "qsize"):
        return queue.qsize()
    if "_openarm_chunk" in getattr(engine, "__dict__", {}):  # GR00T's chunk-wise decoding (policy_loading.py)
        return len(engine.__dict__["_openarm_chunk"])
    policy = getattr(engine, "_policy", None)
    if getattr(policy, "_action_queue", None) is not None:  # pi0.5, GR00T
        return len(policy._action_queue)
    queues = getattr(policy, "_queues", None)  # SmolVLA
    if isinstance(queues, dict) and "action" in queues:
        return len(queues["action"])
    return None


def install() -> None:
    """lerobot-rollout only: log every tick and chunk of the episodic strategy (normal and RTC inference)."""
    command = sys.argv[0].rsplit("/", 1)[-1] if sys.argv else ""
    if "rollout" not in command or not _enabled():
        return
    import lerobot.policies.rtc.action_queue as action_queue
    import lerobot.rollout.strategies.episodic as episodic

    from .policy_loading import sync_module

    sync = sync_module()
    strategy = episodic.EpisodicStrategy
    if getattr(strategy, "_openarm_eval_metrics", False):
        return
    inference_type = (_argv_value("--inference.type") or "").lower()
    mode = inference_type if inference_type in ("rtc", "remote") else "normal"
    model = _argv_value("--policy.path") or "policy"
    setup0, loop0, teardown0 = strategy.setup, strategy._policy_loop, strategy.teardown
    send0, get0, merge0 = episodic.send_next_action, sync.SyncInferenceEngine.get_action, action_queue.ActionQueue.merge

    def setup(self, ctx):
        global LOG
        setup0(self, ctx)
        try:
            engine = self._engine
            _time_policy(getattr(engine, "_policy", None))
            policy = getattr(engine, "_policy", None)
            LOG = EvalLog(mode, model, ctx.runtime.cfg.fps, getattr(getattr(policy, "config", None), "type", None))
            LOG.engine = engine
        except Exception as e:
            print(f"[openarm eval] timing log off: {e!r}", flush=True)

    def _policy_loop(self, *args, **kwargs):
        if LOG is not None:
            LOG.begin_episode()
        try:
            return loop0(self, *args, **kwargs)
        finally:
            if LOG is not None:
                LOG.end_episode()

    def teardown(self, ctx):
        try:
            return teardown0(self, ctx)
        finally:
            if LOG is not None:
                LOG.finish()

    def send_next_action(obs_processed, obs_raw, ctx, interpolator, timer=None):
        action = send0(obs_processed, obs_raw, ctx, interpolator, timer)
        if LOG is not None:
            LOG.tick(action, obs_raw, _queue_size(ctx.policy.inference))
        return action

    def get_action(self, obs_frame):  # normal: the chunk is computed inside this call, the robot waits
        policy = getattr(self, "_policy", None)
        before, start = getattr(policy, "_openarm_calls", 0), time.perf_counter()
        out = get0(self, obs_frame)
        if LOG is not None and getattr(policy, "_openarm_calls", 0) != before:
            last = policy._openarm_last or {}
            LOG.chunk(latency_s=time.perf_counter() - start, compute_s=last.get("compute_s"),
                      n_actions=getattr(policy.config, "n_action_steps", 0) or 0, dropped=0)
        return out

    def merge(self, original_actions, processed_actions, real_delay, action_index_before_inference=None, **kwargs):
        # kwargs: task, and on update_dev_v2 also provenance / snapshot; its merge returns whether it merged
        out = merge0(self, original_actions, processed_actions, real_delay, action_index_before_inference, **kwargs)
        if LOG is not None:  # RTC: the inference thread merges each new chunk, skipping real_delay actions
            policy = getattr(getattr(LOG, "engine", None), "_policy", None)
            last = getattr(policy, "_openarm_last", None) or {}
            n = int(processed_actions.shape[0])
            LOG.chunk(latency_s=(time.perf_counter() - last["start"]) if "start" in last else real_delay / LOG.fps,
                      compute_s=last.get("compute_s"), n_actions=n, delay_steps=real_delay, guided=last.get("guided"),
                      dropped=min(max(int(real_delay), 0), n) if self.cfg.enabled else 0)
        return out

    strategy.setup, strategy._policy_loop, strategy.teardown = setup, _policy_loop, teardown
    episodic.send_next_action = send_next_action
    sync.SyncInferenceEngine.get_action = get_action
    action_queue.ActionQueue.merge = merge
    strategy._openarm_eval_metrics = True


# -- robot_client (async, async_rtc) ---------------------------------------------------------------------
def client_class(base, log: EvalLog):
    """`base` (lerobot's RobotClient, or our episode/display subclass of it) feeding `log`.

    A tick is one pass of lerobot's control loop: it sends the next queued action, or none when the queue
    is empty. A chunk is timed from the observation it was computed from (its first action carries that
    observation's timestamp) to its arrival; the actions whose timestep the robot has already executed are
    the ones the client skips. Our policy server adds its compute time and RTC state to each chunk."""

    class MetricsClient(base):
        _openarm_sent = None
        _openarm_seen = None

        def _measured(self):
            # The joints read last (openarm_umeow / openarm_isaac), only if read since the previous tick: the
            # client reads the robot only on ticks that send an observation, and an older reading would count
            # as tracking error.
            measured = getattr(self.robot, "_measured", None)
            if measured is self._openarm_seen:
                return None
            self._openarm_seen = measured
            return measured

        def actions_available(self):
            available = super().actions_available()
            if not available and not self.shutdown_event.is_set():
                log.tick(None, self._measured(), 0)
            return available

        def _action_tensor_to_action_dict(self, action_tensor):
            self._openarm_sent = super()._action_tensor_to_action_dict(action_tensor)
            return self._openarm_sent

        def control_loop_action(self, verbose: bool = False):
            performed = super().control_loop_action(verbose)
            with self.action_queue_lock:
                queued = self.action_queue.qsize()
            log.tick(self._openarm_sent, self._measured(), queued)
            return performed

        def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
            stale = getattr(self, "_stale", lambda: False)() or self.shutdown_event.is_set()
            if incoming_actions and not stale:
                with self.latest_action_lock:
                    latest = self.latest_action
                first = incoming_actions[0]
                info = getattr(first, "_openarm", None) or {}
                log.chunk(latency_s=time.time() - first.get_timestamp(), compute_s=info.get("compute_s"),
                          n_actions=len(incoming_actions), guided=info.get("guided"),
                          dropped=sum(1 for x in incoming_actions if x.get_timestep() <= latest),
                          delay_steps=max(0, latest - first.get_timestep()),
                          mode=("async_rtc" if info.get("rtc") else "async") if info else None)
            return super()._aggregate_action_queues(incoming_actions, aggregate_fn)

    return MetricsClient
