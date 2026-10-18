"""Report on a teleop diagnostics log (diag_log.py, --teleop.diag=true): where does jerky motion come from?

    python -m lerobot_robot_openarm_umeow.diag_report                 # newest log in outputs/teleop_diag
    python -m lerobot_robot_openarm_umeow.diag_report <log folder> [--episode N] [--no-plot]

Per recorded episode it prints the Quest network, the hand's motion, the IK thread, lerobot-record's
loop, the robot's limits and how well each joint followed, then lists every JERK -- a measured joint
whose acceleration jumps -- with what happened around it (+-JERK_WINDOW_S), and writes a timeline
PNG next to the log.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from .common import MOTOR_KEYS
from .diag_log import load

ARM = [k for k in MOTOR_KEYS if not k.endswith("J8.pos")]
JERK_ACC = 15.0  # rad/s^2: a measured joint acceleration (30 Hz) above this is a jerk
JERK_WINDOW_S = 0.1
LOOP_STALL_S = 0.05  # lerobot-record's loop (33 ms at 30 fps) taking longer than this
NET_GAP_S = 0.05  # Quest packets (72 Hz = 14 ms) arriving this far apart
IK_STALL_S = 0.01  # IK ticks (2 ms) this far apart


def _cols(columns: dict, stream: str) -> dict[str, int]:
    return {name: i for i, name in enumerate(columns.get(stream, []))}


def _pct(x, q) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def _stats_ms(x) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)] * 1000
    if not x.size:
        return "n/a"
    return f"median {np.median(x):.1f} / p99 {np.percentile(x, 99):.1f} / max {x.max():.1f} ms"


def _windows(events: list[dict], t_end: float) -> list[tuple[int | None, float, float, str]]:
    """(dataset episode, start, end, outcome) of every recorded episode, from the events."""
    out, start, idx, outcome = [], None, None, "?"
    for e in events:
        if e["text"] == "episode start":
            start, idx, outcome = e["t"], e.get("dataset_episode"), "?"
        elif e["text"] in ("episode saved", "episode discarded") and start is not None:
            outcome = e["text"].split()[1]
        elif e["text"] == "episode end" and start is not None:
            out.append((idx, start, e["t"], outcome))
            start = None
    if start is not None:
        out.append((idx, start, t_end, outcome + " (log ended)"))
    return out


def _sel(arr: np.ndarray | None, t0: float, t1: float) -> np.ndarray:
    if arr is None or not len(arr):
        return np.zeros((0, 1))
    return arr[(arr[:, 0] >= t0) & (arr[:, 0] <= t1)]


def _near(times: np.ndarray, t: float, w: float = JERK_WINDOW_S) -> bool:
    return bool(times.size) and bool(np.any(np.abs(times - t) <= w))


def _lag_ms(ta, a, tb, b, max_ms=400, step_s=0.002, hold=(True, True)) -> tuple[float, float]:
    """How long b trails a (ms) and how alike they are then (0..1): the shift that best lines up their
    velocities (sign-agnostic per column, so joint conventions do not matter). (nan, nan) if too little motion.
    hold: a / b is a command, in force until the next one (else a measurement, interpolated)."""
    a, b = np.atleast_2d(a.T).T, np.atleast_2d(b.T).T
    ok_a, ok_b = np.isfinite(a).all(1), np.isfinite(b).all(1)
    ta, a, tb, b = ta[ok_a], a[ok_a], tb[ok_b], b[ok_b]
    if len(ta) < 10 or len(tb) < 10:
        return float("nan"), float("nan")
    grid = np.arange(max(ta[0], tb[0]), min(ta[-1], tb[-1]), step_s)
    if len(grid) < 3 * max_ms / 1000 / step_s:
        return float("nan"), float("nan")
    # Velocities, averaged over 40 ms: signals that update in steps (72 Hz packets, 30 Hz commands) would
    # otherwise line up with themselves at lag 0 rather than with each other's motion.
    box = np.ones(int(0.04 / step_s)) / int(0.04 / step_s)

    def vel(t, x, held):
        if held:
            on = x[np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)]
        else:
            on = np.stack([np.interp(grid, t, x[:, i]) for i in range(x.shape[1])], 1)
        v = np.diff(on, axis=0)
        return np.stack([np.convolve(v[:, i], box, mode="same") for i in range(v.shape[1])], 1)

    va, vb = vel(ta, a, hold[0]), vel(tb, b, hold[1])
    if np.linalg.norm(va) < 1e-6 or np.linalg.norm(vb) < 1e-6:
        return float("nan"), float("nan")
    best, best_c = 0, -1.0
    for lag in range(0, int(max_ms / 1000 / step_s)):
        x, y = va[: len(va) - lag], vb[lag:]
        num = np.abs((x * y).sum(0))
        c = float(num.sum() / (np.sqrt((x * x).sum(0) * (y * y).sum(0)).sum() + 1e-12))
        if c > best_c:
            best, best_c = lag, c
    return best * step_s * 1000, best_c


def _fk_positions(joints16: np.ndarray) -> np.ndarray | None:
    """EE positions [right xyz, left xyz] of driver16 rows (the IK model), or None without the teleop plugin."""
    try:
        from lerobot_teleoperator_openarm_quest.ik_driver import build_kinematics

        from .common import DEFAULT_IK_XML
    except ImportError:
        return None
    kin = build_kinematics(DEFAULT_IK_XML, "home", "--mode bimanual")
    out = np.empty((len(joints16), 6))
    for i, q in enumerate(joints16):
        r, l = kin.fk_bimanual(q[0:8], q[8:16])
        out[i] = [*r[:3], *l[:3]]
    return out


def report_latency(q, cq, ik, cik, obs, cobs, send, csend) -> None:
    """Where the time between a hand movement and the arm's movement goes, stage by stage."""
    rows = []
    if len(q) > 2 and np.isfinite(q[:, cq["headset_t"]]).all():
        delay = q[:, 0] - q[:, cq["headset_t"]]
        extra = delay - delay.min()
        rows.append(("Wi-Fi: packet delay above the best seen", float(np.median(extra) * 1000), float("nan")))
    live = ik[:, cik["state"]] == 1 if len(ik) else np.zeros(0, bool)
    if live.sum() > 100:
        L = ik[live]
        t = L[:, 0]
        for side in ("r", "l"):
            raw = L[:, [cik[f"raw_{side}_{k}"] for k in "xyz"]]
            tgt = L[:, [cik[f"target_{side}_{k}"] for k in "xyz"]]
            lag, c = _lag_ms(t, raw, t, tgt)
            rows.append((f"1-Euro filter ({'right' if side == 'r' else 'left'} hand)", lag, c))
        joints = [f"{s}{i}" for s in "rl" for i in range(1, 8)]
        raw_j = L[:, [cik[f"ik_{j}"] for j in joints]]
        cmd_j = L[:, [cik[f"cmd_{j}"] for j in joints]]
        rows.append(("IK solution -> teleop speed limit", *_lag_ms(t, raw_j, t, cmd_j)))
        if len(send) > 10:
            keys = [f"{s.upper()}J{i}.pos" for s in "rl" for i in range(1, 8)]
            sent = send[:, [csend[f"sent_{k}"] for k in keys]]
            rows.append(("teleop command -> sent to motors (30 Hz loop)", *_lag_ms(t, cmd_j, send[:, 0], sent)))
            if len(obs) > 10:
                meas = obs[:, [cobs[f"meas_{k}"] for k in keys]]
                rows.append(("sent -> measured (motors following)", *_lag_ms(send[:, 0], sent, obs[:, 0], meas, hold=(True, False))))
                for side, cols in (("right", slice(0, 7)), ("left", slice(7, 14))):
                    rows.append((f"   {side} arm only", *_lag_ms(send[:, 0], sent[:, cols], obs[:, 0], meas[:, cols], hold=(True, False))))
        if len(obs) > 10:
            from .common import (DEFAULT_CALIBRATION, load_calibration, motor_action_to_sim_joints,
                                 sim_joints_to_driver16)

            calib = load_calibration(DEFAULT_CALIBRATION)
            o = obs[(obs[:, 0] >= t[0]) & (obs[:, 0] <= t[-1])]
            mkeys = [f"meas_{k}" for k in MOTOR_KEYS]
            d16 = np.array([sim_joints_to_driver16(motor_action_to_sim_joints(
                dict(zip(MOTOR_KEYS, row[[cobs[k] for k in mkeys]])), calib)) for row in o])
            ee = _fk_positions(d16) if len(d16) > 10 else None
            if ee is not None:
                raw = L[:, [cik[f"raw_{s}_{k}"] for s in "rl" for k in "xyz"]]
                rows.append(("END TO END: hand (PC side) -> measured gripper position", *_lag_ms(t, raw, o[:, 0], ee, hold=(True, False))))
    if not rows:
        return
    print("DELAY, stage by stage (ms behind the previous stage; match = how alike the motions are, 1 = identical):")
    for name, lag, c in rows:
        print(f"  {name:58} {lag:6.0f} ms" + ("" if not np.isfinite(c) else f"   match {c:.2f}"))
    print("  (Wi-Fi is the extra delay only: the headset's base one-way delay cannot be measured from its clock.)")


def report_window(streams, columns, events, t0, t1, title, plot_path=None):
    q, ik, obs, send = (_sel(streams.get(k), t0, t1) for k in ("quest", "ik", "obs", "send"))
    cq, cik, cobs, csend = (_cols(columns, k) for k in ("quest", "ik", "obs", "send"))
    dur = t1 - t0
    print(f"\n=== {title}: {dur:.1f} s ===")

    # Quest network
    net_gaps = np.zeros(0)
    if len(q) > 2:
        arr = q[:, 0]
        gaps = np.diff(arr)
        net_gaps = arr[1:][gaps > NET_GAP_S]
        print(f"Quest packets   {len(q) / dur:.0f}/s | arrival gaps {_stats_ms(gaps)} | {len(net_gaps)} gaps > {NET_GAP_S * 1000:.0f} ms")
        ht = q[:, cq["headset_t"]]
        if np.isfinite(ht).all():
            hgap = np.diff(ht)
            delay = arr - ht
            print(f"  headset clock frame interval {_stats_ms(hgap)} (headset itself slow/dropping frames if high)")
            print(f"  extra network delay {_stats_ms(delay - delay.min())}")
        drained = q[:, cq["drained"]]
        print(f"  packets arriving in bursts (>1 queued): {100 * (drained > 1).mean():.1f}%")
        for side, key in (("right", "vr"), ("left", "vl")):
            bad = (q[:, cq[key]] == 2).mean() * 100
            if bad:
                print(f"  {side} controller INVALID tracking {bad:.1f}% of packets")
        # Hand motion, raw (Unity frame): speed and tremor (residual from a 5-packet moving average).
        for side, c in (("right", "rc"), ("left", "lc")):
            p = q[:, [cq[f"{c}_x"], cq[f"{c}_y"], cq[f"{c}_z"]]]
            if not np.isfinite(p).all():
                continue
            tt = ht if np.isfinite(ht).all() else arr
            dt = np.maximum(np.diff(tt), 1e-3)
            v = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
            k = np.ones(5) / 5
            smooth = np.stack([np.convolve(p[:, i], k, mode="same") for i in range(3)], 1)
            tremor = np.linalg.norm((p - smooth)[2:-2], axis=1) * 1000
            print(f"  {side} hand speed median {np.median(v):.2f} / p99 {_pct(v, 99):.2f} m/s,"
                  f" tracking noise {np.sqrt((tremor ** 2).mean()):.1f} mm rms")
    else:
        print("Quest packets   none logged in this window")

    # IK thread
    ik_stalls = np.zeros(0)
    limited_t = np.zeros(0)
    if len(ik) > 2:
        tick = ik[:, cik["tick_s"]]
        step = ik[:, cik["step_s"]]
        ik_stalls = ik[tick > IK_STALL_S, 0]
        print(f"IK thread       {len(ik) / dur:.0f} ticks/s | tick interval {_stats_ms(tick)} | {len(ik_stalls)} stalls"
              f" > {IK_STALL_S * 1000:.0f} ms | step time {_stats_ms(step)}")
        live = ik[:, cik["state"]] == 1
        raw = ik[:, [cik[f"ik_{s}{i}"] for s in "rl" for i in range(1, 8)]]
        cmd = ik[:, [cik[f"cmd_{s}{i}"] for s in "rl" for i in range(1, 8)]]
        ok = live & np.isfinite(raw).all(1)
        if ok.any():
            lead = np.abs(raw[ok] - cmd[ok])
            limited = lead.max(1) > 0.01
            limited_t = ik[ok][limited, 0]
            names = [f"{s}J{i}" for s in "RL" for i in range(1, 8)]
            worst = names[int(lead.max(0).argmax())]
            print(f"  speed limit (teleop max_joint_speed) holding the command back > 0.01 rad: {100 * limited.mean():.1f}%"
                  f" of LIVE ticks; max lead {lead.max():.2f} rad on {worst}")
            jumps = np.abs(np.diff(raw[ok], axis=0)).max(1)
            print(f"  raw IK solution change per tick: p99 {_pct(jumps, 99):.4f} / max {jumps.max():.3f} rad")

    # lerobot-record loop and robot
    loop_stalls = np.zeros(0)
    if len(send) > 2:
        interval = send[:, csend["interval_s"]]
        loop_stalls = send[interval > LOOP_STALL_S, 0]
        print(f"record loop     {len(send) / dur:.1f} commands/s | interval {_stats_ms(interval)} | {len(loop_stalls)}"
              f" stalls > {LOOP_STALL_S * 1000:.0f} ms")
        if len(obs):
            print(f"  robot read (motors + cameras) {_stats_ms(obs[:, cobs['read_s']])} | command send {_stats_ms(send[:, csend['send_s']])}")
        des = send[:, [csend[f"desired_{k}"] for k in ARM]]
        sent = send[:, [csend[f"sent_{k}"] for k in ARM]]
        clamp = np.abs(des - sent).max(1) > 1e-6
        print(f"  robot step limit (max_joint_speed) clipped {100 * clamp.mean():.1f}% of commands;"
              f" safety holds {int(send[:, csend['hold']].sum())} commands")

    report_latency(q, cq, ik, cik, obs, cobs, send, csend)

    # Following and jerks
    jerks = []
    if len(obs) > 3 and len(send) > 3:
        meas = obs[:, [cobs[f"meas_{k}"] for k in ARM]]
        ts = obs[:, 0]
        sent_t = send[:, 0]
        sent = send[:, [csend[f"sent_{k}"] for k in ARM]]
        idx = np.clip(np.searchsorted(sent_t, ts) - 1, 0, len(sent_t) - 1)  # the command in force at each read
        err = meas - sent[idx]
        dt = np.maximum(np.diff(ts), 1e-3)
        vm = np.diff(meas, axis=0) / dt[:, None]
        am = np.diff(vm, axis=0) / dt[1:, None]
        dts = np.maximum(np.diff(sent_t), 1e-3)
        vc = np.diff(sent, axis=0) / dts[:, None]
        ac = np.diff(vc, axis=0) / dts[1:, None]
        print(f"{'joint':6}{'err p99':>9}{'acc p99 cmd':>12}{'meas':>7}{'jerks':>7}")
        for j, k in enumerate(ARM):
            n = int((np.abs(am[:, j]) > JERK_ACC).sum())
            print(f"{k[:-4]:6}{_pct(np.abs(err[:, j]), 99):9.3f}{_pct(np.abs(ac[:, j]), 99):12.1f}{_pct(np.abs(am[:, j]), 99):7.1f}{n:7d}")
        cmd_acc_t = sent_t[1:-1][np.abs(ac).max(1) > JERK_ACC]
        for i in np.where(np.abs(am).max(1) > JERK_ACC)[0]:
            t = ts[i + 1]
            j = int(np.abs(am[i]).argmax())
            causes = [name for name, times in (
                ("record-loop stall", loop_stalls), ("Quest packet gap", net_gaps), ("IK thread stall", ik_stalls),
                ("command itself jerked", cmd_acc_t), ("teleop speed limit active", limited_t)) if _near(times, t)]
            jerks.append((t - t0, ARM[j][:-4], float(am[i, j]), causes or ["motor (command smooth, nothing else)"]))
    if jerks:
        print(f"JERKS (measured acceleration > {JERK_ACC:g} rad/s^2): {len(jerks)}")
        tally: dict[str, int] = {}
        for _, _, _, causes in jerks:
            for c in causes:
                tally[c] = tally.get(c, 0) + 1
        for c, n in sorted(tally.items(), key=lambda x: -x[1]):
            print(f"  {n:4d} x {c}")
        for t, joint, a, causes in jerks[:40]:
            print(f"   t={t:6.2f}s {joint:4} {a:+7.1f} rad/s^2  <- {', '.join(causes)}")
        if len(jerks) > 40:
            print(f"   ... {len(jerks) - 40} more")
    elif len(obs) > 3:
        print(f"no jerks (measured acceleration never above {JERK_ACC:g} rad/s^2)")

    if plot_path is not None and len(obs) > 3 and len(send) > 3:
        _plot(plot_path, t0, q, cq, ik, cik, obs, cobs, send, csend, jerks, title)


def _plot(path, t0, q, cq, ik, cik, obs, cobs, send, csend, jerks, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meas = obs[:, [cobs[f"meas_{k}"] for k in ARM]]
    sent = send[:, [csend[f"sent_{k}"] for k in ARM]]
    count = {k: 0 for k in ARM}
    for _, joint, _, _ in jerks:
        count[joint + ".pos"] += 1
    worst = sorted(ARM, key=lambda k: -count[k])[:4]
    fig, ax = plt.subplots(3 + len(worst), 1, figsize=(16, 3 * (3 + len(worst))), sharex=True)
    if len(q) > 1:
        ax[0].plot(q[1:, 0] - t0, np.diff(q[:, 0]) * 1000, lw=.6, label="Quest packet gap")
    ax[0].plot(send[1:, 0] - t0, send[1:, csend["interval_s"]] * 1000, lw=.6, label="record loop interval")
    if len(ik) > 1:
        ax[0].plot(ik[:, 0] - t0, ik[:, cik["tick_s"]] * 1000, lw=.4, alpha=.6, label="IK tick")
    ax[0].set_ylabel("ms"); ax[0].set_ylim(0, 150); ax[0].legend(loc="upper right")
    for c, side in (("rc", "right"), ("lc", "left")):
        if len(q) > 2:
            p = q[:, [cq[f"{c}_x"], cq[f"{c}_y"], cq[f"{c}_z"]]]
            v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(q[:, 0]), 1e-3)
            ax[1].plot(q[1:, 0] - t0, v, lw=.6, label=f"{side} hand")
    ax[1].set_ylabel("hand m/s"); ax[1].set_ylim(0, 2); ax[1].legend(loc="upper right")
    dt = np.maximum(np.diff(obs[:, 0]), 1e-3)
    am = np.abs(np.diff(np.diff(meas, axis=0) / dt[:, None], axis=0) / dt[1:, None]).max(1)
    ax[2].plot(obs[2:, 0] - t0, am, lw=.6, color="r"); ax[2].axhline(JERK_ACC, ls="--", color="k", lw=.6)
    ax[2].set_ylabel("max |acc| rad/s^2")
    for a, k in zip(ax[3:], worst):
        j = ARM.index(k)
        a.plot(send[:, 0] - t0, sent[:, j], lw=.8, label="command")
        a.plot(obs[:, 0] - t0, meas[:, j], lw=.8, label="measured")
        for t, joint, _, _ in jerks:
            if joint + ".pos" == k:
                a.axvline(t, color="r", lw=.5, alpha=.5)
        a.set_ylabel(f"{k[:-4]} ({count[k]} jerks)"); a.legend(loc="upper right")
    ax[-1].set_xlabel("s"); fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)
    print(f"timeline: {path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", help="log folder (default: the newest in outputs/teleop_diag)")
    ap.add_argument("--episode", type=int, help="only this dataset episode index")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    if args.log:
        path = Path(args.log)
    else:
        logs = sorted(Path("outputs/teleop_diag").glob("*/"))
        if not logs:
            sys.exit("no logs in outputs/teleop_diag -- record with --teleop.diag=true")
        path = logs[-1]
    streams, columns, events, meta = load(path)
    if not streams:
        sys.exit(f"{path}: nothing logged yet")
    t_end = max(float(v[-1, 0]) for v in streams.values())
    print(f"log {path}")
    windows = _windows(events, t_end)
    if not windows:
        t_start = min(float(v[0, 0]) for v in streams.values())
        windows = [(None, t_start, t_end, "whole log, no episode markers")]
    for idx, t0, t1, outcome in windows:
        if args.episode is not None and idx != args.episode:
            continue
        name = f"episode {idx}" if idx is not None else "log"
        plot = None if args.no_plot else path / f"timeline_{'ep%03d' % idx if idx is not None else 'all'}_{int(t0)}.png"
        report_window(streams, columns, events, t0, t1, f"{name} ({outcome})", plot)


if __name__ == "__main__":
    main()
