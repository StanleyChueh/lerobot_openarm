#!/usr/bin/env python
"""Move the real dual-arm OpenArm follower to the sim's default rest pose -- every arm joint at
0 rad except joint4 (the elbow) at pi/2, gripper fully open, all mapped through calibration.
See rest_pose_action() on why joint4 is not 0.

Run this BEFORE mirror_bridge.py / replay_sim_dataset.py / replay_hf_sim_episode.py whenever the
real arm has drifted from that pose (e.g. left wherever a previous session's teleop ended), so
those scripts' startup handshake passes cleanly instead of aborting on a large real-vs-sim delta.

It goes to the rest pose FROM WHEREVER THE ARM IS. It used to gate itself behind
run_startup_handshake(), i.e. "if the arm is more than --handshake-tolerance from the rest pose,
abort and tell the human to move it by hand" -- which inverts this script's whole purpose: being
far from the rest pose is the reason to run it, so the check fired exactly when the script was
needed and its advice ("move the real arm to match") was the job it was being asked to do. It now
uses approach_pose() (sim_bridge_common.py) like every other script here: a speed-limited ramp
behind a typed confirmation, refusing only a travel distance so large it implies broken
calibration rather than a displaced arm -- and --max-delta defaults high enough that no plausible
pair of readings trips it. Tighten it if you want that gate back; --yes skips the prompt.

Tracking is measured, not assumed. The approach ramp and a following hold at the target are both
sampled, printed as a per-joint summary, and (unless --no-plot) written out as a commanded-vs-
actual plot. A residual of tens of milliradians at the hold is normal steady-state error for
these gains, not a failure -- see approach_pose()'s docstring.

A flat "actual" trace has two opposite causes, and this script no longer guesses between them.
A CAN read that never refreshed returns retained Motor state -- the previous position, which is
plausible and unchanging -- and a joint that is not moving reports the same float, bit for bit,
on perfectly healthy feedback. Position alone cannot separate them, and neither can "the reading
sat still while the command swept away from it": a joint too weakly driven to break its own
static friction produces exactly that. Measured 2026-08-26, the old test on that rule reported
"feedback stopped refreshing" for LJ8 -- a gripper at kp=3.0 commanded to travel 0.0245 rad, i.e.
a channel telling the truth about a joint physically incapable of responding -- and instructed the
operator not to read it as a gains problem.

So freshness is now COUNTED. OpenArmFollower opens a passive tap on each CAN interface
(can_monitor.py) that tallies the state frames every motor actually puts on the wire; SocketCAN
delivers each frame to every bound socket, so the tap observes openarm_can's traffic without
consuming any of it. A channel that sent zero frames during a read is stale (no-fb column, red
dots on the plot) and its error is an artefact. A channel that sent frames and still did not move
while the command swept past --stale-travel is a joint that did not follow (purple crosses) --
a real tracking failure, to be chased through gains, friction, or the motor's own fault status.

That fault status is the other thing the tap recovers: the high nibble of data[0] in a DM state
frame is the motor's status/error code, which openarm_can's decoder skips over entirely. It is the
only trustworthy "is this motor armed" signal on the bus. Motor.is_enabled() is not one --
set_enabled() is defined in the library and never called from anywhere, so it reads False for all
sixteen channels forever, which is why a run where five joints tracked to within 0.03 rad still
showed every channel "disabled".

Read the summary travel-first either way: a joint commanded to move 0.01 rad cannot demonstrate
good or bad tracking, so the headline numbers are drawn from the joints that actually travelled on
fresh feedback.

Usage:
  python reset_to_rest_pose.py --calibration calibration.json \\
      --model-path model/openarm_description_leader.urdf

Note for a shell with /opt/ros/humble sourced: ROS's PYTHONPATH and LD_LIBRARY_PATH shadow this
venv's pinocchio and libeigenpy with Python-3.10 / NumPy-1.x builds, so prefix the command with
`env -u PYTHONPATH -u LD_LIBRARY_PATH`.
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # headless -- this process only ever saves a PNG, never shows a window
import matplotlib.pyplot as plt

from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from sim_bridge_common import (
    PLAUSIBLE_JOINT_RANGE,
    approach_pose,
    check_arms_not_crossed,
    get_current_pos_action,
    load_calibration,
    ramp_to,
    sim_init_pose_action,
)

POS_KEYS = [f"LJ{i}.pos" for i in range(1, 9)] + [f"RJ{i}.pos" for i in range(1, 9)]

# The largest travel two plausible position reads can possibly ask for: one joint pegged at
# -PLAUSIBLE_JOINT_RANGE with its target at +PLAUSIBLE_JOINT_RANGE. Using this as the default
# --max-delta means approach_pose()'s refusal gate cannot fire on any reading this codebase is
# willing to trust, which is what "go to rest from wherever the arm is" requires. The gate still
# catches a target outside that range, which no correct calibration can produce -- so what remains
# is a broken-calibration check, not a displaced-arm check. The typed confirmation is what keeps a
# human in the loop; --max-delta is how you put a distance limit back.
UNCONDITIONAL_MAX_DELTA = 2 * PLAUSIBLE_JOINT_RANGE


def rest_pose_action(calib: dict) -> dict:
    """Isaac Sim's own env.reset() pose, mapped through calibration: every arm joint at 0 EXCEPT
    joint4 at pi/2, both grippers fully open.

    This used to put joint4 at 0 along with the rest while still describing itself as matching
    env.reset(). It did not: OPENARM_BI_CFG's init_state (IsaacLab's
    source/isaaclab_assets/isaaclab_assets/robots/openarm.py) starts both elbows bent at 1.570796,
    which is why a live sim packet's LJ4/RJ4 read +1.5708 and a reset run against this function's
    all-zeros left those two joints a full 1.57 rad away from the pose it claimed to have reached.
    Now delegates to the single shared definition in sim_bridge_common.
    """
    return sim_init_pose_action(calib)


class TrackingRecorder:
    """Samples commanded-vs-actual joint positions from inside a ramp_to()/approach_pose() tick.

    Reading the robot back costs tens of milliseconds on this CAN stack, so `every` decimates the
    sampling: at 50 Hz ticks and --plot-hz 20 that is one read per 3 ticks. Overrunning the tick
    budget only slows the ramp down (ramp_to() interpolates by step index, not by clock), so the
    cost is wall-clock, never a faster or jerkier motion -- but that is also why the time axis
    here comes from ramp_to()'s own elapsed clock rather than from the planned duration.

    A read that fails is counted and skipped, never raised: this is a diagnostic riding along on a
    motion that is already in progress, and aborting that motion midway to report a bad
    *measurement* would leave the arm somewhere unintended for no benefit.

    Each sample carries two INDEPENDENT facts about the channel, and keeping them apart is the
    whole point of this class:

      - `frames`: how many state frames that motor actually put on the wire during the read,
        counted by a passive CAN tap (OpenArmFollower.take_feedback_counts, can_monitor.py).
        Zero means get_position() returned retained state and the sample is fiction.
      - `unchanged`: whether the decoded position was bit-identical to the previous sample.

    This used to be inferred from `unchanged` alone, and it cannot be. On this CAN stack a read
    that never refreshed returns retained Motor state -- see _read_motor_positions_once() -- so a
    frozen channel repeats its last position exactly. But so does a joint that is simply not
    moving: positions arrive quantised, so a stationary joint returns the same float every time,
    bit for bit. The two are indistinguishable in the position stream, and they call for opposite
    fixes (chase the bus vs. chase the gains).

    The previous test tried to break the tie with "unchanged WHILE the command swept past
    `stale_travel`", and that does not break it either -- a joint that is stuck, or too weakly
    driven to overcome its own friction, produces exactly that signature. Measured 2026-08-26: it
    reported "feedback stopped refreshing" for LJ8, a gripper at kp=3.0 commanded to travel
    0.0245 rad, i.e. a channel that was faithfully reporting a joint physically incapable of
    responding, and told the operator not to read it as a gains problem. On the same run a frame
    census over both buses (8 null-MIT + 8 refresh per cycle, the exact per-tick load, at 10 Hz
    and 50 Hz) came back at 100% replies on all sixteen channels with zero TX drops.

    So freshness is now counted, never inferred, and `stale_travel` keeps its job on the OTHER
    side of the fork: a channel with fresh frames whose reading still did not move while the
    command swept past `stale_travel` is a joint that did not follow, which is a real tracking
    failure and is reported as one.
    """

    def __init__(self, robot, every: int = 1, stale_travel: float = 0.02):
        self.robot = robot
        self.every = max(1, int(every))
        self.stale_travel = stale_travel
        self.history = {k: {"t": [], "target": [], "actual": [], "unchanged": [], "stale": [],
                            "stuck": [], "frames": []}
                        for k in POS_KEYS}
        self.phase_marks: list[tuple[float, str]] = []
        self.skipped = 0
        # None until the first sample; False if the follower could not open its CAN taps, in which
        # case freshness is UNKNOWN and this class must not claim either verdict.
        self.counting: bool | None = None
        self._offset = 0.0
        self._last_t = 0.0
        self._prev_actual: dict[str, float] = {}
        self._prev_cmd: dict[str, float] = {}
        self._cmd_travel: dict[str, float] = {}

    def start_phase(self, label: str):
        """Continue the time axis into a new segment (e.g. ramp -> hold) and mark the boundary."""
        self._offset = self._last_t
        self.phase_marks.append((self._offset, label))

    def on_tick(self, t: float, cmd: dict, i: int, steps: int):
        self._last_t = self._offset + t
        if i % self.every and i != steps:
            return
        try:
            actual = get_current_pos_action(self.robot, max_attempts=2)
        except Exception as e:  # noqa: BLE001 -- see class docstring: never abort a live motion
            self.skipped += 1
            if self.skipped == 1:
                print(f"  [tracking] skipping samples that fail to read ({type(e).__name__}: {e})")
            return
        # Taken AFTER the read so the window lines up with the frames that produced `actual`.
        # An empty dict means no tap is open, not that no frames arrived -- see self.counting.
        counts = self.robot.take_feedback_counts() if hasattr(self.robot, "take_feedback_counts") else {}
        if self.counting is None:
            self.counting = bool(counts)
        for k in POS_KEYS:
            if k not in cmd or k not in actual:
                continue
            h = self.history[k]
            h["t"].append(self._last_t)
            h["target"].append(cmd[k])
            h["actual"].append(actual[k])
            # Exact equality is the right test for "did not change": positions arrive quantised,
            # so both a retained value and a stationary joint repeat the same float bit for bit.
            # Which of the two it is comes from the frame count, not from this flag.
            unchanged = k in self._prev_actual and actual[k] == self._prev_actual[k]
            if unchanged:
                self._cmd_travel[k] = (self._cmd_travel.get(k, 0.0)
                                       + abs(cmd[k] - self._prev_cmd.get(k, cmd[k])))
            else:
                self._cmd_travel[k] = 0.0
            n_frames = counts.get(k[:-4]) if counts else None
            h["frames"].append(n_frames)
            # stale: the motor sent nothing, so this sample is retained state and its error is an
            # artefact. stuck: the motor DID report, and what it reported is a joint that has not
            # moved while the command walked away from it -- a real tracking failure.
            h["stale"].append(n_frames == 0)
            h["stuck"].append(bool(n_frames) and unchanged
                              and self._cmd_travel[k] > self.stale_travel)
        self._prev_actual = dict(actual)
        self._prev_cmd = dict(cmd)

    @property
    def n_samples(self) -> int:
        return len(self.history[POS_KEYS[0]]["t"])

    def _flagged(self, field: str, threshold: float) -> list[str]:
        out = []
        for k in POS_KEYS:
            flags = self.history[k][field]
            if flags and sum(flags) / len(flags) > threshold:
                out.append(k)
        return out

    def stale_channels(self, threshold: float = 0.2) -> list[str]:
        """Channels that MEASURABLY sent no state frames, for cross-referencing against motor state."""
        return self._flagged("stale", threshold)

    def outage_windows(self) -> dict[str, list[tuple[float, float]]]:
        """Per channel, the [start, end] time spans over which it sent no state frames.

        A percentage cannot tell a channel that drops one frame in three from one that goes away
        for two and a half seconds and comes back, and those are different faults. Spans can, and
        spans shared across several channels are the thing worth seeing: four motors that go quiet
        and return TOGETHER have one cause between them, which is a different investigation from
        four flaky links.
        """
        out = {}
        for k in POS_KEYS:
            h = self.history[k]
            spans, start = [], None
            for t, st in zip(h["t"], h["stale"]):
                if st and start is None:
                    start = t
                elif not st and start is not None:
                    spans.append((start, t))
                    start = None
            if start is not None:
                spans.append((start, h["t"][-1]))
            if spans:
                out[k] = spans
        return out

    def _target_at(self, key: str, t: float) -> float:
        """The commanded value for `key` at the sample nearest time `t`."""
        h = self.history[key]
        if not h["t"]:
            return float("nan")
        i = min(range(len(h["t"])), key=lambda j: abs(h["t"][j] - t))
        return h["target"][i]

    def print_outages(self):
        """Print the outage spans, grouped by the channels that share them."""
        windows = self.outage_windows()
        if not windows:
            return
        print("\nFeedback outages (spans with zero state frames):")
        for k in POS_KEYS:
            if k not in windows:
                continue
            spans = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in windows[k])
            total = sum(b - a for a, b in windows[k])
            print(f"  {k:10s} {len(windows[k])} outage(s), {total:.1f}s total: {spans}")

        # "It happens while the arm moves" is a much weaker statement than "it happens between
        # these two joint angles", and the second one is free: the commanded pose at each outage
        # boundary is already recorded. A fault that opens and closes at repeatable angles is a
        # mechanical one, and this is what tells you WHERE to look.
        movers = [k for k in POS_KEYS
                  if self.history[k]["target"]
                  and max(self.history[k]["target"]) - min(self.history[k]["target"]) >= 0.1]
        if movers:
            print("\n  Commanded pose at each outage boundary (joints that travelled >= 0.1 rad):")
            for k in POS_KEYS:
                if k not in windows:
                    continue
                for a, b in windows[k]:
                    at = lambda t: {m: self._target_at(m, t) for m in movers}
                    lo, hi = at(a), at(b)
                    detail = "  ".join(f"{m[:-4]} {lo[m]:+.2f}->{hi[m]:+.2f}" for m in movers)
                    print(f"    {k:10s} {a:4.1f}-{b:4.1f}s   {detail}")
        # Channels whose outage spans match to within a sample are not independent faults.
        groups: dict[tuple, list[str]] = {}
        for k, spans in windows.items():
            groups.setdefault(tuple((round(a, 1), round(b, 1)) for a, b in spans), []).append(k)
        shared = [g for g in groups.values() if len(g) > 1]
        for g in shared:
            print(f"\n  {', '.join(g)} went quiet and came back TOGETHER, to the sample."
                  " Independent motors do not do that -- they share whatever caused it:"
                  " one CAN branch, one connector, or one power rail. If they are a contiguous"
                  " run of joints, the suspect is the link feeding the first of them.")

    def stuck_channels(self, threshold: float = 0.2) -> list[str]:
        """Channels reporting fresh frames whose joint did not move while the command swept away.

        A predominantly stale channel is excluded even though its fresh samples look stuck: the
        few frames it did send say nothing about whether the joint followed, and reporting it as
        both would hand the operator two contradictory diagnoses for one channel.
        """
        stale = set(self.stale_channels(threshold))
        return [k for k in self._flagged("stuck", threshold) if k not in stale]

    def print_summary(self, hold_start: float | None = None):
        """Per-joint commanded travel, peak error while moving, residual at the hold, and how much
        of the channel never refreshed.

        Peak-during-motion and residual-at-rest are different things and worth separating: the
        first is dominated by how far each joint lags a moving command (a gains/speed question),
        the second by where it settles under gravity once the command stops changing (the number
        the other scripts' handshakes actually see).

        Travel leads the table because both are close to meaningless without it. A joint commanded
        to move 0.01 rad has no room to show a tracking error, so its residual says nothing about
        the controller; only the joints that actually went somewhere carry information about how
        well this arm tracks.
        """
        if not self.n_samples:
            print("\nNo tracking samples collected -- nothing to summarise.")
            return

        rows = []
        for k in POS_KEYS:
            h = self.history[k]
            errs = [abs(a - b) for a, b in zip(h["target"], h["actual"])]
            if not errs:
                continue
            travel = max(h["target"]) - min(h["target"])
            peak = max(errs)
            held = [e for t, e in zip(h["t"], errs) if hold_start is not None and t >= hold_start]
            residual = sum(held) / len(held) if held else errs[-1]
            unchanged = sum(h["unchanged"]) / len(h["unchanged"]) if h["unchanged"] else 0.0
            stale = sum(h["stale"]) / len(h["stale"]) if h["stale"] else 0.0
            stuck = sum(h["stuck"]) / len(h["stuck"]) if h["stuck"] else 0.0
            seen = [n for n in h["frames"] if n is not None]
            frames = sum(seen) / len(seen) if seen else 0.0
            rows.append((k, travel, peak, residual, unchanged, stale, stuck, frames))

        print(f"\nTracking summary ({self.n_samples} samples"
              + (f", {self.skipped} reads skipped" if self.skipped else "") + "):")
        if not self.counting:
            print("  [!] No CAN feedback tap was available, so freshness is UNKNOWN this run and"
                  " the no-fb/stuck columns are meaningless. See can_monitor.py.")
        print(f"  {'joint':10s} {'travel':>9s} {'peak |err|':>11s} {'residual':>9s}"
              f" {'unchgd':>7s} {'no-fb':>6s} {'frames':>7s}")
        print(f"  {'':10s} {'(rad)':>9s} {'(rad)':>11s} {'(rad)':>9s} {'':>7s} {'':>6s} {'/read':>7s}")
        for k, travel, peak, residual, unchanged, stale, stuck, frames in rows:
            if stale > 0.2:
                flag = "  <-- motor sent no state frames"
            elif stuck > 0.2:
                flag = "  <-- feedback fresh, joint did not follow"
            else:
                flag = ""
            print(f"  {k:10s} {travel:9.4f} {peak:11.4f} {residual:9.4f}"
                  f" {unchanged:6.0%} {stale:5.0%} {frames:7.1f}{flag}")

        suspect = [r for r in rows if r[5] > 0.2]
        if suspect:
            print(
                "\nWARNING: " + ", ".join(r[0] for r in suspect) + " sent NO state frames on the"
                " bus during those reads -- counted by the passive CAN tap, not inferred from the"
                " position value. Their positions above are retained state, so the errors shown for"
                " them are an artefact and must not be read as a gains problem."
                "\n  Two causes, and they need different fixes. Frames arriving but not being"
                " drained is a read-scheduling problem (--can-recv-rounds / --can-first-timeout-us,"
                " see _read_motor_positions_once() in openarm_follower.py); the tap counts frames"
                " the kernel delivered, so if the tap saw them and the joint still read stale, that"
                " is the one you have. Frames never sent is a motor or bus problem, and no --can-*"
                " value can recover them. Cross-check against the bus directly:"
                "\n    candump -n 20000 <canport> > /tmp/can.log     # during a run"
                "\n    awk '{print $2}' /tmp/can.log | sort | uniq -c | sort -rn"
                "\n  Reply IDs are 0x11..0x17 for J1..J7 and 0x18 for the gripper; command IDs are"
                " 0x01..0x08."
            )

        stuck_rows = [r for r in rows if r[6] > 0.2 and r[5] <= 0.2]
        if stuck_rows:
            print(
                "\nWARNING: " + ", ".join(r[0] for r in stuck_rows) + " reported fresh state"
                f" frames the whole time and still did not move while the command swept past"
                f" {self.stale_travel:g} rad. This is NOT a feedback problem -- the motor was"
                " talking and what it said was that the joint stayed put. That is a real tracking"
                " failure: too little torque to break static friction at these gains, a mechanical"
                " obstruction, or a motor that has cut its own output (check the status column in"
                " the motor-state table below -- a fault there is the answer)."
                "\n  Confirm one joint at a time: python safe_probe.py --side <side> --joint <n>"
                " --step 0.05"
            )

        # Only joints with somewhere to go can demonstrate a tracking error, so the headline
        # numbers come from those. MOVED_ENOUGH is a couple of times the steady-state band this arm
        # settles into -- below it, peak error and residual are the same quantity wearing two hats.
        MOVED_ENOUGH = 0.1
        # Stuck channels are excluded alongside stale ones. Their residual is the whole commanded
        # travel, so leaving them in makes them the "worst" every time and gets that number
        # printed under a line calling it expected steady-state error, which it is not -- they
        # have their own warning above.
        moved = [r for r in rows if r[1] >= MOVED_ENOUGH and r[5] <= 0.2 and r[6] <= 0.2]
        if not moved:
            print(f"\nNo joint travelled more than {MOVED_ENOUGH:g} rad on a channel that stayed"
                  " fresh and followed its command, so this run does not say much about tracking"
                  " quality.")
            return
        worst_peak = max(moved, key=lambda r: r[2])
        worst_res = max(moved, key=lambda r: r[3])
        print(f"\n  Of the {len(moved)} joint(s) that travelled >= {MOVED_ENOUGH:g} rad on fresh feedback:")
        print(f"    worst peak error: {worst_peak[0]} at {worst_peak[2]:.4f} rad"
              f" over {worst_peak[1]:.3f} rad of travel")
        print(f"    worst residual:   {worst_res[0]} at {worst_res[3]:.4f} rad at the hold"
              " (steady-state error at these gains; expected, not a fault)")

    def save_plot(self, out_path: str):
        """Per-joint commanded-vs-actual time series.

        Uses a minimum y-axis half-range per joint category, same as replay_hf_sim_episode.py's
        save_tracking_plot(), so noise or a small steady-state gap on an otherwise-stationary
        joint can't visually read as a large tracking failure.
        """
        if not self.n_samples:
            print("No tracking samples collected -- skipping plot.")
            return

        ARM_AXIS_TOLERANCE = 0.1
        GRIPPER_AXIS_TOLERANCE = 0.005

        # Only segment boundaries are worth a rule; the first phase starts at t=0, which is the
        # axis edge and marks nothing.
        boundaries = [(t, label) for t, label in self.phase_marks if t > 0]

        ncols = 4
        nrows = (len(POS_KEYS) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

        for idx, name in enumerate(POS_KEYS):
            ax = axes[idx // ncols][idx % ncols]
            h = self.history[name]
            ax.plot(h["t"], h["target"], label="commanded", linewidth=1)
            ax.plot(h["t"], h["actual"], label="actual", linewidth=1, linestyle="--")
            # Without this, a frozen feedback channel draws a clean flat "actual" line that reads
            # as a confidently-measured tracking failure. Marking the samples that never refreshed
            # makes the difference visible instead of leaving it to be inferred from the summary.
            stale_t = [t for t, st in zip(h["t"], h["stale"]) if st]
            stale_v = [v for v, st in zip(h["actual"], h["stale"]) if st]
            if stale_t:
                ax.plot(stale_t, stale_v, linestyle="none", marker=".", markersize=3,
                        color="tab:red", label="no state frame")
            # Same flat line, opposite meaning: here the motor WAS reporting and the joint stayed
            # put anyway. Drawn differently so the plot cannot be read as the comms failure above.
            stuck_t = [t for t, st in zip(h["t"], h["stuck"]) if st]
            stuck_v = [v for v, st in zip(h["actual"], h["stuck"]) if st]
            if stuck_t:
                ax.plot(stuck_t, stuck_v, linestyle="none", marker="x", markersize=3,
                        color="tab:purple", label="fresh, not moving")
            for mark_t, _ in boundaries:
                ax.axvline(mark_t, color="grey", linewidth=0.6, linestyle=":")
            ax.set_title(name, fontsize=9)
            ax.legend(fontsize=7)
            ax.set_xlabel("s", fontsize=7)
            ax.set_ylabel("rad", fontsize=7)

            all_v = h["target"] + h["actual"]
            if all_v:
                tolerance = GRIPPER_AXIS_TOLERANCE if name.endswith("8.pos") else ARM_AXIS_TOLERANCE
                data_min, data_max = min(all_v), max(all_v)
                center = (data_min + data_max) / 2
                half_range = max((data_max - data_min) / 2 * 1.1, tolerance)
                ax.set_ylim(center - half_range, center + half_range)

        for idx in range(len(POS_KEYS), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        marks = ", ".join(f"{label} from {t:.1f}s" for t, label in boundaries)
        fig.suptitle("reset_to_rest_pose: commanded vs. actual joint positions"
                     + (f"  (dotted: {marks})" if marks else ""))
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        print(f"Saved commanded-vs-actual tracking plot to {out_path}")


def print_bus_evidence(evidence: dict | None):
    """Print what the CAN controller and the transmitters themselves said about the wire.

    This is the reading that decides whether a silent channel was struggling electrically or
    simply stopped, and neither half of it is visible anywhere else: openarm_can never looks at
    the ESI bit, and `ip -s link`'s bus-errors column stays at zero unless berr-reporting was
    enabled on the interface, which nothing here does.
    """
    if not evidence:
        return
    esi = {k: v for k, v in sorted(evidence.get("esi", {}).items()) if v}
    errors = evidence.get("errors", {})
    print("\nBus evidence:")
    if errors:
        print("  CAN controller error frames: "
              + ", ".join(f"class {cls:#x} x{n}" for cls, n in sorted(errors.items())))
    else:
        print("  CAN controller raised no error frames -- the adapter saw a clean bus.")
    if esi:
        print("  Error-passive (ESI) frames: " + ", ".join(f"{k} x{v}" for k, v in esi.items())
              + "\n    Those transmitters had accumulated enough errors on the wire to stop"
              " being trusted, and a node that keeps accumulating goes bus-off and vanishes."
              " A channel that sets ESI and then goes silent went silent BECAUSE OF THE WIRE.")
    else:
        print("  No transmitter flagged itself error-passive (no ESI frames).")
    if not errors and not esi:
        print("    So a channel that went silent here was not fighting the bus and losing: it"
              " stopped sending while everything else on that wire stayed clean. That points at"
              " that motor's own supply or its branch of the harness, not at bus integrity.")


def print_motor_health(health: dict | None, suspect: list[str] | None = None):
    """Dump per-motor temperature/torque/velocity after a run, to follow up on a stale channel.

    "The feedback stopped refreshing" has two very different causes -- the read scheduling lost the
    frames (software, fixed with the --can-* knobs) or the motor stopped sending them (hardware) --
    and the numbers here are retained from the LAST frame each motor did send, which is exactly the
    moment worth looking at for a channel that then went quiet.

    The "status" column is the DM motor's OWN status/error nibble, read off the wire by the passive
    tap (can_monitor.py) -- data[0] >> 4 of each state frame, which openarm_can's decoder skips
    entirely. It is the only trustworthy statement about whether a motor is armed, and a fault code
    there (OVERLOAD, OVERCURRENT, MOS-OVERTEMP, ...) is the motor saying it cut its own output,
    which explains a joint that stops following while its feedback keeps arriving.

    is_enabled() is deliberately NOT printed any more. It is dead code in openarm_can:
    Motor::set_enabled() is defined and never called from anywhere in the library, so it returns
    False for all sixteen channels forever -- which is exactly what was seen on a run where five
    joints tracked their command to within 0.03 rad.
    """
    if not health:
        return

    print("\nMotor state (retained from each motor's LAST received frame):")
    print(f"  {'ch':5s} {'t_mos':>6s} {'t_rotor':>8s} {'torque':>8s} {'vel':>8s}"
          f" {'frames':>7s} {'status':>15s}")
    hot, faulted = [], []
    for ch, h in health.items():
        note = "  <-- feedback went quiet after this frame" if suspect and ch + ".pos" in suspect else ""
        frames = h.get("frames")
        print(f"  {ch:5s} {h['t_mos']:6d} {h['t_rotor']:8d} {h['torque']:8.3f}"
              f" {h['velocity']:8.3f} {('-' if frames is None else str(frames)):>7s}"
              f" {h.get('status_name', '?'):>15s}{note}")
        if max(h["t_mos"], h["t_rotor"]) >= 70:
            hot.append(f"{ch} ({max(h['t_mos'], h['t_rotor'])}C)")
        if h.get("faulted"):
            faulted.append(f"{ch} ({h.get('status_name')})")

    if faulted:
        print(f"\n  FAULTED: {', '.join(faulted)} -- these motors reported an error state and cut"
              " their own output. A joint that stops following while its feedback keeps arriving is"
              " explained here, and no gain change will fix it. Power-cycle the arm to clear the"
              " latch, then find out why it tripped before running again.")
    if hot:
        print(f"\n  Running hot: {', '.join(hot)}. Thermal protection is the usual reason a DM motor"
              " cuts out mid-run.")
    if suspect:
        print(
            "\n  A flagged channel's torque and velocity above are from the instant its feedback"
            " stopped, not from the end of the run. Small torque with the joint still short of"
            " its target means it was not pushing when it went quiet; near-peak torque means it"
            " was and could not hold. Neither says whether the motor kept running afterwards --"
            f" probe it directly: python safe_probe.py --side <side> --joint <n> --step 0.05"
        )


def reset_to_rest_pose(
    robot,
    calib: dict,
    *,
    target_action: dict | None = None,
    arm_speed: float = 0.3,
    gripper_speed: float = 1.5,
    max_delta: float = UNCONDITIONAL_MAX_DELTA,
    min_duration: float = 4.0,
    hold_duration: float = 2.0,
    assume_yes: bool = False,
    recorder: TrackingRecorder | None = None,
    rate_hz: float = 50.0,
) -> bool:
    """Drive `robot` to `target_action` (default: calibration's mapped sim rest pose) along a
    speed-limited ramp, then hold there so the settled error can be measured. Returns True if the
    move ran, False if approach_pose() refused it or the confirmation was declined.

    Pass the caller's already-known live target_action when available (e.g. sim's actual current
    pose in mirror_bridge.py) rather than relying on the default -- if sim itself has already moved
    away from its own rest pose (e.g. teleop started before the bridge finished connecting),
    resetting the real arm toward an assumed rest is pointless: it'll just create a new, different
    mismatch against sim's actual current pose instead of fixing the original one.

    The hold at the end is not cosmetic. These joints settle wherever gravity and their gains
    balance, so the pose the arm is actually left in is the held one, not the commanded one -- and
    that held pose is what the next script's startup handshake will read. Measuring it here is the
    difference between "the ramp completed" and "the arm is at the rest pose".
    """
    if target_action is None:
        target_action = rest_pose_action(calib)

    on_tick = recorder.on_tick if recorder is not None else None
    approached = approach_pose(
        robot,
        target_action,
        label="the sim rest pose",
        arm_speed=arm_speed,
        gripper_speed=gripper_speed,
        max_delta=max_delta,
        min_duration=min_duration,
        assume_yes=assume_yes,
        on_tick=on_tick,
        rate_hz=rate_hz,
    )
    if approached is None:
        return False

    if hold_duration > 0:
        print(f"Holding the rest pose for {hold_duration:.1f}s to measure settled error...")
        if recorder is not None:
            recorder.start_phase("hold")
        # A ramp from the target to itself is exactly a hold, and reuses the one send path
        # everything else here goes through rather than open-coding a second one.
        ramp_to(robot, target_action, target_action, hold_duration, rate_hz, on_tick=on_tick)

    print("Done -- arm should now be at rest pose.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", type=str, required=True, help="Path to calibration.json")
    parser.add_argument("--right-port", type=str, default="can0")
    parser.add_argument("--left-port", type=str, default="can1")
    parser.add_argument("--model-path", type=str, required=True, help="Path to openarm_description.urdf for gravity comp")
    parser.add_argument(
        "--max-delta", type=float, default=UNCONDITIONAL_MAX_DELTA,
        help=f"rad; approach_pose() refuses to move a joint further than this in one go."
        f" Default {UNCONDITIONAL_MAX_DELTA:.1f} is deliberately beyond any distance a pair of"
        " plausible readings can produce, so this script goes to the rest pose from wherever the"
        " arm is -- which is the point of it. Lower it (e.g. 1.8, the other scripts' value) to put"
        " a real distance limit back.",
    )
    parser.add_argument(
        "--arm-speed", type=float, default=0.3,
        help="rad/s; per-joint speed cap for the approach ramp. The furthest-travelling joint sets"
        " the duration and the rest arrive with it.",
    )
    parser.add_argument("--gripper-speed", type=float, default=1.5, help="rad/s; separate cap for the gripper channels")
    parser.add_argument(
        "--ramp-duration", type=float, default=4.0,
        help="seconds; MINIMUM ramp time. The actual duration is whatever --arm-speed requires for"
        " the furthest joint, so this only matters for a short move.",
    )
    parser.add_argument(
        "--hold-duration", type=float, default=2.0,
        help="seconds to keep commanding the rest pose after the ramp, so the settled (steady-state)"
        " error is what gets measured and plotted rather than the error mid-motion. 0 disables.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip the typed YES confirmation before moving")
    # These control POLLING. They do not control how long a read waits -- recv_all() returns in
    # 0.04-0.16 ms whatever timeout it is given on this build (measured 2026-08-26), so N rounds
    # of it cover a couple of milliseconds of spinning rather than of patience, and the motors
    # answer 0.26-1.02 ms after the request in ascending CAN-id order. That race is what froze
    # J5-J8's feedback for seconds at a time. --can-deadline-us below is the knob that fixes it;
    # these three are kept because they still decide how cheaply the socket gets emptied.
    parser.add_argument(
        "--can-recv-rounds", type=int, default=16,
        help="recv_all() calls per position read, per arm. Enough must run to empty the socket"
        " every cycle or the backlog drifts and read freshness drifts with it.",
    )
    parser.add_argument(
        "--can-first-timeout-us", type=int, default=2_000,
        help="Microseconds the FIRST recv_all() waits -- the only round with a refresh_all()"
        " response outstanding, so the one that should be patient.",
    )
    parser.add_argument(
        "--can-mop-timeout-us", type=int, default=200,
        help="Microseconds per mop-up round. Paid in full whenever the buffer is already empty,"
        " so it must stay small.",
    )
    parser.add_argument(
        "--can-deadline-us", type=int, default=8_000,
        help="Microseconds a read may WAIT for the motors to answer. This is the knob that"
        " matters: recv_all()'s own timeout does not produce a wait on this build, so the"
        " --can-recv-rounds above buy polling, not patience. A read returns as soon as all"
        " sixteen channels have answered, so this ceiling is only paid when one really is"
        " missing.",
    )
    parser.add_argument(
        "--can-retry-us", type=int, default=3_000,
        help="Microseconds before a still-silent channel's arm is asked again within one read."
        " Pass 0 to disable re-asking. Worth one run at 0 whenever a channel goes quiet for a"
        " long stretch: a re-ask is extra traffic, and if it ever outran a motor's reply it"
        " would sustain the very outage it is trying to fix. Same numbers with and without it"
        " means the outage is not ours.",
    )
    parser.add_argument(
        "--plot", type=str, default="reset_tracking.png",
        help="Path for the per-joint commanded-vs-actual PNG",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not write the tracking plot")
    parser.add_argument(
        "--rate-hz", type=float, default=50.0,
        help="Command frames per second per joint during the ramp. Lower it to test CAN"
        " contention: every tick fires 8 commands back-to-back and all 8 motors then arbitrate to"
        " reply in ascending ID order, so the highest IDs (J5-J8) are the ones whose pending"
        " replies get superseded if the next burst arrives first. Fewer bursts per second gives"
        " them room. Also slows the ramp -- duration is fixed, so this only changes granularity.",
    )
    parser.add_argument(
        "--stale-travel", type=float, default=0.02,
        help="rad; a channel is called stale once the COMMAND has swept this far while its reading"
        " stayed bit-identical. Roughly the steady-state band this arm settles into, so a joint"
        " that is merely holding still is not mistaken for a channel whose feedback died.",
    )
    parser.add_argument(
        "--plot-hz", type=float, default=20.0,
        help="Target sampling rate for tracking. Each sample is a CAN read costing tens of ms, so"
        " sampling faster than this mostly just stretches the ramp in wall-clock. 0 disables"
        " sampling entirely (implies --no-plot).",
    )
    args = parser.parse_args()

    calib = load_calibration(args.calibration)

    robot_cfg = OpenArmFollowerConfig(
        right_port=args.right_port,
        left_port=args.left_port,
        enable_fd=True,
        model_path=args.model_path,
        recv_rounds=args.can_recv_rounds,
        recv_first_timeout_us=args.can_first_timeout_us,
        recv_mop_timeout_us=args.can_mop_timeout_us,
        recv_deadline_us=args.can_deadline_us,
        recv_retry_us=args.can_retry_us,
    )
    robot = OpenArmFollower(robot_cfg)
    robot.connect()
    check_arms_not_crossed(robot, calib)

    recorder = None
    if args.plot_hz > 0:
        ramp_rate_hz = args.rate_hz
        recorder = TrackingRecorder(robot, every=max(1, round(ramp_rate_hz / args.plot_hz)),
                                    stale_travel=args.stale_travel)
        recorder.start_phase("ramp")

    hold_start = None
    try:
        moved = reset_to_rest_pose(
            robot,
            calib,
            arm_speed=args.arm_speed,
            gripper_speed=args.gripper_speed,
            max_delta=args.max_delta,
            min_duration=args.ramp_duration,
            hold_duration=args.hold_duration,
            assume_yes=args.yes,
            recorder=recorder,
            rate_hz=args.rate_hz,
        )
        if moved and recorder is not None:
            hold_start = next((t for t, label in recorder.phase_marks if label == "hold"), None)
    except KeyboardInterrupt:
        print("\nInterrupted. Disabling motors.")
    finally:
        # Sampled BEFORE disconnect (it reads retained Motor state, which teardown may drop) but
        # printed after, so nothing delays de-energising the arm for the sake of a diagnostic.
        try:
            health = robot.get_motor_health()
            evidence = robot.get_bus_evidence()
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] could not read motor health ({type(e).__name__}: {e})")
            health, evidence = None, None
        try:
            robot.disconnect()
        except Exception:
            print("WARNING: error during disconnect -- verify motors are physically de-energized.")
        print("Robot disconnected.")

    # After disconnect: whatever was sampled is worth reporting even if the run was interrupted or
    # refused, and nothing here touches the robot.
    if recorder is not None:
        recorder.print_summary(hold_start)
        recorder.print_outages()
        print_bus_evidence(evidence)
        print_motor_health(health, recorder.stale_channels())
        stuck = recorder.stuck_channels()
        if stuck:
            print(f"\n  Fresh-but-not-following: {', '.join(stuck)}. Their status above is the next"
                  " thing to read; if it says 'enabled', the motor was armed and simply could not"
                  " move the joint at these gains.")
        if not args.no_plot:
            recorder.save_plot(args.plot)
    else:
        print_bus_evidence(evidence)
        print_motor_health(health)


if __name__ == "__main__":
    main()
