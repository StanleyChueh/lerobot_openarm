"""Measure what a 16-joint CAN position read actually costs, and why it is bimodal.

WHY THIS EXISTS
---------------
deploy_smolvla_pickup_jointspace.py's cycle time alternates between two values ~50ms apart --
e.g. 50.1ms / 59.1ms at --inference-hz 20, with the model phase alternating 8.4ms / 58.8ms. That
gap is not scheduler jitter (which is sub-millisecond and unimodal); it is exactly one expiry of
the timeout passed to recv_all() in OpenArmFollower._read_motor_positions_once:

    self.right_arm.refresh_all()
    self.left_arm.refresh_all()
    for _ in range(8):
        self.right_arm.recv_all(50_000)     # <- 50_000 us = 50 ms
        self.left_arm.recv_all(50_000)

The pybind signature is recv_all(self, first_timeout_us: int = 500): the timeout governs how long
it waits for the FIRST frame. With frames already buffered it returns quickly; on an empty buffer
it blocks for the whole 50 ms.

The policy is not the culprit, despite sitting inside the same timed section: the V11 checkpoint's
config has n_action_steps=50, so select_action() runs one forward pass and then serves the next 49
steps from its action queue. That would show up as one large spike every 50 steps, not as the 1:1
alternation actually observed.

Two things then interact. First, refresh_all() is called ONCE but recv_all() eight times, so
rounds 2-8 have no refresh responses of their own to collect. Second, OpenArmFollower.send_action
writes MIT commands and never reads the feedback frame each motor sends back, so every control
cycle leaves ~8 unread frames per arm per send (10 interpolation substeps => ~80/arm) sitting in
the kernel socket buffer. Rounds 2-8 therefore feed on that backlog, and the read is fast or slow
depending on whether the backlog happens to run dry inside one of the rounds.

FIRST RUN, 2026-08-21 (30 samples/setting, arm enabled and still)
-----------------------------------------------------------------
Cost came out as EXACTLY rounds x 2 arms x timeout at every setting but one:

    rounds=1 first=50ms -> 100.2ms      rounds=8 first=50ms -> 801.0ms (the shipped setting)
    rounds=1 first=2ms  ->   4.1ms      rounds=8 first=0.5ms->   8.8ms
    rounds=1 first=0.5ms->   0.1ms  <-- the only early return in the whole sweep

So recv_all() blocks its full timeout essentially always here. That settles what the deploy loop
is doing: a read there costs 8-59ms, not 801ms, so ~all 8 rounds must be returning early on the
backlog send_action leaves behind -- and the 50ms alternation is the one round that occasionally
finds the buffer already dry.

What it does NOT settle is freshness. Every setting scored 0/30 implausible, but the plausibility
check only catches the +-12.5 rad glitch; it is blind to a read that returns RETAINED Motor state.

A first attempt at a freshness proxy -- counting reads not bit-identical to the previous one on a
still arm -- turned out to be worthless, and its 0% result should not be read as evidence of
staleness: DM position resolution is about 3.8e-4 rad (16 bits over the motor's +-12.5 rad range),
which is coarser than anything a limp, resting arm does mechanically. Identical consecutive values
are the EXPECTED outcome there for a perfectly fresh read. Freshness needs motion well above that
quantum, which is what --motion-check does: it backdrives the comparison by hand and reads the
candidate settings strictly alternately, so all of them see the same motion.

WHAT THIS MEASURES
------------------
With NO commands sent at all -- so no backlog exists -- how long does the read take at various
(rounds, first_timeout_us) settings, and does it still return plausible positions? That separates
the two candidate explanations cleanly:

Two columns matter. 'median' is what the read costs, which lands straight in the control cycle's
budget. 'moved' is the freshness proxy: the fraction of reads whose 16 values were not bit-
identical to the previous read. An enabled arm's encoders dither in their low bits, so a setting
that genuinely collects frames should score well above 0%, and one returning retained state should
score near 0%. It is a differential test -- the shipped setting is the baseline, and a candidate
is only considered if it is at least as fresh as that baseline. If the BASELINE scores ~0%, the
test is inconclusive by construction and the script says so instead of recommending anything.

The sweep also includes split-timeout candidates (first=50ms, mop=0.5ms and similar). Those keep
the generous wait on round 1 -- the only round with a refresh_all() response outstanding -- while
bounding rounds 2-8, which in the deploy loop exist only to mop up unread send_action feedback.
Freshness of round 1 is untouched by construction, so a split setting cannot be less fresh than
the baseline for any reason the 50ms value was originally introduced to fix.

SAFETY: connect() calls enable_all() on both arms, so the motors end up ENABLED but UNCOMMANDED
for the duration -- the same state every other script in this repo leaves them in between connect
and its first send (e.g. while deploy waits at its "Type YES" prompt). Nothing here ever calls
send_action, so the arm is never commanded to move, but support it / keep it in a safe resting
position anyway, and keep emergency_disable.py within reach.

Usage:
    python profile_can_read.py               # default sweep, 20 samples per setting
    python profile_can_read.py --samples 50 --yes
"""

import argparse
import statistics
import time

from robots.umeow_openarm_follower.config_openarm_follower import OpenArmFollowerConfig
from robots.umeow_openarm_follower.openarm_follower import OpenArmFollower

# Same URDF the deploy script loads -- send_action needs it for the gravity feedforward.
# Nothing here sends, but OpenArmFollower builds the pinocchio model at construction time.
URDF_PATH = "/home/csl/Stanley_ws/IsaacLab/source/isaaclab_assets/data/v1_camera_isaac/urdf/v1_camera.urdf"


def read_once(robot, rounds: int, first_us: int, mop_us: int = None) -> dict:
    """_read_motor_positions_once(), with its two magic numbers made into parameters.

    mop_us splits the rounds in two: round 1 is the one that can actually collect the responses
    to refresh_all(), so it gets first_us. Rounds 2..N have no refresh of their own outstanding --
    in the deploy loop they only ever mop up the feedback frames send_action left unread -- so
    they get mop_us and must never block. Pass mop_us=None to give every round first_us, which is
    what the shipped code does."""
    if mop_us is None:
        mop_us = first_us
    robot.right_arm.refresh_all()
    robot.left_arm.refresh_all()
    robot.right_arm.recv_all(first_us)
    robot.left_arm.recv_all(first_us)
    for _ in range(rounds - 1):
        robot.right_arm.recv_all(mop_us)
        robot.left_arm.recv_all(mop_us)

    pos = {}
    for i, motor in enumerate(robot.right_arm.get_arm().get_motors()):
        pos[f"RJ{i + 1}.pos"] = motor.get_position()
    pos["RJ8.pos"] = robot.right_arm.get_gripper().get_motor().get_position()
    for i, motor in enumerate(robot.left_arm.get_arm().get_motors()):
        pos[f"LJ{i + 1}.pos"] = motor.get_position()
    pos["LJ8.pos"] = robot.left_arm.get_gripper().get_motor().get_position()
    return pos


def parse_setting(spec: str):
    """"rounds:first_us:mop_us" -> (rounds, first_us, mop_us or None). mop_us 0 means "no split"."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected rounds:first_us:mop_us, got {spec!r}")
    rounds, first_us, mop_us = (int(p) for p in parts)
    return rounds, first_us, (mop_us or None)


def label(setting) -> str:
    rounds, first_us, mop_us = setting
    return f"rounds={rounds} first={first_us}us" + (f" mop={mop_us}us" if mop_us else "")


def motion_check(robot, settings, seconds: float) -> None:
    """Rank settings by freshness against REAL motion, reading them strictly alternately.

    The still-arm bit-identity test cannot do this. DM position resolution is about 3.8e-4 rad
    (16 bits over the motor's +-12.5 rad range), which is coarser than anything a limp, resting
    arm does mechanically -- so identical consecutive values are the expected result there whether
    the read is fresh or stale, and 0% "moved" says nothing. Backdriving one joint by hand puts
    motion far above the quantum.

    Alternating the settings read-by-read is what makes the comparison fair: both see the same
    hand motion over the same interval, so a setting whose positions lag or plateau is showing its
    own staleness, not a difference in what the arm did.
    """
    print("\nMOTION CHECK -- the motors are enabled but uncommanded, so the arm is backdrivable.")
    print("Support its weight, then gently move ONE joint (an elbow is easiest) back and forth")
    print(f"for the next {seconds:.0f}s. Settings are read strictly alternately, so they all see")
    print("the same motion.\n")
    for st in settings:
        print(f"  - {label(st)}")
    if input("\nReady? Type GO: ").strip() != "GO":
        print("Not confirmed, skipping motion check.")
        return

    stats = {st: {"n": 0, "moved": 0, "path": 0.0, "prev": None} for st in settings}
    t_end = time.perf_counter() + seconds
    i = 0
    while time.perf_counter() < t_end:
        st = settings[i % len(settings)]
        i += 1
        pos = read_once(robot, *st)
        d = stats[st]
        d["n"] += 1
        if d["prev"] is not None:
            delta = sum(abs(pos[k] - d["prev"][k]) for k in pos)
            d["path"] += delta
            if delta > 0:
                d["moved"] += 1
        d["prev"] = pos

    print(f"\n{'setting':>34} | {'reads':>6} {'moved':>7} {'path travelled':>15}")
    print("-" * 70)
    for st in settings:
        d = stats[st]
        pct = 100.0 * d["moved"] / max(1, d["n"] - 1)
        print(f"{label(st):>34} | {d['n']:>6} {pct:>6.0f}% {d['path']:>13.3f} rad")

    paths = [stats[st]["path"] for st in settings]
    best = max(paths)
    print()
    if best < 0.05:
        print("Almost no motion recorded -- the arm was not moved enough (or reads are not\n"
              "tracking at ANY setting). Re-run and move a joint through a clear arc.")
        return
    print("Path travelled should agree across settings to within the sampling difference, since\n"
          "they saw the same motion. A setting reporting markedly LESS path is dropping updates.")
    for st in settings:
        ratio = stats[st]["path"] / best
        verdict = "tracks" if ratio > 0.8 else ("LAGS" if ratio > 0.3 else "STALE")
        print(f"  {label(st):>34}: {ratio*100:>3.0f}% of the best path -> {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=20, help="reads per (rounds, timeout) setting")
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="recv_all() call counts to try (current code uses 8)")
    ap.add_argument("--timeouts-us", type=int, nargs="+", default=[500, 2_000, 5_000, 50_000],
                    help="first_timeout_us values to try (current code uses 50000; pybind default 500)")
    ap.add_argument("--settle", type=float, default=0.0,
                    help="seconds to sleep between reads; 0 reproduces the deploy loop's back-to-back reads")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--motion-check", action="store_true",
                    help="skip the cost sweep; instead rank settings for FRESHNESS while you "
                         "backdrive a joint by hand (the only test that can distinguish a fresh "
                         "read from a stale one -- see motion_check())")
    ap.add_argument("--motion-settings", type=parse_setting, nargs="+",
                    default=[(8, 50_000, None), (8, 50_000, 500), (8, 5_000, 200), (1, 500, None)],
                    metavar="ROUNDS:FIRST_US:MOP_US",
                    help="settings to compare in --motion-check (mop_us 0 = no split). Default "
                         "compares the shipped setting, two split candidates, and the cheapest "
                         "setting from the cost sweep.")
    ap.add_argument("--motion-seconds", type=float, default=15.0)
    args = ap.parse_args()

    print(__doc__.split("Usage:")[0].strip()[-600:])
    if not args.yes:
        if input("\nConnecting ENABLES both arms (no motion is ever commanded). Type YES: ").strip() != "YES":
            print("Not confirmed, exiting.")
            return

    robot = OpenArmFollower(OpenArmFollowerConfig(
        right_port="can0", left_port="can1", enable_fd=True, model_path=URDF_PATH, cameras={},
    ))
    robot.connect()

    try:
        # Warm-up: the very first read after connect is not representative (it is also where the
        # transient implausible readings documented in openarm_follower.py cluster).
        for _ in range(5):
            read_once(robot, 8, 50_000)

        if args.motion_check:
            motion_check(robot, args.motion_settings, args.motion_seconds)
            return

        print(f"\n{args.samples} reads per setting, no send_action anywhere (so no feedback backlog).")
        print("'moved' = reads whose 16 values were NOT bit-identical to the previous read. On a")
        print("STILL arm this is expected to be 0% at every setting, fresh or stale: DM position")
        print("resolution is ~3.8e-4 rad, coarser than anything a limp resting arm does, so it")
        print("cannot separate the two. Use --motion-check for that. Treat this column here only")
        print("as a check that the arm was in fact still while the costs below were measured.")
        print(f"\n{'rounds':>6} {'first':>9} {'mop':>8} | {'min':>8} {'median':>8} {'max':>8} "
              f"| {'implaus':>8} {'moved':>7}")
        print("-" * 80)

        settings = [(r, t, None) for r in args.rounds for t in args.timeouts_us]
        # The candidate fix: keep the generous wait on the only round that can collect a fresh
        # response, bound every mop-up round so a dry buffer costs microseconds, not 50ms.
        settings += [(8, 50_000, 500), (8, 50_000, 200), (8, 5_000, 200), (4, 5_000, 200)]

        results = []
        for rounds, first_us, mop_us in settings:
            times, bad, moved, prev = [], 0, 0, None
            for _ in range(args.samples):
                t0 = time.perf_counter()
                pos = read_once(robot, rounds, first_us, mop_us)
                times.append((time.perf_counter() - t0) * 1e3)
                if OpenArmFollower._find_implausible_key(pos) is not None:
                    bad += 1
                if prev is not None and any(pos[k] != prev[k] for k in pos):
                    moved += 1
                prev = pos
                if args.settle:
                    time.sleep(args.settle)
            med = statistics.median(times)
            pct = 100.0 * moved / max(1, args.samples - 1)
            results.append((rounds, first_us, mop_us, med, bad, pct))
            print(f"{rounds:>6} {first_us/1000:>7.1f}ms {(str(mop_us/1000)+'ms') if mop_us else '-':>8} "
                  f"| {min(times):>7.1f}ms {med:>7.1f}ms {max(times):>7.1f}ms "
                  f"| {bad:>3}/{args.samples:<4} {pct:>6.0f}%")

        cur = next((r for r in results if r[0] == 8 and r[1] == 50_000 and r[2] is None), None)
        print()
        if cur is not None:
            print(f"Baseline, the shipped setting (rounds=8, first_timeout_us=50000): "
                  f"{cur[3]:.1f}ms median. A read happens once per control cycle, so that lands "
                  f"directly\nin the cycle budget -- at --inference-hz 20 the whole budget is "
                  f"50.0ms.")
            cheap = min((r for r in results if r[4] == 0), key=lambda r: r[3])
            print(f"Cheapest setting in the sweep: rounds={cheap[0]} first={cheap[1]}us "
                  f"mop={cheap[2]}us -- {cheap[3]:.1f}ms median, "
                  f"{cur[3]/max(cheap[3], 1e-9):.0f}x cheaper.")
        print("\nCOST ONLY. Nothing above establishes that any of these settings still reads FRESH\n"
              "positions -- run --motion-check before changing _read_motor_positions_once.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
