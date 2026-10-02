#!/usr/bin/env python
"""Convert replay_hf_sim_episode_realgrip.py --log-csv recordings into the traces.npz that IsaacLab's
scripts/tools/sysid/fit_sim_actuators.py reads.

The replay logs, per tick and per channel, the target, the COMMANDED position (after the speed clamp) and
the ACTUAL position read right after the command was sent -- all in raw MOTOR units. The fit works in the
sim convention (radians, grippers 0..0.044), so both are mapped through calibration.json with the same
inverse the real-episode recorder uses (sim_bridge_common.motor_action_to_sim_joints).

Alignment: the actual read of row k happens right after command k is sent, so it is (almost entirely) the
response to command k-1 -- the same one-tick alignment as the LeRobot-dataset traces and the deploy
rollout traces. The replay loop's period jitters (diagnostic reads slow it), so each run is resampled to
a uniform --hz grid: command by holding the previous value (it is a zero-order-hold signal), measurement
by linear interpolation.

Several CSVs become several segments of ONE npz -- use that to merge runs at different speed caps.

Usage:
  python sysid_csv_to_trace.py --calibration calibration.json --out traces_probe.npz \\
      --csv run_cap1.0.csv run_cap1.5.csv run_cap2.0.csv [--hz 30]
"""

import argparse
import csv

import numpy as np

from sim_bridge_common import load_calibration, motor_action_to_sim_joints

NAMES = [f"LJ{i}.pos" for i in range(1, 9)] + [f"RJ{i}.pos" for i in range(1, 9)]
SIM_NAME = {
    **{f"{p}J{i}.pos": f"openarm_{s}_joint{i}" for p, s in (("L", "left"), ("R", "right")) for i in range(1, 8)},
    "LJ8.pos": "openarm_left_finger_joint1",
    "RJ8.pos": "openarm_right_finger_joint1",
}


def read_csv(path: str):
    """-> t (N,), commanded (N,16), actual (N,16) in motor units; rows with a failed read are dropped."""
    t, cmd, act = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                a = [float(row[f"{k}_actual"]) for k in NAMES]
                c = [float(row[f"{k}_commanded"]) for k in NAMES]
            except (TypeError, ValueError):
                continue   # blank actual = the state read failed on that tick
            t.append(float(row["t"]))
            cmd.append(c)
            act.append(a)
    return np.array(t), np.array(cmd), np.array(act)


def to_sim(motor: np.ndarray, calib: dict) -> np.ndarray:
    out = np.empty_like(motor)
    for i in range(len(motor)):
        sim = motor_action_to_sim_joints(dict(zip(NAMES, motor[i])), calib)
        out[i] = [sim[SIM_NAME[k]] for k in NAMES]
    return out


def resample(t, cmd, meas, hz):
    grid = np.arange(t[0], t[-1], 1.0 / hz)
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)
    cmd_g = cmd[idx]                                              # zero-order hold
    meas_g = np.stack([np.interp(grid, t, meas[:, j]) for j in range(meas.shape[1])], axis=1)
    return cmd_g, meas_g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=30.0, help="Uniform rate to resample to; must equal the sim env's rate.")
    args = ap.parse_args()

    calib = load_calibration(args.calibration)
    out = {"names": np.array(NAMES), "dt": 1.0 / args.hz, "cap": float("nan")}
    for n, path in enumerate(args.csv):
        t, cmd_m, act_m = read_csv(path)
        if len(t) < 30:
            print(f"[SKIP] {path}: only {len(t)} usable rows")
            continue
        period = np.diff(t)
        cmd, meas = resample(t, to_sim(cmd_m, calib), to_sim(act_m, calib), args.hz)
        out[f"ep{n:03d}_cmd"], out[f"ep{n:03d}_meas"] = cmd, meas
        speed = np.abs(np.diff(cmd[:, [j for j in range(16) if j not in (7, 15)]], axis=0)).max() * args.hz
        print(f"[OK] {path}: {len(t)} rows, source period median {1000 * np.median(period):.1f} ms "
              f"(p99 {1000 * np.percentile(period, 99):.1f}), resampled to {len(cmd)} ticks @ {args.hz:g} Hz, "
              f"peak commanded joint speed {speed:.2f} rad/s")
    np.savez_compressed(args.out, **out)
    print(f"[OUT] {args.out}")


if __name__ == "__main__":
    main()
