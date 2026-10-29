"""Writes the committed mock fixture plugins/tests/fixtures/openarm_telemetry_1/: one telemetry session produced by
the PRODUCTION writer (MotorTelemetry.for_robot, snapshot, close) on the fake DM bus, plus the LeRobot-shaped
dataset a lerobot-record run would have saved alongside it. No hardware, no CAN socket, no network.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/make_telemetry_fixture.py <out dir>

The session follows a real recording's shape, at 30 Hz on the policy clock:

  connect read -> start-pose ramp (50 Hz commands, no reads) -> reset loop with ONE failed snapshot
  -> episode 0 (45 frames, both arms moving) -> 0.4 s pause (episode save) -> reset loop
  -> episode 1 (45 frames): RJ5's replies lost on 2 ticks (stale), a 6-tick safety hold (sent frozen while desired
     moves), the right gripper squeezing -> return ramp -> disable frames -> final row

expected.json lists what a consumer must find (alignment, stale/hold/failed rows, analyzer verdicts). Timestamps
differ per run; everything in expected.json does not.
"""
import json
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import openarm_can as oa
import pyarrow as pa
import pyarrow.parquet as pq

import telemetry_fakes as tf
from telemetry_fakes import FakeWire

import lerobot_robot_openarm_umeow  # noqa: F401  (adds the repo root to sys.path)
from lerobot_robot_openarm_umeow import telemetry as tel

OUT = Path(sys.argv[1])
NAMES = [f"{s}J{i}.pos" for i in range(1, 9) for s in "RL"]  # the datasets' interleaved order
PERIOD = 1 / 30
SYSFS = OUT / ".sysfs"
TX0 = {"R": 120_000, "L": 118_500}  # counters as a long-up interface would show them
# What tools/force_aware/telemetry_check.py must say about the fixture, and why: G1 fails on the deliberately
# failed snapshot; S1/S4 report RJ5's two lost replies; G10 needs DIAG folders the fixture does not have.
INTENDED = {"G1": "FAIL", "G2": "PASS", "G3": "PASS", "G4": "PASS", "G5": "PASS", "G6": "PASS", "G7": "PASS",
            "G8": "PASS", "S5": "PASS", "G10": "UNKNOWN", "G11": "PASS", "G12": "PASS", "G13": "PASS", "S4": "FAIL",
            "B1": "PASS", "S1": "FAIL", "S2": "PASS", "S3": "PASS"}


def write_sysfs(wire):
    for side, port in (("R", "can0"), ("L", "can1")):
        d = SYSFS / port / "statistics"
        d.mkdir(parents=True, exist_ok=True)
        for k in ("rx_packets", "rx_errors", "tx_errors", "rx_dropped", "tx_dropped", "rx_over_errors"):
            (d / k).write_text("0\n")
        (d / "tx_packets").write_text(f"{TX0[side] + wire.tx[side]}\n")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    wire = FakeWire()
    tel.make_tap = lambda port, ids, fd=True: wire.taps["R" if port == "can0" else "L"]
    tel.SYSFS_NET = SYSFS
    tel._ip_link = lambda port: {"flags": ["NOARP", "UP", "LOWER_UP", "ECHO"], "linkinfo": {
        "info_kind": "can", "info_data": {"state": "ERROR-ACTIVE", "berr_counter": {"tx": 0, "rx": 0}},
        "info_xstats": {"restarts": 0, "bus_error": 0, "error_warning": 0, "error_passive": 0, "bus_off": 0}}}
    write_sysfs(wire)

    zero = {k: 0.0 for k in NAMES}
    robot = types.SimpleNamespace(
        name="openarm_umeow",
        config=types.SimpleNamespace(
            send_ids=tf.SEND[:7], recv_ids=tf.RECV[:7], gripper_motor_send_id=0x08, gripper_motor_recv_id=0x18,
            motor_types=tf.TYPES[:7], gripper_motor_type=oa.MotorType.DM4310, right_port="can0", left_port="can1",
            enable_fd=True, model_path="<fixture: no URDF>"),
        umeow_config=types.SimpleNamespace(gripper_squeeze_tau=1.5, max_joint_speed=1.0, calibration=None),
        KPs=[60.0, 50.0, 50.0, 60.0, 20.0, 40.0, 20.0, 3.0], KDs=[2.0, 2.0, 2.0, 2.5, 1.0, 1.2, 1.0, 0.3],
        _motor_channels=wire.channels, get_feedback_status=lambda: {},
        _last_sent=None, _last_desired=None, _fault=None, _motor_fault=None,
        gripper_squeeze_tau={"L": 0.0, "R": 0.0})
    t = tel.MotorTelemetry.for_robot(robot, OUT / "session_root")
    assert t is not None

    def observe():
        t0 = time.perf_counter()
        obs = wire.read()
        t.snapshot(obs, t0, time.perf_counter() - t0, robot)
        return obs

    def act(desired, hold=False):
        robot._last_desired = dict(desired)
        if not hold:
            robot._last_sent = dict(desired)
        tau = {"RJ8": robot.gripper_squeeze_tau["R"], "LJ8": robot.gripper_squeeze_tau["L"]}
        wire.command(robot._last_sent, {}, tau, 50.0, 2.0)

    def pose(k, ep):
        """A smooth trajectory: every joint moves, so each episode is unique (alignment is by value)."""
        x = {n: 0.15 * np.sin(0.07 * k + 0.4 * i + ep) for i, n in enumerate(NAMES)}
        x["RJ8.pos"] = x["LJ8.pos"] = 0.0
        return x

    expected = {"episodes": [], "failed_snapshot_ticks": [], "stale_joint_rows": {}, "hold_ticks": []}
    observe()  # connect-time read: no command yet, cmd_* NaN
    for k in range(25):  # start-pose ramp: commands only
        act({n: v * k / 24 for n, v in pose(0, 0).items()})
        time.sleep(1 / 50)
    states, actions, ep_index = [], [], []
    for ep in range(2):
        for k in range(12):  # reset loop
            if ep == 0 and k == 5:
                wire.motors["LJ3"].fail = True
            obs = observe()
            if ep == 0 and k == 5:
                wire.motors["LJ3"].fail = False
                expected["failed_snapshot_ticks"].append(t._tick - 1)
            act(pose(0, ep))
            time.sleep(PERIOD)
        start_tick = t._tick
        for k in range(45):
            if ep == 1:
                wire.drop = {"RJ5"} if k in (10, 11) else set()
                robot._fault = "fixture hold" if 20 <= k < 26 else None
                robot.gripper_squeeze_tau = {"L": 0.0, "R": -1.5 if k >= 30 else 0.0}
            obs = observe()
            if ep == 1 and 20 <= k < 26:
                expected["hold_ticks"].append(t._tick - 1)
            if ep == 1 and k == 11:
                expected["stale_joint_rows"]["RJ5"] = [t._tick - 1]
            desired = pose(k, ep)
            states.append([obs[n] for n in NAMES])
            actions.append([desired[n] for n in NAMES])
            ep_index.append(ep)
            act(desired, hold=robot._fault is not None)
            time.sleep(PERIOD)
        robot._fault = None
        wire.drop = set()
        expected["episodes"].append({"episode_index": ep, "frames": 45, "start_tick": start_tick})
        time.sleep(0.4)  # episode save: nothing reads or commands
    observe()  # approach_pose() reads the arm once before the return ramp
    for k in range(15):  # return ramp
        act(pose(0, 1))
        time.sleep(1 / 50)
    wire.ctrl("R", 0xFD)
    wire.ctrl("L", 0xFD)
    write_sysfs(wire)
    t.close(robot)

    session = OUT / "session"
    next((OUT / "session_root").iterdir()).rename(session)
    (OUT / "session_root").rmdir()
    m = json.loads((session / "meta.json").read_text())  # the only edit after the writer: say what this is
    m["mock_fixture"] = "fake DM bus (plugins/tests/make_telemetry_fixture.py); NOT hardware data"
    (session / "meta.json").write_text(json.dumps(m, indent=1, default=str))
    shutil.rmtree(SYSFS)

    # The LeRobot v3.0 dataset lerobot-record would have written (tabular part; no cameras in the fixture).
    ds = OUT / "dataset"
    (ds / "meta").mkdir(parents=True)
    (ds / "data/chunk-000").mkdir(parents=True)
    n = len(states)
    vec = {"dtype": "float32", "names": NAMES, "shape": [16]}
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}  # noqa: E731
    info = {"codebase_version": "v3.0", "robot_type": "openarm_umeow", "fps": 30, "total_episodes": 2,
            "total_frames": n, "total_tasks": 1, "chunks_size": 1000, "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 500, "splits": {"train": f"0:{2}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet", "video_path": None,
            "features": {"action": vec, "observation.state": vec, "timestamp": scalar("float32"),
                         "frame_index": scalar("int64"), "episode_index": scalar("int64"),
                         "index": scalar("int64"), "task_index": scalar("int64")}}
    (ds / "meta/info.json").write_text(json.dumps(info, indent=1))
    ep_index = np.asarray(ep_index)
    frame_index = np.concatenate([np.arange(45), np.arange(45)])
    table = pa.Table.from_arrays([
        pa.array(list(np.asarray(actions, np.float32)), pa.list_(pa.float32())),
        pa.array(list(np.asarray(states, np.float32)), pa.list_(pa.float32())),
        pa.array((frame_index / 30).astype(np.float32)), pa.array(frame_index, pa.int64()),
        pa.array(ep_index, pa.int64()), pa.array(np.arange(n), pa.int64()), pa.array(np.zeros(n, np.int64))],
        names=["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"])
    pq.write_table(table, ds / "data/chunk-000/file-000.parquet")

    meta = json.loads((session / "meta.json").read_text())
    expected.update(rows=meta["rows"], schema=meta["schema"], frames_schema=meta["frames_schema"],
                    columns=len(meta["columns"]), final_tick=meta["rows"] - 1,
                    disable_frames_final_row={"L": 8, "R": 8})
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/force_aware"))
    import telemetry_check

    report, _ = telemetry_check.run(session, ds)
    got = {i["id"]: i["status"] for i in report.items}
    assert got == INTENDED, f"analyzer disagrees with the fixture's intent: {got}"
    expected["analyzer"] = {"verdict": report.verdict(), "checks": got}
    (OUT / "expected.json").write_text(json.dumps(expected, indent=1))
    print(f"fixture written to {OUT}: {meta['rows']} rows, {sum(meta['taps'][s]['frames'] for s in 'LR')} frames")


if __name__ == "__main__":
    main()
