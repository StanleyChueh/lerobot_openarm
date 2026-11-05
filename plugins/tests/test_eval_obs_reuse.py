"""8e43f49 (background observations in robot_client.py) vs motor telemetry: how many robot reads -- i.e. telemetry
rows, one per get_observation() -- each control tick makes, and from which thread. Mock robot, no server, no CAN.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_eval_obs_reuse.py

Uses the real _background_observation_client_class and eval_recording.client_class on a fake lerobot RobotClient
base, in the order run_episodes composes them, and drives the per-tick calls lerobot's control loop makes
(control_loop_action, then control_loop_observation). The NEXT exporter's handling of the extra-read case is
checked in test_next_export.py (N15).
"""
import logging
import queue
import sys
import threading
import time
import types

from lerobot_robot_openarm_umeow import eval_recording, robot_client

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


class FakeRobot:
    def __init__(self):
        self.reads, self.actions, self.lock = [], [], threading.Lock()

    def get_observation(self):
        with self.lock:
            self.reads.append((time.perf_counter(), threading.get_ident()))
        return {"RJ1.pos": float(len(self.reads))}

    def send_action(self, action):
        with self.lock:
            self.actions.append((time.perf_counter(), threading.get_ident()))


class FakeBase:  # the attributes lerobot's RobotClient gives the subclasses under test
    action_s = 0.0

    def __init__(self, config):
        self.config = config
        self.robot = FakeRobot()
        self.latest_action_lock, self.latest_action = threading.Lock(), 3
        self.action_queue_lock, self.action_queue = threading.Lock(), queue.Queue()
        self.must_go = threading.Event()
        self.logger = logging.getLogger("fake_client")
        self.sent, self.sender_threads = [], set()

    def control_loop_action(self, verbose=False):
        time.sleep(self.action_s)
        self.robot.send_action({"RJ1.pos": 0.0})
        return {}

    def send_observation(self, obs):
        self.sender_threads.add(threading.get_ident())
        self.sent.append(obs)


class TimedObservation:
    def __init__(self, timestamp, observation, timestep):
        self.timestamp, self.observation, self.timestep, self.must_go = timestamp, observation, timestep, False

    def get_timestep(self):
        return self.timestep


rc = types.SimpleNamespace(RobotClient=FakeBase, TimedObservation=TimedObservation)
Client = eval_recording.client_class(robot_client._background_observation_client_class(rc))
DT = 1 / 30


class Recorder:
    def __init__(self):
        self.frames = 0

    def add(self, observation, action):
        self.frames += 1


def drive(recorder, action_s, ticks=30):
    c = Client(types.SimpleNamespace(environment_dt=DT))
    c.recorder = recorder
    c.action_s = action_s
    main = threading.get_ident()
    per_tick = []
    for _ in range(ticks):
        n0 = len(c.robot.reads)
        c.control_loop_action()
        c.control_loop_observation("task")
        per_tick.append(len(c.robot.reads) - n0)
    time.sleep(0.1)
    c.drop_observations()
    return c, per_tick, main


c, per_tick, main = drive(Recorder(), 0.0)
check("E1 recording + background: one robot read (telemetry row) per tick; the policy reuses the recorder's",
      per_tick == [1] * 30, f"{per_tick[:6]}...")
check("E1 every robot read and every action on the control thread; the sender thread never touches the robot",
      {tid for _, tid in c.robot.reads} == {main} and {tid for _, tid in c.robot.actions} == {main}
      and main not in c.sender_threads and c.sent)
check("E1 the recorder's read precedes the tick's action (row k = the state action k is sent from)",
      all(r[0] < a[0] for r, a in zip(c.robot.reads, c.robot.actions)))
c, per_tick, _ = drive(Recorder(), DT * 1.2, ticks=6)
check("E2 an action that takes longer than environment_dt -> a SECOND read that tick (two telemetry rows, the "
      "second a few ms after the action; NEXT export splits/rejects it, test_next_export N15)",
      per_tick == [2] * 6, f"{per_tick}")
c, per_tick, _ = drive(None, 0.0)
check("E3 no recording: control_loop_observation reads the robot itself (as before 8e43f49)", per_tick == [1] * 30)

print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
