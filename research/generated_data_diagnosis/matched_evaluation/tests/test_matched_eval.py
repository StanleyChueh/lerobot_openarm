"""Unit tests of the matched evaluation (robot_client.py --episode_steps / --poses / --results_dir).

    cd ~/Stanley_ws/lerobot_openarm
    env -u PYTHONPATH -u LD_LIBRARY_PATH .venv-v2/bin/python -m pytest -q research/generated_data_diagnosis/matched_evaluation/tests

CPU only, no simulator, no policy: the simulator and the policy server are replaced by fakes; the control loop is
lerobot's own (the plugin's vendored async_inference.robot_client) wherever the test is about it.
"""

import ast
import json
import re
import threading
import types
from pathlib import Path
from queue import Queue

import pytest

from lerobot_robot_openarm_umeow import matched_eval as me
from lerobot_robot_openarm_umeow import robot_client as rcl

ROOT = Path(__file__).resolve().parents[4]  # lerobot_openarm
POSES = ROOT / "research/generated_data_diagnosis/protocol/eval_poses_100.json"
TASK_MODES = ROOT.parent / ("IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/"
                            "config/openarm/openarm_task_modes.py")
BUDGET = 330
HOLD = 30
REST_Z = 0.38


# -- fakes -----------------------------------------------------------------------------------------------------
def truth(stage=0, right=False, left=False, confirmed=False, z=REST_Z, x=0.25, y=0.0, grip=0, lifted=False):
    return {"can_pose": [x, y, z, 1.0, 0.0, 0.0, 0.0], "can_rest_z": REST_Z, "put_down_margin": 0.005,
            "stage": stage, "receiver_grip_steps": grip, "hold_steps_required": HOLD, "was_lifted": lifted,
            "right_holding_since_pickup": right, "right_holds": right, "left_holds": left,
            "left_grip_confirmed": confirmed}


def episode(right_grasp=40, pass_at=150, release=155, drop_at=None, n=BUDGET, success_tail=True):
    """Per-step records of a scripted hand-over, with the stage machine's own bookkeeping (as the task does it)."""
    steps, stage, grip = [], 0, 0
    for k in range(1, n + 1):
        right = right_grasp is not None and right_grasp <= k < (release or 10 ** 9)
        left = pass_at is not None and k >= pass_at and (drop_at is None or k < drop_at)
        z = REST_Z + (0.04 if (right or left) and k > (right_grasp or 0) + 10 else 0.0)
        if right and stage == 0:
            stage = 1
        if left and stage == 1:
            stage = 2
        carrying = stage >= 2 and left and not right and z > REST_Z + 0.005
        grip = grip + 1 if carrying else 0
        if stage == 2 and grip >= HOLD:
            stage = 3
        success = stage >= 3 and grip >= HOLD if success_tail else False
        steps.append({"sim_step": k, "t": k / 30, "success": success,
                      "handover": truth(stage, right, left, left, z, grip=grip, lifted=stage >= 1)})
    return steps


class FakeSim:
    """lerobot_sim_server.py's protocol: reset (optionally placing the can) and lockstep steps."""

    def __init__(self, success_from_step=None, misplace_m=0.0):
        self.steps = 0
        self.resets = []
        self.success_from_step = success_from_step  # global step from which the task's success term is True
        self.misplace = misplace_m
        self.can = [0.3, 0.0]

    def request(self, cmd, **kw):
        if cmd == "hello":
            return {"protocol": 1, "task": "fake", "task_mode": "handover", "control_hz": 30}
        if cmd == "reset":
            self.resets.append(kw.get("object_xy"))
            self.reset_joints = getattr(self, "reset_joints", []) + [kw.get("joints")]
            xy = kw.get("object_xy") or [0.3, 0.0]
            self.can = [xy[0] + self.misplace, xy[1]]
            return self._obs()
        if cmd == "step":
            self.steps += 1
            return self._obs()
        raise AssertionError(cmd)

    def _obs(self):
        ok = self.success_from_step is not None and self.steps >= self.success_from_step
        return {"joints": {}, "images": {}, "success": ok, "step": self.steps,
                "handover": truth(x=self.can[0], y=self.can[1], stage=3 if ok else 0, grip=HOLD if ok else 0)}


def fake_robot(sim):
    """The parts of OpenArmIsaac that a matched evaluation uses, with OpenArmIsaac's own methods."""
    from lerobot_robot_openarm_umeow.openarm_isaac import OpenArmIsaac

    robot = types.SimpleNamespace(_sim=sim, episode_success=False, steps_since_reset=0, sim_steps=0, scene_resets=0,
                                  last_reset=None, step_log=None, auto_scene_reset=True, _results=[], _latest=None,
                                  calib=None, _start_sim_joints={"j": 0.0}, home_error=0.0, prepared=[])
    robot.reset_scene = types.MethodType(OpenArmIsaac.reset_scene, robot)
    robot._home_error = lambda: robot.home_error  # the real one converts sim joints with calibration.json
    robot.get_observation = lambda: {"j.pos": 0.0}
    robot.prepare_return = lambda obs: robot.prepared.append(obs)

    def send(action=None):
        robot._latest = sim.request("step", joints={})
        robot.episode_success = robot.episode_success or bool(robot._latest["success"])
        robot.sim_steps += 1
        robot.steps_since_reset += 1
        if robot.step_log is not None:
            robot.step_log.append(me.step_record(robot._latest, [0.0] * 16, [0.0] * 16, 0.0))

    robot.step = send
    return robot


# -- poses -----------------------------------------------------------------------------------------------------
def _task_constants():
    """The spawn rule's constants, read from the task's source (importing it needs Isaac Sim)."""
    tree = ast.parse(TASK_MODES.read_text())
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("_RECORD_SPAWN_X_RANGE", "_RECORD_SPAWN_Y_RANGE", "_RECORD_SPAWN_GAUSSIAN", "CAN_SCALE"):
                consts[name] = ast.literal_eval(node.value)
    src = TASK_MODES.read_text()
    arm = re.search(r"_ARM_KEEP_OUT_BOXES = \((.*?)\n\)", src, re.S).group(1)
    hand = re.search(r"_HAND_KEEP_OUT_BOXES_AT_HOME = tuple\(.*?in \((.*?)\n    \)\n\)", src, re.S).group(1)
    growth = max(0.0, consts["CAN_SCALE"] * 0.036625 - 0.030)
    arm_boxes = [tuple(eval(t.replace("_CAN_GROWTH", repr(growth)))) for t in re.findall(r"\(([^()]*?)\),", arm)]
    hand_boxes = [tuple(float(v) for v in t.split(",")) for t in re.findall(r"\(([-0-9., ]+)\)", hand)]
    hand_boxes = [(x0 - growth, x1 + growth, y0 - growth, y1 + growth) for x0, x1, y0, y1 in hand_boxes]
    return consts, arm_boxes, hand_boxes


def test_poses_load_in_trial_order():
    poses, sha = me.load_poses(POSES)
    assert len(poses) == 100
    assert [p["trial"] for p in poses] == list(range(100))
    assert sha == me.load_poses(POSES)[1] and len(sha) == 64
    data = json.loads(POSES.read_text())
    counts = {}
    for p in poses:
        counts[p["region"]] = counts.get(p["region"], 0) + 1
    assert counts == data["region_counts"]


def test_poses_obey_the_tasks_spawn_rule():
    consts, arm_boxes, hand_boxes = _task_constants()
    assert len(arm_boxes) == 2 and len(hand_boxes) == 4
    (x0, x1), (y0, y1) = consts["_RECORD_SPAWN_X_RANGE"], consts["_RECORD_SPAWN_Y_RANGE"]
    for p in me.load_poses(POSES)[0]:
        assert x0 <= p["x"] <= x1 and y0 <= p["y"] <= y1, p
        for bx0, bx1, by0, by1 in arm_boxes + hand_boxes:
            assert not (bx0 <= p["x"] <= bx1 and by0 <= p["y"] <= by1), (p, (bx0, bx1, by0, by1))


def test_pose_file_out_of_order_is_refused(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"poses": [{"trial": 1, "x": 0.2, "y": 0.0}]}))
    with pytest.raises(SystemExit):
        me.load_poses(f)


# -- applying a pose -------------------------------------------------------------------------------------------
def test_reset_places_the_requested_pose_and_verifies_it():
    sim = FakeSim()
    robot = fake_robot(sim)
    robot.reset_scene(object_xy=(0.1402, -0.0003))
    assert sim.resets == [[0.1402, -0.0003]]
    assert robot.last_reset["requested_xy"] == [0.1402, -0.0003]
    assert robot.last_reset["error_xy_m"] < 1e-9
    assert robot.scene_resets == 1


def test_an_arm_left_off_home_is_put_back_with_the_reset_and_only_then():
    sim = FakeSim()
    robot = fake_robot(sim)
    robot.reset_scene(object_xy=(0.2, 0.0))  # at home (0 rad off): the robot is kept where it is
    assert sim.reset_joints == [None] and not robot.last_reset["rehomed"] and robot.prepared == []
    robot.home_error = 0.79  # a return blocked by a dropped can
    robot.reset_scene(object_xy=(0.21, 0.0))
    assert sim.reset_joints[-1] == {"j": 0.0} and robot.last_reset["rehomed"]
    assert robot.last_reset["home_error_rad_before"] == 0.79 and len(robot.prepared) == 1
    robot.home_error = 0.05  # within HOME_TOL_RAD: not touched
    robot.reset_scene(object_xy=(0.22, 0.0))
    assert sim.reset_joints[-1] is None and not robot.last_reset["rehomed"]


def test_a_pose_that_is_not_applied_is_reported_not_replaced():
    robot = fake_robot(FakeSim(misplace_m=0.003))
    with pytest.raises(me.PoseNotApplied, match="mm from the requested"):
        robot.reset_scene(object_xy=(0.2, 0.0))


def test_check_reset_rejects_tilt_height_stage_and_a_server_without_truth():
    t = truth(x=0.2, y=0.0)
    me.check_reset((0.2, 0.0), t)
    with pytest.raises(me.PoseNotApplied, match="tilted"):
        me.check_reset((0.2, 0.0), {**t, "can_pose": [0.2, 0.0, REST_Z, 0.99, 0.1, 0.0, 0.0]})
    with pytest.raises(me.PoseNotApplied, match="resting height"):
        me.check_reset((0.2, 0.0), {**t, "can_pose": [0.2, 0.0, REST_Z + 0.01, 1, 0, 0, 0]})
    with pytest.raises(me.PoseNotApplied, match="stage"):
        me.check_reset((0.2, 0.0), {**t, "stage": 1})
    with pytest.raises(me.PoseNotApplied, match="does not report"):
        me.check_reset((0.2, 0.0), None)


# -- metrics ---------------------------------------------------------------------------------------------------
def test_stable_handover_held_to_the_end():
    m = me.trial_metrics(episode(), BUDGET)
    assert m["success_any"] and m["stable"] and m["holding_end"] and m["complete"]
    assert m["right_grasp_step"] == 40 and m["left_grasp_step"] == 150 and m["right_release_step"] == 155
    assert m["stable_completion_step"] == 155 + HOLD - 1 == m["first_success_step"] == m["stage3_step"]
    assert m["stable_matches_stage"]
    assert all(m[f"holding_at_{k}"] for k in (240, 270, 300, 330))
    assert m["outcome"] == "stable_held_to_end"


def test_stable_then_dropped_counts_for_metric_1_and_2_not_3():
    m = me.trial_metrics(episode(drop_at=250), BUDGET)
    assert m["success_any"] and m["stable"] and not m["holding_end"]
    assert m["holding_at_240"] and not m["holding_at_270"]
    assert not m["success_at_end"]  # the task's term is transient: false once the left hand lets go
    assert m["outcome"] == "stable_then_lost"


def test_pass_lost_within_the_hold_window_is_not_stable():
    m = me.trial_metrics(episode(drop_at=170), BUDGET)  # 15 steps of left-only hold < 30
    assert not m["success_any"] and not m["stable"] and m["max_stage"] == 2
    assert m["outcome"] == "pass_not_secured" and m["longest_carry_steps"] == 15


def test_no_grasp_and_no_pass():
    assert me.trial_metrics(episode(right_grasp=None, pass_at=None, release=None), BUDGET)["outcome"] == "no_right_grasp"
    m = me.trial_metrics(episode(pass_at=None, release=None), BUDGET)
    assert m["outcome"] == "right_grasp_no_pass" and m["left_grasp_step"] is None and m["right_release_step"] is None


def test_metric_2_is_recomputed_independently_of_the_stage_counter():
    steps = episode()
    for s in steps:  # a task that never reached stage 3 (e.g. a different counter) must show as a mismatch
        if s["handover"]["stage"] == 3:
            s["handover"]["stage"] = 2
    m = me.trial_metrics(steps, BUDGET)
    assert m["stable"] and m["stage3_step"] is None and not m["stable_matches_stage"]


def test_a_short_trial_is_not_complete_and_has_no_end_metric():
    m = me.trial_metrics(episode(n=200), BUDGET)
    assert not m["complete"] and m["holding_end"] is None and m["holding_at_240"] is None


# -- the step budget, on lerobot's own control loop -----------------------------------------------------------
def _loop_client(robot, actions_every=1):
    """lerobot's RobotClient (the plugin's vendored copy) + our episode and step-budget subclasses, built without
    a server: the queue is refilled by the test, the observation send is a no-op."""
    from lerobot_robot_openarm_umeow.policy_loading import require_async_inference

    require_async_inference()
    import lerobot.async_inference.robot_client as rc
    import torch
    from lerobot.async_inference.helpers import TimedAction

    cls = me.step_budget_client_class(rcl._episode_client_class(rc))
    c = cls.__new__(cls)
    c.robot = robot
    c.config = types.SimpleNamespace(environment_dt=0.0, aggregate_fn=None)
    c.shutdown_event = threading.Event()
    c.latest_action_lock, c.action_queue_lock = threading.Lock(), threading.Lock()
    c.latest_action, c.action_chunk_size, c._chunk_size_threshold = -1, 50, 0.5
    c.action_queue, c.action_queue_size = Queue(), []
    c.start_barrier = threading.Barrier(1)
    c.must_go = threading.Event()
    c._episode, c._thread_episode = 0, threading.local()
    c.stub = types.SimpleNamespace(Ready=lambda _: None)
    tick = [0]

    def control_loop_observation(task, verbose=False):  # instead of sending to a server: refill the queue
        tick[0] += 1
        if tick[0] % actions_every == 0:
            with c.action_queue_lock:
                base = c.action_queue.qsize() + max(c.latest_action, 0)
                for j in range(10):
                    c.action_queue.put(TimedAction(timestamp=0.0, timestep=base + j + 1, action=torch.zeros(2)))
        return {}

    c.control_loop_observation = control_loop_observation
    new_episode = c.new_episode

    def new_episode_one_thread(pb):  # the real one arms a 2-party barrier for the receiver thread; there is none
        new_episode(pb)
        c.start_barrier = threading.Barrier(1)

    c.new_episode = new_episode_one_thread
    robot.action_features = {"a": float, "b": float}
    robot.send_action = lambda action: robot.step()
    return c, TimedAction


@pytest.mark.parametrize("actions_every", [1, 7])
def test_exactly_the_budget_of_simulation_steps(actions_every):
    robot = fake_robot(FakeSim())
    c, _ = _loop_client(robot, actions_every)
    c.new_episode(types.SimpleNamespace(Empty=lambda: None))
    c.begin_budget(BUDGET)
    c.control_loop(task="t")
    assert robot.sim_steps == BUDGET == c.policy_steps()
    assert c.shutdown_event.is_set() and c.budget_spent()
    c.begin_budget(None)  # no budget: never ends the loop by itself
    assert not c.budget_spent()


def test_no_action_leaks_into_the_next_episode():
    import torch

    robot = fake_robot(FakeSim())
    c, TimedAction = _loop_client(robot)
    c.new_episode(types.SimpleNamespace(Empty=lambda: None))
    c.action_queue.put(TimedAction(timestamp=0.0, timestep=5, action=torch.zeros(2)))
    c.latest_action = 4
    c._thread_episode.episode = c._episode  # this thread is the old episode's receiver
    c.new_episode(types.SimpleNamespace(Empty=lambda: None))
    assert c.action_queue.empty() and c.latest_action == -1 and c.action_chunk_size == -1
    # the old receiver's late chunk is dropped, not queued for the new episode
    c._aggregate_action_queues([TimedAction(timestamp=0.0, timestep=1, action=torch.zeros(2))])
    assert c.action_queue.empty()
    c._thread_episode.episode = c._episode  # the new episode's receiver is accepted
    c._aggregate_action_queues([TimedAction(timestamp=0.0, timestep=1, action=torch.zeros(2))])
    assert c.action_queue.qsize() == 1


def test_the_policy_server_forgets_the_last_chunk_on_ready():
    """RTC history: our server patch clears its previous chunk whenever a client says Ready (new_episode)."""
    from lerobot_robot_openarm_umeow import policy_server as ps
    from lerobot_robot_openarm_umeow.policy_loading import require_async_inference

    require_async_inference()
    import lerobot.async_inference.policy_server as server_module

    ps._patch(server_module)
    server = server_module.PolicyServer.__new__(server_module.PolicyServer)
    server.shutdown_event = threading.Event()
    server._predicted_timesteps_lock = threading.Lock()
    server._openarm_prev = {"start": 7}
    server._openarm_latencies = [0.2]
    server._reset_server()
    assert server._openarm_prev is None and server._openarm_latencies == []
    assert server.observation_queue.empty() and server._predicted_timesteps == set()


# -- the return home cannot make a trial successful, and does not reset the scene ------------------------------
def _patched_return():
    import lerobot.rollout.strategies.core as core

    from lerobot_robot_openarm_umeow import openarm_umeow

    openarm_umeow._slow_down_rollout_returns(1e9)
    return core


def test_the_return_home_neither_resets_the_scene_nor_counts_for_the_trial(monkeypatch):
    core = _patched_return()
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(core, "precise_sleep", lambda s: None, raising=False)
    sim = FakeSim(success_from_step=BUDGET + 5)  # the task's term turns true only DURING the return
    robot = fake_robot(sim)
    robot.reset_scene(object_xy=(0.25, 0.0))
    robot.auto_scene_reset = False
    robot.step_log = []
    for _ in range(BUDGET):
        robot.step()
    steps, robot.step_log = robot.step_log, None
    m = me.trial_metrics(steps, BUDGET)

    wrapper = types.SimpleNamespace(hardware_failure=None, _robot=robot,
                                    get_observation=lambda: {"j.pos": 0.0}, send_action=lambda a: robot.step())
    core.RolloutStrategy.return_to_initial_position(
        types.SimpleNamespace(robot_wrapper=wrapper, initial_position={"j.pos": 1.0}), duration_s=1.0)
    assert sim.steps > BUDGET + 5  # the return stepped the simulation
    assert robot.episode_success  # the time-based mode's flag would have said SUCCESS
    assert not m["success_any"]  # the trial, measured before the return, did not succeed
    assert sim.resets == [[0.25, 0.0]] and robot.scene_resets == 1  # and no second (random) reset happened

    robot.auto_scene_reset = True  # the time-based mode still resets after its return
    core.RolloutStrategy.return_to_initial_position(
        types.SimpleNamespace(robot_wrapper=wrapper, initial_position={"j.pos": 1.0}), duration_s=1.0)
    assert sim.resets == [[0.25, 0.0], None]


# -- the trial loop: numbering, one reset per trial, incomplete trials, resume ---------------------------------
class LoopFixture:
    def __init__(self, tmp_path, poses, script):
        """script: per attempt, what _run_episode does: an int (steps run, then 'reached' if == budget),
        'stall', 'rerun', or an exception to raise."""
        self.sim = FakeSim()
        self.robot = fake_robot(self.sim)
        self.results = me.Results(tmp_path / "res", resume=True)
        self.script = list(script)
        self.events = {"stop_recording": False, "rerecord_episode": False, "exit_early": False}
        robot, events = self.robot, self.events
        client = types.SimpleNamespace(robot=robot, budget=None, start=0)
        client.new_episode = lambda pb: None
        client.begin_budget = lambda n: (setattr(client, "budget", n), setattr(client, "start", robot.sim_steps))
        client.policy_steps = lambda: robot.sim_steps - client.start
        self.client = client
        self.poses = poses
        self.placed = []

        def fake_run(client_, task, time_s, events_, steps, stall_s):
            assert time_s is None and steps == BUDGET
            self.placed.append(list(self.sim.can))
            what = self.script.pop(0)
            if isinstance(what, Exception):
                for _ in range(10):
                    robot.step()
                raise what
            if what == "rerun":
                events["rerecord_episode"] = True
                return "to be run again (Left arrow)", 1.0, 3
            if what == "stall":
                for _ in range(12):
                    robot.step()
                return "stalled: no simulation step for 60 s", 61.0, 12
            for _ in range(what):
                robot.step()
            return ("episode_steps (330) reached" if what == BUDGET else "ended early (Enter)"), 20.0, what

        self.fake_run = fake_run
        self.core = types.SimpleNamespace(RolloutStrategy=types.SimpleNamespace(
            return_to_initial_position=lambda hw, duration_s=3.0: [robot.step() for _ in range(5)]))

    def run(self, monkeypatch, n):
        monkeypatch.setattr(rcl, "_run_episode", self.fake_run)
        monkeypatch.setattr(rcl, "_phase", lambda *a, **k: None)
        monkeypatch.setattr(rcl, "_reset_phase", lambda *a, **k: None)
        matched = {"episode_steps": BUDGET, "poses": self.poses, "results": self.results, "stall_s": 60.0}
        cfg = types.SimpleNamespace(task="t", fps=30, pretrained_name_or_path="ckpt")
        return rcl._run_matched(self.client, cfg, None, matched, None, None, self.events, n, 0.0, self.core, None)


def _poses(n):
    return [{"trial": i, "x": 0.15 + 0.01 * i, "y": 0.001 * i} for i in range(n)]


def test_trials_follow_the_pose_list_one_reset_each(monkeypatch, tmp_path):
    fx = LoopFixture(tmp_path, _poses(3), [BUDGET, BUDGET, BUDGET])
    assert fx.run(monkeypatch, 3)
    assert fx.sim.resets == [[p["x"], p["y"]] for p in _poses(3)]
    assert fx.placed == [[p["x"], p["y"]] for p in _poses(3)]
    trials = me.Results.load(fx.results.root)
    assert [t["trial"] for t in trials] == [0, 1, 2]
    assert all(t["status"] == "complete" and t["policy_steps"] == BUDGET and t["scene_resets_during_trial"] == 0
               for t in trials)
    assert all(t["placed"]["requested_xy"] == [p["x"], p["y"]] for t, p in zip(trials, _poses(3)))
    assert sorted(p.name for p in (fx.results.root / "steps").iterdir()) == [f"trial_00{i}_1.json" for i in range(3)]


def test_rerun_repeats_the_same_pose_and_stall_is_incomplete(monkeypatch, tmp_path):
    fx = LoopFixture(tmp_path, _poses(3), [BUDGET, "rerun", BUDGET, "stall"])
    assert not fx.run(monkeypatch, 3)
    trials = me.Results.load(fx.results.root)
    assert [(t["trial"], t["attempt"], t["status"]) for t in trials] == [
        (0, 1, "complete"), (1, 1, "incomplete"), (1, 2, "complete"), (2, 1, "incomplete")]
    assert fx.placed[1] == fx.placed[2]  # the rerun used the same pose
    assert "stalled" in trials[-1]["ended"] and trials[-1]["policy_steps"] == 12
    assert fx.results.complete == {0, 1}


def test_an_error_stops_the_run_and_resume_runs_only_what_is_missing(monkeypatch, tmp_path):
    fx = LoopFixture(tmp_path, _poses(3), [BUDGET, RuntimeError("Isaac Sim server: boom")])
    with pytest.raises(RuntimeError, match="boom"):
        fx.run(monkeypatch, 3)
    trials = me.Results.load(fx.results.root)
    assert [(t["trial"], t["status"]) for t in trials] == [(0, "complete"), (1, "incomplete")]
    assert trials[1]["ended"].startswith("error:") and "success_incl_return" not in trials[1]

    fx2 = LoopFixture(tmp_path, _poses(3), [BUDGET, BUDGET])  # --resume=true on the same folder
    assert fx2.results.complete == {0}
    assert fx2.run(monkeypatch, 3)
    assert fx2.sim.resets == [[p["x"], p["y"]] for p in _poses(3)[1:]]
    trials = me.Results.load(fx2.results.root)
    assert [(t["trial"], t["attempt"], t["status"]) for t in trials][-2:] == [(1, 2, "complete"), (2, 1, "complete")]


def test_an_existing_results_folder_needs_resume(tmp_path):
    r = me.Results(tmp_path / "res", resume=False)
    r.add({"trial": 0, "attempt": 1, "status": "complete"}, [])
    with pytest.raises(SystemExit, match="resume"):
        me.Results(tmp_path / "res", resume=False)
    assert me.Results(tmp_path / "res", resume=True).complete == {0}


def test_success_is_recorded_from_the_steps_before_the_return(monkeypatch, tmp_path):
    fx = LoopFixture(tmp_path, _poses(1), [BUDGET])
    fx.sim.success_from_step = BUDGET + 2  # only the return (steps 331..335) would see the task succeed
    fx.run(monkeypatch, 1)
    t = me.Results.load(fx.results.root)[0]
    assert t["success_any"] is False and t["success_before_return"] is False and t["success_incl_return"] is True


# -- checkpoint identity -----------------------------------------------------------------------------------------
def test_checkpoint_identity_from_a_hub_snapshot(tmp_path):
    a = tmp_path / "models--org--model_a" / "snapshots" / ("a" * 40)
    b = tmp_path / "models--org--model_b" / "snapshots" / ("b" * 40)
    for d, text in ((a, "{}"), (b, '{"x": 1}')):
        d.mkdir(parents=True)
        (d / "config.json").write_text(text)
    ia, ib = me._checkpoint_identity(str(a)), me._checkpoint_identity(str(b))
    assert ia["hub_repo"] == "org/model_a" and ia["hub_revision"] == "a" * 40
    assert ib["hub_repo"] == "org/model_b" and ia["sha256_config.json"] != ib["sha256_config.json"]
