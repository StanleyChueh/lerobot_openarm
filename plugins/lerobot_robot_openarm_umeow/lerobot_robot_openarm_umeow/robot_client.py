"""lerobot's async robot client, run as evaluation EPISODES (async, and async + RTC).

    python -m lerobot_robot_openarm_umeow.robot_client <lerobot.async_inference.robot_client flags> \
        --num_episodes=10 --episode_time_s=60 --reset_time_s=30 [--display_data=false]

lerobot's robot client runs one open-ended session until Ctrl-C. This runs the same client -- same flags,
same policy server (ours, with or without --rtc=true), same action queue and aggregation -- as a series of
episodes, the way `lerobot-rollout --strategy.type=episodic` does for normal and RTC inference:

  episode  the policy drives the arm for episode_time_s.
  reset    the arm returns SLOWLY to the start pose (the pose reached after connecting) and is checked
           there, exactly as lerobot-rollout's resets on this robot; then reset_time_s for you to reset the
           scene. After the last episode the arm returns to the start pose and the client disconnects.

  keys     Enter, Right arrow / n: end the episode now (Right arrow also ends the reset; Enter does not,
           see episode_keys.py).  Left arrow / r: run this episode again (not counted).  Esc / q: stop --
           the arm returns to the start pose, then disconnects.

Every episode starts clean: the client's action queue and timesteps are emptied, and the policy server is
told a new client is ready, so it forgets the last episode's observations and, with --rtc=true, its last
chunk -- without reloading the policy. Without --num_episodes this is lerobot's client unchanged, apart from
the display.

  record   --record_repo_id=<user>/eval_<name> (needs --num_episodes) saves the episodes as a LeRobot dataset
           and pushes it to the Hub once every episode has been run (eval_recording.py). The name must start
           with eval_. If it exists, --resume=true adds --num_episodes more episodes to it; without it, a fresh
           evaluation replaces it (the local copy at once, the Hub's when the new one is pushed).
           --record_root=<dir> (named eval_...) stores it there instead of the LeRobot cache.
           --push_to_hub=false keeps it local only (the Hub is neither checked nor written).

  matched  simulation only (openarm_isaac): --episode_steps=330 --poses=<poses.json> --results_dir=<dir>
           runs trial i with the can placed at poses[i], for exactly 330 simulated steps instead of
           --episode_time_s of wall clock, and writes each trial's ground-truth success metrics, measured
           before the return to the start pose (matched_eval.py). [--stall_timeout_s=60]

  display  a rerun window, as lerobot-rollout's --display_data=true (on by default here, as in our
           lerobot-rollout; --display_data=false turns it off): the cameras, the joint state, the actions
           sent to the arm, and the status panel (episode, phase, safety holds). Every run closes the rerun
           windows left open and opens a fresh one (rerun_status.py). lerobot's client reads the robot only
           when it sends an observation to the server, and the display shows those reads; it never reads the
           robot itself (that would slow the control loop), so the cameras and joints update when the client
           sends an observation, and the actions on every tick. Display errors are reported once and never
           stop the arm.

  timing   every tick and chunk is logged as in lerobot-rollout (eval_metrics.py): a line per episode, a
           summary table and a plot at the end, in outputs/eval_logs/ (mode async, or async_rtc when the
           policy server runs with --rtc=true).
"""

import sys
import threading
import time
import types
from queue import Queue

from . import eval_metrics, eval_recording, matched_eval
from .episode_keys import episode, init_keyboard_listener
from .policy_loading import require_async_inference

def _bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise SystemExit(f"[openarm] expected true or false, got {value!r}")


_FLAGS = {"num_episodes": int, "episode_time_s": float, "reset_time_s": float, "display_data": _bool,
          "record_repo_id": str, "record_root": str, "resume": _bool, "push_to_hub": _bool,
          "episode_steps": int, "poses": str, "results_dir": str, "stall_timeout_s": float}
_DEFAULTS = {"episode_time_s": 60.0, "reset_time_s": 60.0,  # lerobot-record's / lerobot-rollout's defaults
             "stall_timeout_s": 60.0}
_MATCHED = {"episode_steps", "poses", "results_dir"}  # all three, or none


def _take_flags(argv: list[str]) -> tuple[dict, list[str]]:
    """Our episode flags, and argv without them (lerobot's parser must not see them)."""
    values, rest = {}, argv[:1]
    for arg in argv[1:]:
        name, eq, value = arg.partition("=")
        if eq and name.startswith("--") and name[2:] in _FLAGS:
            values[name[2:]] = _FLAGS[name[2:]](value)
        else:
            rest.append(arg)
    return values, rest


def _episode_client_class(rc):
    """lerobot's RobotClient, with each action-receiving thread tied to the episode that started it.

    A receiver can still be waiting on the server when its episode ends; it must neither keep running into
    the next episode nor put that late chunk into the next episode's queue."""

    class EpisodeClient(rc.RobotClient):
        def __init__(self, config):
            super().__init__(config)
            self._episode = 0
            self._thread_episode = threading.local()

        def _stale(self) -> bool:
            owner = getattr(self._thread_episode, "episode", None)
            return owner is not None and owner != self._episode

        @property
        def running(self):
            return not self.shutdown_event.is_set() and not self._stale()

        def receive_actions(self, verbose: bool = False):
            self._thread_episode.episode = self._episode
            return super().receive_actions(verbose)

        def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
            if self._stale() or self.shutdown_event.is_set():
                return  # a chunk for an episode that has ended
            return super()._aggregate_action_queues(incoming_actions, aggregate_fn)

        def new_episode(self, services_pb2) -> None:
            """Empty the queue and timesteps, and have the server forget the previous episode."""
            self._episode += 1
            with self.action_queue_lock:
                self.action_queue = Queue()
            with self.latest_action_lock:
                self.latest_action = -1
            self.action_chunk_size = -1
            self.must_go.set()
            self.start_barrier = threading.Barrier(2)
            self.stub.Ready(services_pb2.Empty())  # the server resets its observation queue / timesteps / RTC chunk
            self.shutdown_event.clear()

    return EpisodeClient


def _display_client_class(base):
    """`base` that also logs every control tick to rerun: the observation and the last action sent."""
    from lerobot.utils.rerun_visualization import log_rerun_data

    class DisplayClient(base):
        _shown_action = None
        _display_failed = False

        def _show(self, observation) -> None:
            if self._display_failed:
                return
            try:
                log_rerun_data(observation={k: v for k, v in observation.items() if k != "task"},
                               action=self._shown_action)
            except Exception as e:  # display only: never stop the arm for it
                self._display_failed = True
                print(f"[openarm] rerun display stopped: {e!r}", flush=True)

        def control_loop_action(self, verbose: bool = False):
            self._shown_action = super().control_loop_action(verbose)
            return self._shown_action

        def control_loop_observation(self, task: str, verbose: bool = False):
            observation = super().control_loop_observation(task, verbose)
            if observation:
                self._show(observation)
            return observation

        def _ready_to_send_observation(self):
            # Called once per control tick, after the action. A tick that sends no observation to the server
            # shows only the action: the display never reads the robot itself (a read takes 36-53 ms on the
            # real arm, which would slow the control loop). Observations are shown when the client reads them.
            ready = super()._ready_to_send_observation()
            if not ready and not self._display_failed and self._shown_action:
                try:
                    log_rerun_data(action=self._shown_action)
                except Exception as e:
                    self._display_failed = True
                    print(f"[openarm] rerun display stopped: {e!r}", flush=True)
            return ready

    return DisplayClient


def _start_display(rc) -> bool:
    """Open a fresh rerun window (old ones are closed) and make every client log to it. Before the client
    is built: the robot connects in its constructor and starts the status panel only if rerun is active.
    If rerun cannot start, the evaluation runs without it."""
    from lerobot.utils.rerun_visualization import init_rerun

    try:
        init_rerun(session_name="robot_client")
    except Exception as e:  # e.g. RuntimeError: rerun cannot find its viewer (env not activated)
        print(f"[openarm] rerun window not opened ({e!r}); running without it.", flush=True)
        return False
    rc.RobotClient = _display_client_class(rc.RobotClient)
    return True


def _phase(phase: str, episode: int | None = None, total: int | None = None) -> None:
    from .rerun_status import BOARD

    BOARD.records = False
    if episode is not None:
        BOARD.episode, BOARD.total = episode, total
    BOARD.set_phase(phase)


def _run_episode(client, task: str, episode_time_s: float | None, events: dict, episode_steps: int | None = None,
                 stall_s: float = 60.0) -> tuple[str, float, int]:
    """The policy drives until the time is up (or, with episode_steps, until the client's step budget is spent:
    matched_eval.step_budget_client_class) or a key ends it. (why it ended, seconds, actions executed)"""
    n0 = len(client.action_queue_size)  # one entry per executed action
    receiver = threading.Thread(target=client.receive_actions, daemon=True, name="openarm-actions")
    reason = []
    t0 = time.perf_counter()
    progress = [t0, -1]  # (when, simulation steps) of the last step seen

    def stalled() -> bool:
        steps = client.robot.sim_steps
        if steps != progress[1]:
            progress[:] = [time.perf_counter(), steps]
        return time.perf_counter() - progress[0] >= stall_s

    def watch():
        while not client.shutdown_event.is_set():
            if episode_steps is None and time.perf_counter() - t0 >= episode_time_s:
                reason.append(f"episode_time_s ({episode_time_s:g} s) reached")
            elif episode_steps is not None and stalled():
                reason.append(f"stalled: no simulation step for {stall_s:g} s")
            elif events["stop_recording"]:
                reason.append("stopped (Esc)")
            elif events["rerecord_episode"]:
                reason.append("to be run again (Left arrow)")
            elif events["exit_early"]:
                reason.append("ended early (" + ("Enter" if events["end_episode"] else "Right arrow") + ")")
            if reason:
                events["exit_early"] = False  # used here; must not also end the reset
                break
            time.sleep(0.02)
        client.shutdown_event.set()  # ends lerobot's control loop (and this episode's receiver)

    watcher = threading.Thread(target=watch, daemon=True, name="openarm-episode-clock")
    receiver.start()
    watcher.start()
    try:
        with episode(events):
            client.control_loop(task=task)
    finally:
        client.shutdown_event.set()
        seconds = time.perf_counter() - t0
        watcher.join(timeout=1.0)
        # Not waited for: the receiver may sit in a call to the server for a while, and the arm should not.
        # It ends by itself, and whatever it still receives is dropped (see EpisodeClient).
        receiver.join(timeout=0.1)
    if not reason and episode_steps is not None and client.budget_spent():
        reason.append(f"episode_steps ({episode_steps}) reached")
    return (reason[0] if reason else "interrupted"), seconds, len(client.action_queue_size) - n0


def _reset_phase(reset_time_s: float, events: dict) -> None:
    print(f"[openarm] reset the scene: {reset_time_s:g} s (Right arrow ends it early, Esc stops).", flush=True)
    t_end = time.perf_counter() + reset_time_s
    while time.perf_counter() < t_end and not events["stop_recording"]:
        if events["exit_early"]:
            events["exit_early"] = False
            break
        time.sleep(0.05)


def _eval_log(policy_path: str | None, fps: float, policy_type: str | None) -> "eval_metrics.EvalLog | None":
    if not eval_metrics._enabled():
        return None
    # async_rtc once a chunk says the server runs RTC
    return eval_metrics.EvalLog("async", policy_path or "policy", fps, policy_type)


def _episode_timing(log, fps: float) -> dict | None:
    """The timing log's numbers for the episode just ended (latency, RTC guidance, ...)."""
    if log is None:
        return None
    ticks = [x for x in log.ticks if x[1] == log.episode]
    chunks = [c for c in log.chunks if c["episode"] == log.episode]
    if not ticks:
        return None
    m = eval_metrics.summarize(ticks, chunks, fps)
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in m.items()}  # NaN -> null in JSON


def _run_matched(client, cfg, hw, matched: dict, log, recorder, events, num_episodes: int, reset_time_s: float,
                 core, services_pb2) -> bool:
    """Trials 0 .. num_episodes-1 of the pose list (those not yet complete in the results folder), each on its
    pose for exactly episode_steps simulation steps. True if every one of them is complete."""
    robot, results, budget = client.robot, matched["results"], matched["episode_steps"]
    poses = matched["poses"]
    robot.auto_scene_reset = False  # the can is placed right before each trial, and only then
    attempts: dict[int, int] = {}
    for t in results.load(results.root):
        attempts[t["trial"]] = max(attempts.get(t["trial"], 0), t["attempt"])
    todo = [i for i in range(num_episodes) if i not in results.complete]
    if len(todo) < num_episodes:
        print(f"[openarm] {results.root}: {num_episodes - len(todo)} of the {num_episodes} trials already complete;"
              f" running {len(todo)}.", flush=True)
    while todo and not events["stop_recording"]:
        i = todo[0]
        pose = poses[i]
        attempts[i] = attempts.get(i, 0) + 1
        robot.reset_scene(object_xy=(pose["x"], pose["y"]))  # PoseNotApplied: stops the run, never replaced
        placed, resets = robot.last_reset, robot.scene_resets
        client.new_episode(services_pb2)
        robot.step_log = []
        client.begin_budget(budget)
        _phase("ASYNC_POLICY", i + 1, num_episodes)
        print(f"[openarm] trial {i} ({len(results.complete) + 1}/{num_episodes}): can at ({pose['x']:.4f},"
              f" {pose['y']:.4f}); the policy drives for {budget} simulation steps.", flush=True)
        if log is not None:
            log.begin_episode()
        error = None
        t0 = time.perf_counter()
        try:
            reason, seconds, actions = _run_episode(client, cfg.task, None, events, budget, matched["stall_s"])
        except Exception as e:  # stop safely: this trial is incomplete, the run stops after the bookkeeping
            error, reason, seconds, actions = e, f"error: {e!r}", time.perf_counter() - t0, None
        finally:
            steps, robot.step_log = robot.step_log, None
            policy_steps = client.policy_steps()
            client.begin_budget(None)
        if log is not None:
            log.end_episode()
        rerun = events["rerecord_episode"]
        events["rerecord_episode"] = False
        complete = (error is None and not rerun and policy_steps == budget and len(steps) == budget
                    and robot.scene_resets == resets)
        metrics = matched_eval.trial_metrics(steps, budget) if steps else {"steps": 0}
        trial = {"trial": i, "attempt": attempts[i], "status": "complete" if complete else "incomplete",
                 "ended": reason, "pose": pose, "placed": placed, "policy_steps": policy_steps,
                 "sim_seconds": policy_steps / cfg.fps, "wall_seconds": seconds, "actions_executed": actions,
                 "scene_resets_during_trial": robot.scene_resets - resets,
                 "success_before_return": robot.episode_success,  # the time-based mode's flag, at this point
                 **metrics, "timing": _episode_timing(log, cfg.fps), "policy": cfg.pretrained_name_or_path}
        if error is None:
            _phase("RETURNING")
            core.RolloutStrategy.return_to_initial_position(hw, duration_s=3.0)
            trial["success_incl_return"] = robot.episode_success  # what the time-based mode would have reported
        results.add(trial, steps)
        if recorder is not None:
            if complete:
                recorder.save({"trial": i, "success": metrics["success_any"], "stable": metrics["stable"],
                               "holding_end": metrics["holding_end"], "pose": pose, "steps": policy_steps,
                               "seconds": round(seconds, 2), "actions": actions, "ended": reason})
            else:
                recorder.discard()
        print(f"[openarm] trial {i} {trial['status'].upper()} ({reason}): {policy_steps} steps, {seconds:.1f} s wall;"
              + (f" success {metrics['success_any']}, stable {metrics['stable']}, holding at the end"
                 f" {metrics['holding_end']} ({metrics['outcome']})" if steps else ""), flush=True)
        if error is not None:
            raise error
        if not rerun:
            todo.pop(0)  # an incomplete trial is recorded as such, and run again only by --resume=true
        if todo and not events["stop_recording"]:
            _phase("RESETTING")
            _reset_phase(reset_time_s, events)
    return all(i in results.complete for i in range(num_episodes))


def run_episodes(cfg, num_episodes: int, episode_time_s: float, reset_time_s: float, display: bool,
                 record: "eval_recording.Target | None" = None, matched: dict | None = None) -> None:
    import lerobot.async_inference.robot_client as rc
    import lerobot.rollout.strategies.core as core  # imported before connecting: the robot slows its returns

    from lerobot.async_inference.helpers import visualize_action_queue_size
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot
    from lerobot.transport import services_pb2

    display = display and _start_display(rc)
    log = _eval_log(cfg.pretrained_name_or_path, cfg.fps, cfg.policy_type)
    client_cls = _episode_client_class(rc)
    if log is not None:
        client_cls = eval_metrics.client_class(client_cls, log)
    if record is not None:
        client_cls = eval_recording.client_class(client_cls)
    if matched is not None:
        client_cls = matched_eval.step_budget_client_class(client_cls)
    client = client_cls(cfg)  # connects the robot: the arm goes to the start pose
    if matched is not None:
        if not all(hasattr(client.robot, a) for a in ("sim_steps", "step_log", "auto_scene_reset")):
            client.stop()
            raise SystemExit("[openarm] --episode_steps / --poses need --robot.type=openarm_isaac.")
        matched["results"].write_meta(matched_eval.run_meta(
            matched["argv"], matched["poses_path"], matched["poses_sha"], len(matched["poses"]),
            matched["episode_steps"], cfg.pretrained_name_or_path, client.robot._sim.request("hello")))
    recorder = None
    if record is not None:
        try:  # needs the robot's features, so after the client has built it
            recorder = client.recorder = eval_recording.EvalRecorder(
                record, client.robot, cfg.fps, cfg.task, cfg.pretrained_name_or_path)
        except BaseException:
            client.stop()
            raise
    obs = client.robot.get_observation()
    # Wrapped as lerobot-rollout wraps it: update_dev_v2's return_to_initial_position reads hardware_failure off it.
    hw = types.SimpleNamespace(robot_wrapper=ThreadSafeRobot(client.robot),
                               initial_position={k: v for k, v in obs.items() if k.endswith(".pos")})
    if not client.start():  # handshake + the policy is loaded on the server
        client.stop()
        if recorder is not None:
            recorder.finish(push=False)  # nothing saved: removes the empty dataset
        return
    listener, events = init_keyboard_listener()
    results = []
    finished = False  # every episode run, without an error or Ctrl-C
    try:
        if matched is not None:
            finished = _run_matched(client, cfg, hw, matched, log, recorder, events, num_episodes, reset_time_s,
                                    core, services_pb2)
        while matched is None and len([r for r in results if r[1]]) < num_episodes and not events["stop_recording"]:
            number = len([r for r in results if r[1]]) + 1
            client.new_episode(services_pb2)
            _phase("ASYNC_POLICY", number, num_episodes)
            print(f"[openarm] episode {number}/{num_episodes}: the policy drives for up to {episode_time_s:g} s"
                  " (Enter or Right arrow ends it, Left arrow runs it again, Esc stops).", flush=True)
            if log is not None:
                log.begin_episode()
            reason, seconds, actions = _run_episode(client, cfg.task, episode_time_s, events)
            if log is not None:
                log.end_episode()
            counted = not events["rerecord_episode"]
            events["rerecord_episode"] = False
            results.append((number, counted, seconds, actions))
            print(f"[openarm] episode {number}/{num_episodes} {reason}: {seconds:.1f} s, {actions} actions"
                  f" executed ({actions / seconds if seconds else 0:.0f} Hz)."
                  + ("" if counted else " It will be run again."), flush=True)
            sim_results = list(getattr(client.robot, "_results", []))
            _phase("RETURNING")
            core.RolloutStrategy.return_to_initial_position(hw, duration_s=3.0)  # in simulation, reports success
            if recorder is not None:
                if counted:
                    new = getattr(client.robot, "_results", [])[len(sim_results):]
                    recorder.save({"success": bool(new[-1]) if new else None, "seconds": round(seconds, 2),
                                   "actions": actions, "ended": reason})
                else:
                    recorder.discard()
            more = len([r for r in results if r[1]]) < num_episodes
            if more and not events["stop_recording"]:
                _phase("RESETTING")
                _reset_phase(reset_time_s, events)
        if matched is None:
            finished = len([r for r in results if r[1]]) == num_episodes
    finally:
        if listener is not None:
            listener.stop()
        client.stop()  # disconnects: the robot ramps back to where it was before connecting
        done = [r for r in results if r[1]]
        if matched is None:
            print(f"[openarm] {len(done)}/{num_episodes} episode(s) run: "
                  + ", ".join(f"#{n} {s:.0f} s" for n, _, s, _ in done), flush=True)
        else:
            print(f"[openarm] {len(matched['results'].complete)}/{num_episodes} trial(s) complete in"
                  f" {matched['results'].root}.", flush=True)
        if cfg.debug_visualize_queue_size:
            visualize_action_queue_size(client.action_queue_size)
        if log is not None:
            log.finish()
        if display:
            from lerobot.utils.rerun_visualization import shutdown_rerun

            shutdown_rerun()  # the window stays open to look at; the next run replaces it
        if recorder is not None:
            recorder.finish(push=finished)  # last: the arm is disconnected, the upload may take minutes


def main() -> None:
    values, sys.argv[:] = _take_flags(sys.argv)
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    require_async_inference()
    import lerobot.async_inference.robot_client as rc

    display = values.get("display_data", True)
    if "num_episodes" not in values:
        if set(values) - {"num_episodes", "display_data"}:
            raise SystemExit("[openarm] --episode_time_s / --reset_time_s / --record_repo_id / --resume need"
                             " --num_episodes.")
        display = display and _start_display(rc)
        log = _eval_log(eval_metrics._argv_value("--pretrained_name_or_path"),
                        float(eval_metrics._argv_value("--fps") or 30), eval_metrics._argv_value("--policy_type"))
        if log is not None:
            rc.RobotClient = eval_metrics.client_class(rc.RobotClient, log)
        try:
            rc.async_client()  # lerobot's client, unchanged apart from the display and the timing log
        finally:
            if log is not None:
                log.end_episode()
                log.finish()
            if display:
                from lerobot.utils.rerun_visualization import shutdown_rerun

                shutdown_rerun()
        return
    given = _MATCHED & set(values)
    if given and given != _MATCHED:
        raise SystemExit("[openarm] --episode_steps, --poses and --results_dir go together (a matched evaluation);"
                         f" missing {sorted(_MATCHED - given)}.")
    if given and "episode_time_s" in values:
        raise SystemExit("[openarm] --episode_steps replaces --episode_time_s; give one of them.")
    if "stall_timeout_s" in values and not given:
        raise SystemExit("[openarm] --stall_timeout_s needs --episode_steps.")
    if {"record_root", "push_to_hub"} & set(values) and "record_repo_id" not in values:
        raise SystemExit("[openarm] --record_root / --push_to_hub need --record_repo_id.")
    if "resume" in values and not ("record_repo_id" in values or given):
        raise SystemExit("[openarm] --resume needs --record_repo_id or --results_dir.")
    matched = None
    if given:  # checked before anything connects
        poses, sha = matched_eval.load_poses(values["poses"])
        if values["episode_steps"] < 1:
            raise SystemExit("[openarm] --episode_steps must be at least 1.")
        if values["num_episodes"] > len(poses):
            raise SystemExit(f"[openarm] --num_episodes={values['num_episodes']}, but {values['poses']} has"
                             f" {len(poses)} poses.")
        matched = {"episode_steps": values["episode_steps"], "poses": poses, "poses_path": values["poses"],
                   "poses_sha": sha, "stall_s": values.get("stall_timeout_s", _DEFAULTS["stall_timeout_s"]),
                   "results": matched_eval.Results(values["results_dir"], values.get("resume", False)),
                   "argv": list(sys.argv) + [f"--{k}={v}" for k, v in values.items()]}
    record = None
    if "record_repo_id" in values:  # checked before anything connects: never a name that could overwrite a dataset
        record = eval_recording.check_repo_id(values["record_repo_id"], values.get("record_root"),
                                              values.get("resume", False), values.get("push_to_hub", True))
    import draccus

    from lerobot.async_inference.configs import RobotClientConfig

    @draccus.wrap()
    def episodes(cfg: RobotClientConfig):
        run_episodes(cfg, values["num_episodes"], values.get("episode_time_s", _DEFAULTS["episode_time_s"]),
                     values.get("reset_time_s", _DEFAULTS["reset_time_s"]), display, record, matched)

    episodes()


if __name__ == "__main__":
    main()
