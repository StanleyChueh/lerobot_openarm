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
chunk -- without reloading the policy. Nothing is recorded. Without --num_episodes this is lerobot's client
unchanged, apart from the display.

  display  a rerun window, as lerobot-rollout's --display_data=true (on by default here, as in our
           lerobot-rollout; --display_data=false turns it off): the cameras, the joint state, the actions
           sent to the arm, and the status panel (episode, phase, safety holds). Every run closes the rerun
           windows left open and opens a fresh one (rerun_status.py). lerobot's client reads the robot only
           when it sends an observation to the server; for a live view, the ticks in between read it too, as
           lerobot-rollout does on every tick. Display errors are reported once and never stop the arm.
"""

import sys
import threading
import time
import types
from queue import Queue

from .episode_keys import episode, init_keyboard_listener

def _bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise SystemExit(f"[openarm] expected true or false, got {value!r}")


_FLAGS = {"num_episodes": int, "episode_time_s": float, "reset_time_s": float, "display_data": _bool}
_DEFAULTS = {"episode_time_s": 60.0, "reset_time_s": 60.0}  # lerobot-record's / lerobot-rollout's defaults


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
            # reads the robot for the display only.
            ready = super()._ready_to_send_observation()
            if not ready and not self._display_failed:
                try:
                    self._show(self.robot.get_observation())
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


def _run_episode(client, task: str, episode_time_s: float, events: dict) -> tuple[str, float, int]:
    """The policy drives until the time is up or a key ends it. (why it ended, seconds, actions executed)"""
    n0 = len(client.action_queue_size)  # one entry per executed action
    receiver = threading.Thread(target=client.receive_actions, daemon=True, name="openarm-actions")
    reason = []
    t0 = time.perf_counter()

    def watch():
        while not client.shutdown_event.is_set():
            if time.perf_counter() - t0 >= episode_time_s:
                reason.append(f"episode_time_s ({episode_time_s:g} s) reached")
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
    return (reason[0] if reason else "interrupted"), seconds, len(client.action_queue_size) - n0


def _reset_phase(reset_time_s: float, events: dict) -> None:
    print(f"[openarm] reset the scene: {reset_time_s:g} s (Right arrow ends it early, Esc stops).", flush=True)
    t_end = time.perf_counter() + reset_time_s
    while time.perf_counter() < t_end and not events["stop_recording"]:
        if events["exit_early"]:
            events["exit_early"] = False
            break
        time.sleep(0.05)


def run_episodes(cfg, num_episodes: int, episode_time_s: float, reset_time_s: float, display: bool) -> None:
    import lerobot.async_inference.robot_client as rc
    import lerobot.rollout.strategies.core as core  # imported before connecting: the robot slows its returns

    from lerobot.async_inference.helpers import visualize_action_queue_size
    from lerobot.transport import services_pb2

    display = display and _start_display(rc)
    client = _episode_client_class(rc)(cfg)  # connects the robot: the arm goes to the start pose
    obs = client.robot.get_observation()
    hw = types.SimpleNamespace(robot_wrapper=client.robot,
                               initial_position={k: v for k, v in obs.items() if k.endswith(".pos")})
    if not client.start():  # handshake + the policy is loaded on the server
        client.stop()
        return
    listener, events = init_keyboard_listener()
    results = []
    try:
        while len([r for r in results if r[1]]) < num_episodes and not events["stop_recording"]:
            number = len([r for r in results if r[1]]) + 1
            client.new_episode(services_pb2)
            _phase("ASYNC_POLICY", number, num_episodes)
            print(f"[openarm] episode {number}/{num_episodes}: the policy drives for up to {episode_time_s:g} s"
                  " (Enter or Right arrow ends it, Left arrow runs it again, Esc stops).", flush=True)
            reason, seconds, actions = _run_episode(client, cfg.task, episode_time_s, events)
            counted = not events["rerecord_episode"]
            events["rerecord_episode"] = False
            results.append((number, counted, seconds, actions))
            print(f"[openarm] episode {number}/{num_episodes} {reason}: {seconds:.1f} s, {actions} actions"
                  f" executed ({actions / seconds if seconds else 0:.0f} Hz)."
                  + ("" if counted else " It will be run again."), flush=True)
            _phase("RETURNING")
            core.RolloutStrategy.return_to_initial_position(hw, duration_s=3.0)
            more = len([r for r in results if r[1]]) < num_episodes
            if more and not events["stop_recording"]:
                _phase("RESETTING")
                _reset_phase(reset_time_s, events)
    finally:
        if listener is not None:
            listener.stop()
        client.stop()  # disconnects: the robot ramps back to where it was before connecting
        done = [r for r in results if r[1]]
        print(f"[openarm] {len(done)}/{num_episodes} episode(s) run: "
              + ", ".join(f"#{n} {s:.0f} s" for n, _, s, _ in done), flush=True)
        if cfg.debug_visualize_queue_size:
            visualize_action_queue_size(client.action_queue_size)
        if display:
            from lerobot.utils.rerun_visualization import shutdown_rerun

            shutdown_rerun()  # the window stays open to look at; the next run replaces it


def main() -> None:
    values, sys.argv[:] = _take_flags(sys.argv)
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    import lerobot.async_inference.robot_client as rc

    display = values.get("display_data", True)
    if "num_episodes" not in values:
        if set(values) - {"num_episodes", "display_data"}:
            raise SystemExit("[openarm] --episode_time_s / --reset_time_s need --num_episodes.")
        display = display and _start_display(rc)
        try:
            rc.async_client()  # lerobot's client, unchanged apart from the display
        finally:
            if display:
                from lerobot.utils.rerun_visualization import shutdown_rerun

                shutdown_rerun()
        return
    import draccus

    from lerobot.async_inference.configs import RobotClientConfig

    @draccus.wrap()
    def episodes(cfg: RobotClientConfig):
        run_episodes(cfg, values["num_episodes"], values.get("episode_time_s", _DEFAULTS["episode_time_s"]),
                     values.get("reset_time_s", _DEFAULTS["reset_time_s"]), display)

    episodes()


if __name__ == "__main__":
    main()
