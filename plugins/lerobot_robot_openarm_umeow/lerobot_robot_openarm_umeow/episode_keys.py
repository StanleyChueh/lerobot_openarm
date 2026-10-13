"""Enter ends the running evaluation episode, in every evaluation mode.

  normal, RTC      lerobot-rollout --strategy.type=episodic   (patched here by install())
  async, async+RTC python -m lerobot_robot_openarm_umeow.robot_client --num_episodes=...

lerobot's keys stay as they are: Right arrow / n ends the episode or the reset, Left arrow / r runs the
episode again, Esc / q stops. Enter is added and does what Right arrow does DURING an episode: the policy
stops, the arm returns to the start pose, and the next episode starts after reset_time_s. Enter outside an
episode (the return, the reset) is ignored, so pressing it twice never skips the reset and starts the
policy again before the scene is ready. Same for every policy: only the episode loop is involved.

Enter counts only in THIS terminal. lerobot's keys use pynput's global listener on X11, which hears every
window: harmless for arrows and Esc, but Enter is pressed all the time elsewhere (running a command in
another terminal would end the episode). So, with pynput, Enter is read from this terminal's input instead;
without pynput (Wayland, SSH) lerobot already reads every key from this terminal.
"""

import sys
import threading
from contextlib import contextmanager, nullcontext

_CONTROLS = {"right": "right", "n": "right", "left": "left", "r": "left", "esc": "esc", "q": "esc"}


class _Listeners:
    def __init__(self, *listeners):
        self._listeners = [x for x in listeners if x is not None]

    def stop(self) -> None:
        for listener in self._listeners:
            listener.stop()


def init_keyboard_listener():
    """lerobot's init_keyboard_listener() (same backends, keys and events), plus Enter from this terminal."""
    from lerobot.utils.keyboard_input import TerminalKeyListener, apply_recording_control, create_key_listener

    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False,
              "end_episode": False, "_episode_running": False, "_lock": threading.Lock()}

    def on_key(name: str) -> None:
        key = name.lower()
        if key == "enter":
            with events["_lock"]:
                if events["_episode_running"]:
                    print("Enter pressed. Ending the episode...", flush=True)
                    events["end_episode"] = events["exit_early"] = True
                else:
                    print("Enter ignored: no episode is running (Right arrow ends the reset).", flush=True)
        elif key in _CONTROLS:
            apply_recording_control(_CONTROLS[key], events)

    listener = create_key_listener(lambda name: name != "enter" and on_key(name),
                                   controls_help="Right/Left/Esc, or n=next, r=re-record, q=quit")
    if isinstance(listener, TerminalKeyListener):  # lerobot reads this terminal already: Enter from it too
        listener._on_key = on_key
        return listener, events
    terminal = None
    if sys.stdin.isatty():
        terminal = TerminalKeyListener(lambda name: name == "enter" and on_key(name))
        terminal.start()
        print("[openarm] Enter (end the episode) is read from this terminal only -- keep it focused.", flush=True)
    return _Listeners(listener, terminal), events


@contextmanager
def episode(events: dict):
    """Mark an episode as running: Enter ends it. A press it did not use is dropped when it ends."""
    with events["_lock"]:
        events["_episode_running"], events["end_episode"] = True, False
    try:
        yield
    finally:
        with events["_lock"]:
            events["_episode_running"] = False
            if events["end_episode"]:  # Enter is never left over to end the reset
                events["end_episode"] = events["exit_early"] = False


def install() -> None:
    """lerobot-rollout's episodic strategy: our listener, and Enter only while _policy_loop runs."""
    command = sys.argv[0].rsplit("/", 1)[-1] if sys.argv else ""
    if "rollout" not in command:
        return
    import lerobot.rollout.strategies.episodic as episodic

    strategy = episodic.EpisodicStrategy
    if getattr(strategy, "_openarm_enter_key", False):
        return
    policy_loop = strategy._policy_loop

    def _policy_loop(self, *args, **kwargs):
        events = kwargs.get("events", args[2] if len(args) > 2 else None)
        with episode(events) if events is not None and "_lock" in events else nullcontext():
            return policy_loop(self, *args, **kwargs)

    episodic.init_keyboard_listener = init_keyboard_listener  # looked up by EpisodicStrategy.setup()
    strategy._policy_loop = _policy_loop
    strategy._openarm_enter_key = True
