"""lerobot-record episodes driven by the Quest: start on the first X, save after the return home.

lerobot-record starts an episode the moment the previous reset phase ends, ends it on a key press or after
episode_time_s, and records nothing of what happens after that. lerobot calls its module-level
record_loop() by name for both the episodes (with a dataset) and the reset phases (without one), so the
teleop wraps it -- only when the teleoperator is openarm_quest:

  episode   1. WAIT, recording nothing, with the robot held at home and the rerun view alive, until the
               Quest's X puts the teleop LIVE (X is refused at any other time, e.g. during a reset).
            2. RECORD with lerobot's own loop, its clock starting at that X.
            3. 2nd X: the arms return home slowly -- still recorded -- and the episode is SAVED when they
               arrive (lerobot's exit_early flag, set here directly: no key press involved). The grippers
               stay as they were until the episode has ended, then open.
               Y: DISCARD at once (rerecord_episode + exit_early); the arms return home during the reset.
            4. episode_time_s is enforced here, not by lerobot: when it runs out the arms start the same
               saving return as a 2nd X, so the saved episode always ends at home, never cut mid-return.
               (While PAUSED it waits: press Y.) This save always ends the episode, even with
               --teleop.episode_buttons=false (which only stops the Quest's own 2nd X / Y from doing so).
  reset     lerobot's own loop, unchanged. X is refused from the end of an episode until the next one
            waits for it (the reset, and the save); before the first episode it is accepted, and recording
            then starts as soon as the gate sees the arms LIVE.

Keyboard Right / Left / Esc keep working as in plain lerobot-record.
"""

import sys
import threading
import time

from lerobot_robot_openarm_umeow.rerun_status import BOARD
from lerobot_robot_openarm_umeow.shared import TELEOP_STATE

_LIVE = "LIVE"
_FOREVER = 10.0**9  # lerobot's own episode clock is disabled; the cap below replaces it
_STATE = {"x_ok": True}  # X may start driving: before the first episode, and while an episode waits for it


def install(teleop_cls) -> None:
    module = sys.modules.get("lerobot.scripts.lerobot_record")
    if module is None or getattr(module, "_openarm_record_gate", False):
        return
    original = module.record_loop

    def record_loop(*args, **kwargs):
        teleop, events = kwargs.get("teleop"), kwargs.get("events")
        if args or not isinstance(teleop, teleop_cls) or events is None:
            return original(*args, **kwargs)
        driver = teleop.driver
        driver.defer_gripper_open = True
        driver.x_allowed = lambda: _STATE["x_ok"]
        if kwargs.get("dataset") is None:
            return original(*args, **kwargs)  # reset phase: lerobot's own
        if not _wait_for_x(kwargs):
            return None
        if kwargs.get("timer") is not None:
            kwargs["timer"].restart()  # the wait is not part of the episode's cadence
        return _record_episode(original, teleop, events, kwargs)

    module.record_loop = record_loop
    module._openarm_record_gate = True
    BOARD.gated = True


def _decide(events: dict, key: str) -> None:
    """The Quest's decision, applied to lerobot-record's own flags (what its Right / Left keys set)."""
    if key == "right":
        print("[openarm_quest] episode SAVED (arms home).", flush=True)
        events["exit_early"] = True
    elif key == "left":
        print("[openarm_quest] episode DISCARDED.", flush=True)
        events["rerecord_episode"] = True
        events["exit_early"] = True


def _record_episode(original, teleop, events: dict, kw: dict):
    cap = kw.get("control_time_s") or _FOREVER
    done = threading.Event()
    capped = threading.Event()

    def enforce_cap():
        if done.wait(cap):
            return
        state = TELEOP_STATE.snapshot()["state"]
        if state == _LIVE:
            print(f"[openarm_quest] episode_time_s ({cap:g} s) reached: returning home, then saving.", flush=True)
            capped.set()
            teleop.driver.request_return("episode time limit", save=True)
        elif state == "PAUSED":
            print(f"[openarm_quest] episode_time_s ({cap:g} s) reached while PAUSED: press Y to discard.", flush=True)

    def sink(key: str) -> None:
        # The time limit's save always ends the episode; the Quest's own 2nd X / Y only if episode_buttons.
        if teleop.config.episode_buttons or (key == "right" and capped.is_set()):
            _decide(events, key)

    _STATE["x_ok"] = False  # from here until the next episode waits for X
    teleop.episode_sink = sink
    watcher = threading.Thread(target=enforce_cap, daemon=True, name="openarm-episode-cap")
    watcher.start()
    try:
        return original(**{**kw, "control_time_s": _FOREVER})
    finally:
        done.set()
        teleop.episode_sink = None
        # Ended some other way (keyboard arrow, Esc): send the arms home too, unrecorded.
        teleop.driver.request_return("episode ended")
        teleop.driver.release_grippers()


def _wait_for_x(kw: dict) -> bool:
    """Hold until the teleop goes LIVE. False if the session was stopped meanwhile."""
    robot, teleop, events = kw["robot"], kw["teleop"], kw["events"]
    t_proc, r_proc, o_proc = kw["teleop_action_processor"], kw["robot_action_processor"], kw["robot_observation_processor"]
    display = kw.get("display_data", False)
    period = 1.0 / kw["fps"]
    if display:
        from lerobot.utils.visualization_utils import log_visualization_data
    BOARD.set_phase("WAITING")
    print("[openarm_quest] waiting for X to start recording this episode (nothing is recorded yet).", flush=True)
    _STATE["x_ok"] = True
    try:
        while TELEOP_STATE.snapshot()["state"] != _LIVE:
            t0 = time.perf_counter()
            if events["stop_recording"]:
                return False
            if events["exit_early"] or events["rerecord_episode"]:
                events["exit_early"] = events["rerecord_episode"] = False  # nothing recorded yet
            obs = robot.get_observation()
            action = teleop.get_action()
            robot.send_action(r_proc((t_proc((action, obs)), obs)))
            if display:
                log_visualization_data(kw.get("display_mode", "rerun"), observation=o_proc(obs), action=action,
                                       compress_images=kw.get("display_compressed_images", False))
            remaining = period - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        _STATE["x_ok"] = False
    BOARD.set_phase("RECORDING")
    print("[openarm_quest] X: recording.", flush=True)
    return True
