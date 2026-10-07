"""Start each lerobot-record episode on the operator's first X, not on lerobot's clock.

lerobot-record starts recording an episode the moment the previous reset phase ends (and the first one
the moment the robot is connected), so every episode would begin with however long the operator took
to get ready, frozen at the home pose. lerobot calls its module-level record_loop() by name for both the
episodes (with a dataset) and the reset phases (without one), so the teleop wraps it -- only when the
teleoperator is openarm_quest:

  - episode: first a waiting loop that keeps the robot commanded (held at home) and the rerun view
    alive but writes NO frames, until the Quest's X puts the teleop LIVE; then lerobot's own loop
    records, its episode clock starting at that X. Esc still stops the session while waiting. Arrow /
    A / B / Y presses while waiting are dropped: there is no episode yet to save or discard.
  - after each episode, however it ended (X / Y, lerobot's timer, a keyboard arrow), the arms return
    home slowly, so the next episode starts from HELD and not mid-motion.
  - reset phase: lerobot's own loop, ended early if X is pressed (the arms are home) -- the next episode
    then starts recording immediately.
"""

import sys
import threading
import time

from lerobot_robot_openarm_umeow.rerun_status import BOARD
from lerobot_robot_openarm_umeow.shared import TELEOP_STATE

_LIVE = "LIVE"


def install(teleop_cls) -> None:
    module = sys.modules.get("lerobot.scripts.lerobot_record")
    if module is None or getattr(module, "_openarm_record_gate", False):
        return
    original = module.record_loop

    def record_loop(*args, **kwargs):
        teleop, events = kwargs.get("teleop"), kwargs.get("events")
        if args or not isinstance(teleop, teleop_cls) or events is None:
            return original(*args, **kwargs)
        if kwargs.get("dataset") is not None:
            if not _wait_for_x(kwargs):
                return None
            if kwargs.get("timer") is not None:
                kwargs["timer"].restart()  # the wait is not part of the episode's cadence
            try:
                return original(*args, **kwargs)
            finally:
                # Ended by lerobot's timer or a keyboard arrow rather than X / Y: still send the arms home,
                # so the next episode starts from HELD and not mid-motion.
                teleop.driver.request_return("episode ended")
        return _reset_phase(original, kwargs)

    module.record_loop = record_loop
    module._openarm_record_gate = True
    BOARD.gated = True


def _wait_for_x(kw: dict) -> bool:
    """Hold until the teleop goes LIVE. False if the session was stopped meanwhile."""
    robot, teleop, events = kw["robot"], kw["teleop"], kw["events"]
    t_proc, r_proc, o_proc = kw["teleop_action_processor"], kw["robot_action_processor"], kw["robot_observation_processor"]
    display = kw.get("display_data", False)
    period = 1.0 / kw["fps"]
    if display:
        from lerobot.utils.visualization_utils import log_visualization_data
    if TELEOP_STATE.snapshot()["state"] != _LIVE:
        BOARD.set_phase("WAITING")
        print("[openarm_quest] waiting for X to start recording this episode (nothing is recorded yet).", flush=True)
    while TELEOP_STATE.snapshot()["state"] != _LIVE:
        t0 = time.perf_counter()
        if events["stop_recording"]:
            return False
        if events["exit_early"] or events["rerecord_episode"]:
            events["exit_early"] = events["rerecord_episode"] = False  # nothing recorded yet to save / discard
        obs = robot.get_observation()
        action = teleop.get_action()
        robot.send_action(r_proc((t_proc((action, obs)), obs)))
        if display:
            log_visualization_data(kw.get("display_mode", "rerun"), observation=o_proc(obs), action=action,
                                   compress_images=kw.get("display_compressed_images", False))
        remaining = period - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)
    BOARD.set_phase("RECORDING")
    print("[openarm_quest] X: recording.", flush=True)
    return True


def _reset_phase(original, kw: dict):
    """lerobot's reset loop, ended early when the operator presses X to start the next episode."""
    events = kw["events"]
    done = threading.Event()
    fired = threading.Event()
    was_live = TELEOP_STATE.snapshot()["state"] == _LIVE

    def watch():
        while not done.wait(0.02):
            live = TELEOP_STATE.snapshot()["state"] == _LIVE
            if live and not was_live and not fired.is_set():
                fired.set()
                events["exit_early"] = True  # lerobot's loop checks this every tick, then clears it
                return

    watcher = threading.Thread(target=watch, daemon=True, name="openarm-reset-gate")
    watcher.start()
    try:
        return original(**kw)
    finally:
        done.set()
        watcher.join()
        if fired.is_set():
            # If X landed after lerobot's last check, its exit_early would end the next episode at once.
            events["exit_early"] = False
