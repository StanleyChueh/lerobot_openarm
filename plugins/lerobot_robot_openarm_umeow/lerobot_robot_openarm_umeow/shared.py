"""In-process state the openarm_umeow robot publishes for the openarm_quest teleoperator.

lerobot's record / teleoperate loops never hand the robot's state to a teleoperator (only to the
unitree_g1), but both plugins run in the same process. The robot publishes the command it last
actually sent and whether its safety guard is holding the arm; the teleop reads them to start a
reset ramp from where the arm really is, and to freeze its own command when the robot refuses one
-- so the two never fight each other.
"""

import threading
import time


class RobotState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_sent: dict | None = None
        self._last_sent_t = 0.0
        self._fault: str | None = None

    def publish(self, last_sent: dict, fault: str | None) -> None:
        with self._lock:
            self._last_sent = dict(last_sent)
            self._last_sent_t = time.perf_counter()
            self._fault = fault

    def clear(self) -> None:
        with self._lock:
            self._last_sent, self._fault = None, None

    def last_sent(self, max_age_s: float = 0.5) -> dict | None:
        """The motor command last sent, if the robot sent one within max_age_s."""
        with self._lock:
            if self._last_sent is None or time.perf_counter() - self._last_sent_t > max_age_s:
                return None
            return dict(self._last_sent)

    def fault(self) -> str | None:
        with self._lock:
            return self._fault


class TeleopState:
    """The openarm_quest teleop's state, for the rerun status panel (rerun_status.py)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state: str | None = None  # HELD / LIVE / RETURNING / PAUSED, None = no teleop
        self.reason = ""
        self.last_key: str | None = None  # "right" (save) / "left" (discard) last sent from the Quest
        self.last_key_t = 0.0

    def publish(self, state: str, reason: str) -> None:
        with self._lock:
            self.state, self.reason = state, reason

    def episode_key(self, key: str) -> None:
        with self._lock:
            self.last_key, self.last_key_t = key, time.perf_counter()

    def snapshot(self) -> dict:
        with self._lock:
            return {"state": self.state, "reason": self.reason, "last_key": self.last_key,
                    "last_key_t": self.last_key_t}


ROBOT_STATE = RobotState()
TELEOP_STATE = TeleopState()
