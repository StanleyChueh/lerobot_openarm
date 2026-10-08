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
        # The arm joint furthest from its command, as measured: (key, commanded, measured) and when.
        self._worst: tuple[str, float, float] | None = None
        self._worst_t = 0.0
        # True while the robot connects and ramps to its start pose: nothing the teleop commands is
        # executed then, so the teleop must not go LIVE.
        self.connecting = False

    def publish(self, last_sent: dict, fault: str | None) -> None:
        with self._lock:
            self._last_sent = dict(last_sent)
            self._last_sent_t = time.perf_counter()
            self._fault = fault

    def publish_tracking(self, key: str, commanded: float, measured: float) -> None:
        with self._lock:
            self._worst, self._worst_t = (key, commanded, measured), time.perf_counter()

    def tracking(self, max_age_s: float = 0.5) -> tuple[str, float, float] | None:
        """(key, commanded, measured) of the arm joint furthest from its command, if measured within
        max_age_s; None without a robot (or a stale one)."""
        with self._lock:
            if self._worst is None or time.perf_counter() - self._worst_t > max_age_s:
                return None
            return self._worst

    def clear(self) -> None:
        with self._lock:
            self._last_sent, self._fault, self._worst = None, None, None

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
        # The newest Quest packet, summarised (see ik_driver.QuestIKDriver._publish_quest), plus how many
        # packets have arrived, when the last did, and from where.
        self.quest: dict | None = None
        self.packets = 0
        self.packet_t: float | None = None
        self.sender: str | None = None
        self.message = ""  # the teleop's latest terminal message, e.g. why X was ignored

    def publish(self, state: str, reason: str) -> None:
        with self._lock:
            self.state, self.reason = state, reason

    def episode_key(self, key: str) -> None:
        with self._lock:
            self.last_key, self.last_key_t = key, time.perf_counter()

    def log(self, text: str) -> None:
        with self._lock:
            self.message = text

    def publish_quest(self, quest: dict | None, packets: int, packet_t: float | None, sender: str | None) -> None:
        with self._lock:
            self.quest, self.packets, self.packet_t, self.sender = quest, packets, packet_t, sender

    def snapshot(self) -> dict:
        with self._lock:
            return {"state": self.state, "reason": self.reason, "last_key": self.last_key,
                    "last_key_t": self.last_key_t, "quest": self.quest, "packets": self.packets,
                    "packet_t": self.packet_t, "sender": self.sender, "message": self.message}


ROBOT_STATE = RobotState()
TELEOP_STATE = TeleopState()
