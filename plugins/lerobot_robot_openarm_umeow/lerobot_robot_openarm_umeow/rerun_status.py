"""A status panel in lerobot's rerun window: episode number, phase, teleop state and safety holds.

lerobot-record tells nobody but its log what it is doing, and draws rerun with a fixed layout (one
view per camera, an observation and an action plot) that would hide anything else logged. So, from
inside the same process:

  - the phase comes from lerobot's own announcements -- log_say() logs "Recording episode N",
    "Reset the environment", "Re-record episode", "Stop recording" -- read by a logging handler, and
    from thin wrappers around LeRobotDataset.save_episode / finalize;
  - the teleop state (HELD / LIVE / RETURNING / PAUSED) and the robot's SAFETY HOLD come from
    shared.py;
  - the layout gets a "status" text view on top: lerobot's blueprint builder is wrapped, keeping its
    views unchanged below.

Everything here is best-effort display: a failure is logged once and never reaches the control loop.
It only runs when rerun is active (--display_data=true); otherwise start() does nothing.
"""

import logging
import re
import sys
import threading
import time

from .shared import ROBOT_STATE, TELEOP_STATE

logger = logging.getLogger(__name__)

ENTITY = "status"
REFRESH_S = 0.5

_PHASE_STYLE = {
    "STARTING": ("⚪", "starting"),
    "WAITING": ("⏳", "WAITING for X (not recording yet)"),
    "RECORDING": ("🔴", "RECORDING"),
    "POLICY": ("🤖", "POLICY RUNNING (recorded)"),
    "RESETTING": ("🟡", "RESETTING the scene (not recorded)"),
    "SAVING": ("💾", "SAVING"),
    "STOPPING": ("⚫", "STOPPING"),
    "FINALIZING": ("💾", "FINALIZING the dataset"),
    "SHUTDOWN": ("⚫", "returning to rest, then motors off"),
    "TELEOP": ("🟢", "teleoperating (not recording)"),
    "ROLLOUT": ("🟢", "running the policy (lerobot-rollout)"),
}
_VALID = {0: "OK", 1: "STALE", 2: "LOST"}


def _quest_lines(t: dict) -> list[str]:
    """What the Quest is sending, so a dead link or an untracked controller is visible at a glance."""
    now = time.perf_counter()
    q, age = t["quest"], (None if t["packet_t"] is None else now - t["packet_t"])
    if q is None or age is None:
        return ["**Meta Quest:** ❌ NO PACKETS -- start the Quest app; it must send to this PC (UDP 5006),"
                " and the dora dataflow must be stopped"]
    if age > 1.0:
        head = f"**Meta Quest:** ❌ packets STOPPED {age:.0f} s ago (headset asleep / app closed?)"
    else:
        head = f"**Meta Quest:** ✅ connected ({t['sender']}, last packet {age * 1000:.0f} ms ago)"

    def track(v: int) -> str:
        return ("✅ " if v == 0 else "⚠️ " if v == 1 else "❌ ") + _VALID.get(v, str(v))

    def p(xyz) -> str:
        return "-" if xyz is None else f"({xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:+.2f})"

    return [
        head,
        f"headset {track(q['v'])} · right controller {track(q['vr'])} {p(q['rc'])} ·"
        f" left controller {track(q['vl'])} {p(q['lc'])}",
        f"buttons pressed: **{' '.join(q['buttons']) or 'none'}** · triggers R {q['rt']:.2f} L {q['lt']:.2f}"
        f" · grips R {q['rg']:.2f} L {q['lg']:.2f} · sticks R ({q['rstick'][0]:+.2f}, {q['rstick'][1]:+.2f})"
        f" L ({q['lstick'][0]:+.2f}, {q['lstick'][1]:+.2f})",
    ]


_TELEOP_STYLE = {
    "HELD": "⏸ HELD at home, grippers open -- press **X** to start",
    "LIVE": "▶ LIVE -- arms follow the controllers. **X** = save, **Y** = discard",
    "RETURNING": "↩ RETURNING home slowly -- wait",
    "PAUSED": "⛔ PAUSED -- **X** = resume from here, **Y** = return home",
}


def _rerun_active() -> bool:
    try:
        import rerun as rr

        return rr.get_global_data_recording() is not None
    except Exception:
        return False


def _argv_int(flag: str) -> int | None:
    for i, a in enumerate(sys.argv):
        if a.startswith(flag + "="):
            value = a.split("=", 1)[1]
        elif a == flag and i + 1 < len(sys.argv):
            value = sys.argv[i + 1]
        else:
            continue
        try:
            return int(value)
        except ValueError:
            return None
    return None


class _LerobotLogHandler(logging.Handler):
    _RECORDING = re.compile(r"^Recording episode (\d+)")

    def __init__(self, board: "StatusBoard"):
        super().__init__(level=logging.INFO)
        self._board = board

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            m = self._RECORDING.match(msg)
            if m:
                self._board.on_recording(int(m.group(1)))
            elif msg.startswith("Reset the environment"):
                self._board.set_phase("RESETTING")
            elif msg.startswith("Re-record episode"):
                self._board.note(f"episode {self._board.episode} DISCARDED -- recording it again")
            elif msg.startswith("Stop recording"):
                self._board.set_phase("STOPPING")
        except Exception:
            pass


class StatusBoard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.phase = "TELEOP"
        self.phase_t = time.perf_counter()
        self.episode: int | None = None
        self.retry = False
        self.saved: int | None = None
        self.total: int | None = None
        self.last_note = ""
        # Set by the openarm_quest record gate: lerobot's "Recording episode N" then only means "waiting
        # for X"; the gate itself announces RECORDING when it really starts.
        self.gated = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._handler: _LerobotLogHandler | None = None
        self._warned = False

    # ── lifecycle ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running or not _rerun_active():
            return
        self.total = _argv_int("--dataset.num_episodes")
        self._install_hooks()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rerun-status")
        self._thread.start()

    def stop(self, final_phase: str | None = None) -> None:
        if final_phase:
            self.set_phase(final_phase)
            self._log()
        self._running = False
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler = None

    # ── events ──────────────────────────────────────────────────────────────────

    def on_recording(self, episode: int) -> None:
        with self._lock:
            self.retry = self.episode == episode and self.phase in ("RESETTING", "RECORDING")
            self.episode = episode
            if self.saved is None:
                self.saved = episode  # episodes already in the dataset (non-zero with --resume)
        command = sys.argv[0].rsplit("/", 1)[-1] if sys.argv else ""
        self.set_phase("WAITING" if self.gated else "POLICY" if "rollout" in command else "RECORDING")

    def on_saved(self, total_saved: int) -> None:
        with self._lock:
            self.saved = total_saved
        self.note(f"episode {self.episode} SAVED ({total_saved} in the dataset)")

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase, self.phase_t = phase, time.perf_counter()
        self._log()

    def note(self, text: str) -> None:
        with self._lock:
            self.last_note = text
        self._log()

    # ── rendering ───────────────────────────────────────────────────────────────

    def _render(self) -> str:
        with self._lock:
            phase, since, episode, retry = self.phase, time.perf_counter() - self.phase_t, self.episode, self.retry
            saved, total, note = self.saved, self.total, self.last_note
        icon, label = _PHASE_STYLE.get(phase, ("", phase))
        lines = []
        if episode is not None:
            of = f" of {total}" if total else ""
            lines.append(f"# {icon} {label} · episode {episode}{of}{' (re-record)' if retry else ''} · {since:.0f} s")
            lines.append(f"saved in the dataset: **{saved if saved is not None else 0}**{of}")
        else:
            lines.append(f"# {icon} {label} · {since:.0f} s")

        teleop = TELEOP_STATE.snapshot()
        if teleop["state"]:
            lines.extend(_quest_lines(teleop))
            line = f"**Quest:** {_TELEOP_STYLE.get(teleop['state'], teleop['state'])}"
            if teleop["state"] == "PAUSED" and teleop["reason"]:
                line += f"  \n reason: {teleop['reason']}"
            lines.append(line)
            if teleop["message"]:
                lines.append(f"teleop says: *{teleop['message']}*")
            if phase == "RESETTING":
                pending = "DISCARDED" if (teleop["last_key"] == "left" and teleop["last_key_t"] > self.phase_t - 5) else "SAVED"
                lines.append(f"episode {episode} will be **{pending}** when the reset ends"
                             + (" (Y now = discard instead)" if pending == "SAVED" else ""))

        fault = ROBOT_STATE.fault()
        if fault:
            lines.append(f"**⛔ ROBOT SAFETY HOLD:** {fault}")
        if note:
            lines.append(f"last: {note}")
        return "\n\n".join(lines)

    def _log(self) -> None:
        if not self._running:
            return
        try:
            import rerun as rr

            rr.log(ENTITY, rr.TextDocument(self._render(), media_type=rr.MediaType.MARKDOWN))
        except Exception:
            if not self._warned:
                logger.exception("rerun status panel: logging failed (display only, recording unaffected)")
                self._warned = True

    def _loop(self) -> None:
        while self._running:
            self._log()
            time.sleep(REFRESH_S)

    # ── hooks into lerobot ──────────────────────────────────────────────────────

    def _install_hooks(self) -> None:
        self._handler = _LerobotLogHandler(self)
        logging.getLogger().addHandler(self._handler)
        command = sys.argv[0].rsplit("/", 1)[-1] if sys.argv else ""
        self.set_phase("STARTING" if "record" in command else "ROLLOUT" if "rollout" in command else "TELEOP")
        _wrap_dataset(self)
        _wrap_blueprint()


def _wrap_dataset(board: StatusBoard) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if getattr(LeRobotDataset, "_openarm_status_wrapped", False):
        LeRobotDataset._openarm_status_board = board
        return
    LeRobotDataset._openarm_status_wrapped = True
    LeRobotDataset._openarm_status_board = board
    save, finalize = LeRobotDataset.save_episode, LeRobotDataset.finalize

    def save_episode(self, *args, **kwargs):
        b = LeRobotDataset._openarm_status_board
        b.set_phase("SAVING")
        out = save(self, *args, **kwargs)
        b.on_saved(self.num_episodes)
        return out

    def finalize_(self, *args, **kwargs):
        # lerobot-record finalizes twice (its video-encoding context manager on leaving the loop, then
        # record() itself); announce it once.
        if not getattr(self, "_openarm_finalize_announced", False):
            self._openarm_finalize_announced = True
            LeRobotDataset._openarm_status_board.set_phase("FINALIZING")
        return finalize(self, *args, **kwargs)

    LeRobotDataset.save_episode = save_episode
    LeRobotDataset.finalize = finalize_


def _wrap_blueprint() -> None:
    """Put a status text view above lerobot's own views (same views, same order, unchanged)."""
    import lerobot.utils.rerun_visualization as viz

    if getattr(viz, "_openarm_status_wrapped", False):
        return
    viz._openarm_status_wrapped = True

    def build_blueprint(observation_paths, action_paths, image_paths):
        import rerun.blueprint as rrb

        views = [rrb.Spatial2DView(origin=path, name=path) for path in sorted(image_paths)]
        if observation_paths:
            views.append(rrb.TimeSeriesView(name="observation", contents=sorted(observation_paths)))
        if action_paths:
            views.append(rrb.TimeSeriesView(name="action", contents=sorted(action_paths)))
        return rrb.Blueprint(
            rrb.Vertical(rrb.TextDocumentView(origin=ENTITY, name="status"), rrb.Grid(*views), row_shares=[1, 5])
        )

    viz._build_blueprint = build_blueprint


def _port_busy(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return False
        except OSError:
            return True


def _patch_spawn() -> None:
    """lerobot opens its viewer with rr.spawn() on rerun's default port 9876, and rerun silently REUSES
    any viewer already there -- e.g. one left open by an earlier session -- so this session's data goes
    to that old (often hidden) window and no new window appears. Open a fresh one on a free port instead.
    Runs when the plugin is imported, which lerobot does before it starts rerun."""
    try:
        import rerun as rr
    except Exception:
        return
    if getattr(rr, "_openarm_spawn_patched", False):
        return
    original = rr.spawn

    def spawn(*, port: int = 9876, **kwargs):
        if _port_busy(port):
            import socket

            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                free = s.getsockname()[1]
            print(f"[openarm] another rerun viewer already holds port {port} (an old session?): opening a NEW"
                  f" window on port {free} for this one. Close the old one with: pkill -f 'rerun --port={port}'",
                  flush=True)
            port = free
        return original(port=port, **kwargs)

    rr.spawn = spawn
    rr._openarm_spawn_patched = True


_patch_spawn()
BOARD = StatusBoard()
