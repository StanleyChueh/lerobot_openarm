"""Teleop diagnostics log: what happened between the Quest and the motors, at full rate.

Turned on by --teleop.diag=true (openarm_quest). Every stream is a table of float64 rows stamped with
time.perf_counter(), so they line up with each other:

  quest   one row per Quest packet the receiver kept: arrival, the headset's own clock, validity,
          both controllers' raw poses (Unity frame), triggers, packets drained with it
  ik      one row per IK-thread tick (500 Hz): tick interval, solve time, state, both EE targets,
          the raw IK solution and the rate-limited command
  obs     one row per robot read (lerobot-record's loop, 30 Hz): motor read time, total read time
          (cameras included), measured joints
  send    one row per robot command: interval since the previous one, desired, sent, velocity
          feed-forward, safety hold
  events  teleop messages, state changes, episode start / save / discard (events.jsonl)

Rows are kept in memory and written every FLUSH_S to outputs/teleop_diag/<start time>/chunk_*.npz, so a
crash loses at most that much. Report: python -m lerobot_robot_openarm_umeow.diag_report.
Cost when off: one attribute check per call site.
"""

import json
import threading
import time
from pathlib import Path

import numpy as np

FLUSH_S = 10.0


class DiagLog:
    def __init__(self) -> None:
        self.enabled = False
        self.dir: Path | None = None
        self._lock = threading.Lock()
        self._rows: dict[str, list] = {}
        self._columns: dict[str, list[str]] = {}
        self._events: list[dict] = []
        self._chunk = 0
        self._thread: threading.Thread | None = None

    def start(self, root: str | Path = "outputs/teleop_diag", **meta) -> Path:
        if self.enabled:
            return self.dir
        self.dir = Path(root) / time.strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "meta.json").write_text(json.dumps(
            {"wall_time": time.time(), "perf_counter": time.perf_counter(), **meta}, indent=1, default=str))
        self.enabled = True
        self._thread = threading.Thread(target=self._flush_loop, daemon=True, name="openarm-diag")
        self._thread.start()
        print(f"[openarm_diag] logging teleop diagnostics to {self.dir}", flush=True)
        return self.dir

    def row(self, stream: str, columns, values) -> None:
        """One row of `stream`, named by `columns` (a constant per stream; column 0 is perf_counter())."""
        arr = np.asarray(values, dtype=np.float64)
        with self._lock:
            if stream not in self._columns:
                self._columns[stream] = list(columns)
            self._rows.setdefault(stream, []).append(arr)

    def event(self, text: str, **fields) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._events.append({"t": time.perf_counter(), "text": text, **fields})

    def stop(self) -> None:
        if not self.enabled:
            return
        self.enabled = False
        self._flush()
        print(f"[openarm_diag] saved -- report: python -m lerobot_robot_openarm_umeow.diag_report {self.dir}",
              flush=True)

    def _flush_loop(self) -> None:
        while self.enabled:
            time.sleep(FLUSH_S)
            if self.enabled:
                self._flush()

    def _flush(self) -> None:
        with self._lock:
            rows, self._rows = self._rows, {k: [] for k in self._rows}
            events, self._events = self._events, []
            columns = dict(self._columns)
        try:
            data = {k: np.stack(v) for k, v in rows.items() if v}
            if data:
                self._chunk += 1
                np.savez(self.dir / f"chunk_{self._chunk:05d}.npz", **data)
            (self.dir / "columns.json").write_text(json.dumps(columns, indent=1))
            if events:
                with open(self.dir / "events.jsonl", "a") as f:
                    for e in events:
                        f.write(json.dumps(e, default=str) + "\n")
        except Exception as e:  # never take the recording down
            print(f"[openarm_diag] write failed: {e!r}", flush=True)


DIAG = DiagLog()


def load(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, list[str]], list[dict], dict]:
    """(streams, columns, events, meta) of one log folder; every stream's column 0 is perf_counter()."""
    path = Path(path)
    parts: dict[str, list] = {}
    for f in sorted(path.glob("chunk_*.npz")):
        with np.load(f) as z:
            for k in z.files:
                parts.setdefault(k, []).append(z[k])
    streams = {k: np.concatenate(v) for k, v in parts.items()}
    columns = json.loads((path / "columns.json").read_text()) if (path / "columns.json").exists() else {}
    events = []
    if (path / "events.jsonl").exists():
        events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines() if line]
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
    return streams, columns, events, meta
