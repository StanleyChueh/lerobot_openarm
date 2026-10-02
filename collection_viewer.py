#!/usr/bin/env python
"""Live view of the real cameras and the collection status, in a rerun window, while mirror_bridge.py
records a real-robot dataset (--record-root).

Kept off the mirror loop's critical path on purpose: the frames come from the cameras' own background
read threads (OpenCVCamera.read_latest, the same buffer the recorder reads -- nothing extra is asked
of the cameras), a daemon thread here downsizes and JPEG-encodes them at --view-hz, and drawing them
is done by the rerun viewer, a separate process. cv2's resize/encode and rerun's logging all release
the GIL, so the mirror loop is not held up by the encoding either.

The status panel shows what record_demos_openarm.py says it is doing ("status" episode events:
ready / recording / saving / discarding / resetting) plus the real recorder's own frame count.
"""

import logging
import os
import sys
import threading
import time

import cv2

logger = logging.getLogger("collection_viewer")

VIEW_SCALE = 0.5  # 640x480 -> 320x240: plenty to monitor by, a quarter of the encode cost
JPEG_QUALITY = 75

# Matches the state names record_demos_openarm.py sends.
STATE_COLORS = {
    "READY": "🟡",
    "RECORDING": "🔴",
    "RETURNING": "🟠",
    "SAVING": "🟢",
    "SAVED": "🟢",
    "DISCARDING": "⚪",
    "RESETTING": "🔵",
}


class CollectionViewer:
    def __init__(self, cameras: dict, hz: float, recorder=None):
        """cameras: {dataset_key: connected OpenCVCamera}. recorder: the RealEpisodeRecorder, read
        (never written) for its frame count."""
        import rerun as rr
        import rerun.blueprint as rrb

        self._rr = rr
        self._cameras = cameras
        self._period = 1.0 / hz
        self._recorder = recorder
        self._lock = threading.Lock()
        self._status = {"state": "STARTING", "episode": None, "total": None, "text": "waiting for the sim"}
        self._status_dirty = True

        # rr.spawn() looks the viewer binary up on PATH, and the bridge is started by the venv's
        # python directly, without the venv activated.
        os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
        rr.init("openarm_collection", spawn=False)
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.TextDocumentView(origin="status", name="Status"),
                rrb.Horizontal(*(rrb.Spatial2DView(origin=f"cameras/{key}", name=key) for key in cameras)),
                row_shares=[1, 4],
            ),
            collapse_panels=True,
        )
        # The viewer keeps every frame it is sent; cap it so an hours-long session does not eat RAM.
        rr.spawn(memory_limit="2GB", hide_welcome_screen=True, default_blueprint=blueprint)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="collection_viewer", daemon=True)
        self._thread.start()
        print(f"[VIEWER] rerun window up: {', '.join(cameras)} at {hz:g} Hz")

    def set_status(self, event: dict) -> None:
        """A "status" episode event from the sim: {"state", "episode", "total", "text"}."""
        with self._lock:
            self._status = {k: event.get(k) for k in ("state", "episode", "total", "text")}
            self._status_dirty = True

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        rr = self._rr
        next_tick = time.perf_counter()
        last_frames = None
        while not self._stop.is_set():
            try:
                rr.set_time("wall", timestamp=time.time())
                for key, cam in self._cameras.items():
                    try:
                        frame = cam.read_latest(max_age_ms=1000)
                    except (TimeoutError, RuntimeError):
                        continue
                    small = cv2.resize(frame, None, fx=VIEW_SCALE, fy=VIEW_SCALE, interpolation=cv2.INTER_AREA)
                    # Frames are RGB; imencode expects BGR.
                    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if ok:
                        rr.log(f"cameras/{key}", rr.EncodedImage(contents=jpeg.tobytes(), media_type="image/jpeg"))

                frames = self._recorder.frames if self._recorder is not None and self._recorder.recording else None
                with self._lock:
                    dirty = self._status_dirty or frames != last_frames
                    self._status_dirty = False
                    status = dict(self._status)
                if dirty:
                    last_frames = frames
                    rr.log("status", rr.TextDocument(self._status_markdown(status, frames),
                                                     media_type=rr.MediaType.MARKDOWN))
            except Exception:
                # The viewer is a convenience: it must never take the bridge (and the arm) down.
                logger.exception("[VIEWER] update failed -- continuing")

            next_tick += self._period
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                self._stop.wait(remaining)
            else:
                next_tick = time.perf_counter()

    def _status_markdown(self, status: dict, frames: int | None) -> str:
        state = status.get("state") or "?"
        episode, total = status.get("episode"), status.get("total")
        if episode is None:
            ep = ""
        elif total:
            ep = f"Episode **{episode} / {total}**  ·  "
        else:
            ep = f"Episode **{episode}**  ·  "
        line = f"## {ep}{STATE_COLORS.get(state, '')} {state}"
        if status.get("text"):
            line += f"\n\n{status['text']}"
        if frames is not None:
            line += f"\n\nreal frames: {frames} ({frames / self._recorder.fps:.1f}s)"
        return line
