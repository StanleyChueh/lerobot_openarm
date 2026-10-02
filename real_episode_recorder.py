#!/usr/bin/env python
"""Record the REAL robot's cameras and joints into a LeRobot v3 dataset while mirror_bridge.py
makes it follow an Isaac Sim teleop session.

Used by mirror_bridge.py's --record-root (which record_demos_openarm.py's --real_arm_dataset sets).
The Isaac Sim process still owns the episode lifecycle -- button X / N / Y / R and the auto-success
check all live there -- so it sends each decision here as a UDP JSON event on a separate port:

  {"event": "start" | "save" | "reset", "seq": int, "t": float, "demo": int}

(It also sends {"event": "status", ...} for the operator's display; mirror_bridge.py hands those to
collection_viewer.py and they never reach this recorder.)

  start  -- button X armed the episode (or a reset finished, in the always-armed modes). Frames are
            recorded from here on.
  save   -- the sim exported this episode as a demo. The real episode is saved too.
  reset  -- the sim reset the scene. An episode still being recorded was NOT saved, so it is dropped.

Events go to their own port rather than riding on the joint packets because the bridge's joint
receiver keeps only the newest packet: a "save" overtaken by the next joint packet would be lost.

What one frame holds, in exactly the layout the sim datasets and the SmolVLA deploy scripts use
(16D, LJ1.pos..LJ8.pos then RJ1.pos..RJ8.pos, arm joints in sim radians, grippers in the sim's
0.0 (closed) .. 0.044 (open) finger-joint convention -- see convert_hdf5_to_lerobot.py in IsaacLab
and _get_obs_sim_gripper in deploy_smolvla_pickup_jointspace.py):

  observation.state          the real arm's MEASURED joints this tick, mapped through calibration.json
  action                     --record-action-source command (default): the sim joint target the real
                             arm was told to follow this tick (the mirror packet, before the bridge's
                             speed clamp), grippers binarised to 0.0/0.044 by the sim side.
                             next_state: the NEXT tick's measured state -- the same proxy
                             convert_hdf5_to_lerobot.py uses for sim data; costs each episode's last frame.
  observation.images.<cam>   the real cameras, 640x480 RGB, streamed straight into the episode's MP4
                             (streaming_encoding), so save_episode() does not stall the mirror loop.
"""

import json
import os
import queue
import shutil
import socket
import threading
import time

import numpy as np

from sim_bridge_common import motor_action_to_sim_joints

STATE_NAMES = [f"LJ{i}.pos" for i in range(1, 9)] + [f"RJ{i}.pos" for i in range(1, 9)]
# The sim joint each dataset column comes from. Only finger_joint1 per gripper: joint2 is its mimic.
SIM_JOINT_FOR_NAME = {
    **{f"{p}J{i}.pos": f"openarm_{side}_joint{i}" for p, side in (("L", "left"), ("R", "right")) for i in range(1, 8)},
    "LJ8.pos": "openarm_left_finger_joint1",
    "RJ8.pos": "openarm_right_finger_joint1",
}
GRIPPER_MIN, GRIPPER_MAX = 0.0, 0.044

CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
# A camera frame older than this means the camera stopped delivering; the episode is then dropped
# on save rather than written with a frozen view in it.
CAMERA_MAX_AGE_MS = 200

DEFAULT_CAMERAS = "body_cam=rs_body,wrist_cam=rs_wrist_left,right_wrist_cam=rs_wrist_right"


def parse_camera_spec(spec: str) -> dict[str, str]:
    """"body_cam=rs_body,wrist_cam=rs_wrist_left" -> {"body_cam": "rs_body", ...}."""
    cameras = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        key, sep, device = item.partition("=")
        if not sep or not key or not device:
            raise ValueError(f"bad camera entry {item!r} -- expected <dataset_key>=<video index or udev alias>")
        cameras[key] = device
    return cameras


def sim_joints_to_vector(sim_joints: dict) -> np.ndarray:
    vec = np.array([sim_joints[SIM_JOINT_FOR_NAME[n]] for n in STATE_NAMES], dtype=np.float32)
    vec[[7, 15]] = np.clip(vec[[7, 15]], GRIPPER_MIN, GRIPPER_MAX)
    return vec


class EpisodeEventReceiver:
    """Background UDP listener that queues EVERY event, in order (unlike LatestPacketReceiver)."""

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._queue.put(json.loads(data.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    def drain(self) -> list[dict]:
        events = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                return events

    def stop(self):
        self._stop.set()
        self._sock.close()


class RealEpisodeRecorder:
    """Owns the cameras and the LeRobotDataset. Everything is called from the bridge's main loop."""

    def __init__(self, *, root: str, repo_id: str | None, task: str, fps: int, cameras: dict[str, str],
                 calib: dict, resume: bool, overwrite: bool, action_source: str, vcodec: str):
        if action_source not in ("command", "next_state"):
            raise ValueError(f"action_source must be 'command' or 'next_state', got {action_source!r}")
        self.root = os.path.abspath(os.path.expanduser(root))
        self.repo_id = repo_id or f"local/{os.path.basename(self.root.rstrip('/'))}"
        self.task = task
        self.fps = fps
        self.camera_devices = cameras
        self.calib = calib
        self.resume = resume
        self.overwrite = overwrite
        self.action_source = action_source
        self.vcodec = vcodec

        self.dataset = None
        self.cameras = {}
        self.recording = False
        self._frames = 0
        self._broken_reason = None
        self._pending = None  # next_state only: the frame still waiting for its action
        self._saved_this_session = 0

    @property
    def frames(self) -> int:
        """Frames added to the episode being recorded so far."""
        return self._frames

    # ── setup / teardown ───────────────────────────────────────────────────────

    def connect(self) -> None:
        # Imported here so mirror_bridge.py without --record-root never pays for lerobot's dataset
        # stack or the deploy module (which pulls in torch).
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        from deploy_smolvla_async import _video_index
        from deploy_smolvla_pickup_jointspace import _usb_reset_for_video_node

        features = {
            "action": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
            "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
            **{
                f"observation.images.{key}": {
                    "dtype": "video", "shape": (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
                    "names": ["height", "width", "channels"],
                }
                for key in self.camera_devices
            },
        }

        writer_kwargs = dict(
            # Named explicitly so lerobot does not probe for torchcodec, which cannot load in this
            # venv (no matching FFmpeg) and prints a page of tracebacks saying so. The backend only
            # matters for DECODING; recording never decodes.
            video_backend="pyav",
            streaming_encoding=True,
            encoder_queue_maxsize=60,
            encoder_threads=2,
            rgb_encoder=RGBEncoderConfig(vcodec=self.vcodec),
        )
        exists = os.path.exists(os.path.join(self.root, "meta", "info.json"))
        if exists and self.overwrite:
            print(f"[REAL REC] --overwrite: removing {self.root}")
            shutil.rmtree(self.root)
            exists = False
        if exists and not self.resume:
            raise SystemExit(f"[REAL REC] {self.root} already holds a dataset. Pass --resume to append"
                             " to it or --overwrite to start again.")
        if exists:
            self.dataset = LeRobotDataset.resume(self.repo_id, root=self.root, **writer_kwargs)
            have = {k: tuple(v["shape"]) for k, v in self.dataset.features.items() if k in features}
            want = {k: tuple(v["shape"]) for k, v in features.items()}
            if have != want or self.dataset.fps != self.fps:
                raise SystemExit(f"[REAL REC] {self.root} was recorded with different features/fps"
                                 f" ({have} @ {self.dataset.fps}fps) than this run ({want} @ {self.fps}fps).")
            print(f"[REAL REC] Resuming {self.root}: {self.dataset.num_episodes} episodes already in it.")
        else:
            if os.path.exists(self.root):
                # LeRobotDatasetMetadata.create refuses an existing directory, even an empty one.
                os.rmdir(self.root)
            self.dataset = LeRobotDataset.create(
                self.repo_id, self.fps, features, root=self.root, robot_type="openarm",
                use_videos=True, **writer_kwargs,
            )
            print(f"[REAL REC] Created {self.root} (repo_id {self.repo_id}).")

        indices = {key: _video_index(dev) for key, dev in self.camera_devices.items()}
        for index in indices.values():
            _usb_reset_for_video_node(index)
        time.sleep(1.0)
        for key, index in indices.items():
            cam = OpenCVCamera(OpenCVCameraConfig(
                index_or_path=index, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=self.fps))
            cam.connect()
            self.cameras[key] = cam
            print(f"[REAL REC] {key} <- /dev/video{index} ({self.camera_devices[key]})")
        print(f"[REAL REC] task: {self.task!r}  |  action = {self.action_source}  |  {self.fps} fps")

    def close(self) -> None:
        if self.dataset is not None:
            if self.recording:
                print("[REAL REC] Shutting down mid-episode -- that episode is discarded.")
                self._discard()
            self.dataset.finalize()
            print(f"[REAL REC] Finalized {self.root}: {self.dataset.meta.total_episodes} episodes"
                  f" ({self._saved_this_session} this session).")
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass

    # ── episode lifecycle (events from the sim) ────────────────────────────────

    def handle_event(self, event: dict) -> None:
        kind = event.get("event")
        demo = event.get("demo")
        if kind == "start":
            if self.recording:
                print("[REAL REC] start while an episode was still open -- discarding it.")
                self._discard()
            self.recording = True
            self._frames = 0
            self._broken_reason = None
            self._pending = None
            print(f"[REAL REC] Recording real episode {self.dataset.meta.total_episodes} (sim demo {demo}).")
        elif kind == "save":
            if not self.recording:
                print("[REAL REC] save received with no episode being recorded -- ignored.")
                return
            self._pending = None  # next_state: the last frame has no next state; drop it
            if self._broken_reason is not None:
                print(f"[REAL REC] NOT saving this episode: {self._broken_reason}. The sim kept its"
                      " demo, so the two datasets now differ by one episode.")
                self._discard()
            elif self._frames < 2:
                print("[REAL REC] NOT saving: the episode has fewer than 2 frames.")
                self._discard()
            else:
                t0 = time.perf_counter()
                try:
                    self.dataset.save_episode()
                except Exception as e:
                    print(f"[REAL REC] Saving the episode FAILED ({e}) -- discarded.")
                    self._discard()
                    return
                self.recording = False
                self._saved_this_session += 1
                print(f"[REAL REC] Saved real episode {self.dataset.meta.total_episodes - 1}:"
                      f" {self._frames} frames ({self._frames / self.fps:.1f}s), save took"
                      f" {(time.perf_counter() - t0) * 1e3:.0f}ms. Total {self.dataset.meta.total_episodes}.")
        elif kind == "reset":
            if self.recording:
                print(f"[REAL REC] Episode discarded ({self._frames} frames).")
                self._discard()
        else:
            print(f"[REAL REC] Unknown event {event!r} -- ignored.")

    def _discard(self) -> None:
        try:
            self.dataset.clear_episode_buffer()
        except Exception as e:
            print(f"[REAL REC] clearing the episode buffer failed: {e}")
        self.recording = False
        self._frames = 0
        self._pending = None

    # ── per tick ───────────────────────────────────────────────────────────────

    def record(self, measured_motor_action: dict | None, commanded_sim_joints: dict) -> None:
        """Add this tick's frame. measured_motor_action is get_current_pos_action(robot) (raw motor
        units, None if that read failed); commanded_sim_joints is the latest mirror packet's joints
        (sim names/units)."""
        if not self.recording or self._broken_reason is not None:
            return
        if measured_motor_action is None:
            # A skipped tick would silently shift every later frame's timestamp by 1/fps.
            self._broken_reason = "a joint read failed mid-episode"
            print(f"[REAL REC] {self._broken_reason} -- this episode will be discarded.")
            return
        state = sim_joints_to_vector(motor_action_to_sim_joints(measured_motor_action, self.calib))
        images = {}
        for key, cam in self.cameras.items():
            try:
                images[f"observation.images.{key}"] = cam.read_latest(max_age_ms=CAMERA_MAX_AGE_MS)
            except (TimeoutError, RuntimeError) as e:
                if self._broken_reason is None:
                    self._broken_reason = f"camera {key} stopped delivering frames ({e})"
                    print(f"[REAL REC] {self._broken_reason} -- this episode will be discarded.")
                return

        if self.action_source == "command":
            self._add(state, sim_joints_to_vector(commanded_sim_joints), images)
        else:
            if self._pending is not None:
                prev_state, prev_images = self._pending
                self._add(prev_state, state, prev_images)
            self._pending = (state, images)

    def _add(self, state: np.ndarray, action: np.ndarray, images: dict) -> None:
        # Nothing the dataset writer raises (an encoder thread dying, a full disk) may reach the
        # mirror loop that calls this: it would stop the real arm mid-demo. The episode is dropped.
        try:
            self.dataset.add_frame({"observation.state": state, "action": action, **images, "task": self.task})
        except Exception as e:
            self._broken_reason = f"the dataset writer failed ({e})"
            print(f"[REAL REC] {self._broken_reason} -- this episode will be discarded.")
            return
        self._frames += 1
