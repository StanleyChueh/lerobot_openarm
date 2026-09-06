"""
Asynchronous GR00T N1.7 rollout on the real OpenArm follower -- the GR00T counterpart of
deploy_smolvla_async.py, same wire-free threading split (control thread vs inference thread), same
hardware layer (cameras, calibration, gripper binarization, speed clamp), different model.

Read deploy_smolvla_async.py's own docstring first for WHY async exists at all (decoupling
inference from the control period) and for the full RATE / ActionQueue / ObservationBus
explanation -- none of that is repeated here because none of it changed. This docstring only
covers what is different for GR00T, all found and fixed while getting
scripts/imitation_learning/lerobot/gr00t_server.py (IsaacLab, same checkpoint family) working:

1. THIS CHECKPOINT USES RELATIVE ACTIONS -- predict_action_chunk() output is joint-position DELTAS
   relative to the observation the chunk was predicted from (see its config.json:
   use_relative_actions=True, relative_exclude_joints=["LJ8", "RJ8"] for the two grippers, which
   stay absolute). That has one consequence this file cannot copy from the SmolVLA version:
   deploy_smolvla_async.py's AsyncPolicy._predict_chunk() postprocesses ONE TIMESTEP AT A TIME
   (`postprocess(chunk[:, i, :])` in a loop) because SmolVLA's actions are absolute and independent
   per step. GR00T's own postprocessor step (GrootN17ActionDecodeStep) explicitly REJECTS that:

       "GrootN17ActionDecodeStep cannot decode native relative actions one step at a time. Decode
       the full action chunk returned by predict_action_chunk while the matching
       GrootN17PackInputsStep state is still cached, then queue the decoded absolute actions."

   Each row of the chunk is a delta relative to the SAME cached observation state, added back by
   the decode step; feeding it one row at a time collapses the batch dimension it uses to do that
   and raises NotImplementedError. So _predict_chunk here calls self.postprocess() ONCE on the
   whole (1, actions_per_chunk, ACTION_DIM) tensor -- still stateless overall (nothing carries
   between calls; each forward pass supplies its own reference state) -- and only splits into
   per-timestep dicts (make_robot_action per row) AFTER that single decode call.

2. LOADING: strict=False and a CPU-first load, not model.to(device) then done.
   - strict=False: GR00T's Qwen backbone ties its (unused) LM head weight to the input embedding
     (see lerobot's groot_n1_7._tie_unused_qwen_lm_head). The checkpoint's safetensors file lists
     embed_tokens.weight separately from that tie; the freshly constructed (already-tied) model
     does not, so a strict load raises "Unexpected key(s)" on a weight that loads correctly under
     lm_head.weight either way -- confirmed harmless, not a missing weight.
   - CPU-first: GrootPolicy.from_pretrained() (via the base PreTrainedPolicy.from_pretrained) moves
     the policy to config.device BEFORE returning, and this checkpoint's saved config.json says
     device="cuda". Casting to bf16 AFTER that needs the fp32 and bf16 copies resident on the GPU
     at the same time, and CUDA's caching allocator does not hand the freed fp32 blocks back to the
     driver afterward -- measured on the IsaacLab side: 13.6GB resident, barely under the full fp32
     footprint, instead of the ~6GB bf16 needs. Forcing config.device="cpu" before from_pretrained()
     keeps the fp32 load AND the cast in system RAM (plentiful), so the GPU only ever receives the
     already-bf16-sized model in the one .to(device) call that follows. See --dtype below.

3. CHUNK SIZE: this checkpoint's chunk_size/n_action_steps is 16, not SmolVLA's 50 (see its
   config.json). --actions-per-chunk here is bounded by THIS checkpoint's chunk_size, so
   deploy_smolvla_async.py's own example (--actions-per-chunk 50) would be rejected outright --
   leave --actions-per-chunk unset (defaults to model.config.chunk_size) unless you have a specific
   reason to keep fewer than the full 16.

4. TASK / cameras / action layout are UNCHANGED: this checkpoint
   (ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz_gr00t) is a GR00T fine-tune of the
   exact same dataset the SmolVLA checkpoint above is, converted through the same pipeline, so the
   16D LJ1..LJ8+RJ1..RJ8 layout, ACTION_NAMES, TASK string, and the observation.images.{body_cam,
   wrist_cam,right_wrist_cam} camera keys (confirmed against this checkpoint's own config.json --
   no --rename_map was used) all match deploy_smolvla_pickup_jointspace.py's constants exactly, so
   this file imports them unchanged rather than redefining them.

Usage (cameras/calibration identical to the SmolVLA scripts; --actions-per-chunk left at this
checkpoint's own chunk_size of 16 -- do not copy SmolVLA's --actions-per-chunk 50):
  python deploy_gr00t_async.py \\
      --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background_30hz_gr00t \\
      --body-cam-index rs_body --wrist-cam-index rs_wrist_left --right-wrist-cam-index rs_wrist_right \\
      --calibration calibration.json \\
      --control-hz 30 --max-joint-speed 1.5 --chunk-size-threshold 0.8 \\
      --max-episode-seconds 25 --max-episodes 20

Everything from ActionQueue down through the control loop and plotting is a straight copy of
deploy_smolvla_async.py's own classes/loop (the async harness itself has nothing policy-specific in
it) with only the two changes above; consult that file's docstring for the reasoning behind
--chunk-size-threshold, --aggregate-fn, the speed clamp, and the tracking plots.
"""

import argparse
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.utils.feature_utils import hw_to_dataset_features

from deploy_smolvla_pickup_jointspace import (
    ACTION_DIM,
    ACTION_NAMES,
    FPS,
    GRIPPER_IDX,
    GRIPPER_MAX,
    GRIPPER_MIN,
    LIVE_VIEW_WINDOW,
    ROBOT_TYPE,
    TASK,
    URDF_PATH,
    _binarize_gripper,
    _build_combined_frame,
    _check_checkpoint_matches,
    _cv2_has_gui,
    _get_obs_sim_gripper,
    _gripper_calib,
    _send_action_sim_gripper,
    _usb_reset_for_video_node,
    GRIPPER_OPEN_CMD,
)
from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from sim_bridge_common import (
    approach_pose,
    check_arms_not_crossed,
    load_calibration,
    sim_init_pose_action,
)

# lerobot/async_inference/configs.py's AGGREGATE_FUNCTIONS, verbatim -- see deploy_smolvla_async.py
# for why a weighted blend rather than a hard swap.
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


@dataclass
class TimedAction:
    """One action, tagged with the control-tick index it is meant to be executed on.

    See deploy_smolvla_async.py's TimedAction docstring -- unchanged reasoning, timestep rather
    than arrival order is what makes the aggregation in ActionQueue.ingest coherent."""

    timestep: int
    action: np.ndarray  # (ACTION_DIM,) absolute joint targets, sim convention (0.0..0.044 grippers)


class ActionQueue:
    """Timestep-keyed action queue shared by the control and inference threads.

    Identical to deploy_smolvla_async.py's ActionQueue -- the scheduling policy (dict keyed by
    timestep, popped by min(keys)) has nothing to do with which policy fills it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._actions: dict[int, np.ndarray] = {}
        self._latest_executed = -1
        self.chunk_size = 1  # largest chunk seen, for the fill-ratio trigger

    def ingest(self, chunk: np.ndarray, first_timestep: int, aggregate_fn) -> tuple[int, int, int]:
        """Merge a freshly predicted chunk in. Returns (dropped, merged, added) for logging."""
        dropped = merged = added = 0
        with self._lock:
            self.chunk_size = max(self.chunk_size, len(chunk))
            for i, action in enumerate(chunk):
                ts = first_timestep + i
                if ts <= self._latest_executed:
                    dropped += 1
                elif ts in self._actions:
                    self._actions[ts] = aggregate_fn(self._actions[ts], action)
                    merged += 1
                else:
                    self._actions[ts] = action
                    added += 1
        return dropped, merged, added

    def pop(self) -> TimedAction | None:
        """Take the action for the earliest queued tick, or None if the queue has drained."""
        with self._lock:
            if not self._actions:
                return None
            ts = min(self._actions)
            action = self._actions.pop(ts)
            self._latest_executed = ts
            return TimedAction(timestep=ts, action=action)

    def flush(self) -> None:
        """Drop every queued action and restart tick numbering -- called between episodes."""
        with self._lock:
            self._actions.clear()
            self._latest_executed = -1

    @property
    def latest_executed(self) -> int:
        with self._lock:
            return self._latest_executed

    def qsize(self) -> int:
        with self._lock:
            return len(self._actions)

    def fill_ratio(self) -> float:
        """Queued actions as a fraction of one chunk -- the re-plan trigger's own quantity."""
        with self._lock:
            return len(self._actions) / max(self.chunk_size, 1)


class ObservationBus:
    """Single-slot handoff of the latest observation to the inference thread.

    Identical to deploy_smolvla_async.py's ObservationBus -- maxsize=1 with overwrite, since an
    observation superseded before inference picks it up is worthless regardless of which policy
    would have consumed it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._slot: tuple[dict, int] | None = None
        self._ready = threading.Event()

    def publish(self, obs: dict, timestep: int) -> None:
        snapshot = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        with self._lock:
            self._slot = (snapshot, timestep)
        self._ready.set()

    def take(self, timeout: float) -> tuple[dict, int] | None:
        """Block until an observation is published, then consume it. None on timeout."""
        if not self._ready.wait(timeout):
            return None
        with self._lock:
            slot, self._slot = self._slot, None
            self._ready.clear()
        return slot

    def clear(self) -> None:
        with self._lock:
            self._slot = None
            self._ready.clear()


class AsyncPolicy:
    """The inference thread: latest observation in, action chunk out, forever.

    Structurally identical to deploy_smolvla_async.py's AsyncPolicy. The one real change is inside
    _predict_chunk: the postprocess() call moves from per-timestep (a loop over chunk[:, i, :]) to
    once on the whole chunk, because this checkpoint's relative-action decode step requires the
    batch dimension intact to read back its cached reference state -- see the module docstring."""

    def __init__(self, *, model, preprocess, postprocess, dataset_features, device,
                 actions_per_chunk, aggregate_fn, chunk_size_threshold, obs_bus, action_queue,
                 verbose):
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.dataset_features = dataset_features
        self.device = device
        self.actions_per_chunk = actions_per_chunk
        self.aggregate_fn = aggregate_fn
        self.chunk_size_threshold = chunk_size_threshold
        self.obs_bus = obs_bus
        self.action_queue = action_queue
        self.verbose = verbose

        self.shutdown = threading.Event()
        self.inference_times: deque[float] = deque(maxlen=50)
        self.replans = 0
        self.discarded = 0
        self.thread = threading.Thread(target=self._worker, name="inference", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.shutdown.set()
        self.thread.join(timeout=5.0)

    @torch.inference_mode()
    def _predict_chunk(self, obs: dict) -> np.ndarray:
        """One forward pass -> (K, ACTION_DIM) absolute joint targets in the sim convention.

        predict_action_chunk() rather than select_action(): GrootPolicy.select_action() explicitly
        raises NotImplementedError for a relative-action checkpoint like this one (cached queued
        actions would be decoded against a newer, wrong, observation state) -- see module
        docstring point 1. Asking for the raw chunk and owning the queue ourselves is not a
        workaround for that restriction, it is the documented way around it: GrootPolicy's own
        select_action docstring says to "use predict_action_chunk and postprocess the full chunk
        before queuing actions", which is exactly this method.
        """
        frame = build_inference_frame(
            observation=obs,
            ds_features=self.dataset_features,
            device=self.device,
            task=TASK,
            robot_type=ROBOT_TYPE,
        )
        chunk = self.model.predict_action_chunk(self.preprocess(frame))
        if chunk.ndim == 2:  # (chunk_size, action_dim) -- restore the batch dim postprocess wants
            chunk = chunk.unsqueeze(0)
        chunk = chunk[:, : self.actions_per_chunk, :]

        # ONE call on the whole (1, actions_per_chunk, ACTION_DIM) chunk -- NOT a per-timestep
        # loop like deploy_smolvla_async.py's. GrootN17ActionDecodeStep reads the reference state
        # this chunk's deltas are relative to off a cache the preprocess() call above just filled,
        # and raises NotImplementedError if handed a 2D (B, D) slice instead of the full 3D
        # (B, T, D) chunk -- it cannot decode one relative step in isolation. Splitting into
        # per-timestep dicts only happens AFTER the chunk is already absolute.
        chunk = self.postprocess(chunk)
        actions = [
            make_robot_action(chunk[:, i, :], self.dataset_features)
            for i in range(chunk.shape[1])
        ]
        return np.array(
            [[a[name] for name in ACTION_NAMES] for a in actions], dtype=np.float64
        )

    def _worker(self) -> None:
        while not self.shutdown.is_set():
            taken = self.obs_bus.take(timeout=0.1)
            if taken is None:
                continue
            obs, timestep = taken

            # Re-check the trigger on THIS thread before spending a forward pass -- see
            # deploy_smolvla_async.py's identical comment for the measured reason (avoids doubling
            # the re-plan count for near-zero additional coverage).
            if self.action_queue.fill_ratio() > self.chunk_size_threshold:
                self.discarded += 1
                continue

            start = time.perf_counter()
            try:
                chunk = self._predict_chunk(obs)
            except Exception as e:  # a dead inference thread must not leave the arm running blind
                print(f"[ERROR] inference failed, stopping the rollout: {e!r}", flush=True)
                self.shutdown.set()
                raise
            elapsed = time.perf_counter() - start
            self.inference_times.append(elapsed)
            self.replans += 1

            dropped, merged, added = self.action_queue.ingest(chunk, timestep, self.aggregate_fn)
            if self.verbose:
                print(
                    f"[replan {self.replans}] from tick {timestep} | {elapsed * 1000:.0f}ms | "
                    f"chunk {len(chunk)} -> dropped {dropped} (stale), merged {merged}, "
                    f"added {added} | queue {self.action_queue.qsize()}",
                    flush=True,
                )


def _video_index(spec: str) -> int:
    """Camera CLI argument -> /dev/video index. Identical to deploy_smolvla_async.py's -- see
    there for why the udev-alias form is preferred over a bare index."""
    if spec.isdigit():
        return int(spec)

    path = spec if os.path.isabs(spec) else os.path.join("/dev", spec)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(
            f"{spec!r} is neither a /dev/video index nor an existing device path ({path} not found)")

    node = os.path.basename(os.path.realpath(path))
    if not re.fullmatch(r"video\d+", node):
        raise argparse.ArgumentTypeError(f"{spec!r} resolves to {node}, which is not a /dev/videoN node")
    return int(node[len("video"):])


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="HF repo id or local path of the joint-space GR00T N1.7 checkpoint.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"],
                        help="Parameter dtype to cast the loaded policy to before moving it onto "
                             "--device. This checkpoint's fp32 master weights (~12GB for the 3B "
                             "backbone) are cast down in system RAM before the one GPU transfer -- "
                             "see module docstring point 2 for why that ordering matters (casting "
                             "after the model is already on the GPU measured 13.6GB resident "
                             "instead of ~6GB). predict_action_chunk() already runs its forward "
                             "pass under torch.autocast(dtype=torch.bfloat16) whenever "
                             "config.use_bf16 (True for this checkpoint) regardless of stored "
                             "parameter dtype, so bf16 storage does not change what is computed.")
    parser.add_argument("--body-cam-index", type=_video_index, required=True,
                        help="Camera for observation.images.body_cam. "
                             "Takes a /dev/video index or a udev alias (rs_body).")
    parser.add_argument("--wrist-cam-index", type=_video_index, required=True,
                        help="Camera for observation.images.wrist_cam (the LEFT wrist). "
                             "Takes a /dev/video index or a udev alias (rs_body).")
    parser.add_argument("--right-wrist-cam-index", type=_video_index, default=None,
                        help="Camera for observation.images.right_wrist_cam (dual-arm checkpoints). "
                             "Takes a /dev/video index or a udev alias (rs_body).")
    parser.add_argument("--front-cam-index", type=_video_index, default=None,
                        help="Camera for observation.images.front_cam (older variants). "
                             "Takes a /dev/video index or a udev alias (rs_body).")
    parser.add_argument("--side-cam-index", type=_video_index, default=None,
                        help="EXTRA camera for the live view only -- never fed to the model. "
                             "Takes a /dev/video index or a udev alias (rs_body).")
    parser.add_argument("--calibration", type=str, required=True,
                        help="calibration.json -- required for the gripper raw<->sim mapping (see "
                             "the GRIPPER CALIBRATION note in deploy_smolvla_pickup_jointspace.py).")

    parser.add_argument("--control-hz", type=float, default=30.0,
                        help="Control-loop rate. This checkpoint was fine-tuned from the "
                             "'_30hz' dataset variant, so 30 plays its chunk back at the speed its "
                             "demos were recorded at -- see deploy_smolvla_async.py's --control-hz "
                             "help for why this is not the reactivity knob (--chunk-size-threshold "
                             "is).")
    parser.add_argument("--chunk-size-threshold", type=float, default=0.8,
                        help="Re-plan once the queue holds less than this fraction of a chunk. "
                             "With this checkpoint's chunk_size=16 (not SmolVLA's 50), the re-plan "
                             "interval (1 - this) * --actions-per-chunk ticks is already short at "
                             "the default -- 0.8 is ~3 ticks (~100ms at 30Hz), floored at one "
                             "forward pass. 0.0 waits for the queue to drain entirely.")
    parser.add_argument("--actions-per-chunk", type=int, default=None,
                        help="How many actions to keep from each predicted chunk (default: this "
                             "checkpoint's own chunk_size, 16 -- NOT SmolVLA's 50; passing a value "
                             "above the checkpoint's chunk_size is rejected). Bounds how far ahead "
                             "the queue can run, i.e. how stale an action can get if inference "
                             "stalls; it does not set the re-plan interval on its own -- "
                             "--chunk-size-threshold does.")
    parser.add_argument("--aggregate-fn", choices=tuple(AGGREGATE_FUNCTIONS), default="weighted_average",
                        help="How an incoming chunk blends with the queued actions it overlaps.")
    parser.add_argument("--max-joint-speed", type=float, default=1.0,
                        help="rad/s ceiling for all 16 joints, enforced by clamping how far the "
                             "queued target may move from the CURRENT measured joints each control "
                             "tick. See deploy_smolvla_async.py's --max-joint-speed help for the "
                             "demos' own measured joint-speed percentiles (same dataset).")
    parser.add_argument("--max-episode-seconds", type=float, default=30.0,
                        help="Wall-clock limit per episode.")
    parser.add_argument("--max-episodes", type=int, default=1,
                        help="Episodes to run. Each one ramps back to Isaac Sim's reset pose and "
                             "flushes the action queue -- the re-arm mechanism between episodes.")
    parser.add_argument("--episode-gap-seconds", type=float, default=5.0,
                        help="Pause after each episode's reset pose, before the policy is given "
                             "control -- time to take the can out of the left gripper and place a "
                             "new one.")

    parser.add_argument("--no-start-pose", action="store_true",
                        help="Skip the ramp to Isaac Sim's reset pose before each episode.")
    parser.add_argument("--start-pose-speed", type=float, default=0.3,
                        help="rad/s ceiling for the move to the start pose.")
    parser.add_argument("--max-start-pose-delta", type=float, default=1.8,
                        help="rad; refuse to run if any joint would have to travel further than this.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the typed YES before the FIRST start-pose move (later episodes never prompt).")

    parser.add_argument("--can-recv-rounds", type=int, default=16,
                        help="recv_all() calls per position read, per arm.")
    parser.add_argument("--can-first-timeout-us", type=int, default=2_000,
                        help="Microseconds the first recv_all() waits.")
    parser.add_argument("--can-mop-timeout-us", type=int, default=200,
                        help="Microseconds each mop-up recv_all() waits.")

    parser.add_argument("--no-live-view", action="store_true",
                        help="Disable the live cv2 camera window.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress the per-replan and per-second control-loop lines.")

    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the target-vs-actual tracking plots shown on shutdown.")
    parser.add_argument("--save-plot", type=str, default=None,
                        help="Also write the tracking plots to this path (a '-raw' variant is "
                             "written alongside it). Useful over SSH, where plt.show() has no "
                             "window to open.")
    parser.add_argument("--plot-max-samples", type=int, default=24_000,
                        help="Ticks of tracking history to keep, per joint.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.chunk_size_threshold <= 1.0:
        raise SystemExit(f"--chunk-size-threshold must be in [0, 1], got {args.chunk_size_threshold}")

    calib = load_calibration(args.calibration)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] loading {args.checkpoint} on {device} (dtype={args.dtype})...")

    # CPU-first load -- see module docstring point 2. config.device is forced to "cpu" BEFORE
    # from_pretrained() so the fp32 checkpoint (and the dtype cast below) stay in system RAM; only
    # the already-cast, already-small model is moved to the GPU.
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = "cpu"
    # strict=False: see module docstring point 2 -- the tied Qwen LM-head/embed_tokens key.
    model = GrootPolicy.from_pretrained(args.checkpoint, config=config, strict=False)
    torch_dtype = getattr(torch, args.dtype)
    if torch_dtype is not torch.float32:
        model.to(dtype=torch_dtype)
    model.config.device = device
    model.to(device)
    model.eval()

    preprocess, postprocess = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    actions_per_chunk = args.actions_per_chunk or model.config.chunk_size
    if not 1 <= actions_per_chunk <= model.config.chunk_size:
        raise SystemExit(f"--actions-per-chunk must be in [1, {model.config.chunk_size}]")

    sync_blind_s = model.config.n_action_steps / args.control_hz
    print(f"[INFO] chunk_size {model.config.chunk_size}, keeping {actions_per_chunk} per re-plan.\n"
          f"[INFO] re-plan trigger: queue below {args.chunk_size_threshold:.2f} of a chunk "
          f"(~{actions_per_chunk * (1 - args.chunk_size_threshold) / args.control_hz:.2f}s of queued "
          f"motion consumed), or empty.\n"
          f"[INFO] for reference, a SYNC loop at this rate would look at the cameras once every "
          f"{sync_blind_s:.1f}s ({model.config.n_action_steps} steps).")

    _usb_reset_for_video_node(args.body_cam_index)
    _usb_reset_for_video_node(args.wrist_cam_index)
    if args.right_wrist_cam_index is not None:
        _usb_reset_for_video_node(args.right_wrist_cam_index)
    if args.front_cam_index is not None:
        _usb_reset_for_video_node(args.front_cam_index)
    time.sleep(1.0)

    camera_config = {
        "body_cam": OpenCVCameraConfig(index_or_path=args.body_cam_index, width=640, height=480, fps=FPS),
        "wrist_cam": OpenCVCameraConfig(index_or_path=args.wrist_cam_index, width=640, height=480, fps=FPS),
    }
    if args.right_wrist_cam_index is not None:
        camera_config["right_wrist_cam"] = OpenCVCameraConfig(
            index_or_path=args.right_wrist_cam_index, width=640, height=480, fps=FPS)
    if args.front_cam_index is not None:
        camera_config["front_cam"] = OpenCVCameraConfig(
            index_or_path=args.front_cam_index, width=640, height=480, fps=FPS)
    model_cam_keys = list(camera_config)

    _check_checkpoint_matches(model, preprocess, model_cam_keys)

    robot = OpenArmFollower(OpenArmFollowerConfig(
        right_port="can0",
        left_port="can1",
        enable_fd=True,
        model_path=URDF_PATH,
        cameras=camera_config,  # type: ignore
        recv_rounds=args.can_recv_rounds,
        recv_first_timeout_us=args.can_first_timeout_us,
        recv_mop_timeout_us=args.can_mop_timeout_us,
    ))
    robot.connect()
    check_arms_not_crossed(robot, calib)
    grip = _gripper_calib(calib)

    side_cam = None
    if args.side_cam_index is not None:
        _usb_reset_for_video_node(args.side_cam_index)
        time.sleep(1.0)
        side_cam = OpenCVCamera(OpenCVCameraConfig(
            index_or_path=args.side_cam_index, width=640, height=480, fps=FPS))
        try:
            side_cam.connect()
        except Exception as e:
            print(f"[WARN] side camera at /dev/video{args.side_cam_index} unavailable: {e}")
            side_cam = None

    live_view = not args.no_live_view
    if live_view and not _cv2_has_gui():
        print("[WARN] opencv-python-headless (GUI: NONE) -- continuing without the live window.")
        live_view = False

    full_obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    image_obs_features = {k: v for k, v in full_obs_features.items() if v["dtype"] in ("image", "video")}
    dataset_features = {
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ACTION_NAMES},
        "observation.state": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ACTION_NAMES},
        **image_obs_features,
    }

    action_queue = ActionQueue()
    obs_bus = ObservationBus()
    async_policy = AsyncPolicy(
        model=model,
        preprocess=preprocess,
        postprocess=postprocess,
        dataset_features=dataset_features,
        device=device,
        actions_per_chunk=actions_per_chunk,
        aggregate_fn=AGGREGATE_FUNCTIONS[args.aggregate_fn],
        chunk_size_threshold=args.chunk_size_threshold,
        obs_bus=obs_bus,
        action_queue=action_queue,
        verbose=not args.quiet,
    )
    async_policy.start()

    control_dt = 1.0 / args.control_hz
    max_step_rad = args.max_joint_speed * control_dt
    loop_periods: deque[float] = deque(maxlen=200)
    busy_periods: deque[float] = deque(maxlen=200)

    plot_n = max(0, args.plot_max_samples)
    plotting = not args.no_plot and plot_n > 0
    plot_time: deque[float] = deque(maxlen=plot_n)
    target_log = {k: deque(maxlen=plot_n) for k in ACTION_NAMES}
    actual_log = {k: deque(maxlen=plot_n) for k in ACTION_NAMES}
    raw_log = {k: deque(maxlen=plot_n) for k in ACTION_NAMES}
    STALE_TRAVEL_RAD = 0.02
    stale_log = {k: deque(maxlen=plot_n) for k in ACTION_NAMES}
    prev_actual: dict[str, float] = {}
    prev_cmd: dict[str, float] = {}
    cmd_travel: dict[str, float] = {}
    episode_marks: list[tuple[float, int]] = []
    run_start = time.perf_counter()

    try:
        for ep in range(args.max_episodes):
            print(f"\n===== Episode {ep} =====")

            if not args.no_start_pose:
                if approach_pose(
                    robot, sim_init_pose_action(calib),
                    label="Isaac Sim's reset pose",
                    arm_speed=args.start_pose_speed,
                    max_delta=args.max_start_pose_delta,
                    assume_yes=args.yes or ep > 0,
                ) is None:
                    print("Start-pose approach refused -- not running the policy.")
                    break

            action_queue.flush()
            obs_bus.clear()

            if ep > 0 and args.episode_gap_seconds > 0:
                print(f"[INFO] {args.episode_gap_seconds:.0f}s to reset the scene -- take the can out of "
                      f"the left gripper and place a new one.")
                time.sleep(args.episode_gap_seconds)

            gripper_cmd = {i: GRIPPER_OPEN_CMD for i in GRIPPER_IDX}
            held_target: np.ndarray | None = None

            episode_start = time.perf_counter()
            if plotting:
                episode_marks.append((episode_start - run_start, ep))
            tick = 0
            starved_ticks = 0
            clamped_ticks = 0
            last_report = episode_start
            last_tick_start: float | None = None
            replans_at_start = async_policy.replans

            while time.perf_counter() - episode_start < args.max_episode_seconds:
                tick_start = time.perf_counter()
                if last_tick_start is not None:
                    loop_periods.append(tick_start - last_tick_start)
                last_tick_start = tick_start
                if async_policy.shutdown.is_set():
                    print("[ERROR] inference thread is down -- ending the episode.")
                    break

                obs = _get_obs_sim_gripper(robot, grip)
                current_q = np.array([obs[k] for k in ACTION_NAMES], dtype=np.float64)
                if held_target is None:
                    held_target = current_q.copy()

                if action_queue.fill_ratio() <= args.chunk_size_threshold:
                    obs_bus.publish(obs, timestep=action_queue.latest_executed + 1)
                else:
                    obs_bus.clear()

                timed = action_queue.pop()
                if timed is None:
                    starved_ticks += 1
                    target_q = held_target.copy()
                    raw_q = None
                else:
                    target_q = timed.action.copy()
                    raw_q = target_q.copy()

                    for i in GRIPPER_IDX:
                        gripper_cmd[i] = _binarize_gripper(target_q[i], gripper_cmd[i])
                        target_q[i] = gripper_cmd[i]

                delta = target_q - current_q
                if np.any(np.abs(delta) > max_step_rad + 1e-12):
                    clamped_ticks += 1
                target_q = current_q + np.clip(delta, -max_step_rad, max_step_rad)
                for i in GRIPPER_IDX:
                    target_q[i] = float(np.clip(target_q[i], GRIPPER_MIN, GRIPPER_MAX))
                held_target = target_q

                if plotting:
                    plot_time.append(tick_start - run_start)
                    for j, name in enumerate(ACTION_NAMES):
                        tgt, act = float(target_q[j]), float(current_q[j])
                        target_log[name].append(tgt)
                        actual_log[name].append(act)
                        raw_log[name].append(float(raw_q[j]) if raw_q is not None else float("nan"))
                        if name in prev_actual and act == prev_actual[name]:
                            cmd_travel[name] = (cmd_travel.get(name, 0.0)
                                                + abs(tgt - prev_cmd.get(name, tgt)))
                        else:
                            cmd_travel[name] = 0.0
                        stale_log[name].append(cmd_travel[name] > STALE_TRAVEL_RAD)
                        prev_actual[name], prev_cmd[name] = act, tgt

                target_action = dict(obs)
                for name, value in zip(ACTION_NAMES, target_q):
                    target_action[name] = float(value)
                _send_action_sim_gripper(robot, target_action, grip)

                if live_view:
                    side_frame = None
                    if side_cam is not None:
                        try:
                            side_frame = side_cam.async_read()
                        except Exception as e:
                            print(f"[WARN] side camera read failed: {e}")
                    frame = _build_combined_frame(obs, side_frame, model_cam_keys)
                    if frame is not None:
                        try:
                            cv2.imshow(LIVE_VIEW_WINDOW, frame)
                            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                                raise KeyboardInterrupt
                        except cv2.error as e:
                            print(f"[WARN] live view failed ({e}) -- disabled for the rest of the run.")
                            live_view = False

                tick += 1
                busy_periods.append(time.perf_counter() - tick_start)

                if not args.quiet and time.perf_counter() - last_report >= 1.0:
                    last_report = time.perf_counter()
                    busy = sorted(busy_periods)[len(busy_periods) // 2]
                    loop = (sorted(loop_periods)[len(loop_periods) // 2]
                            if loop_periods else control_dt)
                    infer = async_policy.inference_times
                    infer_ms = (sorted(infer)[len(infer) // 2] * 1000) if infer else float("nan")
                    ep_replans = async_policy.replans - replans_at_start
                    replan_ticks = tick / max(ep_replans, 1)
                    print(
                        f"[t {time.perf_counter() - episode_start:5.1f}s] tick {tick} | "
                        f"control {1.0 / loop:5.1f} Hz (busy {busy * 1000:4.1f}ms of "
                        f"{control_dt * 1000:.0f}ms) | queue {action_queue.qsize():3d} "
                        f"({action_queue.fill_ratio():.2f} chunk) | replan every "
                        f"{replan_ticks * loop * 1000:.0f}ms ({infer_ms:.0f}ms infer) | "
                        f"starved {starved_ticks} | clamped {clamped_ticks}",
                        flush=True,
                    )

                sleep = control_dt - (time.perf_counter() - tick_start)
                if sleep > 0:
                    time.sleep(sleep)

            held = sorted(loop_periods)[len(loop_periods) // 2] if loop_periods else control_dt
            busy = sorted(busy_periods)[len(busy_periods) // 2] if busy_periods else 0.0
            print(
                f"Episode {ep}: {time.perf_counter() - episode_start:.1f}s, {tick} ticks at "
                f"{1.0 / held:.1f} Hz (requested {args.control_hz:.1f}), "
                f"{async_policy.replans - replans_at_start} re-plans, {starved_ticks} starved ticks "
                f"({100.0 * starved_ticks / max(tick, 1):.1f}%), {clamped_ticks} clamped "
                f"({100.0 * clamped_ticks / max(tick, 1):.1f}%)."
            )
            if busy > 0 and args.control_hz * busy < 0.5:
                print(f"[INFO] a tick's work costs {busy * 1000:.1f}ms of the "
                      f"{control_dt * 1000:.0f}ms period ({100 * (1 - args.control_hz * busy):.0f}% "
                      f"idle). Spend that headroom on --chunk-size-threshold (re-plan sooner), not "
                      f"on --control-hz (play faster).")
            if starved_ticks > 0.05 * max(tick, 1):
                print(f"[WARN] {100.0 * starved_ticks / max(tick, 1):.0f}% of ticks had an empty queue. "
                      f"Inference is not keeping ahead of the control loop.")
            if clamped_ticks > 0.25 * max(tick, 1):
                print(f"[WARN] --max-joint-speed {args.max_joint_speed:.2f} bound "
                      f"{100.0 * clamped_ticks / max(tick, 1):.0f}% of ticks.")

    except KeyboardInterrupt:
        print("\n[INFO] interrupted -- stopping.")
    finally:
        async_policy.stop()
        if side_cam is not None:
            side_cam.disconnect()
        if live_view:
            cv2.destroyAllWindows()
        robot.disconnect()
        print("[INFO] robot disconnected (motors are NOT powered off -- use emergency_disable.py).")

        if plotting:
            _show_tracking_plots(plot_time, target_log, actual_log, raw_log,
                                 episode_marks, args.save_plot, stale_log)


def _show_tracking_plots(plot_time, target_log, actual_log, raw_log, episode_marks, save_path=None,
                         stale_log=None):
    """End-of-run target-vs-actual and raw-policy-output plots -- identical to
    deploy_smolvla_async.py's own; see there for the full rationale of each panel."""
    t = list(plot_time)
    if not t:
        print("[WARN] No tracking data to plot.")
        return

    print(f"[INFO] Plotting target vs. actual joint tracking ({len(t)} ticks, "
          f"{t[0]:.0f}-{t[-1]:.0f}s)...")
    if len(t) == plot_time.maxlen:
        print(f"[INFO] history capped at --plot-max-samples ({plot_time.maxlen}); "
              f"showing the most recent {t[-1] - t[0]:.0f}s of the run.")

    marks = [(mt, ep) for mt, ep in episode_marks if t[0] <= mt <= t[-1]]

    def _grid(logs, title, ylabel="rad", legend=None, mark_stale=False):
        fig, axes = plt.subplots(8, 2, figsize=(14, 24), sharex=True)
        for ax, joint in zip(axes.flat, ACTION_NAMES):
            for label, log, style in logs:
                ax.plot(t, list(log[joint]), label=label, linewidth=1, **style)
            if mark_stale and stale_log is not None:
                flags = list(stale_log[joint])
                acts = list(actual_log[joint])
                st = [(ti, a) for ti, a, f in zip(t, acts, flags) if f]
                if st:
                    ax.plot([x for x, _ in st], [y for _, y in st], linestyle="none",
                            marker=".", markersize=3, color="tab:red", label="stale read")
            for mt, ep in marks:
                ax.axvline(mt, color="grey", linewidth=0.6, linestyle=":")
            ax.set_title(joint)
            ax.set_ylabel(ylabel)
            ax.grid(True)
        if legend:
            axes.flat[0].legend(loc="upper right")
        for ax in axes[-1, :]:
            ax.set_xlabel("Time (s)")
        fig.suptitle(title + (f"  (dotted: start of episodes {', '.join(str(ep) for _, ep in marks)})"
                              if marks else ""))
        fig.tight_layout()
        return fig

    try:
        fig = _grid(
            [("target", target_log, {}),
             ("actual", actual_log, {"marker": "o", "markersize": 2})],
            "Both arms + grippers: commanded target vs. measured actual (async, GR00T)",
            legend=True, mark_stale=True,
        )
        fig2 = _grid(
            [("raw queued action", raw_log, {})],
            "Queued policy action before binarize + speed clamp"
            " -- gaps are ticks the queue was starved",
        )

        if stale_log is not None:
            rows = [(k, sum(stale_log[k]) / len(stale_log[k])) for k in ACTION_NAMES
                    if len(stale_log[k])]
            bad = [(k, f) for k, f in rows if f > 0.05]
            if bad:
                print("[WARN] feedback stopped refreshing on: "
                      + ", ".join(f"{k} {f:.0%}" for k, f in sorted(bad, key=lambda r: -r[1])))
                print("       Those channels' 'actual' traces are retained values, and because the"
                      " speed clamp anchors on the measured position, their COMMANDS are affected"
                      " too -- not just the plot. Raise --can-recv-rounds / lower"
                      " --can-first-timeout-us and re-run before reading anything into them.")
            else:
                print("[INFO] no channel showed stale feedback (>5% of ticks).")

        if save_path:
            base, ext = os.path.splitext(save_path)
            ext = ext or ".png"
            fig.savefig(base + ext, dpi=120)
            fig2.savefig(f"{base}-raw{ext}", dpi=120)
            print(f"[INFO] Saved tracking plots to {base}{ext} and {base}-raw{ext}")

        plt.show()
    except Exception as e:
        print(f"[WARN] Could not display the tracking plots ({type(e).__name__}: {e}). "
              f"Re-run with --save-plot PATH to write them to disk instead.")


if __name__ == "__main__":
    main()
