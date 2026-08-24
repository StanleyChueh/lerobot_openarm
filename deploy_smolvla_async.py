"""
Asynchronous SmolVLA rollout on the real OpenArm follower -- the same 16D joint-space checkpoint
deploy_smolvla_pickup_jointspace.py runs, but with inference moved OFF the control loop.

WHY THIS EXISTS. The synchronous script's loop is: read observation -> forward pass -> send action.
Inference is therefore *inside* the control period, so the only way to afford a forward pass is to
amortise it over a long action chunk: n_action_steps=50 at the dataset's own 20 Hz means the
policy looks at the cameras once every 2.5 seconds and the arm is blind in between. Lowering
--n-action-steps buys reactivity but pays for it in control rate, because the forward pass is
still serialised with the motion. There is no setting of that script where the robot both holds
20 Hz and re-plans often.

Async breaks the coupling: a control thread pops one action per tick from a queue and never waits
for the GPU, while an inference thread refills that queue from the LATEST observation whenever the
queue drains past a threshold. The control rate is then bounded by the CAN read, and the re-plan
rate by the forward pass, independently.

RATE. The training data is 20 fps -- meta/info.json says fps=20 and the episode timestamps step
by exactly 50.0ms -- so --control-hz 20 plays the policy's chunk back at the speed its demos were
recorded at, and is the default here. Raising it does NOT make the robot more reactive (that is
--chunk-size-threshold's job); it makes the demo motions play fast-forward.

Measured on this machine, V14_background at 512x512 x3 cameras: 316ms for the first forward pass,
then 84-93ms warm. So the floor on the re-plan interval is ~92ms; where it actually lands is set
by how much queued motion --chunk-size-threshold lets drain first, which is what the two knobs
below trade against each other. Re-plan interval is (1 - threshold) * actions_per_chunk ticks,
floored at one forward pass. From the threading harness at --control-hz 20 with inference stubbed
at 92ms (starved% is the fraction of ticks that found an empty queue and held position -- the 1.7%
here is entirely the cold-start pass):

    --actions-per-chunk  --chunk-size-threshold   re-plan every   starved
             50                  0.50                1200ms         1.7%
             50                  0.80                 500ms         1.7%
             50                  0.90                 250ms         1.7%   <- default
             25                  0.80                 250ms         1.7%
             10                  0.50                 250ms         1.7%
             10                  0.80                  92ms         1.7%   <- GPU-bound
              5                  0.50                 150ms         1.7%

The sync script at this rate looks at the cameras once every 2.5s (50 steps at 20 Hz), so the
default here is 10x fresher. Prefer raising the threshold over shrinking the chunk: every queued
action is re-blended at 0.7 weight on each re-plan, so a long chunk refreshed often stays smoother
than a short chunk replaced outright.

RELATION TO lerobot/async_inference. This is a port of the algorithm in
lerobot_experiment/lerobot/src/lerobot/async_inference/{robot_client,policy_server,configs}.py,
with the gRPC transport removed. Ported faithfully:

  - TimedAction / integer timestep bookkeeping        (helpers.py)
  - the chunk_size_threshold re-plan trigger          (RobotClient._ready_to_send_observation)
  - must-go: an empty queue always forces a re-plan   (RobotClient.control_loop_observation)
  - overlap aggregation between the queued chunk and
    the incoming one, keyed by timestep               (RobotClient._aggregate_action_queues)
  - AGGREGATE_FUNCTIONS by name                       (configs.py)
  - latest-observation-only, maxsize=1 semantics      (PolicyServer.observation_queue)

Deliberately NOT ported -- the two-process gRPC split. Three things make it cost more than it buys
on this setup, all checked rather than assumed:

  - OpenArmFollower is not in lerobot's robot registry, so RobotClient's make_robot_from_config()
    cannot build it; it would have to be re-exposed as a lerobot plugin first.
  - OpenArmFollower.send_action(action, target_vel) takes a second required argument (abd1499),
    while RobotClient.control_loop_action calls send_action(dict) -- a TypeError on the first tick.
  - The gripper calibration boundary (raw <-> sim, see the GRIPPER CALIBRATION note in
    deploy_smolvla_pickup_jointspace.py) has to wrap get_observation/send_action on the ROBOT side.
    RobotClient has no hook there, so the policy server would be normalising raw motor units.

  Also: grpcio is not installed in this env (lerobot.transport raises on import), and the local
  policy_server.py's _get_action_chunk() is hard-coded to pi05's RTC signature --
  predict_action_chunk(obs, inference_delay=..., prev_chunk_left_over=..., execution_horizon=...)
  -- which SmolVLAPolicy.predict_action_chunk(batch, noise=None, **kwargs) would reject.

  None of that is fatal, and the split is worth doing if the policy ever moves to a second machine.
  The thread boundary here is the same boundary, so that change is localised to _inference_worker.

WHAT THIS DOES NOT FIX. Async makes the policy react to what it currently sees. It does not give
it a behaviour it never learned: the training set contains no failed grasps and no empty-table
frames (Mimic diverts failed trials to a separate *_failed.hdf5), so a missed grasp is still
followed by the rest of the hand-over script. Async raises the chance of grasping correctly in the
first place -- it does not add retry. Retry has to come from outside the policy (detect, reset,
re-run) or from adding recovery demos and retraining.

POLICY STATE. Unlike the sync script this never calls model.select_action(), so SmolVLA's internal
action queue is never used and model.reset() is never needed: predict_action_chunk() is stateless,
and this file owns the only queue. Episode-to-episode carryover is handled by flushing OUR queue.

Usage (same checkpoint/cameras/calibration as the sync script):
  python deploy_smolvla_async.py \\
      --checkpoint ethanCSL/openarm_visuomotor_VR_pringles_V14_background \\
      --body-cam-index 4 --wrist-cam-index 16 --right-wrist-cam-index 10 \\
      --calibration calibration.json \\
      --control-hz 20 --max-joint-speed 2.0 --max-episode-seconds 30

Tuning, in the order worth touching:
  --chunk-size-threshold  when to re-plan, as a fraction of a chunk still queued (default 0.9).
                          Higher = re-plans sooner = fresher actions, more GPU. 0.0 re-plans only
                          when the queue empties (closest to the sync script's behaviour). This,
                          not --control-hz, is the reactivity knob.
  --actions-per-chunk     how much of each 50-step chunk to keep (default 50). Lowering it bounds
                          how stale a queued action can be if inference stalls.
  --aggregate-fn          how an incoming chunk blends with the overlapping queued one.
                          weighted_average (default) = 0.3*old + 0.7*new, latest_only = hard swap.
  --control-hz            leave at 20: it is the dataset's own fps, so the chunk plays at demo
                          speed. The loop has ~8ms of work per tick, so the headroom is real but
                          spending it on a higher rate just runs the demos fast-forward.

Everything below the queue -- gripper calibration, Schmitt-trigger binarization, the
--max-joint-speed clamp against measured joints, Isaac Sim's reset pose per episode -- is imported
unchanged from deploy_smolvla_pickup_jointspace.py rather than re-derived, so the two stay in sync.
"""

import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action

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
from sim_bridge_common import approach_pose, load_calibration, sim_init_pose_action

# lerobot/async_inference/configs.py's AGGREGATE_FUNCTIONS, verbatim. These blend an incoming
# chunk's action with the queued action for the SAME timestep -- the overlap that exists because
# inference started from an observation several control ticks ago and the chunk it produced covers
# ticks the queue already has an opinion about. A hard swap (latest_only) is not obviously right:
# the queued action came from a chunk that was internally smooth, and replacing individual actions
# inside it mid-execution is what produces a visible discontinuity in the arm.
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


@dataclass
class TimedAction:
    """One action, tagged with the control-tick index it is meant to be executed on.

    The timestep -- not arrival order -- is what makes async coherent: a chunk predicted from the
    observation at tick 100 covers ticks 100..149 no matter when it lands, so actions for ticks
    already executed can be dropped and actions for ticks still queued can be aggregated against
    what is already there. lerobot's TimedAction also carries a wall-clock timestamp for measuring
    network latency; with no network here that field would only ever be read by a log line."""

    timestep: int
    action: np.ndarray  # (ACTION_DIM,) absolute joint targets, sim convention (0.0..0.044 grippers)


class ActionQueue:
    """Timestep-keyed action queue shared by the control and inference threads.

    A dict rather than lerobot's Queue-of-TimedAction: _aggregate_action_queues rebuilds a fresh
    Queue on every ingest precisely because a Queue cannot be indexed by timestep, and that rebuild
    is the only reason it holds the lock across the whole merge. Keyed by timestep the merge is a
    dict update, and popping is min(keys) -- same semantics, no rebuild."""

    def __init__(self):
        self._lock = threading.Lock()
        self._actions: dict[int, np.ndarray] = {}
        self._latest_executed = -1
        self.chunk_size = 1  # largest chunk seen, for the fill-ratio trigger (RobotClient's own)

    def ingest(self, chunk: np.ndarray, first_timestep: int, aggregate_fn) -> tuple[int, int, int]:
        """Merge a freshly predicted chunk in. Returns (dropped, merged, added) for logging.

        `dropped` is the count of chunk actions whose tick has already been executed -- i.e. how
        many control ticks elapsed while this forward pass ran. That number IS the inference
        latency in ticks, measured rather than assumed, and it is worth watching: if it approaches
        --actions-per-chunk the queue is being refilled with actions that are all already stale."""
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
        """Drop every queued action and restart tick numbering -- called between episodes.

        Without this, episode N+1 opens by executing actions planned from episode N's final
        observation, of a scene that no longer exists and from a pose the arm has since been ramped
        away from. Same reason the sync script calls model.reset() per episode."""
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
        """Queued actions as a fraction of one chunk -- RobotClient._ready_to_send_observation's
        quantity. Re-planning when this drops below --chunk-size-threshold means the next chunk
        arrives while there is still queued motion to cover the forward pass, which is the whole
        reason the control thread never stalls."""
        with self._lock:
            return len(self._actions) / max(self.chunk_size, 1)


class ObservationBus:
    """Single-slot handoff of the latest observation to the inference thread.

    maxsize=1 with overwrite, matching PolicyServer.observation_queue: an observation that has been
    superseded before inference picked it up is worthless, and queueing it would only guarantee the
    policy plans from a stale frame. Publishing overwrites; there is nothing to fall behind on."""

    def __init__(self):
        self._lock = threading.Lock()
        self._slot: tuple[dict, int] | None = None
        self._ready = threading.Event()

    def publish(self, obs: dict, timestep: int) -> None:
        # Copy the camera arrays: lerobot's OpenCVCamera hands back a frame the reader thread may
        # reuse, and the inference thread holds this for the length of a forward pass. Copying only
        # on publish (once per re-plan, not once per control tick) keeps that off the hot path.
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

    This is policy_server.py's GetActions loop with the gRPC and pickle layers deleted. Everything
    it touches (model, preprocessor, postprocessor, CUDA) stays on this thread; everything the
    control thread touches (robot, CAN, cameras) stays on that one. The only shared state is the
    two synchronised objects above, which is what makes the split safe without the process
    boundary lerobot uses to enforce it."""

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
        self.discarded = 0  # observations withdrawn because a chunk landed while they waited
        self.thread = threading.Thread(target=self._worker, name="inference", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.shutdown.set()
        self.thread.join(timeout=5.0)

    @torch.inference_mode()
    def _predict_chunk(self, obs: dict) -> np.ndarray:
        """One forward pass -> (K, ACTION_DIM) absolute joint targets in the sim convention.

        predict_action_chunk() rather than select_action(): select_action pops one action from
        SmolVLA's own internal queue and refills it only when empty, which is exactly the
        chunk-scheduling policy this file replaces. Asking for the raw chunk is what lets the
        queue above own the scheduling instead."""
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

        # Unnormalise one action at a time: the postprocessor pipeline is built for (B, action_dim)
        # per call, not (B, chunk_size, action_dim). Same loop policy_server._predict_action_chunk
        # runs, and the reason it is a loop rather than a reshape.
        actions = [
            make_robot_action(self.postprocess(chunk[:, i, :]), self.dataset_features)
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

            # Re-check the trigger on THIS thread before spending a forward pass. The control
            # thread publishes once per tick for as long as the queue sits below the threshold,
            # so a ~92ms forward pass at 20 Hz leaves ~2 observations published behind it, the last
            # of which is still waiting when the chunk lands. Gating only on the control thread
            # means that one gets consumed before the next tick can withdraw it, and it produces a
            # chunk that is ~94% overlap with what was just queued -- a full forward pass whose
            # only effect is to re-weight actions already decided. Measured on the threading
            # harness: it doubled the re-plan count for no additional coverage.
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="HF repo id or local path of the joint-space SmolVLA checkpoint.")
    parser.add_argument("--body-cam-index", type=int, required=True,
                        help="/dev/video index for observation.images.body_cam.")
    parser.add_argument("--wrist-cam-index", type=int, required=True,
                        help="/dev/video index for observation.images.wrist_cam (the LEFT wrist).")
    parser.add_argument("--right-wrist-cam-index", type=int, default=None,
                        help="/dev/video index for observation.images.right_wrist_cam (dual-arm checkpoints).")
    parser.add_argument("--front-cam-index", type=int, default=None,
                        help="/dev/video index for observation.images.front_cam (the '..._three_cams' variant).")
    parser.add_argument("--side-cam-index", type=int, default=None,
                        help="EXTRA camera for the live view only -- never fed to the model.")
    parser.add_argument("--calibration", type=str, required=True,
                        help="calibration.json -- required for the gripper raw<->sim mapping (see the "
                             "GRIPPER CALIBRATION note in deploy_smolvla_pickup_jointspace.py).")

    parser.add_argument("--control-hz", type=float, default=20.0,
                        help="Control-loop rate. One queued action is sent per tick, so this is also the "
                             "rate the policy's chunk plays back at -- 20 is the training data's own fps "
                             "(meta/info.json fps=20, timestamps stepping by 50.0ms), which is what makes "
                             "demo motions run at demo speed. Raising it does NOT add reactivity, it "
                             "fast-forwards the demos; use --chunk-size-threshold for reactivity. Unlike "
                             "the sync script's --inference-hz this does NOT include a forward pass, so "
                             "the only thing that has to fit in the period is one 16-joint CAN read "
                             "(~8ms measured, i.e. 84%% of a 20 Hz period is idle sleep).")
    parser.add_argument("--chunk-size-threshold", type=float, default=0.9,
                        help="Re-plan once the queue holds less than this fraction of a chunk. THE "
                             "reactivity knob: the interval is (1 - this) * --actions-per-chunk ticks, "
                             "floored at one forward pass (~92ms). Default 0.9 gives 5 ticks = 250ms at "
                             "20 Hz, against 2.5s for the sync script; lerobot's own default is 0.5, which "
                             "here would be 1.2s. 0.0 waits for the queue to drain entirely.")
    parser.add_argument("--actions-per-chunk", type=int, default=None,
                        help="How many actions to keep from each predicted chunk (default: the checkpoint's "
                             "full chunk_size, 50). Unlike the sync script's --n-action-steps this does not "
                             "set the re-plan interval -- --chunk-size-threshold does -- it bounds how far "
                             "ahead the queue can run, i.e. how stale an action can get if inference stalls.")
    parser.add_argument("--aggregate-fn", choices=tuple(AGGREGATE_FUNCTIONS), default="weighted_average",
                        help="How an incoming chunk blends with the queued actions it overlaps.")
    parser.add_argument("--max-joint-speed", type=float, default=1.0,
                        help="rad/s ceiling for all 16 joints, enforced by clamping how far the queued "
                             "target may move from the CURRENT measured joints each control tick (one tick "
                             "authorises max-joint-speed/--control-hz rad). Sized against the demos "
                             "themselves: at the dataset's 20 fps their 99th-percentile joint speed peaks "
                             "at 1.92 rad/s (RJ4) and their 95th at 1.06, so 2.0 passes every demo motion "
                             "through unclipped and 1.5 clips the fastest ~1%% of RJ4/LJ5 transit. Lower it "
                             "for a cautious first run, not as a smoothness control.")
    parser.add_argument("--max-episode-seconds", type=float, default=30.0,
                        help="Wall-clock limit per episode. The demos are 358 frames = 17.9s at the "
                             "dataset's 20 fps, so 30 leaves room for a slow start plus settling. Anything "
                             "under ~20 cuts the hand-over off before a successful demo would have "
                             "finished it.")
    parser.add_argument("--max-episodes", type=int, default=1,
                        help="Episodes to run. Each one ramps back to Isaac Sim's reset pose (which reopens "
                             "both grippers) and flushes the action queue, so this is also the re-arm "
                             "mechanism: put a fresh can down during the reset and the next episode runs.")
    parser.add_argument("--episode-gap-seconds", type=float, default=5.0,
                        help="Pause after each episode's reset pose, before the policy is given control -- "
                             "time to take the can out of the left gripper and place a new one.")

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
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.chunk_size_threshold <= 1.0:
        raise SystemExit(f"--chunk-size-threshold must be in [0, 1], got {args.chunk_size_threshold}")

    calib = load_calibration(args.calibration)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] loading {args.checkpoint} on {device}...")
    model = SmolVLAPolicy.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()
    preprocess, postprocess = make_pre_post_processors(model.config, args.checkpoint)

    actions_per_chunk = args.actions_per_chunk or model.config.chunk_size
    if not 1 <= actions_per_chunk <= model.config.chunk_size:
        raise SystemExit(f"--actions-per-chunk must be in [1, {model.config.chunk_size}]")

    # What the async setup actually buys, stated in the units that matter, before anything moves.
    # The sync equivalent is chunk_size / control_hz seconds of blindness; here it is one forward
    # pass, and the line below is the claim to check against the [replan] lines once it is running.
    sync_blind_s = model.config.n_action_steps / args.control_hz
    print(f"[INFO] chunk_size {model.config.chunk_size}, keeping {actions_per_chunk} per re-plan.\n"
          f"[INFO] re-plan trigger: queue below {args.chunk_size_threshold:.2f} of a chunk "
          f"(~{actions_per_chunk * (1 - args.chunk_size_threshold) / args.control_hz:.2f}s of queued "
          f"motion consumed), or empty.\n"
          f"[INFO] for reference, the SYNC script at this rate would look at the cameras once every "
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
    # Two separate measurements, because they answer different questions and conflating them is
    # actively misleading: busy_periods is how long a tick's WORK takes (CAN read + clamp + send),
    # loop_periods is the wall-clock gap between consecutive ticks, i.e. the rate the arm actually
    # sees. Reporting the first as "control Hz" reads as a loop running 6x faster than it is --
    # the headroom, not the rate. The gap between them is the sleep, and it is the useful number:
    # busy well under control_dt means --control-hz can be raised.
    loop_periods: deque[float] = deque(maxlen=200)
    busy_periods: deque[float] = deque(maxlen=200)

    try:
        for ep in range(args.max_episodes):
            print(f"\n===== Episode {ep} =====")

            # Ramp to Isaac Sim's reset pose first. Both grippers reopen as part of it, which is
            # what re-arms the policy: the terminal state of a successful demo is the start pose
            # with the LEFT gripper closed on the can, so without reopening it the policy's own
            # observation still says "done" and it holds still. Measured over 300 demos: start and
            # end joint medians differ by <0.07 rad on every joint except LJ8 (0.044 -> 0.021).
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

            # Everything planned before this point described a scene that no longer exists.
            action_queue.flush()
            obs_bus.clear()

            if ep > 0 and args.episode_gap_seconds > 0:
                print(f"[INFO] {args.episode_gap_seconds:.0f}s to reset the scene -- take the can out of "
                      f"the left gripper and place a new one.")
                time.sleep(args.episode_gap_seconds)

            # Latched gripper decisions, seeded from where the grippers physically are so the
            # Schmitt trigger's dead band holds the real state rather than an assumed one.
            gripper_cmd = {i: GRIPPER_OPEN_CMD for i in GRIPPER_IDX}
            held_target: np.ndarray | None = None  # what to command while the queue is empty

            episode_start = time.perf_counter()
            tick = 0
            starved_ticks = 0
            clamped_ticks = 0
            last_report = episode_start
            last_tick_start: float | None = None
            replans_at_start = async_policy.replans   # the counter is cumulative across episodes

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

                # RobotClient._ready_to_send_observation, plus its must_go: an empty queue always
                # re-plans regardless of the threshold, because there is nothing left to execute
                # and the alternative is holding position until the next trigger.
                if action_queue.fill_ratio() <= args.chunk_size_threshold:
                    obs_bus.publish(obs, timestep=action_queue.latest_executed + 1)
                else:
                    # The trigger is satisfied again -- a chunk landed. Withdraw any observation
                    # published while it was in flight but not yet picked up, so the next re-plan
                    # starts from a fresh frame instead of one predating the chunk now queued.
                    obs_bus.clear()

                timed = action_queue.pop()
                if timed is None:
                    # Queue starved: hold the last commanded target rather than re-commanding the
                    # measured position. Commanding the measurement makes the arm creep -- each
                    # tick's droop becomes the next tick's setpoint, and under gravity that walks
                    # the joint down instead of holding it.
                    starved_ticks += 1
                    target_q = held_target.copy()
                else:
                    target_q = timed.action.copy()

                    # Binarize both grippers, in the sim convention, before the speed clamp: the
                    # clamp then limits how fast the gripper travels to its endpoint rather than
                    # smearing the decision back into the intermediate widths binarization exists
                    # to remove.
                    for i in GRIPPER_IDX:
                        gripper_cmd[i] = _binarize_gripper(target_q[i], gripper_cmd[i])
                        target_q[i] = gripper_cmd[i]

                # Speed clamp against the CURRENT measured joints, not against the previous target:
                # self-correcting for tracking error, and the only thing bounding how fast the arm
                # is asked to move. One control tick's worth of travel, since one action is sent
                # per tick here -- there is no separate inference period to convert against, which
                # is the ambiguity --speed-clamp-basis exists to resolve in the sync script.
                delta = target_q - current_q
                if np.any(np.abs(delta) > max_step_rad + 1e-12):
                    clamped_ticks += 1
                target_q = current_q + np.clip(delta, -max_step_rad, max_step_rad)
                for i in GRIPPER_IDX:
                    target_q[i] = float(np.clip(target_q[i], GRIPPER_MIN, GRIPPER_MAX))
                held_target = target_q

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
            # Headroom, reported but explicitly NOT a suggestion to raise --control-hz. The
            # control thread does no GPU work, so a tick costing ~8ms against a 50ms period is
            # normal and correct: 20 Hz is the dataset's own fps, and spending the headroom on a
            # higher rate would play the demos fast-forward rather than make the policy react
            # sooner. The idle time is what a re-plan lands in; --chunk-size-threshold spends it.
            if busy > 0 and args.control_hz * busy < 0.5:
                print(f"[INFO] a tick's work costs {busy * 1000:.1f}ms of the "
                      f"{control_dt * 1000:.0f}ms period ({100 * (1 - args.control_hz * busy):.0f}% "
                      f"idle). That headroom is correct at the dataset's 20 fps -- spend it on "
                      f"--chunk-size-threshold (re-plan sooner), not on --control-hz (play faster).")
            # A starved tick is a tick the arm spent holding position because inference could not
            # keep up -- the async equivalent of an overrun, and the number to act on: raise
            # --chunk-size-threshold so re-plans start earlier, or lower --control-hz.
            if starved_ticks > 0.05 * max(tick, 1):
                print(f"[WARN] {100.0 * starved_ticks / max(tick, 1):.0f}% of ticks had an empty queue. "
                      f"Inference is not keeping ahead of the control loop.")
            if clamped_ticks > 0.25 * max(tick, 1):
                print(f"[WARN] --max-joint-speed {args.max_joint_speed:.2f} bound "
                      f"{100.0 * clamped_ticks / max(tick, 1):.0f}% of ticks. The demos' own 99th "
                      f"percentile peaks at 1.92 rad/s (RJ4) at 20 fps, so anything below 2.0 is "
                      f"holding the arm under demo speed.")

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


if __name__ == "__main__":
    main()
