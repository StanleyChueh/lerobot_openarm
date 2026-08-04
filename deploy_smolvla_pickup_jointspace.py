"""
Autonomous SmolVLA rollout on the real OpenArm follower, for a checkpoint trained on JOINT-SPACE
actions/observations (both are LJ1.pos..LJ8.pos -- see scripts/tools/convert_hdf5_to_lerobot.py's
LJ_IDX/LJ_NAMES in the IsaacLab repo) -- e.g. ethanCSL/openarm_visuomotor_no_domain_randomization_
1000_joints. This is the simplified sibling of deploy_smolvla_pickup.py (the EE-delta-action
version): since the model now predicts joint targets directly, there is no IK bridge here at all
-- no ik_bridge.py import, no DifferentialIK solve, no redundant-DOF noise-amplification problem
(that was specifically a DLS/Jacobian artifact of the EE-delta version, structurally absent here).

SAFETY: this moves the real robot autonomously with NO human-in-the-loop correction step, and
runs an UNTESTED checkpoint on real hardware for the first time.
  - Start with MAX_EPISODES = 1 and watch closely before trusting longer unsupervised runs.
  - Start with a conservative --max-joint-speed (e.g. 0.3) and --inference-hz (e.g. 10) for the
    first run, same as any new checkpoint's first real-hardware test -- see deploy_smolvla_
    pickup.py's own history for why (jerky/wrong output on a first run is not unusual).
  - Keep emergency_disable.py within reach.
  - Ctrl+C stops the loop and disconnects the robot; it does not power off the motors.

Usage (two-camera checkpoint, body+wrist only):
  python deploy_smolvla_pickup_jointspace.py \\
      --checkpoint ethanCSL/openarm_visuomotor_no_domain_randomization_1000_joints \\
      --body-cam-index 4 --wrist-cam-index 10 --side-cam-index 12 \\
      --inference-hz 10 --max-joint-speed 0.3 --max-episode-seconds 10

Usage (three-camera checkpoint, body+wrist+front all fed to the model):
  python deploy_smolvla_pickup_jointspace.py \\
      --checkpoint ethanCSL/openarm_visuomotor_no_domain_randomization_1000_joints_three_cams \\
      --body-cam-index 4 --wrist-cam-index 10 --front-cam-index 12 \\
      --inference-hz 10 --max-joint-speed 0.3 --max-episode-seconds 10

--front-cam-index adds a real "front_cam" observation.images input (only pass it for a checkpoint
actually trained with one -- e.g. the "..._three_cams" variant -- since the preprocessor will
otherwise reject/ignore an unexpected image key). --side-cam-index is unrelated: it's an EXTRA
camera shown for humans only, never fed to the model, regardless of checkpoint.

LIVE CAMERA VIEW: whichever cameras are actually active -- the model's own (body/wrist, plus
front if --front-cam-index is given) and/or the view-only --side-cam-index one -- are combined
side-by-side into one cv2.imshow window, each labeled green ("model input") or yellow ("view
only") so it's never ambiguous which cameras the policy actually sees, updated live during
evaluation. Pass --no-live-view to turn this off (e.g. headless/no X server), and/or --save-video
PATH.mp4 to additionally save the same combined view to disk once the run ends.

Action space: action[t] was derived in training data as state[t+1] -- the NEXT measured joint
position, a standard proxy for "commanded joint target" when demos are recorded via position
tracking rather than direct joint-target logging (see convert_hdf5_to_lerobot.py's _load_episode
docstring in IsaacLab). So the model's raw 8D output is interpreted directly as ABSOLUTE target
joint angles for LJ1.pos..LJ8.pos, not deltas and not a semantic +-1 gripper flag like the older
EE-delta checkpoint used. The right arm is held at its current pose (never part of this action
space, same as the EE-delta version).

Speed limiting still applies the same way as the EE-delta version: the raw target is clamped to
move at most --max-joint-speed * (1/--inference-hz) radians from the CURRENT measured joint
angles each inference step, then the existing interpolation loop smooths that move in time across
control substeps. Optional EMA smoothing (--action-smoothing-alpha) is available but defaults to
off (1.0) -- unlike the EE-delta checkpoint, there's no known noise-amplification mechanism here
yet, so start trusting the raw model output and only add smoothing if the diagnostic plot below
shows it's actually needed.

Camera / USB-reset / dataset-loading notes are unchanged from deploy_smolvla_pickup.py -- see
that file's docstring for the full explanation (D435i intermittent color-stream hang and the
USBDEVFS_RESET workaround, why norm stats load from the checkpoint directly with no --dataset
arg, why camera dict keys are "body_cam"/"wrist_cam" not "camera1"/"camera2").

GRIPPER CALIBRATION -- do not skip --calibration: confirmed on real hardware (2026-07) that
OpenArmFollower's raw LJ8.pos range (the 0.0..0.044 rad "sim convention" that the arm joints and
every other script in this project treat as URDF-native and therefore already correct) does NOT
correspond to this gripper's real open/closed positions -- commanding both 0.044 and 0.0 directly
left the real gripper closed firmly either way. replay_hf_sim_episode.py and mirror_bridge.py
never hit this because they always route the gripper through calibration.json's gripper_sim_to_raw()
(open_raw/closed_raw, measured per-unit during Phase 0 setup) rather than sending LJ8.pos raw --
this script now does the same: every read gripper value is mapped raw->sim via raw_to_gripper_sim()
immediately after robot.get_observation(), and every value about to be sent is mapped sim->raw via
gripper_sim_to_raw() immediately before robot.send_action() -- everywhere in between (the model's
input/output, --max-joint-speed clamping, the tracking plots) stays in the same 0.0..0.044 sim
convention the checkpoint was trained on, matching observation.state/action's LJ8.pos semantics
exactly. Only the two boundary points touch raw motor units at all.
"""

import argparse
import fcntl
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from sim_bridge_common import gripper_sim_to_raw, load_calibration, raw_to_gripper_sim

MAX_EPISODES = 1              # start at 1 for a first real-hardware test; raise once trusted
FPS = 30
TASK = "Pick up the cube."
ROBOT_TYPE = "openarm_follower"
URDF_PATH = "/home/csl/Stanley_ws/lerobot_openarm/model/openarm_description.urdf"

LEFT_ARM_STATE_KEYS = [f"LJ{i}.pos" for i in range(1, 8)]
ACTION_NAMES = LEFT_ARM_STATE_KEYS + ["LJ8.pos"]   # identical to observation.state's names --
                                                     # both are the same joint-space representation
GRIPPER_MIN = 0.0     # matches the real gripper's mechanical range (see record_demos_openarm.py's
GRIPPER_MAX = 0.044   # JointMirrorBroadcaster / the URDF's finger joint limits) -- safety clip
                       # only, not a semantic open/close threshold like the EE-delta version used.
                       # Applied in the sim-convention frame, same as everywhere else in this script
                       # (see GRIPPER CALIBRATION note above) -- raw motor units never get clipped
                       # against these bounds directly.


def _get_obs_sim_gripper(robot, grip: dict) -> dict:
    """robot.get_observation(), with LJ8.pos mapped from raw motor units to the 0.0..0.044 sim
    convention (see GRIPPER CALIBRATION note above). Use this everywhere instead of calling
    robot.get_observation() directly, so nothing downstream ever sees a raw gripper reading."""
    obs = robot.get_observation()
    obs["LJ8.pos"] = float(np.clip(
        raw_to_gripper_sim(obs["LJ8.pos"], grip["open_raw"], grip["closed_raw"]),
        GRIPPER_MIN, GRIPPER_MAX,
    ))
    return obs


def _send_action_sim_gripper(robot, action: dict, grip: dict) -> None:
    """robot.send_action(), with LJ8.pos mapped from the 0.0..0.044 sim convention to raw motor
    units immediately before sending -- `action` itself is left untouched (a converted copy is
    sent) so callers can keep logging/plotting the sim-convention value they already have."""
    raw_action = dict(action)
    raw_action["LJ8.pos"] = gripper_sim_to_raw(action["LJ8.pos"], grip["open_raw"], grip["closed_raw"])
    robot.send_action(raw_action)

USBDEVFS_RESET = ord("U") << 8 | 20


def _usb_reset_for_video_node(video_index: int) -> None:
    """Power-cycle the USB device backing /dev/video<video_index> (see module docstring)."""
    sys_path = f"/sys/class/video4linux/video{video_index}/device"
    if not os.path.exists(sys_path):
        print(f"[WARN] {sys_path} not found -- skipping USB reset for video{video_index}")
        return

    d = os.path.realpath(sys_path)
    while d != "/" and not os.path.exists(os.path.join(d, "busnum")):
        d = os.path.dirname(d)
    if not os.path.exists(os.path.join(d, "busnum")):
        print(f"[WARN] could not resolve USB device for /dev/video{video_index} -- skipping reset")
        return

    with open(os.path.join(d, "busnum")) as f:
        busnum = int(f.read().strip())
    with open(os.path.join(d, "devnum")) as f:
        devnum = int(f.read().strip())
    usb_path = f"/dev/bus/usb/{busnum:03d}/{devnum:03d}"

    try:
        fd = os.open(usb_path, os.O_WRONLY)
        try:
            fcntl.ioctl(fd, USBDEVFS_RESET, 0)
        finally:
            os.close(fd)
        print(f"[INFO] USB-reset /dev/video{video_index} ({usb_path})")
    except OSError as e:
        print(f"[WARN] USB reset failed for /dev/video{video_index} ({usb_path}): {e}")


LIVE_VIEW_WINDOW = "OpenArm cameras (body | wrist | side) -- green label = model input, yellow = view only"

# Green = this feed is actually fed to the policy (whichever of body_cam/wrist_cam/front_cam are
# configured via `camera_config` -- see MODEL_CAM_KEYS in main()). Yellow = human-facing only (the
# standalone --side-cam-index camera, never part of the model's input regardless of checkpoint).
_MODEL_INPUT_COLOR = (0, 255, 0)   # green, BGR
_VIEW_ONLY_COLOR = (0, 220, 255)   # yellow, BGR


def _labeled_bgr(rgb_image: np.ndarray, label: str, color: tuple, *, height: int = None, width: int = None) -> np.ndarray:
    """RGB (from lerobot cameras) -> BGR (for cv2), resized to a target height (aspect-preserving
    width) or target width (aspect-preserving height) -- exactly one of the two must be given --
    with a label burned into the top-left corner so the combined view is legible without a legend."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    if height is not None:
        width = int(bgr.shape[1] * height / bgr.shape[0])
    else:
        height = int(bgr.shape[0] * width / bgr.shape[1])
    bgr = cv2.resize(bgr, (width, height))
    cv2.putText(bgr, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return bgr


def _build_combined_frame(
    obs: dict, side_frame: np.ndarray | None, model_cam_keys: list,
) -> np.ndarray | None:
    """Combine whichever model-input cameras are active (model_cam_keys, e.g. body_cam/wrist_cam or
    body_cam/wrist_cam/front_cam -- already captured this step as part of obs) and, if available,
    the standalone side camera's latest frame into one side-by-side BGR image for display/saving."""
    panels = []
    for key in model_cam_keys:
        if key in obs:
            label = f"{key.removesuffix('_cam')} (model input)"
            panels.append((obs[key], label, _MODEL_INPUT_COLOR))
    if side_frame is not None:
        panels.append((side_frame, "side (view only, NOT model input)", _VIEW_ONLY_COLOR))
    if not panels:
        return None
    panel_height = min(img.shape[0] for img, _, _ in panels)
    return np.hstack([_labeled_bgr(img, label, color, height=panel_height) for img, label, color in panels])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint", required=True,
        help=(
            "HF Hub repo id (or local pretrained_model dir) of the trained joint-space SmolVLA "
            "checkpoint to deploy, e.g. "
            "ethanCSL/openarm_visuomotor_no_domain_randomization_1000_joints."
        ),
    )
    parser.add_argument(
        "--body-cam-index", type=int, required=True,
        help="/dev/video index of the RealSense D435i mounted on the robot body (its color/RGB stream).",
    )
    parser.add_argument(
        "--wrist-cam-index", type=int, required=True,
        help="/dev/video index of the RealSense D435i mounted on the wrist (its color/RGB stream).",
    )
    parser.add_argument(
        "--front-cam-index", type=int, default=None,
        help=(
            "/dev/video index of a third camera named 'front_cam', REQUIRED for checkpoints "
            "trained with three cameras (e.g. ..._three_cams) -- unlike --side-cam-index, this "
            "one IS fed to the model as a real observation.images.front_cam input, same as body/ "
            "wrist. Omit entirely for a two-camera (body+wrist only) checkpoint."
        ),
    )
    parser.add_argument(
        "--side-cam-index", type=int, default=None,
        help=(
            "/dev/video index of an EXTRA camera (e.g. a side-view webcam at /dev/video12) shown "
            "in the live visualization / saved video only -- NOT fed to the model regardless of "
            "checkpoint (that's what --front-cam-index is for). Omit to skip it."
        ),
    )
    parser.add_argument(
        "--no-live-view", action="store_true",
        help="Disable the live cv2 window combining the active model-input cameras (+ --side-cam-index, if given) (on by default).",
    )
    parser.add_argument(
        "--save-video", type=str, default=None,
        help="Optional path (e.g. rollout.mp4) to save the same combined camera view to disk once the run ends.",
    )
    parser.add_argument(
        "--calibration", type=str, required=True,
        help=(
            "Path to calibration.json (see calibration.example.json / mirror_bridge.py's Phase 0 "
            "docs). Required for the gripper specifically -- confirmed OpenArmFollower's raw "
            "LJ8.pos range does not match this hardware's real open/closed positions (see "
            "GRIPPER CALIBRATION note above), so this script routes the gripper through the same "
            "gripper_sim_to_raw()/raw_to_gripper_sim() calibration transform replay_hf_sim_"
            "episode.py and mirror_bridge.py already use. The arm joints (LJ1-7) don't need this "
            "-- confirmed to already be correct in URDF-native radians."
        ),
    )
    parser.add_argument(
        "--inference-hz", type=float, default=30.0,
        help=(
            "Model inference / control-loop rate in Hz (default 30, matching the training data's "
            "recording rate). Lowering this gives the arm more time to physically reach each "
            "target before the next one is issued -- try e.g. 10-15 for a first real-hardware run."
        ),
    )
    parser.add_argument(
        "--max-joint-speed", type=float, default=1.0,
        help=(
            "Maximum left-arm + gripper joint speed in rad/s, enforced by clamping how far the "
            "model's raw target may move from the current measured joint angles each inference "
            "step (default 1.0 rad/s ~= 57 deg/s). The interpolation step alone only smooths this "
            "move in time, not in magnitude, so this is the main safety/smoothness control for a "
            "first run -- try e.g. 0.3 to start."
        ),
    )
    parser.add_argument(
        "--action-smoothing-alpha", type=float, default=1.0,
        help=(
            "EMA smoothing factor for the policy's raw 8D joint-target output: "
            "filtered = alpha*raw + (1-alpha)*filtered_prev. 1.0 = no smoothing (default -- there "
            "is no known noise-amplification mechanism for this checkpoint the way the EE-delta "
            "version had an IK-redundancy issue, so start by trusting the raw output). Lower this "
            "only if the diagnostic plot after a run shows the raw target oscillating."
        ),
    )
    parser.add_argument(
        "--max-episode-seconds", type=float, default=10.0,
        help=(
            "Wall-clock time limit per episode in seconds (default 10) -- an episode that hasn't "
            "hit the success condition or been manually stopped by then is cut short as a timeout."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    calib = load_calibration(args.calibration)
    grip = calib["left"]["gripper"]

    model = SmolVLAPolicy.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    # Normalization stats load from the checkpoint's own saved preprocessor/postprocessor
    # safetensors -- no separate --dataset argument needed (see module docstring).
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        args.checkpoint,
    )

    # See module docstring: both D435i units intermittently fail to start their color stream:
    # reset unconditionally right before connecting rather than only when a failure is detected.
    _usb_reset_for_video_node(args.body_cam_index)
    _usb_reset_for_video_node(args.wrist_cam_index)
    if args.front_cam_index is not None:
        _usb_reset_for_video_node(args.front_cam_index)
    time.sleep(1.0)  # let all devices finish re-enumerating before OpenCVCamera opens them

    camera_config = {
        "body_cam":  OpenCVCameraConfig(index_or_path=args.body_cam_index,  width=640, height=480, fps=FPS),
        "wrist_cam": OpenCVCameraConfig(index_or_path=args.wrist_cam_index, width=640, height=480, fps=FPS),
    }
    if args.front_cam_index is not None:
        # A real model input (unlike --side-cam-index) -- for checkpoints trained with three
        # cameras, e.g. ethanCSL/openarm_visuomotor_no_domain_randomization_1000_joints_three_cams.
        camera_config["front_cam"] = OpenCVCameraConfig(index_or_path=args.front_cam_index, width=640, height=480, fps=FPS)
    MODEL_CAM_KEYS = list(camera_config.keys())  # for _build_combined_frame's live-view/video labeling below

    robot_cfg = OpenArmFollowerConfig(
        right_port="can0",
        left_port="can1",
        enable_fd=True,
        model_path=URDF_PATH,
        cameras=camera_config,  # type: ignore
    )
    robot = OpenArmFollower(robot_cfg)
    robot.connect()

    # Optional EXTRA camera, purely for the human-facing live view / saved video below -- never
    # part of the model's input (unlike front_cam above, which -- if given -- is a real input
    # alongside body_cam/wrist_cam via `robot`'s own cameras, see MODEL_CAM_KEYS).
    side_cam = None
    if args.side_cam_index is not None:
        _usb_reset_for_video_node(args.side_cam_index)
        time.sleep(1.0)
        side_cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=args.side_cam_index, width=640, height=480, fps=FPS))
        try:
            side_cam.connect()
        except Exception as e:
            print(f"[WARN] Could not connect side camera at /dev/video{args.side_cam_index}: {e}")
            side_cam = None

    live_view = not args.no_live_view
    video_writer = None

    # Keep only the camera image entries from the generic (all-16-joint) hw feature set -- state
    # and action are both built manually below as the same 8 left-arm+gripper joint names, matching
    # exactly what convert_hdf5_to_lerobot.py wrote into the training dataset. Because both share
    # the identical LJ1.pos..LJ8.pos names, build_inference_frame's state lookup (which requires
    # exact real-observation-dict key matches) and make_robot_action's action labeling (which is
    # purely positional) both work correctly with no separate naming scheme needed for the two.
    full_obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    image_obs_features = {k: v for k, v in full_obs_features.items() if v["dtype"] in ("image", "video")}
    state_features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": LEFT_ARM_STATE_KEYS + ["LJ8.pos"]},
    }
    action_features = {
        "action": {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES},
    }
    dataset_features = {**action_features, **state_features, **image_obs_features}

    model_dt = 1.0 / args.inference_hz
    max_step_rad = args.max_joint_speed * model_dt
    interp_steps = 10
    control_dt = model_dt / interp_steps

    # For the target-vs-actual tracking plot: dense target curve (once per control substep) vs.
    # sparser actual-measured curve (once per inference step) for the same 8 joints.
    PLOT_JOINTS = ACTION_NAMES
    target_time_log, target_log = [], {k: [] for k in PLOT_JOINTS}
    actual_time_log, actual_log = [], {k: [] for k in PLOT_JOINTS}

    # Diagnostic only: the policy's RAW output (before the EMA filter) -- lets us tell whether a
    # drift/oscillation seen in the target-vs-actual plot originates in the policy itself or gets
    # introduced by the smoothing/clamp chain downstream of it.
    raw_time_log, raw_log = [], {k: [] for k in ACTION_NAMES}
    start_time = time.perf_counter()

    try:
        for ep in range(MAX_EPISODES):
            print(f"Starting episode {ep}...")
            first = True
            interp_start = time.perf_counter()
            episode_start = time.perf_counter()
            step = 0
            filtered_target = np.zeros(8)

            while time.perf_counter() - episode_start < args.max_episode_seconds:
                model_start = time.perf_counter()

                # _get_obs_sim_gripper (not robot.get_observation() directly) maps LJ8.pos through
                # calibration -- see GRIPPER CALIBRATION note above -- and clips it to
                # [GRIPPER_MIN, GRIPPER_MAX] as part of that conversion, which also covers the
                # transient bad CAN-bus reads seen before (gripper briefly reading ~-1.0 rad, ~20x
                # past its real range, in the first ~0.7s right after connect): without clipping,
                # the interpolation loop below would blend its start point (prev_action = obs) from
                # that bad reading even though the *target* end point was already clamped.
                obs = _get_obs_sim_gripper(robot, grip)

                # Live cv2 view / video recording of body+wrist(+side) cameras -- purely a
                # human-facing preview, no bearing on the model's input (built from `obs` above).
                side_frame = None
                if side_cam is not None:
                    try:
                        side_frame = side_cam.async_read()
                    except Exception as e:
                        print(f"[WARN] side camera read failed: {e}")

                combined_frame = None
                if live_view or args.save_video:
                    combined_frame = _build_combined_frame(obs, side_frame, MODEL_CAM_KEYS)

                if live_view and combined_frame is not None:
                    try:
                        cv2.imshow(LIVE_VIEW_WINDOW, combined_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            print("[INFO] 'q' pressed in live view window, stopping...")
                            raise KeyboardInterrupt
                    except cv2.error as e:
                        print(f"[WARN] Live view failed ({e}) -- disabling for the rest of the run.")
                        live_view = False

                if args.save_video and combined_frame is not None:
                    if video_writer is None:
                        h, w = combined_frame.shape[:2]
                        video_writer = cv2.VideoWriter(
                            args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), args.inference_hz, (w, h),
                        )
                    video_writer.write(combined_frame)

                if first:
                    first = False
                    filtered_target = np.array([obs[k] for k in ACTION_NAMES], dtype=np.float64)
                    continue

                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task=TASK,
                    robot_type=ROBOT_TYPE,
                )
                obs_processed = preprocess(obs_frame)

                raw_action = model.select_action(obs_processed)
                raw_action = postprocess(raw_action)
                policy_action = make_robot_action(raw_action, dataset_features)  # {LJ1.pos: v, ...}

                raw_target = np.array([policy_action[n] for n in ACTION_NAMES], dtype=np.float64)

                raw_time_log.append(time.perf_counter() - start_time)
                for name in ACTION_NAMES:
                    raw_log[name].append(policy_action[name])

                # Optional EMA smoothing (see --action-smoothing-alpha help; off by default).
                filtered_target = (
                    args.action_smoothing_alpha * raw_target
                    + (1.0 - args.action_smoothing_alpha) * filtered_target
                )

                current_q = np.array([obs[k] for k in ACTION_NAMES], dtype=np.float64)

                # Speed limit: clamp how far the target may move from the CURRENT measured joints
                # this inference step, to --max-joint-speed * model_dt radians -- see module
                # docstring for why this (not the interpolation loop alone) is what actually
                # bounds jerkiness/speed.
                delta_q = np.clip(filtered_target - current_q, -max_step_rad, max_step_rad)
                target_q = current_q + delta_q
                target_q[-1] = float(np.clip(target_q[-1], GRIPPER_MIN, GRIPPER_MAX))  # gripper safety clip

                # Full 16-joint target: start from the current observation (holds the right arm
                # and its gripper constant -- never part of this action space), then overwrite
                # only the left arm + left gripper with the model's target.
                target_action = dict(obs)
                for name, value in zip(ACTION_NAMES, target_q):
                    target_action[name] = float(value)

                model_time = time.perf_counter() - model_start

                # Interpolate from the CURRENT measured observation (not the previous action)
                # toward the new target -- self-correcting against any tracking error.
                prev_action = obs
                next_time = time.perf_counter()
                for i in range(interp_steps):
                    next_time += control_dt
                    alpha = (i + 1) / interp_steps
                    interp_action = {
                        joint: prev_action[joint] + (target_action[joint] - prev_action[joint]) * alpha
                        for joint in target_action.keys()
                        if joint.endswith(".pos")
                    }
                    _send_action_sim_gripper(robot, interp_action, grip)

                    t_now = time.perf_counter() - start_time
                    target_time_log.append(t_now)
                    for k in PLOT_JOINTS:
                        target_log[k].append(interp_action[k])

                    now = time.perf_counter()
                    sleep_time = next_time - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_time = now

                # Measure where the arm actually ended up after this step's interpolated moves.
                settled_obs = _get_obs_sim_gripper(robot, grip)
                actual_time_log.append(time.perf_counter() - start_time)
                for k in PLOT_JOINTS:
                    actual_log[k].append(settled_obs[k])

                interp_time = time.perf_counter() - interp_start
                elapsed = time.perf_counter() - episode_start
                print(
                    f"[step {step} @ {elapsed:.1f}s/{args.max_episode_seconds:.1f}s] "
                    f"model: {model_time*1000:.1f}ms  "
                    f"interp(total): {interp_time*1000:.1f}ms  "
                    f"per step: {(interp_time/interp_steps)*1000:.1f}ms"
                )
                step += 1

            print(f"Episode {ep} ended after {time.perf_counter() - episode_start:.1f}s ({step} steps).")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected, stopping loop...")

    finally:
        try:
            robot.disconnect()
        except Exception as e:
            print(f"[WARN] Robot disconnect failed: {e}")

        if side_cam is not None:
            try:
                side_cam.disconnect()
            except Exception as e:
                print(f"[WARN] Side camera disconnect failed: {e}")

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # no GUI backend available (see live_view's own cv2.error handling above)

        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Saved combined camera view to {args.save_video}")

        print("[INFO] Plotting target vs. actual joint tracking...")
        if target_time_log:
            fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True)
            for ax, joint in zip(axes.flat, PLOT_JOINTS):
                ax.plot(target_time_log, target_log[joint], label="target", linewidth=1)
                ax.plot(actual_time_log, actual_log[joint], label="actual", marker="o", markersize=2, linewidth=1)
                ax.set_title(joint)
                ax.set_ylabel("rad")
                ax.grid(True)
            axes.flat[0].legend(loc="upper right")
            for ax in axes[-1, :]:
                ax.set_xlabel("Time (s)")
            fig.suptitle("Left arm + gripper: commanded target vs. measured actual (joint-space model)")
            fig.tight_layout()

        print("[INFO] Plotting raw policy output (before EMA filter)...")
        if raw_time_log:
            fig2, axes2 = plt.subplots(4, 2, figsize=(12, 14), sharex=True)
            for ax, name in zip(axes2.flat, ACTION_NAMES):
                ax.plot(raw_time_log, raw_log[name], linewidth=1)
                ax.set_title(name)
                ax.grid(True)
            fig2.suptitle("Raw policy output (pre-filter) -- for comparing against the tracking plot")
            fig2.tight_layout()

        if target_time_log or raw_time_log:
            plt.show()
        else:
            print("[WARN] No data to plot.")


if __name__ == "__main__":
    main()
