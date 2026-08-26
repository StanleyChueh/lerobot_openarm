"""
Autonomous SmolVLA rollout on the real OpenArm follower, for a checkpoint trained on JOINT-SPACE
actions/observations (both are the same 16D dual-arm vector LJ1.pos..LJ8.pos then RJ1.pos..RJ8.pos
-- see scripts/tools/convert_hdf5_to_lerobot.py's LJ_IDX/LJ_NAMES + RJ_IDX/RJ_NAMES in the
IsaacLab repo) -- e.g. ethanCSL/openarm_visuomotor_no_domain_randomization_1000_joints. This is
the simplified sibling of deploy_smolvla_pickup.py (the EE-delta-action version): since the model
now predicts joint targets directly, there is no IK bridge here at all -- no ik_bridge.py import,
no DifferentialIK solve, no redundant-DOF noise-amplification problem (that was specifically a
DLS/Jacobian artifact of the EE-delta version, structurally absent here).

SAFETY: this moves the real robot autonomously with NO human-in-the-loop correction step, and
runs an UNTESTED checkpoint on real hardware for the first time.
  - Start with MAX_EPISODES = 1 and watch closely before trusting longer unsupervised runs.
  - Start with a conservative --max-joint-speed (e.g. 0.3) and --inference-hz (e.g. 10) for the
    first run, same as any new checkpoint's first real-hardware test -- see deploy_smolvla_
    pickup.py's own history for why (jerky/wrong output on a first run is not unusual).
  - Keep emergency_disable.py within reach.
  - Ctrl+C stops the loop and disconnects the robot; it does not power off the motors.

Usage (dual-arm checkpoint: body + BOTH wrist cameras fed to the model):
  python deploy_smolvla_pickup_jointspace.py \\
      --checkpoint ethanCSL/<dual_arm_checkpoint> \\
      --body-cam-index 4 --wrist-cam-index 10 --right-wrist-cam-index 12 \\
      --inference-hz 10 --max-joint-speed 0.3 --max-episode-seconds 10

Usage (two-camera checkpoint, body + left wrist only):
  python deploy_smolvla_pickup_jointspace.py \\
      --checkpoint ethanCSL/openarm_visuomotor_no_domain_randomization_1000_joints \\
      --body-cam-index 4 --wrist-cam-index 10 --side-cam-index 12 \\
      --inference-hz 10 --max-joint-speed 0.3 --max-episode-seconds 10

CAMERAS -- the model-input keys must match the training dataset's observation.images.* keys
exactly, so the names here are deliberately the dataset's own, not "left"/"right" symmetric ones:

  --body-cam-index         -> observation.images.body_cam         (required)
  --wrist-cam-index        -> observation.images.wrist_cam        (required; this is the LEFT
                              wrist -- the dataset has no "left_" prefix on it, only the right
                              wrist camera is prefixed)
  --right-wrist-cam-index  -> observation.images.right_wrist_cam  (the dual-arm checkpoints'
                              third camera; omit for an older left-arm-only two-camera checkpoint)
  --front-cam-index        -> observation.images.front_cam        (the older "..._three_cams"
                              variant's third camera; unrelated to the right wrist one)

Only pass the optional ones for a checkpoint actually trained with them -- the preprocessor will
otherwise reject/ignore an unexpected image key, and a camera the checkpoint DOES expect but that
isn't passed here leaves the policy blind on that input. main() prints the checkpoint's own
expected image keys next to the configured ones at startup and warns on any mismatch, so check
that line before trusting a run. --side-cam-index is unrelated to all of the above: it's an EXTRA
camera shown for humans only, never fed to the model, regardless of checkpoint.

LIVE CAMERA VIEW: whichever cameras are actually active -- the model's own (body/wrist, plus the
right-wrist and/or front ones if their indices are given) and/or the view-only --side-cam-index
one -- are combined
side-by-side into one cv2.imshow window, each labeled green ("model input") or yellow ("view
only") so it's never ambiguous which cameras the policy actually sees, updated live during
evaluation. Pass --no-live-view to turn this off (e.g. headless/no X server), and/or --save-video
PATH.mp4 to additionally save the same combined view to disk once the run ends.

The live window needs a cv2 build with GUI support, which the default install does NOT have:
lerobot depends on opencv-python-headless ("GUI: NONE"), where cv2.imshow exists but raises when
called. main() detects that at startup and skips the window with a note rather than letting it
fail mid-rollout. --save-video is unaffected (videoio works headless); to actually get the live
window, replace the wheel: uv pip uninstall opencv-python-headless && uv pip install opencv-python.

Action space: action[t] was derived in training data as state[t+1] -- the NEXT measured joint
position, a standard proxy for "commanded joint target" when demos are recorded via position
tracking rather than direct joint-target logging (see convert_hdf5_to_lerobot.py's _load_episode
docstring in IsaacLab). So the model's raw 16D output is interpreted directly as ABSOLUTE target
joint angles, not deltas and not a semantic +-1 gripper flag like the older EE-delta checkpoint
used. Unlike the EE-delta version (and unlike this script's earlier left-arm-only revision), BOTH
arms are now driven by the model: the vector is laid out as one left-arm block then one right-arm
block (ACTION_NAMES = LJ_NAMES + RJ_NAMES), because that is the order the real robot's action dict
is written and read in. Nothing is held at its current pose any more.

Speed limiting still applies the same way as the EE-delta version, now across all 16 joints: the
raw target is clamped to move at most --max-joint-speed * (one cycle period) radians from the
CURRENT measured joint angles each inference step, then the existing interpolation loop smooths
that move in time across control substeps -- the substeps being spread over whatever is left of
the cycle budget after inference, so the loop period converges on 1/--inference-hz instead of
1/--inference-hz PLUS the inference time. "One cycle period" is the measured period by default,
not the requested one (--speed-clamp-basis): on hardware where a cycle costs more than the
requested budget, converting the speed against the nominal period is what silently turned
--max-joint-speed 1.5 into ~0.49 rad/s at a requested 20 Hz that only held 6.5 Hz.
Optional EMA smoothing (--action-smoothing-alpha) is available but defaults to
off (1.0) -- unlike the EE-delta checkpoint, there's no known noise-amplification mechanism here
yet, so start trusting the raw model output and only add smoothing if the diagnostic plot below
shows it's actually needed.

Camera / USB-reset / dataset-loading notes are unchanged from deploy_smolvla_pickup.py -- see
that file's docstring for the full explanation (D435i intermittent color-stream hang and the
USBDEVFS_RESET workaround, why norm stats load from the checkpoint directly with no --dataset
arg, why camera dict keys are "body_cam"/"wrist_cam"/"right_wrist_cam" not "camera1"/"camera2").

GRIPPER CALIBRATION -- do not skip --calibration: confirmed on real hardware (2026-07) that
OpenArmFollower's raw gripper range for LJ8.pos/RJ8.pos (the 0.0..0.044 rad "sim convention" that
the arm joints and every other script in this project treat as URDF-native and therefore already
correct) does NOT correspond to the grippers' real open/closed positions -- commanding both 0.044 and 0.0 directly
left the real gripper closed firmly either way. replay_hf_sim_episode.py and mirror_bridge.py
never hit this because they always route the gripper through calibration.json's gripper_sim_to_raw()
(open_raw/closed_raw, measured per-unit during Phase 0 setup) rather than sending LJ8.pos raw --
this script now does the same for BOTH grippers: every read gripper value is mapped raw->sim via
raw_to_gripper_sim() immediately after robot.get_observation(), and every value about to be sent is
mapped sim->raw via gripper_sim_to_raw() immediately before robot.send_action() -- each side
against its own calibration.json open_raw/closed_raw (calib["left"]/["right"], see
_gripper_calib()). Everywhere in between (the model's input/output, binarization,
--max-joint-speed clamping, the tracking plots) stays in the same 0.0..0.044 sim convention the
checkpoint was trained on, matching observation.state/action's LJ8.pos/RJ8.pos semantics exactly.
Only the two boundary points touch raw motor units at all.

GRIPPER BINARIZATION: the policy's continuous gripper prediction is snapped to an open or closed
command before being sent, per arm (LJ8.pos/RJ8.pos, GRIPPER_IDX), after the optional EMA filter,
in the sim convention. Both the thresholds and the closed command are read off the training
dataset's own action distribution (152,937 frames of ethanCSL/openarm_visuomotor_VR_pringles_V8_
generated_500), NOT assumed:

  bin            LJ8      RJ8     <- fraction of frames
  0.0140-0.0160  2.1%    16.2%    closed cluster (LJ8 also spreads up to 0.020)
  0.0260-0.0360  26.0%    7.7%    LJ8-only hump: operator modulating grip while holding
  0.0400-0.0420  0.6%     0.6%    the valley -- the only genuinely empty band on BOTH arms
  0.0420-0.0440  58.7%   62.5%    open cluster

Two things that follow, both of which the first single-threshold version got wrong:

  - The thresholds straddle the 0.0400-0.0420 valley (GRIPPER_CLOSE_BELOW / GRIPPER_OPEN_ABOVE)
    with the previous decision held in between -- a Schmitt trigger. A single threshold at
    0.95 * GRIPPER_MAX = 0.0418 lands INSIDE the open cluster, splitting the single most common
    state in the data down the middle, so ordinary prediction noise flipped the command between
    open and closed every inference step. Observed on hardware 2026-08-18: LJ8's commanded target
    oscillated 0.0 <-> 0.044 at the inference rate while the measured gripper never moved off
    open, because a full open<->closed sweep is ~1.18 rad of motor travel (open_raw 0.0 ->
    closed_raw -1.1755) and no motor can track that as a 10 Hz square wave. The right gripper
    escaped it only because its prediction happened to sit clear of the threshold.
  - Closed commands GRIPPER_CLOSED_CMD (0.015), the demos' measured closed position, not
    GRIPPER_MIN. The demos never reach 0.0 at all -- the minimum over the whole dataset is 0.0044
    (LJ8) / 0.0063 (RJ8), and the closed peak of both arms is 0.0155 -- so commanding 0.0 squeezes
    harder than any demo ever did, against an object whose demo grasp equilibrium is 0.015.

CAVEAT: LJ8's 0.026-0.036 hump is not transit. Its runs last 2-4.6 s (30 fps), i.e. the left
gripper is genuinely held part-closed while manipulating, and binarizing collapses that onto the
0.015 grasp. If the left hand's task needs that intermediate width, binarization is the wrong
model for it and the raw prediction should be passed through instead.
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
from attention_overlay import AttentionDump
from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from sim_bridge_common import (
    approach_pose,
    check_arms_not_crossed,
    gripper_sim_to_raw,
    load_calibration,
    raw_to_gripper_sim,
    sim_init_pose_action,
)

MAX_EPISODES = 100              # start at 1 for a first real-hardware test; raise once trusted
FPS = 30
# Must match the training dataset's task string VERBATIM -- SmolVLA conditions on it, and the
# pringles checkpoints were trained on exactly one task string (read off meta/tasks.parquet of
# ethanCSL/openarm_visuomotor_VR_pringles_V12/V13, which V14 inherits). "Pick up the cube." was
# left over from the cube checkpoint and puts the VLM prefix out of distribution on every step.
TASK = "Pick up the Pringles can with the right arm, hand it to the left arm"
ROBOT_TYPE = "openarm_follower"
URDF_PATH = "/home/csl/Stanley_ws/IsaacLab/source/isaaclab_assets/data/v1_camera_isaac/urdf/v1_camera.urdf"

LJ_NAMES = [f"LJ{i}.pos" for i in range(1, 8)] + ["LJ8.pos"]
RJ_NAMES = [f"RJ{i}.pos" for i in range(1, 8)] + ["RJ8.pos"]

# Dual-arm layout is left-arm-block THEN right-arm-block (not interleaved like the articulation),
# because that is the order the real robot's action dict is written and read in -- and the order
# convert_hdf5_to_lerobot.py's LJ_IDX/LJ_NAMES + RJ_IDX/RJ_NAMES wrote the dataset's
# observation.state / action vectors in (its LJ_IDX/RJ_IDX are the *articulation's* interleaved
# joint indices, which is exactly why the dataset's own block layout has to be stated separately).
ACTION_NAMES = LJ_NAMES + RJ_NAMES   # identical to observation.state's names -- both are the same
ACTION_DIM = len(ACTION_NAMES)       # joint-space representation (16 = 2 arms x (7 + gripper))

GRIPPER_NAMES = ("LJ8.pos", "RJ8.pos")
GRIPPER_IDX = [ACTION_NAMES.index(n) for n in GRIPPER_NAMES]   # 7 (left), 15 (right)
GRIPPER_MIN = 0.0     # matches the real gripper's mechanical range (see record_demos_openarm.py's
GRIPPER_MAX = 0.044   # JointMirrorBroadcaster / the URDF's finger joint limits) -- safety clip
                       # only. Applied in the sim-convention frame, same as everywhere else in this
                       # script (see GRIPPER CALIBRATION note above) -- raw motor units never get
                       # clipped against these bounds directly.

# Both grippers are BINARIZED before being commanded, through a Schmitt trigger rather than a
# single threshold (see GRIPPER BINARIZATION note above for the measured distribution these come
# from). Between the two thresholds the previous decision is held, which is what stops the command
# from flipping every inference step when the policy's output sits near the boundary.
GRIPPER_CLOSE_BELOW = 0.035   # predict below this -> close
GRIPPER_OPEN_ABOVE = 0.042    # predict above this -> open;  in between -> keep previous decision
GRIPPER_OPEN_CMD = GRIPPER_MAX   # 0.044, the demos' open position
GRIPPER_CLOSED_CMD = 0.0001       # the demos' *measured* closed position -- NOT GRIPPER_MIN


def _gripper_calib(calib: dict) -> dict:
    """{"LJ8.pos": <left gripper calib section>, "RJ8.pos": <right>} -- both arms are part of this
    action space, so both grippers get calibrated, each against its own open_raw/closed_raw."""
    return {"LJ8.pos": calib["left"]["gripper"], "RJ8.pos": calib["right"]["gripper"]}


def _get_obs_sim_gripper(robot, grip: dict) -> dict:
    """robot.get_observation(), with LJ8.pos/RJ8.pos mapped from raw motor units to the 0.0..0.044
    sim convention (see GRIPPER CALIBRATION note above). Use this everywhere instead of calling
    robot.get_observation() directly, so nothing downstream ever sees a raw gripper reading."""
    obs = robot.get_observation()
    for name, sec in grip.items():
        obs[name] = float(np.clip(
            raw_to_gripper_sim(obs[name], sec["open_raw"], sec["closed_raw"]),
            GRIPPER_MIN, GRIPPER_MAX,
        ))
    return obs


def _send_action_sim_gripper(robot, action: dict, grip: dict, *, debug_tag: str = None) -> None:
    """robot.send_action(), with LJ8.pos/RJ8.pos mapped from the 0.0..0.044 sim convention to raw
    motor units immediately before sending -- `action` itself is left untouched (a converted copy
    is sent) so callers can keep logging/plotting the sim-convention value they already have.

    The empty target_vel is deliberate and is NOT an optional argument: OpenArmFollower.send_action
    gained a required `target_vel` parameter in abd1499, and `vel = target_vel or {}` inside it
    means {} reproduces exactly the dq=0.0 feedforward every MITParam used before that commit.
    Omitting it (as this script did until now, along with every other one-argument caller that
    commit left behind) raises TypeError on the very FIRST send -- which on the hardware looks
    like the arm's CAN LED blinking green once (connect()'s enable_all) and then nothing, because
    no MIT command is ever transmitted.

    debug_tag, when given, prints the exact per-motor values handed to the motors just before the
    call (see --debug-actions) -- raw units for the grippers, since that is what LJ8/RJ8 actually
    receive, with the sim-convention value they came from shown alongside."""
    raw_action = dict(action)
    for name, sec in grip.items():
        raw_action[name] = gripper_sim_to_raw(action[name], sec["open_raw"], sec["closed_raw"])

    if debug_tag is not None:
        parts = []
        for n in ACTION_NAMES:
            if n in GRIPPER_NAMES:
                parts.append(f"{n}={raw_action[n]:+.4f}(sim {action[n]:.4f})")
            else:
                parts.append(f"{n}={raw_action[n]:+.4f}")
        print(f"[send {debug_tag}] " + " ".join(parts), flush=True)

    robot.send_action(raw_action, {})


def _binarize_gripper(value: float, prev_cmd: float) -> float:
    """Snap a continuous sim-convention gripper prediction to the open or closed command.

    Hysteresis, not a single threshold: `prev_cmd` (the last decision, one of GRIPPER_OPEN_CMD /
    GRIPPER_CLOSED_CMD) is held whenever the prediction lands in the dead band between the two
    thresholds. See the GRIPPER BINARIZATION note in the module docstring for the measured
    distribution the thresholds come from and what the single-threshold version did wrong."""
    if value < GRIPPER_CLOSE_BELOW:
        return GRIPPER_CLOSED_CMD
    if value > GRIPPER_OPEN_ABOVE:
        return GRIPPER_OPEN_CMD
    return prev_cmd


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


LIVE_VIEW_WINDOW = "OpenArm cameras -- green label = model input, yellow = view only"

# Green = this feed is actually fed to the policy (whichever of body_cam/wrist_cam/right_wrist_cam/
# front_cam are configured via `camera_config` -- see MODEL_CAM_KEYS in main()). Yellow = human-
# facing only (the standalone --side-cam-index camera, never part of the model's input).
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


def _cv2_has_gui() -> bool:
    """Whether this cv2 build can actually open a window.

    lerobot depends on opencv-python-headless, which is built with "GUI: NONE": cv2.imshow still
    EXISTS there and only raises when called, so this can't be detected with hasattr -- without
    this check the failure surfaces mid-rollout as an OpenCV "rebuild the library with GTK+"
    error, which reads like a missing system library rather than the wrong wheel."""
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            # A real GUI build prints either "GUI: GTK3" or an empty value followed by indented
            # GTK+/Qt sub-entries; only the headless build prints the literal "NONE".
            return line.split(":", 1)[1].strip().upper() != "NONE"
    return True  # unrecognized build info -- let the existing cv2.error path handle it instead


def _build_combined_frame(
    obs: dict, side_frame: np.ndarray | None, model_cam_keys: list,
) -> np.ndarray | None:
    """Combine whichever model-input cameras are active (model_cam_keys, e.g. body_cam/wrist_cam or
    body_cam/wrist_cam/right_wrist_cam -- already captured this step as part of obs) and, if available,
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


def _expected_image_inputs(model, preprocess) -> list:
    """The observation.images.* keys the checkpoint expects to be HANDED, as camera names.

    Not simply model.config.image_features: a checkpoint's preprocessor may carry a
    RenameObservationsProcessorStep, in which case config.image_features holds the POST-rename
    names and feeding those would be wrong. Confirmed on ethanCSL/openarm_visuomotor_VR_pringles_
    V8_generated_500, whose config lists camera1/camera2/camera3 while its rename map is
    right_wrist_cam->camera1, wrist_cam->camera2, body_cam->camera3 -- i.e. the real inputs are
    the dataset's own key names, and comparing against config.image_features directly would report
    a mismatch on a perfectly correct setup. Invert the rename map to recover what to feed."""
    rename = {}
    for step in getattr(preprocess, "steps", []):
        rename.update(getattr(step, "rename_map", None) or {})
    to_input = {dst: src for src, dst in rename.items()}
    return sorted(
        to_input.get(k, k).removeprefix("observation.images.") for k in model.config.image_features
    )


def _image_key_labels(model, preprocess) -> dict:
    """Post-rename image key -> camera name, for labelling the attention panels.

    Same rename inversion as _expected_image_inputs, kept as a mapping instead of a sorted list
    because the attention view needs to label a SPECIFIC key, not just know the set."""
    rename = {}
    for step in getattr(preprocess, "steps", []):
        rename.update(getattr(step, "rename_map", None) or {})
    to_input = {dst: src for src, dst in rename.items()}
    return {
        k: to_input.get(k, k).removeprefix("observation.images.")
        for k in model.config.image_features
    }


def _check_checkpoint_matches(model, preprocess, model_cam_keys: list) -> None:
    """Compare what the checkpoint expects against what this invocation is set up to feed it.

    Both halves are easy to get wrong from the command line and neither fails loudly on its own: a
    forgotten --right-wrist-cam-index just leaves the policy blind on an input it was trained with,
    and an older left-arm-only 8D checkpoint run through this 16D script would have its output
    mapped onto the wrong joints entirely. Camera differences only warn (the preprocessor's own
    handling of an unexpected key is checkpoint-specific), but an action-dimension mismatch is
    fatal -- there is no safe way to interpret it on real hardware.

    Deliberately NOT checked: observation.state's shape. The same checkpoint above records it as
    6 in config.json while its normalization stats, its dataset and its actual runtime tensor are
    all 16 -- that field is stale metadata SmolVLA never enforces (it pads state to max_state_dim
    anyway), so checking it would only produce false alarms."""
    expected_cams = _expected_image_inputs(model, preprocess)
    configured_cams = sorted(model_cam_keys)
    print(f"[INFO] checkpoint's image inputs: {expected_cams}")
    print(f"[INFO] cameras configured here:   {configured_cams}")
    if expected_cams != configured_cams:
        missing = [c for c in expected_cams if c not in configured_cams] or ["none"]
        unexpected = [c for c in configured_cams if c not in expected_cams] or ["none"]
        print(f"[WARN] camera mismatch -- not provided: {missing}, not trained on: {unexpected}")

    action_ft = model.config.action_feature
    if action_ft is not None and int(np.prod(action_ft.shape)) != ACTION_DIM:
        raise SystemExit(
            f"[FATAL] checkpoint's action dimension is {int(np.prod(action_ft.shape))}, but this "
            f"script commands {ACTION_DIM} joints ({ACTION_NAMES}). Refusing to run: the output "
            f"would be mapped onto the wrong joints. Use a checkpoint trained on the dual-arm "
            f"16D joint-space action space (see the Action space note in the module docstring)."
        )


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
        help=(
            "/dev/video index of the RealSense D435i mounted on the LEFT wrist (its color/RGB "
            "stream) -- fed to the model as observation.images.wrist_cam (the dataset's key for "
            "the left wrist has no 'left_' prefix; only the right one is prefixed)."
        ),
    )
    parser.add_argument(
        "--right-wrist-cam-index", type=int, default=None,
        help=(
            "/dev/video index of the camera on the RIGHT wrist, fed to the model as "
            "observation.images.right_wrist_cam. REQUIRED for the dual-arm checkpoints trained "
            "with it (body_cam + wrist_cam + right_wrist_cam); omit for an older left-arm-only "
            "two-camera checkpoint."
        ),
    )
    parser.add_argument(
        "--front-cam-index", type=int, default=None,
        help=(
            "/dev/video index of a camera named 'front_cam', for the older checkpoints trained "
            "with one (e.g. ..._three_cams) -- unlike --side-cam-index, this one IS fed to the "
            "model as a real observation.images.front_cam input, same as body/wrist. Unrelated to "
            "--right-wrist-cam-index. Omit unless the checkpoint was trained with a front camera."
        ),
    )
    parser.add_argument(
        "--side-cam-index", type=int, default=None,
        help=(
            "/dev/video index of an EXTRA camera (e.g. a side-view webcam at /dev/video12) shown "
            "in the live visualization / saved video only -- NOT fed to the model regardless of "
            "checkpoint (that's what --right-wrist-cam-index / --front-cam-index are for). Omit "
            "to skip it."
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
            "target before the next one is issued -- try e.g. 10-15 for a first real-hardware run. "
            "This is a BUDGET, not a guarantee: one cycle costs a full 16-joint CAN read plus the "
            "policy forward, and if that exceeds 1/hz the loop free-runs slower and says so (each "
            "step line prints the achieved Hz, and a one-shot [WARN] names an --inference-hz the "
            "loop can actually hold). Every episode ends with an achieved-vs-requested summary."
        ),
    )
    parser.add_argument(
        "--max-joint-speed", type=float, default=1.0,
        help=(
            "Maximum joint speed in rad/s for all 16 dual-arm joints (grippers included), "
            "enforced by clamping how far the model's raw target may move from the current "
            "measured joint angles each inference "
            "step (default 1.0 rad/s ~= 57 deg/s). What one 'step' is worth in seconds comes from "
            "--speed-clamp-basis, not from --inference-hz directly. The interpolation step alone only smooths this "
            "move in time, not in magnitude, so this is the main safety/smoothness control for a "
            "first run -- try e.g. 0.3 to start."
        ),
    )
    parser.add_argument(
        "--speed-clamp-basis", choices=("measured", "nominal"), default="measured",
        help=(
            "Which cycle period --max-joint-speed is converted against (it is a speed; the clamp "
            "needs a distance per cycle). 'measured' (default) uses the SHORTEST period seen in "
            "the last 20 cycles -- the shortest, not the average, so the distance stays safe on "
            "the loop's fastest cycle -- capped at 2x the requested period, which keeps "
            "--max-joint-speed meaning what it says even when the loop cannot keep up. 'nominal' "
            "uses 1/--inference-hz, the old "
            "behaviour: whenever the loop overruns, the arm moves proportionally slower than "
            "asked (a 20 Hz request holding 6.5 Hz turned --max-joint-speed 1.5 into ~0.49 rad/s). "
            "Use 'nominal' if you deliberately want the slower, more conservative motion."
        ),
    )
    parser.add_argument(
        "--no-attention-dump", dest="attention_dump", action="store_false",
        help=(
            "Skip writing attention dumps. On by default: one PNG per policy re-plan holding the "
            "action expert's cross-attention over each camera AND the VLM's prefix self-attention "
            "over the same images, plus attention_log.csv with each chunk's per-modality shares "
            "(images / task text / robot state) and per-camera spatial entropy. Costs +23ms on a "
            "re-plan step and nothing on the steps that pop a cached action."
        ),
    )
    parser.add_argument(
        "--attention-dir", type=str, default=None,
        help=(
            "Where to write the attention dumps (default: ./attention_<timestamp> in the current "
            "directory). Point two runs at their own directories -- one with the target object "
            "present, one without -- to compare them."
        ),
    )
    parser.add_argument(
        "--n-action-steps", type=int, default=None,
        help=(
            "How many actions from each predicted chunk to execute before running the policy "
            "again. Defaults to the checkpoint's own value, which for these checkpoints is 50 -- "
            "i.e. one look at the cameras every 50 control steps, and blind in between (2.5s at "
            "--inference-hz 20, 6.7s at 7.5). Lower it for closed-loop behaviour: the policy then "
            "re-plans from the CURRENT observation every N steps, at the cost of one forward pass "
            "every N steps -- measured 90ms warm on this machine, i.e. ~9ms/step amortised at "
            "N=10 and ~18ms/step at N=5, against a 50ms budget at 20 Hz. Cannot exceed the "
            "checkpoint's chunk_size."
        ),
    )
    parser.add_argument(
        "--no-probe-timing", dest="probe_timing", action="store_false",
        help=(
            "Skip the startup timing probe. The probe measures the CAN read and camera reads "
            "separately over 20 samples and says whether --inference-hz is reachable at all, "
            "before the arm is asked to move -- it costs about a second and needs no hardware "
            "motion, so there is rarely a reason to skip it."
        ),
    )
    parser.add_argument(
        "--can-recv-rounds", type=int, default=16,
        help=(
            "recv_all() calls per position read, per arm. Each round drains buffered motor "
            "feedback; rounds beyond the first exist to mop up the frames send_action leaves "
            "unread, so enough of them must run to empty the socket every cycle or the backlog "
            "drifts and the read cost drifts with it. Cheap now that they are bounded by "
            "--can-mop-timeout-us (the follower's own default is 8 rounds at the full 50ms, i.e. "
            "up to 801ms per read)."
        ),
    )
    parser.add_argument(
        "--can-first-timeout-us", type=int, default=2_000,
        help=(
            "Microseconds the FIRST recv_all() waits -- the only round with a refresh_all() "
            "response outstanding, so this is the one that should be patient. 50000 (the "
            "follower default) was measured to block in full without collecting anything, so it "
            "buys stalling rather than freshness."
        ),
    )
    parser.add_argument(
        "--can-mop-timeout-us", type=int, default=200,
        help=(
            "Microseconds each mop-up recv_all() waits. Must be small: it is paid in full on "
            "every round that finds the buffer already empty, which is the normal case once the "
            "backlog is drained. Worst-case read cost is roughly "
            "2 * (--can-first-timeout-us + (--can-recv-rounds - 1) * this)."
        ),
    )
    parser.add_argument(
        "--action-smoothing-alpha", type=float, default=1.0,
        help=(
            "EMA smoothing factor for the policy's raw 16D joint-target output: "
            "filtered = alpha*raw + (1-alpha)*filtered_prev. 1.0 = no smoothing (default -- there "
            "is no known noise-amplification mechanism for this checkpoint the way the EE-delta "
            "version had an IK-redundancy issue, so start by trusting the raw output). Lower this "
            "only if the diagnostic plot after a run shows the raw target oscillating."
        ),
    )
    parser.add_argument(
        "--debug-actions", action="store_true",
        help=(
            "Print every joint command handed to robot.send_action(), one line per interpolation "
            "substep (so --inference-hz * the 10 substeps lines per second -- verbose by design). "
            "Grippers are shown in the raw motor units they actually receive, with the "
            "sim-convention value they were converted from in parentheses. Use this to confirm "
            "commands are being transmitted at all, and at what values, when the arm doesn't move."
        ),
    )
    parser.add_argument(
        "--max-episode-seconds", type=float, default=10.0,
        help=(
            "Wall-clock time limit per episode in seconds (default 10) -- an episode that hasn't "
            "hit the success condition or been manually stopped by then is cut short as a timeout."
        ),
    )
    parser.add_argument(
        "--no-start-pose",
        action="store_true",
        help="Skip the move to Isaac Sim's reset pose before each episode and start the rollout"
        " from wherever the arm happens to be. Off by default: every demo the policy learned from"
        " began at that pose, so starting somewhere else puts the policy out of distribution on"
        " step 0. Only useful when the arm is already positioned deliberately.",
    )
    parser.add_argument(
        "--start-pose-speed",
        type=float,
        default=0.3,
        help="rad/s ceiling for the move to the start pose. Deliberately far below"
        " --max-joint-speed: that one bounds a policy step, this one covers a large unattended"
        " repositioning move. The ramp duration is derived from this and the furthest joint.",
    )
    parser.add_argument(
        "--max-start-pose-delta",
        type=float,
        default=1.8,
        help="rad; refuse to run if any joint would have to travel further than this to reach the"
        " start pose. Guards the case where the calibration or zeroing is wrong rather than the"
        " arm merely being out of position -- see approach_pose() in sim_bridge_common.py.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the typed YES confirmation before the FIRST start-pose move (later episodes"
        " never prompt). For unattended runs only -- --max-start-pose-delta still applies.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    calib = load_calibration(args.calibration)
    grip = _gripper_calib(calib)   # both arms' grippers -- see _gripper_calib()

    model = SmolVLAPolicy.from_pretrained(args.checkpoint)

    # OPEN-LOOP HORIZON. SmolVLAPolicy.select_action refills its action queue only when the queue
    # is EMPTY (see _check_get_actions_condition), then executes n_action_steps actions from that
    # one forward pass before looking at another observation. These checkpoints ship
    # chunk_size=50, n_action_steps=50, so the policy sees the world once every 50 control steps
    # and is blind in between -- 2.5s at --inference-hz 20, 6.7s at 7.5. Anything that happens
    # inside a chunk (the object moving, a grasp slipping, the operator intervening) is invisible
    # until the chunk runs out, and two runs from near-identical starts diverge permanently on
    # whatever single frame each chunk boundary happened to land on.
    #
    # This is a DEPLOYMENT choice, not a property of the trained weights: the chunk is still
    # predicted 50 steps long, n_action_steps only decides how many of them get executed before
    # re-planning. Lowering it costs one forward pass more often -- measured 90ms on this machine
    # (2026-08-21, warm; the ~490ms seen on step 0 is cold-start, not the steady-state cost), so
    # against a 50ms budget at 20 Hz a refill overruns by ~40ms. Amortised, n_action_steps=10 adds
    # ~9ms per step and n_action_steps=5 ~18ms, both of which fit; n_action_steps=1 does not.
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= model.config.chunk_size:
            print(f"[ERROR] --n-action-steps must be between 1 and the checkpoint's chunk_size "
                  f"({model.config.chunk_size}); got {args.n_action_steps}.")
            return
        print(f"[INFO] n_action_steps: {model.config.n_action_steps} (checkpoint) -> "
              f"{args.n_action_steps} (--n-action-steps)")
        model.config.n_action_steps = args.n_action_steps
    model.reset()   # rebuild the action queue at the new maxlen, and start from a known-empty one

    horizon_s = model.config.n_action_steps / args.inference_hz
    print(f"[INFO] open-loop horizon: {model.config.n_action_steps} steps per forward pass = "
          f"{horizon_s:.1f}s at --inference-hz {args.inference_hz:.1f}. The policy does NOT look "
          f"at\n       the cameras or joint state again until that many steps have elapsed."
          + ("" if horizon_s <= 1.0 else
             "\n       Pass --n-action-steps to shorten it if you want the policy reacting to "
             "what it currently sees."))

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
    if args.right_wrist_cam_index is not None:
        _usb_reset_for_video_node(args.right_wrist_cam_index)
    if args.front_cam_index is not None:
        _usb_reset_for_video_node(args.front_cam_index)
    time.sleep(1.0)  # let all devices finish re-enumerating before OpenCVCamera opens them

    camera_config = {
        "body_cam":  OpenCVCameraConfig(index_or_path=args.body_cam_index,  width=640, height=480, fps=FPS),
        "wrist_cam": OpenCVCameraConfig(index_or_path=args.wrist_cam_index, width=640, height=480, fps=FPS),
    }
    if args.right_wrist_cam_index is not None:
        # The dual-arm checkpoints' third model input: observation.images.right_wrist_cam, the
        # right-arm counterpart of "wrist_cam" (which is the left wrist -- dataset key naming).
        camera_config["right_wrist_cam"] = OpenCVCameraConfig(index_or_path=args.right_wrist_cam_index, width=640, height=480, fps=FPS)
    if args.front_cam_index is not None:
        # A real model input (unlike --side-cam-index) -- for the older checkpoints trained with a
        # front camera, e.g. ..._1000_joints_three_cams. Unrelated to right_wrist_cam above.
        camera_config["front_cam"] = OpenCVCameraConfig(index_or_path=args.front_cam_index, width=640, height=480, fps=FPS)
    MODEL_CAM_KEYS = list(camera_config.keys())  # for _build_combined_frame's live-view/video labeling below

    # The camera keys and the action dimension both have to match what the checkpoint was trained
    # on, and both are easy to get wrong from the command line (a missing --right-wrist-cam-index
    # leaves the policy blind on an input it expects; a left-arm-only 8D checkpoint driven through
    # this 16D script would command garbage on real hardware). Check both against the checkpoint's
    # own config before the robot is even connected.
    _check_checkpoint_matches(model, preprocess, MODEL_CAM_KEYS)

    # CAN receive tuning -- the single biggest term in the cycle budget. The shipped follower
    # default (8 rounds x 50_000us) costs up to 801ms per read against this loop's 50ms budget at
    # --inference-hz 20, and drifts UPWARD as send_action's unread feedback backlog depletes:
    # measured 110ms -> 210ms -> 310ms cycles within 20s of a rollout, i.e. 9.1 Hz decaying to
    # 3.2 Hz. See _read_motor_positions_once for the full measurement writeup. The defaults below
    # bound the read at ~10ms worst case while draining that backlog every cycle so it can never
    # build up again.
    robot_cfg = OpenArmFollowerConfig(
        right_port="can0",
        left_port="can1",
        enable_fd=True,
        model_path=URDF_PATH,
        cameras=camera_config,  # type: ignore
        recv_rounds=args.can_recv_rounds,
        recv_first_timeout_us=args.can_first_timeout_us,
        recv_mop_timeout_us=args.can_mop_timeout_us,
    )
    robot = OpenArmFollower(robot_cfg)
    robot.connect()
    check_arms_not_crossed(robot, calib)

    # Optional EXTRA camera, purely for the human-facing live view / saved video below -- never
    # part of the model's input (unlike right_wrist_cam/front_cam above, which -- if given -- are
    # real inputs alongside body_cam/wrist_cam via `robot`'s own cameras, see MODEL_CAM_KEYS).
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
    if live_view and not _cv2_has_gui():
        print(
            "[WARN] This cv2 is opencv-python-headless (GUI: NONE -- it is what lerobot itself "
            "depends on), so the live camera window cannot be opened. Continuing without it.\n"
            "       To get the live view back:  uv pip uninstall opencv-python-headless && "
            "uv pip install opencv-python\n"
            "       Or record the same combined view instead:  --save-video rollout.mp4  "
            "(cv2.VideoWriter works fine in the headless build)."
        )
        live_view = False
    video_writer = None

    # Attention dumps. Written to disk rather than shown live: the maps only change when the
    # policy re-plans (every n_action_steps), so a live window spends most of its time redrawing a
    # cached image, and the question these answer -- is the policy actually using its cameras --
    # is one you compare across runs, which needs files. No GUI is required.
    attn_dump = None
    if args.attention_dump:
        attn_dir = args.attention_dir or os.path.join(
            os.getcwd(), f"attention_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        attn_dump = AttentionDump(model, attn_dir, key_labels=_image_key_labels(model, preprocess))
        if attn_dump.enabled:
            print(f"[INFO] attention dumps -> {attn_dir}\n"
                  f"       One PNG per re-plan (every {model.config.n_action_steps} steps): action-"
                  f"expert cross-attention on top, VLM prefix self-attention below,\n"
                  f"       plus attention_log.csv with per-modality shares and spatial entropy.")
        else:
            print(f"[WARN] attention dumps unavailable: {attn_dump.error}")
            attn_dump = None

    # Keep only the camera image entries from the generic hw feature set -- state and action are
    # both built manually below as the same 16 dual-arm joint names in ACTION_NAMES' left-block-
    # then-right-block order, matching exactly what convert_hdf5_to_lerobot.py wrote into the
    # training dataset. Because both share the identical LJ1.pos..LJ8.pos/RJ1.pos..RJ8.pos names,
    # build_inference_frame's state lookup (which requires exact real-observation-dict key matches)
    # and make_robot_action's action labeling (which is purely positional) both work correctly with
    # no separate naming scheme needed for the two.
    full_obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    image_obs_features = {k: v for k, v in full_obs_features.items() if v["dtype"] in ("image", "video")}
    state_features = {
        "observation.state": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ACTION_NAMES},
    }
    action_features = {
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ACTION_NAMES},
    }
    dataset_features = {**action_features, **state_features, **image_obs_features}

    model_dt = 1.0 / args.inference_hz
    interp_steps = 10

    # --max-joint-speed is a SPEED, but the clamp below can only express it as a distance per
    # inference cycle -- so it needs the cycle's real period, not the requested one. The two are
    # not the same thing: one cycle costs a full 16-joint CAN read plus the policy forward, and on
    # this hardware that alone can exceed 1/--inference-hz, in which case the loop free-runs
    # slower than asked. Clamping against the NOMINAL period then scales the arm down by exactly
    # the ratio the loop missed, silently (measured 2026-08-21: 20 Hz requested, ~6.5 Hz achieved,
    # so --max-joint-speed 1.5 behaved as ~0.49 rad/s).
    #
    # cycle_dt is the SHORTEST period seen in the last CYCLE_DT_WINDOW cycles, not the mean and
    # not an EMA: --max-joint-speed is a maximum, so the distance it authorises has to be safe on
    # this loop's FASTEST cycle. Against an average, the cycles that come in under it exceed the
    # stated limit (an 86ms mean over periods alternating 70/120ms authorises 1.75 rad/s on the
    # 70ms ones for a --max-joint-speed of 1.5). The window also caps how long one stalled cycle
    # can influence the clamp, and MAX_CYCLE_STRETCH bounds it outright.
    MAX_CYCLE_STRETCH = 2.0
    CYCLE_DT_WINDOW = 20
    cycle_dt = model_dt          # seeded at nominal until a window of real periods exists
    cycle_periods: list[float] = []
    overruns = 0
    clamp_steps = 0          # steps where --max-joint-speed actually bound at least one joint
    peak_demand_rad = 0.0    # largest single-joint distance the clamp was asked to allow

    def _rate_summary() -> None:
        """Report the rate the loop actually held vs the one asked for, then reset the counters.

        Printed per episode AND on Ctrl+C, because a mismatch here invalidates more than the
        header of the log lines: the policy was trained at a fixed rate, so a loop running at a
        third of it feeds the model a world moving three times slower than any of its demos."""
        nonlocal cycle_periods, overruns, clamp_steps, peak_demand_rad
        if not cycle_periods:
            return
        srt = sorted(cycle_periods)
        med = srt[len(srt) // 2]
        mean = sum(cycle_periods) / len(cycle_periods)
        print(
            f"  rate: requested {args.inference_hz:.1f} Hz, achieved "
            f"{len(cycle_periods) / sum(cycle_periods):.1f} Hz mean / {1.0 / med:.1f} Hz median "
            f"(cycle {mean * 1000:.1f}ms mean, {med * 1000:.1f}ms median, {srt[-1] * 1000:.1f}ms "
            f"worst); {overruns}/{len(cycle_periods)} cycles over budget."
        )
        if args.save_video and abs(1.0 / med - args.inference_hz) > 0.1 * args.inference_hz:
            print(
                f"  note: {args.save_video} carries a {args.inference_hz:.1f} fps header but its "
                f"frames were captured at ~{1.0 / med:.1f} fps, so it plays back "
                f"{args.inference_hz * med:.1f}x fast. Retime without re-encoding:\n"
                f"    ffmpeg -r {1.0 / med:.2f} -i {args.save_video} -c copy retimed.mp4"
            )
        n = len(cycle_periods)
        print(f"  clamp: --max-joint-speed {args.max_joint_speed:.2f} rad/s bound at least one "
              f"joint on {clamp_steps}/{n} steps ({100.0*clamp_steps/max(1, n):.0f}%); largest "
              f"single-joint demand was {peak_demand_rad:.4f} rad "
              f"({peak_demand_rad/max(med, 1e-9):.2f} rad/s at the median cycle).")
        if clamp_steps == 0:
            print("         It never bound -- the policy never asked to move faster than this, so "
                  "raising\n         --max-joint-speed would change nothing about how the arm "
                  "moves.")
        cycle_periods = []
        overruns = 0
        clamp_steps = 0
        peak_demand_rad = 0.0

    # For the target-vs-actual tracking plot: dense target curve (once per control substep) vs.
    # sparser actual-measured curve (once per inference step) for the same 16 joints.
    PLOT_JOINTS = ACTION_NAMES
    target_time_log, target_log = [], {k: [] for k in PLOT_JOINTS}
    actual_time_log, actual_log = [], {k: [] for k in PLOT_JOINTS}

    # Diagnostic only: the policy's RAW output (before the EMA filter) -- lets us tell whether a
    # drift/oscillation seen in the target-vs-actual plot originates in the policy itself or gets
    # introduced by the smoothing/clamp chain downstream of it.
    raw_time_log, raw_log = [], {k: [] for k in ACTION_NAMES}
    # ------------------------------------------------------------------ startup timing probe
    # Answers "can this machine actually hold --inference-hz?" BEFORE the arm is asked to move,
    # by measuring the two terms that dominate a cycle and that no amount of reasoning about the
    # code can predict. Both are inside the loop's `model:` figure, so a slow one is otherwise
    # only visible after the fact, mixed together.
    #
    # The policy forward is deliberately NOT probed: select_action() would populate its action
    # chunk (n_action_steps=50 for these checkpoints) from a pre-episode observation, and the
    # first 50 steps of episode 0 would then replay actions computed for a scene that no longer
    # exists. Its cost shows up in the step lines instead -- expect one large spike every 50
    # steps, when the chunk is refilled, and a ~490ms first pass.
    #
    # Cameras are reported as a DELIVERY RATE per camera rather than as a cost, because that is
    # what bounds the loop: async_read() blocks until a new frame arrives, so the cycle can never
    # outrun the slowest camera no matter how much budget is left over. A first version of this
    # probe timed three async_read() calls inside a free-running loop and reported 27.7ms -- but
    # that loop was polling faster than 30 fps, so the figure was just one frame period minus the
    # CAN time (5.7 + 27.7 = 33.4ms = 30 fps). It would have read as "cameras are too slow" on
    # cameras that were performing exactly to spec.
    if args.probe_timing:
        n = 20
        can_ms = []
        for _ in range(n):
            t0 = time.perf_counter()
            robot._read_motor_positions_stable()   # the CAN half of get_observation(), alone
            can_ms.append((time.perf_counter() - t0) * 1e3)

        # Per-camera DELIVERY RATE, which is the number that actually bounds the loop -- not the
        # cost of a read at some arbitrary polling rate. async_read() blocks until a NEW frame
        # arrives (it clears the frame event on every call), so back-to-back reads are always
        # faster than the camera and each call's duration IS one frame period. Measuring the
        # aggregate cost of three reads in a free-running probe instead measures the probe's own
        # rate against the frame rate -- it returns ~one frame period no matter how fast the
        # cameras are, which is exactly the false alarm this replaced.
        cam_hz = {}
        for name, cam in robot.cameras.items():
            try:
                cam.async_read()          # discard one, so the first timed call starts aligned
                gaps = []
                for _ in range(n):
                    t0 = time.perf_counter()
                    cam.async_read()
                    gaps.append(time.perf_counter() - t0)
                gaps.sort()
                cam_hz[name] = 1.0 / gaps[len(gaps) // 2]
            except Exception as e:
                print(f"[WARN] timing probe could not read {name}: {e}")

        can_med = sorted(can_ms)[len(can_ms) // 2]
        can_worst = max(can_ms)
        print(f"\n[timing probe] {n} samples, arm still, policy not yet running:")
        print(f"  CAN read ({args.can_recv_rounds} rounds, {args.can_first_timeout_us}us first, "
              f"{args.can_mop_timeout_us}us mop): {can_med:.1f}ms median, {can_worst:.1f}ms worst")
        for name, hz in cam_hz.items():
            note = "" if hz >= args.inference_hz else "   <-- SLOWER THAN THE REQUESTED RATE"
            print(f"  {name:<16} delivering {hz:5.1f} fps ({1000/hz:5.1f}ms/frame){note}")

        # The loop reads every camera once per cycle and blocks until each has a NEW frame, so it
        # can never run faster than the slowest camera, whatever the rest of the budget allows.
        slowest_hz = min(cam_hz.values()) if cam_hz else float("inf")
        cam_wait_ms = max(0.0, 1000.0 / slowest_hz - model_dt * 1000.0)
        floor_ms = can_med + cam_wait_ms
        print(f"  floor per cycle, before the policy: {floor_ms:.1f}ms "
              f"(CAN {can_med:.1f}ms + camera wait {cam_wait_ms:.1f}ms) -- budget is "
              f"{model_dt*1000:.1f}ms at --inference-hz {args.inference_hz:.1f}")
        if slowest_hz < args.inference_hz:
            print(f"  VERDICT: --inference-hz {args.inference_hz:.1f} is NOT reachable -- the "
                  f"slowest camera delivers {slowest_hz:.1f} fps, and the loop waits for a new "
                  f"frame from\n           every camera each cycle. Ceiling is {slowest_hz:.1f} Hz "
                  f"until that camera is fixed (auto-exposure lengthening exposure time in low "
                  f"light is the\n           usual cause -- check with: v4l2-ctl -d /dev/video"
                  f"{args.body_cam_index} --get-ctrl=auto_exposure,exposure_time_absolute).")
        elif floor_ms >= model_dt * 1000:
            print(f"  VERDICT: --inference-hz {args.inference_hz:.1f} is NOT reachable -- the floor "
                  f"alone exceeds the budget before the policy has run.\n"
                  f"           The most this loop could hold is {1000/floor_ms:.1f} Hz.")
        else:
            head = model_dt * 1000 - floor_ms
            print(f"  VERDICT: floor fits, leaving {head:.1f}ms of the budget for the policy and "
                  f"the interpolation sends.\n           Cameras are not the limit "
                  f"({slowest_hz:.1f} fps >= {args.inference_hz:.1f} Hz requested); whether 20 Hz "
                  f"holds now depends on the policy's\n           per-step cost, which the step "
                  f"lines below will show.")

    start_time = time.perf_counter()

    try:
        for ep in range(MAX_EPISODES):
            print(f"Starting episode {ep}...")

            # Put the arm at Isaac Sim's reset pose before the policy sees anything. Every demo in
            # the training set begins there (OPENARM_BI_CFG init_state -- arm joints 0 except
            # joint4 at pi/2, grippers open), so starting a rollout from wherever the previous
            # episode happened to end leaves the policy out of distribution on its very first
            # observation. Done per episode, not once per session, so episode 2 starts from the
            # same place episode 1 did. Only the first one prompts: re-confirming an identical,
            # already-approved move before every episode is the kind of gate people learn to
            # answer without reading.
            if not args.no_start_pose:
                if approach_pose(
                    robot, sim_init_pose_action(calib),
                    label="Isaac Sim's reset pose",
                    arm_speed=args.start_pose_speed,
                    max_delta=args.max_start_pose_delta,
                    assume_yes=args.yes or ep > 0,
                ) is None:
                    print("Start-pose approach refused or not confirmed -- not running the policy.")
                    break

            # Discard any actions left in the policy's queue from the previous episode. Without
            # this, episode N+1 opens by replaying up to n_action_steps actions that were planned
            # from episode N's final observation -- of a scene that no longer exists, from a pose
            # the arm has since been ramped away from. Across separate PROCESSES the queue starts
            # empty anyway (from_pretrained calls reset()), so this is about episode-to-episode
            # carryover within one run, which is where it actually bites.
            model.reset()

            first = True
            episode_start = time.perf_counter()
            cycle_start = episode_start   # deadline anchor, re-stamped at the top of every cycle
            rate_warned = False
            step = 0
            filtered_target = np.zeros(ACTION_DIM)
            # Latched binarization decision per gripper (see _binarize_gripper). Starts open so a
            # first prediction landing in the dead band can never clamp a gripper shut on nothing;
            # it is re-seeded from the actual measured gripper on the first step below.
            gripper_cmd = {i: GRIPPER_OPEN_CMD for i in GRIPPER_IDX}

            while time.perf_counter() - episode_start < args.max_episode_seconds:
                cycle_start = time.perf_counter()
                model_start = cycle_start

                # _get_obs_sim_gripper (not robot.get_observation() directly) maps LJ8.pos through
                # calibration -- see GRIPPER CALIBRATION note above -- and clips it to
                # [GRIPPER_MIN, GRIPPER_MAX] as part of that conversion, which also covers the
                # transient bad CAN-bus reads seen before (gripper briefly reading ~-1.0 rad, ~20x
                # past its real range, in the first ~0.7s right after connect): without clipping,
                # the interpolation loop below would blend its start point (prev_action = obs) from
                # that bad reading even though the *target* end point was already clamped.
                obs = _get_obs_sim_gripper(robot, grip)

                # This read doubles as the "where did the arm actually end up" sample for the
                # PREVIOUS cycle's interpolated move: it lands at the same instant the separate
                # settle-read at the bottom of the loop used to, and costs nothing extra. That
                # second read was a second full _read_motor_positions_stable() -- roughly a third
                # of the cycle budget -- spent only to feed the tracking plot.
                actual_time_log.append(time.perf_counter() - start_time)
                for k in PLOT_JOINTS:
                    actual_log[k].append(obs[k])

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
                    # Record the state this episode actually starts from. approach_pose ramps to
                    # Isaac Sim's reset pose but settles with real steady-state error (0.2768 rad
                    # on RJ1 was observed on 2026-08-21, ~16 deg), and that error depends on where
                    # the arm was ramped FROM -- i.e. on how the previous run ended. That makes it
                    # a genuine run-to-run coupling: same command, same scene, measurably
                    # different initial condition. Printing the full vector makes two runs
                    # diffable when they behave differently.
                    print("Episode start state (what the policy's first observation actually was):")
                    print("  " + "  ".join(f"{n}={obs[n]:+.4f}" for n in ACTION_NAMES[:8]))
                    print("  " + "  ".join(f"{n}={obs[n]:+.4f}" for n in ACTION_NAMES[8:]))

                    filtered_target = np.array([obs[k] for k in ACTION_NAMES], dtype=np.float64)
                    # Seed the latch from where the grippers physically are, so the dead band holds
                    # the real current state rather than an assumed one.
                    for i in GRIPPER_IDX:
                        gripper_cmd[i] = _binarize_gripper(filtered_target[i], gripper_cmd[i])
                    continue

                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task=TASK,
                    robot_type=ROBOT_TYPE,
                )
                obs_processed = preprocess(obs_frame)

                # An empty action queue means select_action is about to run a real forward pass
                # rather than pop a cached action -- which is exactly when fresh attention gets
                # captured. Checked BEFORE the call, since the queue is refilled by it.
                replanned = sum(len(q) for q in model._queues.values()) == 0

                raw_action = model.select_action(obs_processed)
                raw_action = postprocess(raw_action)
                policy_action = make_robot_action(raw_action, dataset_features)  # {LJ1.pos: v, ...}

                # Dump only on a re-plan: on every other step the captured maps are the same
                # tensors as the last dump, so writing them again would just multiply identical
                # PNGs and dilute the CSV.
                if attn_dump is not None and replanned:
                    if not attn_dump.enabled:
                        print(f"[WARN] attention dumps stopped: {attn_dump.error}")
                        attn_dump = None
                    else:
                        attn_dump.dump(step=step, t_s=time.perf_counter() - episode_start)

                raw_target = np.array([policy_action[n] for n in ACTION_NAMES], dtype=np.float64)

                raw_time_log.append(time.perf_counter() - start_time)
                for name in ACTION_NAMES:
                    raw_log[name].append(policy_action[name])

                # Optional EMA smoothing (see --action-smoothing-alpha help; off by default).
                filtered_target = (
                    args.action_smoothing_alpha * raw_target
                    + (1.0 - args.action_smoothing_alpha) * filtered_target
                )

                # Binarize both grippers AFTER the EMA filter, not before: filtering a binary
                # decision would smear it straight back into the intermediate values binarization
                # exists to avoid. The speed clamp below still limits how fast the gripper travels
                # to its end point (~2 steps at 0.3 rad/s, 10 Hz over the 0.044 range).
                for i in GRIPPER_IDX:
                    gripper_cmd[i] = _binarize_gripper(filtered_target[i], gripper_cmd[i])
                    filtered_target[i] = gripper_cmd[i]

                current_q = np.array([obs[k] for k in ACTION_NAMES], dtype=np.float64)

                # Speed limit: clamp how far the target may move from the CURRENT measured joints
                # this inference step, to --max-joint-speed * model_dt radians -- see module
                # docstring for why this (not the interpolation loop alone) is what actually
                # bounds jerkiness/speed.
                clamp_dt = cycle_dt if args.speed_clamp_basis == "measured" else model_dt
                max_step_rad = args.max_joint_speed * clamp_dt
                raw_delta = filtered_target - current_q
                # Does --max-joint-speed actually bind? Raising it only changes the arm's
                # behaviour on steps where it does; on every other step it is inert. Worth
                # measuring rather than assuming, because the demand here is not the policy's
                # intended step size -- the clamp is applied against the CURRENT MEASURED joints,
                # so tracking error (approach_pose alone settles ~0.28 rad out) shows up as
                # demand the clamp then rate-limits the arm's catch-up to.
                joints_clamped = int(np.count_nonzero(np.abs(raw_delta) > max_step_rad + 1e-12))
                if joints_clamped:
                    clamp_steps += 1
                peak_demand_rad = max(peak_demand_rad, float(np.max(np.abs(raw_delta))))
                delta_q = np.clip(raw_delta, -max_step_rad, max_step_rad)
                target_q = current_q + delta_q
                for i in GRIPPER_IDX:  # gripper safety clip (both arms)
                    target_q[i] = float(np.clip(target_q[i], GRIPPER_MIN, GRIPPER_MAX))

                # Full 16-joint target: start from the current observation, then overwrite every
                # joint with the model's target -- with the dual-arm action space, ACTION_NAMES
                # already covers all 16 joints, so nothing is held constant from the observation
                # any more (the dict copy just keeps any non-".pos" entries intact).
                target_action = dict(obs)
                for name, value in zip(ACTION_NAMES, target_q):
                    target_action[name] = float(value)

                model_time = time.perf_counter() - model_start

                # Interpolate from the CURRENT measured observation (not the previous action)
                # toward the new target -- self-correcting against any tracking error.
                #
                # PACING: the substeps are spread across whatever is LEFT of this cycle's budget
                # (cycle_start + model_dt), not across a fresh model_dt of their own. Sleeping a
                # full model_dt here, after model_time had already been spent, made the cycle
                # period model_time + model_dt -- so the loop could not reach --inference-hz even
                # in principle: a free policy would have hit it exactly, and a real one misses by
                # its entire inference time. When nothing is left of the budget every substep is
                # still sent, back to back with no sleep (the arm is a position controller, so a
                # compressed ramp converges on the same end point), and the cycle is counted as an
                # overrun below rather than silently stretching the period.
                interp_start = time.perf_counter()
                substep_dt = max(0.0, (cycle_start + model_dt) - interp_start) / interp_steps
                prev_action = obs
                for i in range(interp_steps):
                    alpha = (i + 1) / interp_steps
                    interp_action = {
                        joint: prev_action[joint] + (target_action[joint] - prev_action[joint]) * alpha
                        for joint in target_action.keys()
                        if joint.endswith(".pos")
                    }
                    _send_action_sim_gripper(
                        robot, interp_action, grip,
                        debug_tag=f"step {step} substep {i + 1}/{interp_steps}" if args.debug_actions else None,
                    )

                    t_now = time.perf_counter() - start_time
                    target_time_log.append(t_now)
                    for k in PLOT_JOINTS:
                        target_log[k].append(interp_action[k])

                    sleep_time = (interp_start + substep_dt * (i + 1)) - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                interp_time = time.perf_counter() - interp_start
                cycle_time = time.perf_counter() - cycle_start
                cycle_periods.append(cycle_time)
                cycle_dt = min(
                    min(cycle_periods[-CYCLE_DT_WINDOW:]), MAX_CYCLE_STRETCH * model_dt
                )
                over = cycle_time - model_dt
                if over > 1e-3:
                    overruns += 1
                elapsed = time.perf_counter() - episode_start
                print(
                    f"[step {step} @ {elapsed:.1f}s/{args.max_episode_seconds:.1f}s] "
                    f"cycle: {cycle_time*1000:.1f}ms = {1.0/cycle_time:.1f} Hz"
                    + (f" (OVER budget by {over*1000:.1f}ms)" if over > 1e-3 else "")
                    + f"  model: {model_time*1000:.1f}ms  "
                    f"interp: {interp_time*1000:.1f}ms ({interp_steps} x "
                    f"{(interp_time/interp_steps)*1000:.1f}ms)  "
                    f"clamp: {max_step_rad*1000:.1f}mrad/cycle"
                    + (f" ({joints_clamped} joints hit)" if joints_clamped else "")
                )

                # One-shot, once there is enough evidence that this is not just warm-up (the first
                # forward pass alone ran ~490ms): name the rate the loop CAN hold instead of
                # leaving a wrong --inference-hz sitting in the header of every later line.
                if not rate_warned and len(cycle_periods) >= 20:
                    med = sorted(cycle_periods[-20:])[10]
                    if med > model_dt * 1.1:
                        rate_warned = True
                        print(
                            f"[WARN] loop is holding {1.0/med:.1f} Hz, not the requested "
                            f"{args.inference_hz:.1f} Hz (median cycle {med*1000:.1f}ms vs a "
                            f"{model_dt*1000:.1f}ms budget). The policy is seeing a world moving "
                            f"{model_dt/med:.2f}x slower than its training demos. Pass "
                            f"--inference-hz {1.0/med:.0f} to make the request match reality."
                        )
                step += 1

            print(f"Episode {ep} ended after {time.perf_counter() - episode_start:.1f}s ({step} steps).")
            _rate_summary()

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected, stopping loop...")
        _rate_summary()

    finally:
        # Before anything else in teardown: the whole point of the dumps is the aggregate, and a
        # run ended with Ctrl+C is the normal case, not the exception.
        if attn_dump is not None:
            csv_path = attn_dump.write_csv()
            print("\nAttention summary:")
            for line in attn_dump.summary_lines():
                print(line)
            if csv_path:
                print(f"  csv: {csv_path}")
            attn_dump.detach()

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
            fig, axes = plt.subplots(8, 2, figsize=(14, 24), sharex=True)
            for ax, joint in zip(axes.flat, PLOT_JOINTS):
                ax.plot(target_time_log, target_log[joint], label="target", linewidth=1)
                ax.plot(actual_time_log, actual_log[joint], label="actual", marker="o", markersize=2, linewidth=1)
                ax.set_title(joint)
                ax.set_ylabel("rad")
                ax.grid(True)
            axes.flat[0].legend(loc="upper right")
            for ax in axes[-1, :]:
                ax.set_xlabel("Time (s)")
            fig.suptitle("Both arms + grippers: commanded target vs. measured actual (joint-space model)")
            fig.tight_layout()

        print("[INFO] Plotting raw policy output (before EMA filter)...")
        if raw_time_log:
            fig2, axes2 = plt.subplots(8, 2, figsize=(14, 24), sharex=True)
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
