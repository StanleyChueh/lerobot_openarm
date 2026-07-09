"""
Autonomous SmolVLA rollout on the real OpenArm follower -- pure policy test, no recording,
no teleop leader/correction. Based on deploy_gr00t_stanley.py's clean rollout loop (that one
uses GrootPolicy; this swaps in SmolVLAPolicy the same way deploy_smolvla_stanley.py does) with
deploy_smolvla_stanley.py's and record-loop/leader-arm machinery removed, since you just want to
watch the trained policy run, not capture a new corrected dataset.

SAFETY: this moves the real robot autonomously with NO human-in-the-loop correction step.
  - Start with MAX_EPISODES = 1 and watch closely before trusting longer unsupervised runs.
  - Keep emergency_disable.py within reach.
  - Ctrl+C stops the loop and disconnects the robot; it does not power off the motors.

Usage:
  python deploy_smolvla_pickup.py \
      --checkpoint ethanCSL/openarm_visuomotor_no_domain_randomization_1000 \
      --body-cam-index 4 --wrist-cam-index 10 \
      --inference-hz 30 --max-joint-speed 1.0 --max-episode-seconds 10 \
      --action-smoothing-alpha 0.3 --gripper-hysteresis 0.3 --ik-damping 0.05

If motion looks jerky, tune --max-joint-speed down first (it caps the commanded joint speed
directly); drop --inference-hz too if it's still jerky after that (gives the arm more time to
physically reach each target before the next one is issued). If the gripper flaps open/closed
rapidly, raise --gripper-hysteresis.

If only one or two specific joints oscillate while the rest stay smooth, check the raw-policy-
output diagnostic plot shown after each run (Figure 2): if THAT is already smooth for all 6 axes,
the noise is coming from the IK solver, not the policy -- raise --ik-damping (see its help text
for why OpenArm's 1 redundant DOF can make specific joints sensitive to ordinary EE-delta noise).
Only reach for --action-smoothing-alpha if the raw-output plot itself is noisy -- filtering an
already-smooth input mostly just adds lag without fixing anything.

A target-vs-actual joint tracking plot (Figure 1) is also shown after each run to check how well
the arm follows the commanded trajectory -- large or growing gaps between the two curves indicate
the arm is not keeping up, which usually means --max-joint-speed and/or --inference-hz need to
come down further.

Episode length is a wall-clock time limit (--max-episode-seconds), not a step count -- a fixed
step count wouldn't mean the same duration once --inference-hz is adjustable (e.g. an old fixed
"300 steps at 30Hz" was only ever an approximation of ~10s in the first place, since it doesn't
account for the model's own inference latency each step).

Cameras are opened via OpenCVCameraConfig (plain V4L2 /dev/videoN index), not the RealSense SDK
-- the D435i's color sensor exposes itself as a standard UVC device, and this task only ever
uses the RGB stream (no depth/IMU), so the simpler V4L2 path is enough. Find your indices with
`ls /dev/video*` or `v4l2-ctl --list-devices` on the machine wired to the robot's cameras -- I
have no way to know these from here, and they're passed as CLI arguments rather than hardcoded
so you can assign them per run instead of editing the script.

Note: unlike a RealSense serial number, a /dev/videoN index can shift if cameras are unplugged/
replugged in a different order or the machine reboots -- re-check with the commands above if a
previously-working index suddenly points at the wrong camera (or nothing).

Normalization stats are loaded from the checkpoint's own saved preprocessor/postprocessor
safetensors (uploaded alongside the model by `lerobot-train`'s push) via
`make_pre_post_processors(model.config, checkpoint)` -- no separate dataset repo lookup needed,
so there's no --dataset argument here (an earlier version of this script assumed the dataset repo
shared the checkpoint's name and needed one; that assumption doesn't hold for every checkpoint
and the dataset lookup wasn't even necessary in the first place).

Camera dict keys below are "body_cam"/"wrist_cam" -- the ORIGINAL semantic names used when the
HDF5->LeRobot dataset was built (see scripts/tools/convert_hdf5_to_lerobot.py), not "camera1"/
"camera2". The rename to camera1/camera2 was a --rename_map preprocessing step applied at
`lerobot-train` time and gets loaded back in via `make_pre_post_processors(model.config,
pretrained_model_path, ...)` below -- this script does not need to replicate that mapping itself.

Action space: the checkpoint was trained on EE-space pose deltas for the left arm (see
convert_hdf5_to_lerobot.py: `actions = ep["actions"][:, :7]` = 6D delta pose + 1D gripper),
matching the DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True,
ik_method="dls") used in sim -- NOT joint targets. OpenArmFollower only accepts joint-space
commands and has no IK of its own, so ik_bridge.LeftArmDifferentialIK (same URDF, same DLS math
and lambda as the sim controller) converts the policy's raw 6D delta into target left-arm joint
angles every step -- see ik_bridge.py's module docstring for the full derivation. The right arm
and both grippers were never part of this action space -- the right arm is held at its current
pose and the left gripper is thresholded open/closed from action[6], matching the convention
already used by record_demos_openarm.py's JointMirrorBroadcaster (open=0.044, closed=0.0 rad).

Observation state fed to the model is the left arm's 7 joints only (LJ1.pos..LJ7.pos), matching
what the checkpoint was trained on -- NOT all 16 joints. build_dataset_frame looks state values
up by exact key name against the real observation dict, so this state feature's "names" must be
real OpenArmFollower observation keys (unlike the action feature's names below, which are just
labels this script picks for its own raw model output and never get looked up against anything).

Both D435i units intermittently fail to start their color stream on connect -- reproduced with
plain `v4l2-ctl --stream-mmap` and bare `cv2.VideoCapture`, with no lerobot/robot code involved,
so this is a device/firmware-level hang (rs-enumerate-devices reports firmware 5.15.1.55 against
Intel's recommended 5.17.0.10 for these units), not a bug in this script or in lerobot. Confirmed
fix each time it happens: a USBDEVFS_RESET ioctl on the camera's USB device (no unplug needed,
no sudo needed -- /dev/bus/usb/*/* is world-writable on this machine). Since the hang recurs
unpredictably between runs, this script resets both cameras' USB devices unconditionally right
before connecting rather than only reactively -- see _usb_reset_for_video_node() below. The real
fix is a firmware update for these two D435i units; this reset is a workaround for testing now.
"""

import argparse
import fcntl
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig

from ik_bridge import LeftArmDifferentialIK

MAX_EPISODES = 1              # start at 1 for a first real-hardware test; raise once trusted
FPS = 30
TASK = "Pick up the cube."
ROBOT_TYPE = "openarm_follower"
URDF_PATH = "/home/csl/Stanley_ws/lerobot_openarm/model/openarm_description.urdf"

LEFT_ARM_STATE_KEYS = [f"LJ{i}.pos" for i in range(1, 8)]
ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]
GRIPPER_OPEN_VAL = 0.044   # matches record_demos_openarm.py's JointMirrorBroadcaster convention
GRIPPER_CLOSED_VAL = 0.0

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint", required=True,
        help=(
            "HF Hub repo id (or local pretrained_model dir) of the trained SmolVLA checkpoint to "
            "deploy, e.g. ethanCSL/openarm_visuomotor_no_domain_randomization_1000."
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
        "--inference-hz", type=float, default=30.0,
        help=(
            "Model inference / control-loop rate in Hz (default 30, matching the training data's "
            "recording rate). Lowering this gives the arm more time to physically reach each "
            "target before the next one is issued, which can reduce jerkiness at the cost of "
            "reactivity -- try e.g. 10-15 if motion still looks jerky after tuning --max-joint-speed."
        ),
    )
    parser.add_argument(
        "--max-joint-speed", type=float, default=1.0,
        help=(
            "Maximum left-arm joint speed in rad/s, enforced by clamping how far the IK-derived "
            "target may move from the current measured joint angles each inference step (default "
            "1.0 rad/s ~= 57 deg/s). This is the main jerkiness control: the policy's raw per-step "
            "delta can otherwise demand large instantaneous joint jumps, which the interpolation "
            "step alone only smooths in time, not in magnitude. Lower = smoother but slower."
        ),
    )
    parser.add_argument(
        "--action-smoothing-alpha", type=float, default=0.3,
        help=(
            "EMA smoothing factor for the policy's raw 6D EE-delta output, applied before it goes "
            "into the IK solver: filtered = alpha*raw + (1-alpha)*filtered_prev. 1.0 = no smoothing "
            "(use the raw per-step output directly); lower = smoother but laggier response. Default "
            "0.3. NOTE: logged evidence (the raw-policy-output diagnostic plot) showed the policy's "
            "own output is already smooth -- if a specific joint oscillates while raw output isn't, "
            "that's --ik-damping's problem to fix, not this one."
        ),
    )
    parser.add_argument(
        "--ik-damping", type=float, default=0.05,
        help=(
            "Damping (lambda) for the IK solver's damped-least-squares pseudo-inverse (Isaac Lab's "
            "own sim controller default is 0.01; this deploys with a higher default since the real "
            "arm doesn't need to match sim's exact value). OpenArm's left arm has 7 joints solving "
            "a 6D EE pose, i.e. 1 redundant DOF -- in poses where two joints have a near-parallel "
            "effect on the commanded EE motion, the minimum-norm IK solution can become highly "
            "sensitive to tiny, ordinary EE-delta noise, amplifying it into visible joint-angle "
            "oscillation on just those joints (confirmed via the raw-policy-output plot: smooth "
            "6D input, jittery output on 1-2 joints only). Raising this trades a little tracking "
            "precision for damping that noise; 0 recovers the undamped pseudo-inverse."
        ),
    )
    parser.add_argument(
        "--gripper-hysteresis", type=float, default=0.3,
        help=(
            "Deadband half-width around zero for the raw gripper command (trained range is "
            "roughly [-1, 1], +1=open/-1=close). The commanded gripper only flips open when the "
            "raw signal exceeds +this value, and only flips closed when it drops below -this "
            "value; in between, it holds its previous state. Prevents rapid open/close flapping "
            "when the raw signal hovers near zero. Default 0.3."
        ),
    )
    parser.add_argument(
        "--max-episode-seconds", type=float, default=10.0,
        help=(
            "Wall-clock time limit per episode in seconds (default 10) -- an episode that hasn't "
            "hit the success condition or been manually stopped by then is cut short as a timeout. "
            "This is wall-clock time, not a step count, so it stays meaningful regardless of "
            "--inference-hz (a fixed step count wouldn't: e.g. 300 steps is ~10s at 30Hz but ~30s "
            "at 10Hz)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmolVLAPolicy.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()

    # Normalization stats are loaded from the checkpoint's own saved preprocessor/postprocessor
    # safetensors (uploaded alongside the model by `lerobot-train`'s push), not from a separately
    # pushed dataset repo -- passing pretrained_path here is what makes make_pre_post_processors
    # load those directly, the same way smolvla_server.py's _load_norm_stats does explicitly for
    # the sim-eval path. No --dataset argument needed.
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        args.checkpoint,
    )

    # See module docstring: both D435i units intermittently fail to start their color stream:
    # reset unconditionally right before connecting rather than only when a failure is detected.
    _usb_reset_for_video_node(args.body_cam_index)
    _usb_reset_for_video_node(args.wrist_cam_index)
    time.sleep(1.0)  # let both devices finish re-enumerating before OpenCVCamera opens them

    camera_config = {
        # Camera capture fps is independent of --inference-hz (the control-loop rate below) --
        # this is just the hardware stream rate, fixed at the value already confirmed to connect
        # reliably on both D435i units. Both color nodes only advertise YUYV
        # (v4l2-ctl --list-formats-ext confirms no MJPG mode on either), so no fourcc override.
        "body_cam":  OpenCVCameraConfig(index_or_path=args.body_cam_index,  width=640, height=480, fps=FPS),
        "wrist_cam": OpenCVCameraConfig(index_or_path=args.wrist_cam_index, width=640, height=480, fps=FPS),
    }

    robot_cfg = OpenArmFollowerConfig(
        right_port="can2",
        left_port="can3",
        enable_fd=True,
        model_path=URDF_PATH,
        cameras=camera_config,  # type: ignore
    )
    robot = OpenArmFollower(robot_cfg)
    robot.connect()

    ik = LeftArmDifferentialIK(urdf_path=URDF_PATH, lambda_val=args.ik_damping)

    # Keep only the camera image entries from the generic (all-16-joint) hw feature set -- state
    # is built manually below to select just the 7 left-arm joints the model was actually trained
    # on. Action names are arbitrary labels (see module docstring); make_robot_action maps the
    # model's raw output tensor to these names purely positionally.
    full_obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    image_obs_features = {k: v for k, v in full_obs_features.items() if v["dtype"] in ("image", "video")}
    state_features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": LEFT_ARM_STATE_KEYS},
    }
    action_features = {
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }
    dataset_features = {**action_features, **state_features, **image_obs_features}

    # Smooth the policy's ~1/inference_hz-rate output into finer robot-control steps.
    model_dt = 1.0 / args.inference_hz
    max_step_rad = args.max_joint_speed * model_dt
    interp_steps = 10
    control_dt = model_dt / interp_steps

    # For the target-vs-actual tracking plot: dense target curve (once per control substep) vs.
    # sparser actual-measured curve (once per inference step) for the same 8 joints.
    PLOT_JOINTS = LEFT_ARM_STATE_KEYS + ["LJ8.pos"]
    target_time_log, target_log = [], {k: [] for k in PLOT_JOINTS}
    actual_time_log, actual_log = [], {k: [] for k in PLOT_JOINTS}

    # Diagnostic only: the policy's RAW output (before the EMA filter, before IK) -- lets us tell
    # whether a drift/oscillation seen in the target-vs-actual plot originates in the policy
    # itself or gets introduced by our own filter/IK/clamp chain downstream of it.
    raw_time_log, raw_log = [], {k: [] for k in ACTION_NAMES}
    start_time = time.perf_counter()

    try:
        for ep in range(MAX_EPISODES):
            print(f"Starting episode {ep}...")
            first = True
            interp_start = time.perf_counter()
            episode_start = time.perf_counter()
            step = 0
            filtered_delta_pose = np.zeros(6)
            gripper_is_open = None  # set from the first real observation below

            while time.perf_counter() - episode_start < args.max_episode_seconds:
                model_start = time.perf_counter()

                obs = robot.get_observation()

                if first:
                    first = False
                    gripper_is_open = obs["LJ8.pos"] > (GRIPPER_OPEN_VAL + GRIPPER_CLOSED_VAL) / 2
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
                policy_action = make_robot_action(raw_action, dataset_features)  # {name: value, ...}

                raw_delta_pose_6d = np.array([policy_action[n] for n in ACTION_NAMES[:6]], dtype=np.float64)
                gripper_cmd = policy_action["gripper"]

                raw_time_log.append(time.perf_counter() - start_time)
                for name in ACTION_NAMES:
                    raw_log[name].append(policy_action[name])

                # EMA-smooth the raw EE-delta before IK: a config where the commanded EE motion
                # is near-redundant for some joint amplifies ordinary per-step policy noise into
                # visible joint oscillation on that joint alone, even though the intended EE
                # trajectory is smooth -- confirmed from the tracking plot (LJ2/LJ4 target
                # oscillating while LJ1/3/5/6/7 stay smooth, same policy output feeding all of
                # them through the same IK call).
                filtered_delta_pose = (
                    args.action_smoothing_alpha * raw_delta_pose_6d
                    + (1.0 - args.action_smoothing_alpha) * filtered_delta_pose
                )
                delta_pose_6d = filtered_delta_pose

                current_left_q = np.array([obs[k] for k in LEFT_ARM_STATE_KEYS], dtype=np.float64)
                target_left_q = ik.compute_target_joint_angles(current_left_q, delta_pose_6d)

                # Speed limit: clamp how far the IK-derived target may move from the CURRENT
                # measured joints this inference step, to --max-joint-speed * model_dt radians.
                # The interpolation loop below only smooths this move *in time* across substeps --
                # without this clamp, a large raw policy delta still commands a large jump by the
                # end of the step, which is what produces jerky motion, not just fast motion.
                delta_left_q = np.clip(target_left_q - current_left_q, -max_step_rad, max_step_rad)
                target_left_q = current_left_q + delta_left_q

                # Hysteresis on the gripper decision: the raw signal hovering near zero otherwise
                # flips the commanded state (and the real motor with it) every other step -- see
                # the LJ8 tracking plot. Only flip once the signal clears a margin past zero;
                # otherwise hold the previous commanded state.
                if gripper_cmd > args.gripper_hysteresis:
                    gripper_is_open = True
                elif gripper_cmd < -args.gripper_hysteresis:
                    gripper_is_open = False
                gripper_target = GRIPPER_OPEN_VAL if gripper_is_open else GRIPPER_CLOSED_VAL

                # Full 16-joint target: start from the current observation (holds the right arm
                # and its gripper constant -- never part of this action space), then overwrite
                # only the left arm + left gripper with the IK-derived target.
                target_action = dict(obs)
                for name, value in zip(LEFT_ARM_STATE_KEYS, target_left_q):
                    target_action[name] = float(value)
                target_action["LJ8.pos"] = gripper_target

                model_time = time.perf_counter() - model_start

                # Interpolate from the CURRENT measured observation (not the previous action)
                # toward the new target -- self-correcting against any tracking error, matching
                # deploy_gr00t_stanley.py's pattern.
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
                    robot.send_action(interp_action)

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

                # Measure where the arm actually ended up after this step's interpolated moves --
                # the tracking plot compares this against the dense target curve above.
                settled_obs = robot.get_observation()
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
            fig.suptitle("Left arm + gripper: commanded target vs. measured actual")
            fig.tight_layout()

        print("[INFO] Plotting raw policy output (before EMA filter / IK)...")
        if raw_time_log:
            fig2, axes2 = plt.subplots(4, 2, figsize=(12, 14), sharex=True)
            for ax, name in zip(axes2.flat, ACTION_NAMES):
                ax.plot(raw_time_log, raw_log[name], linewidth=1)
                ax.axhline(0.0, color="gray", linewidth=0.5)
                ax.set_title(name)
                ax.grid(True)
            for ax in axes2.flat[len(ACTION_NAMES):]:
                ax.axis("off")
            for ax in axes2[-1, :]:
                ax.set_xlabel("Time (s)")
            fig2.suptitle("Raw policy output (pre-filter, pre-IK) -- for comparing against the tracking plot")
            fig2.tight_layout()

        if target_time_log or raw_time_log:
            plt.show()
        else:
            print("[WARN] No data to plot.")


if __name__ == "__main__":
    main()
