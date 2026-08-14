#!/usr/bin/env python
"""replay_hf_sim_episode.py, but with a gripper that actually grips a REAL object.

Same replay as that script -- read an episode from a HF-hosted Isaac Sim dataset, handshake,
ramp, speed-clamp, kill switch, plot -- with one thing changed: how the gripper channel is
driven. Use this to replay a sim pick-up onto a real object (e.g. a real Pringles can); use
the original when you want a faithful reproduction of the sim's finger positions.

WHY THIS EXISTS
---------------
The HF dataset's gripper column is the sim's MEASURED finger joint (convert_hdf5_to_lerobot.py
derives both observation.state and action from states/articulation/robot/joint_position). In sim
the can and the fingers are rigid bodies, so a closing gripper STALLS at the can's surface and
the recording holds that stall value forever. Measured over the 10 episodes of pickup_pringle.hdf5:
while the gripper was commanded fully CLOSED (0.0), the measured finger sat at 0.033 m on average
(min 0.022, max 0.043) out of a 0.044 m fully-open travel.

replay_hf_sim_episode.py maps that value proportionally (gripper_sim_to_raw), so a real grasp
frame commands the real gripper to ~75% OPEN and the real can is never squeezed. The real
gripper's pads are compliant and have to keep closing past sim's rigid stopping point to hold
anything -- the same finding JointMirrorBroadcaster.broadcast() documents for live teleop
mirroring (sim/real agreed at first contact ~52.6/53.4 mm, but the real gripper needed ~43.4 mm
for a grip that held), and it fixes it the same way: mirror the open/close TARGET, not the
measured position, and let the real hardware decide how far it actually closes.

So this script's DEFAULT gripper drive is a two-state channel -- calibration's open_raw or
closed_raw, nothing in between -- taking the open/closed decision from one of these sources:

  --grip-json PATH   (preferred) a small precomputed open/close schedule, one entry per recorded
                     step, extracted from that recording's COMMAND columns. Preferred because it
                     needs no h5py in this environment -- see "REGENERATING" below.

  --grip-hdf5 PATH   the IsaacLab recording itself (record_demos_openarm.py HDF5). Its
                     actions[:, 7] / actions[:, 15] are the COMMAND the operator gave, already
                     0.0/0.044 binary, so there is nothing to infer. This is the same
                     command-vs-measured distinction the recording pipeline itself insists on.
                     Needs h5py installed here, which the lerobot env does not have by default.

  --grip-from-state  (fallback, no source recording available) infer it from the HF gripper
                     column with a latch: closed once the finger drops below --grip-close-below
                     having previously been open, open again above --grip-open-above. A
                     heuristic -- the measured value's open and closed ranges genuinely overlap
                     (see numbers above), so verify the printed open/close step list against
                     the episode before trusting it.

  --grip-continuous  NOT two-state: replay the HF dataset's gripper column as a continuous,
                     proportional command, every intermediate aperture included. This is what
                     replay_hf_sim_episode.py does, with one addition that makes it usable on a
                     real object: --grip-input-closed says which dataset value should map to a
                     FULLY closed real gripper, so the sim's stall value need not mean "75% open"
                     on the real hardware. Leave it at 0.0 for a literal reproduction of the sim
                     fingers; set it to the value the sim stalls at (~0.033 for pickup_pringle,
                     read it off the printed profile) to get the same real grip the two-state
                     mode gives while keeping the approach and release motion continuous.

Every mode ends up on the same scale: a per-step "closed fraction", 0.0 = calibration's open_raw,
1.0 = its closed_raw, scaled by --grip-close-frac. The two-state sources only ever emit 0 or 1.

SAFETY
------
Commanding closed_raw against a rigid object is a sustained stall on the DM4310 gripper motor --
exactly the condition mirror_bridge.py's GripperStallWatchdog exists for. That watchdog runs here
too (--no-grip-watchdog to disable) and power-cycles just the gripper motor if it stops responding.
If the can is being crushed, lower --grip-close-frac (1.0 = fully closed, 0.7 = 70% of the way
from open to closed).

REQUIRED BEFORE RUNNING: the same Phase 0 calibration as mirror_bridge.py, and a real can placed
where the sim's can was -- the arm trajectory is replayed open-loop and will close on empty air if
the object is somewhere else.

Usage:
  python replay_hf_sim_episode_realgrip.py \\
      --repo-id ethanCSL/openarm_visuomotor_VR_pringles --episode 0 \\
      --calibration calibration.json --model-path model/openarm_description_leader.urdf \\
      --grip-json pringles_grip_schedule.json \\
      --handshake-tolerance 1.0 --ramp-duration 10.0 --max-steps 3000 \\
      --plot sim_vs_real_realgrip.png

REGENERATING THE SCHEDULE (after re-recording, or for a different task's dataset) -- run this in
an environment that has h5py, e.g. IsaacLab's:

  python - <<'EOF'
  import h5py, json
  SRC = "logs/demos/pickup_pringle.hdf5"     # the recording the HF dataset was converted from
  MID = 0.022                                # half of the gripper's 0.044 m travel
  out = {"source": SRC, "episodes": {}}
  with h5py.File(SRC) as f:
      for i, name in enumerate(sorted(f["data"])):        # same order convert_hdf5_to_lerobot.py used
          a = f["data"][name]["actions"][:]
          out["episodes"][str(i)] = [[bool(r[7] < MID), bool(r[15] < MID)] for r in a]
  json.dump(out, open("pringles_grip_schedule.json", "w"))
  EOF
"""

import argparse
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")  # headless -- this process only ever saves a PNG, never shows a window
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from mirror_bridge import GripperStallWatchdog, _recover_gripper
from reset_to_rest_pose import reset_to_rest_pose
from robots.umeow_openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from sim_bridge_common import (
    ARM_JOINT_KEYS,
    GRIPPER_SIM_OPEN,
    StdinKillSwitch,
    clamp_step,
    compute_target_velocity,
    get_current_pos_action,
    load_calibration,
    ramp_to,
    run_startup_handshake,
)

# Schemas -- identical to replay_hf_sim_episode.py, see its docstring.
LEFT_STATE_NAMES_EEF_ACTION = [f"left_joint_{i}" for i in range(1, 8)]
LEFT_STATE_NAMES_JOINT_ACTION = [f"LJ{i}.pos" for i in range(1, 8)] + ["LJ8.pos"]
DUAL_STATE_NAMES = LEFT_STATE_NAMES_JOINT_ACTION + [f"RJ{i}.pos" for i in range(1, 9)]
POS_KEYS = [f"LJ{i}.pos" for i in range(1, 9)] + [f"RJ{i}.pos" for i in range(1, 9)]

# Layout of record_demos_openarm.py's 16D joint-space action row, i.e. what --grip-hdf5 reads:
#   [0:7] left arm | [7] left gripper | [8:15] right arm | [15] right gripper, in metres.
HDF5_ACTION_DIM = 16
HDF5_LEFT_GRIPPER_COL = 7
HDF5_RIGHT_GRIPPER_COL = 15


def load_trajectory(repo_id: str, episode: int) -> list[dict]:
    """Unchanged from replay_hf_sim_episode.py -- see that file."""
    dataset = LeRobotDataset(repo_id, episodes=[episode], revision="main", force_cache_sync=True)
    states = dataset.select_columns(OBS_STATE)
    actions = dataset.select_columns(ACTION)

    state_names = dataset.features[OBS_STATE]["names"]
    action_names = dataset.features[ACTION]["names"]

    if state_names == LEFT_STATE_NAMES_EEF_ACTION:
        schema = "eef_action"
    elif state_names == LEFT_STATE_NAMES_JOINT_ACTION:
        schema = "joint_action"
    elif state_names == DUAL_STATE_NAMES:
        schema = "dual_joint_action"
    else:
        raise ValueError(
            f"observation.state names {state_names} match no known schema -- expected "
            f"{LEFT_STATE_NAMES_EEF_ACTION} (EE-delta-action dataset), "
            f"{LEFT_STATE_NAMES_JOINT_ACTION} (single-arm joint-space dataset), or "
            f"{DUAL_STATE_NAMES} (dual-arm joint-space dataset)."
        )
    print(f"Dataset schema: {schema} ({len(state_names)}D observation.state)")

    if schema == "eef_action":
        if "left_gripper" not in action_names:
            raise ValueError(f"Expected 'left_gripper' in action names, got {action_names}")
        gripper_idx = action_names.index("left_gripper")

    trajectory = []
    for idx in range(dataset.num_frames):
        state_vals = states[idx][OBS_STATE].tolist()
        if schema == "eef_action":
            gripper_cmd = actions[idx][ACTION][gripper_idx].item()
            trajectory.append({"joints": state_vals, "gripper_mode": "cmd", "gripper_value": gripper_cmd})
        else:
            frame = {
                "joints": state_vals[:7],
                "gripper_mode": "pos",
                "gripper_value": state_vals[7],
            }
            if schema == "dual_joint_action":
                frame["right_joints"] = state_vals[8:15]
                frame["right_gripper_value"] = state_vals[15]
            trajectory.append(frame)
    return trajectory


# ── Where the open/closed decision comes from ─────────────────────────────────


def _check_source_length(label: str, n_source: int, expected_frames: int, allow_mismatch: bool) -> None:
    """convert_hdf5_to_lerobot.py drops each episode's last frame (action[t] = state[t+1] has no
    "next" for it), so a matching source episode is exactly ONE frame longer than the HF one.
    A mismatch means this is not the recording the dataset came from -- or not the same episode --
    and the grip would be applied at the wrong steps, hence the hard stop."""
    if n_source == expected_frames + 1 or allow_mismatch:
        return
    raise ValueError(
        f"{label} has {n_source} steps but the HF episode has {expected_frames} frames (expected"
        f" {expected_frames + 1} -- the converter drops the last frame). This is probably not the"
        " recording this dataset was built from, or not the same episode. Pass"
        " --grip-allow-length-mismatch to replay anyway (the grip will land on whatever steps"
        " line up)."
    )


def load_grip_commands_from_json(path: str, episode: int, expected_frames: int, allow_mismatch: bool):
    """Read a precomputed [(left_closed, right_closed), ...] schedule -- see REGENERATING above.

    Deliberately a dumb file of booleans: the decision of what counts as "closed" is made once,
    in the environment that has the source recording, rather than re-derived on the robot host.
    """
    with open(path) as f:
        blob = json.load(f)
    episodes = blob.get("episodes", {})
    key = str(episode)
    if key not in episodes:
        raise ValueError(f"{path} has episodes {sorted(episodes, key=int)}; no entry for episode {episode}.")
    grips = [(float(bool(left)), float(bool(right))) for left, right in episodes[key]]
    print(f"Gripper commands from {path} :: episode {episode} ({len(grips)} steps,"
          f" source {blob.get('source', 'unrecorded')})")
    _check_source_length(f"{path} episode {episode}", len(grips), expected_frames, allow_mismatch)
    _print_grip_schedule(grips, "left")
    _print_grip_schedule(grips, "right")
    return grips


def load_grip_commands_from_hdf5(path: str, episode: int, expected_frames: int, allow_mismatch: bool):
    """Read one IsaacLab episode's binary gripper COMMANDS as [(left_closed, right_closed), ...].

    Episodes are indexed the way convert_hdf5_to_lerobot.py enumerated them (sorted(f["data"]))
    so --episode N here selects the same episode the HF dataset's episode N was built from.
    """
    try:
        import h5py
    except ImportError as e:
        raise SystemExit(
            "--grip-hdf5 needs h5py, which this environment does not have"
            f" ({e}). Either `pip install h5py` here, or use --grip-json with a schedule"
            " exported from an environment that does (see this file's REGENERATING section)."
        )
    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys())
        if episode >= len(names):
            raise ValueError(f"{path} has {len(names)} episodes; --grip-hdf5-episode {episode} is out of range.")
        name = names[episode]
        actions = f["data"][name]["actions"][:]
    print(f"Gripper commands from {path} :: {name}  {actions.shape}")

    if actions.ndim != 2 or actions.shape[1] != HDF5_ACTION_DIM:
        raise ValueError(
            f"{name}'s actions are {actions.shape}, not Nx{HDF5_ACTION_DIM} -- --grip-hdf5 needs a"
            " record_demos_openarm.py --teleop_device vr_joint_ros2* recording."
        )
    _check_source_length(name, len(actions), expected_frames, allow_mismatch)

    columns = actions[:, [HDF5_LEFT_GRIPPER_COL, HDF5_RIGHT_GRIPPER_COL]]
    midpoint = 0.5 * GRIPPER_SIM_OPEN
    # These columns should be pure 0.0/0.044. Anything sitting mid-travel is a trigger that was
    # not squeezed all the way (or a pre-snap recording) -- still usable via the midpoint test
    # below, but worth flagging since it means the source demo's own grip was partial.
    ambiguous = int(((columns > 0.15 * GRIPPER_SIM_OPEN) & (columns < 0.85 * GRIPPER_SIM_OPEN)).sum())
    if ambiguous:
        print(
            f"  note: {ambiguous} of {columns.size} gripper values are mid-travel rather than"
            " fully open/closed. Treating < half-open as 'closed'. See IsaacLab's"
            " scripts/tools/snap_gripper_actions.py if this recording needs cleaning up."
        )
    grips = [(float(row[0] < midpoint), float(row[1] < midpoint)) for row in columns]
    _print_grip_schedule(grips, "left")
    _print_grip_schedule(grips, "right")
    return grips


class GripLatch:
    """Infer open/closed from the sim's MEASURED finger value, for when no source recording is at
    hand (--grip-from-state).

    Latching, not thresholding: the measured value's "open" and "closed" ranges overlap (a
    stalled grasp reads as high as 0.043 against a 0.044 open), and every episode STARTS with the
    fingers somewhere near closed, before the demo opens them. So a bare threshold both flickers
    mid-grasp and reports a phantom grasp on frame 0. This closes only after having seen the
    gripper genuinely open first, and re-opens only above a separate, higher threshold.
    """

    def __init__(self, close_below: float, open_above: float):
        self._close_below = close_below
        self._open_above = open_above
        self._armed = False   # has this gripper been seen open yet?
        self._closed = False

    def update(self, sim_val: float) -> float:
        if sim_val >= self._open_above:
            self._armed = True
            self._closed = False
        elif self._armed and sim_val <= self._close_below:
            self._closed = True
        return float(self._closed)


def _closed_fraction(value: float, mode: str, input_open: float, input_closed: float) -> float:
    """One recorded gripper sample -> closed fraction in [0, 1]; 0 = fully open, 1 = fully closed.

    `mode` is load_trajectory()'s gripper_mode: "cmd" for the EE-delta schema's +-1 command column
    (+1 = open, as gripper_cmd_to_raw() reads it), "pos" for the joint-space schemas' finger angle
    in metres, whose open/closed ends are given by input_open/input_closed rather than assumed to
    be 0.044/0.0 -- that remap is the whole point of --grip-input-closed (see module docstring).
    """
    if mode == "cmd":
        open_frac = (value + 1.0) / 2.0
    else:
        open_frac = (value - input_closed) / (input_open - input_closed)
    return 1.0 - max(0.0, min(1.0, open_frac))


def grip_fractions_from_dataset(trajectory: list, input_open: float, input_closed: float) -> list:
    """--grip-continuous: the gripper channel replayed straight from the HF dataset, proportionally.

    The right side falls back to fully open when the dataset is single-arm; that value is never
    sent (frame_to_motor_action() holds the right arm instead), it just keeps the pairs uniform.
    """
    grips = [
        (
            _closed_fraction(f["gripper_value"], f["gripper_mode"], input_open, input_closed),
            _closed_fraction(f["right_gripper_value"], f["gripper_mode"], input_open, input_closed)
            if "right_gripper_value" in f else 0.0,
        )
        for f in trajectory
    ]
    print(f"Gripper commands replayed CONTINUOUSLY from the dataset's own gripper column"
          f" (dataset {input_open} -> fully open, {input_closed} -> fully closed):")
    _print_grip_profile(grips, "left", trajectory, "gripper_value")
    _print_grip_profile(grips, "right", trajectory, "right_gripper_value")
    return grips


def _print_grip_schedule(grips: list, side: str) -> None:
    """Print the steps at which one side's gripper changes state, so the operator can sanity-check
    the schedule against the episode BEFORE any of it is sent to a real motor."""
    idx = 0 if side == "left" else 1
    changes = [(i, g[idx]) for i, g in enumerate(grips) if i == 0 or g[idx] != grips[i - 1][idx]]
    closed_frames = sum(1 for g in grips if g[idx] >= 0.5)
    schedule = ", ".join(f"step {i}: {'CLOSE' if c >= 0.5 else 'open'}" for i, c in changes)
    print(f"  {side:5s} gripper: {closed_frames}/{len(grips)} closed frames | {schedule}")


def _fmt_grip(closed_frac: float) -> str:
    """Per-step console label: the two-state modes keep reading CLOSE/open, --grip-continuous
    shows the actual fraction."""
    if closed_frac <= 0.0:
        return "open "
    if closed_frac >= 1.0:
        return "CLOSE"
    return f"{closed_frac:5.2f}"


def _print_grip_profile(grips: list, side: str, trajectory: list, value_key: str) -> None:
    """Continuous counterpart of _print_grip_schedule(): the raw dataset range this side spans and
    a downsampled closed-fraction trace, so the operator can see BEFORE moving a motor whether the
    episode's grasp actually maps to a real squeeze under the current --grip-input-closed."""
    idx = 0 if side == "left" else 1
    raw = [f[value_key] for f in trajectory if value_key in f]
    if not raw:
        print(f"  {side:5s} gripper: no data in this dataset -- held, not commanded")
        return
    fracs = [g[idx] for g in grips]
    peak = max(range(len(fracs)), key=lambda i: fracs[i])
    step = max(1, len(fracs) // 20)
    trace = " ".join(f"{fracs[i]:.2f}" for i in range(0, len(fracs), step))
    print(f"  {side:5s} gripper: dataset value {min(raw):.4f}..{max(raw):.4f} -> closed fraction"
          f" {min(fracs):.2f}..{max(fracs):.2f} (peak {fracs[peak]:.2f} at step {peak})")
    print(f"         every {step} steps: {trace}")


# ── Motor mapping ─────────────────────────────────────────────────────────────


def _grip_raw(sec: dict, closed_frac: float, close_frac: float) -> float:
    """Gripper target in raw motor units, from a closed fraction (0 = open_raw, 1 = closed_raw).

    The two-state sources pass exactly 0.0 or 1.0 -- deliberately NOT the sim-proportional
    gripper_sim_to_raw() mapping, for the reason in the module docstring; --grip-continuous passes
    everything in between. close_frac scales the closed end in either case: < 1.0 stops short of
    the calibrated fully-closed value, for an object that is being crushed at full close.
    """
    grip = sec["gripper"]
    return grip["open_raw"] + close_frac * closed_frac * (grip["closed_raw"] - grip["open_raw"])


def _map_arm(action: dict, prefix: str, sec: dict, joints, closed_frac: float, close_frac: float) -> None:
    """Write one arm's 8 motor keys ({prefix}1..8.pos) into *action*, through its calibration."""
    for n, jkey in enumerate(ARM_JOINT_KEYS, start=1):
        action[f"{prefix}{n}.pos"] = sec["sign"][jkey] * joints[n - 1] + sec["offset_rad"][jkey]
    action[f"{prefix}8.pos"] = _grip_raw(sec, closed_frac, close_frac)


def frame_to_motor_action(frame: dict, grip: tuple, calib: dict, hold_right: dict, close_frac: float) -> dict:
    """Map one recorded frame + its (left, right) closed fractions to a 16-key motor action.

    Right arm: driven from the data when the dataset carries it, otherwise pinned to `hold_right`
    -- including its gripper, which is left exactly where it was rather than being commanded from
    a decision the dataset cannot support.
    """
    _map_arm(action := {}, "LJ", calib["left"], frame["joints"], grip[0], close_frac)
    if "right_joints" in frame:
        _map_arm(action, "RJ", calib["right"], frame["right_joints"], grip[1], close_frac)
    else:
        action.update(hold_right)
    return action


def save_tracking_plot(plot_history: dict, out_path: str) -> None:
    """Per-joint target-vs-actual time series. Unchanged from replay_hf_sim_episode.py, except
    that a gripper subplot's "target" is now this script's command (two-state, or the dataset's
    continuous profile under --grip-continuous) -- the actual trace sitting well short of it
    during a grasp is the object being held, not a tracking failure."""
    ARM_AXIS_TOLERANCE = 0.1
    GRIPPER_AXIS_TOLERANCE = 0.005

    ncols = 4
    nrows = (len(POS_KEYS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, name in enumerate(POS_KEYS):
        ax = axes[idx // ncols][idx % ncols]
        h = plot_history[name]
        ax.plot(h["t"], h["target"], label="target", linewidth=1)
        ax.plot(h["t"], h["actual"], label="actual", linewidth=1, linestyle="--")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("s", fontsize=7)
        ax.set_ylabel("rad", fontsize=7)

        all_v = h["target"] + h["actual"]
        if all_v:
            tolerance = GRIPPER_AXIS_TOLERANCE if name.endswith("8.pos") else ARM_AXIS_TOLERANCE
            data_min, data_max = min(all_v), max(all_v)
            center = (data_min + data_max) / 2
            half_range = max((data_max - data_min) / 2 * 1.1, tolerance)
            ax.set_ylim(center - half_range, center + half_range)

    for idx in range(len(POS_KEYS), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Dataset target vs. real robot actual joint positions (real-grip replay)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved target-vs-actual comparison plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", type=str, required=True, help="HF dataset repo id")
    parser.add_argument("--episode", type=int, default=0, help="Episode index inside the dataset")
    parser.add_argument("--calibration", type=str, required=True, help="Path to calibration.json")
    parser.add_argument("--right-port", type=str, default="can0")
    parser.add_argument("--left-port", type=str, default="can1")
    parser.add_argument("--model-path", type=str, required=True, help="Path to openarm_description.urdf for gravity comp")
    parser.add_argument("--max-joint-speed", type=float, default=1.2, help="rad/s cap applied to every arm joint's per-tick motion.")
    parser.add_argument("--gripper-max-speed", type=float, default=2.0, help="rad/s cap for the gripper channel specifically -- much higher than the arm cap, since gripper commands are a near-instant open/closed toggle, not a smooth trajectory")
    parser.add_argument("--handshake-tolerance", type=float, default=0.1, help="rad; abort startup if any arm joint differs from the first frame by more than this")
    parser.add_argument("--gripper-handshake-tolerance", type=float, default=1.3, help="rad; separate, more generous tolerance for the gripper channel -- open/closed state legitimately varies between episodes")
    parser.add_argument("--ramp-duration", type=float, default=2.0, help="seconds to smoothly move from real current pose to the first frame")
    parser.add_argument("--playback-hz", type=float, default=20.0, help="rate to step through the recorded trajectory (this dataset's fps is 30)")
    parser.add_argument("--max-steps", type=int, default=None, help="only replay the first N steps of the episode -- use for a cautious first test")
    parser.add_argument("--log-csv", type=str, default=None, help="If set, log target/commanded/actual for all 16 channels to this CSV path.")
    parser.add_argument("--plot", type=str, default=None, help="If set, save a per-joint target-vs-actual comparison plot (PNG) to this path.")

    grip = parser.add_argument_group("gripper (this script's reason for existing -- see module docstring)")
    grip.add_argument(
        "--grip-json", type=str, default=None,
        help="Precomputed open/close schedule exported from the source recording's COMMAND columns"
        " (see this file's REGENERATING section). Preferred: exact, and needs no h5py here.",
    )
    grip.add_argument(
        "--grip-hdf5", type=str, default=None,
        help="IsaacLab recording (record_demos_openarm.py HDF5) the HF dataset was converted from."
        " Its binary gripper command column decides open/closed. Preferred: exact, nothing inferred.",
    )
    grip.add_argument(
        "--grip-hdf5-episode", type=int, default=None,
        help="Episode index inside --grip-hdf5/--grip-json. Defaults to --episode (the converter"
        " preserves episode order).",
    )
    grip.add_argument(
        "--grip-allow-length-mismatch", action="store_true", default=False,
        help="Replay even if --grip-hdf5's episode length doesn't match the HF episode's."
        " Only sensible if you know why they differ -- otherwise the grip lands on the wrong steps.",
    )
    grip.add_argument(
        "--grip-from-state", action="store_true", default=False,
        help="No source recording available: infer open/closed from the HF dataset's own measured"
        " gripper column (see GripLatch). Heuristic -- check the printed schedule before running.",
    )
    grip.add_argument("--grip-close-below", type=float, default=0.042, help="--grip-from-state: sim finger value (m) below which the gripper counts as closing, once it has been seen open.")
    grip.add_argument("--grip-open-above", type=float, default=0.0435, help="--grip-from-state: sim finger value (m) at or above which it counts as open again.")
    grip.add_argument(
        "--grip-continuous", action="store_true", default=False,
        help="Replay the dataset's gripper column proportionally instead of as open/closed --"
        " every intermediate aperture, exactly as recorded. Pair with --grip-input-closed to"
        " still get a real grip out of a sim value that stalled short of fully closed.",
    )
    grip.add_argument(
        "--grip-input-open", type=float, default=GRIPPER_SIM_OPEN,
        help=f"--grip-continuous: dataset gripper value that maps to the real gripper fully OPEN"
        f" (default {GRIPPER_SIM_OPEN}, the sim's open travel). Ignored for the +-1 command schema.",
    )
    grip.add_argument(
        "--grip-input-closed", type=float, default=0.0,
        help="--grip-continuous: dataset gripper value that maps to the real gripper fully CLOSED."
        " 0.0 (default) reproduces the sim fingers literally, which on a rigid-body sim grasp"
        " never reaches a real squeeze; set it to the value the sim stalls at (~0.033 for"
        " pickup_pringle -- read it off the printed profile) to stretch that stall into a real"
        " grip. Ignored for the +-1 command schema.",
    )
    grip.add_argument(
        "--grip-close-frac", type=float, default=1.0,
        help="How far toward calibration's closed_raw a fully-closed command goes. 1.0 = fully"
        " closed (a real grip); lower it if the object is being crushed or the motor keeps"
        " overloading. Scales --grip-continuous's whole range too, not just its closed end.",
    )
    grip.add_argument(
        "--no-grip-watchdog", dest="grip_watchdog", action="store_false", default=True,
        help="Disable the gripper stall watchdog. It is on by default here because holding a real"
        " grasp is exactly what trips the DM4310's overcurrent/overload latch; it needs the same"
        " per-step position read as --log-csv, which slows playback below --playback-hz.",
    )
    args = parser.parse_args()

    sources = [bool(args.grip_json), bool(args.grip_hdf5), args.grip_from_state, args.grip_continuous]
    if sum(sources) != 1:
        parser.error(
            "Choose exactly ONE gripper source: --grip-json PATH (preferred two-state), --grip-hdf5"
            " PATH, --grip-from-state, or --grip-continuous (proportional replay of the dataset's"
            " own column). There is no default -- this script exists precisely because that choice"
            " matters."
        )
    if not 0.0 <= args.grip_close_frac <= 1.0:
        parser.error(f"--grip-close-frac must be in [0, 1], got {args.grip_close_frac}")
    if args.grip_continuous and args.grip_input_open == args.grip_input_closed:
        parser.error("--grip-input-open and --grip-input-closed must differ (they define a range).")

    calib = load_calibration(args.calibration)
    print(f"Loading {args.repo_id}, episode {args.episode}...")
    trajectory = load_trajectory(args.repo_id, args.episode)
    print(f"Loaded {len(trajectory)} steps.")

    # Resolved over the FULL episode, before --max-steps truncates it, so the printed schedule
    # describes the whole demo and the step numbers in it mean what they say.
    source_episode = args.episode if args.grip_hdf5_episode is None else args.grip_hdf5_episode
    if args.grip_json:
        grips = load_grip_commands_from_json(
            args.grip_json, source_episode, len(trajectory), args.grip_allow_length_mismatch
        )
    elif args.grip_hdf5:
        grips = load_grip_commands_from_hdf5(
            args.grip_hdf5, source_episode, len(trajectory), args.grip_allow_length_mismatch
        )
    elif args.grip_continuous:
        grips = grip_fractions_from_dataset(trajectory, args.grip_input_open, args.grip_input_closed)
    else:
        left_latch = GripLatch(args.grip_close_below, args.grip_open_above)
        right_latch = GripLatch(args.grip_close_below, args.grip_open_above)
        grips = [
            (
                left_latch.update(frame["gripper_value"]),
                right_latch.update(frame.get("right_gripper_value", GRIPPER_SIM_OPEN)),
            )
            for frame in trajectory
        ]
        print(f"Gripper state inferred from the dataset's measured finger column"
              f" (close<= {args.grip_close_below}, open>= {args.grip_open_above}):")
        _print_grip_schedule(grips, "left"), _print_grip_schedule(grips, "right")

    if args.max_steps is not None:
        trajectory = trajectory[: args.max_steps]
    grips = grips[: len(trajectory)]
    if len(grips) < len(trajectory):
        # Only reachable with --grip-allow-length-mismatch; hold the last known state rather than
        # silently opening the gripper and dropping whatever is being carried.
        print(f"WARNING: only {len(grips)} gripper decisions for {len(trajectory)} frames --"
              " holding the last one for the remainder.")
        grips += [grips[-1]] * (len(trajectory) - len(grips))

    robot_cfg = OpenArmFollowerConfig(
        right_port=args.right_port,
        left_port=args.left_port,
        enable_fd=True,
        model_path=args.model_path,
    )
    robot = OpenArmFollower(robot_cfg)
    robot.connect()

    log_file = None
    log_writer = None
    if args.log_csv:
        log_file = open(args.log_csv, "w", newline="")
        fieldnames = ["step", "t"]
        for k in POS_KEYS:
            fieldnames += [f"{k}_target", f"{k}_commanded", f"{k}_actual"]
        log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        log_writer.writeheader()
        print(f"Logging target/commanded/actual positions to {args.log_csv}")

    collect_diagnostics = bool(args.log_csv or args.plot or args.grip_watchdog)
    plot_history = {k: {"t": [], "target": [], "actual": []} for k in POS_KEYS} if args.plot else None
    watchdog = GripperStallWatchdog() if args.grip_watchdog else None

    try:
        current_action = get_current_pos_action(robot)
        hold_right = {k: v for k, v in current_action.items() if k.startswith("R")}
        dual_arm = "right_joints" in trajectory[0]
        print(
            "Right arm: DRIVEN from the dataset (dual-arm replay)" if dual_arm
            else "Right arm: HELD at its current pose (dataset has no right-arm data)"
        )
        for side in ("left", "right"):
            g = calib[side]["gripper"]
            closed_target = _grip_raw(calib[side], 1.0, args.grip_close_frac)
            print(f"  {side:5s} gripper targets: open={g['open_raw']:+.4f}  closed={closed_target:+.4f}"
                  f" (calibrated closed_raw={g['closed_raw']:+.4f}, --grip-close-frac {args.grip_close_frac})")

        target_action = frame_to_motor_action(trajectory[0], grips[0], calib, hold_right, args.grip_close_frac)

        if not run_startup_handshake(robot, target_action, args.handshake_tolerance, args.gripper_handshake_tolerance):
            retry = input(
                "\nThis is often just the real arm having drifted since a previous session."
                " Try moving it to match the episode's first frame, then re-check? [y/N]: "
            )
            if retry.strip().lower() != "y":
                return
            if not reset_to_rest_pose(robot, calib, target_action=target_action):
                return
            if not run_startup_handshake(robot, target_action, args.handshake_tolerance, args.gripper_handshake_tolerance):
                print("Still mismatched against the first frame after resetting. Aborting.")
                return

        confirm = input(f"Type YES to ramp to the first frame and replay {len(trajectory)} steps: ")
        if confirm.strip() != "YES":
            print("Not confirmed. Aborting without moving the arm.")
            return

        # Started only after the confirmation prompt -- see replay_hf_sim_episode.py for why.
        kill_switch = StdinKillSwitch()
        print("Type 'q' + Enter at any time to stop playback and safely ramp down.")

        print(f"Ramping to first frame over {args.ramp_duration}s...", flush=True)
        current_action = ramp_to(robot, current_action, target_action, args.ramp_duration)
        print("Ramp complete. Replaying. Type 'q' + Enter to stop.")

        dt = 1.0 / args.playback_hz
        max_delta = args.max_joint_speed * dt
        gripper_max_delta = args.gripper_max_speed * dt
        halted = False
        t0 = time.time()
        tracking_errors = {k: [] for k in POS_KEYS}

        for step_idx, (frame, grip_state) in enumerate(zip(trajectory, grips)):
            loop_start = time.time()

            if kill_switch.triggered:
                print(f"\nKill switch pressed at step {step_idx}/{len(trajectory)}. Ramping down and disabling.")
                halted = True
                break

            desired = frame_to_motor_action(frame, grip_state, calib, hold_right, args.grip_close_frac)
            target_action = clamp_step(current_action, desired, max_delta, gripper_max_delta)
            target_vel = compute_target_velocity(current_action, target_action, dt, args.max_joint_speed)
            robot.send_action(target_action, target_vel)
            current_action = target_action

            if collect_diagnostics:
                try:
                    actual = get_current_pos_action(robot)
                except RuntimeError as e:
                    actual = {}
                    print(f"  [diagnostics] state read failed at step {step_idx}, logging blanks: {e}")
                now_t = time.time() - t0
                if log_writer is not None:
                    row = {"step": step_idx, "t": now_t}
                    for k in POS_KEYS:
                        row[f"{k}_target"] = desired.get(k)
                        row[f"{k}_commanded"] = current_action.get(k)
                        row[f"{k}_actual"] = actual.get(k)
                    log_writer.writerow(row)
                for k in POS_KEYS:
                    if k in actual and k in desired:
                        tracking_errors[k].append(abs(actual[k] - desired[k]))
                        if plot_history is not None:
                            plot_history[k]["t"].append(now_t)
                            plot_history[k]["target"].append(desired[k])
                            plot_history[k]["actual"].append(actual[k])

                # A grasp legitimately sits far from its closed target for as long as it holds --
                # the watchdog checks responsiveness, not error. See its docstring.
                if watchdog is not None:
                    for side, prefix in (("left", "L"), ("right", "R")):
                        key = f"{prefix}J8.pos"
                        if key in actual and watchdog.check(side, current_action[key], actual[key]):
                            _recover_gripper(robot, side)

            print(f"  step {step_idx}/{len(trajectory)}"
                  f"  grip L={_fmt_grip(grip_state[0])} R={_fmt_grip(grip_state[1])}")

            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

        if not halted:
            print(f"\nReplay of all {len(trajectory)} steps complete.")

        if log_writer is not None:
            print("\nTracking summary (|actual - target| per channel). The gripper channels are"
                  " EXPECTED to show a large error while an object is held -- that gap is the grip.")
            for k in POS_KEYS:
                errs = tracking_errors[k]
                if errs:
                    print(f"  {k:10s} mean={sum(errs) / len(errs):.4f} rad   max={max(errs):.4f} rad")

        print("Ramping down to a safe hold before disabling...")
        safe_hold = get_current_pos_action(robot)
        ramp_to(robot, current_action, safe_hold, duration_s=1.0)

    except KeyboardInterrupt:
        print("\nInterrupted. Disabling motors.")
    finally:
        if log_file is not None:
            log_file.close()
            print(f"Wrote log to {args.log_csv}")
        if plot_history is not None and any(plot_history[k]["t"] for k in POS_KEYS):
            save_tracking_plot(plot_history, args.plot)
        try:
            robot.disconnect()
        except Exception:
            print("WARNING: error during disconnect -- verify motors are physically de-energized.")
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
