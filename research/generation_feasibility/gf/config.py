"""Generation-feasibility experiment: constants, parameter ranges, criteria and stop rule.

Version `genfeas-0.1-dryrun`. Everything that the simulated pilot will be judged by is written here BEFORE any
simulation is run. Values are either copied from the IsaacLab task code they mirror (cited), declared physical priors
(cited reason), or design choices stated with their reason. A change after the pilot has been run requires a new version
and fresh pilot poses (see GENERATION_FEASIBILITY_PLAN.md §8).
"""

from pathlib import Path

VERSION = "genfeas-0.1-dryrun"

REPO = Path(__file__).resolve().parents[3]
ISAACLAB = Path.home() / "Stanley_ws/IsaacLab"
ISAACLAB_COMMIT_AUDITED = "4aa5e026"

# ------------------------------------------------------------------ data sources (see BASELINE_GENERATOR_AUDIT.md §2)
SIM10_REPO = "ethanCSL/openarm_pringles_lerobot_sim_10_v00"
SIM10_REV = "b6d39dae630b8ce0b5822bee7a40bffc68a28819"
SIM10_ROOT = Path.home() / f".cache/huggingface/lerobot/hub/datasets--ethanCSL--openarm_pringles_lerobot_sim_10_v00/snapshots/{SIM10_REV}"
# true initial can xy of sim_20 episodes 0-19 (= sim_10 episodes 0-9), read from the server HDF5 by the earlier diagnosis
SIM20_START_POSES = REPO / "research/generated_data_diagnosis/data/sim20_start_poses.json"
EPISODE_EVENTS = REPO / "research/generated_data_diagnosis/data/episode_events.csv"
EVAL_POSES = REPO / "research/generated_data_diagnosis/protocol/eval_poses_100.json"
# local Mimic-annotated sources with per-step simulator ground truth (older env: -v0, 20 Hz, pre-foam-pad)
LOCAL_GT_HDF5 = {
    "V9": ISAACLAB / "logs/demos/pickup_pringles_V9_annotated.hdf5",
    "stanley": ISAACLAB / "logs/demos/pickup_pringles_stanley_annotated.hdf5",
}

FPS = 30                       # sim_10_v00 and the current env (OPENARM_CONTROL_HZ default 30)
NAMES = [f"{s}J{i}.pos" for i in range(1, 9) for s in ("R", "L")]   # interleaved RJ1, LJ1, ... RJ8, LJ8
GRIP_OPEN_RAW = {"R": -1.298, "L": 1.2121}                         # calibration.json; closed = 0
FINGER_OPEN = 0.044                                               # sim finger joint, m

# ------------------------------------------------------------------ geometry mirrored from openarm_task_modes.py @4aa5e026
CAN_SCALE = 0.8
CAN_HALF_HEIGHT = CAN_SCALE * 0.2502 / 2        # 0.10008 m
CAN_RADIUS = CAN_SCALE * 0.036625               # 0.0293 m
CAN_REST_Z = 0.380                              # rest height of the nominal can (DATA_GENERATION_AUDIT §2)
PAD_TOP_Z = CAN_REST_Z - CAN_HALF_HEIGHT        # ~0.280 m
EEF_TO_CAN_RADIAL = 0.07                        # EEF_TO_CAN_THRESHOLD
GRASP_AXIAL_TOL = CAN_HALF_HEIGHT + 0.02        # GRASP_AXIAL_TOLERANCE
RECEIVER_APERTURE = (0.015, 0.076)              # HANDOVER_RECEIVER_APERTURE_RANGE (sum of both finger joints)
GRIPPER_THRESHOLD = 0.005
LIFT_MIN = 0.025                                # apply_task_mode lift_height_offset
PUT_DOWN_MARGIN = 0.005                         # HANDOVER_PUT_DOWN_MARGIN
GRASP_POINT_IN_HAND = (0.0, 0.0, 0.0024)        # sim grasp frame vs URDF openarm_*_hand origin (fitted, <2 mm residual)


def scale_steps(n_ref: float, hz: int = FPS) -> int:
    """openarm_sim_timing.scale_steps (Python round, reference 20 Hz)."""
    return max(1, int(round(n_ref * hz / 20.0)))


def scale_odd(n_ref: float, hz: int = FPS) -> int:
    return max(1, int(n_ref * hz / 20.0) | 1)


SEAM_INTERP = scale_steps(15)                   # 22 at 30 Hz
LEFT_INTERP = 2 * SEAM_INTERP - scale_steps(5)  # 36
RELEASE_OVERLAP = scale_steps(5)                # 8: _HANDOVER_RELEASE_OVERLAP_STEPS
RECEIVER_HOLD = scale_steps(20)                 # 30: HANDOVER_RECEIVER_HOLD_STEPS
SMOOTH_WINDOW = scale_odd(5)                    # 7: waypoint_smoothing_window

# ------------------------------------------------------------------ robot contracts (OpenArmUmeow.send_action; same as Gate 1R A)
MAX_JOINT_SPEED = 2.5          # rad/s
MAX_TARGET_JUMP = 0.25         # rad per step: safety hold
STEP_CAP = MAX_JOINT_SPEED / FPS
JOINT_LIMIT_MARGIN = 0.02      # rad inside the URDF limits
ELBOW_MIN = 0.10               # rad: limited-IK keeps the elbow bent (openarm_mimic_actions: lower bound tighter than 0)

# ------------------------------------------------------------------ offline constraint thresholds (dry-run checks; condition C enforces them)
IK_POS_TOL = 0.005             # m
IK_ROT_TOL_DEG = 3.0
ARM_CLEARANCE = 0.03           # m between non-hand link capsules of the two arms (sim self-collision is OFF by default)
HAND_CLEARANCE = 0.015         # m between the two hands' finger capsules outside the dual-contact window
TABLE_CLEARANCE = 0.01         # m above the pad top for finger tips and the carried can bottom (outside the pick)
LINK_RADIUS = 0.045            # m capsule radius for arm links
FINGER_RADIUS = 0.015

# ------------------------------------------------------------------ condition B/C parameter ranges (sampled per candidate, seeded)
B_RANGES = {
    # Widened in the dry run (before any simulation): with +-4 cm / +-25 deg only 6/21 accepted C plans left the source
    # envelope at plan level, so the pilot could not have met S2 even with 100 % execution success.
    "pass_dx": (-0.07, 0.07),           # m: one translation applied to the whole pass (giver present..withdraw, receiver approach..carry)
    "pass_dy": (-0.06, 0.06),
    "pass_dz": (-0.04, 0.05),
    "approach_bump": (0.0, 0.06),       # m: receiver approach via-point offset, zero at both ends
    "grasp_axial": (-0.04, 0.04),       # m: receiver grasp height along the can axis, relative to the source grasp
    "grasp_azimuth_deg": (-45.0, 45.0), # receiver approach azimuth about the can axis, relative to the source grasp
    "approach_speed": (0.8, 1.25),      # time-warp of the receiver approach segment
    "receiver_dwell": (0, 12),          # steps the receiver waits at the grasp pose before closing
    "release_delay": (0, 15),           # extra steps the giver holds after the take latch (on top of RELEASE_OVERLAP)
}
C_SHRINK = (1.0, 0.6, 0.35)            # C: if a candidate violates a constraint, shrink its spatial parameters, at most 3 tries
C_CLOSE_GATE_WAIT = 15                 # C runtime: receiver closes only when the can is in its grasp cylinder; waits <= 15 steps
C_PREGRASP = 0.06                      # C: last 6 cm of the receiver approach is a straight line along its approach axis
# jaw-capture prior (declared geometry): the can centre expressed in a grasp frame (x ~ can axis, y = jaw axis,
# z = approach) must lie between the OPEN jaws: |y| <= open half-aperture - can radius = 0.044 - 0.029, depth z within
# the finger reach, x on the can body. Source attachments measured in GT: (0.048..0.051, -0.003..-0.002, 0.037..0.050).
JAW_CAPTURE = {"y": 0.015, "z": (0.0, 0.075), "x": CAN_HALF_HEIGHT - 0.02}
# nominal giver attachment (can pose in the RIGHT grasp frame, 15 steps after grasp_right), mean over the 40 local GT
# demos (V9 + stanley): position (0.0489, -0.0030, 0.0468) m, std (0.0135, 0.0029, 0.0068); can axis in the grasp frame
# (0.985, -0.050, -0.119). Used as the plan-time prediction for sim_10 sources, which carry no can pose.
NOMINAL_ATTACH_POS = (0.0489, -0.0030, 0.0468)
NOMINAL_ATTACH_AXIS = (0.985, -0.050, -0.119)
ATTACH_STD = (0.0135, 0.0029, 0.0068)
SETTLE_DRIFT = 0.007             # m, median settle->take drift of the can in the giver frame (GT: 5.3 / 7.2 mm)
# nominal receiver attachment (can pose in the LEFT grasp frame, 20 steps after the take), mean of the 40 GT demos;
# condition C builds its receiver grasp from this canonical capture pose (rotated / slid on the can), not from the
# source's receiver pose.
NOMINAL_RECEIVER_POS = (-0.0499, -0.0018, 0.0460)
NOMINAL_RECEIVER_AXIS = (0.989, 0.038, -0.103)
MIN_GRASP_SEPARATION = 0.045           # C: |giver - receiver grasp point| along the can axis or around it, m (two hands on one can)

# ------------------------------------------------------------------ pilot (prepared, NOT executed)
PILOT_N = 24                           # candidates per condition, same 24 can poses for A, B, C
PILOT_POSE_SEED = 2027                 # the poses are drawn from eval_poses_100.json (stratified evaluation spawn rule)
PILOT_PARAM_SEED = 4242
SMOKE_N = 4                            # smoke-stage poses, disjoint from the pilot poses
EPISODE_STEPS = 420                    # fixed simulated budget at 30 Hz (14 s; the longest generated plan + margin)
POST_RELEASE_RETENTION = 30            # steps the receiver must keep the can after the giver releases
SLIP_TOL = 0.015                       # m: drift of the can in the holding hand's frame counted as slip
TILT_MAX_DEG = 30.0                    # beyond the manual-demo maximum (26 deg) counts as a jolt flag
CAN_SPEED_MAX = 1.0                    # m/s: as DATA_GENERATION_AUDIT heuristic flag

# ------------------------------------------------------------------ pre-registered pilot criteria (see GENERATION_FEASIBILITY_PLAN.md §6)
CRITERIA = {
    "P0_replay": "smoke stage: the 10 sources replayed through the pilot runner reach stable success in >= 8/10, and "
                 "condition A reaches stable success in >= 3/4 smoke poses; otherwise the runner/env is broken -> stop, report",
    "S1_executable": "condition C: stable handover success >= 15/24 (62.5 %) AND not more than 3 below condition A",
    "S2_novel": "among C's stable successes, >= 50 % are NOVEL: executed handover descriptor outside the source envelope "
                "(pass point >= 3 cm from every source pass point, OR receiver grasp height/azimuth in the can frame "
                "outside the source range by >= 1 cm / 10 deg), AND C's median novelty ratio >= 0.5 while A's is <= 0.2",
    "S3_quality": "C stable successes: post-transfer retention >= 30 steps in >= 90 %; slip > 1.5 cm rate not more than "
                  "15 points above the P0 source-replay rate in the same env (known-good GT demos already show 27.5 %, "
                  "11/40); 0 joint-target safety-contract violations; 0 arm-arm clearance violations (offline capsule "
                  "check on the measured joints)",
    "S4_constraints_matter": "reported, not a GO condition: C - B stable success (paired by pose, McNemar)",
}
STOP_RULE = ("One pilot run per condition on the frozen 24 poses. If S1, S2 or S3 fails for C: NO-GO for scaling this "
             "generator; do not retune ranges on the same poses. If all hold: GO for a scaled generation experiment "
             "(new poses, a policy-level test) - still NOT evidence of policy benefit or force awareness.")
