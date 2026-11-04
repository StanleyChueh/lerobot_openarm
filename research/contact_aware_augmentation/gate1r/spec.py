"""Gate 1R frozen specification. Written and committed BEFORE the benchmark is run (see GATE1R_SPEC.md).

Every constant below is either a code contract of the recording wrapper, a declared physical prior, or a window choice
made from the development pilot (episodes 0, 4, 8) and stated with its reason. Thresholds that depend on data are
computed by the fold rule `fold_threshold` from development episodes only. Nothing here may be changed after the
benchmark has been run; a change requires a new version, a new holdout and a recorded deviation.
"""

SPEC_VERSION = "gate1r-1.0"

PILOT = (0, 4, 8)                  # development episodes (leave-one-episode-out)
SEALED = (1, 2, 3, 5, 6, 7, 9)      # untouched; may be used ONLY after human GT exists (it does not)

# ---------------------------------------------------------------- A: action-state alignment and feasibility
A = {
    # contracts of OpenArmUmeow.send_action (config_openarm_umeow.py): a request farther than this from the measured
    # state, or a jump larger than this from the previous request, triggers the safety hold. A recorded episode that
    # kept moving cannot contain such requests.
    "max_tracking_error_rad": 0.5,
    "max_command_jump_rad": 0.25,
    "joint_limit_margin_rad": 0.05,   # URDF limits from model/openarm_description.urdf
    # command-response model per arm joint, fitted on development episodes:
    #   s[t+1] - s[t] = alpha_j * (a[t-d] - s[t]) + beta_j * (s[t] - s[t-1])
    "response_lead_candidates": (0, 1, 2, 3),
    "response_floor_rad": 0.004,      # a-priori floor on the per-joint residual RMS threshold (rad / frame)
    # gripper command -> measured jaw response (openness units, 1 = commanded open, 0 = commanded closed)
    "grip_response_frames": 10,
    "grip_min_move": 0.15,
    "grip_uncommanded_move": 0.30,    # a jaw move this large with no command change within +-response frames
}

# ---------------------------------------------------------------- windows (frames; command-defined events)
W = {
    "settle": 10,          # after the giver close command, before the giver attachment is calibrated
    "calib": 15,           # calibration segment length for a hand attachment (fit only here, never on scored frames)
    "min_calib_ok": 12,    # tracker-ok frames required inside a calibration segment
    "recv_offset": 8,      # receiver segment starts this many frames after the giver open command
    "recv_len": 30,        # required receiver evidence: 1 s. Reason: in the pilot the lid leaves the top of the image
                           # 1-2 s after release; a window to the episode end would make every episode UNKNOWN.
    "trans_pre": 10,       # transition window = [receiver close - trans_pre, giver open + trans_post]
    "trans_post": 8,
    "release": (8, 25),    # departure check window after the giver open command
}

# ---------------------------------------------------------------- B/C geometry
G = {
    # observability: the fitted attachment's covariance (Gauss-Newton, sigma^2 = median calibration residual^2) is
    # propagated to the predicted pixels of each scored segment; if the p90 predicted 1-sigma exceeds this fraction of
    # the segment's decision threshold, the segment cannot resolve its threshold -> UNKNOWN. (Revised BEFORE the freeze
    # from a parameter-std rule: the synthetic fixture showed a 15 mm std along the camera ray with 1.5 px noise, which
    # barely moves the projection; parameter std is the wrong quantity.)
    "pred_sigma_max_frac_tau": 0.5,
    "attach_max_norm_m": 0.30,   # declared prior: the lid centre lies within 0.30 m of the holding TCP (can ~0.23 m
                                 # tall plus pads). Beyond it the object is not attached to that hand.
    "min_depth_m": 0.05,         # projected point must be in front of the camera
    "pred_stat": "p90",          # statistic of per-frame prediction residuals in a scored segment
    "run_len": 5,                # consecutive frames above 2*tau also FAIL (short excursions)
    "B_floor_px": 10.0,          # a-priori: the G1 robot-reprojection budget (8 px) plus margin
    "C_floor_px": 12.0,
    "smooth_frames": 3,          # running median over the transition residual (one-frame tracker blips)
    # C (transition, departure) uses attachments fitted on each hand's WHOLE hold (calibration + prediction segments);
    # B uses the calibration segment only. Neither ever fits a frame it scores. (Set BEFORE the freeze: with
    # 15-frame calibrations the transition prediction could not resolve its threshold at 1.5 px tracker noise.)
    "C_attachment_fit": "whole_hold",
}

# ---------------------------------------------------------------- D: evidence
D = {
    "min_coverage": 0.80,        # tracker-ok fraction required in EVERY required segment, else UNKNOWN
    "required_segments": ("giver_calib", "giver_pred", "transition", "recv_calib", "recv_pred", "release"),
}

MARGIN = 1.5                     # tau = max(MARGIN * max over development episodes, floor); for "low" stats tau = min/MARGIN


def fold_threshold(dev_values, floor=0.0, kind="high"):
    v = [x for x in dev_values if x is not None and x == x]
    if not v:
        return None
    return max(MARGIN * max(v), floor) if kind == "high" else min(v) / MARGIN


# ---------------------------------------------------------------- pre-registered falsification criteria (development)
CRITERIA = {
    "K1_originals": "combined verifier: 0/3 original episodes FAIL and <= 1/3 UNKNOWN",
    "K1_hard_negatives": "combined verifier: <= 1 FAIL among the 6 valid hard negatives (10 % frame dropout, 1.5 px noise)",
    "K2_codex_action": "+1 rad on all 14 arm action channels: FAIL in 3/3",
    "K2_codex_excursion": "60 px transition-only excursion: FAIL in 3/3",
    "K2_codex_missing_receiver": "receiver evidence removed: never PASS (UNKNOWN or FAIL) in 3/3",
    "K3_temporal": "video-state offsets of 6 and 10 frames, both directions: FAIL in 3/3 each",
    "K4_action_offset": "all-channel action offsets >= 0.1 rad: FAIL in 3/3 each",
    "K5_no_silent_pass": "no missing-evidence variant whose coverage falls below D.min_coverage is ever PASS",
}
STOP_RULE = ("Evaluated once on the development folds. If any of K1-K5 fails, recommend NO-GO for this verifier line: "
             "no further threshold or window tuning in a new round. If all hold, the result is at most PARTIAL, because "
             "(i) the sealed test requires human ground truth that does not exist, and (ii) absolute contact "
             "certification needs measured geometry and G1 calibration evidence, so it is UNKNOWN by design.")
