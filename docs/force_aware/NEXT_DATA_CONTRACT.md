# NEXT data contract for OpenArm (`openarm_next_export/1`)

**Producer:** `tools/force_aware/next_export.py`. **Consumer:** `tools/force_aware/next_model.py`, or the
official FACTR2 trainer through the H5 view (§7).

**Input:** `openarm_telemetry/1` sessions ([TELEMETRY_SCHEMA.md](TELEMETRY_SCHEMA.md)). Any other schema string
is refused.

Background and decisions: [NEXT_OPENARM_ADAPTATION.md](NEXT_OPENARM_ADAPTATION.md).

## 1. Model I/O (per arm, one model each)

| symbol | source column (row k) | unit | shape |
|---|---|---|---|
| `q_k` | `<J>.q` | rad, motor-side angle, the motors' stored zero (not URDF; same `calibration.json` required) | 7 |
| `dq_k` | `<J>.dq` | rad/s, motor-reported | 7 |
| `cmd_q_k` | `<J>.cmd_q` | rad, the MIT position target **transmitted** and in effect before row k's measurement | 7 |
| `tau_k` | `<J>.tau` | **N·m nominal, uncalibrated** DM firmware current-based effort | 7 |

* **Joints:** `J` = `LJ1..LJ7` (left model) or `RJ1..RJ7` (right model), proximal to distal, **selected by name**.
  The telemetry order (L then R) and the LeRobot dataset order (interleaved `RJ1, LJ1, …`) are never used as
  indices.
* **Features:** `x_k = [q_k, dq_k, cmd_q_k − q_k]`. The three inputs are cast to float32 first, then the
  difference is taken, exactly as the official dataset does. The feature index is block-major: `[0:7]` q,
  `[7:14]` dq, `[14:21]` cmd − q.
* **Window:** rows `k−H+1 … k`, all from **one segment** (§5). Default H = 30 (1.0 s at 30 Hz).
* **Target:** `tau_k`, at the window's last row. Not a future row.
* **Output:** `tau_free_hat_k`, the predicted effort of contact-free motion.
* **Residual:** `tau_ext_hat_k = tau_k − tau_free_hat_k`. It is an *effort residual* in nominal N·m. Do not call
  it torque or force while `provenance.torque_calibration` is null.

## 2. Causality

Every value used for row k was sampled no later than row k's snapshot (`t`).

* `cmd_q_k` was transmitted at `t − cmd_age_s ≤ t`.
* The measurement was acquired at `t − state_age_s`.
* A valid row requires `state_age_s ≤ cmd_age_s`: the measurement came **after** the command it is paired with.
* Row k+1 (which holds action k's command) is never read for row k. It may only serve as a supervision target
  in other studies, never as a NEXT input.
* The one session-level quantity is the reference gain pair (§4, `gain_change`). It only rejects rows and never
  changes a value.

## 3. Export layout (`<export root>/<session id>/`)

| file | rows | columns |
|---|---|---|
| `left_arm.parquet`, `right_arm.parquet` | one per telemetry row, same `tick` | `tick, kind, t, t_wall` · `valid` (bool) · `reject` (int64 bitmask, §4) · `segment` (int64, −1 = invalid) · `contact_free` (bool, attestation, independent of `valid`) · `t_acquired` = `t − max_j state_age_s` (NaN if unknown) · `hold` · for each of the 7 joints, verbatim float64: `q, dq, tau, cmd_q, cmd_dq, cmd_kp, cmd_kd, cmd_tau, sent, desired, state_age_s, state_fresh, stale, cmd_n, cmd_age_s, status, rx_n` |
| `left_gripper.parquet`, `right_gripper.parquet` | same | the same row fields for J8 alone, plus `squeeze_tau` |
| `manifest.json` | – | `schema`; `source` (path, session id, sha256 of `meta.json` and each part, `closed`, `errors`); `settings`; `contact_free` (mode, intervals, attestation); `hardware_signature`; `gains_meta_mismatch`; `effort` (`calibrated`: false); `arm_joints`, `gripper_joints`; per arm/gripper: `rows`, `valid`, `contact_free_valid`, `reject_counts`, `segments`, `segment_lengths`; `reject_bits` |

Exported values are bit-identical to the source (test N3). NaN stays NaN. Nothing is filled, interpolated or
resampled. The source folder is never written. Existing exports are never overwritten.

## 4. Row validity (per arm: all 7 joints must pass; the gripper is checked on J8 alone)

| bit | reason | rejected when | why |
|---|---|---|---|
| 1 | `final_row` | `kind` ≠ 0 | after disconnect; not an observation |
| 2 | `snapshot_failed` | `snapshot_ok` ≠ 1 | joint columns are NaN |
| 4 | `tap_down` | `<side>.tap_ok` ≠ 1 | freshness unknown |
| 8 | `nan` | any of `q, dq, tau, cmd_q, cmd_dq, cmd_kp, cmd_kd, cmd_tau, state_age_s, cmd_age_s` is NaN | unknown is never assumed |
| 16 | `stale` | `stale` ≠ 0 (NaN included) | value too old or untraceable |
| 32 | `not_fresh` | `state_fresh` ≠ 1 (NaN included); turn off only with `--allow-unfresh` | value does not answer this tick's refresh (it is the previous command's reply) |
| 64 | `state_age` | `state_age_s` > `meta.stale_s` (0.05 s) | bounded age, even when fresh |
| 128 | `no_command` | no MIT command yet | no `cmd_q` |
| 256 | `cmd_count` | `cmd_n` ≠ 1 | not the one-command-per-tick loop (start ramp backlog, skipped command, extra read) |
| 512 | `cmd_age` | `cmd_age_s` > 1.5 periods | the command in effect is not this tick's (first row after a pause) |
| 1024 | `causal` | `state_age_s` > `cmd_age_s` | the measurement predates its command |
| 2048 | `hold` | `hold` ≠ 0 | frozen re-send. A hold is triggered by large tracking error, which can itself be contact |
| 4096 | `motor_fault` | `motor_fault` ≠ 0 | wrapper watchdog |
| 8192 | `status` | DM status nibble ≠ 1 | not enabled, or a fault code |
| 16384 | `saturated` | \|tau\| ≥ tMax − LSB, \|dq\| ≥ vMax − LSB, \|cmd_q\| ≥ pMax − LSB or \|cmd_tau\| ≥ tMax − LSB (`meta.decode_limits`) | clipped values carry no information |
| 32768 | `gain_change` | `cmd_kp`/`cmd_kd` differ from the session's most common transmitted pair for that joint | the free-motion effort depends on the gains |
| 65536 | `ctrl_frame` | `<side>.ctrl_frames` ≠ 0 | enable/disable/set-zero frame on this arm |
| 131072 | `bus` | `<side>.rx_dropped` grew since the previous row (NaN = unknown), or `bus_errors` ≠ 0 | lost frames break the freshness trace |
| 262144 | `clock` | `t` not strictly increasing | spacing undefined |

Row-level faults (`snapshot_failed`, `hold`, `motor_fault`, `clock`) reject both arms. Side and joint faults
reject only the affected arm.

## 5. Segments

A **segment** is a maximal run of valid rows with consecutive `tick` and spacing `t[k] − t[k−1]` within
[0.5, 1.5] nominal periods (fps = 30 by default, `--fps`).

* Anything else ends the segment: an invalid row, a tick gap (lost write), a late tick, or an extra read.
* Windows never cross a segment boundary.
* For training, the rows are first intersected with `contact_free`, then re-segmented.
* The streaming estimator clears its buffer on the same conditions, so batch and streaming output are identical
  (test M5).

## 6. Contact-free attestation and splits

* **Attestation:** nothing is contact-free by default. Choose exactly one mode:
  * `--contact-free-all`: the whole session;
  * `--contact-free-intervals`: a JSON list of `{t_wall_start, t_wall_end, block}`;
  * `--contact-free-episodes <local dataset root>`: the kept episodes of the dataset recorded in that session,
    located by bit-exact value alignment (TELEMETRY_SCHEMA §7). An episode matching zero or several windows is
    an error. Resets and pauses between episodes stay unattested.

  Every mode also needs `--attested-by` (who, and on what evidence: video reviewed, protocol block). The
  attestation is stored in the manifest. Training and the H5 view refuse sessions without contact-free valid
  rows.
* **Split:** by session (telemetry folder). `split_sessions()` takes explicit validation sessions, or a
  deterministic sha256 order with `ceil(val_fraction · n)` validation sessions. No session is on both sides;
  neither side may be empty.
* **Normalization:** official formula, fitted on training sessions only.
* **Hardware signature:** `motor_types`, `decode_limits`, transmitted gains, `calibration_sha256`,
  `urdf_sha256`, `openarm_can`, `torque_calibration`, `gripper_squeeze_tau_config`, `max_joint_speed`, `mock`.
  * All sessions of one model must be identical in these. Mock and real sessions never mix.
  * Estimation refuses a session whose signature differs from the model's (override:
    `allow_signature_change`, analysis only).

## 7. FACTR2 H5 view (`factr2_next_h5_v1`)

`next_export.py --h5 <exports...> --arm <left|right> --out <file>` writes the official recorder's layout:

* one `ep_XXXX` per contact-free segment;
* keys `<arm>_joint_pos`, `<arm>_joint_vel`, `<arm>_joint_cmd`, `<arm>_measured_joint_torque`, each with
  `data` (n × 7, float32) and `timestamps` (int64 ns, from `t_wall`);
* episode attributes `session_id`, `segment`, `first_tick`, `rows`;
* file attribute `openarm_next_export` holding provenance and the signature.

Write one file per arm and per split. The official `train.yaml` then needs `arm_mode: <arm>`, keys
`"{arm}_joint_pos"` and so on, and `history: 30`.

## 8. Run directory (`next_model.py train`)

The official layout:

* `model.pt`: `model_state_dict`, `input_size` 21, `output_size` 7, `history`, `model`;
* `config.yaml`;
* `normalization.npz`: `x_mean`, `x_std` (21), `y_mean`, `y_std` (7);
* `metrics.json`.

Plus `openarm.json`: official repository and commit, arm, joints, history, fps, train and validation session
ids, hardware signature, and target and input definitions.

The module names (`lstm`, `head`) and shapes match the official `LSTMRegressor`. Its `load_checkpoint` reads
these runs. The parity test exists but was not executed in Gate 3A.

## 9. Grippers

J8 has its own tables and validity and is never part of an arm model. Its effort includes the squeeze
feed-forward (`squeeze_tau`, also in J8 `cmd_tau`). A squeezing gripper is a valid gripper row: the squeeze is
a known input, not a fault. Modelling J8 is out of scope for Gate 3A.
