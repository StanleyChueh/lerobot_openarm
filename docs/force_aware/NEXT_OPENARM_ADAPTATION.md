# Gate 3A: adapting FACTR2 / NEXT to bimanual OpenArm v1.0

**Branch:** `research/factr2-next-openarm`, from `8e43f49` (which descends from Gate 2A `f558bc2`). The commit SHA is
in the delivery note. **Status:** offline gate only. Nothing connected to a robot or opened a CAN socket. Nothing
moved a motor, calibrated, changed a gain, ran on a GPU or uploaded a dataset.

**Question:** can NEXT estimate contact-related external joint effort for battery insertion (left arm holds the
charger, right arm inserts the cell) from the OpenArm telemetry we already log?

**Answer so far:** the pipeline exists and is verified on mock data. Whether the *estimate* is useful is
**UNKNOWN**: no real telemetry exists yet. That needs Gate 2 (bench) and then contact-free collection
([GATE3_READINESS.md](GATE3_READINESS.md)).

## 1. What was delivered

| item | file |
|---|---|
| telemetry → NEXT exporter (no ROS; numpy + pandas + pyarrow, h5py only for the FACTR2 H5 view) | `tools/force_aware/next_export.py` |
| NEXT windows, normalization, official-architecture LSTM, CPU training, residual estimation, streaming estimator | `tools/force_aware/next_model.py` |
| adapter tests on the committed Gate-2A fixture plus 20 corrupted copies | `plugins/tests/test_next_export.py` |
| mock-only model smoke tests, plus optional parity against the pinned official code | `plugins/tests/test_next_model.py` |
| 8e43f49 compatibility: robot reads per tick under background observations | `plugins/tests/test_eval_obs_reuse.py` |
| data contract | [NEXT_DATA_CONTRACT.md](NEXT_DATA_CONTRACT.md) |
| hardware readiness, GO / NO-GO | [GATE3_READINESS.md](GATE3_READINESS.md) |

No production file changed. That covers the follower, `openarm_umeow.py`, `telemetry.py`, the gains, the guards, the
teleop and the 16-D interface. Check it with:

```bash
git diff 8e43f49 -- plugins/lerobot_robot_openarm_umeow plugins/lerobot_teleoperator_openarm_quest robots   # empty
```

## 2. Official NEXT, audited

**Source:**

* Repository: `https://github.com/philiphan0109/factr2_next`, pinned at
  **`0df02c5b26e33911078152b6c587d6e8a160d279`** (2026-09-04, "Update README.md"), Apache-2.0. It is cloned
  outside this repository and is not vendored. Our code reimplements the parts we need.
* Paper: arXiv 2606.12406, *FACTR 2: Learning External Force Sensing for Commodity Robot Arms Improves Policy
  Learning* (v2, 2026-08-12). The abstract says NEXT trains in about 1 minute on about 10 minutes of free-motion
  data. Equation and appendix references below are as cited in the code's comments. The PDF itself was not
  reviewed.

| part | official behaviour (file) | consequence for OpenArm |
|---|---|---|
| robot | AgileX Piper, 6 arm joints. Gripper on separate topics, **not** in NEXT (`piper_control/piper_node.py`) | OpenArm: J1–J7 per arm. J8 kept out (§4.5) |
| effort | `GetArmHighSpdInfoMsgs().motor_i.effort / 1000`, the driver's reported effort, not a torque sensor | same situation: DM firmware `tau`, uncalibrated |
| command | `joint_pos_cmd` topic: the teleop target published *to* the driver, before any driver-side limiting | we use the **transmitted** MIT target `cmd_q`, after the step limit and quantisation (§4.2) |
| rate | recorder timer 50 Hz; 4 JointState topics synced by `ApproximateTimeSynchronizer` (slop 0.03 s) | OpenArm logs at the 30 Hz control tick |
| H5 | `ep_XXXX/<key>/{data, timestamps}`, schema `factr2_next_h5_v1` (`data_collection/h5_writer.py`) | reproduced as an optional view (§3.2) |
| windows | `x = [q, dq, q_cmd − q]` in float32; `history` = 50 consecutive rows **per episode**; `y = tau_m` at the window's **last** row; episodes shorter than `history` are skipped (`training/dataset.py`) | identical, but per **segment** of valid rows, not per episode |
| validity | none: rows are whatever the synchronizer produced. No freshness, gap or saturation handling. Episodes = recording toggles | the main addition here (§4.3) |
| split | separate val H5 files, else the last 10 % of *windows* with a 50-window buffer (or random windows) (`training/train.py`) | replaced by a **session-level** split |
| normalization | mean/std over training windows × time (+1e-6), training data only | identical |
| model | stateless LSTM, 2 × 128, unidirectional, dropout 0.1; head Linear(128→256)-ReLU-Dropout-Linear(256→7); output from the last step. MLP/GRU alternatives exist (`training/models.py`) | identical module names and shapes, so the state dicts are interchangeable |
| training | Adam 1e-3, batch 2048, 20 epochs, MSE, CPU default, seed 0; `arm_mode: both` trains one model per arm | identical defaults; CPU enforced |
| checkpoint | `model.pt` (`model_state_dict`, `input_size`, `output_size`, `history`, `model`), `config.yaml`, `normalization.npz`, `metrics.json` | identical layout, plus `openarm.json` (provenance) |
| inference | `HistoryBuffer` deque of H rows; predict once full; `tau_ext_raw = measured − pred`; then EMA (α 0.2), an L1 "contact magnitude" and hysteresis 1.0/2.0. The README says these post-processing steps are runtime/demo additions, not part of the paper's experiments (`inference/inference_node.py`) | raw residual only. Smoothing and thresholds are left to the contact study. Our buffer is **cleared** on any invalid row or gap; the official one never resets |

The contract we reproduce:

```
x_k              = [q_k, dq_k, cmd_q_k − q_k]                     (21 = 3 × 7, float32)
tau_free_hat_k   = f_theta(x_{k−H+1 .. k})                         (stateless LSTM)
training target  = tau_k  on verified contact-free rows only
tau_ext_hat_k    = tau_k − tau_free_hat_k                          (uncalibrated effort residual)
```

## 3. How it is adapted

### 3.1 Export (per telemetry session)

```bash
python tools/force_aware/next_export.py <telemetry session> --out <export root> \
    --contact-free-all --attested-by "<who>, <how verified: video file / protocol block>" --block A1
```

The export writes `<export root>/<session id>/`:

* `left_arm.parquet` and `right_arm.parquet`: one row per telemetry row. Each holds the source columns of J1–J7
  **verbatim** (joined by name), plus `valid`, `reject` (a bitmask of reasons), `segment`, `contact_free` and
  `t_acquired`.
* `left_gripper.parquet` and `right_gripper.parquet`: J8 with the same rules, plus `squeeze_tau`.
* `manifest.json`: the source sha256s, the settings, the attestation, the hardware signature, rejection counts
  per reason and the segment lengths.

The source folder is read-only to the exporter. It refuses to write inside it and never overwrites an
export.

### 3.2 Training and estimation

```bash
python tools/force_aware/next_model.py train --arm left --exports <exp>/<s1> ... --val-sessions <sX> --out <runs>
python tools/force_aware/next_model.py estimate --run <run> --export <exp>/<s> --arm left --out residual.parquet
# optional: the official trainer's H5 input, one ep_XXXX per contact-free segment
python tools/force_aware/next_export.py --h5 <exp>/<s1> <exp>/<s2> --arm left --out next_left_train.h5
```

Training is CPU-only by construction (`device != cpu` raises). Gate 3A ran it only on synthetic mock
sessions, for a few seconds.

## 4. Decisions, and why

### 4.1 Separate left and right models, arm joints only

NEXT is arm-specific: the official bimanual mode trains one model per arm. Here the two arms are also on
separate CAN interfaces, with separate validity. A row can be valid for the left arm and invalid for the right
(fixture tick 81: RJ5 stale). So each arm has its own table, validity, segments and model.

### 4.2 `q_cmd` = the transmitted MIT target in effect before the measurement (`cmd_q` on the same row)

Telemetry offers three candidates:

* `desired`: the policy/teleop request;
* `sent`: after the wrapper's step limit;
* `cmd_q`: decoded from the MIT frame on the wire, after openarm_can's clamping and 16-bit quantisation.

The motor's PD law acts on `cmd_q`:

```
tau ≈ kp·(cmd_q − q) + kd·(cmd_dq − dq) + cmd_tau
```

So `cmd_q − q` is the tracking error the effort answers. On row k, `cmd_q` is the command in effect **before**
row k's measurement (TELEMETRY_SCHEMA §2). It is causal, and it is the command that produced row k's `tau`.
Two checks back this up:

* `cmd_q` equals `sent` within 1 LSB on valid rows (test N6);
* each valid row requires the measurement to have been acquired **after** that command was sent
  (`state_age_s ≤ cmd_age_s`; reject reason `causal`).

Row k + 1 is never read for row k: test N6 scrambles every later row and the earlier export does not change.

This differs from the official Piper setup, which logs the teleop target. On OpenArm the step limit makes
`desired` and `cmd_q` differ during fast motion. Using `desired` would put the step limiter's behaviour into
the model's input.

### 4.3 Validity: reject, never repair

A row is valid for an arm only if all 7 joints pass every rule in
[NEXT_DATA_CONTRACT.md](NEXT_DATA_CONTRACT.md) §4. The rules cover freshness, staleness, the tap, NaN,
saturation, command count and age, causality, hold, faults, status, gain changes, control frames, bus loss and
the clock.

The defaults are strict:

* `state_fresh` must be 1. The value must answer *this* tick's refresh, not the previous command's reply. The
  Gate-2 check S1 measures how often that holds on hardware.
* `stale` = NaN or `state_fresh` = NaN means unknown, so the row is rejected.

Repeated values from a silent motor are rejected, never interpolated (test N9). Valid rows form **segments**:
consecutive ticks whose spacing is 0.5–1.5 nominal periods. No window crosses a segment boundary. A late tick,
a lost write or an extra read starts a new segment (N10, N15).

### 4.4 Hardware signature

A NEXT model learns the effort of one specific controller. It is valid only for the same motor types, decode
limits, **transmitted** kp/kd, calibration (zero offsets), URDF (the `cmd_tau` gravity model) and openarm_can
version.

* The exporter records these in each manifest.
* Training refuses to mix signatures, and estimation refuses a session whose signature differs from the
  model's.
* Transmitted gains are compared with `meta.gains`, and any mismatch is reported. The mock fixture shows one:
  its fake bus transmits kp 49.94 on every joint while `meta.gains` says [60, 50, 50, 60, 20, 40, 20, 3]. On
  hardware, any mismatch is a NO-GO finding ([GATE3_READINESS.md](GATE3_READINESS.md) §3).

### 4.5 Grippers

J8 is never an input or a target of the arm models. Its `cmd_tau` includes the squeeze feed-forward, and a
squeeze closes a force loop through the held object (BATTERY_CONTACT_COLLECTION §2). The exporter writes J8 to
separate gripper tables with the same validity rules plus `squeeze_tau`, so a 1-D gripper model can be studied
later. No gripper model is trained or proposed in Gate 3A.

### 4.6 Session-level splits

Windows from one session are strongly correlated, and the official within-file split puts neighbouring windows
on both sides. Here a **session** (one robot connect = one telemetry folder) goes entirely to train or to
validation:

* explicit validation sessions, or a deterministic hash order;
* normalization is fitted on training sessions only;
* a session on both sides, or an empty side, is refused (N13, M2).

Grouping by protocol block (A1–A4) is recommended on top ([GATE3_READINESS.md](GATE3_READINESS.md) §4).

### 4.7 Rate and history

OpenArm logs at 30 Hz, and the official history is 50 rows at 50 Hz = 1.0 s. The default here is
**H = 30 (1.0 s)**. This is a hypothesis to test, not a result. Also note that `dq` has coarse 12-bit resolution
(0.022 rad/s on J1–J2). Whether `dq` or a finite difference of `q` is the better velocity input is an open
question, and it is left untouched.

### 4.8 What NEXT does *not* give us here

* **Not a joint torque.** `tau` is the firmware's current-based estimate with an unknown scale per joint
  (`provenance.torque_calibration` = null). `tau_ext_hat` is an *effort residual* in nominal N·m. Turning it
  into torque or force needs the known-mass block B3 and the load cell (BATTERY_CONTACT_COLLECTION §3, §6).
* **Holding objects is not "contact-free" in the effort sense.** In the task, the left arm carries the charger
  and the right arm carries the cell. A model trained only with empty grippers will report the payload as
  external effort. Free-motion data must therefore include held-object blocks (A3/A4), or a separate model per
  grip condition. This is a design choice for Gate 3B, not settled here.
* **Internal insertion force.** It appears in both arms with opposite sign through each arm's Jacobian
  (BATTERY_CONTACT_COLLECTION §1). The bimanual consistency check belongs to the contact analysis, not to
  NEXT itself.

## 5. Offline results (this commit)

All runs used the shared venv against **this worktree's** plugin code:

```bash
PYTHONPATH=$PWD/plugins/lerobot_robot_openarm_umeow:$PWD/plugins/lerobot_teleoperator_openarm_quest
```

The venv's editable install points at `~/Stanley_ws/lerobot_openarm`, so the override is needed.

| suite | result |
|---|---|
| `test_next_export.py` (new) | 49/49 PASS. A mutation check (cmd-age rule off, freshness off, spacing band off, NaN-as-fresh) makes it fail each time |
| `test_next_model.py` (new) | 16/16 PASS. Official parity P0–P4: **SKIP** (see below) |
| `test_eval_obs_reuse.py` (new) | 5/5 PASS |
| `test_telemetry_fixture.py` / `test_telemetry.py` / `test_telemetry_record.py` | 9/9, 37/37, 10/10 PASS |
| `test_robot_guard.py` / `test_return_lag.py` / `test_record_gate.py` / `test_rerun_status.py` | 10/10, 3/3, 9/9, 4/4 PASS |
| `test_teleoperate_mock.py`, `test_record_mock.py` (no PASS lines; exit 0) | PASS |
| `test_quest_safety.py` | 43/44. Check P fails: a test-timing race, diagnosed in §6 |

**Official parity not executed.** `test_next_model.py` P0–P4 import the pinned official package from
`FACTR2_NEXT_SRC` and check:

* that its `NextTorqueDataset` yields our windows bit-for-bit;
* that its `fit_normalization` equals ours;
* that its `LSTMRegressor` loads our state dict with identical outputs;
* that its `load_checkpoint` + `HistoryBuffer` reproduce our residual.

Running downloaded third-party code was not authorized in this session, so these checks print SKIP. The
architecture and checkpoint layout were matched by reading the source. To run them:

```bash
git clone https://github.com/philiphan0109/factr2_next <dir> && git -C <dir> checkout 0df02c5
FACTR2_NEXT_SRC=<dir>/src/factr2_next python plugins/tests/test_next_model.py <scratch>
```

**Mock smoke result:**

* setup: 4 training and 1 validation synthetic session, H = 10, LSTM 32 hidden, 40 epochs on CPU, about 3 s;
* validation loss fell from 0.63 to 0.09 (normalized);
* an injected +2.0 effort on RJ4, in a session never seen in training, came out with a residual mean of 1.86,
  against a 0.15 median off-window.

This shows only that the pipeline is wired correctly. **It says nothing about OpenArm.**

## 6. Quest safety check P: diagnosed with mocks

`test_quest_safety.py` check **P "out of reach: says so"** failed in 7 of 8 runs on record (3/3 on 2026-10-29; 4 of 5 here, at load average 1.2–1.9). The last Gate 2A note said
the cause was UNKNOWN and suspected machine load. That is not the cause.

Evidence (the mock Quest, the real `QuestIKDriver`, every IK solve logged, `kin.solve` and `_check_reach`
wrapped in a scratch copy of the test; the test itself is unchanged):

* The reach notice is display-only. It fires when the end-effector gap is > 3 cm on a reach check that runs
  every 0.2 s, and the gap has lasted > 0.5 s. So it fires 0.6–0.8 s after the gap crosses 3 cm, depending on
  the phase of the 0.2 s check grid.
* In the full test, the earlier scenarios leave the headset yawed 15° and the hands displaced. The 1 m push
  then crosses 3 cm at **+2.49 s**, and the IK solution jumps > 0.2 rad (the safety pause) at **+3.27 s**, only
  0.78 s later.
* The check-grid phase is fixed by the test's own sleeps, and it lands within about 10 ms of the crossing.
  * run A: a check at +2.493 s saw the gap, so the notice came at +3.10 s → PASS;
  * run B: the check was at +2.484 s, 9 ms early, so the notice was due at +3.29 s, after the jump → FAIL.
  * This knife-edge explains the mostly-fail record.
* In isolation (fresh start, no headset yaw) the margin is 1.3 s and P passes.
* The same test with the push slowed from 4 s to 6 s (same path) gives a 1.17 s margin. It passed **3/3, 44/44**.
* The safety half of P ("the arm stops, PAUSED") passes in every run.
* No package (mink, mujoco, openarm_control) or IK scene file changed after 2026-10-24. `test_quest_safety.py`
  is unchanged since 2026-10-08 and `ik_driver.py` since 2026-10-21.

**Conclusion:**

* The failure is a timing-margin defect in the test, not a regression and not load-dependent in any way that
  matters.
* Production teleop is unaffected.
* Suggested fix (test only, not applied here, because it changes a safety test): in `plugins/tests/test_quest_safety.py`,
  `move("rc", 2, -1.00, 4.0)` → `move("rc", 2, -1.00, 6.0)`.

## 7. 8e43f49 (evaluation at 30 Hz) and telemetry

8e43f49 changes only the async eval client (`robot_client.py`), the recorder hook and the eval summary. It does
not touch the follower, the telemetry, the guards or the 16-D interface.

What it changes for telemetry, which writes one row per `get_observation()`, verified with a mock client
(`test_eval_obs_reuse.py`):

* With `--record_repo_id` and background observations, the recorder reads the robot **before** the tick's
  action, and the policy reuses that reading. The result is **one row per tick**. Before 8e43f49 there were two.
  Robot reads and actions stay on the control thread. The sender thread never touches the robot (E1).
* If a tick's action takes longer than `environment_dt`, the observation is read **again** a few ms after the
  action, giving two rows that tick (E2). The exporter splits the extra row off (spacing < 0.5 period). The
  next tick saw no new command (`cmd_n` = 0) and is rejected, so no window spans the irregularity (N15).
* Without recording, lerobot's client reads the robot only when it sends an observation. Async-eval telemetry
  is then sparse, as it was before 8e43f49.

**NEXT free-motion data must come from `lerobot-record` (teleop)**, where every tick reads the robot. Eval
sessions are usable only for later residual analysis.

## 8. Unknowns

| unknown | resolved by |
|---|---|
| does the official code accept our H5, normalization and checkpoint bit-for-bit? (P0–P4) | running the parity test with the pinned clone (needs approval to execute it) |
| fraction of rows that are valid on hardware (`state_fresh`, `stale`, `cmd_n`) | Gate 2 report S1, G5; then the export's `reject_counts` |
| `tau` scale and noise floor per joint | block B, B3 |
| whether a 30 Hz, H = 30 LSTM predicts OpenArm free-motion effort well enough | Gate 3B data, then a CPU training run (separate approval) |
| payload: one model per grip condition, or payload in the training data? | Gate 3B design review |
| does NEXT beat R0/R1 (PD effort, quasi-static model) on contact detection? | BATTERY_CONTACT_COLLECTION §6 criteria |
