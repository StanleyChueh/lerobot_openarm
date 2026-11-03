# Gate 3 readiness: from offline NEXT adapter to contact-free OpenArm data

**Status (Gate 3A, offline):** **GO**. The adapter, the tests and the procedure below are ready. **Nothing in this
document has been executed on hardware.** Each hardware step needs its own explicit approval:

1. **Gate 2 bench:** telemetry OFF / ON / OFF, no contact (§2);
2. **Gate 3B:** contact-free NEXT data collection (§3), only after a Gate-2 report of GO.

Training on real data (Gate 3C, CPU) and every contact block (BATTERY_CONTACT_COLLECTION C/D) are later gates.
They are not covered here.

## 1. Offline gate result (Gate 3A)

| requirement | result |
|---|---|
| worktree `~/Stanley_ws/lerobot_openarm_factr2`, branch `research/factr2-next-openarm`, started clean at `8e43f494`, `f558bc2` an ancestor | verified |
| production code unchanged (follower, wrapper, telemetry, teleop, gains, guards, 16-D interface) | `git diff 8e43f49 -- plugins/lerobot_robot_openarm_umeow plugins/lerobot_teleoperator_openarm_quest robots` is empty |
| official NEXT pinned and audited | `philiphan0109/factr2_next` @ `0df02c5` ([NEXT_OPENARM_ADAPTATION.md](NEXT_OPENARM_ADAPTATION.md) §2) |
| exporter, data contract, mock fixture validation, model smoke | all new suites pass (§5 of the adaptation doc) |
| official parity P0–P4 (official code run on our files) | **not executed**: running downloaded code needs your approval. Open item, not a blocker for hardware data collection |
| existing offline regressions | pass, except `test_quest_safety` P: a diagnosed test-timing race, not a regression (adaptation doc §6) |
| 8e43f49 compatibility | verified with a mock client; no change needed |

## 2. Gate 2 bench: telemetry OFF → ON → OFF (supervised, no contact)

This is [GATE2_HARDWARE_READINESS.md](GATE2_HARDWARE_READINESS.md) §5–§7, unchanged, run in A1-B-A2 order. It is
restated here with the exact pre-checks for this branch.

### 2.1 Before power-on (all must hold; abort criteria in GATE2 §5)

* Physical: workspace clear (reach + 0.3 m), **no battery or charger**, hardware E-stop that cuts motor power,
  tested once with the motors powered. One operator in the headset, one supervisor whose only job is the E-stop.
  Arms hanging at rest at power-on.
* Software, read-only checks:

```bash
cd ~/Stanley_ws/lerobot_openarm                       # the checkout the venv's editable install points at
git status --short                                    # must print nothing
git rev-parse HEAD                                    # 8e43f49... (or a descendant: see the next line)
git diff 8e43f49 HEAD -- plugins/lerobot_robot_openarm_umeow plugins/lerobot_teleoperator_openarm_quest robots   # must be empty
ip -details -statistics link show can0; ip -details -statistics link show can1   # UP, ERROR-ACTIVE, berr 0
pgrep -af "lerobot-(record|teleoperate|rollout)|robot_client" || echo "no other CAN user"
```

* Offline suites green on that checkout (GATE2 §2). Run `test_quest_safety` on an idle machine. Check P may still
  fail by timing (adaptation doc §6). Its second half, "the arm stops (PAUSED)", **must** pass.
* Never: `lerobot-calibrate`, `--robot.type=openarm_follower`, set-zero, motor parameter queries, gain edits.

### 2.2 Runs

Use the exact `run` function of GATE2 §6: three `lerobot-record` runs of 3 × 40 s episodes, free motion within
about 20 cm of home, grippers opened and closed twice in the air, fresh local roots, `--dataset.push_to_hub=false`.

```bash
run bench_A1 ""                                      # telemetry OFF
run bench_B  "--robot.telemetry_dir=$G/telemetry"    # telemetry ON
run bench_A2 ""                                      # telemetry OFF again
```

### 2.3 Analysis (offline, after the session)

```bash
python tools/force_aware/telemetry_check.py $G/telemetry/<session> --dataset $G/bench_B --baseline-dataset $G/bench_A1 \
  --diag outputs/teleop_diag/<B> --baseline-diag outputs/teleop_diag/<A1> --baseline-diag outputs/teleop_diag/<A2> \
  --json $G/gate2_report.json
# NEXT dry run on the bench session (no attestation: shows validity only)
python tools/force_aware/next_export.py $G/telemetry/<session> --out $G/next_dryrun
```

### 2.4 Gate 2 GO / NO-GO (for Gate 3B)

**GO** only if every item holds:

* every **gate** check G1–G8, G10–G13 and B1 is PASS (GATE2 §4). G10 INCONCLUSIVE means the A/B/A runs are
  repeated longer;
* **S1 PASS** (`state_fresh` ≥ 99 %, `stale` ≤ 1 %, `state_age_s` p99 ≤ 5 ms per joint). If S1 FAILs, the strict
  exporter rejects those rows. Continuing then needs a written decision: fix the follower, or approve
  `--allow-unfresh` with the measured age bound;
* **S2 PASS** (status 1, no faults, no holds in free motion) and **S4 PASS** (lost replies ≤ 1 %);
* NEXT dry run: per arm, `valid` ≥ 90 % of the episode rows (the `reject_counts` say why if not), and
  `gains_meta_mismatch` is **empty**. On hardware, transmitted kp/kd must equal `meta.gains`, the follower's
  `KPs`/`KDs` at connect. Today these are [60, 50, 50, 60, 20, 40, 20, 3] / [2, 2, 2, 2.5, 1, 1.2, 1, 0.3] in
  `openarm_follower.py:65-66`; the "joint3 raised to kp=150" comment above them is stale. The tolerance is one
  MIT quantisation step. Any mismatch means the model would learn a different controller than the one
  documented.

**NO-GO:** any gate FAIL, any abort event, any `[openarm_telemetry] NOT recording`, `snapshot failed` or `tap
unavailable` warning, or a gain mismatch.

## 3. Gate 3B: contact-free NEXT data collection (after Gate 2 GO, separate approval)

### 3.1 Design

* **Volume:**
  * the paper trains on about 10 minutes of free motion per robot;
  * target here: **≥ 20 min of valid contact-free rows per arm** (36 000 rows at 30 Hz), over **≥ 12 sessions**,
    so that the session-level split keeps ≥ 4 sessions out for validation.
* **Coverage:**
  * the battery-task workspace, including the meeting region of both hands;
  * three speed bands: slow ≤ 0.2 rad/s, task-like, and fast ≤ `max_joint_speed`;
  * every grip condition the task uses.
* **Payload:** in the task, the left arm holds the charger and the right arm holds the cell. A model trained
  only on empty grippers would report the payload as external effort. So the held-object blocks below are part
  of the free-motion data. Whether to train one model per condition or a pooled model is decided in Gate 3C.
* **One `lerobot-record` run = one robot connect = one telemetry session.** Never collect two blocks in one run.

| block (cf. BATTERY_CONTACT_COLLECTION §4 A) | objects | sessions × episodes × length | attestation |
|---|---|---|---|
| N-A1 | empty grippers **open** | 3 × 4 × 45 s | `--contact-free-all` (workspace empty for the whole session) |
| N-A2 | grippers **closed** on nothing (squeeze active) | 3 × 4 × 45 s | `--contact-free-all` |
| N-A3 | left holds the **unpowered** charger, right holds the **dummy AA**; never touching each other or anything else | 3 × 4 × 45 s | `--contact-free-episodes` (the objects are handed over between episodes, which is contact) |
| N-A4 | as N-A3, along the real insertion approach, **stopping 2–3 cm short** of the bay | 3 × 5 × 30 s | `--contact-free-episodes` |

That is about 32 min recorded, with margin for rejected rows. Use the masses, the dummy cell and the unpowered
charger from BATTERY_CONTACT_COLLECTION §3 (instrument safety). Weigh both objects and log them in `block.json`.

### 3.2 Exact commands (per session)

Run on the Gate-2 GO commit from `~/Stanley_ws/lerobot_openarm` with the same §2.1 checks. Use `CAMS` from
GATE2 §6.

```bash
D=$HOME/force_aware_data/next_$(date +%Y%m%d); mkdir -p $D
blk() {   # $1 = block id (N-A1..N-A4), $2 = session no., $3 = episodes, $4 = episode seconds, $5 = task text
  R=$D/$1_s$2; test ! -e $R || { echo "$R exists, refusing"; return 1; }
  lerobot-record \
    --robot.type=openarm_umeow --robot.right_port=can0 --robot.left_port=can1 --robot.cameras="$CAMS" \
    --robot.telemetry_dir=$D/telemetry \
    --teleop.type=openarm_quest --teleop.diag=true \
    --dataset.repo_id=local/openarm_next_${1//-/_}_s$2 --dataset.root=$R --dataset.push_to_hub=false \
    --dataset.single_task="$5" --dataset.num_episodes=$3 --dataset.fps=30 \
    --dataset.episode_time_s=$4 --dataset.reset_time_s=20 \
    --dataset.streaming_encoding=true --dataset.encoder_threads=2 --display_data=true
}
blk N-A1 1 4 45 "NEXT free motion: empty grippers open, no contact"
# ... N-A1 s2, s3; N-A2 s1..s3 ("grippers closed, no contact"); N-A3 s1..s3 ("holding charger and dummy cell, no contact");
#     N-A4 s1..s3 with 5 x 30 s ("insertion approach, stop 2-3 cm short, no contact")
```

Operator rules:

* Any episode with a touch (objects to each other, table, cables, a person) is **discarded with Y**. It never
  enters the dataset, so `--contact-free-episodes` cannot attest it.
* In N-A1/N-A2 nobody enters the workspace during the run. If someone does, the session is exported with
  `--contact-free-episodes` instead of `--contact-free-all`.
* Write `$R/block.json`: block, objects and masses, operator, supervisor, telemetry session folder name, any
  deviation.

### 3.3 Export (offline, after each session)

```bash
S=$D/telemetry/<session folder of that run>
python tools/force_aware/telemetry_check.py $S --dataset $R --json $R/gate_check.json   # G1-G8, G11-G13, S1-S5 again
python tools/force_aware/next_export.py $S --out $D/next_export --contact-free-all \
    --attested-by "<operator>; body-cam video of $R reviewed by <second person>" --block N-A1     # N-A1 / N-A2
python tools/force_aware/next_export.py $S --out $D/next_export --contact-free-episodes $R \
    --attested-by "<operator>; body-cam video of $R reviewed by <second person>" --block N-A3     # N-A3 / N-A4
```

A second person reviews the body-cam video of every kept episode before the export. The dataset's own videos
are the evidence; no new hardware is needed.

### 3.4 Gate 3B GO / NO-GO (data ready for Gate 3C training, which is a separate approval)

**GO** when all hold:

* each session's `telemetry_check` has every gate check PASS, S1 PASS (or the approved mitigation of §2.4), and
  S2/S4 PASS. The exception is G10 (on/off timing), which is UNKNOWN without telemetry-off runs (exit status 3);
  Gate 2 settled it for this commit;
* per arm, ≥ 20 min of `contact_free_valid` rows in total, and every block represented;
* all exports share one `hardware_signature` (`check_compatible` passes), with `mock` false and
  `gains_meta_mismatch` empty;
* the planned split is written down **before** training: validation = one whole session per block (4 sessions),
  test sessions for the contact study kept separate;
* all speed bands reached, checked from `dq` in the exports, not from memory.

**NO-GO:**

* any abort event, fault or hold during a session (that session is excluded and re-collected);
* a hardware-signature split across sessions (for example a recalibration or gain change mid-campaign);
* a review that finds contact in an attested episode (the session's attestation is withdrawn and the export
  redone).

## 4. What Gate 3C will need to decide (not here)

* H = 30 at 30 Hz versus other histories, `dq` versus a finite difference of `q`, and payload as one model per
  condition or pooled. Selection uses validation sessions only.
* Report NEXT against the R0/R1 baselines of BATTERY_CONTACT_COLLECTION §2 on contact blocks C/D, using the
  pre-registered criteria of §6 there.
* `tau` remains uncalibrated. Newton figures need block B3 and the load cell.
* Training is CPU-only (`next_model.py` enforces it). The official parity test P0–P4 should be run once before
  any training result is reported.

## 5. Do not

* Run any of §2–§3 without explicit approval for that session.
* Train on, or tune against, telemetry from sessions that lack a Gate-2-quality check.
* Attest contact-free from memory: attestation needs `--attested-by` with the evidence named.
* Mix sessions with different hardware signatures, or mock and real data.
* Use J8 (grippers) in an arm model, or call `tau_ext_hat` a torque or a force.
* Upload any dataset or export.
