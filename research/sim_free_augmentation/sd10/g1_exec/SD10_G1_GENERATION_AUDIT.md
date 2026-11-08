# SD10-G1 generation audit (implementer's self-audit; independent Codex review pending)

2026-11-08. Scope: what generator ran, what was adapted, how conformance to the sealed contract was verified, and
every deviation. This is **not** an independent verification.

## 1. Existing generator, located and reused unchanged

| Stage | Frozen component (hash-verified against `locked_generation_config.json`) | Use in G1 |
|---|---|---|
| Source loader | `sfa/data.py` (ten-only snapshot) | **Not called.** Replaced by `g1lib.PoolData`, injected as the frozen recon's `data` |
| Source selection | `locked_targets.json` assignments | Recomputed per attempt from locked initial-lid pixels; identical for all 300 rows; no runner-up |
| Segmentation / tracking | `contact_aware_augmentation/gate1/lid_tracker.py` (HSV lid tracker from verified pool click); `sfa/segment.py` (SAM2.1-small video masks, bf16) | Unchanged, new pool-keyed cold cache |
| Object pose / attachment | `v2/sfa2/recon.py` (joint closure-constrained attachment, standing can on lid ray); `v3/sfa3/recon3.py` closure-only receiver fallback | Unchanged; `rc.estimate_lag` wrapped to the registered fixed lag −1.625 (rms curve kept); fixed inherited camera |
| FK/IK | `sfa/robot.py` (`Robot.ik` DLS, 80 iterations, damping 1e-3, margin 0.02) via `planner3.ik_follow` | Unchanged |
| Trajectory construction | `v3/sfa3/planner3.py` G+ waypoint interpolation | Unchanged; spec from `make_spec(var={})` plus the registered initial-pose waypoints |
| Object–gripper constraints | planner3 holder bookkeeping (table → giver `X_R` → receiver `X_L`, Γ=I) | Unchanged |
| Action labels | `planner3.actions` (`q[t+L] + a_src[t] − q_src[t+L]`, gripper ZOH) | Unchanged; export `[3,T−6)` |
| RGB synthesis | `v3/sfa3/render3.py` layered homography renderer | **Not run** (G1-A not CONTINUE) |
| LeRobot export | — | Not produced: no accepted bundle exists |
| Legacy checks | `v3/sfa3/verify3.py`, `labels.py` (source-relative) | Not used as gates; G1 uses the absolute G1-A gates in `g1gates.py` |

## 2. Common adapter (identical for every condition)

- **Initial pose.** `init = trans(target − p_src) · about(R(a_src → −z), p_src)`. The shortest rotation was used in all 271 prepared attempts; the +x 180° fallback never triggered. It is applied to the standing can and to the giver reach_end (via) / grasp / lift (contact) waypoints, exactly as planner3 inserts them for its own initial shifts.
- **Retained source settings.** D = Γ = I, τ = t, scale 1. Native durations and commands are kept.
- **Export.** Rows `[3,T−6)` with a native row map and timestamps `(row−3)/30`. States and actions are float32.
- **No condition-specific branch.** No code path depends on the condition label. The fixed-source-6 control confirms this (identical arrays, ≤1e-6, under all three labels).

## 3. Frozen-condition verification (STEP 2)

- **Source sets.** Exactly Original10 0–9, SD10 1,6,7,11,13,14,27,34,42,47 and Random10 5,6,7,8,18,25,29,32,43,44, from `locked_generation_config.json`.
- **Targets and settings.** Every condition uses the same 100 targets (`locked_targets.json`, SHA verified). Target definitions, selection rule, IK settings, attachment rules, budget (one attempt, 0 retries, 0 shrinks), seed 20261108 (recorded; geometry is deterministic) and output format are identical.
- **Recorded per target and condition:**
  - the actual source used;
  - its locked score and distance;
  - the recomputed ranking (all equal to locked).
- **Forbidden data.**
  - v4A records 10/17/24/31/38/45 are not in the allowed union (assertion).
  - `PoolData.load` asserts body camera only. No wrist stream of any record was opened, and no evaluator-only data was processed.

## 4. Conformance evidence before the first locked-target attempt

- **Identity fixture (source 6).** Planned q and actions equal the source exactly (deviation 0).
- **Analytic 2 cm translation fixture.** Exported can reaches the requested lid (error 3e-18 m). Labels reconstruct (5.9e-8 rad).
- **Corruptions, each rejected by its intended gate:**
  - source-action substitution;
  - one-joint 6-frame delay;
  - 3-frame gripper command shift;
  - 3-frame state delay;
  - 5 mm initial can offset;
  - 20° can rotation after capture;
  - copied static can.
- **Code freeze.** Code was committed before the first locked-target attempt. The runner refuses to start with uncommitted `g1_exec` code, and that guard fired once.

## 5. Results integrity

- **Primary rows.** All 300 are present and unique (target, condition) pairs. Every failure is recorded with its gates; nothing is missing.
- **Controls.** 30 controls and 23 identity diagnostics are present.
- **Errors.** There were 0 exceptions and 0 timeouts. Per-attempt wall time is about 1.7 s, against a 120 s cap.
- **Preparation.** The cold cache was empty at start. 20/23 sources prepared; the 3 failures are recorded with traceback and tracker evidence (`README.md` addendum).
- **Hashes.** Per-attempt arrays are stored outside Git with SHA-256 (`outputs/store_manifest.json`, 411 files).

## 6. Deviations and implementer decisions (all disclosed before outcomes unless stated)

1. **Preparation failures.** Handled per protocol (Y_A = 0, no runner-up or repair). An explicit "failed preparation → INCONCLUSIVE" scorer rule and a preparation-only descriptive subset were added after preprocessing but before any locked-target attempt.
2. **Floor/ceiling numbers** (max yield < 0.10 / min > 0.90) were declared before launch. The protocol names floor/ceiling without a value.
3. **Gate 3** uses all planned frames 0…T−1. Exported rows are checked with FK of the exported states (stricter than export rows only).
4. **Gate 9** additionally requires exported states equal to the planned q.
5. **RGB stage withheld** on the owner's instruction ("do not launch RGB synthesis automatically if the geometry gate fails"). This overrides the protocol's "render all complete finite plans". No G1-B evidence exists.
6. **No independent blinded assessor was available.** A positive G1-B conclusion was unavailable regardless.
7. **Launch-time failures.** Two runner launches stopped before any target attempt, on a metadata error and the clean-tree guard. Both were fixed and committed; no partial outputs existed.

## 7. Not established

- Physical or force validity, collision-free execution, dynamics and real-robot success.
- Calibrated metric poses and RGB fidelity.
- Any VLA benefit.
- Novelty.

The registered height prior (lid z 0.32 m) is ≈13 cm below every reconstructed source lid (0.417–0.481 m). This is a
pre-existing, documented, unvalidated design choice and was not altered.
