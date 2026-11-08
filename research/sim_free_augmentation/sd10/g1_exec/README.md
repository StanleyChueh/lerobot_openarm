# SD10-G1 execution package (implements the sealed contract; does not alter it)

Contract: `../generation_v1/` (`SD10_G1_EXPERIMENT_PROTOCOL.md`, `SD10_G1_METRICS.md`, `locked_targets.json`,
`locked_generation_config.json`, `paired_stats.py`; seal `2dd6ad94…8e9a`; Codex pre-execution decision **GO**).
Bundle tracked unchanged at `04968cb`.

**Worktree.** `../lerobot_openarm_sd10_g1`, branch `research/sd10-g1-generation`. This file was committed **before
the first locked-target attempt**.

**Storage.**
- Large artifacts (caches, plates, per-attempt arrays, any MP4) live outside Git in `/home/stanley/Stanley_ws/sd10_g1_store`, with content hashes.
- Small summaries go in `outputs/` (gitignored; force-added).

| File | Role |
|---|---|
| `g1lib.py` | seals; pool input adapter; cold pool-keyed reconstruction (frozen v2/v3 code, fixed lag −1.625, fixed camera); SE(3) initial-pose adapter; planner3 G+; `[3,T−6)` export |
| `g1gates.py` | G1-A gates 1–9 with absolute thresholds from `locked_generation_config.json` |
| `attempt_g1.py` | one condition-agnostic attempt (frozen selection recomputed and checked) |
| `preprocess_g1.py` | 23-source cold preprocessing, ID order, 600 s cap per source |
| `conformance_g1.py` | analytic/identity fixtures and label/object corruptions (source 6, scratch store, no locked target) |
| `run_geometry.py` | 300 primary + 30 source-6 controls + 23 identity diagnostics; fork per job, 120 s cap, 4 threads |
| `score_g1.py` | completeness, invariance, `paired_stats.meaningful_yield`, coverage, failures, decision |

## Implementation choices (declared before launch)

1. **Inputs.** `sfa.data.load` (ten-only snapshot) is never called.
   - A pool adapter is injected as the frozen recon's `data`. It reads pool `2c1bc860` by capture ID, body camera only, and verifies each capture's state/action SHA-256 against the locked config.
   - Wrist streams cannot be opened (assertion).
   - v4A records 10/17/24/31/38/45 are not in the allowed union (assertion).
2. **Reconstruction.** The frozen `sfa2.recon.build` runs with:
   - the frozen `sfa3.recon3` closure-only fallbacks (flagged per source);
   - a new cold cache per source;
   - lid tracks from the verified pool clicks.

   `rc.estimate_lag` is wrapped so its rms curve is kept as a diagnostic while the registered fixed lag −1.625 is used. The camera is the inherited `BODY_CAM`, with no refit.
3. **Initial pose.** `init = trans(target − p_src) · about(R_short(a_src → −z), p_src)`; the opposite-axis fallback is +world-x 180°. The planner3 spec for `var={}` (D=I, Γ=I, τ=t, scale 1) gets the same reach_end (via) / grasp / lift (contact) giver waypoints `init·T_w(t)` that planner3 itself inserts for initial shifts. `table_new = init·table`. Plan with frozen `planner3.plan(…, "G+")`.
4. **Export and labels.**
   - Rows `[3, T−6)` with a native row map; timestamps `(row−3)/30`.
   - float32 states and actions; can poses are float64.
   - Actions are planner3's own construction (`q[t+L] + a_src[t] − q_src[t+L]`, gripper source ZOH).
5. **Gate details.**
   - **Gate 3 (IK residual)** is evaluated on every planned frame 0…T−1. Exported rows use FK of the exported float32 states; the remaining rows use the planned q needed for action leads. Requested targets are recomputed with planner3's own `waypoint_delta`.
   - **Gate 5 (clearance/penetration)** uses full-mesh hull vertices plus the robot's seed-0 samples (≤400 per link) on every exported row. Distal links are link5–7, hand and fingers; the can-penetration test uses all non-finger arm links.
   - **Gate 9 (labels)** additionally requires the exported states to equal the planned q (≤1e-6).
6. **Determinism and controls.** Geometry is CPU float64 pinocchio with no randomness; the registered seed is recorded. Source-6 controls are independent forked executions, compared array-by-array (≤1e-6).
7. **Decision rule for "floor/ceiling" (protocol names it but gives no number).** Fixed here before any attempt:
   - **Floor:** max condition yield < 0.10.
   - **Ceiling:** min condition yield > 0.90.

   Either gives INCONCLUSIVE. STOP requires a non-floor comparison whose paired CI upper bound is < 0.10 against at least one control.
8. **RGB stage (G1-B) is conditional.** Per the experiment owner's instruction, RGB synthesis is not launched automatically if the G1-A decision is not CONTINUE. That overrides the protocol's "render all complete plans", and the deviation is disclosed here.

   No independent blinded assessor exists inside this execution. Implementer ratings cannot be the primary G1-B evidence, so a positive G1-B artifact conclusion is unavailable regardless.

## Pre-launch observations (source-only fixtures; recorded before any locked target was attempted)

- **Fixture results.** The conformance fixtures pass:
  - identity reproduces source q and actions exactly;
  - source-action substitution, a 6-frame single-joint delay, a 3-frame gripper shift, a 3-frame state delay, a 5 mm initial can offset, a 20° can rotation after capture and a copied static can are each rejected by their intended gate.
- **Height-prior discrepancy.** Historical v3 reconstructions place the source lid at z ≈ 0.44–0.48 m (implied table ≈ 0.21–0.25 m); source 6 here is at 0.456 m. The registered candidate lid plane is z = 0.32 m (table 0.09 m), so every attempt in every condition also moves the can ≈ 13 cm down along its camera ray. This is the registered, unvalidated camera/height prior, and it is not changed. It is common to all conditions and dominates source-to-target translation.
- **Source 6 identity diagnostic** (init = I) fails absolute gates 2, 5 and 8, all from source reconstruction inconsistencies that retargeting preserves:
  - axis tilt 15.9°;
  - hand-mesh/can penetration 15.7 mm;
  - table→giver bookkeeping jump 49.9 mm.

  If this is typical, the G1-A yield may be at the floor in every condition. Thresholds, targets and the generator are unchanged regardless.

## Pre-attempt addendum (after cold preprocessing, before any locked-target attempt)

The official cold preprocessing (`outputs/preprocess.json`) completed for 20/23 sources. **SD10 sources 11, 42 and 47
failed inside the frozen v2/v3 reconstruction.** The frozen Gate-1 HSV lid tracker lost the can:
- **11:** 1/249 frames ok (lost from frame 1).
- **42:** 0/44 giver-hold frames.
- **47:** 0/40 giver-hold and 0/62 receiver frames.

The frozen `recon3` closure-only fallback handles only a missing receiver window, so `joint_fit`/`fit_point` receive an
empty frame list. 5 other sources (6, 27, 32, 34, 44) prepared through the flagged closure-only receiver fallback.

Handling, per protocol ("preprocessing failure of that selected source makes the corresponding attempt fail; do not try
the runner-up"):
- Those attempts are recorded `source_preprocessing_failed`, Y_A = 0, in the registered all-100 analysis. That is 29 SD10 targets: 7 + 10 + 12.
- No tracker, reconstruction, fallback source or threshold change is made.
- The protocol's decision section lists "failed preparation" as INCONCLUSIVE. The scorer therefore returns INCONCLUSIVE whenever any assigned source failed preparation and the both-controls PASS does not hold.
- A descriptive paired comparison on targets whose assigned sources prepared in all three conditions is also reported. It is conditioned on preparation only, never on geometric outcome, and is not a replacement endpoint.
