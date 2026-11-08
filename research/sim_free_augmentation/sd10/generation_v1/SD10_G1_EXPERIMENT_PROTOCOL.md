# SD10-G1: generation utility of fixed source sets

2026-11-08. **Preregistered, not executed. GO for the registered offline generation screen only**, subject to the common adapter/conformance checks below. No VLA training, physics engine, robot/CAN access or pushes. This document does not authorize changing any historical or v4A frozen file.

Actual repository HEAD: `8d31437e5779ef80aca9ffd3e6fa94b288d54f5e`, branch `research/contact-aware-augmentation`; cached upstream 20 commits behind, no remote freshness assumed. Notion Guide v0.0.6, Research HQ/Home, Current State, Ledger, Decisions, Prior Work and Next Experiments were read with actual MCP calls. The enabled Stanley handoff Skill was loaded. Git and pinned local inputs are authoritative.

## Question and preserved history

H1: Does SD10 increase **complete geometric generation yield** on common target configurations versus Original10?

H2: Does it reduce geometric inconsistencies and visible RGB artifacts? These are separate endpoints; geometric acceptance cannot prove better pixels.

H3: Does either benefit exceed the same fixed Random10 set? Comparison with Original10 alone is insufficient to endorse curation.

SD10-L1 tested real-only VLA learning. Its reported primary FAIL and no established SD10-over-Random10 advantage remain intact. The local `sd10/l1_exec/L1_RESULTS.md` is read as Claude's report, not independently rescored in this task. Its negative result does **not** reject generation utility. Likewise, “era/pace explains learning” remains a supported hypothesis, not an isolated causal result. No new selection or learning experiment is registered here.

## Fixed evidence and conditions

| Condition | Captures in the common real-v00 pool |
|---|---|
| Original10 | 0–9 |
| SD10 | 1,6,7,11,13,14,27,34,42,47 |
| Random10 | 5,6,7,8,18,25,29,32,43,44 |

Each has exactly ten real captures. Use **all RGB from the same encoding**: `ethanCSL/openarm_pringles_lerobot_real_v00@2c1bc860347c44fc0fbae7567469d36ee49baa8d`. The original-ten dataset `ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98115ab6827a4ec2655335c536b5dd085b` supplies lineage evidence only: episode i is the same capture/state/action as pool i, separately encoded. Do not blend the two encodings.

Source IDs are immutable. No replacement of a failed source/reconstruction and no outcome-dependent ranking. No pool 10/17/24/31/38/45 images or derived evaluation evidence may be processed; no wrist stream is needed at all. Only the selected source's body stream/state/actions and its pre-existing initial-lid annotation enter that attempt. Shared robot/camera/object/model priors are identical. Another condition's source footage is not an appearance/reference library.

The configuration binds all available dataset content hashes, original SD10 hashes, selected capture state/action fingerprints, source event/quality metadata, inherited code/robot hashes, SAM checkpoint and RAFT weights. Current remote `main` equality is not claimed.

## Exactly 100 common targets

`locked_targets.json` freezes a **10×10 inclusive lattice** in native body pixels:

`u=linspace(260,440,10)`, `v=linspace(100,360,10)`, row-major IDs T000–T099.

There are 16 central cells (row/column 3–6), 48 intermediate and 36 outer-edge cells. These are **edges of the registered image ROI**, not surveyed physical workspace boundaries. Near/far is distance to the nearest Original10 initial lid center ≤40/>40 px, common to every condition. There are 54 near and 46 far targets; the latter are the predeclared underrepresented positions. Tags may overlap region categories. Counts/distances are planning metadata, not outcomes. No targets were selected from L1 errors or generation results.

For a common computational world frame, intersect each body-camera ray with **candidate lid plane z=0.32 m** using the fixed inherited `sfa.robot.BODY_CAM` prior. The can prior is radius 0.0375 m, height 0.23 m; implied candidate table plane z=0.09 m. Candidate coordinates span approximately x=0.184–0.467, y=−0.074–0.115 m. These numbers are **estimated candidate coordinates, not verified metric object poses or reachable workspace**. Calibration lineage is unvalidated; the ROI, height and reachability are hypotheses. No post-output calibration, rejection-based target pruning or target shrink is permitted.

All conditions request the **same candidate lid position and upright axis** (lid-to-bottom `[0,0,-1]`). Cylinder yaw is unconstrained and retains source material-frame gauge: this is a position/axis screen, not matched full textured 6-D orientation ground truth. Record the resulting full orientation and source-to-target rotation. Inherited camera and source reconstruction do not validate absolute yaw or physical RGB accuracy.

## Generator and source assignment

Use one common **G+ geometry planner**, based on frozen `v3/sfa3/planner3.py`, and the same frozen body-only `v3/sfa3/render3.py` layered renderer. This is not a C-versus-G+ experiment. No new learned generation model, condition-specific repair or filter.

At each target, score the ten available sources by squared initial-lid distance in native body pixels. Choose the minimum; exact ties use lower episode ID. All rankings, chosen IDs and scores are already frozen in `locked_targets.json`. Motion speed, quality, IK outputs and acceptance do not change this ranking. Preprocessing failure of that selected source makes the corresponding attempt fail; do not try the runner-up.

The allowed common implementation work is an input/initial-pose adapter, isolated from historical files:

1. Load the pinned 50-pool by canonical capture ID; **do not call the old `sfa.data.load` unchanged**, which reads the ten-only snapshot.
2. Rebuild the same v3/v2 body-only reconstruction for the 23-source union, fixed common camera and body lag −1.625 frames. Use the pinned SAM-small/RAFT checkpoints and source-only tracks/attachments. No camera refinement or target-view fitting. Per-source closure-only/low-visibility fits remain explicitly flagged. Never turn those fallbacks into measured contact evidence.
3. Generalize the initial transform consistently for every condition: translate to the common candidate lid position and apply the shortest rotation aligning the source can axis to down, retaining free yaw. Apply that **one SE(3)** to the initial can and giver reach/grasp/lift waypoints, then use ordinary G+ waypoint interpolation back to the source handover. Do not merely move the object image while leaving the reaching robot unchanged.
4. Keep `D=I`, `Gamma=I`, `tau(t)=t`, scale=1 and native source duration/commands. No handover/regrasp/timing perturbation is added. The source handover differs between captures; only initial position/axis targets are common. This is a source-set package comparison, not identical counterfactual full paths.
5. Preserve native action construction: arm command at t is `q_gen[t+L] + (a_src[t]−q_src[t+L])`, with source lead L selected by the frozen 0–6 scan and lower-argmin tie; gripper commands retain source zero-order chronology. No command replay on changed states, altered gripper commands or time shift fitted against generated RGB.

### Fixed valid export support

The legacy action constructor produces NaNs in its terminal lead segment. Do not pad those or call an incomplete bundle complete. **All conditions export the same rule `[3,T−6)` in native source row time**, T−9 output rows, 30 fps. This removes the first three timing-edge rows and last six command-support rows for every source, irrespective of its fitted lead. Require all command transitions and at least 15 post-donor-release frames inside this interval. Complete means a complete handover plus this registered retained tail, not all original T rows. Store the native-row map and 3/30 s origin; output-relative timestamps alone are insufficient provenance.

A generated body frame at native i uses continuous state/object time `i−1.625`, and source reference time under the same fixed map. Source lag is a model convention, not known exposure timing. No integer-only object lookup, mixed camera clocks or per-arm timestamp rule. All exported 16-D states/actions must be finite and aligned; use body MP4 640×480/30 fps with exact row count. Independent saved-array label reconstruction must catch source-action substitution, per-joint delay and gripper-command shifts on numeric fixtures before the run. It does not establish motor execution.

## Attempts, computation and sensitivities

Primary: **one selected-source attempt per target per condition, 300 total**, scale 1, no retries, alternate seeds, source fallback, target repair scales 0.7/0.49 or winner selection. All attempts/rejections/timeouts remain registered outputs. An unsuccessful attempt may still have a finite complete plan; render it for the separate artifact screen, within budget, even when it fails G1-A. Partial/nonfinite plans cannot be rendered and remain missing/failed RGB in the full denominator.

Common source preprocessing: 600 s/source, all 23 newly processed in ID order with pool-keyed caches; old ten-only caches are not a free advantage. Shared captures use exactly the same immutable asset hashes. Charge each arm its own ten-source standalone preprocessing/model-load cost; also report actual union/deduplicated cost. Identity re-render/source-fit diagnostics are part of this cap and use each source's original pose, not retargeting.

Each primary/sensitivity attempt: 120 s geometry/checks + 120 s RGB/encode/read-back, four CPU threads and one common recorded GPU device/model/precision. No process parallelism that hides condition-dependent contention. Target order is common T000…T099; rotate arm execution order by target index mod3. On the same hardware, include data/model loads, preprocessing, selection, rejected attempts, validation, encoding and I/O in cold and amortized throughput. Report per-output-frame cost too, since native source durations differ. Maximum phase budgets, rather than observed execution speed, are frozen here. Actual implementation/environment hashes must be registered before first attempt.

**Fixed-source negative control:** at T000,T011,…,T099, independently execute source **6** under each of the three arm labels (30 extra attempts, no cache alias to fake replication). Same inputs/seeds must yield identical numerical arrays within 1e−6 and identical deterministic-render pre-encode pixels; otherwise condition-dependent plumbing invalidates comparison. These are separate controls, never fallback attempts or primary successes. They test implementation invariance, not the benefit of three source sets. Total candidate budget **330**, plus 23 source identity diagnostics, no additional primary search. No retry after failed same-source agreement; diagnose before unblinding comparative scores.

Record source utilization, near/far, region, reference distance, SE(3) magnitude, source duration/pace, command handover timing and source observation quality. Descriptive leave-one-assigned-source-out sensitivity uses saved results only; no new generation or reweighting of the primary endpoint. Large dependence on one source must be disclosed.

Do not attribute a gain solely to spatial diversity: source recording era, initial robot state, motion speed, duration, image quality and attachment/reconstruction quality also differ. The fixed-source control tests condition invariance, not isolation of these mediators. This is generation utility of the fixed source sets under the registered method.

## Gates and decision

Detailed thresholds and failure denominators are fixed in `SD10_G1_METRICS.md` and the machine configuration. All-frame FK/IK, joint/action limits, temporal/attachment/receiver geometry and label support define **G1-A engineering consistency**. They are not force, collision-free execution or successful physical contact certification. No source-dependent relaxation of primary thresholds; report old source-relative verifier scores separately if desired.

**G1-B is a limited body-RGB artifact screen.** It cannot close the v4A calibrated real-view gate, validate wrist images or claim physical RGB accuracy. Score all complete finite rendered plans, not just geometry acceptances. Preserve missing/UNKNOWN data and provenance fractions. At least one assessor independent of implementation must label anonymous full clips before seeing condition identities/scores. Source identity may be inferable from the images: blinding is a procedure, not a perfect guarantee. A second assessor and disagreement report are recommended; without one, explicitly label evidence single-assessor. No image or score feedback to generator tuning.

H1/H3 geometric success requires **≥10 percentage points** more complete geometric targets than **each** control, both paired-target 95% CI lower bounds >0, both spatial-block sensitivity lower bounds >0 and both exact paired McNemar p≤0.05. This conjunction tests the registered both-controls claim; do not select the easier comparison. Ten points is a practical screen threshold, not a power guarantee.

H2 is separate: full-denominator clean RGB/bundle yield, plus independently assessed gross-artifact rates on the fixed common-complete subset. A geometry gain alone cannot pass H2. See minimum coverage/quality rules in the metrics file. Unknown physical RGB fidelity stays UNKNOWN even with a positive artifact result.

No target-score-driven edits, “better sources” reselection or extra models. Stop after fixed budgets. A systematic implementation/data-integrity failure makes the study INCOMPLETE; stop the comparison and preserve artifacts. Infrastructure-only exact replay may resume the same interrupted task/seed with documented cost and no usable alternate output; no algorithmic retry or shopping for outputs.

After execution, Codex must verify all 300 primary rows, the 30 controls, source assignments, input/output hashes, actual selected-source evidence, exported arrays/MP4 frames, independent artifact annotations and both statistical analyses. Only then create `CODEX_SD10_G1_FINAL_REVIEW.md`.

- **CONTINUE SD10:** registered meaningful geometric gain over both controls, valid invariance/data checks; report H2 separately. If RGB artifacts remain poor, only geometric-source utility may continue, with v4A repair still required and no VLA training authorization.
- **STOP SD10 generation-utility branch:** a technically valid completed comparison rules out the registered ≥10-point benefit against at least one control (CI upper <0.10), without a credible separate registered RGB advantage. Preserve the finite-target scope; no universal no-benefit claim. Return to v4A/v4C.
- **INCONCLUSIVE:** intervals span meaningful benefit/no benefit, geometry ceiling/floor, failed preparation, sparse common RGB or audit defects. End this budget and return to v4A/v4C rather than repeat/tune SD10 indefinitely. No final scientific verdict is issued at preregistration.

## Novelty boundary and prior work

| Work | Already established mechanism relevant here | Meaning for a positive SD10 result |
|---|---|---|
| [DemoGen](https://www.roboticsproceedings.org/rss21/p157.html) | Transfers demonstrated trajectories to new object configurations and edits point-cloud observations. | Source-distribution sensitivity of an existing geometry approach, not first spatial augmentation. |
| [R2RGen](https://arxiv.org/abs/2510.08547) | Shared-3D parsing, group-wise trajectory augmentation/backtracking and camera-aware point-cloud postprocessing. | Source coverage and grouped interactions are prior art; this screen adds no new mechanism. |
| [CP-Gen](https://proceedings.mlr.press/v305/lin25b.html) | Object-relative keypoint-trajectory constraints and optimized joint configurations for pose/geometry changes. | Preserving constraints or using IK is not novel by itself. |
| [Real2Edit2Real](https://arxiv.org/abs/2512.19402) | Metric reconstruction, 3D trajectory editing and depth/action/edge/ray-conditioned multi-view video. | A body homography artifact screen is not equivalent evidence of calibrated RGB generation. |
| [D-CODA](https://proceedings.mlr.press/v305/liu25f.html) | Coordinated wrist-view generation with corresponding joint labels and contact-related constrained optimization. | Bimanual contact/label coupling already has direct prior work. |

A positive result supports **A: source-distribution sensitivity**. Input plumbing/SE(3)/label/provenance corrections count as **B: engineering repairs** and must be credited to all conditions. It establishes **C: no new generation mechanism**. A genuine contribution would require an identifiable generation-time interaction/observability mechanism and controlled baselines, not ordinary source curation. Sources checked from primary paper/project records on 2026-11-08. No force-conditioned claim is tested.

The lock binds this protocol, targets/config, metrics, pre-execution audit and model-free reproducibility/statistics checks. Future adapter code must receive a separate immutable execution registration/hash before outputs, and match this contract; its absence today is not a completed implementation audit. All frozen v4A/v4B/v4B.1/SD10-L1 records remain untouched. Preregistration performs no source segmentation, IK, rendering or training.
