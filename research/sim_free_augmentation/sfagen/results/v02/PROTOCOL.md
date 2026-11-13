# v02 frozen protocol — 2026-11-13

Approved scope: Step 2 local generation and export; no policy training, hardware, physics, publication or commits.
Base: research/sfagen-wrist-render @ 77e52d4. Source: 10 real Pringles demonstrations pinned at
cebdee98115ab6827a4ec2655335c536b5dd085b. Prior diagnosis remains in ../eval_v00/.

## Hypotheses and matched conditions

Uniform arena placements may improve spatial coverage. Right-wrist reconstruction is an ablation, not a proven fix
for late grasping. Use the current sfagen can reconstruction/body renderer: the legacy lid-pose comparison showed
no established overall gain. Neither dataset establishes physical fidelity or policy performance.

Both conditions use identical candidate IDs, states, actions, body images and camera clocks. One copies both wrists
with validity flags; the other re-renders only the right wrist using the existing refined mount/RAFT renderer.
Left/right lags are −0.25/−1.0 frames in both conditions. The previously failed Pringles wrist fidelity gate remains
a limitation even if export/codec checks pass. Each condition gets a synthetic-only export and a separate mixture
containing the original 10 real episodes. No upload in this step.

## Placement and budget

3 cm lattice centers: x = 12,15,18,21,24,27,30 cm; y = −3,0,3,6,9 cm (35 targets).
Cells extend 1.5 cm around their centers and contain the measured evaluation arena. These approximate positions
reuse development/evaluation information and are not an untouched test set. Assign each target to the nearest
1–3 sources between 0.5 and 8 cm away: 89 source–target attempts. Keep plan.f1_plan unchanged.
Also try all 8 existing ±3/±5 cm axis-wise handover-location shifts on each source (80 attempts), plus ±10-frame
handover timing shifts (20 attempts). Maximum 189 attempts, one pass each. Preserve every rejection. No retry with
weaker thresholds or alternate target selection after seeing results. Wall-time budget: 24 GPU-hours; stop and
report incompleteness if exceeded. Render/export only passing candidates. No quality-based resampling.

## Timing amendment approved by user before generation

Keep v00-style ±10-frame timing variants. Spatial variants must have source-time slope 1 everywhere. Every variant
must have exactly the source frame count, unchanged fps and both source endpoints. Timing variants may have local
non-unit slope, but preserve the initial grasp clock and unit slope from receiver close through giver release.
Use the v3 non-dual piecewise-linear knots, with no global slowdown. Reject non-monotone/invalid schedules.
Gripper commands use zero-order hold; arm labels preserve the output-frame command lead. Independently recompute
the reference and detect corrupted labels. Commands remain nominal kinematics, not hardware validation.

## Gates frozen before results

Existing v01 quality gates remain: finite arrays; URDF ∪ original source state/action limits; action speed ≤1.25×,
acceleration ≤1.5× the real-source envelope; table/mesh/hand clearances and placement checks; contact IK ≤1 mm/0.5°,
blend ≤5 mm/2°; start/untouched/rejoin constraints. Do not clip labels or slow trajectories to pass. Time-warped
references must not enlarge the original source joint/action envelope. Use existing fast necessary-condition rejection.
Run tools/quality_report.py and sfagen verify on both exports, including frame spans, label SHA, no complete real
episodes in synthetic-only output, flags and CPU 50-action load check. Decoded body and sampled rendered wrist
PSNR must be ≥30 dB against raw generated frames (a codec gate, not physical RGB fidelity). Encode H.264 CRF12
yuv444p identically in both conditions to avoid the known v01 compression failure; do not lower the threshold.
Independently check fixed duration, spatial identity clocks, timing knots/events and matched candidate labels.

## Reporting and stop rule

Show all 35 targets, attempt/pass counts by family, rejection causes, accepted rear-corner coverage, nearest distances
to all 40 measured starts, and duration/source-time charts. Save decoded three-camera phase sheets and matched wrist
examples, including distortion. Link exact configs/hashes and local datasets in a visual HTML report. Coverage gaps,
failed gates and absent hardware/policy evidence must remain visible. Stop after this bounded batch and all checks;
do not proceed to Step 3 training or Step 4 benchmark execution.
