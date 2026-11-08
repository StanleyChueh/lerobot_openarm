# SD10-G1 results: generation utility of fixed source sets (Original10 / SD10 / Random10)

**Date:** 2026-11-08. **Status: Executed — Pending Independent Codex Review.** This is not scientifically verified.
It is an offline, generation-only engineering screen. No VLA training, physics engine, robot/CAN access or external push.

## Decision

- **Registered G1-A decision: INCONCLUSIVE.** Two causes:
  - a geometry floor: 0/100 complete geometric demonstrations in **every** condition;
  - failed source preparation: 3 SD10 sources (11, 42, 47) failed inside the frozen reconstruction.
- **G1-B (RGB artifacts): NOT MEASURED.** RGB synthesis was not launched because G1-A is not CONTINUE.
- **Independent physical RGB validity: NOT ESTABLISHED.**

**Recommendation (stopping rule): STOP SD10-specific generation work; RETURN TO v4A geometry/camera calibration and
v4C interaction-aware generation.** SD10 does not outperform Original10 or Random10: no condition produced a single
complete demonstration. This is **not an informative negative about source selection**. The floor is set by
reconstruction/calibration inconsistencies that every source shares (§4), so this screen cannot distinguish the source
sets at all. No new selection method, SmolVLA retraining or project expansion is proposed.

## 1. Provenance

| Item | Value |
|---|---|
| Contract | `sd10/generation_v1/` (seal `2dd6ad94…8e9a`), Codex pre-execution decision GO; tracked unchanged at `04968cb` |
| Worktree / branch | `../lerobot_openarm_sd10_g1`, `research/sd10-g1-generation` |
| Code frozen before any locked-target attempt | `1190329`, plus pre-attempt addenda `258c00a` (preparation-failure handling), `0975203`, `36719e2`, `28de18b` |
| Launch HEAD | `28de18b` (`outputs/geometry_registration.json`) |
| Data | pool `ethanCSL/openarm_pringles_lerobot_real_v00@2c1bc860`, body camera only; source state/action hashes verified against the locked config |
| Models | SAM2.1-hiera-small `ee5bba1d` (bf16, RTX 4090) for masks; pinocchio float64 FK/IK on CPU (4 threads) |
| Seals rechecked | G1, L1, both v4A seals, all inherited code/robot hashes and the 25 preserved SD10 files (`check_seals()` before every stage) |

The frozen planner3/render3/v2-reconstruction files were **not modified**. Common adapter code lives only in `g1_exec/`.

The two first launch attempts stopped before any target was attempted:
- a package-metadata lookup error;
- the clean-tree guard catching an uncommitted figure script.

Both were fixed and committed; no attempt output existed before the recorded run.

## 2. G1-A results (300 primary attempts, same 100 targets, one locked source per target, no retries)

| | Original10 | SD10 | Random10 |
|---|---:|---:|---:|
| **Complete geometric yield Y_A** | **0/100** | **0/100** | **0/100** |
| Near (54) / far (46) | 0 / 0 | 0 / 0 | 0 / 0 |
| Central 16 / intermediate 48 / edge 36 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Attempts with failed source preparation | 0 | **29** (sources 11: 7, 42: 10, 47: 12) | 0 |
| Complete finite plans | 100 | 71 | 100 |
| Gate 3 IK residual failures | 100 | 71 | 100 |
| Gate 4 limits/speed/accel failures | 100 | 71 | 100 |
| Gate 5 clearance/penetration failures | 28 | 19 | 33 |
| Gate 7 receiver-capture failures | 92 | 43 | 63 |
| Gate 8 object-continuity failures | 100 | 71 | 100 |
| Gates 1, 2, 6, 9 failures | 0 | 0 | 0 |
| Median selection distance (px) [p95] | 37.9 [98.1] | 37.5 [70.9] | 44.6 [74.8] |
| Median SE(3) translation (cm) | 16.1 | 15.4 | 15.6 |
| Median SE(3) rotation (axis straightening, °) | 13.1 | 13.6 | 13.6 |
| Median grasp-waypoint request / IK miss (mm) | 148 / 100 | 143 / 134 | 149 / 143 |
| Median max accel (rad/s²) | 197 | 203 | 227 |
| Median table→giver can jump (mm) | 110 | 135 | 148 |
| Attempt time, sum (median per attempt) | 171.5 s (1.72 s) | 121.5 s (1.68 s) | 165.4 s (1.67 s) |
| Standalone ten-source preprocessing | 324 s | 278 s (3 failed) | 308 s |

The pooled 23-source cold preprocessing took 688 s. There were no exceptions and no timeouts. Accepted-target coverage
is empty in every arm (the candidate hull over all 100 targets is 0.047 m²).

**Registered paired statistics** (`paired_stats.meaningful_yield`):

| Comparison | Difference | Paired CI | Spatial-block CI | McNemar p | Result |
|---|---|---|---|---|---|
| SD10 − Original10 | 0.00 | [0, 0] | [0, 0] | 1.0 | PASS = false |
| SD10 − Random10 | 0.00 | [0, 0] | [0, 0] | 1.0 | PASS = false |

The registered floor rule (declared before launch) applies: max yield < 0.10. The protocol's "failed preparation" rule
also applies. Both make the verdict INCONCLUSIVE, not STOP-as-informative-negative.

The descriptive analysis on the 71 targets whose assigned sources prepared in all three arms gives the same result:
difference 0.

**Fixed-source-6 invariance control:** passes. All 10 targets give identical arrays (≤1e-6) and identical Y_A under all
three condition labels, so no condition-dependent plumbing is present.

**Leave-one-assigned-source-out:** every difference is 0.

## 3. Answers to the registered questions

1. **More complete demonstrations?** No. 0/100 in all three arms. SD10 additionally lost 29 targets to preparation failures.
2. **Better geometric consistency?** Not in the registered sense: every complete plan in every arm fails gates 3, 4 and 8.
   - SD10's prepared plans fail gate 7 (receiver capture) less often: 43/71 vs 92/100 and 63/100.
   - That difference tracks *which sources* have consistent receiver-closure reconstructions, not position diversity.
   - It is descriptive only.
3. **Smaller retargeting magnitude?**
   - **Pixel distance:** not on the median (37.5 vs 37.9 px for Original10; 44.6 for Random10), though the tail is shorter (p95 70.9 vs 98.1 px).
   - **Metric SE(3) translation:** similar in all three arms (15–16 cm). It is dominated by a common ≈13 cm height shift that is a property of the registered height prior, not of the source set (§4).
4. **Fewer rendering artifacts?** Not measured. RGB synthesis was not launched (G1-A not CONTINUE); no blinded independent assessor exists.
5. **Better than Random10?** No.
6. **Far targets without sacrificing centre targets?** No arm produced any complete far or centre target.
7. **Pace / recording quality.** The only condition-specific differences come from source recordings.
   - **Preparation failures:** the frozen HSV lid tracker loses the can in SD10 sources 11 (from frame 1), 42 and 47; 42 and 47 are the two fastest recordings.
   - **Receiver-capture consistency** varies by source reconstruction.
   - Neither is a spatial-diversity effect. Recording pace/quality is therefore a material confound in SD10's preparation loss.
8. **Further SD10 generation research justified?** No. Return to v4A/v4C.

## 4. Why the floor occurs (descriptive decomposition, `outputs/floor_diagnostics.json`)

**(a) Every source fails before any retargeting.**
- Identity diagnostics (init = I, all 20 prepared sources) pass gates 1, 3, 4, 6 and 9 — exports, IK, limits, holds and labels are sound.
- **All 20 fail gate 8:** the frozen reconstruction's standing-can pose and its giver-hold pose disagree by 9–96 mm (median 40 mm) at the table→giver switch, against a 5 mm gate. Retargeting with one SE(3) preserves this jump, so Y_A = 0 is unavoidable for every source.
- 13/20 also fail gate 7 (receiver capture 0–44 mm, median 16 mm), and 8/20 fail gate 5 (hand mesh inside the can cylinder).

**(b) The registered targets are not reachable with source grasp orientations.**
- Reconstructed source lid heights are 0.417–0.481 m (median 0.454 m), against the registered candidate lid plane of 0.32 m. Every attempt therefore also moves the can ≈13 cm down along its camera ray.
- The geometry-only overlays show the planned can projecting smaller than the real can at the same pixel.
- Median grasp-waypoint requests are 143–149 mm. No attempt in any arm reaches the grasp waypoint within 2 mm (minimum miss 21 mm), and the miss grows with the requested displacement (r = 0.69–0.92).
- The IK misses also produce the gate-4 acceleration violations.

**(c) Frozen reconstruction cannot prepare three SD10 sources.** The Gate-1 HSV lid tracker loses the can, and the recon3 fallback covers only a missing receiver window (`README.md` addendum).

All three are camera/height-calibration and reconstruction problems: exactly v4A's open prerequisites. None is a property of source spatial coverage.

## 5. Scope and limitations

- Deterministic finite grid of 100 image-space targets, not random robot trials.
- Candidate metric poses are estimates under an unvalidated camera/height prior.
- Gates are kinematic and bookkeeping screens: no force, contact, dynamics or collision-free validation.
- Several gates are construction-aligned.
- The floor/ceiling numbers were declared by the implementer before launch, because the protocol names floor/ceiling without a number.
- The RGB stage was withheld on the experiment owner's instruction, which overrides the protocol's "render all complete plans"; this is disclosed.
- No novelty claim. A positive result would only have shown source-distribution sensitivity of an existing generator.

## 6. Artifacts

- **`outputs/`:**
  - `results.json` — decision, yields, paired statistics, failure categories, details, coverage, LOO, pace, identity
  - `geometry_attempts.jsonl` — all 353 records
  - `floor_diagnostics.json`
  - `preprocess.json`
  - `conformance.json`
  - `geometry_registration.json`
  - `geometry_run.json`
  - `store_manifest.json` — 411 files outside Git, with SHA-256
- **`figures/`:**
  - `source_vs_generation_yield.png`
  - `failure_categories.png`
  - `source_distance_vs_validity.png`
  - `target_acceptance_map.png`
  - `visual_comparisons.png` — geometry-only overlays on the deterministic T000/T033/T066/T099 sample; **no RGB synthesis**
- **Large arrays and caches:** `/home/stanley/Stanley_ws/sd10_g1_store` (0.06 GiB, hashed). No RGB videos exist.

Reproduce (in the worktree, from `sd10/g1_exec`):

```bash
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 /home/stanley/Stanley_ws/lerobot_openarm/.venv/bin/python -B"
CUDA_VISIBLE_DEVICES=0 $PY preprocess_g1.py   # requires an empty store (cold cache)
$PY conformance_g1.py 6
CUDA_VISIBLE_DEVICES= $PY run_geometry.py
$PY score_g1.py; $PY diagnose_floor.py; $PY figures_g1.py; $PY manifest_g1.py
```
