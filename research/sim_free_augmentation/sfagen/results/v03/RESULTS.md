# v03 results: even-coverage Pringles datasets, strict and loose distortion caps (2026-11-13)

Status: **local only, nothing pushed.** The results are first-party (Claude) and await independent review. They are
kinematic and geometric checks only: no physics, no hardware replay, no policy results.
Protocol: `PROTOCOL.md` (frozen) + Amendments 1–3, hashes in `FREEZE.json`.
Run dir: `artifacts/sfagen_v03_run/`. Viewer: `viewer/index.html`.

## 1. Generation (90-cell 3 cm grid, x 9–36 cm, y −9 to 15 cm, F1W planner, all sources ≤ 12 cm)

| run | change | attempts passing | targets with ≥1 pass (of 83 tried) |
|---|---|---:|---:|
| `gen` (v1, kept as negative) | as pre-registered | 43 / 416 | 28 |
| `gen_a1` | Amendment 1: appearance support check + action-aware IK bounds for F1W | 125 / 416 | 40 |
| `gen_a2` | Amendment 2: 0.005 rad action-bound margin | 137 / 416 | 40 |
| `gen_a3` | Amendment 3: plate appearance test in Lab instead of HSV | 189 / 416 | 45 |

The amendments relaxed implementation gates that were broken, never the distortion threshold. Each one is disclosed
with evidence in `PROTOCOL.md`.

Final status of the 90 grid cells:
- 41 covered by strict;
- 4 covered by loose only;
- 18 rejected because the can is out of the body camera's view (x ≤ 12 cm, and (15, 6));
- 8 rejected for `robot_near` (the other arm, y ≥ 12);
- 7 rejected for joint limits;
- 5 rejected for speed;
- 7 with no source demo within 12 cm.

## 2. Distortion score and thresholds
- **Score:** per frame, (missing + spilled silhouette pixels) / silhouette pixels; the episode score is the 95th
  percentile over frames.
- **Thresholds** (`thresholds.json`, rule frozen before any v03 render):
  - strict = 2.289, the worst of 66 v02 episodes with shift ≤ 4 cm;
  - loose = 4.578.
- **Rendered grid episodes:** 84; median score 1.06, max 3.57; 10 above strict, **0 above loose**. The loose cap
  removed nothing, so loose = every grid episode that passed the checks.
- **Known validity limits:**
  - the score is roughly flat in shift (offset ≈ 0.45 from mask spill);
  - it does **not** catch ghost/duplicate-can or smeared-can artifacts.

  Claude's visual audit (`visual_audit.json`, not blinded) flagged 22 kept episodes, 13 of them inside strict.

## 3. Exports (8 local LeRobot v3.0 datasets)
| export dir | synthetic-only | mixed (10 real + synthetic) |
|---|---|---|
| `strict_copied` | 149 eps / 45,116 frames | 159 / 47,691 |
| `strict_right_wrist` | 149 / 45,116 | 159 / 47,691 |
| `loose_copied` | 159 / 48,214 | 169 / 50,789 |
| `loose_right_wrist` | 159 / 48,214 | 169 / 50,789 |

- **Composition:**
  - strict = 75 hand-over (v02 interaction) + 74 grid episodes over 41 targets;
  - loose = 75 + 84 over 45 targets.
- **Checks, all 8 datasets:**
  - `sfagen verify`: LeRobot load, label SHA, replay checks, video spans, no all-black frames, 0 episodes identical to a
    real episode, 0 real episodes in the synthetic-only sets, correct `is_synthetic` flag in the mixed sets;
  - `quality_report` gates PASS: joint limits, speed (max ratio 1.23), acceleration (1.50), table, start state,
    gripper, labels;
  - body decode PSNR 43.4 dB; rendered-wrist decode PSNR ≥ 39.5 dB;
  - independent `check_exports.py`: OK for strict and loose.
- **Wrist:** `wrist_view_valid` = 0.36 (left wrist) and 0.86 (right wrist). In the rendered right wrist, 64% of frames
  are rendered and 2.4% of pixels are inpainted, with visible speckle holes. The Pringles wrist-fidelity gate failed
  earlier, so the copied wrist is the main condition and the rendered wrist is an ablation.
- **Cards:** `cards/*.md`. Push tool: `tools/push_v03.py`, which creates new repos only, tags `v3.0` and verifies the
  tag. Run it with `PUBLISH=1` only after approval.

## 4. Coverage of the 40 measured evaluation starts (worst nearest can start, cm; lower is better)
| set | front-right | rear-right | rear-left | front-left | middle | worst | ≤ 2 cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| real 10 | 3.0 | 6.6 | 5.6 | 3.3 | 3.3 | 6.63 | 14/40 |
| + v00 | 1.1 | 3.9 | 2.8 | 1.4 | 1.2 | 3.91 | 28/40 |
| + v02b | 2.2 | 3.5 | 3.9 | 1.4 | 1.7 | 3.91 | 24/40 |
| + v03 strict | 1.0 | 4.9 | 2.7 | 1.4 | 1.7 | 4.91 | 31/40 |
| + v03 loose | 1.0 | 1.9 | 2.7 | 1.4 | 1.7 | 2.70 | 36/40 |
| strict, visually flagged removed | 1.0 | 4.9 | 2.7 | 1.4 | 2.0 | 4.91 | 27/40 |
| loose, visually flagged removed | 1.0 | 4.9 | 2.7 | 1.4 | 2.0 | 4.91 | 27/40 |

The ≤ 2 cm counts use the same 40 starts in every row.
- **Where strict beats v00:** front-right, rear-left and in the count within 2 cm (31 vs 28).
- **Where strict is worse:** at rear-right it is worse than v00 (4.9 vs 3.9 cm). v00's rear-right cell is not in the
  F1W grid that passed the checks.

**Key finding (negative):** loose's entire rear-right gain comes from episodes that carry visible ghost-can flags
(demos 2 and 8 shifted forward from the rear row). With the flagged episodes removed, loose = strict. The rear-right
corner therefore cannot be covered cleanly from the current 10 demos. The clean fix is to record new real demos
(section 6).

## 5. Visual audit (decoded export frames, `viewer/index.html`)
- **Strict ep 125 and ep 127** (demo 7, targets 29 and 39): the can is split at lift; the lid stays behind while the
  body moves. Score ≤ 2.19, so they pass strict.
- **Loose-only ep 142** (demo 8, target 21, score 3.57): duplicate can at grasp and lift.
- **Many episodes:** a mildly transparent arm at mid-reach.
- **Copied wrist:** clean, but table and background are off by the shift.
- **Rendered right wrist:** close-up can frames show speckle holes.

## 6. Recording recipe (`RECORDING.md`)
- **Hole-based recipe:** record 6 can-start positions, 2 demos each: (15,−6), (18,15), (30,−9), (15,3), (21,−9),
  (24,12) cm.
- The pre-registered radius rule gave r = 12 cm and 0 demos, so it is uninformative because the score is flat. It is
  reported, not used.
- **Discrete modes** (battery slots, object types, grasps, orientations) are never synthesized and need ≥ 2 real demos
  each.

## 7. Limitations
- **Score validity:** the score misses ghost and smear artifacts. Strict still holds 13 flagged episodes, and the
  visual flags are single-rater and not blinded.
- **Out-of-view cells:** x ≤ 12 cm cells are outside the body view and cannot be generated from any demo.
- **Copied wrist:** background inconsistent with the body view under shift.
- **Rendered wrist:** failed the fidelity gate.
- **Unvalidated:** there is no policy evidence yet that distortion-capped coverage beats near-source-only data. That
  is the main experiment: train on strict vs loose vs v00 and evaluate on the 40 starts, real and LIBERO
  (`../libero_scoping/PLAN.md`).

---

# Revision 2 (2026-11-14): episodes as long and as fast as the source demo

**Why.** The user inspected the published rev-1 datasets.
- The rev-1 grid episodes were in slow motion: the median was 1.09× the source length, 20/84 grid episodes were
  > 1.5×, and the worst was 3.56× (998 frames, about 33 s against an 8–9 s demo).
- Cause: when a shifted reach broke the speed/accel envelope, the F1W planner picked a slower clock from its bank
  (`local` ≤ 3×, `window` / `uniform` ≤ 6×).
- Rev 1 is kept as a negative result: it stays at Hub tag `v03-rev1` and in the local `artifacts/sfagen_v03_run`.
- **Disclosed miss:** the rev-1 independent check recorded 39 `duration_mismatch` per export, but my wrapper did not
  gate on them. The rev-2 check gates on `duration_mismatch`, `envelope_violations`, `identical_to_real` and
  `mixed_syn_equals_syn`.

**What changed** (pre-registered in `PROTOCOL.md`, Amendments 4–5; `FREEZE.json`):

| Change | Rule |
|---|---|
| No retiming (Am. 4) | `time_scales: [1.0]`: the identity clock |
| Wide spatial blend (Am. 4) | The shift is absorbed from the previous gripper event +12 frames (or frame 0) until `next_event − 10` |
| `duration` gate (Am. 4) | Every synthetic frame count == its source frame count |
| `speed_similarity` gate (Am. 4) | Moved-arm TCP peak speed and edited-span mean speed must be 0.8–1.25× the source |
| Minimum distortion (Am. 4) | Per target, render all passing candidates; keep the 2 lowest-score ones (ties: smaller shift) |
| `object_mask_grow: 60` (Am. 5) | Removes the ghost lid crescent at the old can position (found in the visual audit) |

**Unchanged:**
- joint speed ≤ 1.25× and accel ≤ 1.5× of the real envelope;
- the distortion thresholds (strict 2.289 / loose 4.578, from `thresholds.json`);
- the grid and the coverage metric.

**Generation.** 92/416 grid candidates pass, on 39 targets (rev 1: 189/416). The loss comes from the speed/accel
gates, which rev 1 had passed by slowing down.

**Hand-over variants.** 75 were re-checked (`results/v03r2/interaction_timing.json`). 4 `timing_-10` variants
(sources 4–7) fail the peak-speed ratio (0.72–0.79) and are excluded (`timing_excluded.json`).

### Duration and speed (exported data, `results/v03r2/duration_check.json` + quality reports)

| | v00 | v03 rev 1 (loose) | **v03 rev 2** (strict and loose) |
|---|---|---|---|
| Length / source | 1.000 (96/96) | median 1.09, max 3.56 | **1.000 for 130/130 (strict) and 138/138 (loose)** |
| Episode seconds | = source | up to ~33 s | **8.03–9.33 s (source 8.03–9.33 s)** |
| TCP peak speed / source | — | — | 0.86–1.16 |
| TCP mean speed (edited span) / source | — | — | 0.92–1.15 |
| Max joint speed / accel vs real envelope | ≤ 1.25 / 1.5 | ≤ 1.25 / 1.5 | 1.22 / 1.49 |

### Exports (`artifacts/sfagen_v03r2_run/{strict,loose}_{copied,right_wrist}`)

| Export | Synthetic-only | Mixed | Synthetic content |
|---|---|---|---|
| strict | 130 | 140 | 71 hand-over + 59 grid, 35 targets |
| loose | 138 | 148 | 71 hand-over + 67 grid, 39 targets |

All 4 export pairs pass `verify`, every quality gate (including `duration` and `speed_similarity`) and the gated
`check_exports`. 0 real episodes are in the synthetic-only sets.

### Coverage (40 eval starts, worst nearest can start in cm; `results/v03r2/coverage.png`)

| Set | Worst | Within 2 cm | ① front-right | ② rear-right | ③ rear-left | ④ front-left | ⑤ middle |
|---|---|---|---|---|---|---|---|
| real only | 6.63 | 14/40 | 2.98 | 6.63 | 5.59 | 3.30 | 3.33 |
| +v00 | 3.91 | 28/40 | 1.12 | 3.91 | 2.81 | 1.44 | 1.23 |
| +v03 rev 1 loose | 2.70 | 36/40 | 1.03 | 1.91 | 2.70 | 1.37 | 1.68 |
| **+v03 rev 2 strict** | 6.63 | 28/40 | 1.03 | **6.63** | 2.70 | 1.37 | 1.68 |
| **+v03 rev 2 loose** | **2.70** | **35/40** | 1.03 | 1.91 | 2.70 | 1.37 | 1.68 |

- **Strict leaves rear-right (②) uncovered:** no rear-right cell scores under 2.289.
- **Loose covers rear-right** with targets 20–22, from demos 2 and 8. Unlike rev 1, these episodes are **not**
  visually flagged.

### Visual audit (`results/v03r2/visual_audit.json`, `audit_sheets_grow/`)
- 13 flags, 11 of them inside strict:
  - notched can top at t=0 (sources 4/5/6, near the left arm);
  - small ghost sliver at lift (7/038, 7/048);
  - dark arc on the lid at grasp (source 9, targets 030–040).
- The ghost-lid crescent seen with the rev-1 renderer is gone after Amendment 5.
- The audit is single-rater and not blinded.

### Review sample (shown to the user before any upload)
- `results/v03r2/review/{strict,loose}/index.html`: real source vs synthetic side by side, with a running clock (corners,
  middle, largest shift, one hand-over variant).
- Viewer: `results/v03r2/viewer/index.html`.
