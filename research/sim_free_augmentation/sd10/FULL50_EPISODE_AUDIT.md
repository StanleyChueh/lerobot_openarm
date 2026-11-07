# SD10: audit of the 50-episode Pringles pool and ten-source selection

**Status:**
* PROPOSED source-diversity study. Source curation only.
* Selection executed; pending independent Codex review.
* No VLA training, augmentation, generation, robot or CAN use, or external push.

**Provenance:**
* Date 2026-11-07, Claude Code, branch `research/contact-aware-augmentation`.
* Protocol frozen at `f4ca482` before any per-episode feature was computed. Amendment A1 was appended before any selection ran (see `ten_source_selection_protocol.md`).

**Unchanged by this work:**
* the original-ten dataset;
* the v4A lock: `sha256sum -c` passes, seal `91eea62d…`;
* v3/v4B/v4B.1 results;
* all evaluation exclusions.

## 1. Data verification (STEP 1)

| item | result |
|---|---|
| Pool | `ethanCSL/openarm_pringles_lerobot_real_v00` @ `2c1bc860347c44fc0fbae7567469d36ee49baa8d`, equal to Hub `main` on 2026-11-07 |
| Original ten | `ethanCSL/openarm_pringles_lerobot_real_10_v00` @ `cebdee98115ab6827a4ec2655335c536b5dd085b`, equal to Hub `main` |
| File integrity | Every local file's SHA-256 equals the Hub LFS SHA-256, or its git blob SHA-1 for small files. Pool wrist MP4s were read from the v4A read-only fetch at the same revision; their hashes were verified. |
| Episodes / frames | 50 / 12,115 (lengths 188–282, median 248.5), 30 fps; original ten 10 / 2,575 |
| Cameras | `body_cam`, `wrist_cam` (left), `right_wrist_cam`; AV1 640×480 in both datasets |
| State / action | 16-D float32 (RJ1, LJ1, …, RJ8, LJ8), finite |
| Tabular | Contiguous frame indices; timestamps = index/30 (error < 1 ms); episode/global indices consistent; video windows equal the lengths |
| Packets | For all 3 streams × 50 episodes (and the ten): packet count = length, regular PTS |
| Pixels | Body: all 50 decoded. Wrists: 44 non-locked decoded. No black or white frames. Isolated duplicated frames, 2–18 per episode summed over streams, never 2 in a row (longest run 1). |
| v4A wrist exclusion | The wrist streams of records 10/17/24/31/38/45 were checked at packet/PTS level only. Pixels were never decoded: `Dataset.decode` refuses them. |

## 2. Duplication and provenance

* **The original ten are pool episodes 0–9.**
  * For each original-ten episode *i*, the state and action arrays are bit-identical to pool episode *i*. It is the unique exact match among the 50, even where other pool episodes have the same length (3↔4, 5↔14/26, 8↔11/34, 9↔24).
  * The **videos are separately encoded**: no compressed packet and no decoded frame is identical, and the mean |Δ| is 0.9–1.2 grey levels per pixel.
  * They are the same captures at the same alignment. On every one of the 30 camera pairs the zero frame shift is best, and a ±1 shift is ≥ 3.99× worse.
* **Consequence:** do not assume byte-identical RGB between the two datasets. Any pixel-level work should name which encoding it uses.
* **Within the pool:** no exact state/action duplicates and no identical compressed video. The nearest trajectories differ by 0.043 rad RMS (episodes 37↔38; median nearest-neighbour 0.058 rad).

## 3. Measurements (STEP 2)

| # | quantity | method | status |
|---|---|---|---|
| 1 | Initial can centre in body pixels | Frame-0 lid-ellipse centre: a SAM2.1-small whole-can mask prompted by the existing verified human click, then a box-prompted lid-only SAM mask and an ellipse fit (A1). All 50 accepted on visual QA. | Lid fill ratio 0.85–1.00. The click is within 25 px of the lid centre for 48/50; the clicks for 11 (90 px) and 38 (29 px) are on the body. |
| 2 | Metric 3D position | **Not estimated.** The body-camera extrinsic is an inherited, unvalidated silhouette fit; v4A refused it. | Image proxy only. Cross-check: lid v vs FK right-grasp x has ρ = −0.99, and lid u vs grasp y has ρ = −0.97. The ordering is consistent with kinematics; metric accuracy is unknown. |
| 3 | Right grasp pose | FK (pinocchio, follower URDF) of measured state at the commanded right close `rc`: TCP xyz, approach and yaw. | Grasp x 0.09–0.40 m, y −0.08–0.11 m, yaw 13–79°. |
| 4 | Left receiving pose | FK at the left close `lc`: left TCP xyz and yaw; inter-TCP separation. | Left x 0.19–0.27, y −0.03–0.03, z 0.46–0.53 m; separation 0.108–0.162 m. |
| 5 | Handover timing | Command events `rc < lc ≤ ro` (openness crossing 0.5); measured hold and release onsets from the gripper state. | Grasp→receive 1.77–3.00 s (median 2.37). Command overlap ro − lc 0–14 frames (median 4). The measured dual-hold is short (−3 to 12 frames, median 1), with the usual ~2-frame action lead. |
| 6 | Evidence of transfer | (a) **Measured** gripper state, not commands: right closure stops at 0.40–0.44 openness (blocked by the can, not empty ≈0), released at the end; left stops at 0.34–0.38 and holds to the end with drift ≤ 0.024. (b) **Manual keyframe review** (`annotations/manual_transfer_review.json`). | All 50 SUCCESS. 44 judged from body plus both wrists, 6 locked from body only. **Not force validation.** |
| 7 | Visibility / blur | Laplacian-variance sharpness of the frame-0 can crop (254–971) and of the wrists at rc/lc (R 4–22, L 1–32; the wrists are close-up and often defocused). Can tracked by SAM2.1 video propagation. | The tracked visibility (1.00 everywhere) does not discriminate and is reported only. Cables partly cross the lid at frame 0 in 18, 26 and 31 (also near 5 and 9). |
| 8 | Camera / data integrity | §1. | All 50 pass. |

**Visual notes:**
* In 6, 32, 33, 38, 44 and 47 the can visibly tilts in the left grasp after the right release.
* Episode 12 starts at the far image-top edge.

**Recording-order trend:** later recordings are faster (length vs index ρ = −0.68; grasp→receive time ρ = −0.77). Mean length is 257.5 frames for episodes 0–9, 248.7 for 10–39 and 207.8 for 40–49. The original ten are the earliest, slowest recordings.

**Contact sheets:**
* `figures/contact_sheet_full50.jpg`: all 50 candidates with the measured lid centre and status.
* `figures/contact_sheet_original_ten.jpg` and `figures/contact_sheet_sd10.jpg`: start, rc+20, lc and ro+10, the wrist views, and the measured features.
* QA sheets (regenerable, not committed): `outputs/review/`.

### Per-episode audit (full table: `full50_episode_features.csv`)

| ep | role | lid (u, v) px | T | rc / lc / ro | grasp→receive s | right grasp TCP xyz (m) | left receive TCP xyz (m) | grip open R / L | Q1–Q6 | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | orig10 | (372, 204) | 273 | 97 / 176 / 181 | 2.63 | (0.240, 0.027, 0.287) | (0.272, -0.007, 0.461) | 0.40 / 0.34 | PASS |  |
| 1 | **SD10** orig10 | (403, 145) | 263 | 87 / 172 / 176 | 2.83 | (0.305, 0.006, 0.292) | (0.254, -0.031, 0.467) | 0.43 / 0.35 | PASS |  |
| 2 | orig10 | (322, 292) | 280 | 101 / 188 / 198 | 2.90 | (0.152, 0.089, 0.308) | (0.243, -0.033, 0.490) | 0.43 / 0.35 | PASS |  |
| 3 | orig10 | (346, 208) | 250 | 82 / 156 / 165 | 2.47 | (0.234, 0.057, 0.302) | (0.226, -0.013, 0.507) | 0.43 / 0.34 | PASS |  |
| 4 | orig10 | (390, 134) | 250 | 87 / 163 / 168 | 2.53 | (0.318, 0.013, 0.298) | (0.250, -0.001, 0.494) | 0.42 / 0.36 | PASS |  |
| 5 | orig10 | (334, 123) | 255 | 86 / 164 / 169 | 2.60 | (0.334, 0.059, 0.294) | (0.226, -0.011, 0.493) | 0.42 / 0.34 | PASS | cable crosses/near the can at frame 0 |
| 6 | **SD10** orig10 | (349, 145) | 241 | 82 / 161 / 166 | 2.63 | (0.309, 0.053, 0.284) | (0.262, 0.022, 0.487) | 0.43 / 0.34 | PASS | can visibly tilted in the left grasp after the right release |
| 7 | **SD10** orig10 | (324, 235) | 261 | 87 / 170 / 179 | 2.77 | (0.209, 0.080, 0.302) | (0.237, -0.020, 0.510) | 0.42 / 0.35 | PASS |  |
| 8 | orig10 | (322, 356) | 249 | 83 / 160 / 164 | 2.57 | (0.126, 0.076, 0.303) | (0.216, -0.019, 0.512) | 0.43 / 0.35 | PASS |  |
| 9 | orig10 | (295, 215) | 253 | 85 / 159 / 168 | 2.47 | (0.233, 0.093, 0.310) | (0.234, -0.029, 0.508) | 0.42 / 0.38 | PASS | cable crosses/near the can at frame 0 |
| 10 | v4A-locked | (445, 296) | 229 | 89 / 155 / 158 | 2.20 | (0.176, -0.014, 0.279) | (0.219, 0.002, 0.514) | 0.42 / 0.37 | PASS (Q5 body-only) |  |
| 11 | **SD10** | (345, 92) | 249 | 86 / 158 / 168 | 2.40 | (0.361, 0.044, 0.290) | (0.229, 0.002, 0.512) | 0.43 / 0.38 | PASS | human click on the body, not the lid |
| 12 | eligible | (338, 37) | 245 | 88 / 165 / 174 | 2.57 | (0.403, 0.040, 0.308) | (0.246, -0.012, 0.499) | 0.43 / 0.37 | PASS | can at the far (top) edge of the workspace at frame 0; can at the far image-top edge |
| 13 | **SD10** | (267, 351) | 279 | 103 / 188 / 194 | 2.83 | (0.137, 0.112, 0.306) | (0.217, 0.000, 0.525) | 0.43 / 0.34 | PASS |  |
| 14 | **SD10** | (439, 368) | 255 | 87 / 164 / 168 | 2.57 | (0.140, -0.013, 0.285) | (0.215, 0.008, 0.518) | 0.43 / 0.36 | PASS |  |
| 15 | eligible | (326, 189) | 256 | 85 / 162 / 167 | 2.57 | (0.271, 0.065, 0.294) | (0.217, -0.011, 0.516) | 0.43 / 0.37 | PASS |  |
| 16 | eligible | (318, 296) | 252 | 86 / 171 / 174 | 2.83 | (0.168, 0.074, 0.297) | (0.199, -0.012, 0.513) | 0.43 / 0.38 | PASS |  |
| 17 | v4A-locked | (445, 253) | 237 | 80 / 151 / 155 | 2.37 | (0.230, -0.037, 0.286) | (0.235, 0.025, 0.500) | 0.43 / 0.35 | PASS (Q5 body-only) |  |
| 18 | eligible | (323, 147) | 247 | 85 / 149 / 159 | 2.13 | (0.305, 0.054, 0.296) | (0.216, 0.010, 0.500) | 0.43 / 0.36 | PASS | cable crosses/near the can at frame 0; cable partially over the lid |
| 19 | eligible | (365, 237) | 256 | 86 / 157 / 163 | 2.37 | (0.231, 0.035, 0.303) | (0.225, -0.002, 0.510) | 0.43 / 0.38 | PASS |  |
| 20 | eligible | (388, 317) | 266 | 92 / 173 / 176 | 2.70 | (0.175, 0.033, 0.296) | (0.232, -0.014, 0.518) | 0.43 / 0.36 | PASS |  |
| 21 | eligible | (331, 287) | 248 | 85 / 158 / 163 | 2.43 | (0.200, 0.060, 0.299) | (0.209, -0.014, 0.511) | 0.42 / 0.36 | PASS |  |
| 22 | eligible | (362, 410) | 276 | 101 / 177 / 181 | 2.53 | (0.091, 0.047, 0.308) | (0.207, 0.009, 0.511) | 0.43 / 0.36 | PASS |  |
| 23 | eligible | (432, 329) | 252 | 87 / 162 / 166 | 2.50 | (0.154, -0.012, 0.278) | (0.241, 0.034, 0.509) | 0.42 / 0.35 | PASS |  |
| 24 | v4A-locked | (359, 117) | 253 | 86 / 159 / 162 | 2.43 | (0.339, 0.025, 0.303) | (0.243, 0.011, 0.502) | 0.43 / 0.35 | PASS (Q5 body-only) |  |
| 25 | eligible | (369, 152) | 242 | 84 / 147 / 160 | 2.10 | (0.292, 0.027, 0.292) | (0.224, 0.008, 0.508) | 0.43 / 0.34 | PASS |  |
| 26 | eligible | (317, 131) | 255 | 83 / 153 / 167 | 2.33 | (0.334, 0.061, 0.318) | (0.238, 0.021, 0.514) | 0.42 / 0.35 | PASS | cable crosses/near the can at frame 0; cable partially over the lid |
| 27 | **SD10** | (459, 157) | 230 | 84 / 148 / 152 | 2.13 | (0.321, -0.076, 0.287) | (0.245, 0.018, 0.495) | 0.43 / 0.35 | PASS |  |
| 28 | eligible | (285, 351) | 264 | 92 / 169 / 171 | 2.57 | (0.118, 0.103, 0.304) | (0.192, -0.010, 0.522) | 0.42 / 0.36 | PASS |  |
| 29 | eligible | (253, 389) | 282 | 96 / 186 / 191 | 3.00 | (0.095, 0.107, 0.324) | (0.213, 0.001, 0.520) | 0.40 / 0.34 | PASS |  |
| 30 | eligible | (248, 293) | 265 | 90 / 169 / 175 | 2.63 | (0.178, 0.119, 0.297) | (0.232, -0.009, 0.521) | 0.42 / 0.35 | PASS |  |
| 31 | v4A-locked | (286, 189) | 270 | 91 / 162 / 167 | 2.37 | (0.257, 0.094, 0.291) | (0.207, -0.002, 0.528) | 0.42 / 0.37 | PASS (Q5 body-only) | cable crosses/near the can at frame 0; cable partially over the lid |
| 32 | eligible | (443, 299) | 239 | 85 / 149 / 151 | 2.13 | (0.189, -0.009, 0.288) | (0.229, -0.002, 0.506) | 0.43 / 0.35 | PASS | can visibly tilted in the left grasp after the right release |
| 33 | eligible | (393, 191) | 231 | 76 / 139 / 153 | 2.10 | (0.262, 0.018, 0.283) | (0.226, 0.018, 0.517) | 0.42 / 0.36 | PASS | can visibly tilted in the left grasp after the right release |
| 34 | **SD10** | (358, 349) | 249 | 87 / 156 / 159 | 2.30 | (0.142, 0.052, 0.293) | (0.212, 0.002, 0.519) | 0.43 / 0.38 | PASS |  |
| 35 | eligible | (334, 277) | 238 | 76 / 141 / 142 | 2.17 | (0.190, 0.050, 0.298) | (0.225, -0.006, 0.521) | 0.41 / 0.38 | PASS |  |
| 36 | eligible | (326, 223) | 236 | 89 / 156 / 158 | 2.23 | (0.246, 0.074, 0.292) | (0.220, 0.009, 0.532) | 0.43 / 0.38 | PASS |  |
| 37 | eligible | (385, 180) | 225 | 87 / 153 / 159 | 2.20 | (0.291, 0.004, 0.297) | (0.231, 0.007, 0.514) | 0.42 / 0.37 | PASS |  |
| 38 | v4A-locked | (407, 214) | 213 | 80 / 144 / 148 | 2.13 | (0.242, -0.006, 0.291) | (0.251, 0.019, 0.526) | 0.43 / 0.38 | PASS (Q5 body-only) | can visibly tilted in the left grasp after the right release; human click on the body, not the lid |
| 39 | eligible | (367, 244) | 223 | 89 / 155 / 157 | 2.20 | (0.214, 0.026, 0.296) | (0.236, 0.014, 0.516) | 0.43 / 0.37 | PASS |  |
| 40 | eligible | (384, 352) | 227 | 85 / 151 / 151 | 2.20 | (0.129, 0.029, 0.295) | (0.221, 0.015, 0.518) | 0.42 / 0.36 | PASS |  |
| 41 | eligible | (371, 206) | 220 | 83 / 149 / 149 | 2.20 | (0.253, 0.037, 0.306) | (0.239, 0.002, 0.519) | 0.42 / 0.37 | PASS |  |
| 42 | **SD10** | (341, 217) | 188 | 64 / 121 / 123 | 1.90 | (0.252, 0.058, 0.294) | (0.224, 0.004, 0.522) | 0.43 / 0.38 | PASS |  |
| 43 | eligible | (325, 231) | 203 | 65 / 125 / 126 | 2.00 | (0.222, 0.059, 0.305) | (0.234, -0.011, 0.525) | 0.43 / 0.38 | PASS |  |
| 44 | eligible | (375, 127) | 205 | 76 / 139 / 140 | 2.10 | (0.322, 0.018, 0.295) | (0.227, 0.034, 0.516) | 0.43 / 0.35 | PASS | can visibly tilted in the left grasp after the right release |
| 45 | v4A-locked | (339, 168) | 206 | 76 / 133 / 135 | 1.90 | (0.288, 0.043, 0.306) | (0.255, 0.015, 0.504) | 0.42 / 0.35 | PASS (Q5 body-only) |  |
| 46 | eligible | (339, 256) | 216 | 77 / 133 / 134 | 1.87 | (0.223, 0.049, 0.297) | (0.214, -0.002, 0.536) | 0.42 / 0.34 | PASS |  |
| 47 | **SD10** | (417, 283) | 200 | 76 / 129 / 130 | 1.77 | (0.194, -0.004, 0.292) | (0.242, 0.030, 0.530) | 0.43 / 0.37 | PASS | can visibly tilted in the left grasp after the right release |
| 48 | eligible | (372, 177) | 192 | 70 / 124 / 127 | 1.80 | (0.278, 0.006, 0.306) | (0.242, 0.024, 0.536) | 0.44 / 0.38 | PASS |  |
| 49 | eligible | (334, 266) | 221 | 81 / 142 / 145 | 2.03 | (0.213, 0.051, 0.296) | (0.231, 0.008, 0.532) | 0.43 / 0.36 | PASS |  |

## 4. Selection (STEP 3)

**Gates:**
* All 50 episodes pass Q1–Q6. Exclusions: only the 6 v4A-locked records (10, 17, 24, 31, 38, 45), on protocol grounds rather than quality.
* **44 eligible.**

**Rule** (protocol §4, A1 position feature):
* Distance on normalised (lid u, lid v) plus 0.25 × (left-receive x, y, z, grasp→receive time).
* Exact k-center (k = 10) → radius-constrained p-median → no-good-cut tie check (scipy `milp`/HiGHS).
* Random-ten seed 20261107; null draws seed 20261108 × 10,000.

**Result: SD10 = [1, 6, 7, 11, 13, 14, 27, 34, 42, 47]**
* Overlap with the original ten: 1, 6, 7.
* Optimal k-center radius R* = 1.430 normalised units; p-median cost 35.2235.
* **Tie disclosure:** the subset with 0 instead of 1 has an identical coverage cost. Episodes 0 and 1 form a two-point cluster. The protocol's pre-registered −1e-4·q quality tie-break chose 1 (q 0.35 vs 0.25). No stage-3 exact tie remained.
* Not only outliers:
  * 2/10 lie on the eligible convex hull;
  * 2 centres serve only themselves (27, 47);
  * the others represent clusters of 2–12 episodes. Episode 42 represents the 12-episode late-session cluster.

### Position coverage (body pixels; targets = all 50 pool can positions)

| metric | original ten | **SD10** | random ten | random-ten null median [5–95 %] |
|---|---|---|---|---|
| covering radius (px) ↓ | 118 (better than 31 % of random) | **65** (better than 100 %) | 95 | 112 [87, 151] |
| mean nearest source (px) ↓ | 36.0 (27 %) | **28.9** (94 %) | 34.5 | 33.4 [28.6, 41.4] |
| within 40 px ↑ | 72 % | **74 %** | 58 % | 66 % [54, 74] |
| within 20 px ↑ | **38 %** | 30 % | 30 % | 34 % [28, 42] |
| hull area (k px²); fraction of pool hull | 12.4; 0.23 (7th pct) | **35.7; 0.66** (96th pct) | 25.5; 0.47 | 22.3 [11.4, 35.3] |
| min source spacing (px) | 17 | **26** | 5 | 16.5 [4.8, 32.8] |
| v4A-locked six: covering radius (px) | 117 | **65** | 73 | — |
| v4A-locked six within 40 px | **4/6** | 3/6 | 3/6 | — |

**Reading:**
* SD10 halves the worst-case gap (118 → 65 px) and triples the covered hull, at the cost of fewer very-close (≤ 20 px) matches in the dense central cluster.
* The original ten are narrow: 7th-percentile hull, worse than a typical random ten on worst-case coverage, concentrated in the upper-centre of the image.
* Near the six locked v4A positions, the original ten have more within 40 px (4 vs 3), but SD10 has a much smaller worst case.

**Interaction and quality** (`outputs/selection_details.json`; `figures/source_diversity_comparison.png`):

| | original ten | SD10 | random ten |
|---|---|---|---|
| grasp→receive time (s) | 2.47–2.90 (sd 0.14) | 1.77–2.83 (sd 0.36) | 2.00–3.00 (sd 0.33) |
| episode length (frames) | 241–280 | 188–279 | 203–282 |
| right-grasp y (m) | 0.006–0.093 | −0.076–0.112 | −0.009–0.107 |
| left-receive x / y / z spread (sd, cm) | 1.7 / 1.6 / 1.7 | 1.6 / 1.8 / 1.9 | 1.4 / 1.7 / 1.1 |
| mean quality q (tie-break score) | 0.43 | 0.49 | 0.56 |
| all gates pass | yes | yes | yes |

**Interaction diversity is modest across all sets.** The left-receive location varies by only about ±3–4 cm across the whole pool (x 0.19–0.27, y −0.03–0.03, z 0.46–0.53 m); the operator received at almost the same place every time. SD10 mainly adds diversity in **timing and right-grasp location**, not in receiver geometry. Real interaction variation (receiver grasp, dual-hold) is not available in this pool; that remains a v4C generation target.

## 5. Evaluation leakage risks (STEP 4)

1. **v4A:** records 10/17/24/31/38/45 are excluded from SD10 and none was selected. The v4A lock files are unchanged (`sha256sum -c` passes; seal `91eea62d…`). Their wrist pixels were never decoded here.
2. **Reuse of recordings 10–49:** SD10 takes 7 sources from 10–49 (11, 13, 14, 27, 34, 42, 47). These recordings were the v3–v4B.1 development evaluation pool.
   * A model trained on SD10 **must not** be evaluated on any of these 7.
   * SD10 results are **not comparable** to the frozen v4B/v4B.1 40-recording scores.
   * The remaining 10–49 recordings are development data, not an untouched test.
3. **The 40 px "far" bin** in v4B was defined relative to the original ten. Under SD10 it changes meaning, so any new study must re-declare its own bins.
4. **Selection–evaluation coupling:** coverage was optimised over the 44 eligible positions, so pool-coverage metrics are in-sample by construction. Generalisation needs prospectively recorded, locked episodes; that is separately authorised operator work.
5. **Recording-order confound:** SD10 includes faster, later-session episodes (42 and 47, lengths 188/200). Any original-ten-vs-SD10 difference mixes position coverage with operator pace and session drift. A random ten and a stratified or time-matched ten should be reported alongside it.

## 6. Is the selection scientifically defensible?

**Yes, as a source-curation step, with these caveats:**
* quality gates and selection rule frozen before features, deterministic, no downstream scores, all exclusions and tie-breaks disclosed;
* one documented pre-selection amendment (A1, a measurement-validity fix);
* the image-space position proxy is consistent with independent kinematics (|ρ| ≥ 0.97).

**Limits:**
* Positions are image coordinates, not metric.
* The quality gates were not discriminative: all 50 passed. "Quality-first" therefore had no practical effect beyond the v4A exclusion and the 0-vs-1 tie-break.
* The transfer review was done by a model annotator, not a human. It is backed by measured gripper-state evidence but is not force validation.
* Receiver-interaction diversity in the pool is small.
* Selecting 10 of 50 supports a "10 input demonstrations" claim, not a "10 collected demonstrations" acquisition-cost claim.
* Diversity-based demonstration selection is prior art, not a novelty.

**Not established:** any effect of SD10 on augmentation quality or policy performance. That needs the separately planned 2×2 study (original/SD10 × real-only/augmented, plus a random ten) on fresh, locked evaluation recordings.

## 7. Reproduce

```bash
cd research/sim_free_augmentation/sd10
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python"
env -u PYTHONPATH -u LD_LIBRARY_PATH ../../../.venv/bin/python scripts/verify_data.py   # Hub metadata read (or --offline)
$PY scripts/verify_alignment.py
$PY scripts/extract_features.py        # GPU, SAM2.1-small, ~4 min
$PY scripts/lid_center.py              # A1 lid centres + QA sheet
$PY scripts/make_review_sheets.py      # Q5/Q6 review sheets
$PY scripts/build_features.py          # -> full50_episode_features.csv
$PY scripts/select10.py                # -> selected_10_episodes.json, outputs/selection_details.json
$PY scripts/figures.py
$PY scripts/make_manifest.py           # -> source_provenance_manifest.json (+ v4A lock check)
```

SAM inference runs in bf16, so lid centres may differ at sub-pixel level across GPUs. The selection depends only on `full50_episode_features.csv`, which is committed.
