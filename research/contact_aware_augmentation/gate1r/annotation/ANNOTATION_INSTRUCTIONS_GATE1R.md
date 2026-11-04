# Gate 1R independent human annotation: instructions

**Purpose.** These annotations are the only ground truth for:

* **G1:** robot-landmark reprojection through FK and the body_cam model;
* **G2:** lid-tracker accuracy, **with coverage enforced per phase**;
* **EV:** the timing of visible contact events.

Without them the sealed episodes (1, 2, 3, 5, 6, 7, 9) cannot be tested. **Nothing is pre-filled, and no automatic
prediction is ever shown or used as ground truth.**

## Scope

* **Data:** Pringles `real_10_v00` @ `cebdee98`, body_cam, episodes **0, 4, 8** only. Do not open the sealed episodes.
* **Keypoints:** frames listed in `outputs/gate1r/annotation/manifest_keypoints.csv`:
  * key frames;
  * dense grasp/lift and transition windows, every 2 frames;
  * about 15 % shuffled repeats to measure your own repeatability.
* **Events:** 4 visible events per episode.
* **Time:** roughly 1.5–2 h for keypoints and 15 min for events per annotator.
* **Annotators:** two independent annotators are strongly preferred. Ground truth then requires agreement (≤ 6 px, ≤ 2
  frames). With one annotator the result is reported as single-annotator.

## Keypoints (in this order, on every frame)

| key | group | required | where |
|---|---|---|---|
| `obj_lid_center` | object | yes | centre of the pale circular lid; estimate from the rim if partly hidden |
| `obj_bottom_center` | object | no | centre of the can's bottom edge, if visible |
| `R_pad_tip_1` | right gripper | yes | tip of the right gripper's foam pad nearer the image **top** |
| `R_pad_tip_2` | right gripper | yes | tip of the other right pad |
| `R_wrist_tag` | right gripper | no | centre of the orange tag on the right wrist |
| `L_pad_tip_1` / `L_pad_tip_2` / `L_wrist_tag` | left gripper | yes / yes / no | the same for the left arm |

**Visibility and confidence** are required for every keypoint:

* **visible:** left-click. Press `1` / `2` / `3` *before* the click to set confidence:
  * 3 = within about 2 px (the default);
  * 2 = within about 5 px;
  * 1 = rough.
  Only confidence ≥ 2 enters ground truth.
* **occluded:** `o`. The point is in the image but hidden (by an arm, the other pad, the can).
* **out of frame:** `f`.

When in doubt, use a lower confidence or `o`. Occluded and out-of-frame points are never treated as negatives.

Other keys:
* `z`: undo;
* `n` / space: next frame (only once all 8 are answered);
* `p`: previous frame;
* `q`: save and quit.

Re-running resumes where you stopped.

## Events (per episode, by scrubbing the video)

| event | definition |
|---|---|
| `right_first_contact` | first frame the right pads visibly touch the can |
| `lift_off` | first frame the can is visibly off the table |
| `left_first_contact` | first frame the left pads visibly touch the can |
| `right_release` | first frame the right pads are visibly separated from the can |

* **Navigation:** `a` / `d` = ±1 frame, `w` / `s` = ±10 frames.
* **Marking:** set confidence with `1`–`3`, then `m` to mark the current frame, or `x` if the event is not observable.
  Command events are **not** shown. Judge only what you see.

## Commands

```bash
cd ~/Stanley_ws/lerobot_openarm
T=research/contact_aware_augmentation/gate1r/annotation/annotate_gate1r.py
O=research/contact_aware_augmentation/outputs/gate1r/annotation
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python"
$PY $T keypoints --annotator <ID> --manifest $O/manifest_keypoints.csv --out $O/keypoints_<ID>.csv
$PY $T events    --annotator <ID> --out $O/events_<ID>.csv
# afterwards
$PY research/contact_aware_augmentation/gate1r/annotation/evaluate_gate1r_annotations.py \
    --keypoints "$O/keypoints_*.csv" --events "$O/events_*.csv" \
    --gate1 research/contact_aware_augmentation/outputs/gate1 --out research/contact_aware_augmentation/outputs/gate1r/annotation_eval.json
```

**Provenance on every row:**
* annotator and session id;
* tool SHA-256 and repository commit;
* dataset revision;
* video SHA-256 and SHA-1 of the exact decoded frame shown;
* UTC time.

Do not edit the CSVs by hand. Corrections are made by re-annotating in the tool.

## Gates computed from your annotations (fixed in advance)

| gate | rule |
|---|---|
| G1 | jaw midpoint (mean of the two pad tips) and wrist tags vs FK, held out by episode: median ≤ 8 px and p90 ≤ 15 px, with ≥ 20 ground-truth points per gripper |
| G2 | tracker vs human lid centre: median ≤ 6 px **and** tracker-ok on ≥ 90 % of human-visible lid frames, separately for the grasp/lift and transition windows |
| EV | every event is ground truth or explicitly not observable for all 3 episodes |
| sealed test | allowed only when G1, G2 and EV pass **and** the sealed episodes' required windows have human annotations (a separate manifest, made after the pilot gates pass) |
