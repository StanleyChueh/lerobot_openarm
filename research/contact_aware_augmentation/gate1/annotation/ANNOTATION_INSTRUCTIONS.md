# Gate 1 human annotation: instructions

**Purpose.** These clicks are the only human ground truth Gate 1 will use. They are needed for two of its criteria:

* **G1:** robot-keypoint reprojection through FK and the body_cam model; median ≤ 8 px, p90 ≤ 15 px, held out by episode.
* **G2:** accuracy of the automatic lid tracker; median ≤ 6 px.

They also let the verifier be re-scored on human lid points, which separates score-model error from tracker error.
**Nothing here is filled in automatically.** The tracker, FK and Claude never write ground truth.

## What to annotate

* **Episodes and frames:** 137 frames from `outputs/gate1/annotation/annotation_manifest.csv`. They are body_cam frames of
  Pringles `real_10_v00` (rev `cebdee98`), episodes **0, 4, 8** only. Episodes 1, 2, 3, 5, 6, 7, 9 are sealed for a later
  test: **do not view them.**
* **What the 137 frames are:**
  * 8 key frames per episode;
  * dense windows every 2 frames around grasp/lift and around the take-over;
  * 18 repeats of earlier frames, shuffled to the end, to measure your own repeatability. Answer them honestly, without
    looking back.
* **Time:** about 45–60 minutes for one annotator. A second annotator, at least on the dense take-over frames, makes
  disagreement measurable.

## The five keypoints (in this order on every frame)

| # | key | where to click |
|---|---|---|
| 1 | `lid_center` | centre of the can's pale circular lid. If part of the rim is hidden, estimate the centre of the full ellipse. |
| 2 | `right_wrist_tag` | centre of the **orange** plastic tag on the robot's **right** arm (the arm on the image right at frame 0) |
| 3 | `left_wrist_tag` | centre of the orange tag on the **left** arm |
| 4 | `right_jaw_mid` | midpoint between the **tips** of the right gripper's two foam pads |
| 5 | `left_jaw_mid` | the same for the left gripper |

**Visibility codes (required for every keypoint):**

* **visible:** you can place the point within about 3 px. Just left-click.
* **uncertain:** you can see roughly where it is, but not within about 3 px. This covers motion blur, partial occlusion
  over more than half of the lid rim, a tag seen edge-on, or pads you cannot tell apart. Press `u`, then click.
  **Uncertain points are excluded from ground truth.**
* **occluded:** not in view or out of the frame. Press `o` (no click). Occluded points are never counted as negatives.

When unsure between visible and uncertain, choose **uncertain**. Fewer, reliable points are better than many doubtful ones.

## Running the tool

```bash
cd ~/Stanley_ws/lerobot_openarm
env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python research/contact_aware_augmentation/gate1/annotation/annotate_gate1.py \
    annotate --annotator <your initials> \
    --manifest research/contact_aware_augmentation/outputs/gate1/annotation/annotation_manifest.csv \
    --out research/contact_aware_augmentation/outputs/gate1/annotation/annotations_<your initials>.csv
```

* The window shows the frame at 2× size. The top bar names the keypoint to click next.
* **Keys:**
  * left click: place the keypoint;
  * `u`: mark the next click as uncertain;
  * `o`: occluded;
  * `z`: undo;
  * `n` / space: next frame (only when all 5 are answered);
  * `p`: previous frame;
  * `q`: save and quit.
* The CSV is saved after every frame. Re-running the same command resumes where you stopped.
* The tool is blind: it never shows the tracker, FK projections or another person's clicks. Do not open
  `outputs/gate1/figures/` before you finish.

## Afterwards

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH .venv/bin/python research/contact_aware_augmentation/gate1/annotation/evaluate_annotations.py \
    --annotations "research/contact_aware_augmentation/outputs/gate1/annotation/annotations_*.csv" \
    --gate1 research/contact_aware_augmentation/outputs/gate1 \
    --out research/contact_aware_augmentation/outputs/gate1/annotation_eval.json
python3 research/contact_aware_augmentation/gate1/make_gate1_metrics.py
```

The ground-truth rule was fixed in `evaluate_annotations.py` before any annotation existed:

* only `visible` answers count;
* with two or more annotators, a point is kept only if all of them marked it visible within 6 px of each other;
* repeats measure repeatability and are not double-counted.

The robot keypoints' link attachment (link5/6/7 or hand_tcp) and 3D offset are chosen on the two development episodes
of each fold. They are then scored on the held-out episode, using the video-to-state lag already frozen for that fold.
