"""Dataset cards (synthetic-only + real+synthetic) for both tasks from the same template; every number is read from files."""
import json, sys
from pathlib import Path
import numpy as np
import yaml
S = Path(__file__).resolve().parents[1]
TASKS = {"pringles": dict(cfg="openarm_pringles_handover", syn="ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00",
                          mix="ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v00", name="bimanual hand-over", tag="handover",
                          prev_syn=("v04-65ep-two-pipelines", "7179061"), prev_mix=("v00-75ep-two-pipelines", "13d7231"),
                          objects="the Pringles can (a 7.5 × 23 cm cylinder)", target="hand-over point", keys=["can", "handover"],
                          held=[0, 2, 3, 4, 6, 7, 9], xb="xbench_pringles_final.json"),
         "battery": dict(cfg="openarm_battery_insertion", syn="ethanCSL/openarm_insertbattery_lerobot_real_to_sim_10_3cam_v00",
                         mix="ethanCSL/openarm_insertbattery_lerobot_real_and_sim_10_3cam_v00", name="bimanual battery insertion", tag="insertion",
                         prev_syn=("v01-65ep-v01", "3a977b0"), prev_mix=("v00-75ep-two-pipelines", "53de718"),
                         objects="the 18650 battery (a 1.85 × 6.5 cm cylinder) and the charging station (box, size from the image)", target="insertion point",
                         keys=["battery", "station", "insertion"], held=[3, 5, 8, 9], xb="xbench_battery_final.json")}
task = sys.argv[1]; t = TASKS[task]; W = S / f"configs/work/{t['cfg']}"
prov = json.loads((W / "dataset/meta/sfa_provenance.json").read_text()); eps = prov["episodes"]; ver = json.loads((W / "verification.json").read_text())
cov = json.loads((S / "logs/coverage.json").read_text()); xb = json.loads((S / f"logs/{t['xb']}").read_text())["summary"]
cal = json.loads((W / "calibration.json").read_text()); tab = json.loads((W / "table_height.json").read_text())
fam = {f: sum(e["family"] == f for e in eps) for f in ("object_position", "interaction_location")}
cfg = yaml.safe_load((S / f"configs/{t['cfg']}.yaml").read_text())
FRONT = "---\nlicense: apache-2.0\ntask_categories:\n- robotics\ntags:\n- LeRobot\n- openarm\n- data-augmentation\n- synthetic\n- bimanual\n- " + t["tag"] + "\nconfigs:\n- config_name: default\n  data_files: data/*/*.parquet\n---\n\n"


def cov_table():
    rows = ["| position (vs the other real recordings) | 10 sources | + previous revision | **+ this revision** |", "|---|---:|---:|---:|"]
    for k in t["keys"]:
        c = cov[f"{task}/{k}"]; n = c["n_other_real"]
        rows.append(f"| {k}: within 2 cm (of {n}) | {c['source']['within_2cm']} | {c['published']['within_2cm']} | **{c['unified']['within_2cm']}** |")
        rows.append(f"| {k}: area covered (cm²; other real recordings span {c['other_real_hull_cm2']:.0f}) | {c['source']['hull_cm2']:.0f} | {c['published']['hull_cm2']:.0f} | **{c['unified']['hull_cm2']:.0f}** |")
    worse = [k for k in t["keys"] if cov[f"{task}/{k}"]["unified"]["within_2cm"] < cov[f"{task}/{k}"]["published"]["within_2cm"]]
    note = ("\n\n**Lower than the previous revision:** " + ", ".join(worse) + " coverage. This pipeline rejects object positions where the "
            "arm, on its way to the object, would come closer to the object than in the recording (robot-proximity rule); the "
            "previous revision had no such rule. The previous revision stays loadable by its tag.") if worse else ""
    return "\n".join(rows) + note


method = f"""## How it is made: one pipeline for every task

Generated with [`sfagen`](https://huggingface.co/datasets/{t['syn']}/blob/main/sfagen/README.md), the same code and settings that
generate the {'battery-insertion' if task == 'pringles' else 'Pringles hand-over'} dataset. Only the task description differs: source
dataset, objects ({t['objects']}) and the shift grid. Full method: `PIPELINE.md`; positioning against prior work: `NOVELTY.md`.

1. **Rig calibration (automatic).**
   - **Body camera:** render-and-compare against the robot's own silhouette; held-out IoU {cal['heldout_iou']['initial']:.3f} → {cal['heldout_iou']['fitted']:.3f}.
   - **Table height:** {tab['z']:.3f} m ({tab['rule']}).
2. **Reconstruction (per source demo, no labels).**
   - **Robot masks:** SAM2.1, prompted from the projected URDF.
   - **Objects:** segmented as what disappears when picked up. Known-size cylinders are posed by grasp-constrained render-and-compare.
   - **Time offset:** each demo's video/state offset is chosen photometrically.
3. **Two variation families** (kinematic edits of the recorded motion):
   - **Object position** ({fam['object_position']} episodes): an object is moved ±3/±6 cm; the arm that grasps it follows (approach and grasp unchanged relative to the object) and rejoins the recording at the lift apex.
   - **Interaction location** ({fam['interaction_location']} episodes, the {t['target']}): the whole two-arm interaction is moved ±3/±5 cm; both hands and everything they hold move together, so every recorded contact relation is kept.
4. **Replay checks.** Every published episode passes all of them, on the commanded actions:
   - joint limits;
   - speed ≤ 1.25× and acceleration ≤ 1.5× the real demos;
   - arm–arm / arm–body mesh clearance and hand distance not below the source;
   - table clearance;
   - start pose;
   - untouched parts bit-identical;
   - IK exact at contact (≤ 1 mm / 0.5°);
   - object placement (in view, on the table, robot never closer to it than in the recording).
5. **Body camera: mesh-guided forward warping of real pixels** (no generative model). Every robot/object pixel moves in 3-D with its part, hidden surfaces are borrowed from other frames, and the background is relit. Frames where the scene equals the recording are the recorded frames.
6. **Wrist cameras.** Real recorded frames; `wrist_view_valid` and `wrist_view_pose_offset` say where they are exact.
"""

bench = f"""## Image fidelity, measured against real ground truth

**Method.** Re-render real frame j+k from real frame j only (k = 4/8/15), compared with the real frame inside a region that
depends on no pipeline. Held-out episodes {t['held']} ({xb['pairs']} pairs), never used for any tuning decision; 95% CI by episode bootstrap.

| renderer | PSNR (dB) | SSIM |
|---|---:|---:|
| copy the source frame (no motion) | {xb['copy_psnr']:.2f} | {xb['copy_ssim']:.3f} |
| previous revision's renderer | {xb['published_psnr']:.2f} | {xb['published_ssim']:.3f} |
| **this revision (sfagen)** | **{xb['unified_psnr']:.2f}** | **{xb['unified_ssim']:.3f}** |

Difference from the previous renderer: {xb['unified_minus_published_psnr'][0]:+.2f} dB [{xb['unified_minus_published_psnr'][1]:.2f}, {xb['unified_minus_published_psnr'][2]:.2f}], better in {100 * xb['unified_minus_published_psnr'][3]:.0f}% of pairs.
"""

limits = """## Limitations

- **Synthetic state.** `observation.state` is a planned kinematic candidate. Force, friction and contact are unchecked, and no episode has been replayed on hardware yet. Start with a ±3 cm episode, slowly.
- **Visible artifacts.** Arms re-posed close to the camera, or entering from the image edge, can show fragments. Watch the comparison videos.
- **Narrow variation.** Translations only (≤ 6 cm); no object rotation; the cable of the station (battery task) moves rigidly near the station.
- **Wrist cameras.** Not re-rendered.
- **No training evidence.** No policy has been trained on this revision yet.
"""

videos = f"""## Real vs synthetic, side by side

[highlights](https://huggingface.co/datasets/{t['syn']}/blob/main/comparison_videos/{task}_unified_highlights.mp4) ·
[all episodes, 2×](https://huggingface.co/datasets/{t['syn']}/blob/main/comparison_videos/{task}_unified_all_2x.mp4)
(left: real source; right: synthetic; the status bar says whether the frame is edited).
"""

rows = []
for e in eps:
    r = e["replay_checks"]; sh = e["shift_m"]
    rows.append(f"| {e['export_episode_index']} | {e['source_real_episode']} | `{e['variation']}` | {e['family'].replace('_', ' ')} | {e['n_frames']} | "
                f"({100 * sh[0]:+.0f}, {100 * sh[1]:+.0f}) | {e['frames_edited']} | {r['speed_ratio_to_real_max']:.2f} | {r['accel_ratio_to_real_max']:.2f} | {r['min_mesh_distance_mm']:.0f} |")
table = "| ep | source | variation | family | frames | shift (cm) | edited frames | max speed / real | max accel / real | min arm distance (mm) |\n|---:|---:|---|---|---:|---|---:|---:|---:|---:|\n" + "\n".join(rows)

syn_card = FRONT + f"""# {t['syn'].split('/')[1]}

OpenArm {t['name']} ("{prov['task']}"), three cameras (`body_cam`, `wrist_cam` = left wrist, `right_wrist_cam`), LeRobot v3.0, 30 fps.

**Every episode is synthetic.**
- **Count:** {ver['synthetic']['episodes']} episodes ({ver['synthetic']['frames']:,} frames).
- **Source:** generated without a simulator from the 10 real demonstrations of [`{cfg['source']['repo_id']}`](https://huggingface.co/datasets/{cfg['source']['repo_id']}) (revision `{cfg['source']['revision'][:8]}`), all {len(ver['synthetic']['sources_used'])} used.
- **Variation families:** {fam['object_position']} object-position + {fam['interaction_location']} interaction-location ({t['target']}).

The same episodes plus the 10 real demos: [`{t['mix']}`](https://huggingface.co/datasets/{t['mix']}).

## Revision history

| tag | content |
|---|---|
| `{t['prev_syn'][0]}` (`{t['prev_syn'][1]}`) | previous revision (task-specific pipeline) |
| current (`v3.0`) | **{ver['synthetic']['episodes']} episodes from the single sfagen pipeline shared by both OpenArm tasks** |

{method}
## What the synthetic data adds (positions by forward kinematics at gripper events; the other real recordings were never used for generation)

{cov_table()}

{bench}
{videos}
## Verification

- **Loading:** the dataset loads in LeRobot.
- **Labels:** state/action SHA-256 equal the plans.
- **Synthetic only:** {ver['synthetic']['episodes_identical_to_a_real_episode']} episodes equal a real episode.
- **Replay checks:** all pass.
- **Video:** every camera decodes with no black frames; the body camera matches the renders ({ver['synthetic']['observation.images.body_cam']['psnr_vs_rendered_db']} dB after encoding).

{limits}
## Episodes

{table}
"""
mix_card = FRONT + f"""# {t['mix'].split('/')[1]}

**Real + synthetic** OpenArm {t['name']} ("{prov['task']}"), three cameras, LeRobot v3.0, 30 fps.

| episodes | what |
|---|---|
| 0–9 | the 10 real demonstrations of [`{cfg['source']['repo_id']}`](https://huggingface.co/datasets/{cfg['source']['repo_id']}) @ `{cfg['source']['revision'][:8]}` (original videos, not re-encoded) |
| 10–{ver['mixed']['episodes'] - 1} | the {ver['synthetic']['episodes']} synthetic episodes of [`{t['syn']}`](https://huggingface.co/datasets/{t['syn']}) (v3.0) |

**Telling them apart.**
- **Per frame:** `is_synthetic` is 0 for real and 1 for synthetic.
- **Per episode:** `meta/sfa_provenance.json` gives each episode's source, family, shift and replay checks.

These extra features (`is_synthetic`, `wrist_view_valid`, `wrist_view_pose_offset`) have no `observation.` prefix, so policies ignore them. Use them to filter or re-weight.

**Checks.**
- **Real episodes:** labels identical to the source ({ver['mixed']['real_labels_identical_to_source']}).
- **Flag:** `is_synthetic` correct on every frame ({ver['mixed']['is_synthetic_flag_correct']}).

**Previous revision:** tag `{t['prev_mix'][0]}` (`{t['prev_mix'][1]}`).

{method}
{bench}
{videos}
{limits}"""
(W / "README_synthetic.md").write_text(syn_card); (W / "README_mixed.md").write_text(mix_card); print("ok", task, len(syn_card), len(mix_card))
