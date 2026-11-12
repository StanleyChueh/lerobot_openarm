"""Dataset cards (Hub README.md) for the v01 few-source candidates, filled from the quality report, the frozen selection
and the wrist gate. usage: make_cards_v01.py N [N ...]   (N or N_wristrender)"""
WRIST_MAIN = ("copied: the time-remapped real source frame. The registered wrist gate failed for Pringles, so the frames are not "
              "re-rendered, and `wrist_view_valid` = 0 wherever the scene changed.")
WRIST_AB = ("both cameras re-rendered by the moving-camera warp (study B), ablation. `wrist_view_valid` = 1 where a frame "
            "is an exact copy or has <= 5 % inpainted pixels.")
BANNER = """
> **Ablation.** Here both wrist cameras are **re-rendered** (sfagen.wrist), although the registered wrist gate FAILED
> for Pringles (it beat copying within episodes, +0.8 to +1.0 dB, but not reliably across episodes at camera moves of
> 3 cm or more). It is otherwise identical to `..._{n}_3cam_v01`, which has copied wrist frames and validity flags.
> Train both to test whether rendered wrist views help.
"""
import json, subprocess, sys
from pathlib import Path
S = Path(__file__).resolve().parents[1]; R = S / "results"


def card(n, v=""):
    q = json.loads((R / f"datasets_v01/fewsrc{n}{v}/quality_report.json").read_text()); sel = json.loads((R / "fewsrc/datasets.json").read_text())[str(n)]
    gate = {}
    for p in sorted(R.glob("wrist/final_*/summary.json")):
        for cam, r in json.loads(p.read_text())["cameras"].items():
            gate.setdefault(cam, []).append((p.parent.name.replace("final_", ""), r["gate"]["pass"], r["T1"]["render_minus_copy_db"], r["T1"]["ci95"]))
    sha = subprocess.run(["git", "-C", str(S), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    t, im = q["trajectories"], q["images"]; wr = im["wrist_rendered_cameras"]
    wl = "\n".join(f"| `{cam}` | " + "; ".join(f"{k}: {'pass' if ok else 'FAIL'} ({m:+.2f} dB [{ci[0]:+.2f}, {ci[1]:+.2f}])" for k, ok, m, ci in v) +
                   f" | {'re-rendered' if cam in wr else 'copied, invalid where the scene changed'} | {im['wrist_view_valid_frac'][cam]:.2f} |" for cam, v in gate.items())
    return f"""---
license: apache-2.0
task_categories: [robotics]
tags: [LeRobot, openarm, synthetic, simulation-free, few-shot]
configs:
- config_name: default
  data_files: data/*/*.parquet
---
# OpenArm Pringles hand-over: synthetic-only, from {n} real demo{'s' if n > 1 else ''} (sfagen v01 candidate{", rendered-wrist ABLATION" if v else ""})
{BANNER.format(n=n) if v else ""}
**{q['episodes']} synthetic episodes / {q['frames']} frames, 3 cameras, 30 fps, LeRobot v3.0.** These are generated from the real demos
{sel['sources']} of [`ethanCSL/openarm_pringles_lerobot_real_10_v00@cebdee98`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_10_v00).
The dataset contains **no real episodes**. No simulator and no learned generator were used.
**Candidate for a SmolVLA few-source screen; not yet validated by policy training or on the robot.**

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("ethanCSL/openarm_pringles_lerobot_real_to_sim_{n}_3cam{v}_v01", revision="v3.0")
# mask wrist cameras where batch["wrist_view_valid"] == 0 (order: wrist_cam, right_wrist_cam)
```

## How it was made (sfagen, code commit `{sha[:12]}`, uploaded under `sfagen/`)
- **Demos:** chosen by a rule frozen before any evaluation: k-center over the recorded can positions. The order is
  3, 8, 5, 1, …, so N=1 → {{3}}, N=2 → {{3, 8}}, N=4 → {{1, 3, 5, 8}}.
- **Variation (F1W):** the can starts elsewhere on the table, **shifted at most 11 cm**. That is the measured
  body-camera limit r* (study R1). The grasping arm's reach is translated rigidly by IK, and the time is retimed to fit
  the speed and acceleration envelope of **{'this demo' if n == 1 else f'these {n} demos'} only**. Every variant that passes that envelope is kept.
- **Body camera:** mesh-guided forward warping of the real pixels, with a clean plate and real-pixel borrowing. This is
  the published renderer, unchanged: the can-fix candidates did not pass their registered test (study A).
- **Wrist cameras:** {WRIST_AB if v else WRIST_MAIN}

## Quality gate (recomputed from this export; `quality_report.json`)
| Check | Result |
|---|---|
| finite, joint limits, start state = source, grippers bit-identical to source | {'pass' if all(t['gate'][k] for k in ('finite', 'joint_limits', 'start_state', 'gripper')) else 'FAIL'} |
| max speed / acceleration ratio vs the {n}-demo envelope (limits 1.25 / 1.5) | {t['max_speed_ratio']:.2f} / {t['max_accel_ratio']:.2f} |
| max jerk ratio (reported) | {t['max_jerk_ratio']:.2f} |
| lowest arm point vs source (table) | {t['min_lowest_z_m']:.4f} m vs {t['source_lowest_z_m']:.4f} m |
| labels = generator labels (SHA-256), no episode equals a real one | {'pass' if t['gate']['labels_sha'] and t['gate']['no_real_episode'] else 'FAIL'} |
| video spans, decoding, body decoded vs rendered | {'pass' if im['verify_passed'] else 'FAIL'}, {im['body_decoded_vs_rendered_psnr_db']} dB |
| body pixels inpainted / borrowed (mean) | {100 * im['body_inpainted_frac_mean']:.3f} % / {100 * im['body_borrowed_frac_mean']:.3f} % |
| body flicker vs source (median ratio) | {im['body_flicker_ratio_vs_source_median']:.2f} |
| can shift (median / max) | {t['shift_cm']['median']:.1f} / {t['shift_cm']['max']:.1f} cm |
| episode length (median synthetic / source) | {t['length_frames']['synthetic_median']:.0f} / {t['length_frames']['source_median']:.0f} frames |
| LeRobot load + 50-step action chunk | {'pass' if q['load_test']['ok'] else 'FAIL'} |
| can-position coverage of 40 unseen real recordings (within 2 cm; proxy) | {q.get('coverage', {}).get('within_2cm', 'n/a')} / 40 |

**Wrist cameras** (wrist gate on held-out Pringles demos 5–9 and on battery; T1 render − copy, 95 % CI):

| Camera | Gate | In this dataset | `wrist_view_valid` = 1 |
|---|---|---|---|
{wl}

## Limits
- Labels are nominal kinematic labels, not measured execution. No physics, force, or hardware replay was done.
- Body-camera fidelity: about 16 dB PSNR against real frames when the can is moved across episodes. The dominant error
  is the reconstructed object pose (about 1–2 cm).
- Coverage is an FK proxy on reused development recordings.
- Wrist mounts are photometric consistency calibrations, not ChArUco extrinsics.

Full study, protocols and reproduction commands: [GitHub `research/sfagen-wrist-render`](https://github.com/StanleyChueh/lerobot_openarm/tree/research/sfagen-wrist-render).
"""


if __name__ == "__main__":
    for n in sys.argv[1:]:
        k, _, v = n.partition("_"); v = f"_{v}" if v else ""
        (R / f"datasets_v01/fewsrc{k}{v}/README.md").write_text(card(int(k), v)); print("card", n)
