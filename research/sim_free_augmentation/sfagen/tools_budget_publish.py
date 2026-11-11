"""Source-budget Pringles datasets (N = 1..9 real source demos): independent re-check, dataset cards, Hub upload.
usage: tools_budget_publish.py check|cards|push   (push: DRY_RUN=1 prints the plan; repos are created PRIVATE)"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
S = Path(__file__).resolve().parent; sys.path.insert(0, str(S))
MODE = sys.argv[1]; NS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(1, 10))
REPO = "ethanCSL/openarm_pringles_lerobot_real_to_sim_{}_3cam_v00"
sel = json.loads((S / "logs/budget_selection.json").read_text()); div = {int(k): v for k, v in json.loads((S / "logs/budget/diversity.json").read_text()).items()}
W = lambda n: S / f"configs/work/pringles_src{n}"

if MODE == "check":
    from sfagen.context import Context
    from sfagen.checks import Checker
    ctx = Context(S / "configs/openarm_pringles_handover_dense.yaml"); ck = Checker(ctx); cs = ck.cfg
    real = {e: [x.astype(np.float32) for x in ctx.src.arrays(e)] for e in range(10)}
    kin = {e: ck.kin(real[e][1].astype(float)) for e in range(10)}; lowz = {e: float(ck.lowest(real[e][0].astype(float)).min()) for e in range(10)}
    out = {}
    for n in NS:
        root = W(n) / "dataset"; prov = json.loads((root / "meta/sfa_provenance.json").read_text()); sub = sel[str(n)]["sources"]
        df = pd.read_parquet(sorted((root / "data").rglob("*.parquet"))); ev = np.max([kin[e][0] for e in sub], 0); ea = np.max([kin[e][1] for e in sub], 0)
        mz = min(lowz[e] for e in sub); worst = {"speed": 0.0, "accel": 0.0, "table_margin_m": 1.0}; same_real = 0
        for i, d in df.groupby("episode_index"):
            d = d.sort_values("frame_index"); Sx = np.stack(d["observation.state"].to_numpy()).astype(np.float32); Ax = np.stack(d["action"].to_numpy()).astype(np.float32)
            v, a = ck.kin(Ax.astype(float)); worst["speed"] = max(worst["speed"], float((v / ev).max())); worst["accel"] = max(worst["accel"], float((a / ea).max()))
            worst["table_margin_m"] = min(worst["table_margin_m"], float(ck.lowest(Sx.astype(float)).min()) - (mz - cs["table_tol_m"]))
            same_real += any(len(r[0]) == len(Sx) and (np.array_equal(r[0], Sx) or np.array_equal(r[1], Ax)) for r in real.values())
        srcs = sorted({e["source_real_episode"] for e in prov["episodes"]})
        out[n] = {"episodes": int(df.episode_index.nunique()), "frames": len(df), "sources_used": srcs, "sources_subset_of_budget": set(srcs) <= set(sub),
                  "contains_real_episodes": prov["contains_real_episodes"], "episodes_identical_to_any_of_the_10_real_demos": int(same_real),
                  "max_speed_ratio_vs_budget_sources": round(worst["speed"], 3), "max_accel_ratio_vs_budget_sources": round(worst["accel"], 3),
                  "min_table_margin_m": round(worst["table_margin_m"], 4),
                  "pass": set(srcs) <= set(sub) and same_real == 0 and worst["speed"] <= cs["speed_ratio"] + 1e-6 and worst["accel"] <= cs["accel_ratio"] + 1e-6 and worst["table_margin_m"] >= -1e-6}
        print(n, out[n], flush=True)
    p = S / "logs/budget/independent_check.json"; old = json.loads(p.read_text()) if p.exists() else {}
    old.update({str(k): v for k, v in out.items()}); p.write_text(json.dumps(old, indent=1))

if MODE == "cards":
    chk = json.loads((S / "logs/budget/independent_check.json").read_text())
    rows = []
    for m in range(1, 10):
        d = div[m]; c, h = d["sources_plus_synthetic"]["can"], d["sources_plus_synthetic"]["handover"]
        rows.append(f"| [{m}](https://huggingface.co/datasets/{REPO.format(m)}) | {d['sources']} | {d['synthetic_episodes']} | {d['families']['object_position']} / {d['families']['interaction_location']} "
                    f"| {c['distinct_positions']} | {c['x_range_cm']:.0f} × {c['y_range_cm']:.0f} | {d['sources_only']['can']['unseen_real_within_2cm']} → **{c['unseen_real_within_2cm']}** "
                    f"| {d['sources_only']['handover']['unseen_real_within_2cm']} → **{h['unseen_real_within_2cm']}** |")
    rows.append("| [10 (reference)](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00) | 0–9 | 96 | 29 / 67 | – | – | 9 → 24 | 33 → 40 |")
    table = "\n".join(rows)
    for n in NS:
        d = div[n]; v = json.loads((W(n) / "verification.json").read_text())["synthetic"]; ic = chk[str(n)]
        srcs = ", ".join(map(str, d["sources"])); c, h = d["sources_plus_synthetic"]["can"], d["sources_plus_synthetic"]["handover"]
        card = f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
- openarm
- data-augmentation
- synthetic
- bimanual
- handover
- source-budget-study
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# openarm_pringles_lerobot_real_to_sim_{n}_3cam_v00

OpenArm bimanual hand-over ("{json.loads((W(n) / 'dataset/meta/sfa_provenance.json').read_text())['task']}"), three cameras (`body_cam`, `wrist_cam` = left wrist, `right_wrist_cam`), LeRobot v3.0, 30 fps.

**Synthetic only, generated from {n} real source demo{'s' if n > 1 else ''}.**
- **Count:** {v['episodes']} episodes ({v['frames']:,} frames); no real episode is included.
- **Source:** real episode{'s' if n > 1 else ''} {srcs} of [`ethanCSL/openarm_pringles_lerobot_real_10_v00`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_10_v00) (revision `cebdee98`). The other real demos are not used.
- **Variation families:** {d['families']['object_position']} object-position (can moved on the table) + {d['families']['interaction_location']} interaction-location (hand-over point moved).

Part of a **source-budget study**: the same pipeline run with 1, 2, …, 9 real source demos (demos 0…N−1, a fixed nested choice, not hand-picked). The 10-demo dataset is [`openarm_pringles_lerobot_real_to_sim_10_3cam_v00`](https://huggingface.co/datasets/ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v00).

## How diverse is it?

![positions](assets/positions_{n}.png)

Positions come from forward kinematics at the gripper-close events (right hand closing on the can; left hand closing at the hand-over). Grey points are the 40 other real recordings of the task (`openarm_pringles_lerobot_real_v00`, episodes ≥ 10), never used for generation.

| | real sources only | sources + this dataset |
|---|---:|---:|
| distinct can positions | {d['sources_only']['can']['distinct_positions']} | {c['distinct_positions']} |
| can position spread (x × y) | {d['sources_only']['can']['x_range_cm']:.1f} × {d['sources_only']['can']['y_range_cm']:.1f} cm | {c['x_range_cm']:.1f} × {c['y_range_cm']:.1f} cm |
| unseen real can positions within 2 cm | {d['sources_only']['can']['unseen_real_within_2cm']}/40 | **{c['unseen_real_within_2cm']}/40** |
| distinct hand-over positions | {d['sources_only']['handover']['distinct_positions']} | {h['distinct_positions']} |
| hand-over spread (x × y) | {d['sources_only']['handover']['x_range_cm']:.1f} × {d['sources_only']['handover']['y_range_cm']:.1f} cm | {h['x_range_cm']:.1f} × {h['y_range_cm']:.1f} cm |
| unseen real hand-over positions within 2 cm | {d['sources_only']['handover']['unseen_real_within_2cm']}/40 | **{h['unseen_real_within_2cm']}/40** |

### All budgets

| N sources | demos | synthetic episodes | object / hand-over episodes | distinct can positions | can spread (cm) | unseen can within 2 cm (sources → + synthetic) | unseen hand-over within 2 cm |
|---|---|---:|---|---:|---|---|---|
{table}

## Side-by-side video

`comparison_videos/real_vs_synthetic_all_2x.mp4`: every synthetic episode next to its real source (body camera, 2× speed).

## How it is made

Generated with `sfagen` (code and config in `sfagen/`; method in `PIPELINE.md`; prior work in `NOVELTY.md`), the same pipeline as the 10-demo dataset, with:
- **Shift grid (dense):** can moved by up to ±9 cm in x and y (3 cm steps, diagonals included); hand-over point moved by ±3 cm (and diagonals, ≤ 4.3 cm). Larger hand-over shifts were left out because the moved arms render poorly.
- **Replay limits from these {n} demo{'s' if n > 1 else ''} only:** speed ≤ 1.25× and acceleration ≤ 1.5× the maximum of the source demo{'s' if n > 1 else ''} used, lowest arm point ≥ their minimum − 1 cm. All other replay checks (joint limits, arm–arm clearance, hand distance, IK accuracy at contact, object on table and in view, robot not closer to the object than recorded) are relative to the episode's own source.
- **Rig calibration reused:** body-camera calibration and table height are the rig values fitted once on this setup (from the 10 source demos), treated as a property of the rig, not of the demos.
- **Every passing episode is included;** none were removed by hand.

## Verification

- Loads in LeRobot: {v['loads_in_lerobot']}; state/action SHA-256 equal the plan: {v['labels_sha_ok']}.
- Episodes identical to any of the 10 real demos: {ic['episodes_identical_to_any_of_the_10_real_demos']}; sources used: {ic['sources_used']}.
- Re-checked from the exported parquet: max speed ratio {ic['max_speed_ratio_vs_budget_sources']} (≤ 1.25), max acceleration ratio {ic['max_accel_ratio_vs_budget_sources']} (≤ 1.5) against the {n} source demo{'s' if n > 1 else ''}.
- All frames of all three cameras decode, no black frames; body camera vs rendered frames: {v['observation.images.body_cam']['psnr_vs_rendered_db']} dB after AV1 encoding.

Per-episode provenance (source episode, variation, shift, schedule, replay-check values): `meta/sfa_provenance.json`.

## Limitations

- Only positions vary: the grasp style, timing and motion shape come from the {n} source demo{'s' if n > 1 else ''}.
- Wrist cameras are the real recorded wrist frames (`wrist_view_valid` = 0 where the synthetic wrist pose differs; `wrist_view_pose_offset` gives the offset).
- No physics check and no hardware replay; no policy has been trained on this dataset yet.
"""
        (W(n) / "README_budget.md").write_text(card); print("card", n)

if MODE == "push":
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(); T = dict(repo_type="dataset"); DRY = os.environ.get("DRY_RUN") == "1"
    code = sorted(S.glob("sfagen/*.py")) + [S / "README.md", S / "tools_budget_select.py", S / "tools_budget_build.py", S / "tools_budget_figs.py", S / "tools_budget_publish.py",
                                           S / "tools_source_budget.py", S / "configs/_openarm_robot.yaml", S / "configs/openarm_pringles_handover_dense.yaml"]
    for n in NS:
        R = REPO.format(n); root = W(n) / "dataset"
        ops = [CommitOperationAdd(path_in_repo=str(p.relative_to(root)), path_or_fileobj=str(p)) for p in sorted(root.rglob("*")) if p.is_file()]
        ops += [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(W(n) / "README_budget.md")),
                CommitOperationAdd(path_in_repo="PIPELINE.md", path_or_fileobj=str(S / "PIPELINE.md")),
                CommitOperationAdd(path_in_repo="NOVELTY.md", path_or_fileobj=str(S / "NOVELTY.md")),
                CommitOperationAdd(path_in_repo="verification.json", path_or_fileobj=str(W(n) / "verification.json")),
                CommitOperationAdd(path_in_repo=f"sfagen/configs/pringles_src{n}.yaml", path_or_fileobj=str(S / f"configs/pringles_src{n}.yaml")),
                CommitOperationAdd(path_in_repo="assets/budget_selection.json", path_or_fileobj=str(S / "logs/budget_selection.json")),
                CommitOperationAdd(path_in_repo="assets/diversity.json", path_or_fileobj=str(S / "logs/budget/diversity.json")),
                CommitOperationAdd(path_in_repo="assets/independent_check.json", path_or_fileobj=str(S / "logs/budget/independent_check.json")),
                CommitOperationAdd(path_in_repo=f"assets/positions_{n}.png", path_or_fileobj=str(S / f"logs/budget/positions_{n}.png")),
                CommitOperationAdd(path_in_repo="assets/positions_overview.png", path_or_fileobj=str(S / "logs/budget/positions_overview.png")),
                CommitOperationAdd(path_in_repo="comparison_videos/real_vs_synthetic_all_2x.mp4", path_or_fileobj=str(W(n) / "compare_all_2x.mp4"))]
        ops += [CommitOperationAdd(path_in_repo=f"sfagen/{p.relative_to(S)}", path_or_fileobj=str(p)) for p in code]
        exists = api.repo_exists(R, **T)
        print(R, "| exists" if exists else "| new (private)", "|", len(ops), "files", flush=True)
        if DRY:
            continue
        assert not exists, f"{R} already exists; refusing to overwrite without tagging"
        api.create_repo(R, private=True, **T)
        c = api.create_commit(R, operations=ops, commit_message=f"sfagen source-budget dataset: synthetic only, from {n} real source demo(s); replay-checked; verified", **T)
        api.create_tag(R, tag="v3.0", revision=c.oid, **T); print("   main", c.oid, flush=True)
