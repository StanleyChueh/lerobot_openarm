"""Publish OpenArm Pringles v04 (R04 even spread, user approval 2026-11-16) to 2 NEW public Hub repos.
usage: push_openarm_v04.py cards          -> results/r04/cards/<repo>.md (numbers read from files)
       PUBLISH=1 push_openarm_v04.py push -> new repos only; one commit each; tag v3.0 (verified); remote file check.
Same contract as sfagen/tools/push_v03.py `publish`."""
import json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

WT = Path(__file__).resolve().parents[4]; SB = WT / "research/sim_free_augmentation/simbench"; R04 = SB / "results/r04"
W = WT / "artifacts/simbench_runs/r04_openarm/all_export"; CARDS = R04 / "cards"; TYPE = dict(repo_type="dataset")
SRC = "ethanCSL/openarm_pringles_lerobot_real_10_v00"; OLD = "ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v03_loose"
REPOS = {"dataset": "ethanCSL/openarm_pringles_lerobot_real_to_sim_10_3cam_v04", "dataset_mixed": "ethanCSL/openarm_pringles_lerobot_real_and_sim_10_3cam_v04"}
COMMIT = os.popen(f"git -C {WT} rev-parse HEAD").read().strip()


def card(kind):
    info = json.loads((W / kind / "meta/info.json").read_text()); d = json.loads((R04 / "density.json").read_text())["openarm_pringles"]
    chk = json.loads((R04 / "openarm_v04_check.json").read_text()); sel = json.loads((WT / "artifacts/simbench_runs/r04_openarm/selection.json").read_text())
    fam = chk["dataset"]["families"]; rb, ra = d["eval_regions_before"], d["eval_regions_after"]
    rows = "\n".join(f"| {k} | {rb[k]:.1f} | {ra[k]:.1f} |" for k in rb)
    mixed = kind == "dataset_mixed"
    return f"""---
license: apache-2.0
task_categories: [robotics]
tags: [LeRobot, openarm, synthetic, sfagen]
---
# OpenArm Pringles hand-over: {'10 real + ' if mixed else ''}100 synthetic demos, evenly spread over the table (sfagen v04)

{'Real + synthetic mixture (`is_synthetic` marks the synthetic episodes; real episodes come first).' if mixed else '**Synthetic only (no real episodes, checked).**'}
{info['total_episodes']} episodes, {info['total_frames']:,} frames, {info['fps']} fps, LeRobot v3.0. Generated without a simulator from the
10 real demos of [`{SRC}`](https://huggingface.co/datasets/{SRC}) (rev cebdee98). Every synthetic episode has exactly its
source demo's duration (1.000x) and speed.

## What is new vs v03 rev 2 (`{OLD}`)
A real-robot evaluation of the v03 model succeeded almost always at the front/middle can positions but sometimes failed at
the rear (close to the robot). Its training set was {d['region_max_over_min_before']:.1f}x denser at the front than at the
rear. v04 fixes the data side:
1. **Exactly 100 synthetic episodes**, chosen one at a time: the 3 cm cell of the table with the fewest episodes so far
   (real demos count as already occupying their cell), ties by largest distance to everything already chosen, then the
   least-used source demo, then the lowest distortion score.
2. **Pipeline v1** candidates: v0 Cartesian blend, with a joint-space fallback where it fails near folded/singular arm
   poses (rescues rear targets); partial-visibility rule (the can must be visibly in view).

Training episodes (real + synthetic) within 4 cm of the 40 measured evaluation starts, mean per region:

| region | v03 rev 2 loose (138 + 10 real) | **v04 (100 + 10 real)** |
|---|---|---|
{rows}

Densest / thinnest region: {d['region_max_over_min_before']:.1f} -> **{d['region_max_over_min_after']:.1f}** (pre-registered goal 1.5: near miss;
front-right stays somewhat denser). {len(chk['rear_episodes'])} synthetic episodes have the can in the rear (x < 17 cm).
Density map: `assets/density.png`.

## Contents
- families: {', '.join(f'{k} {v}' for k, v in fam.items())} (v03 rev 2 also had 60 hand-over location + 11 timing
  variants; most were not selected because they keep the can at the front real-demo positions)
- synthetic episodes per source demo: {sel['per_source']}
- per-episode provenance (source demo, variation, shift, label hashes): `meta/sfa_provenance.json`
- wrist cameras: copied from the source demo (can-relative geometry exact; background off by the shift); `wrist_view_valid`
- checks: `sfagen verify` PASS; every candidate passed the kinematic/geometric gates (joint limits, speed/acceleration vs the
  real demos, IK, collision, table, can in view). No physics, no hardware replay, no policy result yet.
- known image artifacts: a faint smudge at the old can position in some frames; ghost/duplicate-can artifacts are possible
  (the distortion score does not catch all of them).

## Load
`LeRobotDataset("{REPOS[kind]}", revision="v3.0")`

## Code
sfagen, branch `research/sfagen-simbench`, commit `{COMMIT[:10]}` (local; not yet public on GitHub). Selection:
`research/sim_free_augmentation/simbench/tools/balanced_select.py` (`N=100`); protocol `simbench/results/r04/PROTOCOL.md`.
"""


mode = sys.argv[1]
if mode == "cards":
    CARDS.mkdir(parents=True, exist_ok=True)
    for k, R in REPOS.items():
        (CARDS / f"{R.split('/')[1]}.md").write_text(card(k)); print("card", R)
elif mode == "push":
    api = HfApi(); dry = os.environ.get("PUBLISH") != "1"; done = {}
    for k, R in REPOS.items():
        cp = CARDS / f"{R.split('/')[1]}.md"; assert cp.exists(), cp
        if api.repo_exists(R, **TYPE):
            raise RuntimeError(f"{R} already exists; this tool never overwrites")
        fs = {str(p.relative_to(W / k)): p for p in sorted((W / k).rglob("*")) if p.is_file()}
        fs["assets/density.png"] = R04 / "density.png"
        ops = [CommitOperationAdd(path_in_repo=a, path_or_fileobj=str(p)) for a, p in fs.items()] + [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(cp))]
        print(R, "| new public repo | upload", len(ops), "files", f"{sum(p.stat().st_size for p in fs.values()) / 1e6:.0f} MB", flush=True)
        if dry:
            continue
        api.create_repo(R, private=False, exist_ok=False, **TYPE)
        c = api.create_commit(R, operations=ops, commit_message="sfagen v04: 100 synthetic episodes evenly spread over the table (R04)", **TYPE)
        api.create_tag(R, tag="v3.0", revision=c.oid, **TYPE)
        if {x.name: x.target_commit for x in api.list_repo_refs(R, **TYPE).tags}.get("v3.0") != c.oid:
            raise RuntimeError(f"{R}: v3.0 tag does not point at {c.oid}")
        remote = set(api.list_repo_files(R, revision="v3.0", **TYPE)); miss = [a for a in list(fs) + ["README.md"] if a not in remote]
        if miss:
            raise RuntimeError(f"{R}: missing {miss[:5]}")
        done[R] = c.oid; print("  ok", c.oid[:10], flush=True)
    if not dry:
        (R04 / "hub_publish_openarm_v04.json").write_text(json.dumps({"code_commit": COMMIT, "repos": done}, indent=1))
