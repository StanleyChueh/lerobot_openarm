"""Publish the R03 RoboTwin datasets (user approval 2026-11-16: all 3 tasks, 9 repos) to NEW public Hub repos.
usage: push_robotwin.py cards          -> results/r03/cards/<repo>.md (every number read from files; review first)
       PUBLISH=1 push_robotwin.py push -> 9 new public dataset repos; refuses existing repos; one commit each;
                                          tags v3.0 and verifies the tag; then re-checks the remote file list.
Without PUBLISH=1, `push` is a dry run. Same contract as sfagen/tools/push_v03.py `publish`."""
import json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

WT = Path(__file__).resolve().parents[4]; SB = WT / "research/sim_free_augmentation/simbench"
VER = os.environ.get("VER", "r03")   # r03 = first publication; r04 = even N=100 selection (overwrite of the 6 synthetic repos)
DATA = WT / f"artifacts/simbench_data/robotwin_sfagen_{'v1r3' if VER == 'r03' else 'v1r4'}"; R03 = SB / "results/r03"; CARDS = SB / f"results/{VER}/cards"
R04 = SB / "results/r04"
OWNER = "ethanCSL"; TYPE = dict(repo_type="dataset")
TASKS = ["place_empty_cup", "stack_blocks_two", "handover_block"]
KINDS = {"source": "source_10", "synthetic": "sim_aug_10", "mixed": "source_and_sim_aug_10"}
QUALITY = {"stack_blocks_two": "good: the block is moved and the arm stays coherent; small arm fragments at the image edge",
           "place_empty_cup": "mediocre: the cup is moved correctly, but the arm shows black fragments while carrying the cup and cups near the image edge show ghosting",
           "handover_block": "poor for the block-position episodes (the 20 cm block re-renders distorted, arm fragments); the handover location/timing episodes look better"}
COMMIT = os.popen(f"git -C {WT} rev-parse HEAD").read().strip()


def repo(t, kind):
    return f"{OWNER}/robotwin_{t}_lerobot_{KINDS[kind]}"


def card(t, kind):
    info = json.loads((DATA / t / kind / "meta/info.json").read_text()); sub = json.loads((DATA / t / "source/subset.json").read_text())
    RR = R03 if VER == "r03" else R04
    dist = json.loads((RR / "distribution.json").read_text())[t]; chk = json.loads((RR / "datasets_check.json").read_text())[t]
    fam = chk.get("synthetic", {}).get("families", {})
    import pandas as pd; task_str = str(pd.read_parquet(DATA / t / kind / "meta/tasks.parquet").index[0])
    head = f"""---
license: other
license_name: robotwin-upstream
license_link: https://github.com/RoboTwin-Platform/RoboTwin/blob/main/LICENSE
task_categories: [robotics]
tags: [LeRobot, robotwin, aloha-agilex, sim-free-augmentation, sfagen]
configs:
- config_name: default
  data_files: data/*/*.parquet
---
# RoboTwin 2.0 `{t}` ({'10 source demos' if kind == 'source' else 'sfagen synthetic' + (' + 10 source demos' if kind == 'mixed' else ' only')})

LeRobot v3.0 dataset, {info['total_episodes']} episodes / {info['total_frames']} frames at {info['fps']} fps, robot `aloha-agilex`
(14-D joint state/action, head camera `cam_high` + 2 wrist cameras, 640x480).

**Source demos.** The first 10 `demo_clean` episodes of `{t}` in
[`lerobot/robotwin_unified`](https://huggingface.co/datasets/{sub['source_repo']}) at revision `{sub['source_revision'][:12]}`
(global episode indices {sub['source_episodes'][0]}-{sub['source_episodes'][-1]}), split with LeRobot's `split_dataset`
(videos re-encoded once, AV1 CRF 30).
"""
    if kind == "source":
        body = "\nThis repo contains only those 10 source demos (no synthetic data). They are the inputs of the companion `sim_aug` datasets.\n"
    else:
        body = f"""
**Synthetic episodes.** Made by *sfagen* (simulation-free augmentation; no physics simulator, no simulator state used)
from the 10 source demos and the robot URDF only: the object (or the hand-over) is moved, the arm trajectory is
retargeted by IK (pipeline v1: Cartesian blend, joint-space fallback near singular poses), and the head-camera
image is re-rendered from the source frames. Every synthetic episode has exactly its source demo's length (1.000x).
- families: {', '.join(f'{k} {v}' for k, v in fam.items())}
- shift from its source demo (object position, or hand-over location for interaction episodes): median {dist['shift_cm']['median']:.1f} cm, max {dist['shift_cm']['max']:.1f} cm
- synthetic episodes per source demo: {dist['per_source']}
- target positions: 3 cm lattice within 12 cm of the source object positions, kept when the object is visibly in the
  head camera (>= 50 % of its points, or as visible as in its source demo)
{{EVEN}}
- all candidates passed sfagen's kinematic/visual gates (speed/acceleration vs the source demos, IK, joint limits,
  collision, table, object placement); `sfagen verify` passed.
{'- contains the 10 source demos as well; per-frame flag `is_synthetic` (0 = source, 1 = synthetic).' if kind == 'mixed' else '- contains NO source episodes.'}
- per-episode provenance (source demo, family, shift): `meta/sfa_provenance_merged.json`.

**Image-quality warning ({t}).** {QUALITY[t]}. These are research data from a pipeline under development (the camera
calibration is still 16-35 px off); inspect the videos before training on them.
"""
    tail = f"""
**Load** (`lerobot` >= 0.4): `LeRobotDataset("{repo(t, kind)}", revision="v3.0")`. Task string: "{json.dumps(task_str)[1:-1]}" (see `meta/tasks.parquet`).

**Provenance.** sfagen code commit `{COMMIT[:10]}` (branch `research/sfagen-simbench`; local, not yet public). Upstream data:
RoboTwin 2.0 (MIT-licensed code, https://github.com/RoboTwin-Platform/RoboTwin); the upstream LeRobot dataset declares no
license, so the RoboTwin terms apply. Please cite RoboTwin 2.0 when using these data.
"""
    even = ""
    if VER != "r03" and kind != "source":
        d = json.loads((R04 / "density.json").read_text())[t]
        even = (f"- **even spread (R04):** exactly 100 synthetic episodes, chosen round by round over 3 cm cells (the emptiest cell\n"
                f"  first, ties by distance to everything already chosen; real demos count as occupied) so no area of the table is\n"
                f"  over-represented. Training-density evenness (episodes within 4 cm over the target lattice): CV {d['before']['cv']:.2f}\n"
                f"  -> {d['after']['cv']:.2f} vs the first release (r03, {d['synthetic_before']} synthetic episodes; still reachable at tag `r03`).")
    return (head + body + tail).replace("{EVEN}", even)


def files(t, kind):
    root = DATA / t / kind; out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "_mixed_tmp" not in p.parts:
            out[str(p.relative_to(root))] = p
    if kind != "source":
        out["assets/distribution.png"] = (R03 if VER == "r03" else R04) / "distribution.png"
        if VER != "r03":
            out["assets/density.png"] = R04 / "density.png"
    return out


mode = sys.argv[1]
if mode == "cards":
    CARDS.mkdir(parents=True, exist_ok=True)
    for t in TASKS:
        for k in KINDS:
            (CARDS / f"{repo(t, k).split('/')[1]}.md").write_text(card(t, k)); print("card", repo(t, k))
elif mode == "overwrite":
    # VER=r04: tag the current v3.0 commit as r03 (verified), replace every file in one commit, move v3.0 (verified)
    api = HfApi(); dry = os.environ.get("PUBLISH") != "1"; done = {}
    from huggingface_hub import CommitOperationDelete
    for t in TASKS:
        for k in ("synthetic", "mixed"):
            R = repo(t, k); fs = files(t, k); cp = CARDS / f"{R.split('/')[1]}.md"; assert cp.exists(), cp
            tagc = lambda: {x.name: api.repo_info(R, revision=x.name, **TYPE).sha for x in api.list_repo_refs(R, **TYPE).tags}  # noqa: E731
            tags = tagc(); assert "v3.0" in tags, f"{R}: no v3.0"
            old = tags.get("r03") or tags["v3.0"]
            ops = [CommitOperationAdd(path_in_repo=a, path_or_fileobj=str(p)) for a, p in fs.items()] + [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(cp))]
            keep = {o.path_in_repo for o in ops}; stale = [f for f in api.list_repo_files(R, revision=old, **TYPE) if f not in keep and f != ".gitattributes"]
            print(R, "| overwrite | r03", old[:10], "| upload", len(ops), "delete", len(stale), flush=True)
            if dry:
                continue
            if "r03" not in tags:
                api.create_tag(R, tag="r03", revision=old, tag_message="first release (R03), kept for reproducibility", **TYPE)
            if tagc().get("r03") != old:
                raise RuntimeError(f"{R}: r03 tag missing; nothing overwritten")
            c = api.create_commit(R, operations=[CommitOperationDelete(path_in_repo=f) for f in stale] + ops,
                                  commit_message="sfagen R04: exactly 100 synthetic episodes, evenly spread over the table", **TYPE)
            api.delete_tag(R, tag="v3.0", **TYPE); api.create_tag(R, tag="v3.0", revision=c.oid, **TYPE)
            T = tagc()
            if T.get("v3.0") != c.oid or T.get("r03") != old:
                raise RuntimeError(f"{R}: tags not as expected {T}")
            remote = set(api.list_repo_files(R, revision="v3.0", **TYPE)); miss = [a for a in keep if a not in remote]
            if miss:
                raise RuntimeError(f"{R}: missing {miss[:5]}")
            done[R] = {"v3.0": c.oid, "r03": old}; print("  ok", c.oid[:10], flush=True)
    if not dry:
        (R04 / "hub_overwrite.json").write_text(json.dumps({"code_commit": COMMIT, "repos": done}, indent=1))
elif mode == "push":
    api = HfApi(); dry = os.environ.get("PUBLISH") != "1"; done = {}
    for t in TASKS:
        for k in KINDS:
            R = repo(t, k); fs = files(t, k); cp = CARDS / f"{R.split('/')[1]}.md"
            assert cp.exists(), f"write and review cards first: {cp}"
            if api.repo_exists(R, **TYPE):
                raise RuntimeError(f"{R} already exists; this tool never overwrites")
            ops = [CommitOperationAdd(path_in_repo=a, path_or_fileobj=str(p)) for a, p in fs.items()] + [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(cp))]
            print(R, "| new public repo | upload", len(ops), "files", f"{sum(p.stat().st_size for p in fs.values()) / 1e6:.0f} MB", flush=True)
            if dry:
                continue
            api.create_repo(R, private=False, exist_ok=False, **TYPE)
            c = api.create_commit(R, operations=ops, commit_message=f"sfagen RoboTwin {t}: {k} dataset (R03)", **TYPE)
            api.create_tag(R, tag="v3.0", revision=c.oid, **TYPE)
            if {x.name: x.target_commit for x in api.list_repo_refs(R, **TYPE).tags}.get("v3.0") != c.oid:
                raise RuntimeError(f"{R}: v3.0 tag does not point at {c.oid}")
            remote = set(api.list_repo_files(R, revision="v3.0", **TYPE))
            missing = [a for a in list(fs) + ["README.md"] if a not in remote]
            if missing:
                raise RuntimeError(f"{R}: missing on the Hub: {missing[:5]}")
            done[R] = c.oid; print("  ok", c.oid[:10], len(remote), "files", flush=True)
    if not dry:
        (R03 / "hub_publish.json").write_text(json.dumps({"code_commit": COMMIT, "repos": done}, indent=1))
