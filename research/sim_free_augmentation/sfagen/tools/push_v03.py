"""v03 datasets: write dataset cards (review first), then publish to NEW Hub repos only with explicit PUBLISH=1.
usage: python tools/push_v03.py cards          -> results/v03/cards/<repo>.md (8 cards, every number read from files)
       PUBLISH=1 python tools/push_v03.py push -> 8 new public repos (refuses existing repos; tags v3.0; verifies tag)
Without PUBLISH=1, `push` is a dry run. Quality gates of all exports are checked before the first write."""
import json, os, sys, tempfile
from pathlib import Path
from huggingface_hub import CommitOperationAdd
from _provenance import HERE as S, upload_provenance, git_state
from push_unified import export_files
TYPE = dict(repo_type='dataset')


def publish(api, R, new, extra, dry):
    """Same contract as push_fewsrc.publish (new repo only, tag v3.0, verify), with a v03 commit message."""
    if api.repo_exists(R, **TYPE):
        raise RuntimeError(f'{R} already exists; this tool never overwrites')
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(p)) for k, p in new.items()] + extra
    print(R, '| new public repo | upload', len(ops), 'files', flush=True)
    if dry:
        return None
    api.create_repo(R, private=False, exist_ok=False, **TYPE)
    c = api.create_commit(R, operations=ops, commit_message='sfagen v03 even-coverage dataset (F1W grid, distortion-capped); quality-gated', **TYPE)
    api.create_tag(R, tag='v3.0', revision=c.oid, **TYPE)
    if {t.name: t.target_commit for t in api.list_repo_refs(R, **TYPE).tags}.get('v3.0') != c.oid:
        raise RuntimeError(f'{R}: v3.0 tag does not point at {c.oid}')
    return c.oid


def verify_pass(v):
    """Same rule as sfagen/verify.py (which raises instead of writing a flag) plus the mixed-set checks."""
    s, m = v['synthetic'], v['mixed']; cams = [k for k, x in s.items() if isinstance(x, dict) and 'decoded' in x]
    return bool(s['loads_in_lerobot'] and s['labels_sha_ok'] and s['all_replay_checks_pass'] and s['video_spans']['pass']
                and s['episodes_identical_to_a_real_episode'] == 0 and all(s[c]['decoded'] == s['frames'] and not s[c]['any_all_black_frame'] for c in cams)
                and m['is_synthetic_flag_correct'] and m['real_labels_identical_to_source'] and m['real_episodes'] == 10)


RUN = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03_run'); V = S / 'results/v03'
SRC = 'ethanCSL/openarm_pringles_lerobot_real_10_v00'


GH = 'https://github.com/StanleyChueh/lerobot_openarm'; BRANCH = 'research/sfagen-v02'; SUB = 'research/sim_free_augmentation/sfagen'


def jobs():
    commit = git_state()['commit']; T = json.loads((V / 'thresholds.json').read_text()); cov = json.loads((V / 'coverage.json').read_text())
    sel = json.loads((V / 'selection.json').read_text())['selection']; aud = json.loads((V / 'visual_audit.json').read_text())
    out = []
    for th in ('strict', 'loose'):
        for cond, suf in (('copied', ''), ('right_wrist', '_wristrender')):
            q = json.loads((V / f'quality_{th}_{cond}/quality_report.json').read_text())
            ver = json.loads((RUN / f'{th}_{cond}/verification.json').read_text())
            for mixed in (False, True):
                kind = 'real_and_sim' if mixed else 'real_to_sim'; root = RUN / f'{th}_{cond}' / ('dataset_mixed' if mixed else 'dataset')
                info = json.loads((root / 'meta/info.json').read_text())
                repo = f'ethanCSL/openarm_pringles_lerobot_{kind}_10_3cam{suf}_v03_{th}'
                ts = cov['table'][f'+v03 {th}']; flagged = sum((f['source'], f['key']) in {tuple(x) for x in sel[th]['episodes']} for f in aud['flags'])
                card = f"""---
license: apache-2.0
task_categories: [robotics]
tags: [LeRobot, openarm, synthetic, sfagen]
---
# OpenArm Pringles hand-over: {'10 real + ' if mixed else ''}synthetic even-coverage demos (sfagen v03, {th} distortion cap{', rendered right wrist' if cond == 'right_wrist' else ''})

{'Real + synthetic mixture' if mixed else '**Synthetic only (no real episodes).**'} {info['total_episodes']} episodes, {info['total_frames']:,} frames, {info['fps']} fps, LeRobot v3.0.
Generated without a simulator from the 10 real demos of [`{SRC}`](https://huggingface.co/datasets/{SRC}) (rev cebdee98).

- **Variation:** can start positions on a 3 cm arena grid ({sel[th]['grid_targets']} targets, F1W planner) + v02 hand-over location/timing variants.
- **Distortion cap ({th}):** body-camera score (missing + spilled silhouette pixels, 95th percentile over frames) ≤ {sel[th]['threshold']:.2f}
  (strict = worst v02 episode {T['strict']:.2f}; loose = 2 × strict). The score misses some ghost/duplicate-can artifacts:
  {flagged} kept episodes carry a visual-audit flag (`results/v03/visual_audit.json`).
- **Wrist cameras:** {'right wrist rendered by sfagen (ablation; the Pringles wrist-fidelity gate failed); left wrist copied' if cond == 'right_wrist' else 'copied from the source demo (can-relative geometry exact; table/background off by the shift)'}; `wrist_view_valid` flags invalid frames.
- **Coverage proxy** (40 measured eval starts): worst nearest can start {ts['worst_all40_cm']} cm, {ts['within_2cm']}/40 within 2 cm (real only: 6.63 cm, 14/40).
- **Checks:** sfagen verify {'PASS' if verify_pass(ver) else 'FAIL'}; quality gates {'PASS' if q['pass'] else 'FAIL'}. Kinematic/geometric only:
  no physics, no hardware replay, no policy result yet.

## How it was made
1. **Source:** 10 real teleoperated OpenArm demos (right arm grasps the Pringles can and hands it to the left arm), 3 cameras
   (body, left wrist, right wrist). Positions are in sfagen's robot world frame (small x = rear of the arena).
2. **Targets:** a 3 cm grid of new can start positions (x 9–36 cm, y −9 to 15 cm). For each cell, the 10 nearest real demos
   within 12 cm are tried.
3. **Trajectory (F1W planner, `sfagen/workspace_plan.py`):** the near-object reach/grasp is moved rigidly with the can;
   transit uses a vertical detour; IK uses action-aware joint bounds; the clock is re-timed. The hand-over and the left
   arm are unchanged. Gates (`sfagen/checks.py`):
   - joint limits, speed and acceleration within the real demos' envelope;
   - can in the body view and on the table (Lab appearance test);
   - arms not too close.
4. **Body camera (`sfagen/render.py`):** forward-warps the source frames using depth + robot meshes (re-posed arm, moved can).
   Missing pixels are borrowed from other frames or inpainted.
5. **Distortion cap (`sfagen/distortion.py`):** per frame, (missing + spilled silhouette pixels) / silhouette; the episode
   score is the 95th percentile over frames. Episodes above the {th} threshold are dropped. Thresholds were frozen before
   rendering (`results/v03/PROTOCOL.md`, `FREEZE.json`); the amendments are disclosed there.
6. **Wrist cameras:** {'left copied; right re-rendered by sfagen (`sfagen/wrist.py`) for the new arm pose and can position' if cond == 'right_wrist' else 'both copied from the source demo, frame-aligned to the re-timed clock'}.
7. **Also included:** v02 hand-over variants (75 episodes): 60 hand-over locations moved ±3/±5 cm in x or y, and 15 hand-over timings shifted ±10 frames.
8. **Export:** LeRobot v3.0 (`sfagen build`). Each episode's source demo, variation, shift and label hashes are in
   `meta/sfa_provenance.json`. {'Real episodes come first; the `is_synthetic` column marks the synthetic ones.' if mixed else 'No real episode is included (checked).'}

## Code (to locate and reproduce)
- **Repository:** [{GH}]({GH}), branch [`{BRANCH}`]({GH}/tree/{BRANCH}/{SUB}), commit [`{commit[:10]}`]({GH}/tree/{commit}/{SUB}) (full SHA `{commit}`).
- **Pipeline:** `{SUB}/sfagen/`. Configs: `configs/openarm_pringles_v03*.yaml`. Run scripts: `results/v03/run_scripts/`.
- **Results and figures:** `results/v03/RESULTS.md`, `coverage.png`, `visual_audit.json`.
- **This repo also contains:** a snapshot of the exact code under `sfagen/`, and `provenance.json` (commit, per-file
  sha256, uv.lock hash, SAM2 revision).
"""
                out.append(dict(repo=repo, root=root, card=card, quality=V / f'quality_{th}_{cond}/quality_report.json', ok=bool(q['pass'] and verify_pass(ver))))
    return out


if __name__ == '__main__':
    J = jobs(); (V / 'cards').mkdir(exist_ok=True)
    for j in J: (V / 'cards' / (j['repo'].split('/')[1] + '.md')).write_text(j['card'])
    if sys.argv[1] == 'cards':
        print('\n'.join(f"{j['repo']}  gates={'PASS' if j['ok'] else 'FAIL'}" for j in J)); sys.exit()
    assert sys.argv[1] == 'push'
    dirty = [l for l in git_state()['dirty_files'] if l[3:].startswith(('sfagen/', 'configs/', 'tools/push_v03.py'))]
    if dirty: raise RuntimeError(f'uncommitted code would not match the cited commit: {dirty}')
    if not all(j['ok'] for j in J): raise RuntimeError('a quality/verify gate failed; nothing published')
    from huggingface_hub import HfApi
    api = HfApi(); code = sorted(S.glob('sfagen/*.py')) + sorted(S.glob('configs/openarm_pringles_v03*.yaml'))
    prov = Path(tempfile.mkdtemp()) / 'provenance.json'; prov.write_text(json.dumps(upload_provenance(code), indent=1))
    dry = os.environ.get('PUBLISH') != '1'; res = {}
    for j in J:
        extra = [CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=str(V / 'cards' / (j['repo'].split('/')[1] + '.md'))),
                 CommitOperationAdd(path_in_repo='quality_report.json', path_or_fileobj=str(j['quality'])),
                 CommitOperationAdd(path_in_repo='provenance.json', path_or_fileobj=str(prov)),
                 CommitOperationAdd(path_in_repo='assets/coverage.png', path_or_fileobj=str(V / 'coverage.png'))]
        extra += [CommitOperationAdd(path_in_repo=f'sfagen/{p.relative_to(S)}', path_or_fileobj=str(p)) for p in code]
        res[j['repo']] = publish(api, j['repo'], export_files(j['root']), extra, dry)
    print(json.dumps(res, indent=1), '(dry run)' if dry else '')
