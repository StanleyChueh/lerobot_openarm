"""Build v03 work dirs. all/ = v02 interaction candidates (location + timing; renders hard-linked from calib/, already
instrumented) + amended v03 grid candidates (gen_a1/). all_wrist/ shares cache/candidates/body with all/ via symlinks.
v02 object_position candidates are excluded (replaced by the grid). Nothing under sfagen_v02_run is modified."""
import hashlib, json, os, shutil
R2 = os.environ.get('REV', '1') == '2'  # amendment 4: rev-2 grid from sfagen_v03r2_run/gen; interaction renders still from rev-1 calib/
from pathlib import Path

RUN1 = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03_run'); CAL = RUN1 / 'calib'
RUN = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03r2_run') if R2 else RUN1; GEN = RUN / ('gen' if R2 else 'gen_a3')
A = RUN / 'all'; W = RUN / 'all_wrist'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
for d in (A, W):
    if d.exists(): raise FileExistsError(d)
A.mkdir(); W.mkdir()
for name in ('calibration.json', 'table_height.json', 'lowest_arm.json'):
    shutil.copy2(CAL / name, A / name); shutil.copy2(CAL / name, W / name)
shutil.copytree(CAL / 'cache', A / 'cache'); (A / 'candidates').mkdir(); (A / 'body').mkdir()
added = {}
for ep in sorted((GEN / 'candidates').iterdir()):
    dst = A / 'candidates' / ep.name; dst.mkdir()
    old = json.loads((CAL / 'candidates' / ep.name / 'checks.json').read_text())
    keep = {k: v for k, v in old.items() if v.get('family') in ('interaction_location', 'interaction_timing')}
    new = json.loads((ep / 'checks.json').read_text()); gm = json.loads((ep / 'generation_manifest.json').read_text())
    assert gm['complete'] and not set(keep) & set(new)
    om = json.loads((CAL / 'candidates' / ep.name / 'generation_manifest.json').read_text())
    for k in keep:
        shutil.copy2(CAL / 'candidates' / ep.name / f'{k}.npz', dst / f'{k}.npz'); assert sha(dst / f'{k}.npz') == om['candidate_sha256'][k]
        if keep[k]['pass'] and (CAL / f'body/{ep.name[2:]}_{k}.npy').exists():
            for ext in ('npy', 'json', 'fingerprint'):
                os.link(CAL / f'body/{ep.name[2:]}_{k}.{ext}', A / f'body/{ep.name[2:]}_{k}.{ext}')
    for k in new:
        shutil.copy2(ep / f'{k}.npz', dst / f'{k}.npz'); assert sha(dst / f'{k}.npz') == gm['candidate_sha256'][k]
    (dst / 'checks.json').write_text(json.dumps({**keep, **new}, indent=1, default=float))
    m = {**gm, 'candidates': len(keep) + len(new), 'passed': sum(v['pass'] for v in {**keep, **new}.values()),
         'candidate_sha256': {**{k: om['candidate_sha256'][k] for k in keep}, **gm['candidate_sha256']},
         'v03_merge': dict(v02_interaction_from=str(CAL / 'candidates' / ep.name), v02_code_sha256=om['code_sha256'])}
    (dst / 'generation_manifest.json').write_text(json.dumps(m, indent=1))
    added[ep.name] = dict(interaction=sorted(k for k, v in keep.items() if v['pass']), grid_pass=sorted(k for k, v in new.items() if v['pass']))
for name in ('cache', 'candidates', 'body'): (W / name).symlink_to(A / name)
(W / 'wrist').mkdir()
(RUN / 'MERGE.json').write_text(json.dumps(added, indent=1))
print({e: (len(v['interaction']), len(v['grid_pass'])) for e, v in added.items()})
