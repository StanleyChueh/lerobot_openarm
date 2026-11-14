"""v03 driver steps (CPU). usage: python v03_steps.py <step>   steps: thresholds | select | finalize
thresholds : score the v02 calibration re-render; strict = max score at |shift| <= 4 cm, loose = 2 x strict
select     : per grid target, the next passing candidates to render (2 smallest-shift; up to 4 if none <= loose)
finalize   : choose kept episodes per threshold, write selection.json
"""
import json, os, sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v02_worktree/research/sim_free_augmentation/sfagen')
from sfagen.distortion import score_file

R2 = os.environ.get('REV', '1') == '2'  # amendment 4: rev-2 run dir/results; thresholds are read from rev 1 (unchanged)
R = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/' + ('sfagen_v03r2_run' if R2 else 'sfagen_v03_run')); OUT = Path(sys.path[0]) / ('results/v03r2' if R2 else 'results/v03')
OUT.mkdir(exist_ok=True); GEN = R / ('gen' if R2 else 'gen_a3')
TH = Path(sys.path[0]) / 'results/v03/thresholds.json'
V02 = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v02_run/copied')
shift = lambda v: float(np.hypot(*v['shift_m'][:2]))  # noqa: E731


def checks(work):
    out = {}
    for f in sorted(glob.glob(str(work / 'candidates/ep*/checks.json'))):
        e = int(f.split('/ep')[-1].split('/')[0])
        for k, v in json.loads(Path(f).read_text()).items():
            out[(e, k)] = v
    return out


def thresholds():
    E = json.loads((V02 / 'dataset/meta/sfa_provenance.json').read_text())['episodes']
    rows = [dict(source=e['source_real_episode'], key=e['variation'], family=e['family'], shift_cm=100 * float(np.hypot(*e['shift_m'][:2])),
                 score=score_file(R / f"calib/body/{e['source_real_episode']}_{e['variation']}.json")) for e in E]
    strict = max(r['score'] for r in rows if r['shift_cm'] <= 4 + 1e-9)
    t = dict(strict=strict, loose=2 * strict, rule='strict = max v02 score at |shift| <= 4 cm; loose = 2 x strict',
             n_calibration=len(rows), n_le_4cm=sum(r['shift_cm'] <= 4 + 1e-9 for r in rows), rows=rows)
    (OUT / 'thresholds.json').write_text(json.dumps(t, indent=1)); print({k: v for k, v in t.items() if k != 'rows'})


def select(rendered_scores=None):
    C = checks(GEN); T = json.loads(TH.read_text())
    by = {}
    for (e, k), v in C.items():
        if v['pass']:
            by.setdefault(k, []).append((shift(v), e))
    want = []
    for k, lst in sorted(by.items()):
        lst.sort(); done = [(s, e) for s, e in lst if (R / f'all/body/{e}_{k}.json').exists()]
        good = [1 for s, e in done if score_file(R / f'all/body/{e}_{k}.json') <= T['loose']]
        n = len(lst) if R2 else 2 if len(done) < 2 else (min(4, len(done) + 1) if len(good) < 1 and len(done) < len(lst) else len(done))
        want += [(e, k) for s, e in lst[:n] if (s, e) not in done]
    (R / 'render_todo.json').write_text(json.dumps(want)); print('targets with pass', len(by), 'to render', len(want))


def finalize():
    T = json.loads(TH.read_text()); C = checks(R / 'all')
    bad = {tuple(x) for x in json.loads((OUT / 'timing_excluded.json').read_text())} if R2 else set()
    rows = []
    for f in sorted(glob.glob(str(R / 'all/body/*.json'))):
        stem = Path(f).stem; e, k = int(stem.split('_', 1)[0]), stem.split('_', 1)[1]; v = C[(e, k)]
        rows.append(dict(source=e, key=k, family=v['family'], shift_cm=100 * shift(v), score=score_file(f),
                         target_xy_m=v['marks'].get('target_xy_m'), timing_excluded=(e, k) in bad))
    sel = {}
    for name in ('strict', 'loose'):
        th = T[name]; ok = [r for r in rows if not r['timing_excluded']]; keep = [r for r in ok if r['family'] != 'object_workspace' and r['score'] <= th]
        grid = {}
        order = (lambda r: (r['score'], r['shift_cm'])) if R2 else (lambda r: r['shift_cm'])  # rev 2: least distortion first
        for r in sorted((r for r in ok if r['family'] == 'object_workspace' and r['score'] <= th), key=order):
            grid.setdefault(r['key'], [])
            if len(grid[r['key']]) < 2: grid[r['key']].append(r)
        keep += [r for v in grid.values() for r in v]
        sel[name] = dict(threshold=th, episodes=[(r['source'], r['key']) for r in keep], n=len(keep), grid_targets=len(grid))
    (OUT / 'selection.json').write_text(json.dumps(dict(selection=sel, rendered=rows), indent=1))
    print({n: (s['n'], s['grid_targets']) for n, s in sel.items()}, 'rendered', len(rows))


{'thresholds': thresholds, 'select': select, 'finalize': finalize}[sys.argv[1]]()
