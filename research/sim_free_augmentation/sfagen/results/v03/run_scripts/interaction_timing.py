"""Amendment 4, item 5: re-check the reused v02 hand-over candidates (interaction_location / interaction_timing) with the
same duration + TCP speed-similarity gate as the grid (Checker.timing_similarity). Writes results/v03r2/timing_excluded.json
(the passing interaction candidates that fail) and interaction_timing.json (all ratios). CPU.
usage: REV=2 python interaction_timing.py"""
import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
S = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v02_worktree/research/sim_free_augmentation/sfagen'); sys.path.insert(0, str(S))
from sfagen.context import Context
from sfagen.checks import Checker

ctx = Context(S / 'configs/openarm_pringles_v03r2.yaml'); ck = Checker(ctx); r = ctx.robot; OUT = S / 'results/v03r2'; OUT.mkdir(exist_ok=True)
rows, bad = [], []
for e in ctx.episodes():
    S0 = ctx.src.arrays(e)[0].astype(np.float64); rec = SimpleNamespace(state=S0, T=len(S0))
    C = json.loads((ctx.work / f'candidates/ep{e}/checks.json').read_text())
    for k, v in sorted(C.items()):
        if not v['pass'] or v['family'] not in ('interaction_location', 'interaction_timing'): continue
        z = np.load(ctx.work / f'candidates/ep{e}/{k}.npz'); St = z['state'].astype(np.float64)
        moved = [a for a in r.arms if len(St) != len(S0) or np.abs(St[:, r.sidx[a]] - S0[:, r.sidx[a]]).max() > 1e-9]
        t = ck.timing_similarity(rec, dict(state=St, edited=z['edited'], moved_arms=moved))
        rows.append(dict(source=e, key=k, moved_arms=moved, **t))
        if not t['timing_ok']: bad.append([e, k])
(OUT / 'interaction_timing.json').write_text(json.dumps(rows, indent=1)); (OUT / 'timing_excluded.json').write_text(json.dumps(bad))
pk = np.array([x for t in rows for x in t.get('tcp_peak_speed_ratio', [])]); mn = np.array([x for t in rows for x in t.get('tcp_mean_speed_ratio_edited', [])])
print(len(rows), 'interaction candidates;', len(bad), 'fail;', 'duration mismatches', sum(t['frames_out'] != t['source_frames'] for t in rows),
      'peak %.2f-%.2f mean %.2f-%.2f' % (pk.min(), pk.max(), mn.min(), mn.max()))
