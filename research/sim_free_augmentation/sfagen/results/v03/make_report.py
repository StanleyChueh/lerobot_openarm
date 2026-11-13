"""v03 coverage table + arena map (CPU). usage: python results/v03/make_report.py  (run from sfagen/)"""
import json, collections, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
O = Path('results/v03'); RUN = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03_run')
pl = json.loads(Path('results/v02/planned_targets.json').read_text()); P = np.array(pl['eval_xy_m']) * 100
cov = json.loads(Path('results/eval_v00/coverage.json').read_text()); real = np.array(cov['real_xy_cm']); v00 = np.array(cov['v00_xy_cm'])
v2b = json.loads(Path('results/v02b/coverage.json').read_text()); G2 = np.array(pl['targets']['positions_xy_m']) * 100
G = np.array(json.loads((RUN / 'grid.json').read_text())['positions_xy_m']) * 100
sel = json.loads((O / 'selection.json').read_text()); a3 = set(json.loads((RUN / 'amendment3_targets.json').read_text())['added_by_amendment3'])
rend = {(r['source'], r['key']): r for r in sel['rendered']}
tgt = lambda name: sorted({int(k.split('_')[2]) for e, k in sel['selection'][name]['episodes'] if k.startswith('can_target')})  # noqa: E731
S, L = tgt('strict'), tgt('loose')
sets = {'real': real, '+v00': np.vstack([real, v00]), '+v02b': np.vstack([real, G2[v2b['v02_targets'] + v2b['v02b_new_targets']]]),
        '+v03 strict': np.vstack([real, G[S]]), '+v03 loose': np.vstack([real, G[L]])}
grp = [('① front-right', 0, 4), ('② rear-right', 4, 8), ('③ rear-left', 8, 11), ('④ front-left', 11, 16), ('⑤ middle', 16, 20)]
nn = lambda D: np.linalg.norm(P[:, None] - D[None], axis=2).min(1)  # noqa: E731
table = {k: dict(per_position_worst_cm={g: round(float(nn(D)[a:b].max()), 2) for g, a, b in grp}, worst_all40_cm=round(float(nn(D).max()), 2),
                 median_cm=round(float(np.median(nn(D))), 2), within_2cm=int((nn(D) <= 2).sum()), synthetic_can_positions=int(len(D) - len(real)))
         for k, D in sets.items()}
# grid cell status from the final generation
gen = json.loads((RUN / 'gen_a3_table.json').read_text()); status = {}
for t in range(len(G)):
    a = gen.get(str(t), [])
    if not a: status[t] = 'no source within 12 cm'
    elif t in S: status[t] = 'covered (strict)'
    elif t in L: status[t] = 'covered (loose only)'
    elif any(w == 'PASS' for *_, w in a): status[t] = 'passed checks, over loose'
    else:
        c = collections.Counter(w.split(',')[0].split('+')[0] for *_, w in a); status[t] = 'rejected: ' + c.most_common(1)[0][0]
(O / 'coverage.json').write_text(json.dumps(dict(table=table, strict_targets=S, loose_targets=L, amendment3_targets=sorted(a3),
                                                 grid_status={str(t): s for t, s in status.items()}), indent=1, ensure_ascii=False))
col = {'covered (strict)': '#1f6f3f', 'covered (loose only)': '#7fc28f', 'passed checks, over loose': '#d4a017', 'no source within 12 cm': '#eeeeee'}
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4)); w = .16; x = np.arange(5)
pal = {'real': '#9aa3b5', '+v00': '#df8434', '+v02b': '#6c8ebf', '+v03 strict': '#1f6f3f', '+v03 loose': '#7fc28f'}
for i, k in enumerate(sets):
    ax[0].bar(x + (i - 2) * w, [table[k]['per_position_worst_cm'][g] for g, _, _ in grp], w, label=k, color=pal[k])
ax[0].set_xticks(x, [g for g, _, _ in grp]); ax[0].set_ylabel('Worst nearest can start (cm)'); ax[0].axhline(2, color='grey', ls=':')
ax[0].set_title('Coverage of the 40 evaluation starts (lower is better)'); ax[0].legend(fontsize=8)
seen = set()
for t, s in status.items():
    c = col.get(s, '#c0392b' if s.startswith('rejected') else '#999'); lab = s if s.startswith(('covered', 'passed', 'no source')) else 'rejected (kinematic / view / table)'
    ax[1].scatter(G[t, 1], G[t, 0], marker='s', s=150, c=c, edgecolors='k' if t in a3 else 'none', linewidths=1.2, label=None if lab in seen else lab); seen.add(lab)
    if s.startswith('rejected'): ax[1].annotate(s.split(': ')[1][:6], (G[t, 1], G[t, 0]), fontsize=5, ha='center', va='center', color='w')
ax[1].scatter(real[:, 1], real[:, 0], marker='*', s=120, c='#34415d', label='real demo starts', zorder=3)
ax[1].scatter(P[:, 1], P[:, 0], facecolors='none', edgecolors='#c0392b', s=30, label='40 eval starts', zorder=3)
ax[1].scatter([], [], marker='s', facecolors='none', edgecolors='k', label='admitted by amendment 3')
ax[1].set_xlabel('y (cm)'); ax[1].set_ylabel('x (cm), small x = rear'); ax[1].set_aspect('equal'); ax[1].invert_xaxis()
ax[1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1)); ax[1].set_title('3 cm arena grid, final generation')
fig.tight_layout(); [fig.savefig(O / f'coverage.{e}', dpi=150) for e in ('png', 'svg')]
print(json.dumps(table, indent=1, ensure_ascii=False)); print(collections.Counter(status.values()))
