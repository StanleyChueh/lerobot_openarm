"""v02b coverage chart + accepted map (CPU). usage: python make_report.py  (run from sfagen/)"""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
O = Path('results/v02b'); pl = json.loads(Path('results/v02/planned_targets.json').read_text())
G = np.array(pl['targets']['positions_xy_m']) * 100; P = np.array(pl['eval_xy_m']) * 100
cov = json.loads(Path('results/eval_v00/coverage.json').read_text()); real = np.array(cov['real_xy_cm']); v00 = np.array(cov['v00_xy_cm'])
v2 = sorted({int(r['variation'].rsplit('_', 1)[1]) for r in json.loads(Path('results/v02/summary.json').read_text())['rows'] if r['family'] == 'object_position' and r['pass']})
new = sorted({int(k[11:14]) for ks in json.loads(Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v02b_run/MERGE.json').read_text())['new_passing'].values() for k in ks})
grp = [('① front-right', 0, 4), ('② rear-right', 4, 8), ('③ rear-left', 8, 11), ('④ front-left', 11, 16), ('⑤ middle', 16, 20)]
sets = {'real': real, '+v00': np.vstack([real, v00]), '+v02': np.vstack([real, G[v2]]), '+v02b': np.vstack([real, G[v2 + new]])}
nn = lambda D: np.linalg.norm(P[:, None] - D[None], axis=2).min(1)
table = {k: dict(per_position_worst_cm={g: round(float(nn(D)[a:b].max()), 2) for g, a, b in grp}, worst_all40_cm=round(float(nn(D).max()), 2),
                 median_cm=round(float(np.median(nn(D))), 2), within_2cm=int((nn(D) <= 2).sum())) for k, D in sets.items()}
(O / 'coverage.json').write_text(json.dumps(dict(v02_targets=v2, v02b_new_targets=new, table=table), indent=1))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
w = .2; x = np.arange(5); col = {'real': '#9aa3b5', '+v00': '#df8434', '+v02': '#6c8ebf', '+v02b': '#2f5d9e'}
for i, k in enumerate(sets):
    ax[0].bar(x + (i - 1.5) * w, [table[k]['per_position_worst_cm'][g] for g, _, _ in grp], w, label=k, color=col[k])
ax[0].set_xticks(x, [g for g, _, _ in grp]); ax[0].set_ylabel('Worst nearest can start (cm)'); ax[0].axhline(2, color='grey', ls=':')
ax[0].set_title('Coverage of the first 20 evaluation starts, lower is better'); ax[0].legend()
ax[1].scatter(G[:, 1], G[:, 0], marker='s', s=60, facecolors='none', edgecolors='#c9ccd3', label='rejected grid target')
ax[1].scatter(G[v2, 1], G[v2, 0], marker='s', s=60, c='#6c8ebf', label='v02 accepted')
ax[1].scatter(G[new, 1], G[new, 0], marker='s', s=80, c='#2f5d9e', edgecolors='k', label='v02b new')
ax[1].scatter(v00[:, 1], v00[:, 0], marker='x', s=25, c='#df8434', label='v00 synthetic')
ax[1].scatter(real[:, 1], real[:, 0], marker='*', s=110, c='#34415d', label='real sources')
ax[1].scatter(P[:, 1], P[:, 0], facecolors='none', edgecolors='#c0392b', s=30, label='40 eval starts')
ax[1].set_xlabel('y (cm)'); ax[1].set_ylabel('x (cm), small x = rear'); ax[1].set_aspect('equal'); ax[1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1)); ax[1].grid(alpha=.15)
fig.tight_layout(); [fig.savefig(O / f'coverage.{e}', dpi=150) for e in ('png', 'svg')]
print(json.dumps(table, indent=1, ensure_ascii=False))
