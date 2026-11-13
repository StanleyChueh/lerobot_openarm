"""Local HTML viewer for the v02b Pringles datasets (CPU). usage: python make_viewer.py  (run from sfagen/)
Writes results/v02b/viewer/index.html: every episode with source/variation/shift, an arena map, and for each can-position
episode the body + right-wrist frames at t=0 and grasp (synthetic body, source body, copied wrist, rendered wrist)."""
import base64, collections, html, io, json
from pathlib import Path
import av, cv2, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

RUN = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v02b_run'); O = Path('results/v02b/viewer'); O.mkdir(exist_ok=True)
C, R, M = RUN / 'copied/dataset', RUN / 'right_wrist/dataset', RUN / 'copied/dataset_mixed'
BODY, RW = 'observation.images.body_cam', 'observation.images.right_wrist_cam'
eps = json.loads((C / 'meta/sfa_provenance.json').read_text())['episodes']
info = json.loads((C / 'meta/info.json').read_text())
new = {k for ks in json.loads((RUN / 'MERGE.json').read_text())['new_passing'].values() for k in ks}
meta = lambda root: pd.read_parquet(sorted((root / 'meta/episodes').rglob('*.parquet')))
MC, MR, MM = meta(C), meta(R), meta(M)
valid = pd.read_parquet(sorted((C / 'data').rglob('*.parquet')), columns=['episode_index', 'frame_index', 'wrist_view_valid'])

def decode(root, m, ep, cam, which):
    row = m[m.episode_index == ep].iloc[0]; p = f'videos/{cam}/'; t0 = float(row[p + 'from_timestamp'])
    path = root / f"videos/{cam}/chunk-{int(row[p + 'chunk_index']):03d}/file-{int(row[p + 'file_index']):03d}.mp4"
    want = {int(k): None for k in which}
    with av.open(str(path)) as f:
        f.streams.video[0].thread_type = 'AUTO'
        for fr in f.decode(video=0):
            k = int(round((float(fr.pts * fr.time_base) - t0) * info['fps']))
            if k in want: want[k] = fr.to_ndarray(format='rgb24')
            if k > max(want): break
    assert all(v is not None for v in want.values()), (root, ep, cam, want.keys())
    return want

def jpg(a, w=240):
    a = cv2.resize(a, (w, int(a.shape[0] * w / a.shape[1])), interpolation=cv2.INTER_AREA)
    return 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()

# ------------------------------------------------------------------ arena map
pl = json.loads(Path('results/v02/planned_targets.json').read_text()); G = np.array(pl['targets']['positions_xy_m']) * 100
P = np.array(pl['eval_xy_m']) * 100; cov = json.loads(Path('results/eval_v00/coverage.json').read_text()); real = np.array(cov['real_xy_cm'])
can = [e for e in eps if e['family'] == 'object_position']
xy = np.array([e['schedule']['target_xy_m'] for e in can]) * 100
acc = sorted({int(e['variation'][11:14]) for e in can})
assert acc == sorted(json.loads(Path('results/v02b/coverage.json').read_text())['v02_targets'] + json.loads(Path('results/v02b/coverage.json').read_text())['v02b_new_targets'])
assert np.allclose(np.sort(xy, 0), np.sort(G[[int(e['variation'][11:14]) for e in can]], 0), atol=1e-6)
cmap = plt.get_cmap('tab10'); fig, ax = plt.subplots(figsize=(7.5, 5.6))
rej = [j for j in range(len(G)) if j not in acc]
ax.scatter(G[rej, 1], G[rej, 0], marker='s', s=140, facecolors='none', edgecolors='#c9ccd3', label=f'grid target, rejected ({len(rej)})')
for s in sorted({e['source_real_episode'] for e in can}):
    k = [i for i, e in enumerate(can) if e['source_real_episode'] == s]
    ax.scatter(xy[k, 1] + (s - 5) * .12, xy[k, 0], s=34, color=cmap(s), label=f'synthetic from demo {s}')
for s, (x, y) in enumerate(real): ax.scatter(y, x, marker='*', s=180, color=cmap(s), edgecolors='k', zorder=3); ax.annotate(str(s), (y, x), xytext=(5, 4), textcoords='offset points', fontsize=8)
ax.scatter(P[:, 1], P[:, 0], facecolors='none', edgecolors='#c0392b', s=40, label='40 eval starts')
ax.scatter([], [], marker='*', s=150, color='grey', edgecolors='k', label='real demo start (number = demo)')
ax.set_xlabel('y (cm)'); ax.set_ylabel('x (cm), small x = rear'); ax.set_aspect('equal'); ax.grid(alpha=.15)
ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1)); fig.tight_layout(); b = io.BytesIO(); fig.savefig(b, format='png', dpi=130)
MAP = 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()

# ------------------------------------------------------------------ rows
fam_name = {'object_position': 'can position', 'interaction_location': 'handover location', 'interaction_timing': 'handover timing'}
rows, cards = [], []
for e in eps:
    i = e['export_episode_index']; s = e['source_real_episode']; sh = np.array(e['shift_m'][:2]) * 100
    v = np.stack(valid[valid.episode_index == i].wrist_view_valid.to_numpy()); assert len(v) == e['n_frames']
    tag = '<b>new in v02b</b>' if e['variation'] in new else ''
    rows.append(f"<tr data-src='{s}' data-fam='{e['family']}'><td>{i}</td><td>{s}</td><td>{fam_name[e['family']]}</td><td>{html.escape(e['variation'])}</td>"
                f"<td>{sh[0]:+.1f}, {sh[1]:+.1f}</td><td>{e['n_frames']}</td><td>{v[:, 1].mean():.2f}</td><td>{tag}</td></tr>")
    if e['family'] != 'object_position': continue
    tg = e['schedule']['t_grasp']; K = [0, tg]
    sb = decode(C, MC, i, BODY, K); rb = decode(M, MM, s, BODY, K); cw = decode(C, MC, i, RW, K); rw = decode(R, MR, i, RW, K)
    cells = ''.join(f"<figure><img src='{jpg(d[k])}'><figcaption>{lab} · {'t=0' if k == 0 else f'grasp t={k}'}</figcaption></figure>"
                    for k in K for d, lab in ((sb, 'synthetic body'), (rb, f'real demo {s} body'), (cw, f'copied right wrist (valid {int(v[k, 1])})'), (rw, 'rendered right wrist')))
    tx, ty = np.array(e['schedule']['target_xy_m']) * 100
    cards.append(f"<section class='card' data-src='{s}'><h3>ep {i} · demo {s} · {e['variation']} · can at x={tx:.0f}, y={ty:.0f} cm · shift ({sh[0]:+.1f}, {sh[1]:+.1f}) cm {tag}</h3><div class='grid'>{cells}</div></section>")

src_n = collections.Counter(e['source_real_episode'] for e in eps); fam_n = collections.Counter(fam_name[e['family']] for e in eps)
assert len(eps) == info['total_episodes'] == 108 and sum(e['n_frames'] for e in eps) == info['total_frames']
opts = ''.join(f"<option value='{s}'>demo {s} ({src_n[s]} eps)</option>" for s in range(10))
page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>v02b Pringles dataset</title>
<style>:root{{--bg:#fff;--fg:#1d2330;--mut:#667;--line:#e3e5ea}}@media(prefers-color-scheme:dark){{:root{{--bg:#15181e;--fg:#e6e8ec;--mut:#99a;--line:#2b2f38}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1150px;padding:16px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}}td,th{{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}}figure{{margin:0}}img{{width:100%;border-radius:3px}}figcaption{{font-size:11px;color:var(--mut)}}
.card{{border-top:1px solid var(--line);padding:8px 0}}h3{{font-size:13.5px;margin:4px 0}}.scroll{{max-height:420px;overflow:auto}}img.map{{max-width:100%}}</style></head><body>
<h1>v02b Pringles synthetic dataset</h1>
<p>{info['total_episodes']} synthetic episodes, {info['total_frames']:,} frames, all generated from the 10 real demos of
<code>ethanCSL/openarm_pringles_lerobot_real_10_v00</code> (rev cebdee98). The mixed set adds those 10 real demos (118 episodes).
Families: {', '.join(f'{k} {v}' for k, v in fam_n.items())}. Local only, not uploaded.</p>
<p>The copied right wrist is the source demo's frame at the same time (the can sits correctly relative to the gripper; table and background are off by the shift).
Rendered right wrist is the ablation condition. <i>valid</i> is the <code>wrist_view_valid</code> flag, which SmolVLA training does not read.</p>
<h2>Where the cans start</h2><img class='map' src='{MAP}' alt='arena map'>
<p>{len(can)} can-position episodes at {len(acc)} of 35 grid targets. Demos 1, 4, 5, 6 produced none. Small horizontal offsets separate overlapping sources.</p>
<h2>All episodes</h2><p>Filter by demo: <select id='f'><option value=''>all</option>{opts}</select></p>
<div class='scroll'><table><thead><tr><th>ep</th><th>demo</th><th>family</th><th>variation</th><th>shift x, y (cm)</th><th>frames</th><th>right-wrist valid frac</th><th></th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Can-position episodes, decoded frames</h2>{''.join(cards)}
<script>document.getElementById('f').onchange=e=>{{const v=e.target.value;document.querySelectorAll('[data-src]').forEach(n=>n.style.display=!v||n.dataset.src===v?'':'none')}}</script>
</body></html>"""
(O / 'index.html').write_text(page); print(O / 'index.html', len(page) // 1024, 'KB', dict(src_n), dict(fam_n), 'targets', len(acc))
