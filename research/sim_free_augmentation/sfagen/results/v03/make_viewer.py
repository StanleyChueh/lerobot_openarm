"""Local HTML viewer for the v03 Pringles datasets (CPU). usage: python results/v03/make_viewer.py  (run from sfagen/)
Reads the loose exports (a superset of strict). Writes results/v03/viewer/index.html:
- the coverage figure;
- every episode with demo / family / variation / shift / distortion score / strict-loose membership / visual-audit flag;
- for each grid (can-position) episode, decoded frames at t=0 and grasp (synthetic body, source body, copied wrist,
  rendered wrist) plus lift."""
import base64, collections, html, json, os
from pathlib import Path
import av, cv2, numpy as np, pandas as pd

R2 = os.environ.get('REV', '1') == '2'  # amendment 4 (rev 2): same-duration datasets
RUN = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/' + ('sfagen_v03r2_run' if R2 else 'sfagen_v03_run')); V = Path('results/v03r2' if R2 else 'results/v03'); O = V / 'viewer'; O.mkdir(exist_ok=True)
RUN1 = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03_run')
C, R, M = RUN / 'loose_copied/dataset', RUN / 'loose_right_wrist/dataset', RUN / 'loose_copied/dataset_mixed'
BODY, RW = 'observation.images.body_cam', 'observation.images.right_wrist_cam'
eps = json.loads((C / 'meta/sfa_provenance.json').read_text())['episodes']; info = json.loads((C / 'meta/info.json').read_text())
sel = json.loads((V / 'selection.json').read_text()); strict = {tuple(x) for x in sel['selection']['strict']['episodes']}
score = {(r['source'], r['key']): r['score'] for r in sel['rendered']}; T = json.loads(Path('results/v03/thresholds.json').read_text())
flags = {(f['source'], f['key']): f['flag'] for f in json.loads((V / 'visual_audit.json').read_text())['flags']} if (V / 'visual_audit.json').exists() else {}
a3 = set(json.loads((RUN1 / 'amendment3_targets.json').read_text())['added_by_amendment3'])
meta = lambda root: pd.read_parquet(sorted((root / 'meta/episodes').rglob('*.parquet')))  # noqa: E731
MC, MR, MM = meta(C), meta(R), meta(M); LEN = dict(zip(MC.episode_index, MC.length)); SRCLEN = dict(zip(MM.episode_index, MM.length))
mix = json.loads((M / 'meta/sfa_provenance.json').read_text())['episodes']
real_idx = {e['source_real_episode']: e['episode_index'] for e in mix if not e['synthetic']}


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
    assert all(v is not None for v in want.values()), (root, ep, cam)
    return want


def jpg(a, w=230):
    a = cv2.resize(a, (w, int(a.shape[0] * w / a.shape[1])), interpolation=cv2.INTER_AREA)
    return 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 78])[1]).decode()


FAM = {'object_workspace': 'can position (grid)', 'interaction_location': 'handover location', 'interaction_timing': 'handover timing'}
rows, cards = [], []
for e in eps:
    i, s, k = e['export_episode_index'], e['source_real_episode'], e['variation']; sh = np.array(e['shift_m'][:2]) * 100
    sc = score[(s, k)]; ins = (s, k) in strict; fl = flags.get((s, k), '')
    tag = ('strict + loose' if ins else 'loose only') + (' · amendment 3 cell' if k.startswith('can_target') and int(k.split('_')[2]) in a3 else '')
    rows.append(f"<tr data-src='{s}' data-set='{'strict' if ins else 'loose'}'><td>{i}</td><td>{s}</td><td>{FAM.get(e['family'], e['family'])}</td><td>{html.escape(k)}</td>"
                f"<td>{sh[0]:+.1f}, {sh[1]:+.1f}</td><td>{LEN[i]} / {SRCLEN[real_idx[s]]} ({LEN[i] / 30:.1f} s, ×{LEN[i] / SRCLEN[real_idx[s]]:.3f})</td><td>{sc:.2f}</td><td>{tag}</td><td>{html.escape(fl)}</td></tr>")
    if e['family'] != 'object_workspace': continue
    m = e['schedule']; tg = int(m.get('t_grasp_out', m['t_grasp'])); N = e['n_frames']; K = [0, tg, min(tg + 25, N - 1)]
    sfi = e['source_frame_index']; src_k = [int(round(sfi[x])) for x in K]
    sb = decode(C, MC, i, BODY, K); rb = decode(M, MM, real_idx[s], BODY, sorted(set(src_k))); cw = decode(C, MC, i, RW, K); rw = decode(R, MR, i, RW, K)
    lab = ['t=0', f'grasp t={tg}', 'lift +25']
    cells = ''.join(f"<figure><img src='{jpg(sb[x])}'><figcaption>synthetic body · {l}</figcaption></figure>"
                    f"<figure><img src='{jpg(rb[sk])}'><figcaption>real demo {s} body · src t={sk}</figcaption></figure>"
                    f"<figure><img src='{jpg(cw[x])}'><figcaption>copied right wrist · {l}</figcaption></figure>"
                    f"<figure><img src='{jpg(rw[x])}'><figcaption>rendered right wrist · {l}</figcaption></figure>" for x, sk, l in zip(K, src_k, lab))
    tx, ty = np.array(m.get('target_xy_m', [np.nan, np.nan])) * 100
    cards.append(f"<section class='card' data-src='{s}' data-set='{'strict' if ins else 'loose'}'><h3>ep {i} · demo {s} · {k} · can at x={tx:.0f}, y={ty:.0f} cm · shift {np.hypot(*sh):.1f} cm · "
                 f"score {sc:.2f} · {tag}{' · <b>flag: ' + html.escape(fl) + '</b>' if fl else ''}</h3><div class='grid'>{cells}</div></section>")

n_s = len(strict); fam_n = collections.Counter(FAM.get(e['family'], e['family']) for e in eps)
cov = base64.b64encode((V / 'coverage.png').read_bytes()).decode()
opts = ''.join(f"<option value='{s}'>demo {s}</option>" for s in range(10))
page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>v03 Pringles datasets</title>
<style>:root{{--bg:#fff;--fg:#1d2330;--mut:#667;--line:#e3e5ea}}@media(prefers-color-scheme:dark){{:root{{--bg:#15181e;--fg:#e6e8ec;--mut:#99a;--line:#2b2f38}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}}figure{{margin:0}}img{{width:100%;border-radius:3px}}figcaption{{font-size:11px;color:var(--mut)}}
.card{{border-top:1px solid var(--line);padding:8px 0}}h3{{font-size:13px;margin:4px 0}}.scroll{{max-height:420px;overflow:auto}}img.map{{max-width:100%}}</style></head><body>
<h1>v03 Pringles datasets: even arena coverage, strict / loose distortion caps</h1>
<p>Source: the 10 real demos of <code>ethanCSL/openarm_pringles_lerobot_real_10_v00</code> (rev cebdee98). <b>loose</b> = {len(eps)} synthetic episodes; <b>strict</b> = {n_s} (subset).
Families (loose): {', '.join(f'{a} {b}' for a, b in fam_n.items())}. Distortion score = 95th percentile over frames of (missing + spilled silhouette pixels) / silhouette;
strict ≤ {T['strict']:.2f} (worst v02 episode), loose ≤ {T['loose']:.2f}. Local only, nothing uploaded. Flags are Claude's visual audit (not blinded).</p>
<h2>Coverage</h2><img class='map' src='data:image/png;base64,{cov}' alt='coverage'>
<h2>All episodes</h2><p>Demo: <select id='f'><option value=''>all</option>{opts}</select> Set: <select id='g'><option value=''>loose (all)</option><option value='strict'>strict only</option><option value='loose'>loose-only extras</option></select></p>
<div class='scroll'><table><thead><tr><th>ep</th><th>demo</th><th>family</th><th>variation</th><th>shift x, y (cm)</th><th>frames / source (seconds, ratio)</th><th>score</th><th>set</th><th>visual flag</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Grid episodes: decoded frames</h2>{''.join(cards)}
<script>const f=document.getElementById('f'),g=document.getElementById('g');const up=()=>document.querySelectorAll('[data-src]').forEach(n=>n.style.display=(!f.value||n.dataset.src===f.value)&&(!g.value||n.dataset.set===g.value)?'':'none');f.onchange=up;g.onchange=up</script>
</body></html>"""
(O / 'index.html').write_text(page); print(O / 'index.html', len(page) // 1024, 'KB', len(eps), 'episodes', len(cards), 'cards')
