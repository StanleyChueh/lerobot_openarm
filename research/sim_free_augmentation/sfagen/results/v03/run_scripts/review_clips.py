"""Small review sample for the user (before any upload): side-by-side MP4s decoded from the EXPORTED dataset.
Left = real source demo, right = synthetic episode; both played at the dataset fps with a running clock, so duration
and speed can be judged by eye. Episodes: the grid episodes nearest each arena corner and the middle, the largest shift,
and one hand-over variant. usage: REV=2 python review_clips.py <strict|loose>  -> results/v03r2/review/<th>/*.mp4 + index.html"""
import json, os, sys
from pathlib import Path
import av, cv2, numpy as np, pandas as pd

R2 = os.environ.get('REV', '1') == '2'; TH = sys.argv[1]
RUN = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/' + ('sfagen_v03r2_run' if R2 else 'sfagen_v03_run'))
S = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v02_worktree/research/sim_free_augmentation/sfagen')
O = S / ('results/v03r2' if R2 else 'results/v03') / 'review' / TH; O.mkdir(parents=True, exist_ok=True)
C, M = RUN / f'{TH}_copied/dataset', RUN / f'{TH}_copied/dataset_mixed'; BODY = 'observation.images.body_cam'
info = json.loads((C / 'meta/info.json').read_text()); fps = info['fps']
eps = json.loads((C / 'meta/sfa_provenance.json').read_text())['episodes']
mix = json.loads((M / 'meta/sfa_provenance.json').read_text())['episodes']; real_idx = {e['source_real_episode']: e['episode_index'] for e in mix if not e['synthetic']}
meta = lambda root: pd.read_parquet(sorted((root / 'meta/episodes').rglob('*.parquet')))  # noqa: E731
MC, MM = meta(C), meta(M)


def frames(root, m, ep):
    row = m[m.episode_index == ep].iloc[0]; p = f'videos/{BODY}/'; t0 = float(row[p + 'from_timestamp']); n = int(row['length'])
    path = root / f"videos/{BODY}/chunk-{int(row[p + 'chunk_index']):03d}/file-{int(row[p + 'file_index']):03d}.mp4"; out = [None] * n
    with av.open(str(path)) as f:
        for fr in f.decode(video=0):
            k = int(round((float(fr.pts * fr.time_base) - t0) * fps))
            if 0 <= k < n: out[k] = fr.to_ndarray(format='rgb24')
            if k >= n: break
    assert all(x is not None for x in out), (root, ep)
    return out


grid = [e for e in eps if e['family'] == 'object_workspace']
xy = {e['export_episode_index']: np.array(e['schedule']['target_xy_m']) * 100 for e in grid}
pick = {}
for name, key in (('rear-left (small x, large y)', lambda p: p[0] - p[1]), ('rear-right (small x, small y)', lambda p: p[0] + p[1]),
                  ('front-left (large x, large y)', lambda p: -p[0] - p[1]), ('front-right (large x, small y)', lambda p: -p[0] + p[1])):
    pick.setdefault(min(xy, key=lambda i: key(xy[i])), name)
mid = np.mean(list(xy.values()), 0); pick.setdefault(min(xy, key=lambda i: np.linalg.norm(xy[i] - mid)), 'middle')
pick.setdefault(max(grid, key=lambda e: np.hypot(*e['shift_m'][:2]))['export_episode_index'], 'largest shift')
pick.setdefault(next(e['export_episode_index'] for e in eps if e['family'] == 'interaction_location'), 'hand-over location variant')
rows = []
for i, why in pick.items():
    e = next(x for x in eps if x['export_episode_index'] == i); s = e['source_real_episode']
    syn = frames(C, MC, i); src = frames(M, MM, real_idx[s]); n = max(len(syn), len(src)); h, w = syn[0].shape[:2]
    out = O / f'ep{i:03d}_demo{s}_{e["variation"]}.mp4'
    with av.open(str(out), 'w') as f:
        st = f.add_stream('h264', rate=fps); st.width, st.height, st.pix_fmt = 2 * w, h, 'yuv420p'; st.options = {'crf': '23'}
        for t in range(n):
            a = src[min(t, len(src) - 1)].copy(); b = syn[min(t, len(syn) - 1)].copy()
            for img, lab, L in ((a, f'real demo {s}', len(src)), (b, f'synthetic ep {i}', len(syn))):
                cv2.putText(img, f'{lab}  {min(t, L - 1) / fps:5.2f} / {L / fps:.2f} s', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)
            for pk in st.encode(av.VideoFrame.from_ndarray(np.hstack([a, b]), format='rgb24')): f.mux(pk)
        for pk in st.encode(): f.mux(pk)
    sh = np.hypot(*e['shift_m'][:2]) * 100
    rows.append(f"<h3>{why}: ep {i}, demo {s}, {e['variation']}, shift {sh:.1f} cm, {len(syn)} vs {len(src)} frames ({len(syn) / fps:.2f} s vs {len(src) / fps:.2f} s)</h3>"
                f"<video src='{out.name}' controls loop muted width='100%'></video>")
    print(out.name, why, len(syn), len(src))
(O / 'index.html').write_text("<!doctype html><meta charset='utf-8'><title>v03 rev 2 review clips</title><body style='font:14px system-ui;max-width:1100px;margin:auto;padding:16px'>"
                              f"<h1>Review sample ({TH}): left real source, right synthetic, both at {fps} fps</h1>" + ''.join(rows))
