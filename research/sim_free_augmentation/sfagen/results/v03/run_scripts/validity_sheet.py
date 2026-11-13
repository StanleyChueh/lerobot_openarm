"""Metric validity sheet: top-5 and bottom-5 v02 episodes by distortion score (calib re-render), each at its worst
frame (rendered | source frame), saved to results/v03/validity_sheet.jpg. usage: python validity_sheet.py"""
import json, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, '/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v02_worktree/research/sim_free_augmentation/sfagen')
from sfagen.distortion import frame_scores
R = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03_run'); OUT = Path(sys.path[0]) / 'results/v03'
T = json.loads((OUT / 'thresholds.json').read_text()); rows = sorted(T['rows'], key=lambda r: r['score'])
pick = [('LOW', r) for r in rows[:5]] + [('HIGH', r) for r in rows[-5:]]
tiles = []
for tag, r in pick:
    stem = R / f"calib/body/{r['source']}_{r['key']}"
    fr = json.loads(stem.with_suffix('.json').read_text()); s = frame_scores(fr); j = int(s.argmax())
    img = np.load(stem.with_suffix('.npy'), mmap_mode='r')[j]
    src = int(fr[j].get('source_frame', j))
    import pyarrow  # noqa: F401  (Source loader dependency)
    from sfagen.data import Source
    import yaml
    cfg = yaml.safe_load((Path(sys.path[0]) / 'configs/openarm_pringles_v03.yaml').read_text())
    S = Source(cfg['source']); body = S.body; ep = S.load(r['source'], cams=[body]); real = ep.frames[body][min(src, len(ep.frames[body]) - 1)]
    t = np.hstack([cv2.resize(np.asarray(img), (320, 240)), cv2.resize(real, (320, 240))])
    cv2.putText(t, f"{tag} score {r['score']:.3f} ep{r['source']} {r['key']} shift {r['shift_cm']:.1f}cm frame {j}", (4, 14), 0, .38, (255, 255, 0), 1)
    tiles.append(t)
cv2.imwrite(str(OUT / 'validity_sheet.jpg'), cv2.cvtColor(np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 10, 2)]), cv2.COLOR_RGB2BGR))
print('saved', [(t, round(r['score'], 3)) for t, r in pick])
