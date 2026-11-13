"""Derive v03 configs from the frozen v02 configs (same rig, calibration, render settings, checks).

gen         : F1W planner on a 3 cm arena grid, every source within 12 cm of a target (no nearest-k limit)
all         : grid + v02's interaction_location / interaction_timing (their candidates are copied from v02, not regenerated)
wristrender : same as `all`, with the right-wrist renderer config of v02 (ablation condition)
"""
import json, yaml
import numpy as np
from pathlib import Path
S = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v02_worktree/research/sim_free_augmentation/sfagen')
RUN = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03_run')
xs = np.round(np.arange(0.09, 0.36 + 1e-9, 0.03), 2); ys = np.round(np.arange(-0.09, 0.15 + 1e-9, 0.03), 2)
grid = [[float(x), float(y)] for x in xs for y in ys]                      # x-major; superset of the v02 grid
for src, name, work in [('openarm_pringles_v02.yaml', 'openarm_pringles_v03_gen.yaml', 'gen'),
                        ('openarm_pringles_v02.yaml', 'openarm_pringles_v03.yaml', 'all'),
                        ('openarm_pringles_v02_wristrender.yaml', 'openarm_pringles_v03_wristrender.yaml', 'all_wrist')]:
    cfg = yaml.safe_load((S / 'configs' / src).read_text())
    ot = cfg['variations']['object_targets']
    ot.update(positions_xy_m=grid, planner='f1w', nearest_sources=10, max_shift_m=0.12, min_shift_m=0.005)
    cfg['variations'] = {'object_targets': ot} if work == 'gen' else {**cfg['variations'], 'object_targets': ot}
    cfg.setdefault('plan', {}).update(action_aware_bounds=True, action_bound_margin_rad=0.005)  # amendments 1+2
    cfg['checks'].update(support_check='appearance', plate_appearance_space='lab')  # amendments 1 + 3
    cfg['work_dir'] = str(RUN / ('gen_a3' if work == 'gen' else work))
    (S / 'configs' / name).write_text(yaml.safe_dump(cfg, sort_keys=False))
src = {int(k): np.array(v) for k, v in ot['source_positions_xy_m'].items()}
att = sum(1 for t in grid for p in src.values() if 0.005 <= np.linalg.norm(np.array(t) - p) <= 0.12 + 1e-12)
(RUN / 'grid.json').write_text(json.dumps(dict(positions_xy_m=grid, x=xs.tolist(), y=ys.tolist(), attempts=att), indent=1))
print('targets', len(grid), 'attempts', att)
