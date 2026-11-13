"""Create the 4 export work dirs (strict/loose x copied/right_wrist) from results/v03/selection.json.
Each dir: copies of the small rig files, symlinks to all/{cache,candidates}, and hard links of only the selected renders,
so `sfagen build` (which exports every passing candidate that has a current render) exports exactly the selection.
Prints the build commands (run by the caller). usage: python make_exports.py"""
import json, os, shutil
from pathlib import Path
S = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v02_worktree/research/sim_free_augmentation/sfagen')
R = Path('/home/stanley/Stanley_ws/lerobot_openarm/artifacts/sfagen_v03_run')
sel = json.loads((S / 'results/v03/selection.json').read_text())['selection']
cmds = []
for name in ('strict', 'loose'):
    eps = sel[name]['episodes']
    for cond, cfg, rid, mid in (('copied', 'configs/openarm_pringles_v03.yaml', f'real_to_sim_10_3cam_v03_{name}', f'real_and_sim_10_3cam_v03_{name}'),
                                ('right_wrist', 'configs/openarm_pringles_v03_wristrender.yaml', f'real_to_sim_10_3cam_wristrender_v03_{name}', f'real_and_sim_10_3cam_wristrender_v03_{name}')):
        d = R / f'{name}_{cond}'
        if d.exists(): raise FileExistsError(d)
        d.mkdir(); (d / 'body').mkdir()
        for f in ('calibration.json', 'table_height.json', 'lowest_arm.json'): shutil.copy2(R / 'all' / f, d / f)
        for f in ('cache', 'candidates'): (d / f).symlink_to(R / 'all' / f)
        for e, k in eps:
            for ext in ('npy', 'json', 'fingerprint'): os.link(R / f'all/body/{e}_{k}.{ext}', d / f'body/{e}_{k}.{ext}')
        if cond == 'right_wrist':
            (d / 'wrist').mkdir()
            for e, k in eps:
                for ext in ('npy', 'json', 'fingerprint'):
                    os.link(R / f'all_wrist/wrist/{e}_{k}_right_wrist_cam.{ext}', d / f'wrist/{e}_{k}_right_wrist_cam.{ext}')
        cmds.append(f'-m sfagen build {cfg} --work-dir {d} --device cuda:0 --repo-id ethanCSL/openarm_pringles_lerobot_{rid} --mixed-repo-id ethanCSL/openarm_pringles_lerobot_{mid}')
        print(d.name, len(eps))
(R / 'build_cmds.txt').write_text('\n'.join(cmds) + '\n')
