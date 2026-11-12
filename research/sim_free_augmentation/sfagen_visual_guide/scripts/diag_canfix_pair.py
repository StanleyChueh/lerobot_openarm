"""Stage 9 figure canfix_ghost_lid_debug.jpg: one R1 cross-episode pair (A=2 -> B=4, grasp phase) rendered with a render-config
override. Rows: source A | ground truth B; A with robot/object masks (red R arm, blue L arm, green can) | B; render | provenance
(blue plate, green warped, yellow borrowed, red inpainted). Run from sfagen/tools:
  python ../../sfagen_visual_guide/scripts/diag_canfix_pair.py OUT.jpg 2 4 grasp '{"object_mask_grow":60}'"""
import sys, json, numpy as np, cv2
sys.path.insert(0, '.'); sys.path.insert(0, '..')
import xfer_bench as xb
from sfagen.context import Context
from sfagen.recon import Recon
from sfagen.render import Renderer
OUT = sys.argv[1]; ea, eb = int(sys.argv[2]), int(sys.argv[3]); phase = sys.argv[4]
ctx = Context('../configs/openarm_pringles_handover.yaml'); ra, rb = Recon(ctx, ea, 'cuda:0'), Recon(ctx, eb, 'cuda:0')
def frames_of(r):
    o = r.objects[0]; other = [s for s in r.arms if s != o["arm"]][0]; tg = int(o["grasp_frame"])
    tl = min([c for c in r.ev[other]["closes"] if c > tg], default=r.T - 1); return xb.video_frames(tg, tl, r.lag, r.T)
ja, tb = frames_of(ra)[phase][1], frames_of(rb)[phase][1]
print('holder A', ra.holder_at(0, ja + ra.lag), 'holder B', rb.holder_at(0, tb + rb.lag), 'grasp', ra.objects[0]['grasp_frame'], rb.objects[0]['grasp_frame'], ja, tb)
print('can A', np.round(ra.obj_pose(0, ja + ra.lag)[:3, 3], 3), 'can B', np.round(rb.obj_pose(0, tb + rb.lag)[:3, 3], 3))
print('table A', np.round(ra.objects[0]['table'][:3, 3], 3), 'table B', np.round(rb.objects[0]['table'][:3, 3], 3))
U = Renderer(ra, 'cuda:0', {**ctx.cfg['render'], **json.loads(sys.argv[5] if len(sys.argv)>5 else '{}')})
q = {s: rb.gv[s][tb] for s in rb.arms}; g = {s: float(rb.grip_v[s][tb]) for s in rb.grip_v}
img, info = U.render(ja, q, g, [rb.obj_pose(0, tb + rb.lag)]); print(info)
pal = np.array([[0,0,0],[0,0,255],[0,255,0],[255,255,0],[255,0,255],[255,0,0],[0,255,255]], np.uint8)
S = U.source_state(ja); cat = S['cat'].cpu().numpy(); cm = np.zeros_like(img); cm[cat == 0] = (255, 0, 0); cm[cat == 1] = (0, 0, 255); cm[cat == 2] = (0, 255, 0)
top = np.concatenate([ra.F[ja], rb.F[tb]], 1); bot = np.concatenate([img, (0.5 * img + 0.5 * pal[U.last_prov]).astype(np.uint8)], 1)
mid = np.concatenate([(0.5 * ra.F[ja] + 0.5 * cm).astype(np.uint8), (0.5 * rb.F[tb] + 0.5 * cm * 0).astype(np.uint8)], 1)
cv2.imwrite(OUT, cv2.cvtColor(np.concatenate([top, mid, bot], 0), cv2.COLOR_RGB2BGR))
