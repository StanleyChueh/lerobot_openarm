import sys, json, numpy as np, cv2, torch
from pathlib import Path; S = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(S))
from sfagen.context import Context
from sfagen.recon import Recon
from sfagen.render import Renderer
from sfagen.bench import ssim_masked
study = {0: -1.75, 1: -1.75, 2: -0.75, 3: -1.5, 4: -1.75, 5: -1.25, 6: -2.25, 7: -1.0, 8: -1.0, 9: -1.75}
ctx = Context(S / "configs/openarm_pringles_handover.yaml"); rows = []
for e in [int(x) for x in sys.argv[1].split(",")]:
    r = Recon(ctx, e, "cuda:0"); F = r.F
    med = torch.tensor(F[::3], device="cuda:0").float().median(0).values.cpu().numpy()
    fgm = lambda j: cv2.morphologyEx((np.abs(F[j].astype(np.int16) - med.astype(np.int16)).max(2) > 30).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0  # noqa: E731
    res = {}
    for name, L in (("default", r.lag), ("study", study[e])):
        r.lag = float(L); r._video_states(); R = Renderer(r, "cuda:0", {**ctx.cfg["render"], "part_fill": False}); ps = []
        for k in (4, 8, 15):
            for j in range(10, r.T - k - 10, 24):
                t = j + k; m = cv2.dilate((fgm(j) | fgm(t)).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
                q = {s: r.gv[s][t] for s in r.arms}; g = {s: float(r.grip_v[s][t]) for s in r.grip_v}
                im, _ = R.render(j, q, g, [r.obj_pose(kk, t + r.lag) for kk in range(len(r.objects))], exclude=(t - 5, t + 5))
                ps.append(10 * np.log10(255 ** 2 / max(float(((im.astype(float) - F[t]) ** 2)[m].mean()), 1e-6)))
        res[name] = float(np.mean(ps)); del R; torch.cuda.empty_cache()
    print(e, "lag default", -1.75, "study", study[e], {k: round(v, 3) for k, v in res.items()}, "study-default %.3f dB" % (res["study"] - res["default"]), flush=True)
