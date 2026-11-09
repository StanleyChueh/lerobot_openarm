import json, sys, numpy as np, cv2, torch
sys.path.insert(0, '.')
import mgw
from sfa3 import recon3
OUT = "/tmp/claude-1000/-home-stanley-Stanley-ws-lerobot-openarm/fa7abc16-fc24-4e89-be66-4d1e5716fc93/scratchpad/"
V31 = mgw.SFA / "v3_1/outputs"
E, key, t = 0, "init_x-6__G+__s1.0", 60
e = {(x["E"], x["key"]): x for x in json.load(open("../3cam_v3/selection.json"))["selected"]}[(E, key)]; a, b = e["rows"]
rec = recon3.build(E, with_plate=True, log=lambda *a: None)
R = mgw.MGWForward(rec, device="cuda:1")
z = np.load(V31 / f"ep{E}/candidates/{key}.npz"); vid = mgw.candidate_video(rec, z)
j, q, g, can = vid[a + t]
S = R.src_points(j); part = S["part"]
Mt = R.placements(q, g, can)
Ms_t = torch.tensor(Mt, dtype=torch.float32, device=R.dev); Ms_s = torch.tensor(S["Ms"], dtype=torch.float32, device=R.dev)
rel = Ms_t @ torch.linalg.inv(Ms_s)
ys, xs = ((part >= 0) & (torch.tensor(R.part_side, device=R.dev)[part.clamp(min=0)] == "R") if False else (part >= 0)).nonzero(as_tuple=True)
side = np.array(R.part_side)[part[ys, xs].cpu().numpy()]
keep = torch.tensor(side == "R", device=R.dev); ys, xs = ys[keep], xs[keep]
P = S["P"][ys, xs]; pr = part[ys, xs]; Rm = rel[pr]
Pn = torch.einsum("nij,nj->ni", Rm[:, :3, :3], P) + Rm[:, :3, 3]
uv, zz = R.project(Pn); uv = uv.cpu().numpy()
print("source R px", len(ys), "z range", float(S["sdepth"][ys, xs].min()), float(S["sdepth"][ys, xs].max()))
print("target uv x", np.percentile(uv[:, 0], [5, 50, 95]).round(0), "y", np.percentile(uv[:, 1], [5, 50, 95]).round(0))
print("in frame frac", float(((uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)).mean()))
_, lab, _ = R.zbuf(Mt); lab = lab.cpu().numpy(); tR = (lab >= 0) & (np.array(R.part_side)[np.clip(lab, 0, None)] == "R")
yy, xx = np.nonzero(tR); print("target CAD R x", np.percentile(xx, [5, 50, 95]), "y", np.percentile(yy, [5, 50, 95]))
_, labs, _ = R.zbuf(S["Ms"]); labs = labs.cpu().numpy(); sR = (labs >= 0) & (np.array(R.part_side)[np.clip(labs, 0, None)] == "R")
yy, xx = np.nonzero(sR); print("source CAD R x", np.percentile(xx, [5, 50, 95]), "y", np.percentile(yy, [5, 50, 95]))
img = rec.F[j].copy(); u = uv.astype(int); ok = (u[:, 0] >= 0) & (u[:, 0] < 640) & (u[:, 1] >= 0) & (u[:, 1] < 480)
canvas = np.zeros_like(img); canvas[u[ok, 1], u[ok, 0]] = rec.F[j][ys.cpu().numpy()[ok], xs.cpu().numpy()[ok]]
ov = (0.5 * canvas + 0.5 * np.stack([tR * 255] * 3, -1)).astype(np.uint8)
cv2.imwrite(OUT + "dbg2.jpg", cv2.cvtColor(np.concatenate([img, ov], 1), cv2.COLOR_RGB2BGR))
alias = e["alias"]; old = np.load(V31 / f"cache/ep{E}/raw/{alias}.npy", mmap_mode="r")[a + t]
new = np.load("../3cam_v3/work/body/0_init_x-6__G+__s1.0.npy")[t]
_, labc, _ = R.zbuf(Mt); labc = labc.cpu().numpy(); tm = ((labc >= 0) & (np.array(R.part_side)[np.clip(labc, 0, None)] != "B")).astype(np.uint8)
cs, _ = cv2.findContours(tm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
o1, o2 = np.ascontiguousarray(old).copy(), new.copy()
for im in (o1, o2): cv2.drawContours(im, cs, -1, (0, 255, 0), 1)
cv2.imwrite(OUT + "dbg3.jpg", cv2.cvtColor(np.concatenate([o1, o2], 1), cv2.COLOR_RGB2BGR))
