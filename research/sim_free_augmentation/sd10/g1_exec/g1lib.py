"""SD10-G1 execution library: implements the sealed contract in ../generation_v1 (read-only) without altering it.

Common to every condition (no condition-specific code path exists in this module):
* pool-keyed input adapter (all RGB/state/action from real_v00@2c1bc860; body camera only; no wrist stream is opened);
* cold, pool-keyed source reconstruction = frozen v2/v3 code (sfa2.recon.build + sfa3.recon3 closure-only fallbacks)
  with the registered fixed body lag -1.625 frames and the fixed inherited camera (no refit);
* initial-pose adapter: one SE(3) (shortest rotation of the source can axis to world-down about the source lid point,
  then translation of the lid to the target candidate; +world-x 180 deg fallback for opposite axes) applied to the
  standing can and to the giver reach/grasp/lift waypoints, exactly as planner3 does for its own initial shifts;
* frozen planner3 G+ (D=I, Gamma=I, tau=t, scale 1), frozen ik_follow / Robot.ik, frozen action construction;
* export support [3, T-6) in native rows; nine absolute G1-A gates (SD10_G1_METRICS.md).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

EXEC = Path(__file__).resolve().parent                       # sd10/g1_exec
SD = EXEC.parent
SFA = SD.parent                                              # research/sim_free_augmentation
REPO = SFA.parents[1]                                        # worktree root
GEN = SD / "generation_v1"                                   # sealed bundle (read-only)
MAIN = Path("/home/stanley/Stanley_ws/lerobot_openarm")      # main checkout: untracked v4a seals + .venv
STORE = Path(os.environ.get("SD10_G1_STORE", "/home/stanley/Stanley_ws/sd10_g1_store"))   # large artifacts, outside Git
OUT = EXEC / "outputs"
SEAL_SHA = "2dd6ad940d06f243c096e331b2b6fc1187036ae2a4ed9d8caef5507efd0d9e8a"
FIXED_LAG = -1.625
EXPORT_HEAD, EXPORT_TAIL = 3, 6

for p in (str(SFA / "v3"), str(SFA / "v2"), str(SFA), str(SD)):
    if p not in sys.path:
        sys.path.insert(0, p)

CFG = json.loads((GEN / "locked_generation_config.json").read_text())
TGT = json.loads((GEN / "locked_targets.json").read_text())
ACC = CFG["acceptance"]
CONDS = list(CFG["conditions"])
UNION = CFG["data"]["allowed_source_union"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def check_seals():
    """G1 seal (from repo root), L1 seal, both v4A seals (main checkout, untracked there)."""
    bad = []
    for line in (GEN / "locked_artifacts.sha256").read_text().splitlines():
        h, rel = line.split(None, 1)
        if sha(REPO / rel.strip()) != h:
            bad.append(rel)
    assert sha(GEN / "locked_artifacts.sha256") == SEAL_SHA and not bad, bad
    assert sha(SD / "learning_v1/locked_artifacts.sha256") == CFG["learning_seal_sha256"]
    for rel, h in CFG["historical_seals"].items():
        assert sha(MAIN / rel) == h, rel
    for rel, h in CFG["inherited_code_and_robot_sha256"].items():
        assert sha(REPO / rel) == h, rel
    for rel, h in CFG["preserved_source_selection_sha256"].items():
        assert sha(REPO / rel) == h, rel
    assert sha(GEN / "locked_targets.json") == CFG["targets_sha256"]
    return True


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


# ----------------------------------------------------------------------------- pool input adapter
class PoolData:
    """Replacement for sfa.data inside the frozen recon code: pool episode by canonical capture ID, body cam only."""
    def __init__(self):
        from sd10 import common as C
        from sfa import data as d0
        self.CAMS, self.action_lead, self.command_events = d0.CAMS, d0.action_lead, d0.command_events   # unchanged helpers
        self.C = C
        self.pool = C.pool()
        assert C.POOL_REV == CFG["data"]["pool_revision"]

    def decode_body(self, e):
        import av
        p, t0, _ = self.pool.video(self.C.BODY, e)
        n = len(self.pool.rows(e))
        out = []
        with av.open(str(p)) as c:
            st = c.streams.video[0]
            c.seek(int(max(t0 - 0.5, 0) / st.time_base), stream=st)
            for fr in c.decode(st):
                if fr.time is None or fr.time < t0 - 0.5 / 30:
                    continue
                k = int(round((fr.time - t0) * 30))
                if k >= n:
                    break
                if k == len(out):
                    out.append(fr.to_ndarray(format="rgb24"))
        assert len(out) == n, (e, len(out), n)
        return np.stack(out), str(p)

    def load(self, e, cams=None):
        from sfa.data import Episode
        assert e in UNION, f"capture {e} is not an allowed G1 source"
        cams = cams or (self.CAMS[0],)
        assert tuple(cams) == (self.CAMS[0],), "G1 never opens wrist streams"
        S32, A32 = self.pool.vec(e, "observation.state"), self.pool.vec(e, "action")
        meta = CFG["source_metadata"][str(e)]
        assert hashlib.sha256(S32.tobytes()).hexdigest() == meta["state_sha256"], e
        assert hashlib.sha256(A32.tobytes()).hexdigest() == meta["action_sha256"], e
        F, _ = self.decode_body(e)
        S, A = S32.astype(float), A32.astype(float)
        return Episode(e, 30.0, S, A, {self.CAMS[0]: F}, self.command_events(A))


def cache_dir(e):
    return STORE / "cache" / f"pool_ep{e:02d}"


def build_source(e, device="cuda:0", with_plate=True):
    """Cold pool-keyed reconstruction with the frozen v2/v3 code; fixed lag and camera."""
    from sfa3 import recon3                      # installs the frozen closure-only fallbacks on sfa2.recon
    recon2 = recon3.recon2
    import lid_tracker as lt
    data = PoolData()
    recon2.data = data                           # pool adapter instead of the ten-only snapshot loader
    _est = recon2.rc.estimate_lag

    def fixed_lag(robot, cam, g, side, frames, uv, lags=np.arange(-3.0, 1.01, 0.25)):
        diag = _est(robot, cam, g, side, frames, uv, lags)            # diagnostic curve only
        return {"lag": FIXED_LAG, "rms_by_lag": diag["rms_by_lag"], "fitted_lag_diagnostic": diag["lag"]}

    recon2.rc.estimate_lag = fixed_lag
    cd = cache_dir(e)
    cd.mkdir(parents=True, exist_ok=True)
    lt_path = cd / f"lid_track_ep{e}.npz"
    recon2.LID_TRACK = cd / "lid_track_ep{e}.npz"
    recon3.FALLBACK.clear()
    if not lt_path.exists():
        F, _ = data.decode_body(e)
        click = recon3.clicks()[e]
        tr = lt.track(F, click)
        np.savez_compressed(lt_path, uv=tr.uv, centroid=tr.centroid, area=tr.area, status=np.array(tr.status),
                            flags=np.array(["|".join(f) for f in tr.flags]), init_click=np.array(click))
    rec = recon2.build(e, cd, device=device, with_plate=with_plate, log=lambda *a: None)
    rec.info["receiver_from_closure_only"] = bool(recon3.FALLBACK.get("receiver_from_closure_only", False))
    rec.info["init_click"] = recon3.clicks()[e]
    rec.info["lid_track_sha256"] = sha(lt_path)
    assert rec.lag == FIXED_LAG
    return rec


def light(rec):
    return dataclasses.replace(rec, F=None, masks=None, plate=None)


def asset_hash(rec):
    h = hashlib.sha256()
    for a in (rec.state, rec.action, rec.X["R"], rec.X["L"], rec.table, rec.lid_uv, rec.lid_ok.astype(np.uint8)):
        h.update(np.ascontiguousarray(np.nan_to_num(a, nan=-999.0)).tobytes())
    h.update(json.dumps([rec.lag, rec.lead, rec.ev, rec.z_table], default=float).encode())
    return h.hexdigest()


def save_light(rec, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(light(rec)))


def load_light(e):
    return pickle.loads((STORE / "assets" / f"pool_ep{e:02d}_geometry.pkl").read_bytes())


# ----------------------------------------------------------------------------- initial pose + plan
def shortest_rotation(a, b):
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:                                   # opposite axes: deterministic +world-x 180 deg fallback
        from sfa2 import se3
        return se3.rot([1, 0, 0], 180.0)[:3, :3], "x180_fallback"
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c), "shortest"


def init_transform(rec, target_xyz):
    """one SE(3): rotate about the source lid point so the can axis is world-down, then move the lid to the target."""
    from sfa2 import se3
    p0, a0 = rec.table[:3, 3].copy(), rec.table[:3, 2].copy()
    Rm, how = shortest_rotation(a0, [0.0, 0.0, -1.0])
    R4 = np.eye(4)
    R4[:3, :3] = Rm
    init = se3.trans(np.asarray(target_xyz, float) - p0) @ se3.about(R4, p0)
    return init, {"rotation_rule": how, "source_lid_xyz": p0, "source_axis": a0,
                  "translation_m": float(np.linalg.norm(np.asarray(target_xyz) - p0)),
                  "rotation_deg": float(np.degrees(np.arccos(np.clip(a0 @ [0, 0, -1.0], -1, 1))))}


def make_spec(rec, robot, init):
    """planner3.make_spec with var={} (D=I, Gamma=I, tau=t, scale 1), then the registered initial-pose waypoints:
    identical to planner3's own init-shift branch, with the generalized SE(3) in place of a planar shift."""
    from sfa3 import planner3 as pl
    spec = pl.make_spec(rec, robot, {"id": "g1"}, 1.0)
    m, Tw = spec.marks, spec.warped["T"]
    assert np.array_equal(spec.tau, np.arange(rec.T, dtype=float))
    assert all(np.allclose(x, np.eye(4)) for x in (spec.D, spec.Gamma, spec.init)) and not np.any(spec.a)
    head = [(t, init @ Tw["R"][t], "contact" if t != m["reach_end"] else "via") for t in (m["reach_end"], m["grasp"], m["lift"])]
    wp = {"R": head + list(spec.waypoints["R"]), "L": list(spec.waypoints["L"])}
    return dataclasses.replace(spec, init=init, table_new=init @ rec.table, waypoints=wp)


def requested_targets(spec):
    from sfa3 import planner3 as pl
    Tw = spec.warped["T"]
    return {s: [pl.waypoint_delta(spec.waypoints[s], Tw[s], t) @ Tw[s][t] for t in range(len(Tw[s]))] for s in "RL"}


def plan(rec, robot, spec):
    from sfa3 import planner3 as pl
    return pl.plan(rec, robot, spec, "G+")


def export(rec, p):
    rows = np.arange(EXPORT_HEAD, rec.T - EXPORT_TAIL)
    return {"native_rows": rows, "timestamp_s": (rows - EXPORT_HEAD) / 30.0, "native_origin_s": EXPORT_HEAD / 30.0,
            "state": p["state"][rows].astype(np.float32), "action": p["action"][rows].astype(np.float32),
            "can": p["can"][rows], "fps": 30}
