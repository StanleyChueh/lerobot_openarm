"""Gate 1 CPU pilot: lid tracking, rigid-attachment verifier, failure injection, LOEO evaluation (no hardware/GPU).

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 .venv/bin/python \
        research/contact_aware_augmentation/gate1/run_gate1.py --out research/contact_aware_augmentation/outputs/gate1

Episodes: Pringles real_10_v00 eps 0, 4, 8 only. Episodes 1, 2, 3, 5, 6, 7, 9 are SEALED for a later test and are
never decoded, tracked or scored here (asserted). Evaluation is leave-one-episode-out: lag, reference attachment points
and every threshold are frozen on the two development episodes before the held-out episode and its corruptions are
scored. Nothing here is ground truth except the injection labels (known by construction) and the human-verified
frame-0 click used to initialise the tracker.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate1_core as gc  # noqa: E402
import lid_tracker as lt  # noqa: E402
from gate1_core import g0  # noqa: E402

REPO = gc.REPO
PRINGLES = Path.home() / ".cache/huggingface/lerobot/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_10_v00/snapshots/cebdee98115ab6827a4ec2655335c536b5dd085b"
POOL = Path.home() / ".cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/2c1bc860347c44fc0fbae7567469d36ee49baa8d"
ANN = REPO / "research/dataset_diagnostics/annotations/real_v00.csv"
PILOT = (0, 4, 8)
SEALED = {1, 2, 3, 5, 6, 7, 9}
CAM = "observation.images.body_cam"
LAGS = np.round(np.arange(-3.0, 1.001, 0.25), 2)
METHODS = ("A", "Bp", "B", "C", "E", "B_rel", "C_rel", "C_rel_take")

_G = {}  # per-process globals (episode data)


# ----------------------------------------------------------------------------------------------- data
def decode_episode(view, e: int) -> np.ndarray:
    assert e not in SEALED, f"episode {e} is sealed"
    import av
    rel, t0 = view.video_path(CAM, e)
    n = int(view.episodes.loc[e]["length"])
    out = []
    with av.open(str(view.root / rel)) as c:
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
    return np.stack(out)


def init_clicks() -> dict:
    return {int(r["episode_index"]): (float(r["u"]), float(r["v"])) for r in csv.DictReader(open(ANN))
            if r["dataset"] == "real_v00_pool" and r["verified"] == "1"}


def load_pilot(out: Path):
    from tools.dataset_diagnostics.loader import load_dataset
    P = load_dataset(root=PRINGLES, label="pringles_real_10_v00", episodes=list(PILOT))  # sealed episodes never loaded
    assert set(P.vectors("action")) == set(PILOT), sorted(P.vectors("action"))
    ev = {r["episode"]: r for r in g0.episode_events(P)}  # action(command)-defined events
    Q, A = P.vectors("observation.state"), P.vectors("action")
    clicks = init_clicks()
    tracks, frames = {}, {}
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    for e in PILOT:
        F = decode_episode(P, e)
        tr = lt.track(F, clicks[e])
        tracks[e] = tr
        frames[e] = F
        np.savez_compressed(out / "tracks" / f"lid_track_ep{e}.npz", uv=tr.uv, centroid=tr.centroid, area=tr.area,
                            status=np.array(tr.status), flags=np.array(["|".join(f) for f in tr.flags]),
                            init_click=np.array(clicks[e]))
    data = {e: {"q": Q[e], "a": A[e], "names": P.names("action"),
                "events": {k: ev[e][k] for k in ("right_close", "left_close", "right_open")},
                "uv": tracks[e].uv, "ok": tracks[e].ok()} for e in PILOT}
    return P, data, tracks, frames


def episode(e: int, lag: float) -> gc.Episode:
    key = (e, float(lag))
    if key not in _G["ep_cache"]:
        d = _G["data"][e]
        _G["ep_cache"][key] = gc.Episode(e, d["q"], d["a"], d["names"], gc.fk_hands(_G["fk"], d["q"], lag),
                                         dict(d["events"]), d["uv"], d["ok"])
    return _G["ep_cache"][key]


def tail_end(e: int) -> int:
    return int(np.where(_G["data"][e]["ok"])[0].max()) - gc.WINDOW["tail_trim"]


def _init_worker(data):
    from tools.dataset_diagnostics.loader import load_dataset
    P = load_dataset(root=PRINGLES, label="p", episodes=list(PILOT))
    _G["data"], _G["fk"], _G["ep_cache"] = data, g0.FK(P), {}


# ----------------------------------------------------------------------------------------------- folds
def estimate_lag(dev, cam):
    cost = {}
    for lag in LAGS:
        c = 0.0
        for e in dev:
            r = gc.score_C(gc.sample_from_episode(episode(e, lag)), cam, tail_end(e))
            c += r["res_giver"] ** 2 + r["res_recv"] ** 2
        cost[float(lag)] = c
    return min(cost, key=cost.get), cost


def run_fold(test: int, cam_dev: dict, cam_test: dict, label: str = "nominal", lag_override=None,
             with_B: bool = True) -> dict:
    dev = [e for e in PILOT if e != test]
    lag, cost = (lag_override, None) if lag_override is not None else estimate_lag(dev, cam_dev)
    # references from development originals only
    devC = {e: gc.score_C(gc.sample_from_episode(episode(e, lag)), cam_dev, tail_end(e)) for e in dev}
    refC = {"giver": np.mean([np.array(devC[e]["p_giver"]) for e in dev], 0),
            "recv": np.mean([np.array(devC[e]["p_recv"]) for e in dev], 0)}
    refB = None
    if with_B:
        devB = {e: gc.score_B(gc.sample_from_episode(episode(e, lag)), cam_dev, tail_end(e)) for e in dev}
        refB = {h: np.mean([np.array(devB[e]["B_p"][h]) for e in dev], 0) for h in gc.HANDS}

    def score(s, cam, tail):
        r = {"name": s.name, "family": s.family, "valid": s.valid, "joints_ep": s.joints_ep, "video_ep": s.video_ep}
        r.update({f"C.{k}": v for k, v in gc.score_C(s, cam, tail, refC).items()})
        if with_B:
            r.update({f"B.{k}": v for k, v in gc.score_B(s, cam, tail, refB).items()})
        r.update({f"A.{k}": v for k, v in gc.score_A(s, tail).items()})
        r.update({f"E.{k}": v for k, v in gc.score_E(s).items()})
        return r

    dev_rows = [score(gc.sample_from_episode(episode(e, lag)), cam_dev, tail_end(e)) for e in dev]
    th = gc.freeze_thresholds(dev_rows)
    E = episode(test, lag)
    fit_c = gc.score_C(gc.sample_from_episode(E), g0.BODY_CAM, tail_end(test))  # injection geometry, nominal camera
    fits = {"giver": np.array(fit_c["p_giver"]), "recv": np.array(fit_c["p_recv"])}
    samples = [gc.sample_from_episode(E)] + gc.inject(E, fits, g0.BODY_CAM)
    samples += [gc.cross_pair(episode(j, lag), E) for j in dev]
    rows = []
    for s in samples:
        tail = tail_end(test) if s.family != "cross_episode" else min(tail_end(test), tail_end(s.video_ep), len(s.uv) - 1)
        r = score(s, cam_test, tail)
        r["fold_test_ep"] = test
        for m in METHODS:
            if m in ("B", "Bp", "B_rel") and not with_B:
                continue
            sc, fl, why = gc.method_score(r, th, m)
            r[f"{m}.score"], r[f"{m}.flag"], r[f"{m}.why"] = sc, fl, why
            if m in gc.LOGIC:  # the same decision with the command-only phase-logic rule switched off
                r[f"{m}.flag_geometry_only"] = gc.method_score(dict(r, **{gc.LOGIC[m]: True}), th, m)[1]
        r["artifact"] = gc.artifact_features(s, tail)
        rows.append(r)
    return {"test": test, "dev": dev, "lag": lag, "lag_cost": cost, "thresholds": th,
            "refC_mm": {k: (v * 1e3).tolist() for k, v in refC.items()},
            "refB_mm": {k: (v * 1e3).tolist() for k, v in refB.items()} if refB else None,
            "dev_rows": dev_rows, "rows": rows, "label": label}


def _fold_job(args):
    test, cam_dev, cam_test, label, lag_override, with_B = args
    return run_fold(test, cam_dev, cam_test, label, lag_override, with_B)


# ----------------------------------------------------------------------------------------------- summaries
def summarize(folds: list[dict], methods=METHODS) -> dict:
    rows = [r for f in folds for r in f["rows"]]
    out = {"n_samples": len(rows), "n_valid": sum(r["valid"] for r in rows)}
    fams = sorted({r["family"] for r in rows if not r["valid"]})
    for m in methods:
        if f"{m}.score" not in rows[0]:
            continue
        val = [r for r in rows if r["valid"]]
        inv = [r for r in rows if not r["valid"]]
        res = {"false_positives": f"{sum(r[f'{m}.flag'] for r in val)}/{len(val)}",
               "false_positive_episodes": [r["fold_test_ep"] for r in val if r[f"{m}.flag"]],
               "auc_all": gc.auc([r[f"{m}.score"] for r in inv], [r[f"{m}.score"] for r in val]),
               "detected_all": f"{sum(r[f'{m}.flag'] for r in inv)}/{len(inv)}",
               "unevaluable": sum(not np.isfinite(r[f"{m}.score"]) for r in rows),
               "per_family": {}, "per_variant": {}}
        for fam in fams:
            fi = [r for r in inv if r["family"] == fam]
            res["per_family"][fam] = {"detected": f"{sum(r[f'{m}.flag'] for r in fi)}/{len(fi)}",
                                      "auc": gc.auc([r[f"{m}.score"] for r in fi], [r[f"{m}.score"] for r in val])}
        # a corruption only counts as "detected beyond its original" if the paired original was NOT flagged
        orig_flag = {r["fold_test_ep"]: r[f"{m}.flag"] for r in val}
        res["detected_beyond_original"] = f"{sum(r[f'{m}.flag'] and not orig_flag[r['fold_test_ep']] for r in inv)}/{len(inv)}"
        for fam in fams:
            fi = [r for r in inv if r["family"] == fam]
            res["per_family"][fam]["detected_beyond_original"] = f"{sum(r[f'{m}.flag'] and not orig_flag[r['fold_test_ep']] for r in fi)}/{len(fi)}"
        if f"{m}.flag_geometry_only" in rows[0]:
            og = {r["fold_test_ep"]: r[f"{m}.flag_geometry_only"] for r in val}
            for fam in fams:
                fi = [r for r in inv if r["family"] == fam]
                res["per_family"][fam]["geometry_only_beyond_original"] = (
                    f"{sum(r[f'{m}.flag_geometry_only'] and not og[r['fold_test_ep']] for r in fi)}/{len(fi)}")
        for name in sorted({r["name"] for r in inv}):
            vi = [r for r in inv if r["name"] == name]
            res["per_variant"][name] = f"{sum(r[f'{m}.flag'] for r in vi)}/{len(vi)}"
            res.setdefault("per_variant_beyond_original", {})[name] = (
                f"{sum(r[f'{m}.flag'] and not orig_flag[r['fold_test_ep']] for r in vi)}/{len(vi)}")
        out[m] = res
    return out


def artifact_check(folds) -> dict:
    """AUC of each track-only feature, valid originals vs track-modifying corruptions (0.5 = indistinguishable)."""
    rows = [r for f in folds for r in f["rows"]]
    val = [r for r in rows if r["valid"]]
    out = {}
    for fam in ("displacement_world", "displacement_hand", "displacement_px", "temporal", "slip", "cross_episode"):
        inv = [r for r in rows if r["family"] == fam]
        out[fam] = {k: round(gc.auc([r["artifact"][k] for r in inv], [r["artifact"][k] for r in val]), 3)
                    for k in val[0]["artifact"]}
        out[fam]["n_ok_equal_to_paired_original"] = f"{sum(r['artifact']['n_ok'] == next(v['artifact']['n_ok'] for v in val if v['fold_test_ep'] == r['fold_test_ep']) for r in inv)}/{len(inv)}"
    # paired, within-episode comparison: the max |delta| of each feature between a corruption and its own original
    paired = {}
    for r in rows:
        if r["valid"] or r["family"] in ("phase",):
            continue
        o = next(v for v in val if v["fold_test_ep"] == r["fold_test_ep"])
        for k, v in r["artifact"].items():
            rel = abs(v - o["artifact"][k]) / max(abs(o["artifact"][k]), 1e-9)
            paired.setdefault(r["family"], {}).setdefault(k, []).append(rel)
    out["paired_relative_change_median"] = {f: {k: round(float(np.median(v)), 4) for k, v in d.items()} for f, d in paired.items()}
    return out


# ----------------------------------------------------------------------------------------------- task 5
def projection_reconciliation(pilot_folds) -> dict:
    """Why the Gate-0 19 px scatter is not a calibration error, and what the 8 px criterion refers to."""
    from scipy.optimize import least_squares
    from tools.dataset_diagnostics.loader import load_dataset
    pool = load_dataset(root=POOL, label="pool")
    eps = [e for e in sorted(pool.episodes.index) if e not in SEALED]
    assert not SEALED & set(eps)
    clicks = init_clicks()
    fk = g0.FK(pool)
    Q = pool.vectors("observation.state")
    ev = {r["episode"]: r for r in g0.episode_events(pool)}
    tcp, uvc = [], []
    for e in eps:
        k = ev[e]["right_close"]
        T = fk.frames(Q[e][k])["openarm_right_hand_tcp"]
        tcp.append(T)
        uvc.append(clicks[e])
    tcp, uvc = np.array(tcp), np.array(uvc)
    n = len(eps)

    def loo(model):
        errs = []
        for i in range(n):
            tr = np.array([j for j in range(n) if j != i])
            errs.append(model(tr, i))
        return np.array(errs)

    def m_raw(tr, i):
        return np.linalg.norm(g0.project(tcp[i][None, :3, 3])[0][0] - uvc[i])

    def m_pixbias(tr, i):
        b = np.median(uvc[tr] - g0.project(tcp[tr][:, :3, 3])[0], 0)
        return np.linalg.norm(g0.project(tcp[i][None, :3, 3])[0][0] + b - uvc[i])

    def m_world(tr, i):
        f = least_squares(lambda b: (g0.project(tcp[tr][:, :3, 3] + b)[0] - uvc[tr]).ravel(), np.zeros(3), loss="soft_l1", f_scale=5)
        return np.linalg.norm(g0.project(tcp[i][None, :3, 3] + f.x)[0][0] - uvc[i])

    def m_hand(tr, i):
        f = least_squares(lambda p: (gc.project_points(tcp[tr], p, g0.BODY_CAM) - uvc[tr]).ravel(), np.zeros(3), loss="soft_l1", f_scale=5)
        return np.linalg.norm(gc.project_points(tcp[i][None], f.x, g0.BODY_CAM)[0] - uvc[i])

    out = {"episodes": eps, "n": n, "excluded_sealed": sorted(SEALED),
           "what_is_compared": "projected right TCP at the first right-close COMMAND vs the human click on the can at "
                               "frame 0 (different material point, different time; the can may be pushed in between)"}
    for name, m in (("raw", m_raw), ("constant_pixel_bias_LOO", m_pixbias), ("world_3d_offset_LOO", m_world),
                    ("hand_frame_3d_offset_LOO", m_hand)):
        e = loo(m)
        out[name] = {"median_px": round(float(np.median(e)), 1), "p90_px": round(float(np.percentile(e, 90)), 1),
                     "max_px": round(float(e.max()), 1)}
    # same material point, same time: the lid track vs the rigid model during the giver/receiver holds (pilot)
    rows = [r for f in pilot_folds for r in f["rows"] if r["valid"]]
    out["same_point_same_time_rigid_residual_px"] = {
        f"ep{r['fold_test_ep']}": {"giver_rms": round(r["C.res_giver"], 2), "receiver_rms": round(r["C.res_recv"], 2),
                                   "n_giver": r["C.n_giver"], "n_receiver": r["C.n_recv"]} for r in rows}
    # pixel scale of 1 cm at the lid during the giver hold (nominal camera)
    scale = []
    for e in PILOT:
        E = episode(e, pilot_folds[[f["test"] for f in pilot_folds].index(e)]["lag"])
        c = gc.score_C(gc.sample_from_episode(E), g0.BODY_CAM, tail_end(e))
        w = gc.windows(gc.sample_from_episode(E), tail_end(e))["giver"]
        X = E.T_hand["right"][w][:, :3, :3] @ np.array(c["p_giver"]) + E.T_hand["right"][w][:, :3, 3]
        for ax in np.eye(3):
            d = np.linalg.norm(g0.project(X + 0.01 * ax)[0] - g0.project(X)[0], axis=1)
            scale.append(d)
    scale = np.concatenate(scale)
    out["px_per_cm_at_lid_giver_hold"] = {"median": round(float(np.median(scale)), 1), "p10": round(float(np.percentile(scale, 10)), 1),
                                          "p90": round(float(np.percentile(scale, 90)), 1),
                                          "note": "over world x/y/z unit displacements; depth-axis displacements project small"}
    return out


# ----------------------------------------------------------------------------------------------- M2 counterexamples
def m2_counterexamples(fold) -> dict:
    """Gate-0 M2 (image-space TCP-offset drift) vs the rigid model, on (i) a constant 20 px offset and (ii) a
    perfectly rigid synthetic lid point driven by the real right-hand trajectory of the held-out episode."""
    E = episode(fold["test"], fold["lag"])
    s = gc.sample_from_episode(E)
    w = gc.windows(s, tail_end(fold["test"]))["giver"]
    w = w[s.ok[w]]
    tcp = g0.project(E.T_hand["right"][w][:, :3, 3])[0]

    def m2(uv):
        d = (uv - tcp) - (uv[0] - tcp[0])
        return float(np.max(np.linalg.norm(d, axis=1)))
    c = gc.score_C(s, g0.BODY_CAM, tail_end(fold["test"]))
    p = np.array(c["p_giver"])
    rigid = gc.project_points(E.T_hand["right"][w], p, g0.BODY_CAM)
    f_rigid = gc.fit_point(E.T_hand["right"][w], rigid, g0.BODY_CAM)
    f_off = gc.fit_point(E.T_hand["right"][w], s.uv[w] + np.array([20.0, 0]), g0.BODY_CAM)
    return {"episode": fold["test"],
            "real_track": {"M2_px": round(m2(s.uv[w]), 1), "rigid_rms_px": round(c["res_giver"], 2)},
            "real_plus_const_20px": {"M2_px": round(m2(s.uv[w] + np.array([20.0, 0])), 1),
                                     "rigid_rms_px": round(f_off["rms"], 2),
                                     "fitted_point_shift_mm": round(float(np.linalg.norm(f_off["p"] - p) * 1e3), 1)},
            "synthetic_rigid_point": {"M2_px": round(m2(rigid), 1), "rigid_rms_px": round(f_rigid["rms"], 3)}}


# ----------------------------------------------------------------------------------------------- figures
def figures(out: Path, folds, frames, tracks):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for f in folds:
        e, lag = f["test"], f["lag"]
        E = episode(e, lag)
        s = gc.sample_from_episode(E)
        c = gc.score_C(s, g0.BODY_CAM, tail_end(e))
        pg, pr = np.array(c["p_giver"]), np.array(c["p_recv"])
        F, tr = frames[e], tracks[e]
        ev = E.events
        ks = sorted(set([0, ev["right_close"] - 20, ev["right_close"], ev["right_close"] + 15, ev["right_close"] + 40,
                         (ev["right_close"] + ev["left_close"]) // 2, ev["left_close"] - 5, ev["left_close"], ev["right_open"],
                         ev["right_open"] + 15, ev["right_open"] + 35, tail_end(e)]))
        tiles = []
        for k in ks:
            im = np.ascontiguousarray(F[k][..., ::-1])
            if not np.isnan(tr.uv[k, 0]):
                col = {"ok": (0, 255, 0), "uncertain": (0, 165, 255)}[tr.status[k]]
                cv2.drawMarker(im, tuple(map(int, tr.uv[k])), col, cv2.MARKER_CROSS, 28, 2)
            for p, h, col in ((pg, "right", (0, 0, 255)), (pr, "left", (255, 0, 0))):
                u = gc.project_points(E.T_hand[h][k:k + 1], p, g0.BODY_CAM)[0]
                cv2.circle(im, tuple(map(int, u)), 9, col, 2)
            cv2.putText(im, f"ep{e} f{k} {tr.status[k]}", (8, 26), 0, 0.8, (0, 255, 255), 2)
            tiles.append(cv2.resize(im, (320, 240)))
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(fig_dir / f"verifier_overlay_ep{e}.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    # score distributions
    rows = [r for f in folds for r in f["rows"]]
    fams = ["valid"] + sorted({r["family"] for r in rows if not r["valid"]})
    fig, axes = plt.subplots(1, len(METHODS), figsize=(4 * len(METHODS), 4.2), sharey=False)
    for ax, m in zip(axes, METHODS):
        for i, fam in enumerate(fams):
            v = [r[f"{m}.score"] for r in rows if (r["valid"] if fam == "valid" else r["family"] == fam)]
            v = np.log10(np.clip(np.asarray(v, float), 1e-3, None))
            ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)), v, s=12,
                       c="k" if fam == "valid" else "C%d" % i)
        ax.axhline(0, color="r", lw=0.8)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels(fams, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{m}: log10(score/threshold)")
    fig.tight_layout()
    fig.savefig(fig_dir / "score_by_family.png", dpi=110)
    plt.close(fig)
    # lag curves
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for f in folds:
        if f["lag_cost"]:
            ax.plot(list(f["lag_cost"]), list(f["lag_cost"].values()), label=f"dev {f['dev']} (test {f['test']})")
    ax.set_xlabel("lag (frames; pose at t+lag)")
    ax.set_ylabel("dev sum of squared rigid RMS (px^2)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "lag_estimation.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
CAL_PERTURB = ([("rot_x", (d, 0, 0), (0, 0, 0), 1.0) for d in (0.5, 1.0, 2.0)] +
               [("rot_y", (0, d, 0), (0, 0, 0), 1.0) for d in (0.5, 1.0, 2.0)] +
               [("rot_z", (0, 0, d), (0, 0, 0), 1.0) for d in (0.5, 1.0, 2.0)] +
               [("trans_x", (0, 0, 0), (d, 0, 0), 1.0) for d in (0.005, 0.01, 0.02)] +
               [("trans_y", (0, 0, 0), (0, d, 0), 1.0) for d in (0.005, 0.01, 0.02)] +
               [("trans_z", (0, 0, 0), (0, 0, d), 1.0) for d in (0.005, 0.01, 0.02)] +
               [("focal", (0, 0, 0), (0, 0, 0), f) for f in (0.97, 1.03)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--skip-calibration", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P, data, tracks, frames = load_pilot(out)
    _init_worker(data)
    track_summary = {}
    for e in PILOT:
        st = tracks[e].status
        track_summary[e] = {"frames": len(st), "ok": st.count("ok"), "uncertain": st.count("uncertain"), "lost": st.count("lost"),
                            "flags": {k: sum(k in f for f in tracks[e].flags) for k in ("border", "area_jump", "low_solidity", "center_disagree", "jump", "no_component")},
                            "tail_end": tail_end(e), "events_action": data[e]["events"]}

    jobs = [(t, g0.BODY_CAM, g0.BODY_CAM, "nominal", None, True) for t in PILOT]
    with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(data,)) as ex:
        folds = list(ex.map(_fold_job, jobs))
    nominal = summarize(folds)
    print("nominal done", round(time.time() - t0, 1), "s", flush=True)

    # calibration sensitivity (C, A, E only; B is costlier and is reported at the nominal camera)
    cal = {}
    if not args.skip_calibration:
        jobs, keys = [], []
        for name, rd, tm, fs in CAL_PERTURB:
            cam = gc.perturbed_cam(rd, tm, fs)
            mag = max(np.abs(rd).max(), np.abs(tm).max() * 1e3, abs(fs - 1) * 100) * (np.sign(fs - 1) if name == "focal" else 1)
            tag = f"{name}_{mag:+g}{'deg' if 'rot' in name else 'mm' if 'trans' in name else 'pct'}"
            for mode in ("consistent", "drift"):
                for t in PILOT:
                    lag = next(f["lag"] for f in folds if f["test"] == t)
                    jobs.append((t, cam if mode == "consistent" else g0.BODY_CAM, cam, f"{tag}|{mode}", lag, False))
                    keys.append((tag, mode))
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(data,)) as ex:
            res = list(ex.map(_fold_job, jobs))
        by = {}
        for (tag, mode), f in zip(keys, res):
            by.setdefault((tag, mode), []).append(f)
        for (tag, mode), fl in by.items():
            ss = summarize(fl, methods=("C", "C_rel"))
            val = [r for f in fl for r in f["rows"] if r["valid"]]
            cal.setdefault(tag, {})[mode] = {
                m: {"false_positives": ss[m]["false_positives"], "auc_all": round(ss[m]["auc_all"], 3),
                    "detected_all": ss[m]["detected_all"], "detected_beyond_original": ss[m]["detected_beyond_original"],
                    "per_family": {k: v["detected_beyond_original"] for k, v in ss[m]["per_family"].items()}}
                for m in ("C", "C_rel")} | {
                "valid_res_giver_px": [round(r["C.res_giver"], 2) for r in val],
                "valid_res_recv_px": [round(r["C.res_recv"], 2) for r in val],
                "valid_C_rel_why": {r["fold_test_ep"]: r["C_rel.why"] for r in val}}
        print("calibration done", round(time.time() - t0, 1), "s", flush=True)

    metrics = {
        "schema": "contact_aware_augmentation/gate1_metrics/1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo": g0.git_sha(REPO),
        "dataset": {"repo_id": "ethanCSL/openarm_pringles_lerobot_real_10_v00", "revision": "cebdee98115ab6827a4ec2655335c536b5dd085b"},
        "pilot_episodes": list(PILOT), "sealed_episodes_not_used": sorted(SEALED),
        "evaluation": "leave-one-episode-out over the 3 pilot episodes: lag, reference attachment points and thresholds "
                      "frozen on the 2 development episodes; the held-out episode's original and all its corruptions scored",
        "ground_truth": {"human": "none yet for tracking/robot keypoints (see ANNOTATION_INSTRUCTIONS.md); the tracker is "
                                  "initialised from the human-verified frame-0 click only",
                         "injection_labels": "known by construction"},
        "tracker": {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in lt.PARAMS.items()}, "per_episode": track_summary},
        "verifier": {"floors": gc.FLOORS, "margin": gc.MARGIN, "windows": gc.WINDOW},
        "folds": [{k: f[k] for k in ("test", "dev", "lag", "lag_cost", "thresholds", "refC_mm", "refB_mm")} for f in folds],
        "nominal": nominal,
        "artifact_check": artifact_check(folds),
        "m2_counterexamples": [m2_counterexamples(f) for f in folds],
        "projection_reconciliation": projection_reconciliation(folds),
        "calibration_sensitivity": cal,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(out / "samples.jsonl", "w") as fh:
        for f in folds:
            for r in f["rows"]:
                fh.write(json.dumps(r, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
    (out / "metrics_full.json").write_text(json.dumps(metrics, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    figures(out, folds, frames, tracks)
    print(json.dumps(nominal, indent=1, default=str)[:6000])
    print("total", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
