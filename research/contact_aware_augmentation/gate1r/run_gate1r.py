"""Gate 1R benchmark: the multi-component verifier vs the original Gate 1 verifier, development folds only.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 .venv/bin/python \
        research/contact_aware_augmentation/gate1r/run_gate1r.py --out research/contact_aware_augmentation/outputs/gate1r

Leave-one-episode-out over the pilot (0, 4, 8). Per fold, the lag, the command-response model, every threshold and the
camera are frozen from the two development episodes (camera: inherited body_cam model, never refit). The sealed
episodes are not loaded: run_sealed() refuses to run without human ground truth (task 9 stop).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gate1")]
import corruptions_r as cr  # noqa: E402
import gate1_core as gc  # noqa: E402
import spec as S  # noqa: E402
import verifier_r as vr  # noqa: E402
from gate1_core import g0  # noqa: E402

REPO = gc.REPO
PRINGLES = Path.home() / ".cache/huggingface/lerobot/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_10_v00/snapshots/cebdee98115ab6827a4ec2655335c536b5dd085b"
LAGS = np.round(np.arange(-3.0, 1.001, 0.25), 2)
METHODS = ("A", "B", "C", "D", "combined", "combined_minus_A", "combined_minus_B", "combined_minus_C", "combined_minus_D",
           "gate1_C", "gate1_C_rel")


def camera_hash(cam) -> str:
    return hashlib.sha256(json.dumps({k: cam[k] for k in ("pos", "quat_wxyz", "fx", "fy", "cx", "cy")}, sort_keys=True).encode()).hexdigest()[:16]


def load(gate1_dir: Path):
    from tools.dataset_diagnostics.loader import load_dataset
    P = load_dataset(root=PRINGLES, label="p", episodes=list(S.PILOT))
    assert set(P.vectors("action")) == set(S.PILOT)
    Q, A = P.vectors("observation.state"), P.vectors("action")
    eps = {}
    for e in S.PILOT:
        d = np.load(gate1_dir / "tracks" / f"lid_track_ep{e}.npz")
        eps[e] = vr.Sample("original", "valid", "valid", e, e, Q[e], A[e], P.names("action"), d["uv"], d["status"] == "ok")
    return P, eps


class Poses:
    def __init__(self, fk):
        self.fk, self.cache = fk, {}

    def __call__(self, q, lag):
        key = (hashlib.sha1(np.ascontiguousarray(q).tobytes()).hexdigest(), float(lag))
        if key not in self.cache:
            self.cache[key] = gc.fk_hands(self.fk, q, lag)
        return self.cache[key]


def estimate_lag(dev, eps, poses, cam):
    cost = {}
    for lag in LAGS:
        c = 0.0
        for e in dev:
            s = eps[e]
            P = poses(s.q, lag)
            ev = vr.command_events(s)
            for hand, lo, hi in ((ev["giver"], ev["g_close"] + S.W["settle"], ev["r_close"] - S.W["trans_pre"]),
                                 (ev["receiver"], ev["g_open"] + S.W["recv_offset"], ev["g_open"] + S.W["recv_offset"] + S.W["recv_len"])):
                idx = np.arange(lo, min(hi, len(s.uv)))
                sel = idx[s.ok[idx]]
                c += gc.fit_point(P[hand][sel], s.uv[sel], cam)["rms"] ** 2
        cost[float(lag)] = c
    return min(cost, key=cost.get), cost


def joint_limits(fk, names):
    m = fk.model
    out = {}
    for j, n in enumerate(names):
        if n.endswith("8.pos"):
            continue
        side = "right" if n[0] == "R" else "left"
        qi = m.idx_qs[m.getJointId(f"openarm_{side}_joint{n[2]}")]
        out[j] = (float(m.lowerPositionLimit[qi]), float(m.upperPositionLimit[qi]))
    return out


def freeze_fold(test, eps, poses, fk, cam):
    dev = [e for e in S.PILOT if e != test]
    names = eps[test].names
    arm = [i for i, n in enumerate(names) if not n.endswith("8.pos")]
    lag, cost = estimate_lag(dev, eps, poses, cam)
    model = vr.ResponseModel(arm).fit([(eps[e].q, eps[e].a) for e in dev])
    dev_rms = {e: model.rms(eps[e].q, eps[e].a) for e in dev}
    tau_A = {j: S.fold_threshold([dev_rms[e][j] for e in dev], S.A["response_floor_rad"]) for j in arm}
    stats = {e: vr.dev_statistics(eps[e], {}, poses(eps[e].q, lag), cam) for e in dev}
    tau = {"giver_pred": S.fold_threshold([stats[e].get("giver_pred") for e in dev], S.G["B_floor_px"]),
           "recv_pred": S.fold_threshold([stats[e].get("recv_pred") for e in dev], S.G["B_floor_px"]),
           "transition": S.fold_threshold([stats[e].get("transition") for e in dev], S.G["C_floor_px"]),
           "release": S.fold_threshold([stats[e].get("release") for e in dev], kind="low")}
    return {"test": test, "dev": dev, "lag": lag, "lag_cost": cost, "model": model, "tau_A": tau_A, "tau": tau,
            "limits": joint_limits(fk, names), "dev_stats": stats, "dev_response_rms": dev_rms}


# ----------------------------------------------------------------------------------------------- Gate 1 baseline
def gate1_status(s: vr.Sample, g1fold: dict, poses, tail_g1: int) -> dict:
    ev = vr.command_events(s)
    raw = ev["raw"]
    try:
        events = {"right_close": raw["right"][0][0], "left_close": raw["left"][0][0], "right_open": raw["right"][0][1]}
    except IndexError:
        return {m: "FAIL" for m in ("gate1_C", "gate1_C_rel")}
    gs = gc.Sample(s.name, s.family, s.label == "valid", s.joints_ep, s.video_ep, poses(s.q, g1fold["lag"]), events, s.uv, s.ok)
    refC = {k: np.array(v) / 1e3 for k, v in g1fold["refC_mm"].items()}
    row = {f"C.{k}": v for k, v in gc.score_C(gs, g0.BODY_CAM, min(tail_g1, len(s.uv) - 1), refC).items()}
    out = {}
    for m, key in (("gate1_C", "C"), ("gate1_C_rel", "C_rel")):
        sc, flag, why = gc.method_score(row, g1fold["thresholds"], key)
        out[m] = "FAIL" if flag else "PASS"          # what the Gate 1 verifier reported; unevaluable was not a status
        out[m + "_why"] = why
    return out


def statuses(r: dict) -> dict:
    comp = {"A": r["A"]["status"], "B": r["B"]["status"], "C": r["C"]["status"], "D": r["D"]["status"]}
    out = dict(comp)
    out["combined"] = r["status"]
    for drop in ("A", "B", "C", "D"):
        out[f"combined_minus_{drop}"] = vr.combine([v for k, v in comp.items() if k != drop])
    return out


def summarize(rows: list[dict]) -> dict:
    out = {}
    for m in METHODS:
        fam = {}
        for r in rows:
            key = (r["label"], r["family"], r["name"])
            fam.setdefault(key, {"PASS": 0, "FAIL": 0, "UNKNOWN": 0})[r["st"][m]] += 1
        per_label = {}
        for (label, family, name), c in fam.items():
            per_label.setdefault(label, {}).setdefault(family, {})[name] = c
        out[m] = per_label
    return out


def criteria(rows: list[dict]) -> dict:
    def cnt(pred, m="combined"):
        sel = [r for r in rows if pred(r)]
        return {s: sum(r["st"][m] == s for r in sel) for s in ("PASS", "FAIL", "UNKNOWN")} | {"n": len(sel)}
    orig = cnt(lambda r: r["name"] == "original")
    hard = cnt(lambda r: r["family"] == "hard_negative")
    k = {}
    k["K1_originals"] = {"counts": orig, "met": orig["FAIL"] == 0 and orig["UNKNOWN"] <= 1}
    k["K1_hard_negatives"] = {"counts": hard, "met": hard["FAIL"] <= 1}
    a1 = cnt(lambda r: r["name"] == "action_offset_all_1rad")
    k["K2_codex_action"] = {"counts": a1, "met": a1["FAIL"] == a1["n"]}
    ex = cnt(lambda r: r["name"] == "excursion_gap_60px")
    k["K2_codex_excursion"] = {"counts": ex, "met": ex["FAIL"] == ex["n"]}
    mr = cnt(lambda r: r["name"] == "missing_receiver_gate1_window")
    k["K2_codex_missing_receiver"] = {"counts": mr, "met": mr["PASS"] == 0}
    tm = {n: cnt(lambda r, n=n: r["name"] == n) for n in ("time_lag_6", "time_lead_6", "time_lag_10", "time_lead_10")}
    k["K3_temporal"] = {"counts": tm, "met": all(v["FAIL"] == v["n"] for v in tm.values())}
    ao = {n: cnt(lambda r, n=n: r["name"] == n) for n in ("action_offset_all_0.1rad", "action_offset_all_0.25rad", "action_offset_all_0.5rad", "action_offset_all_1rad")}
    k["K4_action_offset"] = {"counts": ao, "met": all(v["FAIL"] == v["n"] for v in ao.values())}
    me = [r for r in rows if r["family"] == "missing_evidence" and r["required_coverage_min"] < S.D["min_coverage"]]
    k["K5_no_silent_pass"] = {"n_below_coverage": len(me), "silent_pass": sum(r["st"]["combined"] == "PASS" for r in me),
                              "met": all(r["st"]["combined"] != "PASS" for r in me)}
    k["all_met"] = all(v["met"] for v in k.values() if isinstance(v, dict) and "met" in v)
    return k


def run_sealed(*_):
    raise SystemExit("REFUSED: the sealed episodes may be scored only after human ground truth exists "
                     "(outputs/gate1r/annotation_eval.json status EVALUATED). Current status: PENDING.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate1", default=str(HERE.parent / "outputs/gate1"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P, eps = load(Path(args.gate1))
    fk = g0.FK(P)
    poses = Poses(fk)
    cam = g0.BODY_CAM
    g1 = {f["test"]: f for f in json.loads((Path(args.gate1) / "metrics_full.json").read_text())["folds"]}
    tail_g1 = {e: int(np.where(eps[e].ok)[0].max()) - gc.WINDOW["tail_trim"] for e in S.PILOT}
    rows, folds = [], []
    for test in S.PILOT:
        fold = freeze_fold(test, eps, poses, fk, cam)
        orig = eps[test]
        P_orig = poses(orig.q, fold["lag"])
        others = {j: (eps[j].uv, eps[j].ok, len(eps[j].uv)) for j in fold["dev"]}
        samples = cr.suite(orig, P_orig, cam, tail_g1[test], others)
        for s in samples:
            ps = poses(s.q, fold["lag"])
            r = vr.verify(s, fold, ps, cam)
            tg = min(tail_g1[s.joints_ep], tail_g1[s.video_ep]) if s.family == "cross_episode" else tail_g1[test]
            g1s = gate1_status(s, g1[test], poses, tg)
            r["st"] = statuses(r) | {k: v for k, v in g1s.items() if not k.endswith("_why")}
            r["gate1_why"] = {k: v for k, v in g1s.items() if k.endswith("_why")}
            r["fold"] = test
            r["required_coverage_min"] = min(r["D"]["coverage"].values()) if r["D"].get("coverage") else 0.0
            r["meta"] = s.meta
            rows.append(r)
        folds.append({k: v for k, v in fold.items() if k not in ("model",)} |
                     {"response_lead_d": fold["model"].d, "tau_A": {eps[test].names[j]: v for j, v in fold["tau_A"].items()},
                      "dev_response_rms": {e: {eps[test].names[j]: v for j, v in d.items()} for e, d in fold["dev_response_rms"].items()},
                      "limits": None})
    crit = criteria(rows)
    res = {"spec_version": S.SPEC_VERSION, "criteria": crit, "verdict_by_stop_rule": "PARTIAL" if crit["all_met"] else "NO-GO",
           "camera": {"source": cam["source"], "hash": camera_hash(cam), "refit": False,
                      "status": "inherited silhouette fit on overlapping pool imagery; independent validation (G1) pending"},
           "folds": folds, "summary": summarize(rows), "n_samples": len(rows), "runtime_s": round(time.time() - t0, 1)}
    default = lambda x: x.tolist() if hasattr(x, "tolist") else (None if isinstance(x, float) and x != x else str(x))  # noqa: E731
    (out / "results_full.json").write_text(json.dumps(res, indent=1, default=default))
    with open(out / "samples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=default) + "\n")
    print(json.dumps({"criteria": crit, "verdict_by_stop_rule": res["verdict_by_stop_rule"], "runtime_s": res["runtime_s"]}, indent=1, default=default))


if __name__ == "__main__":
    main()
