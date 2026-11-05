"""v3 export: re-validate every candidate at export time, then write LOCAL LeRobot v3 datasets (body_cam only) and
validate them by read-back. Nothing is pushed.

    env -u PYTHONPATH -u LD_LIBRARY_PATH HF_HUB_OFFLINE=1 ../../../.venv/bin/python scripts/export3.py

Datasets (outputs/lerobot/, git-ignored):
  R            the 10 real source episodes (real body_cam frames, real state/action) - training baseline
  Gp_matched   G+ candidates of the MATCHED set           } same (source episode, variation, common scale) pairs,
  C_matched    C candidates of the MATCHED set            } both conditions pass every export check
  C_all        every C candidate that passes the export checks at its own scale (full-data generation)
  R+Gp, R+C    training sets for the screen: R followed by the matched synthetic episodes
Export checks (recomputed here, not read from results.json): candidate hash == recorded hash; label construction,
per-joint lead and gripper chronology; v3 verifier; IK and common contact-waypoint tolerance; giver-window image lag
within 0.5 frame of the source lag; receiver-window lag ok or explicitly 'unknown' (counted); frozen v3 image QA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from sfa3 import labels as lb, planner3 as pl, recon3, verify3 as v3  # noqa: E402
from sfa3.recon3 import data, rb  # noqa: E402

CFG = json.loads((HERE / "configs/v3.json").read_text())
NAMES = list(rb.NAMES)
G_ = {}


def results(E):
    return json.loads((HERE / f"outputs/ep{E}/results.json").read_text())


def load_cand(E, key):
    z = np.load(HERE / f"outputs/ep{E}/candidates/{key}.npz")
    return {"state": z["state"], "action": z["action"], "valid": z["valid"], "can": z["can"], "tau": z["tau"],
            "q": {"R": z["qR"], "L": z["qL"]}, "table_new": z["table_new"]}


def image_qa(m, ident, qa, init):
    """frozen v3 rule (configs/v3.json image_qa), relative to the identity re-render of the same episode."""
    why = []
    for w in ("giver_hold", "receiver_hold"):
        ref = ident["lid"][w]["coverage"]
        cur = m["lid"][w]
        if ref <= 0:
            continue                               # no reference evidence in this window (e.g. ep6 receiver)
        if cur["coverage"] < qa["lid_coverage_ratio_min"] * ref:
            why.append(f"{w} lid coverage {cur['coverage']:.2f} < {qa['lid_coverage_ratio_min']} x {ref:.2f}")
        elif cur["median_px"] is not None and cur["median_px"] > qa["lid_median_px_max"]:
            why.append(f"{w} lid error {cur['median_px']:.1f} px")
    if init:
        r = m.get("relocation", {})
        pg = r.get("pregrasp_lid", {})
        if pg.get("coverage", 0) < qa["pregrasp_lid_coverage_min"] or (pg.get("median_px") or 1e9) > qa["pregrasp_lid_median_px_max"]:
            why.append(f"relocated can: lid coverage {pg.get('coverage')} median {pg.get('median_px')}")
        if (r.get("old_location_red_frac_median") or 0) > qa["old_location_red_residual_max"]:
            why.append(f"old location red residual {r.get('old_location_red_frac_median'):.2f}")
    return not why, why


def check_job(args):
    E, key = args
    rec, V, res = G_["rec"][E], G_["V"][E], G_["res"][E]
    c = load_cand(E, key)
    s = res["candidates"][key]
    out = {"episode": E, "key": key, "reasons": []}
    h = lb.candidate_hash(c)
    if h != s["hash"]:
        out["reasons"].append("hash mismatch")
    L = lb.validate(rec, c, G_["sl"][E], G_["sc"][E])
    if not L["ok"]:
        out["reasons"].append("labels")
    v = V(c)
    if not v["ok"]:
        out["reasons"].append("verifier:" + ",".join(v["failed"]))
    if not (s["ik"]["ok"] and s["targets_ok"]):
        out["reasons"].append("ik/targets")
    alias = res["render_alias"][key]
    m = res["clips"][alias]["metrics"]
    if abs(m["lag_giver"]["lag"] - rec.lag) > 0.5:
        out["reasons"].append(f"giver image lag {m['lag_giver']['lag']}")
    out["receiver_lag_status"] = m["lag_receiver"]["status"]
    if m["lag_receiver"]["status"] == "mismatch":
        out["reasons"].append(f"receiver image lag {m['lag_receiver'].get('lag')}")
    ident = res["clips"][res["render_alias"]["identity__C__s1.0"]]["metrics"]
    init = np.linalg.norm(s["init_shift_m"]) > 0
    ok_qa, why = image_qa(m, ident, CFG["image_qa"], init)
    out["image_qa_ok"], out["image_qa_why"] = ok_qa, why
    if not ok_qa:
        out["reasons"].append("image_qa")
    out["ok"] = not out["reasons"]
    out["hash"] = h
    return out


def features():
    return {"observation.state": {"dtype": "float32", "shape": (16,), "names": NAMES},
            "action": {"dtype": "float32", "shape": (16,), "names": NAMES},
            "observation.images.body_cam": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}}


def task_str():
    import pandas as pd
    return str(pd.read_parquet(data.ROOT / "meta/tasks.parquet").index[0])


def real_rows(E):
    ep = data.load(E, cams=(data.CAMS[0],))
    return ep.state.astype(np.float32), ep.action.astype(np.float32), ep.frames[data.CAMS[0]]


def synth_rows(E, key, alias, rec_lag, lead):
    c = load_cand(E, key)
    frames = np.load(HERE / f"outputs/cache/ep{E}/raw/{alias}.npy", mmap_mode="r")
    a, b = max(0, int(np.ceil(-rec_lag))), len(c["state"]) - lead
    return c["state"][a:b].astype(np.float32), c["action"][a:b].astype(np.float32), frames[a:b], (a, b)


def write(name, items, task):
    """items: list of dict(kind, E, key, alias) in episode order."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root = HERE / "outputs/lerobot" / name
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=f"local/{name}", fps=30, features=features(), root=root, robot_type="openarm_umeow",
                               use_videos=True, streaming_encoding=True)
    man = []
    t0 = time.time()
    for k, it in enumerate(items):
        if it["kind"] == "real":
            st, ac, fr = real_rows(it["E"])
            rows = [0, len(st)]
        else:
            st, ac, fr, rows = synth_rows(it["E"], it["key"], it["alias"], it["lag"], it["lead"])
        assert np.isfinite(st).all() and np.isfinite(ac).all(), it
        for i in range(len(st)):
            ds.add_frame({"observation.state": st[i], "action": ac[i], "observation.images.body_cam": np.ascontiguousarray(fr[i]), "task": task})
        ds.save_episode()
        man.append(it | {"episode_index": k, "rows": rows, "n_frames": len(st),
                         "state_sha256": hashlib.sha256(st.tobytes()).hexdigest(), "action_sha256": hashlib.sha256(ac.tobytes()).hexdigest()})
    ds.finalize()
    meta = {"name": name, "episodes": man, "task": task, "cameras_generated": ["observation.images.body_cam"],
            "cameras_omitted": ["observation.images.wrist_cam", "observation.images.right_wrist_cam"],
            "state_semantics": {"real": "measured", "synthetic": "planned kinematic candidate (not measured)"},
            "source_dataset": CFG["dataset"], "heldout_never_used": CFG["heldout"], "physics_used": False,
            "force_validity": "unknown"}
    (root / "meta/sfa_provenance.json").write_text(json.dumps(meta, indent=1, default=float))
    return root, man, time.time() - t0


def validate(name, root, man):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.feature_utils import dataset_to_policy_features
    info = json.loads((root / "meta/info.json").read_text())
    f = info["features"]
    rep = {"name": name}
    rep["schema"] = bool(info["codebase_version"].startswith("v3") and info["fps"] == 30 and f["action"]["names"] == NAMES
                         and f["observation.state"]["names"] == NAMES and not any("wrist" in k for k in f)
                         and list(f["observation.images.body_cam"]["shape"]) == [480, 640, 3])
    ds = LeRobotDataset(f"local/{name}", root=root)
    pf = dataset_to_policy_features(ds.meta.features)
    rep["policy_features"] = {k: [str(v.type), list(v.shape)] for k, v in pf.items()}
    epi = np.array(ds.hf_dataset["episode_index"])
    fi = np.array(ds.hf_dataset["frame_index"])
    ts = np.array(ds.hf_dataset["timestamp"])
    S = np.stack(ds.hf_dataset["observation.state"])
    A = np.stack(ds.hf_dataset["action"])
    per, psnr = [], []
    for e in man:
        idx = np.flatnonzero(epi == e["episode_index"])
        ok_b = len(idx) == e["n_frames"] and np.array_equal(fi[idx], np.arange(len(idx))) and \
            np.abs(ts[idx] - np.arange(len(idx)) / 30).max() < 1e-4 and (idx == np.arange(idx[0], idx[0] + len(idx))).all()
        st, ac = S[idx].astype(np.float32), A[idx].astype(np.float32)
        ok_a = hashlib.sha256(st.tobytes()).hexdigest() == e["state_sha256"] and hashlib.sha256(ac.tobytes()).hexdigest() == e["action_sha256"]
        # decode a sample of frames per episode
        sample = idx[:: max(1, len(idx) // 6)]
        dec = np.stack([(ds[int(i)]["observation.images.body_cam"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for i in sample])
        if e["kind"] == "real":
            src = real_rows(e["E"])[2][sample - idx[0]]
        else:
            src = np.load(HERE / f"outputs/cache/ep{e['E']}/raw/{e['alias']}.npy", mmap_mode="r")[e["rows"][0] + (sample - idx[0])]
        mse = float(((dec.astype(np.float32) - src.astype(np.float32)) ** 2).mean())
        p = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
        psnr.append(p)
        lead = lb.joint_leads(st.astype(float), ac.astype(float), np.ones(len(st), bool))
        per.append({"episode_index": e["episode_index"], "boundaries_ok": bool(ok_b), "arrays_exact": bool(ok_a), "psnr_db": p,
                    "median_joint_lead": float(np.median([v["lead_frac"] for v in lead.values()]))})
    rep["episodes"] = len(man)
    rep["frames"] = int(len(epi))
    rep["boundaries_ok"] = all(p["boundaries_ok"] for p in per)
    rep["arrays_exact"] = all(p["arrays_exact"] for p in per)
    rep["psnr_db_min"], rep["psnr_db_median"] = float(min(psnr)), float(np.median(psnr))
    rep["per_episode"] = per
    rep["ok"] = bool(rep["schema"] and rep["boundaries_ok"] and rep["arrays_exact"] and rep["psnr_db_min"] > 30)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    t_all = time.time()
    eps = CFG["source_episodes"]
    robot = rb.Robot()
    for E in eps:
        rec = recon3.build(E, with_plate=False, log=lambda *a: None)
        G_.setdefault("rec", {})[E] = rec
        G_.setdefault("V", {})[E] = v3.Verifier3(rec, robot)
        G_.setdefault("res", {})[E] = results(E)
        G_.setdefault("sl", {})[E] = lb.joint_leads(rec.state, rec.action, np.ones(rec.T, bool))
        G_.setdefault("sc", {})[E] = lb.gripper_chronology(rec.state, rec.action, np.ones(rec.T, bool))
    # candidates to check: own-scale accepted + common-scale plans (all non-identity)
    jobs = []
    for E in eps:
        res = G_["res"][E]
        for vid, pinfo in res["plans"].items():
            if vid == "identity":
                continue
            for c in pl.CONDITIONS:
                for s in {pinfo["own_scale"][c], pinfo["common_scale"]} - {None}:
                    jobs.append((E, f"{vid}__{c}__s{s}"))
    t0 = time.time()
    with get_context("fork").Pool(args.workers) as P:
        checks = {(r["episode"], r["key"]): r for r in P.map(check_job, jobs)}
    t_check = time.time() - t0
    # matched set and C_all
    matched, c_all, table = [], [], []
    for E in eps:
        res = G_["res"][E]
        rec = G_["rec"][E]
        for vid, pinfo in res["plans"].items():
            if vid == "identity":
                continue
            fam = pinfo["family"]
            cs = pinfo["common_scale"]
            row = {"episode": E, "variation": vid, "family": fam, "own_scale": pinfo["own_scale"], "common_scale": cs}
            for c in pl.CONDITIONS:
                s = pinfo["own_scale"][c]
                ck = checks.get((E, f"{vid}__{c}__s{s}")) if s is not None else None
                row[f"{c}_export_ok"] = bool(ck and ck["ok"])
                row[f"{c}_reasons"] = ck["reasons"] if ck else ["not accepted at any scale"]
                row[f"{c}_receiver_lag"] = ck["receiver_lag_status"] if ck else None
            if cs is not None:
                ok = all(checks[(E, f"{vid}__{c}__s{cs}")]["ok"] for c in pl.CONDITIONS)
                row["matched_ok"] = ok
                if ok:
                    matched.append((E, vid, cs, fam))
            else:
                row["matched_ok"] = False
            s = pinfo["own_scale"]["C"]
            if s is not None and checks[(E, f"{vid}__C__s{s}")]["ok"]:
                c_all.append((E, vid, s, fam))
            table.append(row)
    out = HERE / "outputs/dataset_export"
    out.mkdir(parents=True, exist_ok=True)
    (out / "export_checks.json").write_text(json.dumps({"rows": table, "checks": [checks[k] for k in sorted(checks)],
                                                        "matched": matched, "c_all": c_all, "check_s": t_check}, indent=1, default=float))
    print(f"export checks: {len(jobs)} candidates in {t_check:.0f} s; matched pairs {len(matched)}, C_all {len(c_all)}", flush=True)
    task = task_str()

    def item(E, vid, s, c):
        res = G_["res"][E]
        k = f"{vid}__{c}__s{s}"
        return {"kind": "synthetic", "E": E, "key": k, "alias": res["render_alias"][k], "variation": vid, "condition": c,
                "scale": s, "family": res["plans"][vid]["family"], "lag": G_["rec"][E].lag, "lead": G_["rec"][E].lead,
                "candidate_hash": checks[(E, k)]["hash"]}
    real = [{"kind": "real", "E": E, "condition": "real"} for E in eps]
    sets = {"R": real, "Gp_matched": [item(E, v, s, "G+") for E, v, s, f in matched],
            "C_matched": [item(E, v, s, "C") for E, v, s, f in matched], "C_all": [item(E, v, s, "C") for E, v, s, f in c_all]}
    sets["R+Gp"] = sets["R"] + sets["Gp_matched"]
    sets["R+C"] = sets["R"] + sets["C_matched"]
    reports, timing = {}, {}
    for name, items in sets.items():
        root, man, t_w = write(f"sfa_v3_{name.replace('+', '_plus_')}", items, task)
        t1 = time.time()
        rep = validate(f"sfa_v3_{name.replace('+', '_plus_')}", root, man)
        timing[name] = {"write_encode_s": t_w, "validate_s": time.time() - t1, "episodes": len(man), "frames": rep["frames"]}
        reports[name] = rep
        print(f"{name}: {len(man)} episodes, {rep['frames']} frames, ok={rep['ok']} psnr>={rep['psnr_db_min']:.1f} "
              f"write {t_w:.0f} s validate {timing[name]['validate_s']:.0f} s", flush=True)
        (out / f"{name}_manifest.json").write_text(json.dumps(man, indent=1, default=float))
    # leakage: no held-out pool episode shares a state fingerprint with any source episode
    from tools.dataset_diagnostics.loader import load_dataset
    import glob
    pool = load_dataset(root=glob.glob(str(Path.home() / ".cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/*"))[0], label="pool")
    S = pool.vectors("observation.state")
    fp = lambda x: hashlib.sha1(np.round(x[:60], 4).tobytes()).hexdigest()  # noqa: E731
    src_fp = {fp(G_["rec"][E].state) for E in eps}
    held = CFG["heldout"]["episodes"]
    leak = [e for e in held if fp(S[e]) in src_fp]
    reports["leakage"] = {"heldout": held, "heldout_sharing_source_fingerprint": leak, "ok": not leak,
                          "synthetic_sources": sorted({it["E"] for n in ("C_all", "Gp_matched") for it in sets[n]})}
    (out / "validation.json").write_text(json.dumps({"reports": reports, "timing": timing, "total_s": time.time() - t_all}, indent=1, default=float))
    print(json.dumps({"leakage_ok": not leak, "total_s": round(time.time() - t_all)}))


if __name__ == "__main__":
    main()
