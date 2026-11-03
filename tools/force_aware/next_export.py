"""Offline adapter: OpenArm telemetry (openarm_telemetry/1) -> NEXT free-motion data, one model per arm.

NEXT (FACTR2, github.com/philiphan0109/factr2_next @ 0df02c5) learns the motor effort of contact-free motion
from a history of x = [q, dq, q_cmd - q] and estimates  tau_ext_hat = tau_measured - tau_free_hat.  This module
turns telemetry sessions into that contract without ROS. Reads files only; needs numpy + pandas + pyarrow
(h5py only for the optional FACTR2-format H5 view). It imports nothing from the robot plugin.

    python tools/force_aware/next_export.py <session_dir> --out <export_root> \\
        [--contact-free-all | --contact-free-intervals intervals.json | --contact-free-episodes <dataset root>] \\
        [--attested-by NAME --block A1] [--fps 30]
    python tools/force_aware/next_export.py --h5 <export_root>/<session> [...] --arm left --out next_left_train.h5

The contract (units, ordering, every rejection rule) is docs/force_aware/NEXT_DATA_CONTRACT.md. In short:

  * arms: J1..J7 per side, selected BY NAME (LJ1..LJ7, RJ1..RJ7). Grippers (J8) go to separate tables and are
    never part of an arm model's input or target.
  * row k:  q_k, dq_k (motor-reported), tau_k (motor-reported effort, UNCALIBRATED), and cmd_q_k = the MIT position
    target actually transmitted and in effect before row k's measurement (TELEMETRY_SCHEMA section 2). Every
    value on row k was sampled no later than row k's snapshot: nothing from row k+1 is ever read for row k.
  * a row is valid for an arm only if every one of its 7 joints is valid (reasons are a bitmask, REJECT below).
    Nothing is interpolated, filled or carried forward. NaN is unknown and is rejected, never treated as fresh.
  * valid rows are cut into segments: consecutive ticks, spacing within DT_BAND periods. A NEXT window never
    crosses a segment boundary.
  * training eligibility needs an explicit contact-free attestation (whole session, or t_wall intervals).

Source files are opened read-only and never written; their sha256 is stored in the export manifest.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = "openarm_telemetry/1"
EXPORT_SCHEMA = "openarm_next_export/1"
H5_SCHEMA = "factr2_next_h5_v1"  # the official recorder's H5 schema (data_collection/h5_writer.py)
ARMS = {"left": "L", "right": "R"}
ARM_JOINTS = {arm: [f"{s}J{i}" for i in range(1, 8)] for arm, s in ARMS.items()}
GRIPPER = {arm: f"{s}J8" for arm, s in ARMS.items()}
ARM_FIELDS = ("q", "dq", "tau", "cmd_q", "cmd_dq", "cmd_kp", "cmd_kd", "cmd_tau", "sent", "desired",
              "state_age_s", "state_fresh", "stale", "cmd_n", "cmd_age_s", "status", "rx_n")
# Fields that must be finite on a valid row (the NEXT inputs/target and what decides their validity).
REQUIRED = ("q", "dq", "tau", "cmd_q", "cmd_dq", "cmd_kp", "cmd_kd", "cmd_tau", "state_age_s", "cmd_age_s")

DEFAULT_FPS = 30.0
DT_BAND = (0.5, 1.5)      # consecutive rows of one segment are this many nominal periods apart
CMD_AGE_MAX_PERIODS = 1.5  # the command in effect must be this tick's (sent after the previous snapshot)
GAIN_TOL = 1e-6            # transmitted gains are quantised: any change is a real change

# Rejection bitmask (per arm row). Stored as int64 column `reject`; 0 = valid.
REJECT = {
    "final_row": 1 << 0,        # kind = 1: written after disconnect, not an observation
    "snapshot_failed": 1 << 1,  # snapshot_ok != 1: every joint column is NaN
    "tap_down": 1 << 2,         # tap_ok != 1 for this arm: freshness unknown
    "nan": 1 << 3,              # a required value is NaN (unknown)
    "stale": 1 << 4,            # stale != 0 (1 = too old or untraceable; NaN = unknown)
    "not_fresh": 1 << 5,        # state_fresh != 1: value does not answer this tick's refresh
    "state_age": 1 << 6,        # state_age_s > max_state_age_s
    "no_command": 1 << 7,       # no MIT command transmitted yet
    "cmd_count": 1 << 8,        # cmd_n != 1: not the one-command-per-tick control loop (ramp, reset, stall)
    "cmd_age": 1 << 9,          # the command in effect is older than CMD_AGE_MAX_PERIODS
    "causal": 1 << 10,          # measurement acquired BEFORE the command in effect was sent
    "hold": 1 << 11,            # wrapper safety hold: command is a frozen re-send (also possibly contact)
    "motor_fault": 1 << 12,     # wrapper motor watchdog fault
    "status": 1 << 13,          # DM status nibble != 1 (not enabled, or a fault code)
    "saturated": 1 << 14,       # |tau| or |dq| at its decode limit, or the command clamped
    "gain_change": 1 << 15,     # cmd_kp / cmd_kd differ from the session's transmitted gains
    "ctrl_frame": 1 << 16,      # enable / disable / set-zero frame on this arm since the previous row
    "bus": 1 << 17,             # frames lost (rx_dropped grew) or CAN error frames since the previous row
    "clock": 1 << 18,           # t not strictly increasing
}
REASONS = {v: k for k, v in REJECT.items()}


class ExportError(ValueError):
    pass


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_session(session_dir):
    """(rows in tick order, meta, {file: sha256}) of one session folder. Columns are checked by NAME against
    meta['columns']; their order in the Parquet files is irrelevant."""
    session_dir = Path(session_dir)
    meta_path = session_dir / "meta.json"
    if not meta_path.is_file():
        raise ExportError(f"{session_dir}: no meta.json")
    meta = json.loads(meta_path.read_text())
    if meta.get("schema") != SCHEMA:
        raise ExportError(f"{session_dir}: schema {meta.get('schema')!r}, this exporter reads {SCHEMA!r} only")
    parts = sorted(session_dir.glob("part-*.parquet"))
    if not parts:
        raise ExportError(f"{session_dir}: no part-*.parquet")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    missing = set(meta["columns"]) - set(df.columns)
    extra = set(df.columns) - set(meta["columns"])
    if missing or extra:
        raise ExportError(f"{session_dir}: columns differ from meta.columns (missing {sorted(missing)[:5]}, "
                          f"extra {sorted(extra)[:5]})")
    df = df.sort_values("tick", kind="stable").reset_index(drop=True)
    if df["tick"].duplicated().any():
        raise ExportError(f"{session_dir}: duplicate ticks")
    hashes = {p.name: sha256(p) for p in [meta_path, *parts]}
    return df, meta, hashes


def _limits(meta, joint):
    lim = meta["decode_limits"][joint]
    p, v, t = float(lim["pMax"]), float(lim["vMax"]), float(lim["tMax"])
    return p, v, t, 2 * p / 65535, 2 * v / 4095, 2 * t / 4095  # LSBs: 16-bit q, 12-bit dq / tau


def _transmitted_gains(df, joint, loop):
    """The session's transmitted (kp, kd) for one joint: the most common pair on control-loop rows."""
    kp, kd = df[f"{joint}.cmd_kp"].to_numpy(), df[f"{joint}.cmd_kd"].to_numpy()
    ok = loop & np.isfinite(kp) & np.isfinite(kd)
    if not ok.any():
        return None
    pairs, counts = np.unique(np.stack([kp[ok], kd[ok]], axis=1), axis=0, return_counts=True)
    return [float(x) for x in pairs[np.argmax(counts)]]


def arm_mask(df, meta, joints, side, fps, max_state_age_s, require_fresh=True):
    """(reject bitmask per row, transmitted gains {joint: [kp, kd]}) for one arm's joints."""
    n = len(df)
    rej = np.zeros(n, dtype=np.int64)
    col = lambda c: df[c].to_numpy(dtype=np.float64)  # noqa: E731

    def flag(name, cond):
        rej[np.asarray(cond, dtype=bool)] |= REJECT[name]

    flag("final_row", col("kind") != 0)
    flag("snapshot_failed", col("snapshot_ok") != 1)  # NaN != 1 is True: unknown is rejected
    flag("tap_down", col(f"{side}.tap_ok") != 1)
    flag("hold", col("hold") != 0)
    flag("motor_fault", col("motor_fault") != 0)
    flag("ctrl_frame", col(f"{side}.ctrl_frames") != 0)
    drop = col(f"{side}.rx_dropped")
    grew = np.ones(n, dtype=bool)
    grew[1:] = ~(drop[1:] == drop[:-1])  # NaN on either side -> unknown -> rejected
    grew[0] = ~(drop[0] == 0)
    flag("bus", grew | (col(f"{side}.bus_errors") != 0))
    t = col("t")
    clock = np.zeros(n, dtype=bool)
    clock[1:] = ~(np.diff(t) > 0)
    flag("clock", clock | ~np.isfinite(t))

    period = 1.0 / fps
    cand_loop = (col("kind") == 0) & (col("snapshot_ok") == 1)
    gains = {}
    for j in joints:
        p_max, v_max, t_max, q_lsb, v_lsb, t_lsb = _limits(meta, j)
        v = {f: col(f"{j}.{f}") for f in ARM_FIELDS}
        flag("nan", np.any([~np.isfinite(v[f]) for f in REQUIRED], axis=0))
        flag("no_command", ~np.isfinite(v["cmd_q"]))
        flag("stale", v["stale"] != 0)
        if require_fresh:
            flag("not_fresh", v["state_fresh"] != 1)
        flag("state_age", ~(v["state_age_s"] <= max_state_age_s))
        flag("cmd_count", v["cmd_n"] != 1)
        flag("cmd_age", ~(v["cmd_age_s"] <= CMD_AGE_MAX_PERIODS * period))
        # row k's tau was produced under row k's command only if the measurement came after that command
        flag("causal", ~(v["state_age_s"] <= v["cmd_age_s"]))
        flag("status", v["status"] != 1)
        flag("saturated", (np.abs(v["tau"]) >= t_max - t_lsb) | (np.abs(v["dq"]) >= v_max - v_lsb)
             | (np.abs(v["cmd_q"]) >= p_max - q_lsb) | (np.abs(v["cmd_tau"]) >= t_max - t_lsb))
        g = _transmitted_gains(df, j, cand_loop & (v["cmd_n"] == 1))
        gains[j] = g
        if g is None:
            flag("no_command", np.ones(n, dtype=bool))
        else:
            flag("gain_change", ~((np.abs(v["cmd_kp"] - g[0]) <= GAIN_TOL) & (np.abs(v["cmd_kd"] - g[1]) <= GAIN_TOL)))
    return rej, gains


def segments(valid, tick, t, fps):
    """Segment id per row (-1 = invalid). A segment is a maximal run of valid rows with consecutive ticks whose
    spacing is within DT_BAND nominal periods. Nothing is bridged: a gap ends the segment."""
    seg = np.full(len(valid), -1, dtype=np.int64)
    lo, hi = DT_BAND[0] / fps, DT_BAND[1] / fps
    cur, prev = -1, None
    for i in range(len(valid)):
        if not valid[i]:
            prev = None
            continue
        if prev is None or tick[i] - tick[prev] != 1 or not (lo <= t[i] - t[prev] <= hi):
            cur += 1
        seg[i] = cur
        prev = i
    return seg


def episode_intervals(df, dataset_root, block=None):
    """t_wall intervals of every episode of a LOCAL LeRobot dataset recorded in this session, found by value:
    an episode's observation.state equals the telemetry `q` of a run of consecutive kind-0 rows bit-for-bit in
    float32 (TELEMETRY_SCHEMA section 7). Raises if an episode matches no window or more than one."""
    root = Path(dataset_root)
    names = json.loads((root / "meta/info.json").read_text())["features"]["observation.state"]["names"]
    data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").glob("chunk-*/file-*.parquet"))],
                     ignore_index=True)
    obs = df[df["kind"] == 0]
    tele = np.stack([obs[f"{n.removesuffix('.pos')}.q"].to_numpy() for n in names], axis=1).astype(np.float32)
    out = []
    for ep, g in data.groupby("episode_index"):
        states = np.stack(g.sort_values("frame_index")["observation.state"].to_numpy()).astype(np.float32)
        T = len(states)
        hits = [i for i in np.flatnonzero((tele[: len(tele) - T + 1] == states[0]).all(axis=1))
                if np.array_equal(tele[i:i + T], states)]
        if len(hits) != 1:
            raise ExportError(f"episode {ep} ({T} frames) matched {len(hits)} telemetry windows (need exactly 1)")
        rows = obs.iloc[hits[0]:hits[0] + T]
        out.append({"t_wall_start": float(rows["t_wall"].iloc[0]), "t_wall_end": float(rows["t_wall"].iloc[-1]),
                    "episode_index": int(ep), "first_tick": int(rows["tick"].iloc[0]), "frames": T, "block": block})
    return out


def contact_free_mask(t_wall, contact_free_all, intervals):
    if contact_free_all:
        return np.ones(len(t_wall), dtype=bool)
    m = np.zeros(len(t_wall), dtype=bool)
    for iv in intervals or []:
        m |= (t_wall >= float(iv["t_wall_start"])) & (t_wall <= float(iv["t_wall_end"]))
    return m


def hardware_signature(meta, gains):
    """Everything a NEXT model is specific to. Sessions with different signatures must not share a model."""
    prov = meta.get("provenance", {})
    return {
        "motor_types": meta.get("motor_types"),
        "decode_limits": meta.get("decode_limits"),
        "transmitted_gains": gains,
        "calibration_sha256": prov.get("calibration_sha256"),
        "urdf_sha256": prov.get("urdf_sha256"),  # cmd_tau = gravity of this URDF
        "openarm_can": prov.get("openarm_can"),
        "torque_calibration": prov.get("torque_calibration"),
        "gripper_squeeze_tau_config": meta.get("gripper_squeeze_tau_config"),
        "max_joint_speed": meta.get("max_joint_speed"),
        "mock": bool(meta.get("mock_fixture")),
    }


def export_session(session_dir, out_root, *, fps=DEFAULT_FPS, contact_free_all=False, intervals=None,
                   attestation=None, max_state_age_s=None, require_fresh=True):
    """Write <out_root>/<session id>/{left,right}_arm.parquet, {left,right}_gripper.parquet, manifest.json.
    Returns the manifest. Refuses to write inside the source session."""
    session_dir = Path(session_dir).resolve()
    out_dir = Path(out_root).resolve() / session_dir.name
    if out_dir == session_dir or session_dir in out_dir.parents:
        raise ExportError("refusing to write inside the source session folder")
    if out_dir.exists():
        raise ExportError(f"{out_dir} exists; exports are never overwritten")
    if contact_free_all and intervals:
        raise ExportError("give either contact_free_all or intervals, not both")
    if (contact_free_all or intervals) and not (attestation or {}).get("attested_by"):
        raise ExportError("a contact-free attestation needs attested_by (who verified it, and how)")
    df, meta, hashes = load_session(session_dir)
    stale_s = float(meta.get("stale_s", 0.05))
    max_state_age_s = stale_s if max_state_age_s is None else float(max_state_age_s)
    tick = df["tick"].to_numpy(dtype=np.int64)
    t = df["t"].to_numpy(dtype=np.float64)
    t_wall = df["t_wall"].to_numpy(dtype=np.float64)
    cfree = contact_free_mask(t_wall, contact_free_all, intervals)
    base = {"tick": tick, "kind": df["kind"].to_numpy(), "t": t, "t_wall": t_wall}
    out_dir.mkdir(parents=True)
    arms, grippers, all_gains = {}, {}, {}
    for arm, side in ARMS.items():
        joints = ARM_JOINTS[arm]
        rej, gains = arm_mask(df, meta, joints, side, fps, max_state_age_s, require_fresh)
        all_gains.update(gains)
        valid = rej == 0
        seg = segments(valid, tick, t, fps)
        age = np.stack([df[f"{j}.state_age_s"].to_numpy() for j in joints], axis=1)
        table = {**base, "valid": valid, "reject": rej, "segment": seg, "contact_free": cfree,
                 "t_acquired": t - age.max(axis=1)}  # oldest joint's acquisition; NaN if any is unknown
        for f in ARM_FIELDS:
            for j in joints:
                table[f"{j}.{f}"] = df[f"{j}.{f}"].to_numpy(dtype=np.float64)  # verbatim, by name
        table["hold"] = df["hold"].to_numpy()
        pd.DataFrame(table).to_parquet(out_dir / f"{arm}_arm.parquet", index=False)
        arms[arm] = _summary(rej, seg, valid, cfree)

        # Gripper: same row validity rules on J8 alone, plus the squeeze feed-forward. Never in the arm models.
        g = GRIPPER[arm]
        grej, ggains = arm_mask(df, meta, [g], side, fps, max_state_age_s, require_fresh)
        all_gains.update(ggains)
        gvalid = grej == 0
        gseg = segments(gvalid, tick, t, fps)
        gtable = {**base, "valid": gvalid, "reject": grej, "segment": gseg, "contact_free": cfree,
                  "squeeze_tau": df[f"{side}.squeeze_tau"].to_numpy(dtype=np.float64)}
        for f in ARM_FIELDS:
            gtable[f"{g}.{f}"] = df[f"{g}.{f}"].to_numpy(dtype=np.float64)
        pd.DataFrame(gtable).to_parquet(out_dir / f"{arm}_gripper.parquet", index=False)
        grippers[arm] = _summary(grej, gseg, gvalid, cfree)

    meta_gains = meta.get("gains", {})
    mismatch = []
    for arm in ARMS:
        for i, j in enumerate(ARM_JOINTS[arm] + [GRIPPER[arm]]):
            sent = all_gains.get(j)
            try:
                kp_lsb = 500.0 / 4095  # MIT kp: 12 bit over 0..500
                exp = (float(meta_gains["kp"][i]), float(meta_gains["kd"][i]))
            except (KeyError, IndexError, TypeError):
                exp = None
            if sent is None or exp is None or abs(sent[0] - exp[0]) > kp_lsb or abs(sent[1] - exp[1]) > 5.0 / 4095:
                mismatch.append({"joint": j, "transmitted": sent, "meta_gains": exp})
    manifest = {
        "schema": EXPORT_SCHEMA,
        "source": {"session_dir": str(session_dir), "session_id": session_dir.name, "schema": meta["schema"],
                   "sha256": hashes, "closed": bool(meta.get("closed")), "rows": int(len(df)),
                   "errors": meta.get("errors")},
        "settings": {"fps": fps, "dt_band_periods": list(DT_BAND), "max_state_age_s": max_state_age_s,
                     "require_fresh": require_fresh, "cmd_age_max_periods": CMD_AGE_MAX_PERIODS},
        "contact_free": {"all": bool(contact_free_all), "intervals": intervals or [], "attestation": attestation},
        "hardware_signature": hardware_signature(meta, all_gains),
        "gains_meta_mismatch": mismatch,
        "effort": {"column": "tau", "units": "N*m nominal, DM firmware current-based estimate",
                   "calibrated": meta.get("provenance", {}).get("torque_calibration") is not None},
        "arm_joints": ARM_JOINTS, "gripper_joints": GRIPPER,
        "arms": arms, "grippers": grippers,
        "reject_bits": REJECT,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _summary(rej, seg, valid, cfree):
    counts = {name: int(((rej & bit) != 0).sum()) for name, bit in REJECT.items() if ((rej & bit) != 0).any()}
    lens = np.bincount(seg[seg >= 0]) if (seg >= 0).any() else np.array([], dtype=int)
    return {"rows": int(len(rej)), "valid": int(valid.sum()), "contact_free_valid": int((valid & cfree).sum()),
            "reject_counts": counts, "segments": int(len(lens)), "segment_lengths": [int(x) for x in lens]}


def describe_reject(mask: int) -> list[str]:
    return [name for bit, name in REASONS.items() if mask & bit]


# -- reading exports ---------------------------------------------------------------------------------------------
def load_export(export_dir):
    export_dir = Path(export_dir)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    if manifest.get("schema") != EXPORT_SCHEMA:
        raise ExportError(f"{export_dir}: not an {EXPORT_SCHEMA} export")
    return manifest


def arm_arrays(export_dir, arm):
    """(table, q, dq, cmd_q, tau) of one arm, float64 arrays (rows x 7) in J1..J7 order, joined by name."""
    tab = pd.read_parquet(Path(export_dir) / f"{arm}_arm.parquet")
    js = ARM_JOINTS[arm]
    get = lambda f: np.stack([tab[f"{j}.{f}"].to_numpy(dtype=np.float64) for j in js], axis=1)  # noqa: E731
    return tab, get("q"), get("dq"), get("cmd_q"), get("tau")


def check_compatible(manifests):
    """Raise unless every export shares one hardware signature (a NEXT model is specific to it)."""
    sigs = [json.dumps(m["hardware_signature"], sort_keys=True) for m in manifests]
    if len(set(sigs)) > 1:
        a = manifests[0]["hardware_signature"]
        for m in manifests[1:]:
            diff = [k for k in a if a[k] != m["hardware_signature"].get(k)]
            if diff:
                raise ExportError(f"session {m['source']['session_id']} differs from "
                                  f"{manifests[0]['source']['session_id']} in {diff}: one NEXT model cannot "
                                  "span different hardware signatures")
    return manifests[0]["hardware_signature"] if manifests else None


def split_sessions(session_ids, val_fraction=0.2, explicit_val=None):
    """Session-level split: every row of a session goes to exactly one side. Deterministic: explicit_val if
    given, else sessions sorted by sha256(id) and the first ceil(val_fraction * n) go to validation."""
    ids = sorted(set(session_ids))
    if len(ids) != len(session_ids):
        raise ExportError("duplicate session ids")
    if explicit_val is not None:
        val = sorted(set(explicit_val))
        unknown = set(val) - set(ids)
        if unknown:
            raise ExportError(f"unknown validation sessions {sorted(unknown)}")
    else:
        order = sorted(ids, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
        val = sorted(order[: int(np.ceil(val_fraction * len(ids)))])
    train = [s for s in ids if s not in val]
    if not train or not val:
        raise ExportError(f"split needs >= 1 session on each side (train {train}, val {val})")
    return train, val


def write_factr2_h5(export_dirs, arm, path, require_contact_free=True):
    """The official recorder's H5 layout, so factr2_next.training.train runs on it unchanged with
    arm_mode: <arm> and keys "{arm}_joint_pos" ... : one ep_XXXX per segment of contact-free valid rows."""
    import h5py

    path = Path(path)
    if path.exists():
        raise ExportError(f"{path} exists; not overwritten")
    manifests = [load_export(d) for d in export_dirs]
    sig = check_compatible(manifests)
    episodes = []
    for d, m in zip(export_dirs, manifests):
        tab, q, dq, cmd, tau = arm_arrays(d, arm)
        use = tab["valid"].to_numpy() & (tab["contact_free"].to_numpy() if require_contact_free else True)
        if require_contact_free and not use.any():
            raise ExportError(f"{m['source']['session_id']}: no contact-free valid rows for the {arm} arm "
                              "(contact-free attestation missing?)")
        seg = tab["segment"].to_numpy()
        # re-segment after the contact-free cut, so an episode never spans a non-contact-free row
        seg = np.where(use, seg, -1)
        t = tab["t"].to_numpy()
        tick = tab["tick"].to_numpy()
        seg = segments(seg >= 0, tick, t, m["settings"]["fps"])
        tw_ns = np.round(tab["t_wall"].to_numpy() * 1e9).astype(np.int64)
        for s in np.unique(seg[seg >= 0]):
            i = np.flatnonzero(seg == s)
            episodes.append((m["source"]["session_id"], int(s), i, q[i], dq[i], cmd[i], tau[i], tw_ns[i]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = H5_SCHEMA
        h5.attrs["session_name"] = f"openarm_{arm}"
        h5.attrs["openarm_next_export"] = json.dumps({
            "arm": arm, "joints": ARM_JOINTS[arm], "sessions": [m["source"]["session_id"] for m in manifests],
            "hardware_signature": sig, "effort_calibrated": False,
            "joint_cmd": "transmitted MIT position target (telemetry cmd_q), in effect before the measurement"})
        for k, (sid, s, idx, q_, dq_, cmd_, tau_, tw) in enumerate(episodes):
            ep = h5.create_group(f"ep_{k:04d}")
            ep.attrs["session_id"], ep.attrs["segment"] = sid, s
            ep.attrs["first_tick"], ep.attrs["rows"] = int(idx[0]), len(idx)
            for key, arr in (("joint_pos", q_), ("joint_vel", dq_), ("joint_cmd", cmd_),
                             ("measured_joint_torque", tau_)):
                g = ep.create_group(f"{arm}_{key}")
                g.create_dataset("data", data=arr.astype(np.float32))
                g.create_dataset("timestamps", data=tw)
    return len(episodes)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="telemetry session folders (export) or export folders (--h5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS)
    ap.add_argument("--contact-free-all", action="store_true", help="the WHOLE session is verified contact-free")
    ap.add_argument("--contact-free-intervals", help="JSON list of {t_wall_start, t_wall_end, block}")
    ap.add_argument("--contact-free-episodes", metavar="DATASET_ROOT",
                    help="the kept episodes of this LOCAL LeRobot dataset (recorded in the session) are contact-free")
    ap.add_argument("--attested-by", help="who verified contact-free, and how (video, protocol block)")
    ap.add_argument("--block", help="protocol block, e.g. A1 (BATTERY_CONTACT_COLLECTION.md)")
    ap.add_argument("--allow-unfresh", action="store_true", help="accept state_fresh = 0 (still bounded by age)")
    ap.add_argument("--h5", action="store_true", help="write the FACTR2 H5 view of existing exports")
    ap.add_argument("--arm", choices=sorted(ARMS))
    a = ap.parse_args(argv)
    if a.h5:
        if not a.arm:
            ap.error("--h5 needs --arm")
        n = write_factr2_h5(a.sessions, a.arm, a.out)
        print(f"wrote {a.out}: {n} episodes ({a.arm} arm)")
        return 0
    if sum(map(bool, (a.contact_free_all, a.contact_free_intervals, a.contact_free_episodes))) > 1:
        ap.error("give at most one of --contact-free-all / --contact-free-intervals / --contact-free-episodes")
    intervals = json.loads(Path(a.contact_free_intervals).read_text()) if a.contact_free_intervals else None
    if a.contact_free_episodes:
        if len(a.sessions) != 1:
            ap.error("--contact-free-episodes takes exactly one session (the one the dataset was recorded in)")
        intervals = episode_intervals(load_session(a.sessions[0])[0], a.contact_free_episodes, a.block)
        print(f"{len(intervals)} episodes aligned: first ticks {[iv['first_tick'] for iv in intervals]}")
    att = ({"attested_by": a.attested_by, "block": a.block, "dataset": a.contact_free_episodes}
           if (a.contact_free_all or intervals) else None)
    for s in a.sessions:
        m = export_session(s, a.out, fps=a.fps, contact_free_all=a.contact_free_all, intervals=intervals,
                           attestation=att, require_fresh=not a.allow_unfresh)
        for arm in ARMS:
            print(f"{m['source']['session_id']} {arm}: {m['arms'][arm]['valid']}/{m['arms'][arm]['rows']} valid, "
                  f"{m['arms'][arm]['contact_free_valid']} contact-free, {m['arms'][arm]['segments']} segments, "
                  f"rejects {m['arms'][arm]['reject_counts']}")
        if m["gains_meta_mismatch"]:
            print(f"  WARNING transmitted gains differ from meta.gains on {len(m['gains_meta_mismatch'])} joints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
