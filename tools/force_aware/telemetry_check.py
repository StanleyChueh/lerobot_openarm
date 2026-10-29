"""Gate-2 offline analysis of one OpenArm telemetry session (openarm_telemetry/1 + openarm_frames/1).

Reads files only: the session folder, optionally the LeRobot dataset recorded alongside it (a LOCAL root, nothing
is downloaded) and optionally two teleop DIAG folders (--teleop.diag=true) for the telemetry-off vs telemetry-on
timing comparison. Needs numpy + pandas + pyarrow; imports nothing from the robot plugin, so it can be copied.

    python tools/force_aware/telemetry_check.py <session_dir> [--dataset <root> [--baseline-dataset <root>]] \\
        [--diag <diag dir of this run> --baseline-diag <off run> [--baseline-diag <2nd off run>]] [--json report.json]

Every check is PASS / FAIL / UNKNOWN (input missing). Thresholds are the constants below and are the ones in
docs/force_aware/GATE2_HARDWARE_READINESS.md section 4. Two groups:

  gate   the telemetry is valid and harmless -- every one must PASS for GO to data collection;
  stack  findings about the existing robot stack (value freshness, motor status, shutdown) -- reported with a
         verdict, reviewed by a person; a FAIL here is a result, not a telemetry defect.

Exit status: 0 every gate check PASS; 1 a gate check FAILED; 3 none failed but some are UNKNOWN.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA, FRAMES_SCHEMA = "openarm_telemetry/1", "openarm_frames/1"
JOINTS = [f"LJ{i}" for i in range(1, 9)] + [f"RJ{i}" for i in range(1, 9)]
DATASET_NAMES = [f"{s}J{i}.pos" for i in range(1, 9) for s in "RL"]  # openarm_umeow datasets: interleaved R/L
DATASET_KEYS = {"action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"}

# -- thresholds (GATE2_HARDWARE_READINESS.md section 4) ---------------------------------------------------------
LOOP_GAP_S = 0.2           # rows further apart than this are not one control loop (episode save, ramp, pause)
CMD_VISIBLE_MIN = 0.99     # joint-rows of the loop with exactly one MIT frame (cmd_n == 1)
REFRESH_VISIBLE_MIN = 0.99  # joint-rows of the loop where the refresh request was seen (state_fresh not NaN)
Q_LSB = 25.0 / 65535       # 16-bit position over +-12.5 rad
CLOCK_STEP_MAX_S = 0.002   # spread of (t_wall - t) over the session: a wall-clock step breaks every *_age_s
LAT_MEDIAN_MAX_S = 0.002   # refresh request -> reply, per joint
LAT_P99_MAX_S = 0.008      # = the follower's recv_deadline_us
REORDER_MAX = 0.001        # refreshes whose reply was delivered BEFORE the request's own echo
LOST_REPLY_MAX = 0.01      # (stack) commands + refreshes a motor never answered
SNAP_P99_MAX_S = 0.001
SNAP_LOOP_MAX_S = 0.005
PERIOD_TOL_S = 0.001       # |median period - 1/fps|
PERIOD_P99_EXTRA_S = 0.010
LATE_FACTOR, LATE_MAX = 1.5, 0.01
AB_MEDIAN_S, AB_P99_S, AB_LATE = 0.0005, 0.002, 0.005
FRESH_MIN, STALE_MAX, AGE_P99_MAX_S = 0.99, 0.01, 0.005


class Report:
    def __init__(self):
        self.items = []

    def add(self, cid, group, name, status, detail=""):
        self.items.append({"id": cid, "group": group, "check": name, "status": status, "detail": detail})

    def check(self, cid, group, name, ok, detail=""):
        self.add(cid, group, name, "UNKNOWN" if ok is None else ("PASS" if ok else "FAIL"), detail)

    def verdict(self):
        gate = [i["status"] for i in self.items if i["group"] == "gate"]
        return "NO-GO" if "FAIL" in gate else ("INCOMPLETE" if "UNKNOWN" in gate else "GO")


# -- loading ------------------------------------------------------------------------------------------------------
def load_session(path):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    parts = sorted(path.glob("part-*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame(columns=meta["columns"])
    df = df.sort_values("tick", kind="stable").reset_index(drop=True)
    fparts = sorted(path.glob("frames-*.parquet"))
    frames = pd.concat([pd.read_parquet(p) for p in fparts], ignore_index=True) if fparts else None
    if frames is not None:
        frames = frames.sort_values(["side", "seq"], kind="stable").reset_index(drop=True)
    return df, meta, frames


def load_dataset(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    data = pd.concat(pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet")))
    return info, data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def load_diag_obs_t(path):
    """perf_counter() of every robot read in a teleop DIAG folder (diag_log.py: chunk_*.npz, stream 'obs')."""
    ts = []
    for f in sorted(Path(path).glob("chunk_*.npz")):
        with np.load(f) as z:
            if "obs" in z:
                ts.append(z["obs"][:, 0])
    return np.concatenate(ts) if ts else np.array([])


def align(obs, states, names):
    """Position in `obs` (kind-0 rows) where the episode's observation.state matches q bit-for-bit in float32,
    or (None, number of matches). Same rule as telemetry.align_to_dataset."""
    tele = np.stack([obs[f"{n.removesuffix('.pos')}.q"].to_numpy() for n in names], axis=1).astype(np.float32)
    states = np.asarray(states, dtype=np.float32)
    T = len(states)
    hits = [i for i in np.flatnonzero((tele[: max(len(tele) - T + 1, 0)] == states[0]).all(axis=1))
            if np.array_equal(tele[i:i + T], states)]
    return (hits[0], 1) if len(hits) == 1 else (None, len(hits))


def loop_mask(t):
    """Rows that continue a steady loop: the previous row is less than LOOP_GAP_S earlier."""
    dt = np.diff(t, prepend=np.nan)
    return dt <= LOOP_GAP_S


def period_stats(t_list):
    dts = np.concatenate([np.diff(t) for t in t_list if len(t) > 1]) if t_list else np.array([])
    dts = dts[dts <= LOOP_GAP_S]
    if not len(dts):
        return None
    return {"n": int(len(dts)), "median": float(np.median(dts)), "p99": float(np.percentile(dts, 99)),
            "max": float(dts.max())}


def _segments(t):
    """Split a perf_counter series into steady-loop segments."""
    if not len(t):
        return []
    cut = np.flatnonzero(np.diff(t) > LOOP_GAP_S) + 1
    return np.split(np.asarray(t), cut)


def late_frac(t_list, period):
    dts = np.concatenate([np.diff(t) for t in t_list if len(t) > 1]) if t_list else np.array([])
    dts = dts[dts <= LOOP_GAP_S]
    return float((dts > LATE_FACTOR * period).mean()) if len(dts) else None


# -- the checks ---------------------------------------------------------------------------------------------------
def _ab(on, off, period):
    """(ok, detail) comparing two lists of loop segments; ok None if either has no loop."""
    a, b = period_stats(on), period_stats(off)
    if a is None or b is None:
        return None, "no loop rows"
    la, lb = late_frac(on, period), late_frac(off, period)
    ok = abs(a["median"] - b["median"]) <= AB_MEDIAN_S and a["p99"] - b["p99"] <= AB_P99_S and la - lb <= AB_LATE
    return ok, (f"median {a['median'] * 1e3:.2f} vs {b['median'] * 1e3:.2f} ms, p99 {a['p99'] * 1e3:.2f} vs "
                f"{b['p99'] * 1e3:.2f} ms, late {la:.3%} vs {lb:.3%}")


def run(session, dataset=None, diag=None, baseline_diag=None, baseline_dataset=None):
    """baseline_diag: one telemetry-off DIAG folder, or a list (A-B-A design: the two off runs must agree with
    each other first, else the comparison is inconclusive)."""
    R = Report()
    df, meta, frames = load_session(session)
    obs = df[df["kind"] == 0]  # index = position in df
    fin = df[df["kind"] == 1]
    sides = ["L", "R"]
    can = meta.get("can_ids", {})
    send, recv, refresh_id = can.get("send", []), can.get("recv", []), can.get("refresh", 0x7FF)

    # G1 session integrity
    ticks = df["tick"].to_numpy()
    ok = (meta.get("schema") == SCHEMA and list(df.columns) == list(meta["columns"]) and meta.get("closed") is True
          and np.array_equal(ticks, np.arange(len(df))) and meta.get("rows") == len(df) and len(fin) == 1
          and all(v == 0 for v in meta.get("errors", {}).values()) and bool((df["snapshot_ok"] == 1).all()))
    R.check("G1", "gate", "session complete: schema, closed, consecutive ticks, no snapshot/writer/poll errors", ok,
            f"{len(df)} rows ({len(obs)} obs), closed={meta.get('closed')}, errors={meta.get('errors')}, "
            f"snapshot_ok=0 rows {int((df['snapshot_ok'] != 1).sum())}")

    # G2 taps
    tap_rows = {s: float(df[f"{s}.tap_ok"].mean()) for s in sides}
    R.check("G2", "gate", "both CAN receive taps open at start and on every row",
            all(meta.get("tap_available", {}).get(s) for s in sides) and all(v == 1.0 for v in tap_rows.values()),
            f"tap_available={meta.get('tap_available')}, tap_ok mean={tap_rows}")

    # G3 frame loss
    taps = meta.get("taps", {})
    drop = {s: float(np.nanmax(df[f"{s}.rx_dropped"])) if df[f"{s}.rx_dropped"].notna().any() else None for s in sides}
    ok = all(d == 0 for d in drop.values()) and all(taps.get(s, {}).get("host_dropped") == 0 for s in sides) \
        if all(d is not None for d in drop.values()) and taps else None
    R.check("G3", "gate", "no frame lost before telemetry read it (kernel queue + host buffer)", ok,
            f"rx_dropped max={drop}, taps={taps}")

    # G4 kernel timestamps and clock
    if frames is not None and len(frames):
        inv = {s: int((np.diff(frames.loc[frames.side == s, "t_ns"].to_numpy(dtype=float)) < 0).sum()) for s in sides}
        nulls = int(frames["t_ns"].isna().sum())
    else:
        inv, nulls = None, None
    untimed = {s: taps.get(s, {}).get("untimed") for s in sides}
    off = (df["t_wall"] - df["t"]).to_numpy()
    spread = float(off.max() - off.min()) if len(off) else None
    ok = None if inv is None or not taps else (nulls == 0 and all(u == 0 for u in untimed.values())
                                               and all(v == 0 for v in inv.values()) and spread <= CLOCK_STEP_MAX_S)
    spread_ms = "n/a" if spread is None else f"{spread * 1e3:.3f} ms"
    R.check("G4", "gate", "every frame kernel-timestamped, stamps non-decreasing in delivery order, no wall-clock step",
            ok, f"untimed={untimed}, null t_ns={nulls}, inversions={inv}, (t_wall - t) spread={spread_ms}")

    # The rows of the steady control loop: dataset episodes if given, else every obs row continuing a loop.
    ep_rows, ep_info, fps = None, [], 30
    if dataset is not None:
        info, data = load_dataset(dataset)
        fps = info.get("fps", 30)
        names = info["features"]["observation.state"]["names"]
        ep_rows = np.zeros(len(obs), bool)
        for ep, g in data.groupby("episode_index"):
            states, actions = np.stack(g["observation.state"]), np.stack(g["action"])
            start, n = align(obs, states, names)
            item = {"episode": int(ep), "frames": len(g), "start": None if start is None else int(start), "matches": n}
            if start is not None:
                pos = int(obs.index[start])
                rows = df.iloc[pos:pos + len(g) + 1]  # + the next row (kind 1 if the session ended there)
                jn = [x.removesuffix(".pos") for x in names]
                des = rows[[f"{j}.desired" for j in jn]].to_numpy()[1:].astype(np.float32)
                item["contiguous"] = bool((rows["tick"].diff().iloc[1:] == 1).all() and (rows["kind"].iloc[:len(g)] == 0).all())
                item["action_eq_desired"] = bool(len(des) == len(actions) and np.array_equal(des, actions.astype(np.float32)))
                ep_rows[start:start + len(g)] = True
            ep_info.append(item)
        loop = ep_rows
    else:
        loop = loop_mask(obs["t"].to_numpy())
        loop[: int(np.argmax(obs[[f"{j}.cmd_n" for j in JOINTS]].fillna(0).to_numpy().sum(1) > 0)) + 1] = False
    L = obs[loop]

    # G5 command and refresh frames visible (loopback)
    if len(L) and all(tap_rows[s] == 1.0 for s in sides):
        cmd = L[[f"{j}.cmd_n" for j in JOINTS]].to_numpy()
        fresh_known = L[[f"{j}.state_fresh" for j in JOINTS]].notna().to_numpy()
        nxt = obs.iloc[np.flatnonzero(loop)[np.flatnonzero(loop) + 1 < len(obs)] + 1]
        gap = max(float(np.nanmax(np.abs(nxt[f"{j}.cmd_q"] - nxt[f"{j}.sent"]).to_numpy(), initial=0)) for j in JOINTS)
        ok = (cmd == 1).mean() >= CMD_VISIBLE_MIN and fresh_known.mean() >= REFRESH_VISIBLE_MIN and gap <= Q_LSB * 1.0001
        R.check("G5", "gate", "the library's MIT and refresh frames are seen via loopback: one MIT frame per joint per tick, "
                "decoded command = host's sent command", ok,
                f"cmd_n==1 on {(cmd == 1).mean():.4f} of {cmd.size} joint-rows (cmd_n values {sorted(set(np.unique(cmd).tolist()))}), "
                f"refresh seen {fresh_known.mean():.4f}, max |cmd_q - sent| {gap / Q_LSB:.2f} LSB")
    else:
        R.check("G5", "gate", "the library's MIT and refresh frames are seen via loopback", None, "no loop rows or a tap closed")

    # G6 transmissions accounted for: kernel tx_packets delta == frames the library transmitted, as seen by the tap
    iface = meta.get("iface", {})
    det, res = {}, {}
    for s in sides:
        st = [((iface.get(k) or {}).get(s) or {}).get("statistics") for k in ("start", "end")]
        if frames is None or not all(st) or drop[s] != 0:
            res[s], det[s] = None, "unavailable (no counters / no frames / frames lost)"
            continue
        f = frames[frames.side == s]
        tx_seen = int(f["can_id"].isin(send).sum() + (f["can_id"] == refresh_id).sum())
        delta = st[1]["tx_packets"] - st[0]["tx_packets"]
        res[s], det[s] = delta == tx_seen, f"tx_packets delta {delta}, library TX frames seen {tx_seen}"
    ok = False if False in res.values() else (None if None in res.values() else True)
    R.check("G6", "gate", "telemetry adds no CAN transmission: every frame the interface sent is a library frame the tap saw",
            ok, "; ".join(f"{s}: {v}" for s, v in det.items()))

    # G7 control frames
    if frames is not None:
        is_ctrl = frames["can_id"].isin(send) & frames["data"].map(lambda b: len(b) == 8 and b[:7] == b"\xff" * 7)
        kinds = frames.loc[is_ctrl, "data"].map(lambda b: {0xFC: "enable", 0xFD: "disable", 0xFE: "set_zero"}.get(b[7], hex(b[7])))
        final_tick = int(fin["tick"].iloc[0]) if len(fin) else -1
        in_obs = kinds[frames.loc[is_ctrl, "tick"] != final_tick]
        R.check("G7", "gate", "no enable / disable / set-zero frame while the robot was being read (kind-0 rows)",
                len(in_obs) == 0 and "set_zero" not in set(kinds),
                f"during rows: {in_obs.value_counts().to_dict()}, whole session: {kinds.value_counts().to_dict()}")
    else:
        R.check("G7", "gate", "no enable / disable / set-zero frame while the robot was being read", None, "no frames log")

    # G8 overhead
    snap = obs["snapshot_s"].to_numpy()
    p99 = float(np.percentile(snap, 99)) if len(snap) else None
    lmax = float(L["snapshot_s"].max()) if len(L) else None
    R.check("G8", "gate", f"snapshot cost: p99 <= {SNAP_P99_MAX_S * 1e3:g} ms over all rows, max <= {SNAP_LOOP_MAX_S * 1e3:g} ms in the loop",
            None if p99 is None or lmax is None else (p99 <= SNAP_P99_MAX_S and lmax <= SNAP_LOOP_MAX_S),
            f"median {np.median(snap) * 1e3:.3f} ms, p99 {p99 * 1e3:.3f} ms, loop max {lmax * 1e3:.3f} ms, "
            f"session max {snap.max() * 1e3:.2f} ms" if p99 is not None and lmax is not None else "")

    # G9 loop timing (telemetry on), and A/B against a telemetry-off run
    period = 1.0 / fps
    segs = [obs["t"].to_numpy()[np.flatnonzero(loop)][i] for i in _contiguous(np.flatnonzero(loop))]
    ps, late = period_stats(segs), late_frac(segs, period)
    R.check("S5", "stack", f"control loop at {fps} Hz with telemetry on: median within {PERIOD_TOL_S * 1e3:g} ms, "
            f"p99 <= period + {PERIOD_P99_EXTRA_S * 1e3:g} ms, late (> {LATE_FACTOR} periods) <= {LATE_MAX:.0%}",
            None if ps is None else (abs(ps["median"] - period) <= PERIOD_TOL_S and ps["p99"] <= period + PERIOD_P99_EXTRA_S
                                     and late <= LATE_MAX),
            "" if ps is None else f"n={ps['n']}, median {ps['median'] * 1e3:.2f} ms, p99 {ps['p99'] * 1e3:.2f} ms, "
                                  f"max {ps['max'] * 1e3:.1f} ms, late {late:.3%}")
    if diag and baseline_diag:
        offs = [baseline_diag] if isinstance(baseline_diag, (str, Path)) else list(baseline_diag)
        on = _segments(load_diag_obs_t(diag))
        off = [_segments(load_diag_obs_t(p)) for p in offs]
        res = [_ab(on, o, period) for o in off]
        ok = None if any(r[0] is None for r in res) else all(r[0] for r in res)
        detail = " | ".join(f"on vs off#{i + 1}: {d}" for i, (_, d) in enumerate(res))
        if len(off) > 1:  # do the telemetry-off runs agree with each other? (symmetric check)
            aa = [_ab(off[0], o, period) for o in off[1:]] + [_ab(o, off[0], period) for o in off[1:]]
            if not all(r[0] for r in aa):
                ok = None
                detail += " || INCONCLUSIVE: the telemetry-off runs differ from each other by more than the thresholds"
        R.check("G10", "gate", "telemetry on vs off (teleop DIAG obs period): median, p99 and late fraction not worse",
                ok, detail)
    else:
        R.check("G10", "gate", "telemetry on vs off timing comparison", None, "pass --diag and --baseline-diag")

    # G11 dataset format and alignment
    if dataset is not None:
        feats = info["features"]
        shape_ok = all(feats[k]["dtype"] == "float32" and list(feats[k]["shape"]) == [16] and feats[k]["names"] == DATASET_NAMES
                       for k in ("observation.state", "action"))
        extra = sorted(k for k in feats if k not in DATASET_KEYS and not k.startswith("observation.images."))
        same = True
        if baseline_dataset is not None:
            base = json.loads((Path(baseline_dataset) / "meta/info.json").read_text())
            same = base["features"] == feats and base.get("robot_type") == info.get("robot_type")
        R.check("G11", "gate", "dataset keeps the 16-D float32 state/action in the interleaved R/L order, no extra features"
                + ("; features identical to the telemetry-off run's" if baseline_dataset is not None else ""),
                shape_ok and not extra and same,
                f"robot_type={info.get('robot_type')}, extra features={extra}, identical to baseline={same if baseline_dataset is not None else 'n/a'}")
        ok = bool(ep_info) and all(e["start"] is not None and e.get("contiguous") and e.get("action_eq_desired") for e in ep_info)
        R.check("G12", "gate", "every episode aligns to exactly one run of consecutive telemetry rows; action[k] == desired on row k+1",
                ok, "; ".join(f"ep {e['episode']}: {e['frames']} frames, start {e['start']}, matches {e['matches']}, "
                              f"contiguous {e.get('contiguous')}, action==desired {e.get('action_eq_desired')}" for e in ep_info))
    else:
        R.check("G11", "gate", "dataset format", None, "pass --dataset")
        R.check("G12", "gate", "dataset alignment", None, "pass --dataset")

    # G13 refresh -> reply latency and delivery order (timestamp accuracy)
    final_tick = int(fin["tick"].iloc[0]) if len(fin) else -1
    lat = _latency(frames[frames["tick"] != final_tick], send, recv, refresh_id) if frames is not None else None
    if lat:
        ok = all(v["median"] is not None and 0 < v["median"] <= LAT_MEDIAN_MAX_S and v["p99"] <= LAT_P99_MAX_S
                 and v["reordered"] <= REORDER_MAX for v in lat.values())
        worst = max(lat.items(), key=lambda kv: kv[1]["p99"])
        medians = ", ".join("%s %.2f" % (j, v["median"] * 1e3) for j, v in lat.items() if v["median"] is not None)
        reordered = max(v["reordered"] for v in lat.values())
        R.check("G13", "gate", f"refresh request -> its reply on the wire: 0 < median <= {LAT_MEDIAN_MAX_S * 1e3:g} ms, "
                f"p99 <= {LAT_P99_MAX_S * 1e3:g} ms, reply delivered before the request's echo <= {REORDER_MAX:.1%}", ok,
                f"medians (ms) {medians}; worst p99 {worst[0]} {worst[1]['p99'] * 1e3:.2f} ms; max reordered {reordered:.3%}")
        lost = {j: v["lost"] for j, v in lat.items()}
        wl = max(lost.items(), key=lambda kv: kv[1])
        R.check("S4", "stack", f"every motor answered its commands and refreshes: lost replies <= {LOST_REPLY_MAX:.0%} per joint",
                all(v <= LOST_REPLY_MAX for v in lost.values()),
                f"worst {wl[0]} {wl[1]:.3%}; " + ", ".join(f"{j} {v:.2%}" for j, v in lost.items() if v > 0))
    else:
        R.check("G13", "gate", "refresh request -> reply latency", None, "no refresh frames in the frames log")
        R.check("S4", "stack", "motor replies", None, "no refresh frames in the frames log")

    # B1 bus health
    errs = {s: float(np.nansum(df[f"{s}.bus_errors"])) for s in sides}
    states, deltas = {}, {}
    for s in sides:
        st = [((iface.get(k) or {}).get(s) or {}) for k in ("start", "end")]
        states[s] = [(x.get("can") or {}).get("state") for x in st]
        a, b = (x.get("statistics") or {} for x in st)
        deltas[s] = {k: b[k] - a[k] for k in ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped", "rx_over_errors")
                     if k in a and k in b}
    known = all(deltas[s] for s in sides)
    ok = all(v == 0 for v in errs.values()) and all(all(x in (None, "ERROR-ACTIVE") for x in states[s]) for s in sides) \
        and all(v == 0 for s in sides for v in deltas[s].values())
    R.check("B1", "gate", "CAN bus healthy: no error frames, interface error counters unchanged, controller ERROR-ACTIVE",
            ok if known else (False if not ok else None), f"error frames {errs}, state {states}, counter deltas {deltas}")

    # S1 freshness of the values the policy saw (a finding about the follower, not about telemetry)
    if len(L) and all(tap_rows[s] == 1.0 for s in sides):
        per = {}
        for j in JOINTS:
            fr, stl, age = L[f"{j}.state_fresh"], L[f"{j}.stale"], L[f"{j}.state_age_s"]
            per[j] = (float(fr.mean()) if fr.notna().any() else float("nan"), float(stl.mean()),
                      float(np.nanpercentile(age, 99)) if age.notna().any() else float("nan"))
        ok = all(f >= FRESH_MIN and s <= STALE_MAX and a <= AGE_P99_MAX_S for f, s, a in per.values())
        worst = min(per.items(), key=lambda kv: (kv[1][0] if kv[1][0] == kv[1][0] else -1))
        R.check("S1", "stack", f"values the policy saw are this tick's: state_fresh >= {FRESH_MIN:.0%}, stale <= {STALE_MAX:.0%}, "
                f"state_age_s p99 <= {AGE_P99_MAX_S * 1e3:g} ms, per joint", ok,
                f"worst {worst[0]}: fresh {worst[1][0]:.3f}, stale {worst[1][1]:.3f}, age p99 {worst[1][2] * 1e3:.2f} ms; "
                f"all: {json.dumps({j: [round(x, 4) for x in v] for j, v in per.items()})}")
    else:
        R.check("S1", "stack", "value freshness", None, "no loop rows or a tap closed")

    # S2 motor status and wrapper guards during the loop
    if len(L):
        status = L[[f"{j}.status" for j in JOINTS]].to_numpy()
        R.check("S2", "stack", "every motor reports status 1 (enabled, no fault); no motor_fault; hold rows counted",
                bool((status == 1).all() and (L["motor_fault"] == 0).all()),
                f"status values {sorted(set(np.unique(status[~np.isnan(status)]).astype(int).tolist()))}, "
                f"motor_fault rows {int((L['motor_fault'] == 1).sum())}, hold rows {int((L['hold'] == 1).sum())}")
    else:
        R.check("S2", "stack", "motor status", None, "no loop rows")

    # S3 shutdown: the final row holds the disable frames, 8 per arm
    if len(fin):
        n = {s: float(fin[f"{s}.ctrl_frames"].iloc[0]) for s in sides}
        R.check("S3", "stack", "shutdown sent the disable frames (8 per arm) after the last observation; nothing else",
                all(v == 8 for v in n.values()), f"final-row ctrl_frames {n}, logged {len(meta.get('ctrl_frames', []))}")
    else:
        R.check("S3", "stack", "shutdown disable frames", None, "no final row")

    return R, {"episodes": ep_info, "latency": lat, "meta_taps": taps}


def _contiguous(idx):
    """Index groups of `idx` that are consecutive integers."""
    if not len(idx):
        return []
    cut = np.flatnonzero(np.diff(idx) != 1) + 1
    return [np.arange(a, b) for a, b in zip(np.r_[0, cut], np.r_[cut, len(idx)])]


def _latency(frames, send, recv, refresh_id):
    """Per joint, in delivery order (seq): a refresh is ANSWERED when the motor's next state frame arrives before
    the next frame sent to it (refresh or MIT). Latency = that reply's stamp - the request echo's stamp.
    `lost` = 1 - replies / requests (every MIT command and refresh gets exactly one reply); `reordered` = refreshes
    unanswered in order beyond what lost replies explain, i.e. replies delivered before their request's echo."""
    out = {}
    is_ref = (frames["can_id"] == refresh_id) & frames["data"].map(lambda b: len(b) >= 3 and b[2] == 0xCC)
    if not is_ref.any():
        return None
    target = frames["data"].map(lambda b: b[0] | (b[1] << 8) if len(b) >= 2 else -1).where(is_ref, -1)
    is_mit = frames["data"].map(lambda b: len(b) == 8 and b[:7] != b"\xff" * 7)
    for s in ("L", "R"):
        for i, (sid, rid) in enumerate(zip(send, recv), start=1):
            side = frames.side == s
            req = frames[side & ((is_ref & (target == sid)) | ((frames.can_id == sid) & is_mit))]
            ref = frames[side & is_ref & (target == sid)]
            rp = frames[side & (frames.can_id == rid)]
            if not len(ref) or not len(rp):
                out[f"{s}J{i}"] = {"n": int(len(ref)), "median": None, "p99": float("inf"), "lost": 1.0, "reordered": 1.0}
                continue
            rp_seq, rp_t = rp["seq"].to_numpy(), rp["t_ns"].to_numpy(dtype=float)
            rq_seq = req["seq"].to_numpy()
            r_seq, r_t = ref["seq"].to_numpy(), ref["t_ns"].to_numpy(dtype=float)
            k = np.searchsorted(rp_seq, r_seq)
            n = np.searchsorted(rq_seq, r_seq, side="right")
            nxt_req = np.where(n < len(rq_seq), rq_seq[np.minimum(n, len(rq_seq) - 1)], np.iinfo(np.int64).max)
            answered = (k < len(rp_seq)) & (rp_seq[np.minimum(k, len(rp_seq) - 1)] < nxt_req)
            lat = (rp_t[k[answered]] - r_t[answered]) / 1e9
            lost = max(0, len(req) - len(rp))
            out[f"{s}J{i}"] = {"n": int(len(ref)), "median": float(np.median(lat)) if len(lat) else None,
                               "p99": float(np.percentile(lat, 99)) if len(lat) else float("inf"),
                               "lost": lost / len(req),
                               "reordered": max(0, int((~answered).sum()) - lost) / len(ref)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--dataset", help="local LeRobot dataset root recorded in the same session")
    ap.add_argument("--diag", help="teleop DIAG folder of this (telemetry-on) run")
    ap.add_argument("--baseline-diag", action="append", help="teleop DIAG folder of a telemetry-off run (repeat for A-B-A)")
    ap.add_argument("--baseline-dataset", help="local dataset root of the telemetry-off run (features must be identical)")
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args(argv)
    R, extra = run(a.session, a.dataset, a.diag, a.baseline_diag, a.baseline_dataset)
    for i in R.items:
        print(f"{i['status']:<8} {i['id']:<4} [{i['group']}] {i['check']}\n         {i['detail']}")
    verdict = R.verdict()
    print(f"\nGATE-2 TELEMETRY VERDICT: {verdict}  (stack findings: "
          f"{', '.join(i['id'] + ' ' + i['status'] for i in R.items if i['group'] == 'stack')})")
    if a.json:
        Path(a.json).write_text(json.dumps({"verdict": verdict, "checks": R.items, **extra}, indent=1, default=str))
    return {"GO": 0, "NO-GO": 1, "INCOMPLETE": 3}[verdict]


if __name__ == "__main__":
    sys.exit(main())
