"""NEXT adapter (tools/force_aware/next_export.py) on the committed Gate-2A mock fixture and corrupted copies of it.
Files only: no robot, no CAN socket, no GPU.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_next_export.py <scratch dir>

Checks the contract in docs/force_aware/NEXT_DATA_CONTRACT.md: values exported verbatim and joined by name, the
fixture's scripted faults rejected exactly where expected, no window across a break, causality (nothing from a
later row changes an earlier row's export), grippers kept apart, no interpolation of stale values, explicit
contact-free attestation, hardware-signature checks, session-level splits, and the source files left untouched.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import lerobot_robot_openarm_umeow  # noqa: F401  (adds the repo root to sys.path)
from lerobot_robot_openarm_umeow import telemetry as tel

REPO = Path(__file__).resolve().parents[2]
FX = Path(__file__).resolve().parent / "fixtures/openarm_telemetry_1"
sys.path.insert(0, str(REPO / "tools/force_aware"))
import next_export as nx  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/next_export_test")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
FAILS = []
ATT = {"attested_by": "test (mock fixture)", "block": "A1"}


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def raises(fn, *exc):
    try:
        fn()
    except exc or Exception as e:  # noqa: B030
        return str(e) or type(e).__name__
    return None


def clone(name, mutate=None, meta_mutate=None, shuffle_columns=False):
    """A copy of the fixture session as ONE part file, optionally modified (a scratch copy; the fixture is
    never written)."""
    df, meta = tel.load(FX / "session")
    if mutate:
        mutate(df)
    if meta_mutate:
        meta_mutate(meta)
    d = OUT / "sessions" / name
    d.mkdir(parents=True)
    if shuffle_columns:
        cols = list(df.columns)
        np.random.default_rng(0).shuffle(cols)
        df = df[cols]
    df.to_parquet(d / "part-00001.parquet", index=False)
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def tables(m, out_root):
    d = Path(out_root) / m["source"]["session_id"]
    return {k: pd.read_parquet(d / f"{k}.parquet") for k in
            ("left_arm", "right_arm", "left_gripper", "right_gripper")}


def bad_ticks(tab):
    return tab.loc[~tab["valid"], "tick"].tolist()


exp = json.loads((FX / "expected.json").read_text())
src_hash = {p.name: nx.sha256(p) for p in sorted((FX / "session").iterdir())}
df, meta = tel.load(FX / "session")

# -- export the fixture itself ----------------------------------------------------------------------------------
m = nx.export_session(FX / "session", OUT / "exp", contact_free_all=True, attestation=ATT)
T = tables(m, OUT / "exp")
check("N1 source files unchanged by the export (sha256 of every file), and recorded in the manifest",
      {p.name: nx.sha256(p) for p in sorted((FX / "session").iterdir())} == src_hash
      and all(m["source"]["sha256"][k] == src_hash[k] for k in m["source"]["sha256"]))
check("N1 refuses to write inside the source session, and never overwrites an export",
      raises(lambda: nx.export_session(FX / "session", FX, contact_free_all=True, attestation=ATT)) is not None
      and raises(lambda: nx.export_session(FX / "session", OUT / "exp", contact_free_all=True, attestation=ATT))
      is not None)

js = {"left": [f"LJ{i}" for i in range(1, 8)], "right": [f"RJ{i}" for i in range(1, 8)]}
check("N2 arm tables hold exactly J1..J7 of their side (no J8, no other side), in J1..J7 order",
      all([c.split(".")[0] for c in T[f"{a}_arm"].columns if c.endswith(".q")] == js[a] for a in js))
check("N2 gripper tables hold J8 of their side only, plus the squeeze feed-forward",
      all([c.split(".")[0] for c in T[f"{a}_gripper"].columns if c.endswith(".q")] == [f"{a[0].upper()}J8"]
          and "squeeze_tau" in T[f"{a}_gripper"] for a in js))
same = True
for a in js:
    for j in js[a] + [f"{a[0].upper()}J8"]:
        tab = T[f"{a}_arm"] if not j.endswith("8") else T[f"{a}_gripper"]
        for f in ("q", "dq", "tau", "cmd_q", "cmd_tau", "state_age_s", "stale", "sent", "desired"):
            same &= np.array_equal(tab[f"{j}.{f}"].to_numpy(), df[f"{j}.{f}"].to_numpy(), equal_nan=True)
check("N3 every exported value equals the source telemetry bit-for-bit (float64, NaN kept as NaN)", same)
check("N3 one export row per source row, ticks identical", all(np.array_equal(t["tick"], df["tick"]) for t in T.values()))

# The fixture's scripted events (expected.json + CODEX_GATE2_HANDOFF section 4).
L, R = T["left_arm"], T["right_arm"]
fail_tick = exp["failed_snapshot_ticks"][0]
holds = exp["hold_ticks"]
stale_r = exp["stale_joint_rows"]["RJ5"]
check("N4 failed snapshot row (tick 6) invalid on both arms and both grippers",
      all(fail_tick in bad_ticks(t) for t in T.values()))
check("N4 final row (kind 1) invalid everywhere", all(exp["final_tick"] in bad_ticks(t) for t in T.values()))
check("N4 hold rows 90-95 invalid on both arms (the command is a frozen re-send)",
      all(set(holds) <= set(bad_ticks(t)) for t in (L, R)))
rj5 = R.set_index("tick")
check("N4 RJ5 stale row (tick 81) invalid on the RIGHT arm only; the left arm keeps it",
      set(stale_r) <= set(bad_ticks(R)) and not set(stale_r) & set(bad_ticks(L))
      and "stale" in nx.describe_reject(int(rj5.loc[stale_r[0], "reject"])))
check("N4 tick 80 (RJ5 value answers the previous command's reply, state_fresh 0) invalid on the right arm",
      80 in bad_ticks(R) and nx.describe_reject(int(rj5.loc[80, "reject"])) == ["not_fresh"])
check("N4 ramp / double-command rows (cmd_n != 1) and the row before any command are invalid",
      {0, 1, 7} <= set(bad_ticks(L)) and "cmd_count" in nx.describe_reject(int(L.set_index("tick").loc[1, "reject"])))
check("N4 first row after the save pause (stale command, cmd_age > 1.5 periods) invalid",
      nx.describe_reject(int(L.set_index("tick").loc[58, "reject"])) == ["cmd_age"])
check("N4 no other rejections on the left arm than the scripted ones",
      set(bad_ticks(L)) == {0, 1, fail_tick, 7, 58, *holds, 115, exp["final_tick"]}, f"{bad_ticks(L)}")

# Segments: consecutive ticks, spacing within the band, only valid rows.
ok = True
for t in T.values():
    v = t[t["segment"] >= 0]
    ok &= bool(v["valid"].all()) and bool((t.loc[~t["valid"], "segment"] == -1).all())
    for _, g in v.groupby("segment"):
        dt = np.diff(g["t"].to_numpy())
        ok &= bool((np.diff(g["tick"].to_numpy()) == 1).all() and ((dt >= 0.5 / 30) & (dt <= 1.5 / 30)).all())
check("N5 segments contain only valid rows, consecutive ticks, spacing within 0.5-1.5 periods", ok)
check("N5 fixture segments: left [2-5, 8-57, 59-89, 96-114], right additionally split at RJ5 80-81",
      m["arms"]["left"]["segment_lengths"] == [4, 50, 31, 19] and m["arms"]["right"]["segment_lengths"] == [4, 50, 21, 8, 19],
      f"{m['arms']['left']['segment_lengths']} {m['arms']['right']['segment_lengths']}")

# Causality: changing everything AFTER row k must not change rows <= k (except where the next row's spacing
# decides a segment boundary, which only ever looks back).
K = 40


def scramble_after(d):
    num = [c for c in d.columns if c not in ("tick", "kind", "t", "t_wall", "obs_t0")]
    d.loc[d["tick"] > K, num] = np.random.default_rng(1).normal(size=(int((d["tick"] > K).sum()), len(num)))


mc = nx.export_session(clone("causal", scramble_after), OUT / "exp_causal", contact_free_all=True, attestation=ATT)
Tc = tables(mc, OUT / "exp_causal")
# (The one session-level rule is the reference gain pair -- the most common transmitted (kp, kd) -- which a few
# scrambled rows cannot move; it only ever rejects rows, it never changes a value.)
check("N6 causal: scrambling every row after k leaves the export of rows <= k identical (values, validity, "
      "segments)", all(T[k][T[k]["tick"] <= K].equals(Tc[k][Tc[k]["tick"] <= K]) for k in T))
loop = L[L["valid"]]
lsb = 25.0 / 65535
check("N6 row k's cmd_q is the transmitted form of row k's `sent` (within 1 LSB): the command in effect "
      "before row k's measurement, not action k",
      all((np.abs(loop[f"{j}.cmd_q"] - loop[f"{j}.sent"]) <= lsb + 1e-12).all() for j in js["left"]))
check("N6 on valid rows the measurement was acquired after the command in effect was sent",
      bool((loop[[f"{j}.state_age_s" for j in js["left"]]].to_numpy()
            <= loop[[f"{j}.cmd_age_s" for j in js["left"]]].to_numpy()).all()))

G = T["right_gripper"]
check("N7 right gripper squeeze rows (squeeze_tau -1.5) stay valid gripper rows; arm validity is independent",
      G.loc[G["squeeze_tau"] == -1.5, "valid"].sum() >= 10 and set(bad_ticks(G)) == set(bad_ticks(T["left_gripper"])))

# Column order irrelevant (join by name).
ms = nx.export_session(clone("shuffled", shuffle_columns=True), OUT / "exp_shuf", contact_free_all=True, attestation=ATT)
Ts = tables(ms, OUT / "exp_shuf")
check("N2 shuffled source column order -> identical export (joined by name, never by index)",
      all(T[k].equals(Ts[k]) for k in T))

# -- corrupted copies -------------------------------------------------------------------------------------------
cases = {
    "nan_tau": (lambda d: d.__setitem__("LJ3.tau", d["LJ3.tau"].where(d["tick"] != 30)), "left", 30, "nan"),
    "stale_nan": (lambda d: d.__setitem__("LJ2.stale", d["LJ2.stale"].where(d["tick"] != 31)), "left", 31, "stale"),
    "fresh_nan": (lambda d: d.__setitem__("RJ2.state_fresh", d["RJ2.state_fresh"].where(d["tick"] != 32)), "right", 32, "not_fresh"),
    "tap_down": (lambda d: d.__setitem__("R.tap_ok", d["R.tap_ok"].where(d["tick"] != 33, 0.0)), "right", 33, "tap_down"),
    "saturated": (lambda d: d.__setitem__("LJ5.tau", d["LJ5.tau"].where(d["tick"] != 34, 10.0)), "left", 34, "saturated"),
    "gain_change": (lambda d: d.__setitem__("RJ1.cmd_kp", d["RJ1.cmd_kp"].where(d["tick"] != 35, 60.0)), "right", 35, "gain_change"),
    "status": (lambda d: d.__setitem__("LJ7.status", d["LJ7.status"].where(d["tick"] != 36, 0xA)), "left", 36, "status"),
    "bus": (lambda d: d.__setitem__("L.rx_dropped", d["L.rx_dropped"].where(d["tick"] < 37, 3.0)), "left", 37, "bus"),
    "ctrl": (lambda d: d.__setitem__("R.ctrl_frames", d["R.ctrl_frames"].where(d["tick"] != 38, 1.0)), "right", 38, "ctrl_frame"),
    "fault": (lambda d: d.__setitem__("motor_fault", d["motor_fault"].where(d["tick"] != 39, 1.0)), "left", 39, "motor_fault"),
}
for name, (mut, arm, tick, reason) in cases.items():
    mm = nx.export_session(clone(name, mut), OUT / f"exp_{name}", contact_free_all=True, attestation=ATT)
    tab = tables(mm, OUT / f"exp_{name}")[f"{arm}_arm"].set_index("tick")
    other = tables(mm, OUT / f"exp_{name}")[f"{'right' if arm == 'left' else 'left'}_arm"].set_index("tick")
    r = nx.describe_reject(int(tab.loc[tick, "reject"]))
    side_wide = reason in ("motor_fault",)
    check(f"N8 {name}: tick {tick} rejected on the {arm} arm for '{reason}'"
          + ("" if side_wide else ", the other arm unaffected"),
          reason in r and (side_wide or bool(other.loc[tick, "valid"])), f"{r}")
mm = tables(nx.export_session(clone("bus_after", cases["bus"][0]), OUT / "exp_bus_after", contact_free_all=True,
                              attestation=ATT), OUT / "exp_bus_after")["left_arm"].set_index("tick")
check("N8 rx_dropped is cumulative: only the row where it GREW is rejected, not every later row",
      not mm.loc[37, "valid"] and mm.loc[38, "valid"])


def repeat_stale(d):  # a motor stopped answering: q/dq/tau repeat, state_age grows, stale = 1
    rows = (d["tick"] >= 40) & (d["tick"] <= 44)
    for f in ("q", "dq", "tau"):
        d.loc[rows, f"LJ4.{f}"] = float(d.loc[d["tick"] == 39, f"LJ4.{f}"].iloc[0])
    d.loc[rows, "LJ4.state_age_s"] = 0.0338 * np.arange(1, 6)
    d.loc[rows, "LJ4.stale"] = (0.0338 * np.arange(1, 6) > 0.05).astype(float)
    d.loc[rows, "LJ4.state_fresh"] = 0.0


mm = nx.export_session(clone("repeat", repeat_stale), OUT / "exp_repeat", contact_free_all=True, attestation=ATT)
tl = tables(mm, OUT / "exp_repeat")["left_arm"]
check("N9 repeated (stale) measurements are rejected, not interpolated: ticks 40-44 invalid, values verbatim",
      set(range(40, 45)) <= set(bad_ticks(tl))
      and np.array_equal(tl.set_index("tick").loc[40:44, "LJ4.q"].to_numpy(),
                         np.full(5, df.loc[df["tick"] == 39, "LJ4.q"].iloc[0])))


def jitter(d):  # one late tick (a 70 ms period) and one duplicated-time row
    d.loc[d["tick"] >= 45, "t"] += 0.036
    d.loc[d["tick"] == 50, "t"] = d.loc[d["tick"] == 49, "t"].iloc[0]


mm = nx.export_session(clone("jitter", jitter), OUT / "exp_jitter", contact_free_all=True, attestation=ATT)
tl = tables(mm, OUT / "exp_jitter")["left_arm"].set_index("tick")
check("N10 a late tick (period 2.1x) starts a new segment; a non-increasing clock is rejected",
      tl.loc[45, "segment"] != tl.loc[44, "segment"] and tl.loc[44, "valid"] and tl.loc[45, "valid"]
      and "clock" in nx.describe_reject(int(tl.loc[50, "reject"])))
mm = nx.export_session(clone("gap", lambda d: d.drop(d.index[d["tick"] == 20], inplace=True)), OUT / "exp_gap",
                       contact_free_all=True, attestation=ATT)
tl = tables(mm, OUT / "exp_gap")["left_arm"].set_index("tick")
check("N10 a missing tick (lost write) ends the segment even though the spacing would pass",
      tl.loc[21, "segment"] != tl.loc[19, "segment"])



def extra_read(d):
    """8e43f49 async eval when the action outlasts a tick (test_eval_obs_reuse E2): row A (recorder) -> action ->
    row B, read again a few ms later (cmd_n 1, the action just sent) -> next tick's row C sees no new command
    (cmd_n 0). Ticks after A shift by one."""
    a = d.index[d["tick"] == 30][0]
    b = d.loc[[a]].copy()
    b["t"] += 0.005
    b["t_wall"] += 0.005
    d.loc[d["tick"] > 30, "tick"] += 1
    b["tick"] = 31
    d.loc[d["tick"] == 32, [f"{j}.cmd_n" for j in tel.JOINTS]] = 0.0
    d.loc[len(d)] = b.iloc[0]
    d.sort_values("tick", inplace=True)
    d.reset_index(drop=True, inplace=True)


mm = nx.export_session(clone("extra_read", extra_read), OUT / "exp_extra", contact_free_all=True, attestation=ATT)
tl = tables(mm, OUT / "exp_extra")["left_arm"].set_index("tick")
check("N15 an extra observation read inside one tick (8e43f49 E2) is split off (spacing < 0.5 period), and the "
      "next tick, which saw no new command, is rejected: no window spans the irregular rows",
      tl.loc[31, "segment"] != tl.loc[30, "segment"] and not tl.loc[32, "valid"]
      and "cmd_count" in nx.describe_reject(int(tl.loc[32, "reject"])) and tl.loc[33, "segment"] != tl.loc[31, "segment"])

# -- attestation, signatures, schema ------------------------------------------------------------------------------
mn = nx.export_session(clone("unattested"), OUT / "exp_unatt")
check("N11 without an attestation nothing is contact-free, and the FACTR2 H5 view refuses it",
      mn["arms"]["left"]["contact_free_valid"] == 0
      and raises(lambda: nx.write_factr2_h5([OUT / "exp_unatt/unattested"], "left", OUT / "x.h5")) is not None)
check("N11 an attestation needs attested_by",
      raises(lambda: nx.export_session(clone("noname"), OUT / "exp_noname", contact_free_all=True)) is not None)
iv = [{"t_wall_start": float(df.loc[df["tick"] == 20, "t_wall"].iloc[0]),
       "t_wall_end": float(df.loc[df["tick"] == 40, "t_wall"].iloc[0]), "block": "A1"}]
mi = nx.export_session(clone("intervals"), OUT / "exp_iv", intervals=iv, attestation=ATT)
ti = tables(mi, OUT / "exp_iv")["left_arm"]
check("N11 interval attestation marks exactly the rows inside [start, end] (ticks 20-40)",
      ti.loc[ti["contact_free"], "tick"].tolist() == list(range(20, 41)))
ivs = nx.episode_intervals(df, FX / "dataset", block="A3")
check("N16 dataset episodes located in the session by value: first ticks [13, 70], 45 frames each",
      [iv["first_tick"] for iv in ivs] == [e["start_tick"] for e in exp["episodes"]] and {iv["frames"] for iv in ivs} == {45})
rc = nx.main([str(clone("episodes")), "--out", str(OUT / "exp_eps"), "--contact-free-episodes", str(FX / "dataset"),
              "--attested-by", "test: dataset video reviewed", "--block", "A3"])
te = pd.read_parquet(OUT / "exp_eps/episodes/left_arm.parquet")
check("N16 --contact-free-episodes: exactly the episode rows are contact-free (resets and pauses are not)",
      rc == 0 and te.loc[te["contact_free"], "tick"].tolist() == list(range(13, 58)) + list(range(70, 115))
      and json.loads((OUT / "exp_eps/episodes/manifest.json").read_text())["contact_free"]["attestation"]["block"] == "A3")
check("N12 effort declared uncalibrated (provenance.torque_calibration is null)",
      m["effort"]["calibrated"] is False and m["hardware_signature"]["torque_calibration"] is None)
check("N12 transmitted gains differing from meta.gains are reported, not hidden (mock bus: kp 49.94 everywhere)",
      {g["joint"] for g in m["gains_meta_mismatch"]} == {f"{s}J{i}" for s in "LR" for i in (1, 4, 5, 6, 7, 8)})
other = nx.export_session(clone("other_hw", lambda d: [d.__setitem__(f"LJ{i}.cmd_kp", d[f"LJ{i}.cmd_kp"] * 0 + 60.0)
                                                       for i in range(1, 8)]), OUT / "exp_hw", contact_free_all=True,
                          attestation=ATT)
check("N12 sessions with different transmitted gains cannot share a model",
      raises(lambda: nx.check_compatible([m, other]), nx.ExportError) is not None)
check("N12 a different schema string is refused",
      raises(lambda: nx.load_session(clone("schema2", meta_mutate=lambda mt: mt.__setitem__("schema", "openarm_telemetry/2"))),
             nx.ExportError) is not None)

# -- splits -------------------------------------------------------------------------------------------------------
ids = [f"2026110{i}_120000_{100 + i}" for i in range(1, 7)]
tr, va = nx.split_sessions(ids, 0.34)
check("N13 session split: disjoint, covers every session, deterministic, 34% -> 3 val of 6",
      not set(tr) & set(va) and sorted(tr + va) == sorted(ids) and (tr, va) == nx.split_sessions(list(reversed(ids)), 0.34)
      and len(va) == 3, f"val {va}")
check("N13 explicit validation sessions honoured; a split with an empty side refused",
      nx.split_sessions(ids, explicit_val=[ids[0]])[1] == [ids[0]]
      and raises(lambda: nx.split_sessions(ids[:1], 0.5), nx.ExportError) is not None)

# -- FACTR2 H5 view ---------------------------------------------------------------------------------------------------
import h5py  # noqa: E402

n = nx.write_factr2_h5([OUT / "exp/session"], "right", OUT / "right.h5")
with h5py.File(OUT / "right.h5") as h5:
    eps = sorted(h5.keys())
    lens = [h5[e]["right_joint_pos"]["data"].shape[0] for e in eps]
    shapes = {h5[eps[1]][k]["data"].shape for k in ("right_joint_pos", "right_joint_vel", "right_joint_cmd",
                                                     "right_measured_joint_torque")}
    first = h5[eps[1]]
    r = R.set_index("tick").loc[8:57]
    match = (np.array_equal(first["right_joint_pos"]["data"][:], r[[f"{j}.q" for j in js["right"]]].to_numpy(np.float32))
             and np.array_equal(first["right_joint_cmd"]["data"][:], r[[f"{j}.cmd_q" for j in js["right"]]].to_numpy(np.float32))
             and np.array_equal(first["right_measured_joint_torque"]["data"][:], r[[f"{j}.tau" for j in js["right"]]].to_numpy(np.float32)))
    attrs = h5.attrs["schema"]
check("N14 FACTR2 H5 view: schema factr2_next_h5_v1, one ep per segment, {arm}_* keys, (n, 7) float32",
      attrs == "factr2_next_h5_v1" and n == 5 and lens == [4, 50, 21, 8, 19] and shapes == {(50, 7)}, f"{lens}")
check("N14 H5 values equal the export (float32 cast only)", match)

print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
