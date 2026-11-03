"""NEXT model smoke tests (tools/force_aware/next_model.py) on MOCK data only. CPU, a few seconds of training.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_next_model.py <scratch dir>

Optional parity with the official implementation: point FACTR2_NEXT_SRC at a checkout of
github.com/philiphan0109/factr2_next pinned at 0df02c5 (its src/factr2_next directory). Without it, the parity
checks print SKIP. The official package is imported from that path only (training/dataset.py, training/train.py,
training/models.py, inference/checkpoint.py, inference/history_buffer.py); nothing ROS is imported.

Synthetic sessions are written in the openarm_telemetry/1 format from the committed fixture's meta, with a toy
arm: q tracks the transmitted command through a first-order lag, and the motor effort is a PD + gravity-like +
friction-like function of the history. The numbers mean nothing about OpenArm; they only exercise the pipeline.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import lerobot_robot_openarm_umeow  # noqa: F401
from lerobot_robot_openarm_umeow import telemetry as tel

REPO = Path(__file__).resolve().parents[2]
FX = Path(__file__).resolve().parent / "fixtures/openarm_telemetry_1"
sys.path.insert(0, str(REPO / "tools/force_aware"))
import next_export as nx  # noqa: E402
import next_model as nm  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/next_model_test")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
FAILS = []
ATT = {"attested_by": "test (synthetic mock)", "block": "A1"}
FPS = 30.0
KP, KD = 500.0 * 409 / 4095, 2.0  # the fixture's transmitted gains (49.94, 2.0)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def synth_session(name, seed, n=600, inject=None, kp=KP):
    """One closed telemetry session of n control-loop rows. inject = (joint, first_row, rows, N*m) adds an
    'external' effort to that joint's measured tau only (the model never sees it in training)."""
    _, meta = tel.load(FX / "session")
    rng = np.random.default_rng(seed)
    t = 1000.0 + np.arange(n) / FPS + rng.normal(0, 2e-4, n)
    cols = {c: np.full(n, np.nan) for c in tel.COLUMNS}
    cols.update({"tick": np.arange(n), "kind": np.zeros(n), "t": t, "t_wall": t + 1.79e9, "obs_t0": t - 0.002,
                 "obs_read_s": np.full(n, 0.002), "snapshot_ok": np.ones(n), "snapshot_s": np.full(n, 1e-4),
                 "hold": np.zeros(n), "motor_fault": np.zeros(n)})
    for s in "LR":
        cols.update({f"{s}.tap_ok": np.ones(n), f"{s}.rx_dropped": np.zeros(n), f"{s}.bus_errors": np.zeros(n),
                     f"{s}.ctrl_frames": np.zeros(n), f"{s}.squeeze_tau": np.zeros(n)})
        for i in range(1, 9):
            j = f"{s}J{i}"
            f = rng.uniform(0.1, 0.4, 3)
            ph = rng.uniform(0, 2 * np.pi, 3)
            cmd = sum(0.25 * np.sin(2 * np.pi * f[k] * (t - t[0]) + ph[k]) for k in range(3))
            q = np.zeros(n)
            q[0] = cmd[0]
            for k in range(1, n):
                q[k] = q[k - 1] + 0.55 * (cmd[k] - q[k - 1])  # tracks the command in effect before row k
            dq = np.gradient(q, t) + rng.normal(0, 0.003, n)
            tau = (kp * (cmd - q) - KD * dq * 0.2 + 1.2 * np.sin(q + i) + 0.15 * np.tanh(dq / 0.05)
                   + rng.normal(0, 0.01, n))
            if inject and inject[0] == j:
                tau[inject[1]:inject[1] + inject[2]] += inject[3]
            sent = cmd + rng.uniform(-1, 1, n) * 1e-4
            cols.update({f"{j}.q": q, f"{j}.dq": dq, f"{j}.tau": tau, f"{j}.t_mos": np.full(n, 30.0),
                         f"{j}.t_rotor": np.full(n, 30.0), f"{j}.status": np.ones(n), f"{j}.rx_n": np.full(n, 2.0),
                         f"{j}.rx_age_s": np.full(n, 3e-4), f"{j}.state_age_s": np.full(n, 3e-4),
                         f"{j}.state_fresh": np.ones(n), f"{j}.stale": np.zeros(n), f"{j}.cmd_q": cmd,
                         f"{j}.cmd_dq": np.gradient(cmd, t), f"{j}.cmd_kp": np.full(n, kp),
                         f"{j}.cmd_kd": np.full(n, KD), f"{j}.cmd_tau": 1.2 * np.sin(cmd + i),
                         f"{j}.cmd_n": np.ones(n), f"{j}.cmd_age_s": np.full(n, 0.03), f"{j}.sent": sent,
                         f"{j}.desired": sent})
    df = pd.DataFrame(cols)[tel.COLUMNS]
    df["tick"] = df["tick"].astype("int64")
    d = OUT / "sessions" / name
    d.mkdir(parents=True)
    df.to_parquet(d / "part-00001.parquet", index=False)
    meta.update({"rows": n, "closed": True, "mock_fixture": "synthetic toy arm (test_next_model.py); NOT hardware data"})
    (d / "meta.json").write_text(json.dumps(meta))
    return d


# -- data ----------------------------------------------------------------------------------------------------------
train_ids = [f"20261103_10000{i}_9{i}" for i in range(4)]
val_ids = ["20261103_110000_95"]
contact_id = "20261103_120000_96"
exp = OUT / "exp"
for k, sid in enumerate(train_ids + val_ids):
    nx.export_session(synth_session(sid, k), exp, contact_free_all=True, attestation=ATT)
INJ = ("RJ4", 300, 30, 2.0)
nx.export_session(synth_session(contact_id, 99, inject=INJ), exp)  # NOT attested: contact present
TR = [exp / s for s in train_ids]
VA = [exp / s for s in val_ids]
H = 10

x, y, ends = nm.arm_windows(TR[0], "right", H)
tab, q, dq, cmd, tau = nx.arm_arrays(TR[0], "right")
k = 123
check("M1 window contract: x = [q, dq, cmd - q] (float32) over H rows ending at k; y = tau at k (not k+1)",
      x.shape == (600 - H + 1, H, 21) and y.shape == (600 - H + 1, 7)
      and np.array_equal(x[k - H + 1][-1], np.concatenate([q[k], dq[k], cmd[k].astype(np.float32) - q[k].astype(np.float32)]).astype(np.float32))
      and np.array_equal(y[k - H + 1], tau[k].astype(np.float32)) and ends[0] == H - 1)
check("M1 joint order: feature block i is RJ{i+1} (by name)",
      np.array_equal(x[0][0][:7], tab[[f"RJ{i}.q" for i in range(1, 8)]].to_numpy(np.float32)[0]))
xc, _, _ = nm.arm_windows(exp / contact_id, "right", H)
check("M2 training windows need the contact-free attestation (contact session yields none)", len(xc) == 0)
try:
    nm._check_sets(TR, TR[:1])
    ok_overlap = False
except nx.ExportError:
    ok_overlap = True
try:
    nm._check_sets(TR, [])
    ok_empty = False
except nx.ExportError:
    ok_empty = True
check("M2 train refuses a session on both sides, and an empty validation set", ok_overlap and ok_empty)

# -- training (CPU, tiny) -------------------------------------------------------------------------------------------
import torch  # noqa: E402

torch.set_num_threads(4)
small = {"hidden_size": 32, "head_hidden": 64, "dropout": 0.0}
run_dir, met = nm.train(TR, VA, "right", OUT / "runs", history=H, model_cfg=small,
                        train_cfg={"epochs": 40, "batch_size": 128, "learning_rate": 3e-3})
check("M3 CPU training runs; validation loss falls (normalized MSE)",
      met["val_loss"][-1] < 0.5 * met["val_loss"][0], f"{met['val_loss'][0]:.3f} -> {met['val_loss'][-1]:.3f}")
z = np.load(run_dir / "normalization.npz")
xtr, ytr = nm.collect(TR, "right", H)
xva, _ = nm.collect(VA, "right", H)
check("M3 normalization fitted on the training sessions only",
      np.allclose(z["x_mean"], xtr.mean(axis=(0, 1))) and not np.allclose(z["x_mean"], np.concatenate([xtr, xva]).mean(axis=(0, 1))))
info = json.loads((run_dir / "openarm.json").read_text())
check("M3 run records arm, joints, history, sessions, hardware signature, official commit",
      info["arm"] == "right" and info["joints"] == [f"RJ{i}" for i in range(1, 8)] and info["history"] == H
      and info["train_sessions"] == sorted(train_ids) and info["val_sessions"] == val_ids
      and info["official"]["commit"].startswith("0df02c5") and info["hardware_signature"]["mock"] is True)
check("M3 official run layout: model.pt, config.yaml, normalization.npz, metrics.json",
      all((run_dir / f).is_file() for f in ("model.pt", "config.yaml", "normalization.npz", "metrics.json")))

# -- inference ------------------------------------------------------------------------------------------------------
run = nm.Run(run_dir)
res = nm.estimate(run, exp / contact_id, "right")
ext = res["RJ4.tau_ext_hat"].to_numpy()
inside = np.zeros(len(res), bool)
inside[INJ[1]:INJ[1] + INJ[2]] = True
base = np.nanmedian(np.abs(ext[~inside & res["has_estimate"].to_numpy()]))
rms_in = np.sqrt(np.nanmean(ext[inside] ** 2))
check("M4 residual = measured - predicted, exactly",
      np.allclose(res["RJ4.tau_ext_hat"], res["RJ4.tau"] - res["RJ4.tau_free_hat"], equal_nan=True))
check("M4 an injected +2.0 effort on RJ4 (a session the model never trained on) stands out in the residual",
      abs(np.nanmean(ext[inside]) - 2.0) < 0.5 and rms_in > 5 * base, f"mean inside {np.nanmean(ext[inside]):.2f}, "
      f"|median| outside {base:.3f}")
check("M4 first H-1 rows of a segment have no estimate (no partial windows)",
      not res["has_estimate"].iloc[:H - 1].any() and res["has_estimate"].iloc[H - 1])

# Streaming estimator = batch, and it resets on an invalid row.
se = nm.StreamingEstimator(run, FPS)
tab, q, dq, cmd, tau = nx.arm_arrays(exp / contact_id, "right")
valid = tab["valid"].to_numpy().copy()
valid[200] = False  # an invalid row mid-stream
outs = [se.push(int(tab["tick"][i]), float(tab["t"][i]), bool(valid[i]), q[i], dq[i], cmd[i], tau[i])
        for i in range(260)]
check("M5 streaming estimator matches the batch estimate (same windows)",
      np.allclose(outs[150], res.loc[150, [f"RJ{i}.tau_ext_hat" for i in range(1, 8)]].to_numpy(np.float32), atol=1e-5))
check("M5 streaming: invalid row -> no output for the next H-1 rows, then resumes",
      outs[200] is None and all(o is None for o in outs[201:201 + H - 1]) and outs[201 + H - 1] is not None)
try:
    nm.estimate(run, exp / contact_id, "left")
    other_arm = False
except nx.ExportError:
    other_arm = True
check("M5 estimate refuses a model trained on the other arm", other_arm)

nx.export_session(synth_session("20261103_130000_97", 7, kp=60.0), exp, contact_free_all=True, attestation=ATT)
try:
    nm.estimate(run, exp / "20261103_130000_97", "right")
    sig_ok = False
except nx.ExportError:
    sig_ok = True
check("M6 estimate refuses an export whose transmitted gains differ from the model's training sessions", sig_ok)
fx = OUT / "fixture_exp"
nx.export_session(FX / "session", fx, contact_free_all=True, attestation={"attested_by": "test", "block": "A1"})
r_fx = nm.estimate(run, fx / "session", "right")
check("M6 the committed fixture runs end to end (export -> windows -> model -> residual), estimates only on "
      "rows with H valid predecessors in one segment",
      r_fx["has_estimate"].sum() == sum(max(0, n - H + 1) for n in [4, 50, 21, 8, 19]), f"{int(r_fx['has_estimate'].sum())}")

# -- parity with the official implementation (pinned) ----------------------------------------------------------------
src = os.environ.get("FACTR2_NEXT_SRC")
if not src:
    print("SKIP  P* official parity (set FACTR2_NEXT_SRC to <factr2_next checkout>/src/factr2_next)")
else:
    root = Path(src).resolve()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    check("P0 official checkout is the pinned commit", head == nm.OFFICIAL["commit"], head)
    sys.path.insert(0, str(root))
    from factr2_next.inference.checkpoint import load_checkpoint  # noqa: E402
    from factr2_next.inference.history_buffer import HistoryBuffer  # noqa: E402
    from factr2_next.training import train as off_train  # noqa: E402
    from factr2_next.training.dataset import NextTorqueDataset  # noqa: E402
    from factr2_next.training.models import build_model  # noqa: E402

    h5 = OUT / "right_train.h5"
    nx.write_factr2_h5(TR, "right", h5)
    keys = {"joint_pos": "{arm}_joint_pos", "joint_vel": "{arm}_joint_vel", "joint_cmd": "{arm}_joint_cmd",
            "measured_joint_torque": "{arm}_measured_joint_torque"}
    ds = NextTorqueDataset([str(h5)], keys, H, arm="right")
    check("P1 official NextTorqueDataset on our H5 yields exactly our windows and targets",
          np.array_equal(ds.x, xtr) and np.array_equal(ds.y, ytr), f"{ds.x.shape} vs {xtr.shape}")
    on = off_train.fit_normalization(ds.x, ds.y)
    check("P2 official fit_normalization == ours", all(np.array_equal(on[k], z[k]) for k in on))
    mcfg = {**nm.DEFAULT_MODEL, **small}
    off = build_model(mcfg, 21, 7, H)
    sd = torch.load(run_dir / "model.pt")["model_state_dict"]
    off.load_state_dict(sd)
    off.eval()
    xb = torch.from_numpy(((xva[:64] - z["x_mean"]) / z["x_std"]).astype(np.float32))
    with torch.inference_mode():
        same = torch.equal(off(xb), run.model(xb))
    check("P3 official LSTMRegressor loads our state_dict (same keys/shapes) and gives identical outputs", same)
    lc = load_checkpoint(run_dir)
    buf = HistoryBuffer(lc.history)
    for i in range(140, 150):
        buf.append(q[i], dq[i], cmd[i])
    xo = (buf.array() - lc.normalization["x_mean"]) / lc.normalization["x_std"]
    with torch.inference_mode():
        yo = lc.model(torch.from_numpy(xo).unsqueeze(0)).numpy()[0] * lc.normalization["y_std"] + lc.normalization["y_mean"]
    ours = run.predict(nm.step_features(q[140:150], dq[140:150], cmd[140:150])[None])[0]
    check("P4 official load_checkpoint + HistoryBuffer on our run == our prediction (residual contract)",
          np.allclose(yo, ours, atol=1e-6) and np.allclose(tau[149] - yo, res.loc[149, [f"RJ{i}.tau_ext_hat" for i in range(1, 8)]].to_numpy(np.float32), atol=1e-5))

print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
