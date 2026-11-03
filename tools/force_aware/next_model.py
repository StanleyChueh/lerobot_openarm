"""NEXT free-motion effort model for one OpenArm arm: windows, normalization, LSTM, CPU training, residuals.

Reproduces the official FACTR2 NEXT contract (github.com/philiphan0109/factr2_next @ 0df02c5,
src/factr2_next/factr2_next/{training,inference}) on next_export.py exports, without ROS:

    x_k   = [q_k, dq_k, cmd_q_k - q_k]          (21 = 3 x 7 arm joints, float32, cmd - q computed in float32)
    input = x over the H rows ending at k       (stateless sliding window; H rows of ONE segment, never bridged)
    y_k   = tau_k                               (motor-reported effort at the window's LAST row, not a future one)
    tau_ext_hat_k = tau_k - tau_free_hat_k      (uncalibrated effort residual; N*m nominal, NOT a joint torque)

Normalization: mean/std over all training windows and time steps (+1e-6), fitted on TRAINING sessions only.
The LSTM, its head and the checkpoint layout (model.pt, config.yaml, normalization.npz, metrics.json) are the
official ones, so the official load_checkpoint() reads a run made here and vice versa.

    python tools/force_aware/next_model.py train --arm left --exports <exp>/<s1> <exp>/<s2> ... \\
        --val-sessions <s2> --out <runs dir> [--history 30 --epochs 20]
    python tools/force_aware/next_model.py estimate --run <run dir> --export <exp>/<s> --arm left --out res.parquet

CPU only: there is no GPU path here by design (Gate 3A is offline).
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import next_export as nx  # noqa: E402

OFFICIAL = {"repo": "https://github.com/philiphan0109/factr2_next",
            "commit": "0df02c5b26e33911078152b6c587d6e8a160d279"}
# Official config/train.yaml model + train defaults. history is the one deliberate change: 50 rows = 1.0 s at the
# official 50 Hz; OpenArm logs at 30 Hz, so 30 rows keep the 1.0 s span (a hypothesis to test, not a result).
DEFAULT_MODEL = {"type": "lstm", "state_mode": "stateless", "hidden_size": 128, "num_layers": 2,
                 "bidirectional": False, "head_hidden": 256, "head_layers": 2, "dropout": 0.1}
DEFAULT_TRAIN = {"batch_size": 2048, "epochs": 20, "learning_rate": 1e-3, "device": "cpu", "seed": 0}
DEFAULT_HISTORY = 30


# -- data -------------------------------------------------------------------------------------------------------
def step_features(q, dq, cmd):
    """Official NextTorqueDataset._episode_arrays: float32 first, then cmd - q."""
    q, dq, cmd = (np.asarray(a, dtype=np.float32) for a in (q, dq, cmd))
    return np.concatenate([q, dq, cmd - q], axis=1)


def segment_runs(seg):
    """[(start, stop)] index ranges of each segment id >= 0, in row order."""
    runs = []
    for s in np.unique(seg[seg >= 0]):
        i = np.flatnonzero(seg == s)
        if not np.array_equal(i, np.arange(i[0], i[-1] + 1)):
            raise ValueError(f"segment {s} is not contiguous")
        runs.append((int(i[0]), int(i[-1]) + 1))
    return runs


def arm_windows(export_dir, arm, history, require_contact_free=True):
    """(x [N, H, 21] float32, y [N, 7] float32, end row index [N]) of every window that lies inside one segment
    of valid (and, for training, contact-free) rows. Re-segments after the contact-free cut."""
    m = nx.load_export(export_dir)
    tab, q, dq, cmd, tau = nx.arm_arrays(export_dir, arm)
    use = tab["valid"].to_numpy(dtype=bool)
    if require_contact_free:
        use &= tab["contact_free"].to_numpy(dtype=bool)
    seg = nx.segments(use, tab["tick"].to_numpy(), tab["t"].to_numpy(), m["settings"]["fps"])
    feats = step_features(q, dq, cmd)
    tau32 = tau.astype(np.float32)
    xs, ys, ends = [], [], []
    for a, b in segment_runs(seg):
        if b - a < history:
            continue
        xs.append(np.stack([feats[i:i + history] for i in range(a, b - history + 1)]))
        ys.append(tau32[a + history - 1:b])  # the window's last row: Paper Sec. 4, Eq. (3)
        ends.append(np.arange(a + history - 1, b))
    if not xs:
        return np.zeros((0, history, 21), np.float32), np.zeros((0, 7), np.float32), np.zeros(0, np.int64)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ends)


def collect(export_dirs, arm, history, require_contact_free=True):
    xs, ys = [], []
    for d in export_dirs:
        x, y, _ = arm_windows(d, arm, history, require_contact_free)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def fit_normalization(x, y):
    """Official train.fit_normalization."""
    return {"x_mean": x.mean(axis=(0, 1)), "x_std": x.std(axis=(0, 1)) + 1e-6,
            "y_mean": y.mean(axis=0), "y_std": y.std(axis=0) + 1e-6}


# -- model (official training/models.py, LSTM path) ------------------------------------------------------------
def _torch():
    import torch
    return torch


def regression_head(input_size, output_size, head_hidden=256, head_layers=2, dropout=0.0):
    nn = _torch().nn
    layers, width = [], input_size
    for _ in range(max(0, int(head_layers) - 1)):
        layers.extend([nn.Linear(width, head_hidden), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        width = head_hidden
    layers.append(nn.Linear(width, output_size))
    return nn.Sequential(*layers)


def build_lstm(model_cfg, input_size, output_size):
    torch = _torch()
    cfg = {**DEFAULT_MODEL, **model_cfg}
    if str(cfg["type"]).lower() != "lstm" or str(cfg["state_mode"]).lower() != "stateless":
        raise ValueError("only the official stateless LSTM is implemented here")

    class LSTMRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=int(input_size), hidden_size=int(cfg["hidden_size"]),
                                      num_layers=int(cfg["num_layers"]), batch_first=True,
                                      bidirectional=bool(cfg["bidirectional"]),
                                      dropout=float(cfg["dropout"]) if int(cfg["num_layers"]) > 1 else 0.0)
            head_in = int(cfg["hidden_size"]) * (2 if cfg["bidirectional"] else 1)
            self.head = regression_head(head_in, int(output_size), int(cfg["head_hidden"]),
                                        int(cfg["head_layers"]), float(cfg["dropout"]))

        def forward(self, x):
            y, _ = self.lstm(x)
            return self.head(y[:, -1])

    return LSTMRegressor()


# -- training ---------------------------------------------------------------------------------------------------
def _check_sets(train_dirs, val_dirs):
    mans = [nx.load_export(d) for d in [*train_dirs, *val_dirs]]
    ids = [m["source"]["session_id"] for m in mans]
    if len(set(ids)) != len(ids):
        raise nx.ExportError(f"a session appears more than once across train/val: {ids}")
    tr = {nx.load_export(d)["source"]["session_id"] for d in train_dirs}
    va = {nx.load_export(d)["source"]["session_id"] for d in val_dirs}
    if tr & va or not tr or not va:
        raise nx.ExportError(f"train/val must be disjoint, non-empty session sets (train {sorted(tr)}, val {sorted(va)})")
    if len({m["hardware_signature"]["mock"] for m in mans}) > 1:
        raise nx.ExportError("mock-fixture and real sessions cannot be mixed")
    return nx.check_compatible(mans), sorted(tr), sorted(va)


def train(train_dirs, val_dirs, arm, out_dir, history=DEFAULT_HISTORY, model_cfg=None, train_cfg=None):
    """Train one arm's NEXT model on CPU from contact-free windows; write an official-layout run directory."""
    torch = _torch()
    import yaml

    tcfg = {**DEFAULT_TRAIN, **(train_cfg or {})}
    if str(tcfg["device"]) != "cpu":
        raise ValueError("CPU only in Gate 3A")
    mcfg = {**DEFAULT_MODEL, **(model_cfg or {})}
    sig, tr_ids, va_ids = _check_sets(train_dirs, val_dirs)
    seed = int(tcfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    xtr, ytr = collect(train_dirs, arm, history)
    xva, yva = collect(val_dirs, arm, history)
    if len(xtr) == 0 or len(xva) == 0:
        raise nx.ExportError(f"no contact-free windows (train {len(xtr)}, val {len(xva)}) at history {history}")
    norm = fit_normalization(xtr, ytr)  # training sessions only
    nrm = lambda x, y: (torch.from_numpy((x - norm["x_mean"]) / norm["x_std"]).float(),  # noqa: E731
                        torch.from_numpy((y - norm["y_mean"]) / norm["y_std"]).float())
    Xtr, Ytr = nrm(xtr, ytr)
    Xva, Yva = nrm(xva, yva)
    model = build_lstm(mcfg, Xtr.shape[-1], Ytr.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=float(tcfg["learning_rate"]))
    bs = int(tcfg["batch_size"])
    gen = torch.Generator().manual_seed(seed)
    metrics = {"arm": arm, "train_loss": [], "val_loss": []}

    def run(X, Y, opt=None):
        model.train(opt is not None)
        order = torch.randperm(len(X), generator=gen) if opt is not None else torch.arange(len(X))
        total = 0.0
        for i in range(0, len(X), bs):
            idx = order[i:i + bs]
            with torch.set_grad_enabled(opt is not None):
                loss = torch.nn.functional.mse_loss(model(X[idx]), Y[idx])
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            total += loss.item() * len(idx)
        return total / max(len(X), 1)

    for _ in range(int(tcfg["epochs"])):
        metrics["train_loss"].append(run(Xtr, Ytr, opt))
        metrics["val_loss"].append(run(Xva, Yva))
    out = Path(out_dir) / f"next_openarm_{arm}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    out.mkdir(parents=True, exist_ok=False)
    torch.save({"model_state_dict": model.state_dict(), "input_size": int(Xtr.shape[-1]),
                "output_size": int(Ytr.shape[-1]), "history": int(history), "model": mcfg}, out / "model.pt")
    cfg = {"data": {"arm_mode": arm, "history": int(history), "keys": {
        "joint_pos": "{arm}_joint_pos", "joint_vel": "{arm}_joint_vel", "joint_cmd": "{arm}_joint_cmd",
        "measured_joint_torque": "{arm}_measured_joint_torque"}}, "model": mcfg, "train": tcfg}
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    np.savez(out / "normalization.npz", **norm)
    metrics.update({"train_windows": int(len(xtr)), "val_windows": int(len(xva))})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "openarm.json").write_text(json.dumps({
        "official": OFFICIAL, "arm": arm, "joints": nx.ARM_JOINTS[arm], "history": int(history),
        "fps": nx.load_export(train_dirs[0])["settings"]["fps"], "train_sessions": tr_ids, "val_sessions": va_ids,
        "hardware_signature": sig, "target": "motor-reported effort tau (uncalibrated)",
        "inputs": "[q, dq, cmd_q - q], cmd_q = transmitted MIT target in effect before the measurement"}, indent=1))
    return out, metrics


# -- inference --------------------------------------------------------------------------------------------------
class Run:
    def __init__(self, run_dir):
        torch = _torch()
        run_dir = Path(run_dir)
        ck = torch.load(run_dir / "model.pt", map_location="cpu")
        self.model = build_lstm(ck.get("model", {}), ck["input_size"], ck["output_size"])
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()
        z = np.load(run_dir / "normalization.npz")
        self.norm = {k: z[k].astype(np.float32) for k in z.files}
        self.history = int(ck["history"])
        info = run_dir / "openarm.json"
        self.info = json.loads(info.read_text()) if info.exists() else {}

    def predict(self, x):
        """Free-motion effort for windows x [N, H, 21] (raw units) -> [N, 7]."""
        torch = _torch()
        if len(x) == 0:
            return np.zeros((0, 7), np.float32)
        xn = (np.asarray(x, np.float32) - self.norm["x_mean"]) / self.norm["x_std"]
        with torch.inference_mode():
            y = self.model(torch.from_numpy(xn)).numpy()
        return (y * self.norm["y_std"] + self.norm["y_mean"]).astype(np.float32)


def estimate(run, export_dir, arm, allow_signature_change=False):
    """Per row of one arm's export: measured effort, predicted free-motion effort and the residual. Rows without
    a full window of valid rows inside one segment ending at them are NaN. Contact need not be absent here."""
    if isinstance(run, (str, Path)):
        run = Run(run)
    m = nx.load_export(export_dir)
    want = run.info.get("hardware_signature")
    if want is not None and want != m["hardware_signature"] and not allow_signature_change:
        raise nx.ExportError("this export's hardware signature differs from the model's training sessions")
    if run.info.get("arm", arm) != arm:
        raise nx.ExportError(f"model is for the {run.info['arm']} arm, not {arm}")
    x, y, ends = arm_windows(export_dir, arm, run.history, require_contact_free=False)
    tab, *_ = nx.arm_arrays(export_dir, arm)
    n = len(tab)
    pred = np.full((n, 7), np.nan, np.float32)
    meas = np.full((n, 7), np.nan, np.float32)
    pred[ends] = run.predict(x)
    meas[ends] = y
    out = {"tick": tab["tick"].to_numpy(), "t": tab["t"].to_numpy(), "t_wall": tab["t_wall"].to_numpy(),
           "valid": tab["valid"].to_numpy(), "contact_free": tab["contact_free"].to_numpy(),
           "has_estimate": np.isin(np.arange(n), ends)}
    for i, j in enumerate(nx.ARM_JOINTS[arm]):
        out[f"{j}.tau"] = meas[:, i]
        out[f"{j}.tau_free_hat"] = pred[:, i]
        out[f"{j}.tau_ext_hat"] = meas[:, i] - pred[:, i]  # Paper Sec. 4, Eq. (2)
    return pd.DataFrame(out)


class StreamingEstimator:
    """Row-by-row residual, as an online consumer would compute it. Mirrors the official HistoryBuffer, except
    that the buffer is CLEARED on an invalid row, a tick gap or a spacing outside DT_BAND, so a window never
    spans a break. push() returns tau_ext_hat [7] or None until H consecutive valid rows have arrived."""

    def __init__(self, run, fps):
        self.run = run if isinstance(run, Run) else Run(run)
        self.fps = float(fps)
        self.rows, self.last = [], None

    def push(self, tick, t, valid, q, dq, cmd, tau):
        lo, hi = nx.DT_BAND[0] / self.fps, nx.DT_BAND[1] / self.fps
        if not valid:
            self.rows, self.last = [], None
            return None
        if self.last is not None and (tick - self.last[0] != 1 or not lo <= t - self.last[1] <= hi):
            self.rows = []
        self.last = (tick, t)
        self.rows.append(step_features(np.asarray(q)[None], np.asarray(dq)[None], np.asarray(cmd)[None])[0])
        self.rows = self.rows[-self.run.history:]
        if len(self.rows) < self.run.history:
            return None
        free = self.run.predict(np.stack(self.rows)[None])[0]
        return np.asarray(tau, np.float32) - free


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=sorted(nx.ARMS))
    t.add_argument("--exports", nargs="+", required=True)
    t.add_argument("--val-sessions", nargs="*", help="session ids for validation (default: hash split)")
    t.add_argument("--val-fraction", type=float, default=0.2)
    t.add_argument("--out", required=True)
    t.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    t.add_argument("--epochs", type=int, default=DEFAULT_TRAIN["epochs"])
    e = sub.add_parser("estimate")
    e.add_argument("--run", required=True)
    e.add_argument("--export", required=True)
    e.add_argument("--arm", required=True, choices=sorted(nx.ARMS))
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "train":
        by_id = {nx.load_export(d)["source"]["session_id"]: d for d in a.exports}
        tr, va = nx.split_sessions(list(by_id), a.val_fraction, a.val_sessions)
        out, met = train([by_id[s] for s in tr], [by_id[s] for s in va], a.arm, a.out, a.history,
                         train_cfg={"epochs": a.epochs})
        print(f"{out}: train {met['train_windows']} / val {met['val_windows']} windows, "
              f"val loss {met['val_loss'][-1]:.4f} (normalized)")
    else:
        if Path(a.out).exists():
            raise SystemExit(f"{a.out} exists; not overwritten")
        estimate(a.run, a.export, a.arm).to_parquet(a.out, index=False)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
