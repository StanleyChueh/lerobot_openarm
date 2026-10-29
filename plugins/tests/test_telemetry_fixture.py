"""The committed mock fixture (plugins/tests/fixtures/openarm_telemetry_1, made by make_telemetry_fixture.py) still
matches the producer's contract, so a consumer's integration tests built on it test the real format. No hardware.

    env -u PYTHONPATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python plugins/tests/test_telemetry_fixture.py

Fails when telemetry.py's columns, frame layout or schema string change without regenerating the fixture (and,
if the meaning changed, bumping openarm_telemetry/N), and when the analyzer's verdict on it drifts.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import lerobot_robot_openarm_umeow  # noqa: F401  (adds the repo root to sys.path)
from lerobot_robot_openarm_umeow import telemetry as tel

REPO = Path(__file__).resolve().parents[2]
FX = Path(__file__).resolve().parent / "fixtures/openarm_telemetry_1"
sys.path.insert(0, str(REPO / "tools/force_aware"))
import telemetry_check as tc  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


exp = json.loads((FX / "expected.json").read_text())
df, meta = tel.load(FX / "session")
fr = tel.load_frames(FX / "session")
part = pq.read_schema(next((FX / "session").glob("part-*.parquet")))
fpart = pq.read_schema(next((FX / "session").glob("frames-*.parquet")))
check("F1 schema strings and row columns are the producer's current ones",
      meta["schema"] == tel.SCHEMA_VERSION == exp["schema"] and meta["frames_schema"] == tel.FRAMES_SCHEMA
      and meta["columns"] == tel.COLUMNS == list(df.columns) and len(tel.COLUMNS) == exp["columns"],
      f"{meta['schema']}, {len(tel.COLUMNS)} columns")
check("F1 Parquet types: tick int64, every other row column float64; frames tick/seq/t_ns int64, side string, "
      "can_id int32, data binary",
      str(part.field("tick").type) == "int64" and all(str(part.field(c).type) == "double" for c in tel.COLUMNS[1:])
      and [str(fpart.field(c).type) for c in tel.FRAME_COLUMNS] == ["int64", "string", "int64", "int32", "binary", "int64"])
check("F2 rows: consecutive ticks, one final row, the failed snapshot row is NaN but kept",
      np.array_equal(df["tick"], np.arange(exp["rows"])) and df["kind"].iloc[-1] == 1 and (df["kind"] == 1).sum() == 1
      and all(df.loc[t, "snapshot_ok"] == 0 and np.isnan(df.loc[t, "LJ1.q"]) for t in exp["failed_snapshot_ticks"]))
info = json.loads((FX / "dataset/meta/info.json").read_text())
data = pd.read_parquet(FX / "dataset/data/chunk-000/file-000.parquet")
starts = []
for ep, g in data.groupby("episode_index"):
    starts.append(tel.align_to_dataset(df, np.stack(g["observation.state"]), info["features"]["observation.state"]["names"]))
check("F3 align_to_dataset finds each episode at its expected tick",
      starts == [e["start_tick"] for e in exp["episodes"]], f"{starts}")
# kind-0 rows only: on the final row every joint the return ramp moved is stale, correctly -- no read refreshed
# the library's state after the ramp's replies.
stale = {j: df.index[(df[f"{j}.stale"] == 1) & (df["kind"] == 0)].tolist() for j in tel.JOINTS}
check("F4 stale observation rows are exactly the expected ones (RJ5 after two lost replies)",
      {j: v for j, v in stale.items() if v} == exp["stale_joint_rows"], f"{ {j: v for j, v in stale.items() if v} }")
check("F4 hold rows are exactly the expected ones, and sent stays frozen on them while desired moves",
      df.index[df["hold"] == 1].tolist() == exp["hold_ticks"]
      and df.loc[exp["hold_ticks"], "RJ1.sent"].nunique() == 1 and df.loc[exp["hold_ticks"], "RJ1.desired"].nunique() > 1)
check("F5 frames: per arm, seq is 0..n-1 and every frame belongs to an existing row",
      all(np.array_equal(g["seq"], np.arange(len(g))) for _, g in fr.groupby("side")) and fr["tick"].isin(df["tick"]).all()
      and len(fr) == sum(meta["taps"][s]["frames"] for s in "LR"), f"{len(fr)} frames")
report, _ = tc.run(FX / "session", FX / "dataset")
got = {i["id"]: i["status"] for i in report.items}
check("F6 telemetry_check's verdict on the fixture is unchanged", got == exp["analyzer"]["checks"]
      and report.verdict() == exp["analyzer"]["verdict"], f"{report.verdict()}; diff {set(got.items()) ^ set(exp['analyzer']['checks'].items())}")
check("F7 the fixture says it is mock data", "NOT hardware" in meta.get("mock_fixture", ""))

print(f"\n{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
