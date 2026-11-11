"""Target positions (where the task goal happens) for real sources, published synthetic episodes and the wider real pool.
All from recorded / published joint states + gripper commands via URDF forward kinematics (no annotation):
  Pringles  hand-over point = right TCP at the left-gripper close command (both hands on the can)
  Battery   station pick    = left TCP at the left-gripper close command (station grasped on the table)
            insertion point = right TCP at the first right-gripper open command (battery released into the station)
            battery grasp   = right TCP at the first right-gripper close command (object position, not a target)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
SFA = Path("/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation")
sys.path[:0] = [str(SFA), str(ROOT / "battery/bsfa")]
from sfa.robot import Robot
from sfa.data import command_events
import bdata
robot = Robot()
PR_POOL = Path.home() / ".cache/huggingface/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_v00/snapshots/2c1bc860347c44fc0fbae7567469d36ee49baa8d/data"
PR_SRC = Path.home() / ".cache/huggingface/lerobot/hub/datasets--ethanCSL--openarm_pringles_lerobot_real_10_v00/snapshots/cebdee98115ab6827a4ec2655335c536b5dd085b/data"


def tcp(S, side, t):
    sp = Robot.split(S.astype(np.float64)); robot.fk(sp["R"][t], sp["L"][t]); return robot.frame(robot.tcp[side])[:3, 3]


def episodes(path):
    df = pd.read_parquet(sorted(Path(path).rglob("*.parquet")))
    for e, d in df.groupby("episode_index"):
        d = d.sort_values("frame_index")
        yield int(e), np.stack(d["observation.state"].to_numpy()).astype(float), np.stack(d["action"].to_numpy()).astype(float)


def pringles(S, A):
    ev = command_events(A); return {"handover": tcp(S, "R", ev["left_close"]).tolist()}


def battery(S, A):
    ev = bdata.grip_events(A)
    if not ev["L"]["closes"] or not ev["R"]["opens"]:
        return None
    return {"station_pick": tcp(S, "L", ev["L"]["closes"][0]).tolist(), "insertion": tcp(S, "R", ev["R"]["opens"][0]).tolist(),
            "battery_grasp": tcp(S, "R", ev["R"]["closes"][0]).tolist()}


def collect(path, fn, skip=()):
    out = {}
    for e, S, A in episodes(path):
        if e in skip:
            continue
        try:
            r = fn(S, A)
        except Exception as ex:   # an episode whose gripper events cannot be parsed is reported, not guessed
            r = None; print("skip", path, e, ex)
        if r is not None:
            out[e] = r
    return out


def main(extra=None):
    res = {"definition": __doc__}
    res["pringles"] = {"source": collect(PR_SRC, pringles), "synthetic_v3": collect(ROOT / "3cam_v3/dataset/data", pringles),
                       "pool": collect(PR_POOL, pringles, skip=range(10))}
    sub = [15, 20, 21, 22, 23, 30, 32, 36, 45, 53]
    res["battery"] = {"source": collect(bdata.ROOT / "data", battery), "synthetic_v00": collect(ROOT / "battery/dataset/data", battery),
                      "pool": collect(ROOT / "battery/pool/data", battery, skip=sub)}
    for name, path in (extra or {}).items():
        task, key = name.split("/"); res[task][key] = collect(path, pringles if task == "pringles" else battery)
    (HERE / "target_positions.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    r = main()
    for task in ("pringles", "battery"):
        for k, v in r[task].items():
            print(task, k, len(v))
