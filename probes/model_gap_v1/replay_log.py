"""lerobot-replay plus a log: same arguments, same robot behaviour, and every tick's sent action and measured joint
positions are written to outputs/model_gap_probe/ep<k>_<time>.csv (so the model can be checked at the real joints).
Usage: python research/sim_free_augmentation/simbench/probes/model_gap_v1/replay_log.py <the usual lerobot-replay arguments>"""
import csv, time
from pathlib import Path

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_replay import ReplayConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins


@parser.wrap()
def replay(cfg: ReplayConfig):
    proc = make_default_robot_action_processor()
    robot = make_robot_from_config(cfg.robot)
    ds = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])
    actions, names = ds.select_columns(ACTION), ds.features[ACTION]["names"]
    out = Path("outputs/model_gap_probe"); out.mkdir(parents=True, exist_ok=True)
    path = out / f"ep{cfg.dataset.episode}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    robot.connect()
    rows, dt, t0 = [], 1.0 / ds.fps, time.perf_counter()
    try:
        for i in range(ds.num_frames):
            a = {n: float(v) for n, v in zip(names, actions[i][ACTION])}
            obs = robot.get_observation()
            robot.send_action(proc((a, obs)))
            rows.append([i, round(time.perf_counter() - t0, 4)] + [a[n] for n in names] + [float(obs.get(n, float("nan"))) for n in names])
            time.sleep(max(0.0, t0 + (i + 1) * dt - time.perf_counter()))
    finally:
        robot.disconnect()
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["frame", "t"] + [f"a_{n}" for n in names] + [f"s_{n}" for n in names]); w.writerows(rows)
        print(f"[probe] wrote {path} ({len(rows)} ticks)")


if __name__ == "__main__":
    register_third_party_plugins()
    replay()
