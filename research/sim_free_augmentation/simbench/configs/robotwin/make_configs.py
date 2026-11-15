"""sfagen configs for the RoboTwin 2.0 pilot tasks (SIM_BENCHMARK_PLAN S1). Pipeline v0 recipe: the reconstruction,
render, checks, plan and video_encoding blocks of configs/openarm_pringles_v03r2_loose.yaml unchanged except the
robot-specific link-name lists (A9); variations = the same recipe (F1W 3 cm grid, sources <= 12 cm, wide blend,
time scale 1.0, interaction shifts +-3/+-5 cm, timing +-10). Per task only `objects`, the arena and episode groups change.

Episode groups (manual step, reported in R01): RoboTwin picks the arm per episode from the object side, while sfagen fixes
`objects.<name>.grasped_by` per config (A13). Each task is split into groups of episodes with the same grasping arms,
read from the gripper commands (close events of the action, threshold 0.5). Every group config lists the frozen
source positions of ALL episodes of its task, so target-to-source assignment is the same as for one config.

Usage: make_configs.py RUN_DIR [POSITIONS_DIR [TASK]]   -> writes <task>_base.yaml and <task>_<group>.yaml next to this file
POSITIONS_DIR/<task>.json = {task: {episode: [x, y]}} from tools/source_positions.py (omit on the first pass)."""
import copy, json, sys
from pathlib import Path
import numpy as np, yaml

HERE = Path(__file__).resolve().parent; SF = HERE.parents[2] / "sfagen"; WT = HERE.parents[4]
DATA = WT / "artifacts/simbench_data"; EMB = DATA / "rt_assets/embodiments/aloha-agilex"
run_dir = Path(sys.argv[1]).resolve(); pos = {k: v for f in sorted(Path(sys.argv[2]).glob("*.json")) if not f.stem.endswith("_recon_summary") for k, v in json.loads(f.read_text()).items()} if len(sys.argv) > 2 else {}
ref = yaml.safe_load((SF / "configs/openarm_pringles_v03r2_loose.yaml").read_text())
cam = yaml.safe_load((HERE / "_camera_guess.yaml").read_text())["camera"]

grip = lambda side, i: {"state_index": i, "open_raw": 1.0, "closed_raw": 0.0, "finger_joints": [f"{side}_joint7", f"{side}_joint8"],  # noqa: E731
                        "finger_open": 0.045, "close_threshold": 0.5}  # RoboTwin: finger = -0.01 + 0.055 * command, clipped at 0
robot = {"urdf": str(EMB / "urdf/arx5_sfagen.urdf"), "package_dirs": [str(EMB / "urdf")],
         "arms": {"L": {"joints": [f"fl_joint{k}" for k in range(1, 7)], "state_indices": list(range(0, 6)), "tcp_frame": "fl_tcp", "gripper": grip("fl", 6)},
                  "R": {"joints": [f"fr_joint{k}" for k in range(1, 7)], "state_indices": list(range(7, 13)), "tcp_frame": "fr_tcp", "gripper": grip("fr", 13)}}}

# A9: OpenArm link-name lists mapped by function (proximal link touching the arm base; proximal links never reach the table)
recon = {**ref["reconstruction"], "sam2_exclude_links": ["link1"]}
checks = {**ref["checks"], "skip_collision_links": ["link1"], "table_skip_links": ["link1", "link2"]}


def arena(xw, yw):
    """task randomisation range (RoboTwin envs/<task>.py, world frame) -> robot root frame (robot_pose: base at
    world (0, -0.65, 0), yaw +90 deg): x_r = y_w + 0.65, y_r = -x_w. Grid: 3 cm, inclusive."""
    xs = np.round(np.arange(yw[0] + 0.65, yw[1] + 0.65 + 1e-9, 0.03), 3); ys = np.round(np.arange(-xw[1], -xw[0] + 1e-9, 0.03), 3)
    return [[float(x), float(y)] for x in xs for y in ys]


box = lambda h=None: {"type": "box", **({"height": h} if h else {})}  # noqa: E731
TASKS = {
    # envs/handover_block.py: box half_size (0.03, 0.03, 0.1), xlim [-0.25, -0.05], ylim [0, 0.25]
    "handover_block": {"arena": arena([-0.25, -0.05], [0.0, 0.25]), "target": "block",
                       "groups": {"L": ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], {"block": {"grasped_by": "L", "grasp_index": 0, "proxy": box(0.2)}})}},
    # envs/place_empty_cup.py: cup 021_cup (size unknown here: box proxy, height from the grasp), xlim [0.15, 0.3] (right) or
    # [-0.3, -0.15] (left), ylim [-0.2, 0.05]
    "place_empty_cup": {"arena": {"R": arena([0.15, 0.3], [-0.2, 0.05]), "L": arena([-0.3, -0.15], [-0.2, 0.05])}, "target": "cup",
                        "groups": {"R": ([0, 1, 2, 4, 6, 7], {"cup": {"grasped_by": "R", "grasp_index": 0, "proxy": box()}}),
                                   "L": ([3, 5, 8, 9], {"cup": {"grasped_by": "L", "grasp_index": 0, "proxy": box()}})}},
    # envs/stack_blocks_two.py: cubes half 0.025, xlim [-0.28, 0.28], ylim [-0.08, 0.05]; block_a = grasped first
    "stack_blocks_two": {"arena": arena([-0.28, 0.28], [-0.08, 0.05]), "target": "block_a",
                         "groups": {"LL": ([0, 7], {"block_a": {"grasped_by": "L", "grasp_index": 0, "proxy": box(0.05)}, "block_b": {"grasped_by": "L", "grasp_index": 1, "proxy": box(0.05)}}),
                                    "RR": ([8], {"block_a": {"grasped_by": "R", "grasp_index": 0, "proxy": box(0.05)}, "block_b": {"grasped_by": "R", "grasp_index": 1, "proxy": box(0.05)}}),
                                    "RL": ([1, 3, 4, 5, 6], {"block_a": {"grasped_by": "R", "grasp_index": 0, "proxy": box(0.05)}, "block_b": {"grasped_by": "L", "grasp_index": 0, "proxy": box(0.05)}}),
                                    "LR": ([2, 9], {"block_a": {"grasped_by": "L", "grasp_index": 0, "proxy": box(0.05)}, "block_b": {"grasped_by": "R", "grasp_index": 0, "proxy": box(0.05)}})}},
}

BLEND = "joint_fallback"  # pipeline v1 (R02 Amendment 2): v0 Cartesian blend, joint-space blend only if v0 fails; same for every robot/task
only = sys.argv[3] if len(sys.argv) > 3 else None  # rewrite one task only (drivers run tasks in parallel)
for task, t in TASKS.items():
    if only and task != only:
        continue
    root = DATA / "robotwin_pilot" / task
    src = {"repo_id": f"local/robotwin_unified_{task}", "root": str(root), "body_camera": "observation.images.cam_high",
           "wrist_cameras": ["observation.images.cam_left_wrist", "observation.images.cam_right_wrist"],
           "wrist_camera_arms": {"observation.images.cam_left_wrist": "L", "observation.images.cam_right_wrist": "R"}}
    base = {"source": {**src, "episodes": list(range(10))}, "robot": robot, "camera": cam, "reconstruction": recon,
            "render": ref["render"], "checks": checks, "calibration": ref["calibration"], "objects": {}, "variations": {},
            "video_encoding": ref["video_encoding"], "work_dir": str(run_dir / task / "rig"), "plan": {**ref["plan"], "blend_space": BLEND}}
    (HERE / f"{task}_base.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    for g, (eps, objs) in t["groups"].items():
        c = copy.deepcopy(base); c["source"]["episodes"] = eps; c["objects"] = objs; c["work_dir"] = str(run_dir / task / g)
        v = copy.deepcopy(ref["variations"]); ot = v["object_targets"]; ot["object"] = t["target"]
        ot["positions_xy_m"] = t["arena"][g] if isinstance(t["arena"], dict) else t["arena"]
        ot["source_positions_xy_m"] = {int(e): p for e, p in pos.get(task, {}).items()} or {}
        c["variations"] = v
        (HERE / f"{task}_{g}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
        print(task, g, eps, len(ot["positions_xy_m"]), "targets", len(ot["source_positions_xy_m"]), "source positions")
