"""Episode selection for v3: every evidence-complete G+ candidate that
  (1) passes the replay checks (render_v4/traj_check.json: limits, speed <= 1.25x and accel <= 1.5x the real-demo
      maximum, mesh collision distance >= the real demos', table clearance, exact start pose, grasp pose = source), and
  (2) leaves the receiving (left) arm exactly on its recorded trajectory (max left-TCP deviation from the source at the
      time map <= 1 mm), so the hand-over itself is the real recording (re-posing both arms there is where image
      synthesis fails, even with the improved renderer)."""
import json, sys
from pathlib import Path
import numpy as np
SFA = Path("/home/stanley/Stanley_ws/lerobot_openarm/research/sim_free_augmentation"); sys.path.insert(0, str(SFA))
from sfa import data
from sfa.robot import Robot
HERE = Path(__file__).resolve().parent; V31 = SFA / "v3_1/outputs"
robot = Robot()
tc = {(c["E"], c["key"]): c for c in json.loads((HERE.parent / "render_v4/traj_check.json").read_text())["candidates"]}
man = json.loads((V31 / "dataset_export/Gp_matched_evidence_complete_manifest.json").read_text())
src = {}
sel, rej = [], []
for e in man:
    E = e["E"]; a, b = e["rows"]
    if E not in src:
        s = data.load(E, cams=()); sp = Robot.split(s.state.astype(np.float64))
        src[E] = np.stack([robot.tcp_traj({"R": sp["R"], "L": sp["L"]}, "L")[:, :3, 3]])[0]
    z = np.load(V31 / f"ep{E}/candidates/{e['key']}.npz")
    S = z["state"][a:b].astype(np.float64); tau = z["tau"][a:b]
    sp = Robot.split(S); L = robot.tcp_traj({"R": sp["R"], "L": sp["L"]}, "L")[:, :3, 3]
    ref = np.stack([np.interp(tau, np.arange(len(src[E])), src[E][:, k]) for k in range(3)], 1)
    dL = float(1000 * np.linalg.norm(L - ref, axis=1).max())
    c = tc[(E, e["key"])]
    reason = []
    if not c["pass"]: reason.append("replay checks")
    if dL > 1.0: reason.append(f"left arm re-posed ({dL:.1f} mm)")
    (rej if reason else sel).append({**e, "left_tcp_dev_mm": dL, "reject": reason, "replay": {k: c[k] for k in (
        "speed_ratio_to_real_max", "accel_ratio_to_real_max", "speed_ratio_to_urdf", "min_mesh_distance_mm", "lowest_arm_z_m",
        "grasp_R_err_mm_deg", "grasp_L_err_mm_deg", "start_state_err_rad", "can_init_xy_m", "source_can_init_xy_m")}})
print("selected", len(sel), "rejected", len(rej))
import collections
print(collections.Counter(x["variation"].split("_")[0] if x["variation"].startswith("init") else x["variation"] for x in sel))
for r in rej: print("  reject", r["E"], r["variation"], r["reject"])
(HERE / "selection.json").write_text(json.dumps({"selected": sel, "rejected": rej}, indent=1, default=float))
