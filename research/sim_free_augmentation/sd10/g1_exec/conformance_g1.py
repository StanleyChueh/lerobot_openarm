"""Pre-launch conformance fixtures (registered as analytic/source-identity fixtures; NOT the 100 locked targets).

Source: one reconstructed capture (default 6). Fixtures:
  F0 identity: init = I at the source's own lid -> planned arms/actions must reproduce the source (|dev| <= 1e-6).
  F1 analytic translation: own lid + (0.02, 0, 0) m at the source lid height -> exported can reaches it exactly,
     giver contact waypoints = init @ source TCP, labels reconstruct.
  Corruptions of F1 that the gates must reject: source-action substitution, one-joint 6-frame delay, gripper command
     shifted 3 frames, exported state delayed 3 frames, initial can offset 5 mm, can rotated 20 deg after receiver
     capture, can held still while the giver moves (copied can).
Writes outputs/conformance.json. Exits non-zero if any fixture fails.
"""
from __future__ import annotations

import copy
import json
import sys

import numpy as np

import attempt_g1 as AT
import g1gates as GG
import g1lib as G


def main(src=6):
    from sfa import robot as rb
    from sfa2 import se3
    rec = G.load_light(src)
    robot, geom = AT.robot_geom()
    rep = {"source": src}
    # F0 identity
    spec0 = G.make_spec(rec, robot, np.eye(4))
    p0 = G.plan(rec, robot, spec0)
    rows = np.arange(3, rec.T - 6)
    dq = max(float(np.abs(p0["q"][s] - rec.g[s]).max()) for s in "RL")
    da = float(np.nanmax(np.abs(p0["action"][rows] - rec.action[rows])))
    rep["F0_identity"] = {"max_q_dev_rad": dq, "max_action_dev_rad": da, "ok": bool(dq <= 1e-6 and da <= 1e-6)}
    # F1 analytic translation at the source lid height
    tgt = {"target_id": "FIXTURE", "candidate_lid_xyz_robot_base_m": (rec.table[:3, 3] + [0.02, 0, 0]).tolist(),
           "requested_lid_uv_native_px": [0.0, 0.0]}
    init, tinfo = G.init_transform(rec, tgt["candidate_lid_xyz_robot_base_m"])
    spec = G.make_spec(rec, robot, init)
    p = G.plan(rec, robot, spec)
    ex = G.export(rec, p)
    base = GG.evaluate(rec, robot, geom, spec, p, ex, tgt)
    g = base["gates"]
    rep["F1_translation"] = {"transform": G.jsonable(tinfo), "Y_A": base["Y_A"], "failed": base["failed_gates"],
                             "init_lid_err_m": g["2_initial_candidate"]["lid_position_err_m"],
                             "init_axis_err_deg": g["2_initial_candidate"]["axis_err_deg"],
                             "label_dev": g["9_labels"]["max_arm_label_dev_rad"], "gate9_ok": g["9_labels"]["ok"],
                             "gate3": {k: g["3_ik_residual"][k] for k in ("max_pos_m_all_frames", "max_rot_deg_all_frames", "contact_waypoint_max_mm")}}
    rep["F1_translation"]["ok"] = bool(g["2_initial_candidate"]["lid_position_err_m"] < 1e-9 and g["9_labels"]["ok"]
                                       and g["1_arrays_support"]["ok"])
    # corruptions: each must flip the named gate(s) from pass to fail
    def corrupt(fn, must):
        e2, p2 = copy.deepcopy(ex), copy.deepcopy(p)
        fn(e2, p2)
        r = GG.evaluate(rec, robot, geom, spec, p2, e2, tgt)
        caught = [k for k in must if not r["gates"][k]["ok"] and g[k]["ok"]]
        return {"must_fail": must, "caught_by": caught, "failed": r["failed_gates"], "ok": bool(caught)}
    arm = [rb.NAMES.index(f"{s}J{i}.pos") for s in "RL" for i in range(1, 8)]
    j0 = rb.NAMES.index("RJ2.pos")
    gR = rb.NAMES.index("RJ8.pos")
    def sub(e, p_):
        e["action"][:, arm] = rec.action[rows][:, arm].astype(np.float32)
    def delay_joint(e, p_):
        x = e["action"][:, j0].copy(); e["action"][6:, j0] = x[:-6]
    def grip_shift(e, p_):
        x = e["action"][:, gR].copy(); e["action"][3:, gR] = x[:-3]
    def state_delay(e, p_):
        x = e["state"].copy(); e["state"][3:] = x[:-3]
    def can_offset(e, p_):
        e["can"] = e["can"].copy(); e["can"][: rec.ev["right_close"] + 8 - 3, 0, 3] += 0.005
    def rot_after_capture(e, p_):
        e["can"] = e["can"].copy(); lc = rec.ev["left_close"] - 3
        R20 = se3.rot([1, 0, 0], 20.0)
        for i in range(lc + 5, len(e["can"])):
            e["can"][i] = e["can"][i] @ R20
    def copied_can(e, p_):
        e["can"] = e["can"].copy(); k = rec.ev["right_close"] + 8 - 3
        e["can"][k:] = e["can"][k - 1]
    tests = {"source_action_substitution": corrupt(sub, ["9_labels"]),
             "one_joint_delay6": corrupt(delay_joint, ["9_labels"]),
             "gripper_cmd_shift3": corrupt(grip_shift, ["9_labels"]),
             "state_delay3": corrupt(state_delay, ["9_labels", "3_ik_residual"]),
             "initial_can_offset5mm": corrupt(can_offset, ["2_initial_candidate", "8_object_continuity"]),
             "can_rotated20_after_capture": corrupt(rot_after_capture, ["6_steady_holds", "8_object_continuity"]),
             "copied_static_can": corrupt(copied_can, ["6_steady_holds", "7_receiver_capture"])}
    rep["corruptions"] = tests
    rep["ALL_PASS"] = bool(rep["F0_identity"]["ok"] and rep["F1_translation"]["ok"] and all(t["ok"] for t in tests.values()))
    G.OUT.mkdir(parents=True, exist_ok=True)
    (G.OUT / "conformance.json").write_text(json.dumps(G.jsonable(rep), indent=1))
    print(json.dumps(G.jsonable({k: (v if k != "corruptions" else {a: (b["ok"], b["caught_by"]) for a, b in v.items()}) for k, v in rep.items()}), indent=1))
    if not rep["ALL_PASS"]:
        sys.exit(1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
