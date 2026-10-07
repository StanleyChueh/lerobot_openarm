"""Dry run of the Quest teleop: no robot, no CAN, nothing moves.

    python -m lerobot_teleoperator_openarm_quest.preview            # print what would be sent
    python -m lerobot_teleoperator_openarm_quest.preview --viewer   # and show the IK pose in MuJoCo

Runs the same OpenArmQuest teleoperator lerobot-record uses, polls get_action() at --fps like the
record loop, and prints the hold/live state, the Quest packet rate, the IK solve rate and the motor
targets. With --viewer the IK model is drawn at the solved joints (written straight to qpos, no
physics), so a wrong pose seen here is the IK / VR mapping, not the robot.
"""

import argparse
import time

import numpy as np

from lerobot_robot_openarm_umeow.common import motor_action_to_sim_joints

from .config_openarm_quest import OpenArmQuestConfig
from .openarm_quest import OpenArmQuest


def main() -> None:
    defaults = OpenArmQuestConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=defaults.port)
    p.add_argument("--calibration", default=defaults.calibration)
    p.add_argument("--ik-xml", default=defaults.ik_xml)
    p.add_argument("--ik-args", default=defaults.ik_args)
    p.add_argument("--ik-hz", type=float, default=defaults.ik_hz)
    p.add_argument("--fps", type=float, default=30.0, help="get_action() polling rate, as in lerobot-record")
    p.add_argument("--print-hz", type=float, default=2.0)
    p.add_argument("--viewer", action="store_true", help="draw the solved pose in a MuJoCo window")
    p.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 = until Ctrl-C")
    args = p.parse_args()

    teleop = OpenArmQuest(
        OpenArmQuestConfig(
            port=args.port,
            calibration=args.calibration,
            ik_xml=args.ik_xml,
            ik_args=args.ik_args,
            ik_hz=args.ik_hz,
            episode_buttons=False,
        )
    )
    teleop.connect()

    viewer = model = data = None
    if args.viewer:
        import mujoco
        import mujoco.viewer

        model = mujoco.MjModel.from_xml_path(args.ik_xml)
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, defaults.keyframe)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
        viewer = mujoco.viewer.launch_passive(model, data)
        qadr = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): model.jnt_qposadr[j] for j in range(model.njnt)
        }

    period = 1.0 / args.fps
    t_end = time.perf_counter() + args.duration if args.duration > 0 else None
    last_print = 0.0
    prev_status = teleop.driver.status()
    prev_t = time.perf_counter()
    prev_action = None
    max_step = 0.0
    try:
        while t_end is None or time.perf_counter() < t_end:
            t0 = time.perf_counter()
            action = teleop.get_action()
            if prev_action is not None:
                max_step = max(max_step, max(abs(action[k] - prev_action[k]) for k in action if not k.endswith("8.pos")))
            prev_action = action

            if viewer is not None:
                if not viewer.is_running():
                    break
                sim = motor_action_to_sim_joints(action, teleop.calib)
                with viewer.lock():
                    for name, value in sim.items():
                        if name in qadr:
                            data.qpos[qadr[name]] = value
                    mujoco.mj_forward(model, data)
                viewer.sync()

            if t0 - last_print >= 1.0 / args.print_hz:
                st = teleop.driver.status()
                span = t0 - prev_t
                pkt_hz = (st["packets"] - prev_status["packets"]) / span if span > 0 else 0.0
                ik_hz = (st["solves"] - prev_status["solves"]) / span if span > 0 else 0.0
                age = "none yet" if st["packet_age_s"] is None else f"{st['packet_age_s'] * 1e3:.0f} ms"
                print(
                    f"\n[{st['state']}{(': ' + st['reason']) if st['state'] == 'PAUSED' else ''}] quest {pkt_hz:5.1f} Hz (last {age}) | IK {ik_hz:5.0f} Hz"
                    f" ({st['failed_solves']} failed) | biggest arm step between frames {max_step:.3f} rad"
                )
                for prefix in ("L", "R"):
                    vals = " ".join(f"{action[f'{prefix}J{i}.pos']:+.3f}" for i in range(1, 9))
                    print(f"  {prefix}J1..8 (motor rad): {vals}")
                prev_status, prev_t, last_print, max_step = st, t0, t0, 0.0

            remaining = period - (time.perf_counter() - t0)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
