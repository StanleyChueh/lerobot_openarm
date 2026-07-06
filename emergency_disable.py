#!/usr/bin/env python
"""Minimal panic-button script: register motors, then disable_all(). Nothing else.

Does not enable, does not read positions, does not send any MIT command. The
only CAN traffic this sends is the motor-registration setup (local, no bus
effect) and the disable frame itself. Use when an arm is armed/enabled in an
unknown state and you need to de-energize it with the least possible
additional risk. A physical power cut / E-stop is still preferable if one is
available -- use this only when it isn't.
"""

import argparse

import openarm_can as oa

MOTOR_TYPES = [
    oa.MotorType.DM8009, oa.MotorType.DM8009,
    oa.MotorType.DM4340, oa.MotorType.DM4340,
    oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310,
]
SEND_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
RECV_IDS = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
GRIPPER_MOTOR_TYPE = oa.MotorType.DM4310
GRIPPER_SEND_ID = 0x08
GRIPPER_RECV_ID = 0x18

DEFAULT_PORT_FOR_SIDE = {"left": "can3", "right": "can2"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--port", type=str, default=None)
    args = parser.parse_args()
    port = args.port or DEFAULT_PORT_FOR_SIDE[args.side]

    print(f"Disabling {args.side} arm on {port}...")
    arm = oa.OpenArm(port, True)
    arm.init_arm_motors(MOTOR_TYPES, SEND_IDS, RECV_IDS)
    arm.init_gripper_motor(GRIPPER_MOTOR_TYPE, GRIPPER_SEND_ID, GRIPPER_RECV_ID)
    arm.disable_all()
    print("disable_all() sent. Please visually confirm the arm's status LEDs actually went red/off --"
          " do not assume this succeeded just because the script didn't error.")


if __name__ == "__main__":
    main()
