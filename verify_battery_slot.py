"""Check physical insertion in the live battery sim server using two drop tests.

Start ``lerobot_sim_server.py --task Isaac-InsertBattery-OpenArm-IK-Abs-v0`` first.
The aligned battery must settle below the lid; a battery on the solid lid must remain above it.
"""

import argparse

from lerobot_robot_openarm_umeow.sim_link import SimClient

CHARGER_XY = (0.325, 0.070)
SLOT_XY = (-0.011, 0.0215)
PAD_Z = 0.280
CHARGER_HEIGHT = 0.0375
BATTERY_RADIUS = 0.00725
SLOT_DEPTH = 0.009


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5710)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    client = SimClient(port=args.port)
    hello = client.connect()
    if hello["task"] != "Isaac-InsertBattery-OpenArm-IK-Abs-v0":
        raise SystemExit(f"Wrong server task: {hello['task']}")
    results = {}
    for label, offset in (("slot", SLOT_XY), ("lid", (0.025, -0.015))):
        x, y = CHARGER_XY[0] + offset[0], CHARGER_XY[1] + offset[1]
        start_z = PAD_Z + CHARGER_HEIGHT + BATTERY_RADIUS + 0.012
        obs = client.request("reset", joints=None, objects={
            "charger": [*CHARGER_XY, 0.0], "battery": [x, y, start_z, 0.0],
        })
        joints = obs["joints"]
        for _ in range(args.steps):
            obs = client.request("step", joints=joints)
        charger_z = obs["objects"]["charger"][2]
        battery_z = obs["objects"]["battery"][2]
        results[label] = battery_z - charger_z
        print(f"{label}: battery centre {results[label]:.4f} m above charger centre", flush=True)
    client.close()
    expected = CHARGER_HEIGHT / 2 - SLOT_DEPTH + BATTERY_RADIUS
    if abs(results["slot"] - expected) >= 0.004 or results["lid"] - results["slot"] <= 0.005:
        raise SystemExit(f"FAIL: {results}, expected slot centre {expected:.4f} m")
    print("PASS: a real contact simulation seats the battery in the open bay", flush=True)


if __name__ == "__main__":
    main()
