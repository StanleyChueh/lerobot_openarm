import time
import argparse

from robots.umeow_openarm_follower import OpenArmFollowerConfig, OpenArmFollower


def to_float(x):
    try:
        return float(x.item())
    except AttributeError:
        return float(x)


def ramp_joint(robot, base_action, joint_name, target_value, steps=30, dt=0.03):
    start_value = to_float(base_action[joint_name])

    for i in range(1, steps + 1):
        alpha = i / steps
        cmd = dict(base_action)
        cmd[joint_name] = start_value + alpha * (target_value - start_value)
        robot.send_action(cmd)
        time.sleep(dt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", type=str, default="LJ1.pos")
    parser.add_argument("--step", type=float, default=3.0, help="degrees")
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    robot_config = OpenArmFollowerConfig(
        right_port="can2",
        left_port="can3",
        enable_fd=True,
        model_path="/home/csl/lerobot_openarm/model/openarm_description_leader.urdf",
    )

    robot = OpenArmFollower(robot_config)
    robot.connect()

    left_joints = [
        "LJ1.pos",
        "LJ2.pos",
        "LJ3.pos",
        "LJ4.pos",
        "LJ5.pos",
        "LJ6.pos",
        "LJ7.pos",
    ]

    try:
        obs = robot.get_observation()
        print("Current observation:")
        for k in left_joints + ["LJ8.pos"]:
            print(k, obs.get(k))

        if args.all:
            test_joints = left_joints
        else:
            test_joints = [args.joint]

        for joint_name in test_joints:
            obs = robot.get_observation()
            base_action = dict(obs)

            if joint_name not in base_action:
                print(f"Joint not found: {joint_name}")
                print("Available keys:", list(base_action.keys()))
                return

            current = to_float(base_action[joint_name])
            target = current + args.step

            print("\n================================")
            print(f"Testing {joint_name}")
            print(f"Current: {current:.4f}")
            print(f"Target : {target:.4f}")
            print("Press Enter to move, or Ctrl+C to stop.")
            input()

            print(f"Moving {joint_name} to {target:.4f}")
            ramp_joint(robot, base_action, joint_name, target)
            time.sleep(args.hold)

            obs_after = robot.get_observation()
            print(f"Observed after move: {joint_name} = {obs_after.get(joint_name)}")

            print(f"Returning {joint_name} to {current:.4f}")
            base_after = dict(robot.get_observation())
            ramp_joint(robot, base_after, joint_name, current)
            time.sleep(args.hold)

            obs_return = robot.get_observation()
            print(f"Observed after return: {joint_name} = {obs_return.get(joint_name)}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
