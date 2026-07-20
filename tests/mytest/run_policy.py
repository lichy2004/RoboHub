"""
Run the workstation-side policy client for cross-machine testing.

conda activate robohub_policy
python tests/mytest/run_policy.py \
    --host 10.40.5.70 \
    --port 8765
"""

import argparse

from robohub.communication import RobotClient
from robohub.policies.my_policy import MyPolicy
from robohub.robots.my_robot import MyRobot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RoboHub policy client")
    parser.add_argument(
        "--host", required=True, help="Robot machine IP address or hostname"
    )
    parser.add_argument("--port", type=int, default=8765, help="Robot server TCP port")
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Request timeout in seconds"
    )
    parser.add_argument(
        "--action-value", type=float, default=0.5, help="Constant test action value"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = RobotClient(host=args.host, port=args.port, timeout=args.timeout)
    policy = MyPolicy({"action_value": args.action_value, "arm_size": 2})
    robot = MyRobot(client, policy)

    try:
        observation = robot.get_observation()
        action = policy.infer(observation)
        robot.set_action(action)

        image_shapes = {name: image.shape for name, image in observation.rgb.items()}
        depth_shapes = {name: depth.shape for name, depth in observation.depth.items()}
        print(f"RGB shapes: {image_shapes}")
        print(f"Depth shapes: {depth_shapes}")
        print(f"Action sent successfully: left_arm={action.left_arm}")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
