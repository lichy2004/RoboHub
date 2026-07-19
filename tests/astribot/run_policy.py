"""
Run a fake policy against the Astribot RoboHub server.

conda activate robohub_policy
python tests/astribot/run_policy.py \
    --host 10.40.5.70 \
    --port 8765
"""

import argparse
import time
from pathlib import Path

import numpy as np

from robohub.communication import RobotClient
from robohub.policies.base import Policy
from robohub.robots.astribot import AstribotRobot
from robohub.schemas import Action, Observation
from robohub.utils.config import load_config


_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "robohub"
    / "robots"
    / "astribot"
    / "configs"
    / "default.yaml"
)


class FakePolicy(Policy):
    """Return a zero joint-position action with Astribot dimensions."""

    def __init__(self, action_value: float = 0.0) -> None:
        self.action_value = action_value

    def infer(self, observation: Observation) -> Action:
        del observation
        value = self.action_value
        return Action(
            left_arm=np.full(7, value, dtype=np.float64),
            left_gripper=np.full(1, value, dtype=np.float64),
            right_arm=np.full(7, value, dtype=np.float64),
            right_gripper=np.full(1, value, dtype=np.float64),
            torso=np.full(4, value, dtype=np.float64),
            head=np.full(2, value, dtype=np.float64),
            base=np.full(3, value, dtype=np.float64),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fake Astribot policy")
    parser.add_argument("--host", default="127.0.0.1", help="Robot server IP or hostname")
    parser.add_argument("--port", type=int, default=8765, help="Robot server port")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout")
    parser.add_argument("--action-value", type=float, default=0.0, help="Constant action value")
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH, help="YAML config path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    client = RobotClient(host=args.host, port=args.port, timeout=args.timeout)
    policy = FakePolicy(args.action_value)
    robot = AstribotRobot(client, policy, config)
    try:
        start_time = time.perf_counter()
        observation = robot.get_observation()
        elapsed_time = time.perf_counter() - start_time
        action = policy.infer(observation)
        # robot.set_action(action)
        print(f"Observation received in {elapsed_time:.3f} s ({elapsed_time * 1000:.1f} ms)")
        print(f"RGB shapes: { {name: image.shape for name, image in observation.rgb.items()} }")
        print(f"Depth shapes: { {name: image.shape for name, image in observation.depth.items()} }")
        print(f"Joint shapes: position={observation.joints_position.shape}, "
              f"velocity={observation.joints_velocity.shape}, "
              f"torque={observation.joints_torque.shape}")
        print("Fake action sent successfully")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
