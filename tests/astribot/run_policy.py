"""Visualize live observations from the Astribot RoboHub server.

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
from robohub.processing.point_cloud import get_point_cloud
from robohub.processing.transforms import transform_points
from robohub.processing.visual import Visual
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
    parser = argparse.ArgumentParser(description="Visualize live Astribot observations")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Robot server IP or hostname",
    )
    parser.add_argument("--port", type=int, default=8765, help="Robot server port")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout")
    parser.add_argument(
        "--visual-host",
        default="0.0.0.0",
        help="Viser server host",
    )
    parser.add_argument(
        "--visual-port",
        type=int,
        default=8080,
        help="Viser server port",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="YAML config path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cameras_config = config["robot"]["cameras"]
    head_rgb_config = cameras_config["rgb"]["head"]
    head_depth_config = cameras_config["depth"]["head"]

    color_intrinsics = np.asarray(
        head_rgb_config["intrinsic"]["camera_matrix"],
        dtype=np.float64,
    )
    depth_intrinsics = np.asarray(
        head_depth_config["intrinsic"]["camera_matrix"],
        dtype=np.float64,
    )
    depth_to_color = np.asarray(
        head_depth_config["depth_to_color"],
        dtype=np.float64,
    )
    camera_to_head = np.asarray(
        head_rgb_config["extrinsic"]["transform"],
        dtype=np.float64,
    )
    head_link = head_rgb_config["extrinsic"]["parent_frame"]

    client = RobotClient(host=args.host, port=args.port, timeout=args.timeout)
    policy = FakePolicy()
    robot = AstribotRobot(client, policy, config)
    visual: Visual | None = None
    try:
        visual = Visual(
            host=args.visual_host,
            port=args.visual_port,
            camera_fov_deg=float(head_rgb_config["fov_deg"]),
        )
        print(
            f"Visualizing Astribot observations at "
            f"http://{args.visual_host}:{args.visual_port}"
        )
        frame_count = 0
        while True:
            cycle_started = time.perf_counter()
            observation = robot.get_observation()
            visual.update(
                observation.joints_position,
                observation.rgb,
            )

            points_camera, colors = get_point_cloud(
                observation.rgb["head"],
                observation.depth["head"],
                color_intrinsics,
                depth_intrinsics,
                depth_to_color,
                depth_scale=float(head_depth_config["depth_scale"]),
                stride=int(head_depth_config["point_cloud_stride"]),
                min_depth=float(head_depth_config["min_depth"]),
                max_depth=float(head_depth_config["max_depth"]),
            )
            point_count = points_camera.shape[0]
            if point_count:
                points_head = transform_points(points_camera, camera_to_head)
                head_to_base = robot.forward_kinematics(
                    observation.joints_position,
                    head_link,
                )
                points_base = transform_points(points_head, head_to_base)
                visual.update(
                    point_cloud=points_base,
                    point_cloud_colors=colors,
                )

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.perf_counter() - cycle_started
                print(
                    f"Frame {frame_count}: {elapsed_time * 1000:.1f} ms, "
                    f"{point_count} points"
                )
    except KeyboardInterrupt:
        print("Visualization stopped.")
    finally:
        if visual is not None:
            visual.close()
        robot.close()


if __name__ == "__main__":
    main()
