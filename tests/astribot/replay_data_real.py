#!/usr/bin/env python3
"""Replay a recorded Astribot trajectory on the real robot."""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

from robohub.communication import RobotClient
from robohub.policies.base import Policy
from robohub.robots.astribot import AstribotRobot
from robohub.schemas import Action, Observation
from robohub.utils.config import load_config

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = (
    _REPOSITORY_ROOT
    / "src"
    / "robohub"
    / "robots"
    / "astribot"
    / "configs"
    / "default.yaml"
)
_DATA_PATH = (
    _REPOSITORY_ROOT
    / "data"
    / "robot"
    / "astribot"
    / "hdf5_output_TheaExp0716"
    / "TheaExp0716_episode_0.hdf5"
)


class _ReplayPolicy(Policy):
    def infer(self, observation: Observation) -> Action:
        del observation
        raise RuntimeError("ReplayPolicy does not support inference")


def _load_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as file:
        qpos = np.asarray(
            file["joints_dict/joints_position_command"],
            dtype=np.float64,
        )
        timestamps = np.asarray(
            file["joints_dict/command_timestamp"],
            dtype=np.float64,
        )

    if qpos.ndim != 2 or qpos.shape[1] != 25:
        raise ValueError(f"Expected trajectory shape (N, 25), got {qpos.shape}")
    if len(qpos) == 0:
        raise ValueError("Trajectory must contain at least one frame")
    if timestamps.shape != (qpos.shape[0],):
        raise ValueError(
            f"Expected timestamps shape ({qpos.shape[0]},), got {timestamps.shape}"
        )
    if not np.all(np.isfinite(qpos)):
        raise ValueError("Trajectory contains non-finite joint positions")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("Trajectory contains non-finite timestamps")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("Trajectory timestamps must be strictly increasing")
    return qpos, timestamps


def _recorded_to_robohub_qpos(qpos: np.ndarray) -> np.ndarray:
    qpos_array = np.asarray(qpos, dtype=np.float64)
    if qpos_array.ndim != 2 or qpos_array.shape[1] != 25:
        raise ValueError(f"Expected trajectory shape (N, 25), got {qpos_array.shape}")
    return np.concatenate(
        (
            qpos_array[:, 7:14],
            qpos_array[:, 14:15],
            qpos_array[:, 15:22],
            qpos_array[:, 22:23],
            qpos_array[:, 23:25],
            qpos_array[:, 3:7],
            qpos_array[:, 0:3],
        ),
        axis=1,
    )


def _replay_trajectory(
    robot: AstribotRobot,
    qpos: np.ndarray,
    timestamps: np.ndarray,
    replay_speed: float,
) -> None:
    current_qpos = np.asarray(
        robot.get_observation().joints_position,
        dtype=np.float64,
    )
    robot.set_action_interpolated(current_qpos, qpos[0])
    print("Robot moved to the first trajectory frame")

    replay_started = time.perf_counter()
    first_timestamp = timestamps[0]
    for index in range(1, len(qpos)):
        deadline = replay_started + (timestamps[index] - first_timestamp) / replay_speed
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)

        robot.set_action(robot.qpos_to_action(qpos[index]))
        if index % 100 == 0 or index + 1 == len(qpos):
            print(f"Replayed action {index + 1}/{len(qpos)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Astribot HDF5 data on the real robot"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Robot server IP or hostname",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Robot server port",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="YAML config path",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=_DATA_PATH,
        help="HDF5 trajectory path",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="Trajectory replay speed multiplier",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replay_speed <= 0.0:
        raise ValueError("replay-speed must be positive")

    recorded_qpos, timestamps = _load_trajectory(args.data)
    qpos = _recorded_to_robohub_qpos(recorded_qpos)
    duration = timestamps[-1] - timestamps[0]
    print(f"Loaded {len(qpos)} actions from {args.data} ({duration:.3f} s)")

    config = load_config(args.config)
    client = RobotClient(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
    )
    robot = AstribotRobot(client, _ReplayPolicy(), config)
    try:
        robot.reset()
        print("Robot reset completed")
        _replay_trajectory(robot, qpos, timestamps, args.replay_speed)
        print("Trajectory replay completed")
    except KeyboardInterrupt:
        print("Trajectory replay interrupted")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
