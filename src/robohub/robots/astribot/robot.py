"""Workstation-side Astribot workflow and action validation."""

from collections.abc import Mapping
from pathlib import Path
from time import sleep
from typing import Any

import numpy as np

from robohub.communication import RobotClient
from robohub.policies import Policy
from robohub.robots.base import Robot
from robohub.schemas import Action, Observation, RawImage, RawObservation

_DEFAULT_URDF_PATH = (
    Path(__file__).resolve().parents[4]
    / "assets"
    / "astribot"
    / "urdf"
    / "astribot_s1_urdf"
    / "astribot_whole_body_maniskill.urdf"
)


class AstribotRobot(Robot):
    """Coordinate an Astribot client and policy on the workstation."""

    def __init__(
        self,
        client: RobotClient,
        policy: Policy,
        config: Mapping[str, Any],
    ) -> None:
        self.client = client
        self.policy = policy
        self.config = config
        self._urdf: Any | None = None

    def _get_urdf(self) -> Any:
        if self._urdf is None:
            try:
                import yourdfpy
            except ImportError as error:
                raise ImportError(
                    "Astribot forward kinematics requires yourdfpy"
                ) from error
            self._urdf = yourdfpy.URDF.load(
                _DEFAULT_URDF_PATH,
                mesh_dir=_DEFAULT_URDF_PATH.parent,
            )
        return self._urdf

    @staticmethod
    def _set_urdf_joint(
        urdf: Any,
        configuration: dict[str, float],
        joint_name: str,
        value: float,
    ) -> None:
        if joint_name not in urdf.joint_map:
            return
        joint = urdf.joint_map[joint_name]
        if joint.limit is not None:
            lower = -np.inf if joint.limit.lower is None else float(joint.limit.lower)
            upper = np.inf if joint.limit.upper is None else float(joint.limit.upper)
            value = float(np.clip(value, lower, upper))
        configuration[joint_name] = value

    @classmethod
    def _urdf_configuration(
        cls,
        urdf: Any,
        joints_position: np.ndarray,
    ) -> np.ndarray:
        qpos = np.asarray(joints_position, dtype=np.float64)
        if qpos.shape != (25,):
            raise ValueError(f"Expected joints_position shape (25,), got {qpos.shape}")
        if not np.all(np.isfinite(qpos)):
            raise ValueError("joints_position contains non-finite values")

        configuration: dict[str, float] = {}
        for index in range(7):
            cls._set_urdf_joint(
                urdf,
                configuration,
                f"astribot_arm_left_joint_{index + 1}",
                float(qpos[index]),
            )
            cls._set_urdf_joint(
                urdf,
                configuration,
                f"astribot_arm_right_joint_{index + 1}",
                float(qpos[index + 8]),
            )
        for index in range(2):
            cls._set_urdf_joint(
                urdf,
                configuration,
                f"astribot_head_joint_{index + 1}",
                float(qpos[index + 16]),
            )
        for index in range(4):
            cls._set_urdf_joint(
                urdf,
                configuration,
                f"astribot_torso_joint_{index + 1}",
                float(qpos[index + 18]),
            )
        return np.asarray(
            [
                configuration.get(joint_name, 0.0)
                for joint_name in urdf.actuated_joint_names
            ],
            dtype=np.float64,
        )

    def forward_kinematics(
        self,
        joints_position: np.ndarray,
        link_name: str,
    ) -> np.ndarray:
        """Return the link pose in the robot base coordinate frame."""
        urdf = self._get_urdf()
        if link_name not in urdf.link_map:
            raise ValueError(f"Unknown Astribot URDF link {link_name!r}")
        configuration = self._urdf_configuration(urdf, joints_position)
        urdf.update_cfg(configuration)
        return np.asarray(
            urdf.get_transform(link_name),
            dtype=np.float64,
        ).copy()

    def get_observation(self) -> Observation:
        raw = self.client.get_observation(RawObservation)
        return self.decode_observation(raw, self.config)

    @classmethod
    def decode_observation(
        cls,
        raw: RawObservation,
        config: Mapping[str, Any],
    ) -> Observation:
        return Observation(
            rgb={name: cls._decode_rgb(image) for name, image in raw.rgb.items()},
            depth={
                name: cls._decode_depth(image, *cls._depth_shape(config, name, image))
                for name, image in raw.depth.items()
            },
            joints_position=raw.joints_position,
            joints_velocity=raw.joints_velocity,
            joints_torque=raw.joints_torque,
        )

    @staticmethod
    def _decode_rgb(image: RawImage) -> np.ndarray:
        import cv2

        payload = np.frombuffer(image.data, dtype=np.uint8)
        bgr = cv2.imdecode(payload, cv2.IMREAD_COLOR)
        if bgr is None and payload.size == image.height * image.width * 3:
            bgr = payload.reshape(image.height, image.width, 3)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(
                f"Failed to decode Astribot RGB image with shape "
                f"({image.height}, {image.width}, 3)"
            )
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _depth_shape(
        config: Mapping[str, Any],
        name: str,
        image: RawImage,
    ) -> tuple[int, int]:
        depth_config = (
            config.get("robot", {}).get("cameras", {}).get("depth", {}).get(name, {})
        )
        resolution = depth_config.get("resolution")
        if resolution is not None and len(resolution) == 2:
            width, height = (int(value) for value in resolution)
        else:
            width, height = image.width, image.height
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid depth resolution for camera {name}")
        return height, width

    @classmethod
    def _decode_depth(
        cls,
        image: RawImage,
        expected_height: int,
        expected_width: int,
    ) -> np.ndarray:
        import cv2

        payload = np.frombuffer(image.data, dtype=np.uint8)
        depth = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
        if depth is None:
            depth = cls._decode_raw_depth(payload, expected_height, expected_width)
        elif depth.ndim == 3:
            depth = depth[..., 0]

        if depth.shape != (expected_height, expected_width):
            if depth.size == expected_height * expected_width:
                depth = depth.reshape(expected_height, expected_width)
            else:
                depth = cv2.resize(
                    depth,
                    (expected_width, expected_height),
                    interpolation=cv2.INTER_NEAREST,
                )
        return np.ascontiguousarray(depth[..., np.newaxis])

    @staticmethod
    def _decode_raw_depth(
        payload: np.ndarray,
        height: int,
        width: int,
    ) -> np.ndarray:
        pixel_count = height * width
        for dtype in (np.uint16, np.float32, np.uint8):
            itemsize = np.dtype(dtype).itemsize
            if payload.nbytes == pixel_count * itemsize:
                return payload.view(dtype).reshape(height, width)
        for header_size in (12, 16, 24, 32):
            remaining = payload.nbytes - header_size
            for dtype in (np.uint16, np.float32):
                if remaining == pixel_count * np.dtype(dtype).itemsize:
                    return payload[header_size:].view(dtype).reshape(height, width)
        raise ValueError(
            f"Unsupported depth payload size {payload.nbytes} for {width}x{height}"
        )

    def set_action(self, action: Action) -> None:
        self.client.set_action(action)

    def set_action_interpolated(
        self,
        current_qpos: np.ndarray,
        target_qpos: np.ndarray,
    ) -> None:
        current = np.asarray(current_qpos, dtype=np.float64)
        target = np.asarray(target_qpos, dtype=np.float64)
        reset_config = self.config["robot"]["reset"]
        interpolation_steps = int(reset_config["interpolation_steps"])
        interpolation_period = float(reset_config["interpolation_period_seconds"])

        delta = target - current
        for step in range(1, interpolation_steps + 1):
            positions = current + delta * (step / interpolation_steps)
            self.set_action(self.qpos_to_action(positions))
            if step < interpolation_steps:
                sleep(interpolation_period)

    def reset(self) -> None:
        current = np.asarray(self.get_observation().joints_position, dtype=np.float64)
        target = np.asarray(self.config["robot"]["default_action"], dtype=np.float64)

        self.policy.reset()
        self.set_action_interpolated(current, target)

    @staticmethod
    def action_to_qpos(action: Action) -> np.ndarray:
        return np.concatenate(
            (
                action.left_arm,
                action.left_gripper,
                action.right_arm,
                action.right_gripper,
                action.head,
                action.torso,
                action.base,
            )
        )

    @staticmethod
    def qpos_to_action(qpos: np.ndarray) -> Action:
        return Action(
            left_arm=qpos[0:7],
            left_gripper=qpos[7:8],
            right_arm=qpos[8:15],
            right_gripper=qpos[15:16],
            head=qpos[16:18],
            torso=qpos[18:22],
            base=qpos[22:25],
        )

    def step(self) -> Action:
        observation = self.get_observation()
        action = self.policy.infer(observation)
        self.set_action(action)
        return action

    def close(self) -> None:
        try:
            self.policy.close()
        finally:
            self.client.close()
