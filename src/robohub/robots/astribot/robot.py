"""Workstation-side Astribot workflow and action validation."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from robohub.communication import RobotClient
from robohub.policies import Policy
from robohub.robots.base import Robot
from robohub.schemas import Action, Observation, RawImage, RawObservation


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
            rgb={
                name: cls._decode_rgb(image)
                for name, image in raw.rgb.items()
            },
            depth={
                name: cls._decode_depth(
                    image, *cls._depth_shape(config, name, image)
                )
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
            config.get("robot", {})
            .get("cameras", {})
            .get("depth", {})
            .get(name, {})
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
            depth = cls._decode_raw_depth(
                payload, expected_height, expected_width
            )
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
                    return payload[header_size:].view(dtype).reshape(
                        height, width
                    )
        raise ValueError(
            f"Unsupported depth payload size {payload.nbytes} "
            f"for {width}x{height}"
        )

    def set_action(self, action: Action) -> None:
        self.client.set_action(action)

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
