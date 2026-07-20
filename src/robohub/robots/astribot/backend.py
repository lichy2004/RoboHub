"""Astribot SDK adapter for RoboHub observations."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from threading import Event, Lock
from typing import Any

import numpy as np

from robohub.schemas import Action, RawImage, RawObservation


_RGB_CAMERAS = {
    "head_rgbd": "head",
    "torso_rgbd": "torso",
    "left_wrist_rgbd": "wrist_left",
    "right_wrist_rgbd": "wrist_right",
}
_DEPTH_CAMERAS = {
    "head_rgbd": "head",
    "torso_rgbd": "torso",
}
_EXPECTED_JOINT_COUNT = 25


class AstribotBackend:
    """Collect standard RoboHub observations from the Astribot SDK."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
        sdk: Any | None = None,
    ) -> None:
        self.config = config or {}
        communication = self.config.get("communication", {})
        self.timeout = float(timeout if timeout is not None else communication.get("timeout", 10.0))
        self.sdk = sdk if sdk is not None else self._create_sdk()
        self._images: dict[tuple[str, str], RawImage] = {}
        self._events = {
            *(("rgb", name) for name in _RGB_CAMERAS.values()),
            *(("depth", name) for name in _DEPTH_CAMERAS.values()),
        }
        self._events = {key: Event() for key in self._events}
        self._lock = Lock()
        self._subscriptions: list[Any] = []
        self._closed = False
        self._activate_cameras()
        self._register_callbacks()

    @staticmethod
    def _create_sdk() -> Any:
        from astribot_sdk.core.astribot_api.astribot_client import Astribot

        return Astribot(node_name="robohub_astribot_backend")

    def _activate_cameras(self) -> None:
        self.sdk.activate_camera()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            info = self.sdk.get_cameras_info() or {}
            if all(info.get(camera, {}).get("activate", False) for camera in _RGB_CAMERAS):
                return
            time.sleep(0.2)
        raise TimeoutError("Timed out waiting for Astribot cameras to activate")

    def _register_callbacks(self) -> None:
        for camera in _RGB_CAMERAS:
            self._subscriptions.append(
                self.sdk.register_image_callback(
                    camera, "color", self._image_callback, need_decode=False
                )
            )
        for camera in _DEPTH_CAMERAS:
            self._subscriptions.append(
                self.sdk.register_image_callback(
                    camera, "depth", self._image_callback, need_decode=False
                )
            )

    def _image_callback(
        self,
        topic_name: str,
        msg: Any,
        width: int,
        height: int,
        array: np.ndarray | None,
    ) -> None:
        camera = self.sdk.get_camera_name_from_topic_name(topic_name)
        image_type = self.sdk.get_image_type_from_topic_name(topic_name)
        if image_type == "color" and camera in _RGB_CAMERAS:
            key = ("rgb", _RGB_CAMERAS[camera])
        elif image_type == "depth" and camera in _DEPTH_CAMERAS:
            key = ("depth", _DEPTH_CAMERAS[camera])
        else:
            return

        image = self._raw_image(msg, array, width, height)
        with self._lock:
            self._images[key] = image
            self._events[key].set()

    @staticmethod
    def _raw_image(
        msg: Any,
        array: np.ndarray | None,
        width: int,
        height: int,
    ) -> RawImage:
        payload = getattr(msg, "data", None)
        if payload is None:
            if array is None:
                raise ValueError("Astribot image callback did not provide image data")
            data = np.asarray(array, dtype=np.uint8).tobytes()
        elif isinstance(payload, bytes):
            data = payload
        else:
            data = bytes(payload)
        return RawImage(data=data, width=int(width), height=int(height))

    def _wait_for_images(self) -> None:
        deadline = time.monotonic() + self.timeout
        missing = set(self._events)
        while missing and time.monotonic() < deadline:
            missing = {key for key, event in self._events.items() if not event.is_set()}
            if missing:
                time.sleep(0.02)
        if missing:
            names = [f"{kind}/{name}" for kind, name in sorted(missing)]
            raise TimeoutError(f"Timed out waiting for Astribot images: {names}")

    @staticmethod
    def _flatten_joint_state(values: Sequence[Sequence[float]], label: str) -> np.ndarray:
        flattened = np.asarray(
            [value for group in values for value in group], dtype=np.float64
        )
        if flattened.shape != (_EXPECTED_JOINT_COUNT,):
            raise ValueError(
                f"Expected {_EXPECTED_JOINT_COUNT} {label} values, got {flattened.shape}"
            )
        return flattened

    def get_observation(self) -> RawObservation:
        self._wait_for_images()
        names = self.sdk.whole_body_names
        position = self._flatten_joint_state(
            self.sdk.get_current_joints_position(names), "joint position"
        )
        velocity = self._flatten_joint_state(
            self.sdk.get_current_joints_velocity(names), "joint velocity"
        )
        torque = self._flatten_joint_state(
            self.sdk.get_current_joints_torque(names), "joint torque"
        )
        with self._lock:
            rgb = {
                name: self._images[("rgb", name)]
                for name in _RGB_CAMERAS.values()
            }
            depth = {
                name: self._images[("depth", name)]
                for name in _DEPTH_CAMERAS.values()
            }
        return RawObservation(
            rgb=rgb,
            depth=depth,
            joints_position=position,
            joints_velocity=velocity,
            joints_torque=torque,
        )

    @staticmethod
    def _validate_action_part(
        values: np.ndarray, expected_size: int, name: str
    ) -> list[float]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (expected_size,):
            raise ValueError(
                f"Expected action.{name} shape ({expected_size},), got {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"action.{name} contains non-finite values")
        return array.tolist()

    def set_action(self, action: Action) -> None:
        if self._closed:
            raise RuntimeError("Cannot send an action after closing the Astribot backend")
        if not self.sdk.get_control_rights_status():
            raise RuntimeError("Astribot control rights are unavailable")

        names = [
            self.sdk.chassis_name,
            self.sdk.torso_name,
            self.sdk.arm_left_name,
            self.sdk.effector_left_name,
            self.sdk.arm_right_name,
            self.sdk.effector_right_name,
            self.sdk.head_name,
        ]
        commands = [
            self._validate_action_part(action.base, 3, "base"),
            self._validate_action_part(action.torso, 4, "torso"),
            self._validate_action_part(action.left_arm, 7, "left_arm"),
            self._validate_action_part(action.left_gripper, 1, "left_gripper"),
            self._validate_action_part(action.right_arm, 7, "right_arm"),
            self._validate_action_part(action.right_gripper, 1, "right_gripper"),
            self._validate_action_part(action.head, 2, "head"),
        ]
        self.sdk.set_joints_position(
            names,
            commands,
            control_way="direct",
            use_wbc=False,
            add_default_torso=False,
        )

    def close(self) -> None:
        if self._closed:
            return
        self.sdk.deactivate_camera()
        self._closed = True
