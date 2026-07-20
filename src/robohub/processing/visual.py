"""Incremental Astribot visualization with Viser."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_CAMERA_LINKS = {
    "head": "head_rgbd",
    "torso": "astribot_torso_link_4",
    "wrist_left": "left_wrist_rgbd",
    "wrist_right": "right_wrist_rgbd",
}
_CAMERA_RPY_OFFSETS_DEG = {
    "head": (0.0, 180.0, 0.0),
    "torso": (90.0, 180.0, 90.0),
    "wrist_left": (0.0, 180.0, 0.0),
    "wrist_right": (0.0, 180.0, 0.0),
}
_DEFAULT_MESH_DIR = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "astribot"
    / "urdf"
    / "astribot_s1_urdf"
)
_DEFAULT_URDF_PATH = _DEFAULT_MESH_DIR / "astribot_whole_body_maniskill.urdf"


def _rpy_to_rotation_matrix(
    rpy: tuple[float, float, float],
) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    rotation_y = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rotation_z = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rotation_z @ rotation_y @ rotation_x


def _rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            (
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            )
        )
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        next_index = (diagonal_index + 1) % 3
        last_index = (diagonal_index + 2) % 3
        scale = (
            np.sqrt(
                1.0
                + matrix[diagonal_index, diagonal_index]
                - matrix[next_index, next_index]
                - matrix[last_index, last_index]
            )
            * 2.0
        )
        xyz = np.zeros(3, dtype=np.float64)
        xyz[diagonal_index] = 0.25 * scale
        xyz[next_index] = (
            matrix[next_index, diagonal_index] + matrix[diagonal_index, next_index]
        ) / scale
        xyz[last_index] = (
            matrix[last_index, diagonal_index] + matrix[diagonal_index, last_index]
        ) / scale
        w = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
        quaternion = np.concatenate(([w], xyz))
    return quaternion / np.linalg.norm(quaternion)


class Visual:
    """Visualize Astribot joints, RGB cameras, and a point cloud."""

    def __init__(
        self,
        urdf_path: str | Path = _DEFAULT_URDF_PATH,
        mesh_dir: str | Path | None = _DEFAULT_MESH_DIR,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        camera_fov_deg: float = 75.0,
        camera_scale: float = 0.1,
        point_size: float = 0.006,
    ) -> None:
        try:
            import viser
            import yourdfpy
            from viser.extras import ViserUrdf
        except ImportError as error:
            raise ImportError(
                "Visual requires the 'astribot-visualization' optional dependencies"
            ) from error

        urdf_path = Path(urdf_path)
        mesh_path = urdf_path.parent if mesh_dir is None else Path(mesh_dir)
        self.urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_path)
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.add_grid("/ground", width=2, height=2)
        self.urdf_visual = ViserUrdf(
            self.server,
            self.urdf,
            root_node_name="/base",
        )
        self.camera_fov = np.deg2rad(camera_fov_deg)
        self.camera_scale = camera_scale
        self.point_size = point_size
        self._camera_handles: dict[str, Any] = {}
        self._point_cloud_handle: Any | None = None

    def _set_joint(
        self,
        configuration: dict[str, float],
        joint_name: str,
        value: float,
    ) -> None:
        if joint_name not in self.urdf.joint_map:
            return
        joint = self.urdf.joint_map[joint_name]
        if joint.limit is not None:
            lower = -np.inf if joint.limit.lower is None else float(joint.limit.lower)
            upper = np.inf if joint.limit.upper is None else float(joint.limit.upper)
            value = float(np.clip(value, lower, upper))
        configuration[joint_name] = value

    def _joint_configuration(
        self,
        joints_position: np.ndarray,
    ) -> np.ndarray:
        qpos = np.asarray(joints_position, dtype=np.float64)
        if qpos.shape != (25,):
            raise ValueError(f"Expected joints_position shape (25,), got {qpos.shape}")
        if not np.all(np.isfinite(qpos)):
            raise ValueError("joints_position contains non-finite values")

        configuration: dict[str, float] = {}
        for index in range(7):
            self._set_joint(
                configuration,
                f"astribot_arm_left_joint_{index + 1}",
                float(qpos[index]),
            )
            self._set_joint(
                configuration,
                f"astribot_arm_right_joint_{index + 1}",
                float(qpos[index + 8]),
            )
        for index in range(2):
            self._set_joint(
                configuration,
                f"astribot_head_joint_{index + 1}",
                float(qpos[index + 16]),
            )
        for index in range(4):
            self._set_joint(
                configuration,
                f"astribot_torso_joint_{index + 1}",
                float(qpos[index + 18]),
            )
        return np.asarray(
            [
                configuration.get(joint_name, 0.0)
                for joint_name in self.urdf.actuated_joint_names
            ],
            dtype=np.float64,
        )

    def _camera_pose(
        self,
        camera_name: str,
        link_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        transform = self.urdf.get_transform(link_name).copy()
        offset = _rpy_to_rotation_matrix(
            tuple(np.deg2rad(_CAMERA_RPY_OFFSETS_DEG[camera_name]))
        )
        rotation = transform[:3, :3] @ offset
        return transform[:3, 3], _rotation_matrix_to_wxyz(rotation)

    def _update_cameras(
        self,
        rgb: Mapping[str, np.ndarray],
    ) -> None:
        for camera_name, image in rgb.items():
            if camera_name not in _CAMERA_LINKS:
                continue
            link_name = _CAMERA_LINKS[camera_name]
            if link_name not in self.urdf.link_map:
                continue

            image_array = np.asarray(image)
            if image_array.ndim != 3 or image_array.shape[2] != 3:
                raise ValueError(
                    f"Expected rgb[{camera_name!r}] shape (H, W, 3), "
                    f"got {image_array.shape}"
                )
            image_array = np.ascontiguousarray(image_array)
            position, wxyz = self._camera_pose(camera_name, link_name)
            handle = self._camera_handles.get(camera_name)
            if handle is None:
                handle = self.server.scene.add_camera_frustum(
                    f"/camera_frustums/{camera_name}",
                    fov=self.camera_fov,
                    aspect=image_array.shape[1] / image_array.shape[0],
                    scale=self.camera_scale,
                    line_width=2.0,
                    image=image_array,
                    wxyz=wxyz,
                    position=position,
                    variant="wireframe",
                )
                self._camera_handles[camera_name] = handle
            else:
                handle.image = image_array
                handle.position = position
                handle.wxyz = wxyz

    def _update_point_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        points_array = np.asarray(points, dtype=np.float32)
        colors_array = np.asarray(colors)
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            raise ValueError(
                f"Expected point_cloud shape (N, 3), got {points_array.shape}"
            )
        if colors_array.shape != points_array.shape:
            raise ValueError(
                "point_cloud_colors must have the same shape as point_cloud"
            )

        points_array = np.ascontiguousarray(points_array)
        colors_array = np.ascontiguousarray(colors_array)
        if self._point_cloud_handle is None:
            self._point_cloud_handle = self.server.scene.add_point_cloud(
                "/point_clouds/rgbd",
                points=points_array,
                colors=colors_array,
                point_size=self.point_size,
                point_shape="circle",
            )
        else:
            self._point_cloud_handle.points = points_array
            self._point_cloud_handle.colors = colors_array

    def update(
        self,
        joints_position: np.ndarray | None = None,
        rgb: Mapping[str, np.ndarray] | None = None,
        point_cloud: np.ndarray | None = None,
        point_cloud_colors: np.ndarray | None = None,
    ) -> None:
        """Update the robot pose and latest sensor visualizations."""
        if joints_position is not None:
            configuration = self._joint_configuration(joints_position)
            self.urdf.update_cfg(configuration)
            self.urdf_visual.update_cfg(configuration)
        if rgb is not None:
            self._update_cameras(rgb)

        if (point_cloud is None) != (point_cloud_colors is None):
            raise ValueError(
                "point_cloud and point_cloud_colors must be provided together"
            )
        if point_cloud is not None and point_cloud_colors is not None:
            self._update_point_cloud(
                point_cloud,
                point_cloud_colors,
            )

    def close(self) -> None:
        """Stop the Viser server."""
        self.server.stop()
