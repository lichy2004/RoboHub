"""Generate a colorized point cloud from aligned RGB-D observations."""

import numpy as np

from robohub.processing.transforms import transform_points


def _validate_intrinsics(
    intrinsics: np.ndarray,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected {name} shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError(f"{name} focal lengths must be positive")
    return matrix


def get_point_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    color_intrinsics: np.ndarray,
    depth_intrinsics: np.ndarray,
    depth_to_color: np.ndarray,
    *,
    depth_scale: float = 0.001,
    stride: int = 1,
    min_depth: float = 0.0,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return color-camera-frame XYZ points and their RGB colors."""
    rgb_array = np.asarray(rgb)
    depth_array = np.asarray(depth)
    if depth_array.ndim == 3 and depth_array.shape[2] == 1:
        depth_array = depth_array[..., 0]

    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected rgb shape (H, W, 3), got {rgb_array.shape}")
    if depth_array.ndim != 2:
        raise ValueError(
            f"Expected depth shape (H, W) or (H, W, 1), got {depth_array.shape}"
        )
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("stride must be a positive integer")
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth_scale must be positive and finite")
    if not np.isfinite(min_depth) or min_depth < 0.0:
        raise ValueError("min_depth must be non-negative and finite")
    if max_depth is not None and (not np.isfinite(max_depth) or max_depth <= min_depth):
        raise ValueError("max_depth must be finite and greater than min_depth")

    color_matrix = _validate_intrinsics(color_intrinsics, "color_intrinsics")
    depth_matrix = _validate_intrinsics(depth_intrinsics, "depth_intrinsics")
    depth_to_color_array = np.asarray(depth_to_color, dtype=np.float64)
    if depth_to_color_array.shape != (4, 4):
        raise ValueError(
            f"Expected depth_to_color shape (4, 4), got {depth_to_color_array.shape}"
        )

    rows, columns = np.indices(depth_array.shape)
    u = columns[::stride, ::stride].reshape(-1).astype(np.float64)
    v = rows[::stride, ::stride].reshape(-1).astype(np.float64)
    z = depth_array[::stride, ::stride].reshape(-1).astype(np.float64) * depth_scale

    valid = np.isfinite(z) & (z > min_depth)
    if max_depth is not None:
        valid &= z < max_depth
    u, v, z = u[valid], v[valid], z[valid]

    x = (u - depth_matrix[0, 2]) * z / depth_matrix[0, 0]
    y = (v - depth_matrix[1, 2]) * z / depth_matrix[1, 1]
    points_depth = np.column_stack((x, y, z))
    points_color = transform_points(points_depth, depth_to_color_array)

    projected_z = points_color[:, 2]
    in_front = np.isfinite(projected_z) & (projected_z > 0.0)
    points_color = points_color[in_front]
    projected_z = projected_z[in_front]

    color_u = np.rint(
        color_matrix[0, 0] * points_color[:, 0] / projected_z + color_matrix[0, 2]
    ).astype(np.int64)
    color_v = np.rint(
        color_matrix[1, 1] * points_color[:, 1] / projected_z + color_matrix[1, 2]
    ).astype(np.int64)
    in_image = (
        (color_u >= 0)
        & (color_u < rgb_array.shape[1])
        & (color_v >= 0)
        & (color_v < rgb_array.shape[0])
    )

    points = np.ascontiguousarray(points_color[in_image].astype(np.float32))
    colors = np.ascontiguousarray(rgb_array[color_v[in_image], color_u[in_image]])
    return points, colors
