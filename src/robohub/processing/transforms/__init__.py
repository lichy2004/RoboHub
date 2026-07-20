"""Coordinate transform utilities."""

import numpy as np


def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """Transform 3D points with a homogeneous transform."""
    points_array = np.asarray(points)
    transform_array = np.asarray(transform, dtype=np.float64)

    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points_array.shape}")
    if transform_array.shape != (4, 4):
        raise ValueError(
            f"Expected transform shape (4, 4), got {transform_array.shape}"
        )
    if not np.all(np.isfinite(transform_array)):
        raise ValueError("Transform contains non-finite values")

    transformed = points_array @ transform_array[:3, :3].T + transform_array[:3, 3]
    return transformed.astype(points_array.dtype, copy=False)


__all__ = ["transform_points"]
