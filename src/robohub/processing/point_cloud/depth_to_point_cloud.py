"""Depth image to point cloud conversion boundary."""

import numpy as np


def depth_to_point_cloud(
    depth: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Convert depth pixels to camera-frame XYZ points."""
    raise NotImplementedError(
        "Depth units, invalid-pixel handling, and camera conventions are not configured"
    )
