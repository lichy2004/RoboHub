#!/usr/bin/env python3
"""
Collect one Astribot observation and save all camera images.

python -m debugpy --listen 5678 --wait-for-client tests/astribot/get_observation.py
"""

from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from robohub.robots.astribot import AstribotBackend, AstribotRobot


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
_OUTPUT_DIR = _REPOSITORY_ROOT / "output" / "astribot_get_observation"


def _describe(name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    finite = array[np.isfinite(array)]
    value_range = (
        f", min={finite.min():.6g}, max={finite.max():.6g}"
        if finite.size
        else ""
    )
    print(f"{name}: shape={array.shape}, dtype={array.dtype}{value_range}")


def _describe_depth(name: str, value: np.ndarray) -> None:
    depth = np.asarray(value).squeeze()
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        print(f"depth/{name}: no valid non-zero pixels")
        return
    percentiles = np.percentile(valid, (1, 50, 99))
    invalid_ratio = 1.0 - valid.size / depth.size
    print(
        f"depth/{name}: valid={valid.size}/{depth.size} "
        f"({100 * (1 - invalid_ratio):.1f}%), "
        f"p01={percentiles[0]:.1f}, median={percentiles[1]:.1f}, "
        f"p99={percentiles[2]:.1f}"
    )


def _colorize_depth(image: np.ndarray) -> np.ndarray:
    depth = np.asarray(image, dtype=np.float32).squeeze()
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        near, far = np.percentile(depth[valid], (2.0, 98.0))
        if far > near:
            clipped = np.clip(depth, near, far)
            normalized[valid] = (
                (clipped[valid] - near) * 255.0 / (far - near)
            ).astype(np.uint8)

    depth_color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    depth_color[~valid] = 0
    return depth_color


def _save_images(observation) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, image in observation.rgb.items():
        path = _OUTPUT_DIR / f"astribot_{name}_rgb.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to save {path}")
        print(f"Saved {path}")

    for name, image in observation.depth.items():
        depth = np.asarray(image).squeeze()
        path = _OUTPUT_DIR / f"astribot_{name}_depth.png"
        if not cv2.imwrite(str(path), _colorize_depth(depth)):
            raise RuntimeError(f"Failed to save {path}")
        print(f"Saved {path}")


def _save_observation(observation) -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = _OUTPUT_DIR / "observation.npy"
    np.save(path, observation, allow_pickle=True)
    print(f"Saved {path}")


def main() -> None:
    with _CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    backend = AstribotBackend(config)
    try:
        started_at = time.perf_counter()
        raw_observation = backend.get_observation()
        backend_elapsed = time.perf_counter() - started_at

        started_at = time.perf_counter()
        observation = AstribotRobot.decode_observation(raw_observation, config)
        decode_elapsed = time.perf_counter() - started_at

        print(
            f"backend.get_observation(): {backend_elapsed:.3f} s "
            f"({backend_elapsed * 1000:.1f} ms)"
        )
        print(
            f"AstribotRobot.decode_observation(): {decode_elapsed:.3f} s "
            f"({decode_elapsed * 1000:.1f} ms)"
        )
        for name, image in observation.rgb.items():
            _describe(f"rgb/{name}", image)
        for name, image in observation.depth.items():
            _describe(f"depth/{name}", image)
            _describe_depth(name, image)
        _describe("joints_position", observation.joints_position)
        _describe("joints_velocity", observation.joints_velocity)
        _describe("joints_torque", observation.joints_torque)
        _save_images(observation)
        _save_observation(observation)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
