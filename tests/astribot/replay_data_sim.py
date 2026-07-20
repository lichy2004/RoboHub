#!/usr/bin/env python3
"""Replay recorded Astribot state and RGB images in Viser."""

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from robohub.processing.visual import Visual

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DATA_PATH = (
    _REPOSITORY_ROOT
    / "data"
    / "robot"
    / "astribot"
    / "hdf5_output_TheaExp0716"
    / "TheaExp0716_episode_0.hdf5"
)
_CAMERA_NAMES = {
    "head": "head",
    "left": "wrist_left",
    "right": "wrist_right",
}


def _recorded_to_robohub_qpos(qpos: np.ndarray) -> np.ndarray:
    qpos_array = np.asarray(qpos, dtype=np.float64)
    if qpos_array.shape != (25,):
        raise ValueError(f"Expected recorded qpos shape (25,), got {qpos_array.shape}")
    return np.concatenate(
        (
            qpos_array[7:14],
            qpos_array[14:15],
            qpos_array[15:22],
            qpos_array[22:23],
            qpos_array[23:25],
            qpos_array[3:7],
            qpos_array[0:3],
        )
    )


def _image_offsets(sizes: np.ndarray) -> np.ndarray:
    sizes_array = np.asarray(sizes, dtype=np.int64)
    if sizes_array.ndim != 1 or np.any(sizes_array <= 0):
        raise ValueError("RGB frame sizes must be a one-dimensional positive array")
    return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(sizes_array[:-1])))


def _decode_rgb(
    group: h5py.Group,
    frame_index: int,
    sizes: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    start = int(offsets[frame_index])
    end = start + int(sizes[frame_index])
    payload = np.asarray(group["rgb"][start:end], dtype=np.uint8)
    bgr = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to decode RGB frame {frame_index}")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _load_metadata(
    file: h5py.File,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[Any, np.ndarray, np.ndarray]]]:
    qpos = np.asarray(
        file["joints_dict/joints_position_state"],
        dtype=np.float64,
    )
    timestamps = np.asarray(
        file["joints_dict/state_timestamp"],
        dtype=np.float64,
    )
    if qpos.ndim != 2 or qpos.shape[1] != 25:
        raise ValueError(f"Expected qpos shape (N, 25), got {qpos.shape}")
    if timestamps.shape != (qpos.shape[0],):
        raise ValueError(
            f"Expected timestamps shape ({qpos.shape[0]},), got {timestamps.shape}"
        )

    image_streams = {}
    for recorded_name, visual_name in _CAMERA_NAMES.items():
        group = file[f"images_dict/{recorded_name}"]
        sizes = np.asarray(group["rgb_size"], dtype=np.int64)
        if sizes.shape != (qpos.shape[0],):
            raise ValueError(
                f"Expected {recorded_name} RGB sizes shape "
                f"({qpos.shape[0]},), got {sizes.shape}"
            )
        image_streams[visual_name] = (
            group,
            sizes,
            _image_offsets(sizes),
        )
    return qpos, timestamps, image_streams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Astribot HDF5 data in Viser")
    parser.add_argument("--data", type=Path, default=_DATA_PATH)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument(
        "--exit-on-complete",
        action="store_true",
        help="Stop the Viser server when replay finishes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replay_speed <= 0.0:
        raise ValueError("replay-speed must be positive")
    if args.start_frame < 0:
        raise ValueError("start-frame must be non-negative")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("max-frames must be positive")

    visual = Visual(
        host=args.host,
        port=args.port,
    )
    try:
        with h5py.File(args.data, "r") as file:
            qpos, timestamps, image_streams = _load_metadata(file)
            stop_frame = len(qpos)
            if args.max_frames is not None:
                stop_frame = min(
                    stop_frame,
                    args.start_frame + args.max_frames,
                )
            if args.start_frame >= stop_frame:
                raise ValueError(
                    f"start-frame {args.start_frame} is outside the recording"
                )

            print(f"Replaying frames {args.start_frame}:{stop_frame} from {args.data}")
            print("The recording has no depth stream; point cloud is skipped.")
            for frame_index in range(args.start_frame, stop_frame):
                cycle_started = time.perf_counter()
                rgb = {
                    camera_name: _decode_rgb(
                        group,
                        frame_index,
                        sizes,
                        offsets,
                    )
                    for camera_name, (
                        group,
                        sizes,
                        offsets,
                    ) in image_streams.items()
                }
                visual.update(
                    _recorded_to_robohub_qpos(qpos[frame_index]),
                    rgb,
                )

                if frame_index + 1 < stop_frame:
                    period = (
                        timestamps[frame_index + 1] - timestamps[frame_index]
                    ) / args.replay_speed
                    remaining = period - (time.perf_counter() - cycle_started)
                    if remaining > 0.0:
                        time.sleep(remaining)

        print("Replay completed.")
        if not args.exit_on_complete:
            print("Press Ctrl+C to close the Viser server.")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("Replay stopped.")
    finally:
        visual.close()


if __name__ == "__main__":
    main()
