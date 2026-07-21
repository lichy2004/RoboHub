"""Convert Astribot recordings to ACT episode files."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
from numpy.typing import NDArray

CAMERA_GROUPS = {
    "head": "cam_high",
    "left": "cam_left_wrist",
    "right": "cam_right_wrist",
}
CAMERA_NAMES = tuple(CAMERA_GROUPS.values())
JOINT_DIM = 25
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def recorded_to_robohub_qpos(qpos: NDArray[np.generic]) -> NDArray[np.float32]:
    """Reorder one or more 25-dimensional Astribot joint vectors."""
    array = np.asarray(qpos)
    if array.ndim < 1 or array.shape[-1] != JOINT_DIM:
        raise ValueError(
            f"Expected recorded qpos shape (..., {JOINT_DIM}), got {array.shape}"
        )
    reordered = np.concatenate(
        (
            array[..., 7:14],
            array[..., 14:15],
            array[..., 15:22],
            array[..., 22:23],
            array[..., 23:25],
            array[..., 3:7],
            array[..., 0:3],
        ),
        axis=-1,
    )
    return np.asarray(reordered, dtype=np.float32)


def _frame_offsets(sizes: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(sizes[:-1], dtype=np.int64))
    )


def _camera_layout(
    group: h5py.Group,
    frame_count: int,
    camera_name: str,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if "rgb" not in group or "rgb_size" not in group:
        raise ValueError(f"Camera {camera_name!r} must contain rgb and rgb_size")

    rgb_dataset = group["rgb"]
    sizes = np.asarray(group["rgb_size"], dtype=np.int64)
    if sizes.shape != (frame_count,):
        raise ValueError(
            f"Expected {camera_name} rgb_size shape ({frame_count},), "
            f"got {sizes.shape}"
        )
    if np.any(sizes <= 0):
        raise ValueError(f"Camera {camera_name!r} contains non-positive frame sizes")
    if rgb_dataset.dtype != np.dtype(np.uint8) or rgb_dataset.ndim != 1:
        raise ValueError(
            f"Expected {camera_name} rgb to be one-dimensional uint8 bytes, "
            f"got shape {rgb_dataset.shape} and dtype {rgb_dataset.dtype}"
        )
    total_size = int(np.sum(sizes, dtype=np.int64))
    if total_size != rgb_dataset.shape[0]:
        raise ValueError(
            f"Camera {camera_name!r} byte count is {rgb_dataset.shape[0]}, "
            f"but rgb_size sums to {total_size}"
        )
    return sizes, _frame_offsets(sizes)


def _write_camera_frames(
    source_group: h5py.Group,
    destination: h5py.Dataset,
    frame_indices: NDArray[np.int64],
    sizes: NDArray[np.int64],
    offsets: NDArray[np.int64],
    camera_name: str,
) -> None:
    rgb_dataset = source_group["rgb"]
    for output_index, frame_index in enumerate(frame_indices):
        start = int(offsets[frame_index])
        end = start + int(sizes[frame_index])
        encoded = np.asarray(rgb_dataset[start:end], dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(
                f"Failed to decode {camera_name} RGB frame {int(frame_index)}"
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        destination[output_index] = cv2.resize(
            rgb,
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )


def _load_episode(
    source: h5py.File,
    source_path: Path,
    downsample: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int64],
    int,
]:
    state_path = "joints_dict/joints_position_state"
    command_path = "joints_dict/joints_position_command"
    if state_path not in source or command_path not in source:
        raise ValueError(
            f"{source_path} must contain {state_path} and {command_path}"
        )
    state = np.asarray(source[state_path])
    command = np.asarray(source[command_path])
    if state.ndim != 2 or state.shape[1] != JOINT_DIM:
        raise ValueError(f"Expected state shape (T, {JOINT_DIM}), got {state.shape}")
    if command.shape != state.shape:
        raise ValueError(f"Expected command shape {state.shape}, got {command.shape}")
    if state.shape[0] == 0:
        raise ValueError(f"{source_path} contains no frames")
    if not np.issubdtype(state.dtype, np.number) or not np.issubdtype(
        command.dtype, np.number
    ):
        raise ValueError("State and command datasets must be numeric")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(command)):
        raise ValueError("State and command datasets must contain finite values")

    for recorded_name in CAMERA_GROUPS:
        group_path = f"images_dict/{recorded_name}"
        if group_path not in source:
            raise ValueError(f"{source_path} is missing {group_path}")

    frame_indices = np.arange(0, state.shape[0], downsample, dtype=np.int64)
    sampled_command = recorded_to_robohub_qpos(command[frame_indices])
    qpos = np.empty_like(sampled_command)
    qpos[0] = recorded_to_robohub_qpos(state[0])
    qpos[1:] = sampled_command[:-1]
    return qpos, sampled_command, frame_indices, state.shape[0]


def convert_episode(
    source_path: str | Path,
    output_path: str | Path,
    downsample: int = 1,
    *,
    overwrite: bool = False,
) -> int:
    """Convert one Astribot recording and return its sampled episode length."""
    source = Path(source_path)
    output = Path(output_path)
    if downsample <= 0:
        raise ValueError("downsample must be a positive integer")
    if not source.is_file():
        raise FileNotFoundError(f"Input episode does not exist: {source}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output episode already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with (
            h5py.File(source, "r") as source_file,
            h5py.File(temporary, "w") as destination,
        ):
            qpos, action, frame_indices, frame_count = _load_episode(
                source_file,
                source,
                downsample,
            )
            camera_layouts = {
                recorded_name: _camera_layout(
                    source_file[f"images_dict/{recorded_name}"],
                    frame_count,
                    recorded_name,
                )
                for recorded_name in CAMERA_GROUPS
            }

            destination.attrs["sim"] = False
            observations = destination.create_group("observations")
            observations.create_dataset("qpos", data=qpos, dtype=np.float32)
            image_group = observations.create_group("images")
            for recorded_name, camera_name in CAMERA_GROUPS.items():
                image_dataset = image_group.create_dataset(
                    camera_name,
                    shape=(
                        len(frame_indices),
                        IMAGE_HEIGHT,
                        IMAGE_WIDTH,
                        3,
                    ),
                    dtype=np.uint8,
                    chunks=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                    compression="gzip",
                )
                sizes, offsets = camera_layouts[recorded_name]
                _write_camera_frames(
                    source_file[f"images_dict/{recorded_name}"],
                    image_dataset,
                    frame_indices,
                    sizes,
                    offsets,
                    recorded_name,
                )
            destination.create_dataset("action", data=action, dtype=np.float32)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(action)


def _natural_sort_key(path: Path) -> tuple[tuple[int, int | str], ...]:
    parts = re.split(r"(\d+)", path.name.casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )


def _input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.casefold() not in {".h5", ".hdf5"}:
            raise ValueError(f"Input file must be HDF5: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    files = sorted(
        (
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.casefold() in {".h5", ".hdf5"}
        ),
        key=_natural_sort_key,
    )
    if not files:
        raise ValueError(f"No HDF5 files found in {input_path}")
    return files


def convert_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    task: str,
    downsample: int = 1,
    *,
    overwrite: bool = False,
) -> list[int]:
    """Convert a file or directory and write dataset metadata."""
    if downsample <= 0:
        raise ValueError("downsample must be a positive integer")
    if not task.strip():
        raise ValueError("task must not be empty")

    sources = _input_files(Path(input_path))
    destination = Path(output_dir)
    outputs = [
        destination / f"episode_{episode_id}.hdf5"
        for episode_id in range(len(sources))
    ]
    metadata_path = destination / "dataset.yaml"
    conflicts = [path for path in [*outputs, metadata_path] if path.exists()]
    if conflicts and not overwrite:
        conflict_list = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"Refusing to overwrite existing output(s): {conflict_list}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    episode_lengths = [
        convert_episode(
            source,
            output,
            downsample,
            overwrite=overwrite,
        )
        for source, output in zip(sources, outputs, strict=True)
    ]
    metadata = {
        "task": task,
        "dataset_dir": str(destination.resolve()),
        "num_episodes": len(sources),
        "episode_lengths": episode_lengths,
        "camera_names": list(CAMERA_NAMES),
        "state_dim": JOINT_DIM,
        "action_dim": JOINT_DIM,
        "downsample": downsample,
    }
    temporary_metadata = destination / ".dataset.yaml.tmp"
    try:
        with temporary_metadata.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=False)
        temporary_metadata.replace(metadata_path)
    finally:
        if temporary_metadata.exists():
            temporary_metadata.unlink()
    return episode_lengths


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Astribot HDF5 recordings to ACT episodes"
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Astribot HDF5 file or directory",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="ACT dataset output directory",
    )
    parser.add_argument(
        "--input",
        dest="input_option",
        type=Path,
        help="Astribot HDF5 file or directory",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir_option",
        type=Path,
        help="ACT dataset output directory",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task name stored in dataset.yaml",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Keep every Nth command and matching image frame",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite episode files and metadata generated by this conversion",
    )
    parsed = parser.parse_args(args)
    for positional_name, option_name in (
        ("input", "input_option"),
        ("output_dir", "output_dir_option"),
    ):
        positional = getattr(parsed, positional_name)
        option = getattr(parsed, option_name)
        if positional is not None and option is not None:
            parser.error(
                f"{positional_name} cannot be provided both positionally and by option"
            )
        value = option if option is not None else positional
        if value is None:
            parser.error(f"{positional_name} is required")
        setattr(parsed, positional_name, value)
        delattr(parsed, option_name)
    return parsed


def main() -> None:
    args = parse_args()
    lengths = convert_dataset(
        args.input,
        args.output_dir,
        args.task,
        args.downsample,
        overwrite=args.overwrite,
    )
    print(f"Converted {len(lengths)} episode(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
