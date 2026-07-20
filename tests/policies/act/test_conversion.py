"""Offline tests for Astribot-to-ACT data conversion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml

from robohub.policies.act.data.convert_astribot import (
    CAMERA_NAMES,
    convert_dataset,
    convert_episode,
    recorded_to_robohub_qpos,
)


def _expected_reorder(values: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            values[..., 7:14],
            values[..., 14:15],
            values[..., 15:22],
            values[..., 22:23],
            values[..., 23:25],
            values[..., 3:7],
            values[..., 0:3],
        ),
        axis=-1,
    ).astype(np.float32)


def _jpeg_bytes(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr)
    if not success:
        raise RuntimeError("Failed to encode synthetic JPEG")
    return encoded


def _write_recording(
    path: Path,
    state: np.ndarray,
    command: np.ndarray,
) -> None:
    frame_count = state.shape[0]
    camera_colors = {
        "head": (220, 20, 10),
        "left": (10, 210, 30),
        "right": (20, 30, 200),
    }
    with h5py.File(path, "w") as output:
        joints = output.create_group("joints_dict")
        joints.create_dataset("joints_position_state", data=state)
        joints.create_dataset("joints_position_command", data=command)
        images = output.create_group("images_dict")
        for camera_index, (camera_name, color) in enumerate(camera_colors.items()):
            encoded_frames = []
            for frame_index in range(frame_count):
                rgb = np.full((12, 16, 3), color, dtype=np.uint8)
                rgb[0, 0] = (frame_index, camera_index, 127)
                encoded_frames.append(_jpeg_bytes(rgb))
            camera = images.create_group(camera_name)
            camera.create_dataset(
                "rgb",
                data=np.concatenate(encoded_frames).astype(np.uint8),
            )
            camera.create_dataset(
                "rgb_size",
                data=np.asarray([len(frame) for frame in encoded_frames]),
            )


class RecordedQposTests(unittest.TestCase):
    def test_reorders_single_and_batched_vectors_exactly(self) -> None:
        vector = np.arange(25, dtype=np.float64)
        batch = np.stack((vector, vector + 100))

        np.testing.assert_array_equal(
            recorded_to_robohub_qpos(vector),
            _expected_reorder(vector),
        )
        converted_batch = recorded_to_robohub_qpos(batch)
        np.testing.assert_array_equal(converted_batch, _expected_reorder(batch))
        self.assertEqual(converted_batch.dtype, np.float32)

    def test_rejects_wrong_last_dimension(self) -> None:
        for value in (np.zeros(24), np.zeros((2, 26)), np.asarray(1.0)):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                recorded_to_robohub_qpos(value)


class ConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.state = np.arange(6 * 25, dtype=np.float64).reshape(6, 25)
        self.command = self.state + 1000
        self.source = self.root / "recording.h5"
        _write_recording(self.source, self.state, self.command)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_convert_episode_preserves_downsampled_timing_and_images(self) -> None:
        output = self.root / "episode.hdf5"

        length = convert_episode(self.source, output, downsample=2)

        self.assertEqual(length, 3)
        sampled_command = _expected_reorder(self.command[[0, 2, 4]])
        expected_qpos = np.vstack(
            (_expected_reorder(self.state[0]), sampled_command[:-1])
        )
        with h5py.File(output, "r") as episode:
            self.assertIs(episode.attrs["sim"], np.False_)
            np.testing.assert_array_equal(episode["action"][:], sampled_command)
            np.testing.assert_array_equal(
                episode["observations/qpos"][:],
                expected_qpos,
            )
            self.assertEqual(episode["action"].dtype, np.dtype(np.float32))
            self.assertEqual(
                episode["observations/qpos"].dtype,
                np.dtype(np.float32),
            )
            images = episode["observations/images"]
            self.assertEqual(set(images), set(CAMERA_NAMES))
            for camera_name in CAMERA_NAMES:
                self.assertEqual(images[camera_name].shape, (3, 480, 640, 3))
                self.assertEqual(images[camera_name].dtype, np.dtype(np.uint8))

    def test_convert_dataset_writes_complete_metadata(self) -> None:
        input_directory = self.root / "input"
        input_directory.mkdir()
        _write_recording(input_directory / "sample10.h5", self.state, self.command)
        _write_recording(
            input_directory / "sample2.h5",
            self.state[:3],
            self.command[:3],
        )
        output_directory = self.root / "dataset"

        lengths = convert_dataset(
            input_directory,
            output_directory,
            task="offline conversion",
            downsample=2,
        )

        self.assertEqual(lengths, [2, 3])
        with (output_directory / "dataset.yaml").open(encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)
        self.assertEqual(
            metadata,
            {
                "task": "offline conversion",
                "dataset_dir": str(output_directory.resolve()),
                "num_episodes": 2,
                "episode_lengths": [2, 3],
                "camera_names": list(CAMERA_NAMES),
                "state_dim": 25,
                "action_dim": 25,
                "downsample": 2,
            },
        )
        self.assertTrue((output_directory / "episode_0.hdf5").is_file())
        self.assertTrue((output_directory / "episode_1.hdf5").is_file())

    def test_refuses_to_overwrite_existing_episode(self) -> None:
        output = self.root / "episode.hdf5"
        output.write_bytes(b"keep me")

        with self.assertRaises(FileExistsError):
            convert_episode(self.source, output)

        self.assertEqual(output.read_bytes(), b"keep me")

    def test_rejects_incorrect_state_or_command_shape(self) -> None:
        cases = (
            (np.zeros((3, 24)), np.zeros((3, 24))),
            (np.zeros((3, 25)), np.zeros((2, 25))),
        )
        for index, (state, command) in enumerate(cases):
            malformed = self.root / f"malformed-{index}.h5"
            _write_recording(malformed, state, command)
            with self.subTest(index=index), self.assertRaises(ValueError):
                convert_episode(malformed, self.root / f"output-{index}.hdf5")


if __name__ == "__main__":
    unittest.main()
