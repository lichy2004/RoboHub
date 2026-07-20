"""Lightweight runtime tests for ACTPolicy without constructing ACT models."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from robohub.policies.act.policy import ACTPolicy


def _flatten_action(action: Any) -> np.ndarray:
    return np.concatenate(
        (
            action.left_arm,
            action.left_gripper,
            action.right_arm,
            action.right_gripper,
            action.head,
            action.torso,
            action.base,
        )
    )


@unittest.skipIf(torch is None, "torch is unavailable; skipping ACT runtime tests")
class ACTPolicyRuntimeTests(unittest.TestCase):
    def _policy(
        self,
        chunks: list[np.ndarray],
        *,
        temporal_agg: bool,
        decay: float = 0.0,
    ) -> tuple[Any, Callable[[], int]]:
        policy = ACTPolicy.__new__(ACTPolicy)
        policy.model = object()
        policy.optimizer = None
        policy.device = torch.device("cpu")
        policy._chunk_size = len(chunks[0])
        policy._temporal_agg = temporal_agg
        policy._temporal_agg_decay = decay
        policy._timestep = 0
        policy._cached_chunk = None
        policy._cache_index = 0
        policy._history = []
        policy._closed = False
        policy._prepare_inputs = lambda observation: (None, None)
        call_count = 0

        def predict_chunk(qpos: object, images: object) -> np.ndarray:
            nonlocal call_count
            chunk = chunks[call_count]
            call_count += 1
            return chunk.copy()

        policy._predict_chunk = predict_chunk
        return policy, lambda: call_count

    def test_to_action_uses_exact_joint_segments(self) -> None:
        values = np.arange(25, dtype=np.float32)

        action = ACTPolicy._to_action(values)

        np.testing.assert_array_equal(action.left_arm, values[0:7])
        np.testing.assert_array_equal(action.left_gripper, values[7:8])
        np.testing.assert_array_equal(action.right_arm, values[8:15])
        np.testing.assert_array_equal(action.right_gripper, values[15:16])
        np.testing.assert_array_equal(action.head, values[16:18])
        np.testing.assert_array_equal(action.torso, values[18:22])
        np.testing.assert_array_equal(action.base, values[22:25])

    def test_non_temporal_inference_consumes_each_chunk_before_predicting(self) -> None:
        chunks = [
            np.stack(
                (
                    np.full(25, 1, dtype=np.float32),
                    np.full(25, 2, dtype=np.float32),
                )
            ),
            np.stack(
                (
                    np.full(25, 3, dtype=np.float32),
                    np.full(25, 4, dtype=np.float32),
                )
            ),
        ]
        policy, prediction_count = self._policy(chunks, temporal_agg=False)

        outputs = [_flatten_action(policy.infer(None)) for _ in range(3)]

        self.assertEqual(prediction_count(), 2)
        for output, expected in zip(outputs, (1, 2, 3), strict=True):
            np.testing.assert_array_equal(output, np.full(25, expected))

    def test_temporal_aggregation_includes_all_zero_valid_prediction(self) -> None:
        chunks = [
            np.zeros((2, 25), dtype=np.float32),
            np.stack(
                (
                    np.full(25, 2, dtype=np.float32),
                    np.full(25, 4, dtype=np.float32),
                )
            ),
            np.full((2, 25), 8, dtype=np.float32),
        ]
        policy, prediction_count = self._policy(chunks, temporal_agg=True)

        outputs = [_flatten_action(policy.infer(None)) for _ in range(3)]

        self.assertEqual(prediction_count(), 3)
        np.testing.assert_array_equal(outputs[0], np.zeros(25))
        np.testing.assert_array_equal(outputs[1], np.ones(25))
        np.testing.assert_array_equal(outputs[2], np.full(25, 6))

    def test_reset_clears_episode_state_and_close_blocks_inference(self) -> None:
        chunks = [
            np.full((2, 25), 3, dtype=np.float32),
            np.full((2, 25), 4, dtype=np.float32),
        ]
        policy, prediction_count = self._policy(chunks, temporal_agg=False)
        policy.infer(None)

        policy.reset()

        self.assertEqual(policy._timestep, 0)
        self.assertIsNone(policy._cached_chunk)
        self.assertEqual(policy._cache_index, 0)
        self.assertEqual(policy._history, [])
        np.testing.assert_array_equal(
            _flatten_action(policy.infer(None)),
            np.full(25, 4),
        )
        self.assertEqual(prediction_count(), 2)

        policy.close()
        policy.close()
        with self.assertRaises(RuntimeError):
            policy.infer(None)


if __name__ == "__main__":
    unittest.main()
