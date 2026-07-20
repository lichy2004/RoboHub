"""Offline tests for ACT configuration handling."""

from __future__ import annotations

import builtins
import copy
import sys
import unittest
from unittest import mock

from robohub.policies.act.configuration import (
    deep_merge,
    load_config,
    validate_config,
)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_use_astribot_dimensions_and_camera_order(self) -> None:
        config = load_config()

        self.assertEqual(config["task"]["state_dim"], 25)
        self.assertEqual(config["task"]["action_dim"], 25)
        self.assertEqual(
            config["task"]["camera_names"],
            ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        )

    def test_deep_merge_does_not_modify_either_input(self) -> None:
        base = {"nested": {"keep": [1, 2], "replace": "old"}, "base_only": True}
        override = {"nested": {"replace": "new", "added": {"value": 3}}}
        original_base = copy.deepcopy(base)
        original_override = copy.deepcopy(override)

        merged = deep_merge(base, override)
        merged["nested"]["keep"].append(99)
        merged["nested"]["added"]["value"] = 4

        self.assertEqual(base, original_base)
        self.assertEqual(override, original_override)
        self.assertEqual(merged["nested"]["replace"], "new")

    def test_validation_rejects_invalid_dimensions(self) -> None:
        for field in ("state_dim", "action_dim"):
            config = load_config()
            config["task"][field] = 24
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(config)

    def test_validation_rejects_invalid_temporal_decay(self) -> None:
        for decay in (-0.1, float("nan"), float("inf"), True, "0.1"):
            config = load_config()
            config["inference"]["temporal_agg_decay"] = decay
            with self.subTest(decay=decay), self.assertRaises(ValueError):
                validate_config(config)

    def test_configuration_import_does_not_require_torch(self) -> None:
        module_names = (
            "robohub.policies.act",
            "robohub.policies.act.configuration",
        )
        saved_modules = {name: sys.modules.get(name) for name in module_names}
        for name in module_names:
            sys.modules.pop(name, None)
        real_import = builtins.__import__

        def reject_torch(
            name: str,
            globals_: object = None,
            locals_: object = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("configuration import attempted to import torch")
            return real_import(name, globals_, locals_, fromlist, level)

        try:
            with mock.patch("builtins.__import__", side_effect=reject_torch):
                from robohub.policies.act.configuration import (
                    load_config as imported_load_config,
                )

                self.assertEqual(imported_load_config()["task"]["state_dim"], 25)
        finally:
            for name in module_names:
                sys.modules.pop(name, None)
                saved = saved_modules[name]
                if saved is not None:
                    sys.modules[name] = saved


if __name__ == "__main__":
    unittest.main()
