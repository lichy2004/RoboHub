"""Configuration loading helpers."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        value = yaml.safe_load(config_file)

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    return value
