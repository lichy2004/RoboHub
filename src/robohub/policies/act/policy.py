from pathlib import Path
from typing import Any

from robohub.policies.base import Policy
from robohub.utils.config import load_config
from robohub.utils.types import Action, Observation


class ACTPolicy(Policy):
    def __init__(self, config_path: str | Path | None = None) -> None:
        path = config_path or Path(__file__).parent / "config" / "default.yaml"
        super().__init__(load_config(path))

    def load_model(self) -> None:
        raise NotImplementedError("ACT model implementation is not configured")

    def encode_obs(self, obs: Observation) -> Any:
        raise NotImplementedError("ACT observation encoder is not configured")

    def get_action(self, obs: Observation) -> Action:
        raise NotImplementedError("ACT inference is not configured")
