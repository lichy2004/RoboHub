from pathlib import Path

from robohub.robots.base import Robot
from robohub.utils.config import load_config
from robohub.utils.types import Action, Observation


class Astribot(Robot):
    def __init__(self, config_path: str | Path | None = None) -> None:
        path = config_path or Path(__file__).parent / "configs" / "default.yaml"
        config = load_config(path)
        super().__init__(config)

    def reset(self) -> None:
        raise NotImplementedError("Astribot hardware SDK is not configured")

    def get_observation(self) -> Observation:
        raise NotImplementedError("Astribot hardware SDK is not configured")

    def set_action(self, action: Action) -> None:
        raise NotImplementedError("Astribot hardware SDK is not configured")

    def close(self) -> None:
        pass
