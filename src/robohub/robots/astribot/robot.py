"""Workstation-side Astribot workflow and action validation."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from robohub.communication import RobotClient
from robohub.policies import Policy
from robohub.robots.base import Robot
from robohub.schemas import Action, Observation


class AstribotRobot(Robot):
    """Coordinate an Astribot client and policy on the workstation."""

    def __init__(
        self,
        client: RobotClient,
        policy: Policy,
        config: Mapping[str, Any],
    ) -> None:
        self.client = client
        self.policy = policy
        self.config = config

    def get_observation(self) -> Observation:
        return self.client.get_observation()

    def set_action(self, action: Action) -> None:
        self.client.set_action(action)

    def step(self) -> Action:
        observation = self.get_observation()
        action = self.policy.infer(observation)
        self.set_action(action)
        return action

    def close(self) -> None:
        try:
            self.policy.close()
        finally:
            self.client.close()
