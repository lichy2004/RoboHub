"""Workstation-side project robot orchestration."""

from robohub.communication import RobotClient
from robohub.policies import Policy
from robohub.robots.base import Robot
from robohub.schemas import Action, Observation


class MyRobot(Robot):
    def __init__(self, client: RobotClient, policy: Policy) -> None:
        self.client = client
        self.policy = policy

    def get_observation(self) -> Observation:
        return self.client.get_observation()

    def set_action(self, action: Action) -> None:
        self.client.set_action(action)

    def reset(self) -> None:
        self.policy.reset()

    def step(self) -> Action:
        observation = self.get_observation()
        action = self.policy.infer(observation)
        self.set_action(action)
        return action

    def close(self) -> None:
        self.policy.close()
        self.client.close()
