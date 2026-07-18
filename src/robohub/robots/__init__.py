from robohub.robots.base import Robot
from robohub.robots.fake_robot import FakeRobot
from robohub.robots.astribot import Astribot

ROBOT_REGISTRY = {
    "astribot": Astribot,
    "fake_robot": FakeRobot,
}

__all__ = ["Astribot", "FakeRobot", "Robot", "ROBOT_REGISTRY"]
