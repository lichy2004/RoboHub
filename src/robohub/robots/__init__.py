from robohub.robots.astribot import Astribot
from robohub.robots.base import Robot
from robohub.robots.cobot_magic import CobotMagic
from robohub.robots.fake_robot import FakeRobot
from robohub.robots.unitree_g1 import UnitreeG1

ROBOT_REGISTRY = {
    "astribot": Astribot,
    "cobot_magic": CobotMagic,
    "fake_robot": FakeRobot,
    "unitree_g1": UnitreeG1,
}

__all__ = ["Astribot", "CobotMagic", "FakeRobot", "Robot", "UnitreeG1", "ROBOT_REGISTRY"]
