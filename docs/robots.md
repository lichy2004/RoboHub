# Robot 开发指南

本文说明 RoboHub 中 Robot 组件的职责、数据接口、目录结构，以及新增机器人适配器的完整流程。

## 1. Robot 的职责

Robot 是 RoboHub 与真实机器人硬件之间的适配层。它负责：

1. 初始化机器人 SDK、控制器、相机和其他传感器。
2. 将机器人移动到安全的默认位置。
3. 采集图像、深度和关节状态，并转换为统一的 `Observation`。
4. 接收统一的 `Action`，校验后转换为机器人 SDK 所需的控制命令。
5. 在程序退出、网络断开或发生异常时释放硬件资源。

Robot 不应负责模型加载、图像归一化或策略推理。这些功能属于 Policy。

## 2. Robot 基础接口

所有机器人必须继承 `src/robohub/robots/base.py` 中的 `Robot`：

```python
from robohub.robots.base import Robot
```

当前基类定义了初始化逻辑以及四个抽象方法，子类必须实现 `reset()`、`get_observation()`、`set_action()` 和 `close()`。

### 2.1 初始化

`Robot` 接收完整的配置映射：

```python
class Robot(ABC):
    def __init__(self, config: Mapping[str, Any]) -> None:
        ...
```

子类应先读取 YAML 配置，再将完整配置传给基类：

```python
path = config_path or Path(__file__).parent / "configs" / "default.yaml"
config = load_config(path)
super().__init__(config)
```

初始化后具有以下公共属性：

- `config: Mapping[str, Any]`：通过 `MappingProxyType` 包装的只读配置。
- `name`：来自 `config["name"]`。
- `joints_order: tuple`：来自 `config["joints_order"]`，并转换为元组。
- `joints_num`：来自 `config["joints_num"]`。

基类会检查 `joints_num == len(joints_order)`，不一致时抛出 `ValueError`。厂商 SDK、相机和控制器等资源可由子类在调用 `super().__init__(config)` 后初始化。

### 2.2 `reset()`

```python
def reset(self) -> None:
    ...
```

让机器人恢复到可开始运行的状态。真实机器人通常根据 `config["default_position"]` 移动到默认关节位置；具体速度、控制模式和安全策略由机器人 SDK 实现。

### 2.3 `get_observation()`

```python
def get_observation(self) -> Observation:
    ...
```

返回 `robohub.utils.types.Observation` 定义的统一观测：

```python
{
    "rgb": {
        "head": np.ndarray,
        "wrist_left": np.ndarray,
        "wrist_right": np.ndarray,
    },
    "depth": {
        "head": np.ndarray,
        "wrist_left": np.ndarray,
        "wrist_right": np.ndarray,
    },
    "joints_position": np.ndarray,
    "joints_velocity": np.ndarray,
    "joints_torque": np.ndarray,
}
```

关节数组的顺序应与 `joints_order` 一致。当前代码只通过 `TypedDict` 描述结构，没有提供运行时的 `validate_observation()`，Robot Server 也不会额外校验观测内容，因此适配器应直接按约定返回数据。

### 2.4 `set_action()`

```python
def set_action(self, action: Action) -> None:
    ...
```

接收 `robohub.utils.types.Action` 定义的动作：

```python
{
    "left_arm": np.ndarray,
    "left_gripper": np.ndarray,
    "right_arm": np.ndarray,
    "right_gripper": np.ndarray,
    "torso": np.ndarray,
    "head": np.ndarray,
    "base": np.ndarray,
}
```

各字段的预期长度由 `config["action_dimensions"]` 描述。适配器负责将这些字段转换为厂商 SDK 所需的命令和顺序。当前基类及 Robot Server 不会对动作字段、维度或数值做通用运行时校验。

### 2.5 `close()`

```python
def close(self) -> None:
    ...
```

释放机器人占用的资源，例如厂商 SDK 会话、相机、串口、Socket、线程和设备句柄。当前 `Robot` 基类没有实现 context manager，调用方应显式调用 `close()`。

## 3. Robot 目录结构

每个机器人使用独立子包：

```text
src/robohub/robots/my_robot/
├── __init__.py
├── robot.py
├── assets/
└── configs/
    └── default.yaml
```

各部分用途：

- `robot.py`：Robot 子类和硬件适配逻辑。
- `configs/default.yaml`：默认关节、动作维度和设备参数。
- `assets/`：URDF、标定文件或机器人描述等静态资源。
- `__init__.py`：导出 Robot 类。

不要把模型权重、采集数据、密钥或大型二进制文件提交到 `assets/`。

## 4. Robot 配置

每个 Robot 的默认配置位于自己的 `configs/default.yaml`。按照当前基类和 `fake_robot` 的格式，配置如下：

```yaml
name: my_robot
joints_order:
  - left_joint_1
  - left_joint_2
  - right_joint_1
  - right_joint_2
joints_num: 4
default_position: [0.0, 0.0, 0.0, 0.0]
image_size: [480, 640]
action_dimensions:
  left_arm: 2
  left_gripper: 1
  right_arm: 2
  right_gripper: 1
  torso: 0
  head: 0
  base: 0
```

字段说明：

- `name`：机器人名称，通常与注册表中的键一致。
- `joints_order`：观测中关节位置、速度和力矩的排列顺序。
- `joints_num`：关节总数，必须等于 `joints_order` 的长度。
- `default_position`：`reset()` 使用的默认关节位置，长度通常与 `joints_num` 相同。
- `image_size`：图像高度和宽度；仅在机器人实现需要时配置。
- `action_dimensions`：七类动作字段对应的数组长度；不支持的部分使用 `0`。

可根据 SDK 增加机器人 IP、串口、设备名称、控制频率、超时、相机参数和控制模式等字段。敏感凭据应通过环境变量或部署系统提供，不应写入 YAML。

## 5. 新增 Robot

下面以 `MyRobot` 为例。

### 第一步：创建目录

```text
src/robohub/robots/my_robot/
├── __init__.py
├── robot.py
├── assets/
└── configs/
    └── default.yaml
```

Python 包名使用 `snake_case`，类名使用 `PascalCase`。

### 第二步：编写配置

在 `configs/default.yaml` 中填写 `name`、真实关节顺序、`joints_num`、默认位置和各动作字段维度。`joints_num` 必须等于 `joints_order` 的长度，关节顺序必须与 `get_observation()` 输出一致。

### 第三步：实现适配器

```python
from pathlib import Path

import numpy as np

from robohub.robots.base import Robot
from robohub.utils.config import load_config
from robohub.utils.types import Action, Observation


class MyRobot(Robot):
    def __init__(self, config_path: str | Path | None = None) -> None:
        path = config_path or Path(__file__).parent / "configs" / "default.yaml"
        config = load_config(path)
        super().__init__(config)
        self._sdk = self._connect_hardware()

    def reset(self) -> None:
        self._sdk.move_joints(self.config["default_position"])

    def get_observation(self) -> Observation:
        return {
            "rgb": self._read_rgb_images(),
            "depth": self._read_depth_images(),
            "joints_position": np.asarray(self._sdk.get_joint_positions()),
            "joints_velocity": np.asarray(self._sdk.get_joint_velocities()),
            "joints_torque": np.asarray(self._sdk.get_joint_torques()),
        }

    def set_action(self, action: Action) -> None:
        self._sdk.send_action(self._convert_action(action))

    def close(self) -> None:
        self._sdk.close()
```

示例中的 SDK 方法只是结构说明，新增机器人时必须替换为真实厂商 API。实现应保持简单，重点是完成统一接口与厂商 SDK 之间的数据转换。除厂商 SDK 或硬件安全确实要求的检查外，不需要编写过多的字段判断、契约测试或主动 `raise`；当前基类只检查 `joints_num` 与 `joints_order` 的长度是否一致。

### 第四步：导出类

在 `src/robohub/robots/my_robot/__init__.py` 中：

```python
from robohub.robots.my_robot.robot import MyRobot

__all__ = ["MyRobot"]
```

### 第五步：注册机器人

修改 `src/robohub/robots/__init__.py`：

```python
from robohub.robots.my_robot import MyRobot

ROBOT_REGISTRY = {
    "astribot": Astribot,
    "cobot_magic": CobotMagic,
    "unitree_g1": UnitreeG1,
    "my_robot": MyRobot,
}
```

注册后，运行入口会自动将名称加入命令行选项：

```bash
robohub-robot my_robot --host 0.0.0.0 --port 8765
```

### 第六步：确保配置被打包

当前 `pyproject.toml` 会包含 `robots/*/configs/*.yaml`。如果新增其他文件类型，例如 JSON 标定文件，需要同步更新 `tool.setuptools.package-data`。

### 第七步：实现检查清单

新增 Robot 完成后确认：

- [ ] 继承 `Robot` 并实现 `reset()`、`get_observation()`、`set_action()` 和 `close()`。
- [ ] 默认配置包含 `name`、`joints_order`、`joints_num`、`default_position` 和 `action_dimensions`。
- [ ] `joints_num` 等于 `joints_order` 的长度。
- [ ] Observation 包含三路 RGB、三路 Depth 和三个关节状态字段，关节顺序与配置一致。
- [ ] 七个 Action 字段都有明确的维度和控制语义。
- [ ] 已在机器人子包的 `__init__.py` 中导出实现类。
- [ ] 已在 `ROBOT_REGISTRY` 中注册。
- [ ] 配置和必要的 assets 会被安装包包含。
- [ ] `close()` 能释放实际占用的硬件资源。

检查应以接口可用和真实硬件行为正确为主，不需要为简单的数据传递添加大量判断、测试或 `raise`。如需自动测试，应使用 fake SDK，避免依赖真实硬件或局域网。

## 6. 与 Policy 的通信流程

Robot 主机通过以下命令启动服务：

```bash
robohub-robot my_robot --host 0.0.0.0 --port 8765
```

基本循环是：

1. Policy Client 发送 `get_observation`。
2. Robot Server 调用 `robot.get_observation()`。
3. Observation 校验成功后通过 TCP 返回。
4. Policy Client 发送 `set_action`。
5. Robot Server 校验 Action 并调用 `robot.set_action()`。
6. Robot Server 返回 `ack` 或 `error`。

TCP 通信不会替代机器人自身的安全控制。真实适配器仍必须实现超时停机、急停、动作限制和控制权限检查。
