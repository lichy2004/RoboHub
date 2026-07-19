# Robot 开发指南

本文档说明 RoboHub 当前的 Robot 架构、机器人端与工作站端的职责边界，以及新增一种机器人时需要实现的内容。

## 1. 整体架构

RoboHub 将机器人部署分为机器人端和工作站端：

```text
机器人端                                              工作站端

Robot SDK / ROS
       │
       ▼
RobotBackend
       │ Observation / Action
       ▼
RobotServer ─────────────── TCP ─────────────── RobotClient
                                                    │
                                                    ▼
                                                  Robot
                                                    │
                                          ┌─────────┴─────────┐
                                          ▼                   ▼
                                    Observation             Policy
                                                              │
                                                              ▼
                                                            Action
```

机器人端负责读取硬件状态并执行动作，工作站端负责调用 Policy。两端只通过标准的 `Observation` 和 `Action` Schema 交换数据。

## 2. 代码结构

当前 Robot 相关目录如下：

```text
src/robohub/
├── backends/
│   └── base.py                    # 通用 RobotBackend Protocol
├── communication/
│   ├── protocol.py                # 通信协议消息
│   ├── serialization.py           # Schema 与 NumPy 序列化
│   ├── robot_client.py            # 通用 TCP 客户端
│   └── robot_server.py            # 通用 TCP 服务端
├── schemas/
│   ├── observation.py             # 标准观测 Schema
│   └── action.py                  # 标准动作 Schema
├── robots/
│   ├── base.py                    # 工作站端 Robot 抽象接口
│   └── my_robot/
│       ├── __init__.py
│       ├── backend.py             # 示例 SDK 和硬件适配层
│       ├── robot.py               # 工作站端流程编排
│       └── configs/
│           └── default.yaml       # 机器人配置模板
└── policies/
    └── my_policy/
        └── policy.py              # 示例 Policy
```

`RobotClient`、`RobotServer` 和通用数据处理属于公共模块，不应在每个机器人目录中重复实现。

## 3. 各组件职责

### 3.1 `RobotBackend`

`RobotBackend` 运行在机器人端，是 RoboHub 与真实硬件之间的适配层。公共接口定义在 `robohub.backends.base`：

```python
class RobotBackend(Protocol):
    def get_observation(self) -> Observation: ...
    def set_action(self, action: Action) -> None: ...
    def close(self) -> None: ...
```

Backend 负责：

- 初始化厂家 SDK、ROS 节点、相机或其他硬件资源；
- 读取 RGB、Depth 和关节状态；
- 将硬件数据转换为标准 `Observation`；
- 将标准 `Action` 转换为硬件控制命令；
- 处理关节名称、顺序、单位和坐标系映射；
- 执行机器人端必要的动作范围、安全状态和控制模式检查；
- 释放 SDK、ROS 和硬件资源。

Backend 不负责：

- TCP 协议和消息收发；
- Policy 推理；
- 工作站端的数据处理流程；
- 为特定 Policy 生成专用数据格式。

当前 `MyRobotBackend` 使用 `MySDK` 生成随机观测，仅用于验证通信链路，不是生产硬件实现。

### 3.2 `RobotServer`

`RobotServer` 是机器人端的公共 TCP 服务，位于 `robohub.communication`。它接收一个 `RobotBackend`，并负责：

- 监听 TCP 端口；
- 处理消息长度前缀、协议版本和序列化；
- 分发 `get_observation` 与 `set_action` 请求；
- 将请求委托给 Backend；
- 返回结果或远端错误；
- 在 Policy 客户端断开后继续等待新连接。

新增机器人时通常不需要继承或复制 `RobotServer`，只需把新的 Backend 注入公共服务端：

```python
backend = NewRobotBackend(config)
server = RobotServer(backend, host="0.0.0.0", port=8765, timeout=30.0)
server.serve_forever()
```

### 3.3 `RobotClient`

`RobotClient` 运行在工作站端，负责连接公共 `RobotServer`：

```python
client = RobotClient(host="192.168.1.100", port=8765, timeout=30.0)
observation = client.get_observation()
client.set_action(action)
```

所有使用相同 TCP 协议和 Schema 的机器人都应复用该客户端。只有协议完全不同或存在额外机器人专用操作时，才需要新增客户端实现。

### 3.4 `Robot`

`Robot` 运行在工作站端，是应用程序使用的统一机器人接口：

```python
class Robot(ABC):
    def get_observation(self) -> Observation: ...
    def set_action(self, action: Action) -> None: ...
    def close(self) -> None: ...
```

当前 `MyRobot` 组合了一个公共 `RobotClient` 和一个 `Policy`：

```text
MyRobot.step()
    ├── RobotClient.get_observation()
    ├── Policy.infer(observation)
    └── RobotClient.set_action(action)
```

`Robot` 负责流程编排，但不应实现 TCP 协议、硬件 SDK 调用或模型内部推理。

如果不同机器人在工作站端没有不同的数据处理或动作校验要求，可以直接复用一个公共 Robot 实现。只有机器人具有特定的工作流、配置映射或处理步骤时，才需要创建机器人专用 `robot.py`。

### 3.5 标准 Schema

机器人端和工作站端必须使用相同的数据契约。

`Observation` 当前包含：

- `rgb`：相机名称到 RGB NumPy 数组的映射；
- `depth`：相机名称到深度 NumPy 数组的映射；
- `joints_position`：关节位置；
- `joints_velocity`：关节速度；
- `joints_torque`：关节力矩。

`Action` 当前包含：

- `left_arm`；
- `left_gripper`；
- `right_arm`；
- `right_gripper`；
- `torso`；
- `head`；
- `base`。

Backend 必须保证数组的维度、顺序、数据类型和单位与配置及 Schema 约定一致。

## 4. 机器人配置

每种机器人在自己的 `configs` 目录中提供一个 `default.yaml`：

```text
robots/new_robot/configs/default.yaml
```

配置包含两个顶层部分：

```yaml
communication:
  host: 0.0.0.0
  port: 8765
  timeout: 30.0

robot:
  joints: {}
  cameras: {}
```

### 4.1 关节配置

关节配置至少包含：

- `count`：关节数量；
- `order`：标准关节排列顺序。

`count` 必须与 `order` 的长度一致。Backend 从 SDK 读取状态和发送动作时，都必须按照该顺序完成映射。

### 4.2 相机配置

相机分为 `rgb` 和 `depth`，每个相机至少应定义：

- `resolution`：`[width, height]`；
- `intrinsic.camera_matrix`：3×3 相机矩阵；
- `intrinsic.distortion_coefficients`：畸变参数；
- `extrinsic.parent_frame`：参考坐标系；
- `extrinsic.child_frame`：相机坐标系；
- `extrinsic.transform`：4×4 齐次变换矩阵。

深度相机还应定义：

- `depth_scale`：原始深度到米的缩放系数；
- `min_depth`：最小有效深度，单位为米；
- `max_depth`：最大有效深度，单位为米。

`my_robot/configs/default.yaml` 可作为新增机器人配置的模板。

## 5. 新增一种机器人

假设新增机器人名为 `new_robot`，推荐目录结构如下：

```text
src/robohub/robots/new_robot/
├── __init__.py
├── backend.py
├── robot.py                 # 仅在需要专用工作站流程时创建
└── configs/
    └── default.yaml
```

### 步骤一：准备配置

复制模板：

```bash
cp -r src/robohub/robots/my_robot src/robohub/robots/new_robot
```

然后修改 `default.yaml`：

1. 设置通信地址、端口和超时；
2. 设置真实关节数量；
3. 按 SDK 与 RoboHub 约定填写关节顺序；
4. 填写所有 RGB 和深度相机；
5. 替换示例内参和外参为真实标定结果；
6. 设置深度单位与有效范围。

### 步骤二：实现 Backend

在 `backend.py` 中实现 `RobotBackend` 所需的三个方法：

```python
class NewRobotBackend:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.sdk = NewRobotSDK(...)

    def get_observation(self) -> Observation:
        # Read hardware and convert it to the standard schema.
        ...

    def set_action(self, action: Action) -> None:
        # Validate and convert the standard action to SDK commands.
        ...

    def close(self) -> None:
        self.sdk.close()
```

实际实现必须完成以下功能。

#### 观测读取

- 读取 RGB 图像；
- 读取 Depth 图像；
- 读取关节位置、速度和力矩；
- 将相机命名映射到配置中的名称；
- 将关节数据映射到配置中的 `order`；
- 统一数组数据类型、shape 和单位；
- 返回标准 `Observation`。

#### 动作执行

- 检查所有动作数组的 shape；
- 将标准动作顺序映射为 SDK 顺序；
- 转换角度、长度、速度等单位；
- 检查关节范围和速度限制；
- 检查机器人当前控制模式；
- 拒绝无效、过期或不安全的动作；
- 调用 SDK 或 ROS 执行动作。

#### 生命周期管理

- 初始化 SDK 或 ROS；
- 等待必要的硬件连接和传感器；
- 在 `close()` 中停止控制并释放资源；
- 保证重复调用 `close()` 不会导致异常。

### 步骤三：决定是否需要专用 Robot

如果新机器人只需要标准的观测、动作和 Policy 调用流程，可以复用公共 Robot 工作流，不需要创建专用通信类。

如果存在机器人特定的工作站逻辑，则在 `robot.py` 中继承 `Robot`。例如：

- 工作站端动作 shape 校验；
- 机器人专用观测组装；
- 发送动作前的关节名称映射；
- 调用公共 processing 函数；
- 特殊控制流程编排。

不要在 `robot.py` 中实现：

- TCP Socket 收发；
- 厂家 SDK 调用；
- ROS Topic 订阅；
- Policy 模型内部逻辑。

### 步骤四：导出公共类

在 `new_robot/__init__.py` 中导出需要公开使用的类：

```python
from robohub.robots.new_robot.backend import NewRobotBackend
from robohub.robots.new_robot.robot import NewRobot

__all__ = ["NewRobot", "NewRobotBackend"]
```

### 步骤五：创建机器人端入口

机器人端应复用公共 `RobotServer`：

```python
backend = NewRobotBackend(config)
server = RobotServer(
    backend,
    host=config["communication"]["host"],
    port=config["communication"]["port"],
    timeout=config["communication"]["timeout"],
)
server.serve_forever()
```

不要在 `new_robot` 目录中复制 `robot_server.py`。

### 步骤六：创建工作站端入口

工作站端复用公共 `RobotClient`：

```python
client = RobotClient(
    host=robot_ip,
    port=config["communication"]["port"],
    timeout=config["communication"]["timeout"],
)
robot = NewRobot(client, policy)
robot.step()
```

不要在 `new_robot` 目录中复制 `robot_client.py`。

## 6. 新机器人验收清单

新增机器人至少应验证以下内容：

### 配置

- [ ] `default.yaml` 可以成功加载；
- [ ] 关节 `count` 等于 `order` 长度；
- [ ] 关节名称唯一且与映射逻辑一致；
- [ ] 相机内参矩阵 shape 为 3×3；
- [ ] 相机外参矩阵 shape 为 4×4；
- [ ] 深度单位和有效范围明确。

### Backend

- [ ] `get_observation()` 返回标准 `Observation`；
- [ ] 所有观测数组具有正确 dtype 和 shape；
- [ ] 关节顺序与配置一致；
- [ ] `set_action()` 验证动作维度和数值范围；
- [ ] SDK 异常能够被服务端返回给客户端；
- [ ] `close()` 可以安全重复调用。

### 通信

- [ ] Policy 端能够获取完整观测；
- [ ] Robot 端能够收到完整动作；
- [ ] NumPy 数组经过通信后 dtype、shape 和数值保持一致；
- [ ] Policy 进程退出后 RobotServer 继续运行；
- [ ] 新 Policy 客户端可以重新连接；
- [ ] 超时和网络断开不会导致硬件处于不安全状态。

### 安全

- [ ] 关节限制来自真实机器人规格；
- [ ] 动作执行前验证控制模式；
- [ ] 对速度、加速度和力矩设置安全限制；
- [ ] 网络断开时定义明确的停止或保持策略；
- [ ] 真机运行前完成低速、空载和急停测试。

## 7. 设计约束

新增机器人时应遵守以下原则：

1. 通信协议保持公共，不为每种机器人复制 Client 或 Server。
2. 标准 Schema 是机器人端与工作站端之间唯一的数据契约。
3. 厂家 SDK 和 ROS 依赖只能进入 Backend 或更底层 Driver。
4. Policy 不直接依赖 Robot SDK、ROS 或网络协议。
5. 无状态数据处理优先放在公共 `processing` 模块。
6. 机器人专用目录只保留硬件适配、配置和必要的工作站流程。
7. 未明确真实硬件单位、坐标系和安全约束前，不应执行真机动作。
