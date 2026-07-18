# Policy 开发指南

本文说明 RoboHub 中 Policy 组件的职责、输入输出接口、目录结构，以及新增策略模型的完整流程。

## 1. Policy 的职责

Policy 是观测到动作的算法适配层，负责：

1. 读取模型配置并加载模型和权重。
2. 将统一的 Robot `Observation` 转换为模型输入。
3. 执行推理并将模型输出转换为统一的 `Action`。
4. 管理模型设备、缓存、历史状态及推理资源。
5. 为训练和评估提供独立入口。

Policy 不应直接调用机器人厂商 SDK，也不应直接管理真实机器人硬件。Robot 与 Policy 通过统一类型及 Socket 通信解耦。

## 2. Policy 基础接口

所有策略必须继承：

```python
from robohub.policies.base import Policy
```

基类位于 `src/robohub/policies/base.py`。

### 2.1 初始化

基类构造函数接收配置：

```python
super().__init__(config)
```

初始化后具有：

- `config: Mapping[str, Any]`：只读策略配置。
- `model: Any`：模型对象，加载前默认为 `None`。

构造函数适合保存配置和建立轻量状态，不建议在 `__init__()` 中加载大型权重。模型加载应放在 `load_model()` 中，使实例创建、配置检查和错误定位更清晰。

### 2.2 `load_model()`

```python
def load_model(self) -> None:
    ...
```

负责：

- 创建模型结构。
- 从 `model_path` 加载 checkpoint。
- 将模型移动到指定设备。
- 切换到推理模式。
- 初始化归一化统计量、tokenizer 或其他预处理组件。

实现要求：

- 模型文件不存在时提供明确错误。
- 校验 checkpoint 与模型结构兼容性。
- 不要静默使用随机初始化模型执行真实机器人控制。
- PyTorch 模型应调用 `eval()`，推理时使用无梯度模式。
- 明确 CPU、CUDA 或其他加速设备的选择规则。
- 重复调用时避免重复占用设备内存，或者明确禁止重复调用。

### 2.3 `encode_obs()`

```python
def encode_obs(self, obs: Observation) -> Any:
    ...
```

将 RoboHub 的统一 Observation 转换为模型输入。典型处理包括：

- 校验 Observation。
- 选择和重排相机。
- RGB resize、crop、归一化和维度转换。
- 深度值缩放或 mask 处理。
- 按模型要求拼接关节状态。
- 添加 batch、时间或历史序列维度。
- 转换为模型框架 tensor 并移动到计算设备。

输入 Observation 格式：

```python
{
    "rgb": {
        "head": np.ndarray,         # (H, W, 3)
        "wrist_left": np.ndarray,   # (H, W, 3)
        "wrist_right": np.ndarray,  # (H, W, 3)
    },
    "depth": {
        "head": np.ndarray,         # (H, W, 1)
        "wrist_left": np.ndarray,   # (H, W, 1)
        "wrist_right": np.ndarray,  # (H, W, 1)
    },
    "joints_position": np.ndarray,
    "joints_velocity": np.ndarray,
    "joints_torque": np.ndarray,
}
```

Policy 必须明确训练时使用的关节顺序，并确保其与 Robot 的 `joints_order` 一致。如果训练数据顺序不同，应在这里执行显式映射，不能假设顺序相同。

### 2.4 `get_action()`

```python
def get_action(self, obs: Observation) -> Action:
    ...
```

完成一次完整推理：

1. 检查模型已加载。
2. 调用 `encode_obs(obs)`。
3. 执行模型前向计算或动作采样。
4. 反归一化模型输出。
5. 将输出拆分为统一 Action 字段。
6. 校验 Action 后返回。

返回格式：

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

要求：

- 所有字段必须存在。
- 每个值必须是一维 NumPy 数组。
- 每个字段长度必须与目标 Robot 的 `action_dimensions` 对齐。
- 不支持的组件使用空数组，例如 `np.empty(0, dtype=np.float32)`。
- 输出不能包含 NaN 或 Inf。
- 动作必须采用目标 Robot 约定的位置、速度、力矩或增量语义。
- 如果模型输出动作块，需定义当前循环发送第一个动作、整块缓存执行或重新规划的策略。

当前代码库没有可供 Policy 调用的通用 `validate_action()`。新增 Policy 应自行校验上述约束，或者先在公共模块中实现验证函数，再在 Policy 和 Robot 两端复用。不要在文档或实现中假设 `PolicyClient` 会自动验证动作；它当前只负责传输。

### 2.5 `close()`

```python
def close(self) -> None:
    ...
```

基类默认将 `self.model` 设置为 `None`。子类可以扩展它来释放：

- GPU/加速器资源。
- 推理服务连接。
- 后台预取线程。
- 视频或日志写入器。
- 历史动作和缓存状态。

如果扩展 `close()`，应确保最终清理 `self.model`，并尽量支持重复调用。

## 3. Policy 目录结构

每个策略使用独立子包：

```text
src/robohub/policies/my_policy/
├── __init__.py
├── policy.py
├── train.py
├── eval.py
├── model/
│   └── __init__.py
├── config/
│   └── default.yaml
└── data/
```

各部分职责：

- `policy.py`：RoboHub Policy 适配器，负责模型加载、编码和推理。
- `model/`：模型结构、损失函数和模型内部组件。
- `config/`：默认训练与推理配置。
- `data/`：数据集读取和转换代码；真实数据集不应提交到仓库。
- `train.py`：训练入口。
- `eval.py`：离线或在线评估入口。
- `__init__.py`：导出 Policy 类。

模型权重、数据集和生成产物不应提交到 Git，也不应默认打入 Python wheel。

## 4. 默认配置

推荐至少包含：

```yaml
name: my_policy
model_path: null
device: cpu
image_size: [224, 224]
camera_names:
  - head
  - wrist_left
  - wrist_right
action_dimensions:
  left_arm: 2
  left_gripper: 1
  right_arm: 2
  right_gripper: 1
  torso: 0
  head: 0
  base: 0
```

根据模型可增加：

- checkpoint 类型和版本。
- 输入图像均值与标准差。
- 关节状态归一化统计量。
- 动作反归一化统计量。
- 历史窗口和动作块长度。
- 推理精度，如 float32、float16 或 bfloat16。
- 采样温度、随机种子或 diffusion steps。
- 语言指令、tokenizer 或任务描述配置。
- 推理频率。

配置中的动作维度和关节顺序必须与目标 Robot 兼容。

## 5. 新增 Policy

下面以 `MyPolicy` 为例。

### 第一步：创建目录

```text
src/robohub/policies/my_policy/
├── __init__.py
├── policy.py
├── train.py
├── eval.py
├── model/
│   └── __init__.py
├── config/
│   └── default.yaml
└── data/
```

Python 包名使用 `snake_case`，类名使用 `PascalCase`，建议以 `Policy` 结尾。

### 第二步：编写配置

在 `config/default.yaml` 中定义模型路径、设备、图像处理参数、动作维度和模型特有参数。不要在配置中提交密钥或机器相关的绝对路径。

### 第三步：实现 Policy

以下是框架示例，模型 API 需要替换为真实实现：

```python
from pathlib import Path
from typing import Any

import numpy as np

from robohub.policies.base import Policy
from robohub.utils.config import load_config
from robohub.utils.types import Action, Observation


class MyPolicy(Policy):
    def __init__(self, config_path: str | Path | None = None) -> None:
        path = config_path or Path(__file__).parent / "config" / "default.yaml"
        super().__init__(load_config(path))
        self.device = self.config.get("device", "cpu")

    def load_model(self) -> None:
        model_path = self.config["model_path"]
        if not model_path:
            raise ValueError("model_path must be configured")
        self.model = self._build_model()
        self._load_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()

    def encode_obs(self, obs: Observation) -> Any:
        validate_observation(obs)
        return self._build_model_inputs(obs)

    def get_action(self, obs: Observation) -> Action:
        if self.model is None:
            raise RuntimeError("Policy model has not been loaded")
        inputs = self.encode_obs(obs)
        output = self._run_inference(inputs)
        action = self._decode_action(output)
        return action

    def close(self) -> None:
        self._clear_runtime_cache()
        super().close()
```

不要复制示例中的虚构内部方法；应根据真实模型实现 `_build_model_inputs()`、推理和动作解码逻辑。

### 第四步：实现动作映射

模型通常输出一个扁平向量。必须按照配置显式拆分：

```python
left_arm = output[0:2]
left_gripper = output[2:3]
right_arm = output[3:5]
right_gripper = output[5:6]
```

然后构造完整 Action。即使 Robot 不支持 torso、head 或 base，也必须返回空的一维数组：

```python
empty = np.empty(0, dtype=np.float32)
```

不要依赖字典遍历顺序推断模型输出布局；动作布局应在配置或代码中明确记录并测试。

### 第五步：导出类

在 `src/robohub/policies/my_policy/__init__.py` 中：

```python
from robohub.policies.my_policy.policy import MyPolicy

__all__ = ["MyPolicy"]
```

### 第六步：注册策略

修改 `src/robohub/policies/__init__.py`：

```python
from robohub.policies.my_policy import MyPolicy

POLICY_REGISTRY = {
    "act": ACTPolicy,
    "pi05": PI05Policy,
    "my_policy": MyPolicy,
}
```

注册后可以运行：

```bash
robohub-policy my_policy --host 192.168.1.10 --port 8765
```

### 第七步：实现训练入口

`train.py` 应负责：

- 读取配置和命令行覆盖参数。
- 创建数据集和 DataLoader。
- 构造模型、优化器及学习率调度器。
- 保存 checkpoint、配置和归一化统计量。
- 支持恢复训练。

训练输出必须包含推理所需的全部信息，尤其是图像预处理、关节顺序及动作归一化参数。

### 第八步：实现评估入口

`eval.py` 应根据部署风险区分：

- 离线评估：读取数据集，计算损失或动作误差，不连接机器人。
- 在线评估：通过 Policy Client 连接 Robot Server，必须具备明确安全确认、超时和停止机制。

建议首先完成离线测试，再进行仿真测试，最后才连接真实硬件。

### 第九步：确保配置被打包

当前 `pyproject.toml` 会包含 `policies/*/config/*.yaml`。如果策略需要其他运行时静态文件，应同步配置 package data。模型权重应由部署流程提供，而不是打入 wheel。

### 第十步：添加测试

至少测试：

1. 默认配置可读取。
2. 缺少或无效 checkpoint 时错误明确。
3. `encode_obs()` 的相机顺序、shape、dtype 和归一化正确。
4. 模型输出可准确拆分为七个 Action 字段。
5. 不支持的动作字段返回空数组。
6. NaN、Inf 和错误动作维度会被拒绝。
7. `close()` 能释放模型及缓存。

使用轻量 fake model 测试，不应在单元测试中下载权重或要求 GPU。

## 6. 与 Robot 的通信流程

策略主机启动：

```bash
robohub-policy my_policy --host 192.168.1.10 --port 8765
```

`PolicyClient.run_forever()` 的循环为：

1. 向 Robot Server 请求 Observation。
2. 接收并还原 NumPy 数组。
3. 调用 `policy.get_action(observation)`。
4. 将 Action 发送到 Robot Server。
5. 等待 Robot Server 返回 `ack`。
6. 重复执行，直到连接关闭、超时或发生错误。

当前 `PolicyClient` 不校验 Observation 或 Action，也不在 `run_forever()` 中提供主动停止条件、重连、频率控制或自动资源清理。`close()` 会尝试向 Robot Server 发送 `close` 并等待 `ack`，但调用方仍应在 `finally` 中分别关闭 client 和 policy。需要固定控制频率时，应结合模型推理耗时、网络延迟和机器人控制周期进行设计，不应只使用未经测量的 `sleep()`。

## 7. Robot 与 Policy 兼容性

一个 Policy 能否控制某个 Robot，至少取决于：

- 相机名称和数量一致。
- RGB/Depth 的单位、颜色空间和预处理一致。
- `joints_order` 与训练数据顺序一致，或存在显式映射。
- Action 七个字段的维度一致。
- 动作控制语义一致，例如位置不能直接作为速度发送。
- 归一化和反归一化参数与训练时一致。
- 控制频率与动作块执行方式一致。

建议后续在配置中加入 Robot/Policy schema 版本和兼容性检查。在此之前，每个新增 Policy 都必须用目标 Robot 配置进行集成测试。

## 8. 实现检查清单

新增 Policy 前确认：

- [ ] 继承 `Policy` 并实现三个抽象方法。
- [ ] 模型在 `load_model()` 中加载，而不是在构造函数中加载。
- [ ] checkpoint 不存在或不兼容时明确失败。
- [ ] `encode_obs()` 与训练时预处理完全一致。
- [ ] 相机和关节顺序有明确配置或映射。
- [ ] `get_action()` 返回全部七个一维 NumPy 数组字段。
- [ ] 动作维度和控制语义与目标 Robot 一致。
- [ ] 模型输出经过正确反归一化，且不含 NaN/Inf。
- [ ] `close()` 释放模型、设备缓存和后台资源。
- [ ] 已在 `POLICY_REGISTRY` 注册，且所有注册项都能从当前源码树导入。
- [ ] 若需要 CLI，`robohub.runtime.policy_runner` 已实际存在并完成参数解析、模型加载及资源清理。
- [ ] 配置文件会包含在安装包中，权重和数据不会被打包。
- [ ] 使用 fake model 完成编码和动作映射测试。
