# RoboHub 通信机制

本文说明 RoboHub 当前 Robot 与 Policy 之间的通信架构、TCP 消息格式、请求处理流程、错误处理方式和现有实现边界。

## 1. 通信架构

RoboHub 使用 TCP Socket 在 Robot 主机和 Policy 主机之间通信。

```text
┌──────────────────────── Policy 主机 ────────────────────────┐
│                                                             │
│  Policy                                                      │
│    ├── encode_obs()                                          │
│    └── get_action()                                          │
│            ▲                         │                       │
│            │ Observation             │ Action                │
│            │                         ▼                       │
│                     PolicyClient                             │
└─────────────────────────────┬───────────────────────────────┘
                              │ TCP
                              │ get_observation / observation
                              │ set_action / ack
                              ▼
┌──────────────────────── Robot 主机 ─────────────────────────┐
│                     RobotServer                              │
│            │                         ▲                       │
│            │ get_observation()       │ set_action()          │
│            ▼                         │                       │
│                         Robot                                │
│    ├── cameras and sensors                                  │
│    └── hardware controllers                                 │
└─────────────────────────────────────────────────────────────┘
```

通信层由三个主要文件构成：

- `src/robohub/communication/protocol.py`：帧协议、序列化和 Socket 收发。
- `src/robohub/communication/robot_server.py`：运行在机器人主机上的 TCP 服务端。
- `src/robohub/communication/policy_client.py`：运行在策略主机上的 TCP 客户端。

`src/robohub/communication/__init__.py` 对外导出 `RobotServer` 和 `PolicyClient`。

## 2. 为什么使用 TCP

当前实现选择 TCP，主要是因为它提供：

- 可靠传输。
- 字节顺序保持。
- 丢包重传。
- 跨主机局域网通信。
- Python 标准库直接支持。

但是，TCP 是字节流协议，不会保留应用层消息边界。一次 `sendall()` 的内容可能被拆成多次 `recv()`，多条消息也可能在接收端一起到达。因此 RoboHub 在 TCP 之上实现了长度前缀帧协议。

## 3. 服务端和客户端角色

### 3.1 Robot Server

Robot Server 运行在连接真实机器人的主机上：

```bash
robohub-robot astribot --host 0.0.0.0 --port 8765
```

参数含义：

- `astribot`：从 `ROBOT_REGISTRY` 选择机器人适配器。
- `--host 0.0.0.0`：监听本机所有网络接口。
- `--port 8765`：监听端口，默认值为 `8765`。
- `--timeout 10.0`：单次 Socket 操作超时，默认 10 秒。
- `--config PATH`：可选的机器人配置路径。

Robot Server 当前使用 IPv4 TCP：

```python
socket.socket(socket.AF_INET, socket.SOCK_STREAM)
```

它设置 `SO_REUSEADDR`，绑定地址后调用 `listen(1)`。当前设计一次处理一个客户端连接，避免多个 Policy 同时控制同一机器人。

### 3.2 Policy Client

Policy Client 运行在部署模型的主机上：

```bash
robohub-policy act --host 192.168.1.10 --port 8765
```

其中 `--host` 必须是 Robot Server 所在主机的局域网 IP，不应填写 Policy 主机自己的 IP。

客户端使用：

```python
socket.create_connection((host, port), timeout=timeout)
```

建立 TCP 连接。连接建立后，同一个 Socket 会在整个控制循环中持续复用，不会为每一步动作重新建立连接。

## 4. 消息帧格式

每条 TCP 消息由外层长度头和 payload 组成：

```text
┌──────────────────────┬──────────────────────────────────────┐
│ 8-byte frame length  │ payload                              │
│ unsigned big-endian  │ metadata length + JSON + array bytes│
└──────────────────────┴──────────────────────────────────────┘
```

### 4.1 外层长度头

外层消息头使用：

```python
struct.Struct("!Q")
```

含义如下：

- `!`：网络字节序，即大端字节序。
- `Q`：8 字节无符号整数。

该整数表示后续 payload 的总字节数，不包含头部自身的 8 字节。

接收端先读取完整的 8 字节，再按该长度读取完整 payload。`_receive_exact()` 会循环调用 `recv()`，直到收满指定字节数，从而正确处理 TCP 拆包。

### 4.2 Payload 格式

payload 由三部分组成：

```text
┌────────────────────────┬──────────────────┬─────────────────┐
│ 4-byte metadata length │ JSON metadata    │ NumPy raw bytes │
│ unsigned big-endian    │ UTF-8            │ contiguous      │
└────────────────────────┴──────────────────┴─────────────────┘
```

前 4 字节使用大端无符号整数，表示 JSON metadata 的字节长度。

JSON 后面紧跟所有 NumPy 数组的连续二进制数据。这样避免将图像和关节数组转换为体积较大的 JSON 数字列表。

### 4.3 完整帧示意

```text
byte offset
0              8              12                 12 + N
│              │               │                     │
▼              ▼               ▼                     ▼
┌──────────────┬───────────────┬─────────────────────┬──────────────┐
│ payload size │ metadata size │ JSON metadata       │ array bytes  │
│ 8 bytes      │ 4 bytes       │ N bytes             │ remaining    │
└──────────────┴───────────────┴─────────────────────┴──────────────┘
```

## 5. JSON 元数据

每条消息的元数据包含：

```json
{
  "version": 1,
  "type": "get_observation",
  "data": {}
}
```

字段含义：

- `version`：协议版本，当前为 `1`。
- `type`：消息类型。
- `data`：消息参数、Observation、Action 或错误信息。

接收端会检查协议版本和消息类型字段。如果版本不是当前支持的版本，消息会被拒绝。

## 6. NumPy 数组序列化

Observation 和 Action 中包含 NumPy 数组。当前实现不使用 `pickle`，而是将每个数组拆分为：

1. JSON 中的数组描述信息。
2. JSON 后的连续原始字节。

数组在 JSON 中会被替换为类似结构：

```json
{
  "__array__": 0,
  "dtype": "<f4",
  "shape": [7]
}
```

字段含义：

- `__array__`：数组在本消息数组列表中的索引。
- `dtype`：NumPy dtype 字符串，包括字节序。
- `shape`：数组形状。

发送端处理流程：

1. 递归遍历 `data` 中的字典、列表和元组。
2. 遇到 NumPy 数组时调用 `np.ascontiguousarray()`。
3. 将 dtype 和 shape 写入 JSON。
4. 将 `tobytes()` 结果按遍历顺序追加到 binary payload。

接收端处理流程：

1. 读取 JSON metadata。
2. 递归寻找带 `__array__` 标记的对象。
3. 根据 dtype 和 shape 计算所需字节数。
4. 从 binary payload 中依次切出对应字节。
5. 使用 `np.frombuffer()` 和 `reshape()` 恢复数组。
6. 调用 `copy()`，使数组不依赖接收消息的临时内存。

这种方式支持嵌套 Observation 和 Action，同时保留数组 shape 和 dtype。

## 7. 当前消息类型

### 7.1 `get_observation`

Policy Client 请求一次机器人观测：

```text
PolicyClient -> RobotServer
```

消息数据为空。

Robot Server 收到后调用：

```python
robot.get_observation()
```

随后使用 `validate_observation(observation, robot.joints_num)` 校验数据。

### 7.2 `observation`

Robot Server 返回观测：

```text
RobotServer -> PolicyClient
```

数据结构：

```python
{
    "observation": observation,
}
```

其中 observation 包含三路 RGB、三路 Depth 和三组关节状态。

### 7.3 `set_action`

Policy Client 发送策略输出：

```text
PolicyClient -> RobotServer
```

数据结构：

```python
{
    "action": action,
}
```

发送前 Policy Client 调用 `validate_action(action)`。Robot Server 接收后再次调用 `validate_action(action)`，然后执行：

```python
robot.set_action(action)
```

双端校验可以尽早发现 Policy 输出错误，并防止未经基础校验的数据直接进入 Robot 适配器。

### 7.4 `reset`

Policy Client 请求机器人复位：

```text
PolicyClient -> RobotServer
```

Robot Server 调用：

```python
robot.reset()
```

成功后返回 `ack`。

### 7.5 `close`

Policy Client 在正常关闭时发送 `close`。Robot Server 返回 `ack`，然后结束当前客户端会话。

这里的 `close` 只结束当前 TCP 控制会话，不会停止 Robot Server 的监听循环，也不会直接调用 `robot.close()`。Robot 资源由 Robot Runner 在进程退出时通过 context manager 释放。

### 7.6 `ack`

表示请求已经成功处理。当前用于响应：

- `set_action`
- `reset`
- `close`

`PolicyClient._expect_ack()` 要求返回类型必须是 `ack`；收到其他类型会抛出 `ProtocolError`。

### 7.7 `error`

Robot Server 处理请求失败时返回：

```python
{
    "message": "error description",
}
```

Policy Client 收到 `error` 后会将服务端错误消息包装为 `ProtocolError` 抛出。

## 8. 一次完整控制循环

`PolicyClient.step()` 实现一次 observation-action 往返：

```text
PolicyClient                  RobotServer                     Robot
     │                            │                             │
     │  get_observation           │                             │
     ├───────────────────────────>│                             │
     │                            │  get_observation()          │
     │                            ├────────────────────────────>│
     │                            │  Observation                │
     │                            │<────────────────────────────┤
     │  observation               │                             │
     │<───────────────────────────┤                             │
     │                            │                             │
     │  policy.get_action(obs)    │                             │
     │  set_action                │                             │
     ├───────────────────────────>│                             │
     │                            │  set_action(action)         │
     │                            ├────────────────────────────>│
     │                            │                             │
     │  ack                       │                             │
     │<───────────────────────────┤                             │
     │                            │                             │
```

具体步骤：

1. Client 发送 `get_observation`。
2. Server 调用 Robot 获取 Observation。
3. Server 校验并发送 `observation`。
4. Client 检查返回消息类型。
5. Client 调用 `policy.get_action(observation)`。
6. Client 校验 Action。
7. Client 发送 `set_action`。
8. Server 再次校验 Action。
9. Server 调用 Robot 执行动作。
10. Server 返回 `ack`。

`PolicyClient.run_forever()` 会持续调用 `step()`，直到客户端关闭或异常终止。

## 9. 连接生命周期

### 9.1 启动

1. Robot Runner 创建 Robot。
2. Robot Runner 创建 `RobotServer`。
3. Server 绑定地址并监听。
4. Policy Runner 创建 Policy。
5. Policy Runner 调用 `policy.load_model()`。
6. Policy Client 连接 Robot Server。
7. Client 进入持续控制循环。

### 9.2 正常关闭

Policy Client 的 `close()` 会尝试：

1. 发送 `close`。
2. 等待 `ack`。
3. 关闭本地 Socket。
4. 将内部 Socket 引用设置为 `None`。

如果连接已经断开，关闭期间的 `ConnectionError`、`OSError` 或 `ProtocolError` 会被忽略，以确保本地 Socket 仍被释放。

Policy Runner 随后调用 `policy.close()`。

Robot Runner 捕获 `KeyboardInterrupt` 后关闭 Server，并通过 Robot context manager 调用 `robot.close()`。

### 9.3 异常断开和超时

Robot Server 为客户端连接设置 Socket timeout。如果发生：

- 客户端断开。
- Socket 读取超时。
- 接收过程中连接关闭。

当前连接处理函数会返回，Server 随后继续等待新的客户端连接。

## 10. 错误处理

### 10.1 协议错误

`ProtocolError` 用于表示：

- 帧超过最大尺寸。
- metadata 不完整或不是合法 JSON。
- 协议版本不支持。
- 数组 metadata 无效。
- 数组数据不完整。
- 存在未被 metadata 使用的额外二进制数据。
- 收到不符合当前流程的消息类型。

### 10.2 Robot Server 错误响应

Server 当前会捕获：

- `KeyError`
- `TypeError`
- `ValueError`
- `ProtocolError`
- `NotImplementedError`

并尽量返回 `error` 消息，而不是立即断开连接。

`ConnectionError` 和 `socket.timeout` 会直接结束当前连接。

未被捕获的硬件异常可能终止当前服务流程。因此具体 Robot 适配器应将可恢复的厂商 SDK 错误转换为语义清晰的异常，并保证硬件安全。

## 11. 消息大小限制

当前最大 frame size 为：

```text
256 MiB
```

该限制包括：

- 4 字节 metadata 长度。
- JSON metadata。
- 所有 NumPy 数组的二进制数据。

发送端和接收端都会检查该限制。限制可以防止错误 shape 或异常请求导致无限制内存分配，但 256 MiB 并不等同于安全的生产配置，应根据实际相机分辨率和部署网络进一步收紧。

## 12. 当前安全特性

当前实现已经具备：

- 不使用 `pickle` 反序列化网络数据。
- 最大消息尺寸限制。
- 协议版本校验。
- JSON 完整性校验。
- NumPy dtype、shape 和实际字节数校验。
- Observation 和 Action 基础格式校验。
- Socket 超时。
- 单客户端控制。
- 服务端错误响应。

但是这些机制只保护通信格式，不构成完整的机器人安全系统。

## 13. 当前限制

目前通信层仍有以下限制：

1. **没有身份认证**：同一网络中能访问端口的主机可能尝试连接。
2. **没有加密**：Observation 和 Action 以明文 TCP 传输。
3. **没有请求 ID**：当前严格按请求/响应顺序工作，不支持并行请求。
4. **没有时间戳**：不能判断 Observation 或 Action 的采集和生成时间。
5. **没有 sequence number**：不能在应用层检测重复帧或过期动作。
6. **没有自动重连**：Policy Client 连接断开后不会自动恢复。
7. **没有心跳**：Server 只能依赖 Socket timeout 感知静默客户端。
8. **没有压缩**：RGB 和 Depth 使用原始 NumPy 字节，网络带宽占用较高。
9. **没有严格实时调度**：循环速度由图像采集、推理、传输和动作执行共同决定。
10. **单客户端串行处理**：不支持监控客户端和控制客户端同时连接。
11. **没有动作过期策略**：通信层没有 deadline 或 action timestamp。
12. **Server 停止唤醒有限**：如果 `accept()` 正在阻塞，跨线程调用 `close()` 的行为依赖操作系统。
13. **错误消息未限制细节**：生产环境中不应向不可信客户端暴露敏感内部信息。

## 14. 真实机器人部署要求

当前通信框架可以用于开发和局域网原型，但连接真实机器人前至少还应实现：

- 网络访问控制或防火墙白名单。
- TLS 或可信隔离网络。
- 客户端身份认证。
- 心跳和控制租约。
- Observation 时间戳与 Action deadline。
- 动作 sequence number 和过期动作丢弃。
- 断连后立即进入安全模式。
- Robot 端独立的速度、位置、力矩和工作空间限制。
- Robot 端急停，不依赖 Policy 主机或 TCP 连接。
- 控制频率监控和超时 watchdog。
- 图像压缩、降采样或更适合高带宽数据的传输方案。

最重要的原则是：Robot 端必须自行保证动作安全。不能假设 Policy 输出永远正确，也不能依赖 TCP 连接关闭来替代急停。

## 15. 本地测试

协议测试使用 `socket.socketpair()` 创建本地连接，不占用固定 TCP 端口：

```bash
pytest tests/test_protocol.py
```

当前测试验证 NumPy 数组可以经过：

```text
send_message -> TCP socket -> receive_message
```

并保持消息类型、dtype、shape 和数组值不变。

后续建议增加：

- TCP 拆包测试。
- 多数组和嵌套字典测试。
- 非法 JSON 测试。
- 超大 frame 测试。
- 截断数组 payload 测试。
- Client/Server 完整 step 测试。
- 超时和断连恢复测试。
- Robot 返回错误时的传播测试。

## 16. 相关公共 API

通信层常用导入：

```python
from robohub.communication import PolicyClient, RobotServer
from robohub.communication.protocol import (
    ProtocolError,
    decode_message,
    encode_message,
    receive_message,
    send_message,
)
```

正常应用通常只需要使用 `PolicyClient` 和 `RobotServer`。直接使用协议函数更适合测试、扩展消息类型或开发其他语言的客户端。
