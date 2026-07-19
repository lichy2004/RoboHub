# RoboHub

RoboHub provides common interfaces and a TCP network runtime for robot hardware and policy models. The robot machine publishes observations and receives actions, while the policy machine performs inference through a `RobotClient`.

## Architecture

```text
Robot machine                                      Policy machine
MySDK -> MyRobotBackend -> RobotServer <- TCP -> RobotClient -> MyRobot -> MyPolicy
```

- `robohub.communication`: framed TCP transport for observations and actions.
- `robohub.robots`: robot interfaces, hardware backends, and workflow orchestration.
- `robohub.policies`: policy interfaces and implementations.
- `robohub.schemas`: shared `Observation` and `Action` data contracts.

## Installation

Install the repository on both machines. The robot and policy machines use separate Conda environments.

### Robot machine

```bash
conda activate robohub_astribot
python -m pip install -e .
```

### Policy machine

```bash
conda activate robohub_policy
python -m pip install -e .
```

Run installation commands from the repository root.