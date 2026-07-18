# RoboHub

RoboHub provides common interfaces and a network runtime for robot hardware and policy models.

## Install

```bash
python -m pip install -e .
```


## Architecture

- `robohub.robots`: the `Robot` interface and Astribot, CobotMagic, and Unitree G1 adapters.
- `robohub.policies`: the `Policy` interface and ACT and PI05 adapters.
- `robohub.communication`: framed TCP transport for NumPy observations and actions.
- `robohub.runtime`: command-line processes for robot and policy hosts.

The included adapters define integration boundaries. Hardware SDK calls, model implementations, joint definitions, and production safety behavior must be configured for the target deployment before controlling a real robot.
