#!/usr/bin/env python3

'''
python tests/run_fake_policy_client.py \
    --port 8765 \
    --host 10.40.5.70
'''

import argparse
import time

from robohub.communication import PolicyClient
from robohub.policies.fake_policy import FakePolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a fake Policy client for cross-host communication testing")
    parser.add_argument("--host", default="10.40.5.70")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = FakePolicy(args.config)
    client = PolicyClient(policy, args.host, args.port, args.timeout)
    try:
        policy.load_model()
        client.connect()
        print(f"Connected to Fake Robot server at {args.host}:{args.port}", flush=True)
        client.reset_robot()
        print("Reset request acknowledged.", flush=True)
        for step in range(1, args.steps + 1):
            observation = client.get_observation()
            action = policy.get_action(observation)
            client.set_action(action)
            print(f"Step {step}/{args.steps}: observation and action exchange succeeded", flush=True)
            if args.interval > 0 and step < args.steps:
                time.sleep(args.interval)
        print(f"Communication test passed: {args.steps} steps completed.", flush=True)
    finally:
        client.close()
        policy.close()


if __name__ == "__main__":
    main()
