#!/usr/bin/env python3

'''
python tests/run_fake_robot_server.py \
    --port 8765 \
    --host 0.0.0.0
'''

import argparse

from robohub.communication import RobotServer
from robohub.robots.fake_robot import FakeRobot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a fake Robot server for cross-host communication testing")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--config")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    robot = FakeRobot(args.config)
    server = RobotServer(robot, args.host, args.port, args.timeout)
    print(f"Fake Robot server listening on {args.host}:{args.port}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Fake Robot server stopped.", flush=True)
    finally:
        server.close()


if __name__ == "__main__":
    main()
