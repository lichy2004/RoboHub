"""
Run the robot-side TCP server for cross-machine testing.

conda activate robohub_robot
python tests/mytest/run_robot.py \
    --host 0.0.0.0 \
    --port 8765
"""

import argparse

from robohub.communication import RobotServer
from robohub.robots.my_robot import MyRobotBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RoboHub robot server")
    parser.add_argument("--host", default="0.0.0.0", help="Address to listen on")
    parser.add_argument("--port", type=int, default=8765, help="TCP port to listen on")
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Connection timeout in seconds"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = RobotServer(
        MyRobotBackend(),
        host=args.host,
        port=args.port,
        timeout=args.timeout,
    )
    print(f"Robot server listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping robot server")
    finally:
        server.close()


if __name__ == "__main__":
    main()
