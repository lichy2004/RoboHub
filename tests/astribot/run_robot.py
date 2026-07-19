"""
Run the Astribot-side RoboHub TCP server.

conda activate robohub_astribot
python tests/astribot/run_robot.py \
    --host 0.0.0.0 \
    --port 8765 
"""

import argparse
from pathlib import Path

from robohub.communication import RobotServer
from robohub.robots.astribot import AstribotBackend
from robohub.utils.config import load_config


_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "robohub"
    / "robots"
    / "astribot"
    / "configs"
    / "default.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Astribot RoboHub server")
    parser.add_argument("--host", default="0.0.0.0", help="Address to listen on")
    parser.add_argument("--port", type=int, default=8765, help="TCP port")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout")
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH, help="YAML config path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    backend = AstribotBackend(config, timeout=args.timeout)
    server = RobotServer(
        backend,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
    )
    print(f"Astribot robot server listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping Astribot robot server")
    finally:
        server.close()


if __name__ == "__main__":
    main()
