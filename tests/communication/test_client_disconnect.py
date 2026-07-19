"""Regression coverage for disconnected policy clients."""

import socket
from threading import Thread
import time

from robohub.communication import RobotClient, RobotServer
from robohub.robots.my_robot import MyRobotBackend


def test_server_accepts_new_client_after_disconnect() -> None:
    backend = MyRobotBackend()
    server = RobotServer(backend, host="127.0.0.1", port=0, timeout=1.0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    deadline = time.monotonic() + 2.0
    while server._socket is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server._socket is not None
    port = server._socket.getsockname()[1]

    disconnected_client = socket.create_connection(("127.0.0.1", port))
    disconnected_client.sendall((100).to_bytes(4, "big") + b"incomplete")
    disconnected_client.close()

    client = RobotClient(host="127.0.0.1", port=port, timeout=10.0)
    observation = client.get_observation()

    assert "head" in observation.rgb
    assert thread.is_alive()

    client.close()
    server.close()
