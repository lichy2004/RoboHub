from __future__ import annotations

import socket

from robohub.communication.protocol import ProtocolError, receive_message, send_message
from robohub.robots.base import Robot


class RobotServer:
    def __init__(self, robot: Robot, host: str = "0.0.0.0", port: int = 8765, timeout: float = 10.0) -> None:
        self.robot = robot
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._running = False

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            self._socket = server
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(1)
            self._running = True
            while self._running:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(self.timeout)
                    self._serve_connection(connection)

    def _serve_connection(self, connection: socket.socket) -> None:
        while self._running:
            try:
                message_type, data = receive_message(connection)
                if message_type == "get_observation":
                    observation = self.robot.get_observation()
                    send_message(connection, "observation", {"observation": observation})
                elif message_type == "set_action":
                    action = data["action"]
                    self.robot.set_action(action)
                    send_message(connection, "ack")
                elif message_type == "reset":
                    self.robot.reset()
                    send_message(connection, "ack")
                elif message_type == "close":
                    send_message(connection, "ack")
                    return
                else:
                    raise ProtocolError(f"Unsupported message type: {message_type}")
            except (ConnectionError, socket.timeout):
                return
            except (KeyError, TypeError, ValueError, ProtocolError, NotImplementedError) as error:
                send_message(connection, "error", {"message": str(error)})

    def close(self) -> None:
        self._running = False
        if self._socket is not None:
            self._socket.close()
