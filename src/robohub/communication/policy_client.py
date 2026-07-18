from __future__ import annotations

import socket

from robohub.communication.protocol import ProtocolError, receive_message, send_message
from robohub.policies.base import Policy
from robohub.utils.types import Action, Observation


class PolicyClient:
    def __init__(self, policy: Policy, host: str, port: int = 8765, timeout: float = 10.0) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: socket.socket | None = None

    def connect(self) -> None:
        self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout)

    def get_observation(self) -> Observation:
        sock = self._require_socket()
        send_message(sock, "get_observation")
        message_type, data = receive_message(sock)
        return data["observation"]

    def set_action(self, action: Action) -> None:
        sock = self._require_socket()
        send_message(sock, "set_action", {"action": action})
        self._expect_ack(sock)

    def run_forever(self) -> None:
        if self._socket is None:
            self.connect()
        while self._socket is not None:
            observation = self.get_observation()
            action = self.policy.get_action(observation)
            self.set_action(action)

    def reset_robot(self) -> None:
        sock = self._require_socket()
        send_message(sock, "reset")
        self._expect_ack(sock)

    def close(self) -> None:
        if self._socket is not None:
            try:
                send_message(self._socket, "close")
                self._expect_ack(self._socket)
            except (ConnectionError, OSError, ProtocolError):
                pass
            self._socket.close()
            self._socket = None

    def _require_socket(self) -> socket.socket:
        if self._socket is None:
            raise RuntimeError("Policy client is not connected")
        return self._socket

    @staticmethod
    def _expect_ack(sock: socket.socket) -> None:
        message_type, data = receive_message(sock)
        if message_type == "error":
            raise ProtocolError(data["message"])
        if message_type != "ack":
            raise ProtocolError(f"Expected ack, received {message_type}")
