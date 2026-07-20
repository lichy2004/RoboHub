"""TCP robot-side service."""

import socket

from robohub.backends.base import RobotBackend
from robohub.communication.errors import ProtocolError
from robohub.communication.protocol import (
    PROTOCOL_VERSION,
    Message,
    MessageHeader,
    MessageType,
)
from robohub.communication.robot_client import _MAX_FRAME_SIZE, _read_exact, _send
from robohub.communication.serialization import MessagePackSerializer
from robohub.schemas import Action


class RobotServer:
    def __init__(self, backend: RobotBackend, *, host: str = "127.0.0.1", port: int = 8765, timeout: float = 5.0) -> None:
        self.backend, self.host, self.port, self.timeout = backend, host, port, timeout
        self._socket: socket.socket | None = None
        self._closed = False
        self._serializer = MessagePackSerializer()

    def _handle(self, request: Message) -> Message:
        if request.header.protocol_version != PROTOCOL_VERSION:
            raise ProtocolError("Unsupported protocol version")
        if request.header.message_type is MessageType.GET_OBSERVATION:
            payload = self.backend.get_observation()
        elif request.header.message_type is MessageType.SET_ACTION:
            if not isinstance(request.payload, Action):
                raise ProtocolError("SET_ACTION payload must be an Action")
            self.backend.set_action(request.payload)
            payload = None
        else:
            raise ProtocolError("Unsupported request type")
        return Message(MessageHeader(MessageType.RESPONSE, request.header.request_id), payload)

    def _serve_connection(self, connection: socket.socket) -> None:
        connection.settimeout(self.timeout)
        while not self._closed:
            request: Message | None = None
            try:
                data = _read_exact(connection, 4)
                size = int.from_bytes(data, "big")
                if size > _MAX_FRAME_SIZE:
                    raise ProtocolError("Message exceeds maximum frame size")
                request = self._serializer.loads(_read_exact(connection, size), Message)
                response = self._handle(request)
            except (ConnectionError, socket.timeout, OSError):
                break
            except Exception as exc:
                request_id = request.header.request_id if request is not None else "unknown"
                response = Message(
                    MessageHeader(MessageType.ERROR, request_id),
                    str(exc),
                    {"error_type": type(exc).__name__},
                )

            try:
                _send(connection, self._serializer.dumps(response))
            except (ConnectionError, socket.timeout, OSError):
                break

    def serve_forever(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.listen(1)
        while not self._closed:
            try:
                connection, _ = self._socket.accept()
            except OSError:
                if self._closed:
                    break
                raise
            with connection:
                self._serve_connection(connection)

    def close(self) -> None:
        self._closed = True
        if self._socket is not None:
            self._socket.close()
        self.backend.close()

    def __enter__(self) -> "RobotServer":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
