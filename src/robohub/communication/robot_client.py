"""TCP workstation-side communication client."""

import socket
from typing import TypeVar
from uuid import uuid4

from robohub.communication.errors import CommunicationTimeoutError, ProtocolError, RemoteError
from robohub.communication.protocol import Message, MessageHeader, MessageType
from robohub.communication.serialization import MessagePackSerializer
from robohub.schemas import Action, Observation, RawObservation

_MAX_FRAME_SIZE = 64 * 1024 * 1024
ObservationT = TypeVar("ObservationT", Observation, RawObservation)


def _send(sock: socket.socket, data: bytes) -> None:
    sock.sendall(len(data).to_bytes(4, "big") + data)


def _receive(sock: socket.socket) -> bytes:
    header = _read_exact(sock, 4)
    size = int.from_bytes(header, "big")
    if size > _MAX_FRAME_SIZE:
        raise ProtocolError("Message exceeds maximum frame size")
    return _read_exact(sock, size)


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ProtocolError("Peer closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


class RobotClient:
    def __init__(self, *, host: str, port: int = 8765, timeout: float = 5.0) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self._socket: socket.socket | None = None
        self._serializer = MessagePackSerializer()

    def _request(self, message_type: MessageType, payload: object = None) -> object:
        if self._socket is None:
            try:
                self._socket = socket.create_connection((self.host, self.port), self.timeout)
                self._socket.settimeout(self.timeout)
            except TimeoutError as exc:
                raise CommunicationTimeoutError("Connection timed out") from exc
        request_id = str(uuid4())
        _send(self._socket, self._serializer.dumps(Message(MessageHeader(message_type, request_id), payload)))
        response = self._serializer.loads(_receive(self._socket), Message)
        if response.header.request_id != request_id:
            raise ProtocolError("Response request ID does not match")
        if response.header.message_type is MessageType.ERROR:
            raise RemoteError(str(response.payload), error_type=response.metadata.get("error_type"))
        return response.payload

    def get_observation(
        self, expected_type: type[ObservationT] = Observation
    ) -> ObservationT:
        payload = self._request(MessageType.GET_OBSERVATION)
        if not isinstance(payload, expected_type):
            raise ProtocolError(
                f"Expected {expected_type.__name__}, got {type(payload).__name__}"
            )
        return payload

    def set_action(self, action: Action) -> None:
        self._request(MessageType.SET_ACTION, action)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> "RobotClient":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
