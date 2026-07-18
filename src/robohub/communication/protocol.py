from __future__ import annotations

import json
import socket
import struct
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1
HEADER = struct.Struct("!Q")
MAX_FRAME_SIZE = 256 * 1024 * 1024


class ProtocolError(RuntimeError):
    pass


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Socket closed while receiving a message")
        chunks.extend(chunk)
    return bytes(chunks)


def encode_message(message_type: str, data: dict[str, Any] | None = None) -> bytes:
    arrays: list[bytes] = []

    def encode_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            contiguous = np.ascontiguousarray(value)
            index = len(arrays)
            arrays.append(contiguous.tobytes())
            return {"__array__": index, "dtype": contiguous.dtype.str, "shape": contiguous.shape}
        if isinstance(value, dict):
            return {key: encode_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode_value(item) for item in value]
        return value

    metadata = json.dumps(
        {"version": PROTOCOL_VERSION, "type": message_type, "data": encode_value(data or {})},
        separators=(",", ":"),
    ).encode("utf-8")
    payload = struct.pack("!I", len(metadata)) + metadata + b"".join(arrays)
    if len(payload) > MAX_FRAME_SIZE:
        raise ProtocolError("Message exceeds maximum frame size")
    return HEADER.pack(len(payload)) + payload


def decode_message(payload: bytes) -> tuple[str, dict[str, Any]]:
    if len(payload) < 4:
        raise ProtocolError("Message metadata length is missing")
    metadata_size = struct.unpack("!I", payload[:4])[0]
    metadata_end = 4 + metadata_size
    if metadata_end > len(payload):
        raise ProtocolError("Message metadata is incomplete")
    try:
        metadata = json.loads(payload[4:metadata_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("Message metadata is invalid") from error
    if metadata.get("version") != PROTOCOL_VERSION or not isinstance(metadata.get("type"), str):
        raise ProtocolError("Unsupported protocol version or message type")

    binary = memoryview(payload)[metadata_end:]
    offset = 0

    def decode_value(value: Any) -> Any:
        nonlocal offset
        if isinstance(value, dict) and "__array__" in value:
            try:
                dtype = np.dtype(value["dtype"])
                shape = tuple(int(size) for size in value["shape"])
                byte_count = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            except (KeyError, TypeError, ValueError) as error:
                raise ProtocolError("Array metadata is invalid") from error
            if byte_count < 0 or offset + byte_count > len(binary):
                raise ProtocolError("Array payload is incomplete")
            array = np.frombuffer(binary[offset : offset + byte_count], dtype=dtype).reshape(shape).copy()
            offset += byte_count
            return array
        if isinstance(value, dict):
            return {key: decode_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode_value(item) for item in value]
        return value

    data = decode_value(metadata.get("data", {}))
    if offset != len(binary) or not isinstance(data, dict):
        raise ProtocolError("Message contains unexpected binary data")
    return metadata["type"], data


def send_message(sock: socket.socket, message_type: str, data: dict[str, Any] | None = None) -> None:
    sock.sendall(encode_message(message_type, data))


def receive_message(sock: socket.socket) -> tuple[str, dict[str, Any]]:
    frame_size = HEADER.unpack(_receive_exact(sock, HEADER.size))[0]
    if frame_size > MAX_FRAME_SIZE:
        raise ProtocolError("Incoming message exceeds maximum frame size")
    return decode_message(_receive_exact(sock, frame_size))
