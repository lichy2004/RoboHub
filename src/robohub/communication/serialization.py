"""Serialization for protocol messages and standard schemas."""

from dataclasses import fields, is_dataclass
from enum import Enum
import json
from typing import Any, Protocol, TypeVar

import msgpack
import numpy as np

from robohub.communication.errors import ProtocolError
from robohub.communication.protocol import Message, MessageHeader, MessageType
from robohub.schemas import Action, Observation, RawImage, RawObservation

T = TypeVar("T")


class Serializer(Protocol):
    def dumps(self, value: object) -> bytes: ...

    def loads(self, data: bytes, expected_type: type[T]) -> T: ...


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"__ndarray__": True, "dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__"):
        return np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"])
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _schema(value: Any, expected_type: type[T]) -> T:
    if expected_type is Message:
        header = MessageHeader(
            message_type=MessageType(value["header"]["message_type"]),
            request_id=value["header"]["request_id"],
            protocol_version=value["header"]["protocol_version"],
        )
        return Message(header, value.get("payload"), value.get("metadata", {}))  # type: ignore[return-value]
    if expected_type in (Observation, Action):
        return expected_type(**value)  # type: ignore[return-value]
    return value


class SchemaSerializer:
    def dumps(self, value: object) -> bytes:
        try:
            return json.dumps(_encode(value), separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"Unable to serialize value: {exc}") from exc

    def loads(self, data: bytes, expected_type: type[T]) -> T:
        try:
            return _schema(_decode(json.loads(data.decode("utf-8"))), expected_type)
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"Unable to deserialize value: {exc}") from exc


_DATACLASS_TYPES = {
    cls.__name__: cls
    for cls in (Message, MessageHeader, Action, Observation, RawImage, RawObservation)
}


def _encode_binary(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": array.dtype.str,
            "shape": array.shape,
            "data": array.tobytes(),
        }
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            "__dataclass__": type(value).__name__,
            "fields": {
                field.name: _encode_binary(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, dict):
        return {str(key): _encode_binary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_binary(item) for item in value]
    return value


def _decode_binary(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__"):
        return (
            np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
            .reshape(value["shape"])
            .copy()
        )
    if isinstance(value, dict) and "__dataclass__" in value:
        class_name = value["__dataclass__"]
        cls = _DATACLASS_TYPES[class_name]
        decoded_fields = {
            key: _decode_binary(item) for key, item in value["fields"].items()
        }
        if cls is MessageHeader:
            decoded_fields["message_type"] = MessageType(
                decoded_fields["message_type"]
            )
        return cls(**decoded_fields)
    if isinstance(value, dict):
        return {key: _decode_binary(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_binary(item) for item in value]
    return value


class MessagePackSerializer:
    """Encode schemas without expanding ndarrays into Python lists."""

    def dumps(self, value: object) -> bytes:
        try:
            return msgpack.packb(_encode_binary(value), use_bin_type=True)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"Unable to serialize value: {exc}") from exc

    def loads(self, data: bytes, expected_type: type[T]) -> T:
        try:
            value = _decode_binary(msgpack.unpackb(data, raw=False))
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Expected {expected_type.__name__}, got {type(value).__name__}"
                )
            return value
        except (
            msgpack.exceptions.UnpackException,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ProtocolError(f"Unable to deserialize value: {exc}") from exc
