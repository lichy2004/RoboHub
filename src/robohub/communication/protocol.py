"""Versioned request and response protocol definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

PROTOCOL_VERSION = 2


class MessageType(str, Enum):
    GET_OBSERVATION = "get_observation"
    SET_ACTION = "set_action"
    RESPONSE = "response"
    ERROR = "error"


@dataclass(slots=True)
class MessageHeader:
    message_type: MessageType
    request_id: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(slots=True)
class Message:
    header: MessageHeader
    payload: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
