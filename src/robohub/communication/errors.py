"""Communication error hierarchy."""


class CommunicationError(RuntimeError):
    """Base error for communication failures."""


class ProtocolError(CommunicationError):
    """Raised when a peer sends an invalid protocol message."""


class CommunicationTimeoutError(CommunicationError):
    """Raised when a communication operation exceeds its deadline."""


class RemoteError(CommunicationError):
    """Raised when the remote endpoint reports an operation failure."""

    def __init__(self, message: str, *, error_type: str | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
