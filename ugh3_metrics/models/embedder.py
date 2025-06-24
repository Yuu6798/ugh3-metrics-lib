from typing import Protocol, Any


class EmbedderProtocol(Protocol):
    def encode(self, s: str) -> Any:
        """Encode sentence ``s`` into an embedding vector."""
        ...
