from typing import Any, List, Protocol, TypedDict

class _Message(TypedDict): ...

class _Choice(TypedDict):
    message: _Message

class _Response(TypedDict):
    choices: List[_Choice]

class _Completions(Protocol):
    def create(self, *, model: str, messages: List[dict[str, Any]], temperature: float = ...) -> _Response: ...

class _Chat:
    completions: _Completions

class OpenAI:
    chat: _Chat

    def __init__(self, *, api_key: str | None = ...) -> None: ...
