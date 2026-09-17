from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class RawMemoryEntry:
    """A minimally transformed conversational utterance."""

    id: str
    conversation_id: str
    speaker: str
    text: str
    timestamp: str
    embedding: list[float]

    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        value = asdict(self)
        if not include_embedding:
            value.pop("embedding")
        return value


@dataclass(slots=True)
class ControllerDecision:
    status: str
    query: str | None = None
    reason: str | None = None
    expand_entry_id: str | None = None


@dataclass(slots=True)
class AgentState:
    original_query: str
    retrieval_history: list[dict[str, Any]]
    current_evidence: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    step: int = 0
