"""Temporal expansion around selected seeds (shared CoD / agent primitive)."""

from __future__ import annotations

from typing import Any

from .controller import merge_evidence
from .store import RawMemoryStore

DEFAULT_LAMBDA = 0.06
DEFAULT_BETA = 0.10


def expand_seeds(
    store: RawMemoryStore,
    conversation_id: str,
    seeds: list[dict[str, Any]],
    expand_window: int,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for hit in seeds:
        context = store.expand_context(conversation_id, hit["id"], expand_window)
        expanded = merge_evidence(expanded, context)
    return merge_evidence(seeds, expanded)
