from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .schema import RawMemoryEntry


_ENTRY_ID_RE = re.compile(r"^(?P<conversation>.+)/session_(?P<session>\d+)/turn_(?P<turn>\d+)$")


class LocalTextEmbedder:
    """Standalone MiniLM embedder (no LightMem dependency)."""

    def __init__(self, model: str, device: str = "cuda") -> None:
        self.model_name = model
        try:
            self.model = SentenceTransformer(
                model, device=device, local_files_only=True
            )
        except (OSError, ValueError):
            self.model = SentenceTransformer(model, device=device)

    def embed(self, text: str) -> list[float]:
        result = self.model.encode(text, convert_to_numpy=True)
        if isinstance(result, np.ndarray):
            return result.tolist()
        return list(result)


def create_embedder(model: str, device: str = "cuda") -> LocalTextEmbedder:
    try:
        return LocalTextEmbedder(model=model, device=device)
    except Exception:
        if device != "cpu":
            return LocalTextEmbedder(model=model, device="cpu")
        raise


def entry_position(entry_id: str) -> tuple[int, int]:
    match = _ENTRY_ID_RE.match(entry_id)
    if not match:
        raise ValueError(f"Invalid raw memory entry ID: {entry_id}")
    return int(match.group("session")), int(match.group("turn"))


def _result(entry: RawMemoryEntry, score: float, latency: float = 0.0) -> dict[str, Any]:
    value = entry.to_dict(include_embedding=False)
    value.update(score=float(score), retrieval_latency=float(latency))
    return value


class RawMemoryStore:
    """Dense retrieval and temporal expansion over unmodified utterances."""

    def __init__(
        self,
        embedder: Any,
        store_dir: str | Path | None = None,
        entries: Iterable[RawMemoryEntry] | None = None,
    ) -> None:
        self.embedder = embedder
        self.store_dir = Path(store_dir).expanduser() if store_dir else None
        self.client = QdrantClient(path=str(self.store_dir)) if self.store_dir else None
        self._embed_lock = threading.Lock()
        self._entries: dict[str, list[RawMemoryEntry]] = defaultdict(list)
        if entries:
            for entry in entries:
                self._entries[entry.conversation_id].append(entry)
            for values in self._entries.values():
                values.sort(key=lambda item: entry_position(item.id))

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def _load_entries(self, conversation_id: str) -> list[RawMemoryEntry]:
        if conversation_id in self._entries:
            return self._entries[conversation_id]
        if self.client is None:
            return []

        points = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=conversation_id,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            points.extend(batch)
            if offset is None:
                break
        entries = [
            RawMemoryEntry(
                id=str(point.payload["id"]),
                conversation_id=str(point.payload["conversation_id"]),
                speaker=str(point.payload["speaker"]),
                text=str(point.payload["text"]),
                timestamp=str(point.payload["timestamp"]),
                embedding=list(point.vector),
            )
            for point in points
        ]
        entries.sort(key=lambda item: entry_position(item.id))
        self._entries[conversation_id] = entries
        return entries

    def search_memory(
        self, conversation_id: str, query: str, top_k: int
    ) -> list[dict[str, Any]]:
        start = time.perf_counter()
        with self._embed_lock:
            query_vector = self.embedder.embed(query)

        if self.client is not None:
            response = self.client.query_points(
                collection_name=conversation_id,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            latency = time.perf_counter() - start
            return [
                {
                    "id": str(point.payload["id"]),
                    "conversation_id": str(point.payload["conversation_id"]),
                    "speaker": str(point.payload["speaker"]),
                    "text": str(point.payload["text"]),
                    "timestamp": str(point.payload["timestamp"]),
                    "score": float(point.score),
                    "retrieval_latency": latency,
                }
                for point in response.points
            ]

        entries = self._load_entries(conversation_id)
        q = np.asarray(query_vector, dtype=float)
        q_norm = np.linalg.norm(q)
        scored = []
        for entry in entries:
            vector = np.asarray(entry.embedding, dtype=float)
            denominator = q_norm * np.linalg.norm(vector)
            score = float(np.dot(q, vector) / denominator) if denominator else 0.0
            scored.append((entry, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        latency = time.perf_counter() - start
        return [_result(entry, score, latency) for entry, score in scored[:top_k]]

    def expand_context(
        self, conversation_id: str, entry_id: str, window: int
    ) -> list[dict[str, Any]]:
        target_session, _ = entry_position(entry_id)
        entries = [
            entry
            for entry in self._load_entries(conversation_id)
            if entry_position(entry.id)[0] == target_session
        ]
        index = next(
            (position for position, entry in enumerate(entries) if entry.id == entry_id),
            None,
        )
        if index is None:
            raise KeyError(f"Unknown entry_id {entry_id!r} in {conversation_id!r}")
        start = max(0, index - window)
        stop = min(len(entries), index + window + 1)
        return [
            {
                **entry.to_dict(include_embedding=False),
                "score": None,
                "expanded_from": entry_id,
                "relative_offset": position - index,
            }
            for position, entry in enumerate(entries[start:stop], start=start)
        ]
