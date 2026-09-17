from __future__ import annotations

import json
from types import SimpleNamespace

from cod.build_raw_memory import parse_timestamp, raw_entries_from_sample
from cod.controller import IterativeMemoryController
from cod.schema import RawMemoryEntry
from cod.store import RawMemoryStore


class FakeEmbedder:
    vectors = {
        "career assists": [1.0, 0.0],
        "another query": [0.0, 1.0],
    }

    def embed(self, text: str) -> list[float]:
        return self.vectors.get(text, [1.0, 0.0])


class FakeCompletions:
    def __init__(self, decisions: list[dict]) -> None:
        self.decisions = iter(decisions)

    def create(self, **kwargs):
        content = json.dumps(next(self.decisions))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=4, total_tokens=14
            ),
        )


class FakeClient:
    def __init__(self, decisions: list[dict]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(decisions))


def entries() -> list[RawMemoryEntry]:
    return [
        RawMemoryEntry(
            id=f"conv/session_1/turn_{index}",
            conversation_id="conv",
            speaker="John" if index % 2 else "Maria",
            text=text,
            timestamp="2023-08-15 14:30:00",
            embedding=embedding,
        )
        for index, (text, embedding) in enumerate(
            [
                ("Before the game.", [0.5, 0.5]),
                ("I had a career-high in assists last Friday.", [1.0, 0.0]),
                ("It was against our rival.", [0.8, 0.2]),
                ("We celebrated afterward.", [0.0, 1.0]),
            ]
        )
    ]


def test_raw_entry_has_no_semantic_categories() -> None:
    entry = entries()[0].to_dict()
    assert set(entry) == {
        "id",
        "conversation_id",
        "speaker",
        "text",
        "timestamp",
        "embedding",
    }
    assert entry["timestamp"] == "2023-08-15 14:30:00"


def test_raw_dataset_entry_preserves_text_and_timestamp() -> None:
    sample = {
        "sample_id": "conv",
        "conversation": {
            "session_1_date_time": "2:32 pm on 15 August, 2023",
            "session_1": [{"speaker": "John", "text": "Exact original utterance."}],
        },
    }
    result = raw_entries_from_sample(sample, [[0.1, 0.2]])
    assert result[0].text == "Exact original utterance."
    assert result[0].timestamp == "2023-08-15 14:32:00"
    assert parse_timestamp("2:32 pm on 15 August, 2023") == result[0].timestamp


def test_dense_search_returns_ids_scores_and_raw_text() -> None:
    store = RawMemoryStore(FakeEmbedder(), entries=entries())
    result = store.search_memory("conv", "career assists", top_k=2)
    assert result[0]["id"] == "conv/session_1/turn_1"
    assert result[0]["score"] == 1.0
    assert result[0]["text"] == "I had a career-high in assists last Friday."
    assert result[0]["retrieval_latency"] >= 0


def test_expand_context_does_not_change_utterances() -> None:
    original = entries()
    store = RawMemoryStore(FakeEmbedder(), entries=original)
    expanded = store.expand_context("conv", "conv/session_1/turn_1", window=1)
    assert [item["text"] for item in expanded] == [
        original[0].text,
        original[1].text,
        original[2].text,
    ]
    assert [item["relative_offset"] for item in expanded] == [-1, 0, 1]


def test_controller_stops_when_evidence_is_sufficient() -> None:
    controller = IterativeMemoryController(
        FakeClient([{"status": "sufficient"}]),
        "fake",
        RawMemoryStore(FakeEmbedder(), entries=entries()),
        top_k=1,
        max_steps=3,
    )
    result = controller.run("career assists", "conv")
    assert result["num_rounds"] == 1
    assert result["stop_reason"] == "sufficient"
    assert result["state"]["actions"][-1]["action"] == "stop"


def test_controller_enforces_max_steps_and_logs_trajectory() -> None:
    decisions = [
        {
            "status": "need_more_evidence",
            "query": "another query",
            "reason": "Need rival context",
            "expand_entry_id": "conv/session_1/turn_1",
        },
        {
            "status": "need_more_evidence",
            "query": "career assists",
            "reason": "Still uncertain",
        },
    ]
    controller = IterativeMemoryController(
        FakeClient(decisions),
        "fake",
        RawMemoryStore(FakeEmbedder(), entries=entries()),
        top_k=1,
        max_steps=2,
        expand_window=1,
    )
    result = controller.run("career assists", "conv")
    assert result["num_rounds"] == 2
    assert result["stop_reason"] == "max_steps"
    assert result["num_tool_calls"] == 3
    assert any(
        action["action"] == "expand_context"
        for action in result["state"]["retrieval_history"]
    )
    assert result["controller_token_usage"]["total_tokens"] == 28
