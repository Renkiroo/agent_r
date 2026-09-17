from __future__ import annotations

from cod.run import build_parser
from cod.schema import RawMemoryEntry
from cod.sde import select_seeds, selective_expand
from cod.store import RawMemoryStore


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


def make_entries(session_lengths: tuple[int, ...] = (6,)) -> list[RawMemoryEntry]:
    entries = []
    for session_index, length in enumerate(session_lengths, start=1):
        for turn_index in range(length):
            entries.append(
                RawMemoryEntry(
                    id=f"conv/session_{session_index}/turn_{turn_index}",
                    conversation_id="conv",
                    speaker="A" if turn_index % 2 else "B",
                    text=f"session {session_index} turn {turn_index}",
                    timestamp=f"2023-01-{session_index:02d} 10:{turn_index:02d}:00",
                    embedding=[1.0, 0.0],
                )
            )
    return entries


def hit(entry: RawMemoryEntry, score: float) -> dict:
    return {**entry.to_dict(include_embedding=False), "score": score}


def test_select_seeds_uses_top_m_scores_deterministically() -> None:
    entries = make_entries()
    results = [hit(entries[0], 0.2), hit(entries[1], 0.9), hit(entries[2], 0.5)]
    seeds = select_seeds(results, 2)
    assert [item["id"] for item in seeds] == [
        "conv/session_1/turn_1",
        "conv/session_1/turn_2",
    ]


def test_window_expansion_deduplicates_overlap() -> None:
    entries = make_entries()
    store = RawMemoryStore(FakeEmbedder(), entries=entries)
    seeds = [hit(entries[2], 0.9), hit(entries[3], 0.8)]
    result = selective_expand(
        store,
        "conv",
        seeds,
        dense_candidates=6,
        strategy="window",
        window=1,
    )
    assert [item["id"] for item in result["evidence"]] == [
        "conv/session_1/turn_2",
        "conv/session_1/turn_3",
        "conv/session_1/turn_1",
        "conv/session_1/turn_4",
    ]
    assert len({item["id"] for item in result["expanded"]}) == 4
    assert result["metadata"]["final_unique_entries"] == 4
    assert result["metadata"]["expansion_window"] == 1
    assert all("turn_index" in item for item in result["evidence"])


def test_session_expansion_is_bounded_and_does_not_cross_sessions() -> None:
    entries = make_entries((30, 3))
    store = RawMemoryStore(FakeEmbedder(), entries=entries)
    result = selective_expand(
        store,
        "conv",
        [hit(entries[15], 1.0)],
        dense_candidates=33,
        strategy="session",
        max_context_turns_per_seed=24,
    )
    assert len(result["expanded"]) == 24
    assert all("/session_1/" in item["id"] for item in result["expanded"])
    assert "conv/session_1/turn_15" in {
        item["id"] for item in result["expanded"]
    }
    metadata = result["metadata"]
    assert metadata["sde_strategy"] == "session"
    assert metadata["max_context_turns_per_seed"] == 24
    assert metadata["dense_candidates"] == 33
    assert metadata["num_seeds"] == 1
    assert metadata["retrieved_token_estimate"] > 0


def test_cli_parses_sde_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--method",
            "sde",
            "--sde-m",
            "10",
            "--sde-window",
            "6",
            "--sde-strategy",
            "window",
        ]
    )
    assert args.method == "sde"
    assert args.sde_m == 10
    assert args.sde_window == 6
    assert args.sde_strategy == "window"
