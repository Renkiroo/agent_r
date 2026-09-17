from __future__ import annotations

from cod.sde import (
    select_seeds,
    select_seeds_oracle,
    select_seeds_session_diverse,
    seed_session_stats,
)


def _hit(session: int, turn: int, score: float, text: str = "x") -> dict:
    return {
        "id": f"conv/session_{session}/turn_{turn}",
        "conversation_id": "conv",
        "speaker": "A",
        "text": text,
        "timestamp": "2023-01-01 00:00:00",
        "score": score,
        "session_index": session,
        "turn_index": turn,
    }


def test_session_diverse_covers_distinct_sessions_before_refill() -> None:
    # Top scores are dominated by session 1; diverse should still take other sessions.
    results = [
        _hit(1, 0, 0.99),
        _hit(1, 1, 0.98),
        _hit(1, 2, 0.97),
        _hit(2, 0, 0.50),
        _hit(3, 0, 0.40),
        _hit(4, 0, 0.30),
        _hit(5, 0, 0.20),
    ]
    seeds = select_seeds_session_diverse(results, m=5)
    sessions = [item["session_index"] for item in seeds]
    assert len(seeds) == 5
    assert set(sessions) == {1, 2, 3, 4, 5}
    assert sessions[0] == 1  # highest session-best first


def test_session_diverse_fills_remaining_by_score() -> None:
    results = [
        _hit(1, 0, 0.90),
        _hit(2, 0, 0.80),
        _hit(1, 1, 0.70),
        _hit(2, 1, 0.60),
        _hit(1, 2, 0.50),
    ]
    seeds = select_seeds(results, m=4, strategy="session_diverse")
    assert [item["id"] for item in seeds] == [
        "conv/session_1/turn_0",
        "conv/session_2/turn_0",
        "conv/session_1/turn_1",
        "conv/session_2/turn_1",
    ]
    stats = seed_session_stats(seeds)
    assert stats["num_seed_sessions"] == 2
    assert stats["seed_diversity"] == 0.5


def test_oracle_forces_gold_then_fills_by_score() -> None:
    results = [
        _hit(1, 0, 0.99),
        _hit(1, 1, 0.90),
        _hit(2, 0, 0.80),
        _hit(3, 0, 0.10, text="gold"),
    ]
    gold_id = "conv/session_3/turn_0"
    seeds = select_seeds_oracle(results, m=3, gold_entry_ids=[gold_id])
    assert seeds[0]["id"] == gold_id
    assert [item["id"] for item in seeds[1:]] == [
        "conv/session_1/turn_0",
        "conv/session_1/turn_1",
    ]


def test_oracle_can_inject_gold_missing_from_dense_top_k() -> None:
    results = [_hit(1, 0, 0.99), _hit(2, 0, 0.80)]
    gold = _hit(9, 3, float("-inf"), text="missing gold")
    seeds = select_seeds(
        results,
        m=2,
        strategy="oracle",
        gold_entry_ids=[gold["id"]],
        gold_entries=[gold],
    )
    assert seeds[0]["id"] == gold["id"]
    assert seeds[1]["id"] == "conv/session_1/turn_0"


def test_score_strategy_unchanged() -> None:
    results = [_hit(2, 0, 0.5), _hit(1, 0, 0.9), _hit(3, 0, 0.7)]
    seeds = select_seeds(results, m=2, strategy="score")
    assert [item["id"] for item in seeds] == [
        "conv/session_1/turn_0",
        "conv/session_3/turn_0",
    ]


def test_cli_parses_seed_strategy() -> None:
    from cod.run import build_parser

    args = build_parser().parse_args(
        [
            "--method",
            "sde",
            "--sde-m",
            "10",
            "--sde-window",
            "4",
            "--seed-strategy",
            "session_diverse",
        ]
    )
    assert args.seed_strategy == "session_diverse"
