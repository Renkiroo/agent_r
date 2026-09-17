from __future__ import annotations

from cod.sde import select_seeds, select_seeds_rescue


def _hit(rank_score: float, session: int, turn: int) -> dict:
    return {
        "id": f"conv/session_{session}/turn_{turn}",
        "conversation_id": "conv",
        "speaker": "A",
        "text": f"s{session}t{turn}",
        "timestamp": "2023-01-01 00:00:00",
        "score": rank_score,
    }


def _ranked_hits(n: int = 60) -> list[dict]:
    # score decreases with rank index
    return [_hit(1.0 - i * 0.01, session=(i % 12) + 1, turn=i) for i in range(n)]


def test_rescue_keeps_core_and_pulls_from_mid_rank_window() -> None:
    results = _ranked_hits(60)
    seeds = select_seeds_rescue(
        results, core_k=8, rescue_k=2, rescue_rank_lo=11, rescue_rank_hi=30
    )
    assert len(seeds) == 10
    assert [item["id"] for item in seeds[:8]] == [results[i]["id"] for i in range(8)]
    # ranks 11-12 are indices 10-11
    assert [item["id"] for item in seeds[8:]] == [
        results[10]["id"],
        results[11]["id"],
    ]
    # ranks 9-10 (indices 8-9) are intentionally skipped
    skipped = {results[8]["id"], results[9]["id"]}
    assert skipped.isdisjoint({item["id"] for item in seeds})


def test_rescue_m5_window_6_30() -> None:
    results = _ranked_hits(60)
    seeds = select_seeds(
        results,
        m=5,
        strategy="rescue",
        rescue_core_k=4,
        rescue_k=1,
        rescue_rank_lo=6,
        rescue_rank_hi=30,
    )
    assert len(seeds) == 5
    assert seeds[4]["id"] == results[5]["id"]  # rank 6
    assert results[4]["id"] not in {item["id"] for item in seeds}  # rank 5 skipped


def test_rescue_does_not_use_oracle_or_question_ids() -> None:
    results = _ranked_hits(20)
    seeds = select_seeds_rescue(
        results, core_k=3, rescue_k=2, rescue_rank_lo=8, rescue_rank_hi=15
    )
    assert [item["id"] for item in seeds] == [
        results[0]["id"],
        results[1]["id"],
        results[2]["id"],
        results[7]["id"],
        results[8]["id"],
    ]


def test_cli_parses_rescue_args() -> None:
    from cod.run import build_parser

    args = build_parser().parse_args(
        [
            "--method",
            "sde",
            "--seed-strategy",
            "rescue",
            "--sde-m",
            "10",
            "--rescue-core-k",
            "8",
            "--rescue-k",
            "2",
            "--rescue-rank-lo",
            "11",
            "--rescue-rank-hi",
            "30",
        ]
    )
    assert args.seed_strategy == "rescue"
    assert args.rescue_core_k == 8
    assert args.rescue_rank_hi == 30
