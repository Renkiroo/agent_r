"""Tests for CoD selector."""

from cod.selector import (
    query_terms,
    select_greedy_unprotected,
    select_with_seed_protection,
)


def _item(i: int, score: float, session: int, text: str) -> dict:
    return {
        "id": f"conv/session_{session}/turn_{i}",
        "score": score,
        "text": text,
    }


def test_query_terms_use_question_entities() -> None:
    terms = [t.lower() for t in query_terms("How many hikes has Joanna been on?")]
    assert "joanna" in terms
    assert "hikes" in terms


def test_same_pool_different_questions_change_selection() -> None:
    pool = [
        _item(i, 0.40 - i * 0.001, 1, f"generic chat filler {i}") for i in range(59)
    ]
    hike = _item(200, 0.305, 7, "Joanna went hiking near Fort Wayne last summer")
    hair = _item(201, 0.304, 8, "Nate dyed his hair last week")
    leftover = _item(58, 0.300, 1, "generic chat filler leftover")
    pool.extend([hike, hair, leftover])
    seeds_hike, _ = select_greedy_unprotected(
        pool, "How many hikes has Joanna been on?", 60, lambda_div=0.06, beta_ent=0.25
    )
    seeds_hair, _ = select_greedy_unprotected(
        pool, "What color did Nate choose for his hair?", 60, lambda_div=0.06, beta_ent=0.25
    )
    hike_ids = {s["id"] for s in seeds_hike}
    hair_ids = {s["id"] for s in seeds_hair}
    assert hike["id"] in hike_ids
    assert hair["id"] in hair_ids
    assert hike_ids != hair_ids


def test_seed_protection_keeps_top_dense_unless_margin() -> None:
    pool = [_item(i, 1.0 - i * 0.001, 1, f"alpha talk {i}") for i in range(60)]
    pool.append(_item(99, 0.10, 2, "completely unrelated zeta"))
    seeds, meta = select_with_seed_protection(
        pool,
        "What did alpha say?",
        60,
        lambda_div=0.06,
        beta_ent=0.10,
        protect_prefix=45,
        max_replacements=15,
        replace_margin=0.03,
    )
    assert len(seeds) == 60
    assert pool[0]["id"] in {s["id"] for s in seeds}
    assert meta["n_replacements"] == 0
    assert pool[60]["id"] not in {s["id"] for s in seeds}


def test_protected_selector_can_promote_complementary_candidate() -> None:
    pool = [
        _item(i, 0.50 - i * 0.0001, 1, "hiking trail sunset water") for i in range(60)
    ]
    pool.append(
        _item(
            80,
            0.48,
            9,
            "Joanna took that pic on a hike last summer near Fort Wayne",
        )
    )
    seeds, meta = select_with_seed_protection(
        pool,
        "How many hikes has Joanna been on?",
        60,
        lambda_div=0.08,
        beta_ent=0.20,
        protect_prefix=45,
        max_replacements=15,
        replace_margin=0.02,
    )
    ids = {s["id"] for s in seeds}
    assert pool[60]["id"] in ids
    assert meta["n_replacements"] >= 1
    assert len(seeds) == 60
