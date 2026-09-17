from __future__ import annotations

from typing import Any

from .controller import merge_evidence
from .store import entry_position


def estimate_tokens(evidence: list[dict[str, Any]]) -> int:
    return sum(max(1, len(item.get("text", "").split())) for item in evidence)


def _score(item: dict[str, Any]) -> float:
    return float(item.get("score", float("-inf")))


def _session_key(item: dict[str, Any]) -> int:
    if item.get("session_index") is not None:
        return int(item["session_index"])
    return entry_position(item["id"])[0]


def seed_session_stats(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    sessions = [_session_key(seed) for seed in seeds]
    unique = sorted(set(sessions))
    return {
        "seed_sessions": unique,
        "num_seed_sessions": len(unique),
        "seed_diversity": (len(unique) / len(seeds)) if seeds else 0.0,
    }


def select_seeds_score(results: list[dict[str, Any]], m: int) -> list[dict[str, Any]]:
    ranked = sorted(results, key=_score, reverse=True)
    return ranked[:m]


def select_seeds_session_diverse(
    results: list[dict[str, Any]], m: int
) -> list[dict[str, Any]]:
    """Pick high-score seeds while covering distinct sessions first.

    Algorithm:
    1. Rank dense hits by score.
    2. Keep the best hit per session.
    3. Take those session representatives in score order until budget is filled.
    4. Fill any remaining slots from leftover hits by score.
    """
    if m <= 0:
        raise ValueError("m must be greater than zero")
    ranked = sorted(results, key=_score, reverse=True)
    best_by_session: dict[int, dict[str, Any]] = {}
    for item in ranked:
        session = _session_key(item)
        if session not in best_by_session:
            best_by_session[session] = item

    session_order = sorted(
        best_by_session.keys(),
        key=lambda session: _score(best_by_session[session]),
        reverse=True,
    )
    seeds: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for session in session_order:
        if len(seeds) >= m:
            break
        item = best_by_session[session]
        seeds.append(item)
        selected_ids.add(item["id"])

    if len(seeds) < m:
        for item in ranked:
            if item["id"] in selected_ids:
                continue
            seeds.append(item)
            selected_ids.add(item["id"])
            if len(seeds) >= m:
                break
    return seeds[:m]


def select_seeds_oracle(
    results: list[dict[str, Any]],
    m: int,
    gold_entry_ids: list[str],
    *,
    gold_entries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Diagnosis-only: force gold entries into the seed set, then fill by score."""
    if m <= 0:
        raise ValueError("m must be greater than zero")
    by_id = {item["id"]: item for item in results}
    if gold_entries:
        for entry in gold_entries:
            by_id.setdefault(entry["id"], entry)

    forced: list[dict[str, Any]] = []
    forced_ids: set[str] = set()
    for gold_id in gold_entry_ids:
        if gold_id in forced_ids:
            continue
        item = by_id.get(gold_id)
        if item is None:
            continue
        forced.append(item)
        forced_ids.add(gold_id)
        if len(forced) >= m:
            return forced[:m]

    ranked = sorted(
        (item for item in results if item["id"] not in forced_ids),
        key=_score,
        reverse=True,
    )
    return forced + ranked[: max(0, m - len(forced))]


def select_seeds_rescue(
    results: list[dict[str, Any]],
    *,
    core_k: int,
    rescue_k: int,
    rescue_rank_lo: int,
    rescue_rank_hi: int,
) -> list[dict[str, Any]]:
    """Keep top core_k seeds, then rescue mid-rank hits into the remaining budget.

    Ranks are 1-indexed over dense results sorted by score descending.
    Rescue candidates must lie in [rescue_rank_lo, rescue_rank_hi] and outside
    the core set. Within the rescue window, candidates are chosen by dense score.
    No gold / oracle / speaker signals are used.
    """
    if core_k < 0 or rescue_k < 0:
        raise ValueError("core_k and rescue_k must be non-negative")
    if core_k + rescue_k <= 0:
        raise ValueError("core_k + rescue_k must be greater than zero")
    if rescue_rank_lo < 1 or rescue_rank_hi < rescue_rank_lo:
        raise ValueError("invalid rescue rank window")

    ranked = sorted(results, key=_score, reverse=True)
    core = ranked[:core_k]
    core_ids = {item["id"] for item in core}

    rescue_pool = []
    for rank, item in enumerate(ranked, start=1):
        if rank < rescue_rank_lo or rank > rescue_rank_hi:
            continue
        if item["id"] in core_ids:
            continue
        rescue_pool.append(item)

    rescue = sorted(rescue_pool, key=_score, reverse=True)[:rescue_k]
    return core + rescue


def select_seeds(
    results: list[dict[str, Any]],
    m: int,
    strategy: str = "score",
    *,
    gold_entry_ids: list[str] | None = None,
    gold_entries: list[dict[str, Any]] | None = None,
    rescue_core_k: int | None = None,
    rescue_k: int | None = None,
    rescue_rank_lo: int | None = None,
    rescue_rank_hi: int | None = None,
) -> list[dict[str, Any]]:
    """Select deterministic expansion seeds from dense retrieval results."""
    if m <= 0:
        raise ValueError("m must be greater than zero")
    if strategy == "score":
        return select_seeds_score(results, m)
    if strategy == "session_diverse":
        return select_seeds_session_diverse(results, m)
    if strategy == "oracle":
        return select_seeds_oracle(
            results,
            m,
            gold_entry_ids or [],
            gold_entries=gold_entries,
        )
    if strategy == "rescue":
        if (
            rescue_core_k is None
            or rescue_k is None
            or rescue_rank_lo is None
            or rescue_rank_hi is None
        ):
            raise ValueError(
                "rescue strategy requires rescue_core_k, rescue_k, "
                "rescue_rank_lo, and rescue_rank_hi"
            )
        seeds = select_seeds_rescue(
            results,
            core_k=rescue_core_k,
            rescue_k=rescue_k,
            rescue_rank_lo=rescue_rank_lo,
            rescue_rank_hi=rescue_rank_hi,
        )
        if len(seeds) > m:
            return seeds[:m]
        return seeds
    raise ValueError(f"Unsupported seed selection strategy: {strategy}")


def _annotate_position(item: dict[str, Any]) -> dict[str, Any]:
    session_index, turn_index = entry_position(item["id"])
    return {
        **item,
        "session_index": session_index,
        "turn_index": turn_index,
    }


def _bounded_session_context(
    store: Any,
    conversation_id: str,
    seed_id: str,
    max_context_turns: int,
) -> list[dict[str, Any]]:
    if max_context_turns <= 0:
        raise ValueError("max_context_turns_per_seed must be greater than zero")

    target_session, _ = entry_position(seed_id)
    session_entries = [
        entry
        for entry in store._load_entries(conversation_id)
        if entry_position(entry.id)[0] == target_session
    ]
    seed_index = next(
        (index for index, entry in enumerate(session_entries) if entry.id == seed_id),
        None,
    )
    if seed_index is None:
        raise KeyError(f"Unknown seed {seed_id!r} in {conversation_id!r}")

    if len(session_entries) <= max_context_turns:
        start, stop = 0, len(session_entries)
    else:
        half = max_context_turns // 2
        start = max(0, seed_index - half)
        stop = start + max_context_turns
        if stop > len(session_entries):
            stop = len(session_entries)
            start = stop - max_context_turns

    return [
        {
            **entry.to_dict(include_embedding=False),
            "score": None,
            "expanded_from": seed_id,
            "relative_offset": index - seed_index,
        }
        for index, entry in enumerate(session_entries[start:stop], start=start)
    ]


def selective_expand(
    store: Any,
    conversation_id: str,
    seeds: list[dict[str, Any]],
    *,
    dense_candidates: int,
    strategy: str = "window",
    window: int = 4,
    max_context_turns_per_seed: int = 24,
    seed_selection_strategy: str = "score",
) -> dict[str, Any]:
    """Expand selected seeds while preserving dense-hit evidence order."""
    if strategy not in {"window", "session"}:
        raise ValueError(f"Unsupported SDE expansion strategy: {strategy}")
    if strategy == "window" and window < 0:
        raise ValueError("window must be non-negative")

    annotated_seeds = [_annotate_position(seed) for seed in seeds]
    expanded: list[dict[str, Any]] = []
    per_seed_counts: dict[str, int] = {}
    for seed in annotated_seeds:
        if strategy == "window":
            context = store.expand_context(conversation_id, seed["id"], window)
        else:
            context = _bounded_session_context(
                store,
                conversation_id,
                seed["id"],
                max_context_turns_per_seed,
            )
        annotated_context = [_annotate_position(item) for item in context]
        per_seed_counts[seed["id"]] = len(annotated_context)
        expanded = merge_evidence(expanded, annotated_context)

    evidence = merge_evidence(annotated_seeds, expanded)
    diversity = seed_session_stats(annotated_seeds)
    metadata = {
        "dense_candidates": dense_candidates,
        "num_seeds": len(annotated_seeds),
        "seed_selection_strategy": seed_selection_strategy,
        "sde_strategy": strategy,
        "expansion_window": window if strategy == "window" else None,
        "max_context_turns_per_seed": (
            max_context_turns_per_seed if strategy == "session" else None
        ),
        "expanded_unique_entries": len(expanded),
        "final_unique_entries": len(evidence),
        "retrieved_token_estimate": estimate_tokens(evidence),
        "per_seed_expansion_counts": per_seed_counts,
        **diversity,
    }
    return {
        "seeds": annotated_seeds,
        "expanded": expanded,
        "evidence": evidence,
        "metadata": metadata,
    }
