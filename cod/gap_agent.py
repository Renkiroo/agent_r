"""Gap-Directed evidence acquisition (selection-only, no LLM).

Actions: ANCHOR → GAP_DETECT → TARGET_SEARCH → ... → STOP
All candidates from a fixed dense top-200 pool. Temporary per-question state only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .scoring import token_set, _score, ranked
from .selector import (
    covered_terms,
    marginal_diversity,
    query_entity_coverage,
    query_terms,
)

GAP_TYPES = (
    "missing_entity",
    "missing_fact_event",
    "missing_additional_evidence",
    "missing_temporal_relation",
    "missing_multi_hop_bridge",
)

_TEMPORAL_CUES = {
    "when",
    "before",
    "after",
    "during",
    "since",
    "until",
    "year",
    "month",
    "day",
    "week",
    "date",
    "time",
    "ago",
    "later",
    "earlier",
    "yesterday",
    "today",
    "tomorrow",
    "last",
    "next",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

_MULTI_EVIDENCE_CUES = {
    "and",
    "also",
    "both",
    "each",
    "all",
    "many",
    "which",
    "what else",
    "besides",
}


@dataclass
class EvidenceState:
    question: str
    terms: list[str]
    selected: list[dict[str, Any]] = field(default_factory=list)
    grounded_entities: set[str] = field(default_factory=set)
    grounded_events: list[str] = field(default_factory=list)
    temporal_hits: int = 0
    unresolved_gaps: list[dict[str, Any]] = field(default_factory=list)
    category: int | None = None
    unsatisfiable_terms: set[str] = field(default_factory=set)

    @property
    def selected_ids(self) -> set[str]:
        return {s["id"] for s in self.selected}


def _mentions_temporal(text: str) -> bool:
    toks = set(re.findall(r"[a-z0-9']+", (text or "").lower()))
    if toks & _TEMPORAL_CUES:
        return True
    return bool(re.search(r"\b(20\d{2}|19\d{2})\b", text or ""))


def _question_wants_temporal(question: str) -> bool:
    q = (question or "").lower()
    toks = set(re.findall(r"[a-z0-9']+", q))
    return bool(toks & _TEMPORAL_CUES) or "when" in toks


def _question_wants_multi(question: str, terms: list[str]) -> bool:
    q = (question or "").lower()
    if len(terms) >= 3:
        return True
    if any(c in q for c in (" and ", " both ", " also ", " as well ")):
        return True
    if re.search(r"\b(who|what|which).+\b(and|also)\b", q):
        return True
    return False


def update_state_from_pick(state: EvidenceState, item: dict[str, Any]) -> None:
    text = item.get("text") or ""
    for term in state.terms:
        if term.lower() in text.lower():
            state.grounded_entities.add(term.lower())
    # light event fingerprint: speaker + content keywords
    speaker = (item.get("speaker") or "").strip()
    content = [
        t
        for t in token_set(text)
        if len(t) >= 5 and t not in _TEMPORAL_CUES
    ][:4]
    if speaker or content:
        state.grounded_events.append(
            f"{speaker}:{'/'.join(content)}" if content else speaker
        )
    if _mentions_temporal(text):
        state.temporal_hits += 1


def detect_gaps(
    state: EvidenceState,
    *,
    pool_text_blob: str | None = None,
) -> list[dict[str, Any]]:
    """Emit unresolved question-specific gaps from temporary evidence state.

    Entity gaps are restricted to query terms that appear somewhere in the
    candidate pool (otherwise they are unclosable and must not loop forever).
    """
    gaps: list[dict[str, Any]] = []
    covered = covered_terms(state.selected, state.terms)
    uncovered = [t for t in state.terms if t.lower() not in covered]
    if pool_text_blob is not None:
        blob = pool_text_blob.lower()
        uncovered = [t for t in uncovered if t.lower() in blob]
    # Drop terms already marked unsatisfiable in this question
    uncovered = [t for t in uncovered if t.lower() not in state.unsatisfiable_terms]

    if uncovered:
        gaps.append(
            {
                "gap_type": "missing_entity",
                "targets": uncovered[:6],
                "detail": f"uncovered_query_terms={uncovered[:6]}",
            }
        )

    if (
        _question_wants_temporal(state.question)
        and state.temporal_hits == 0
        and "temporal" not in state.unsatisfiable_terms
    ):
        # only if pool has any temporal-looking candidate
        if pool_text_blob is None or any(
            cue in (pool_text_blob or "").lower() for cue in _TEMPORAL_CUES
        ) or re.search(r"\b(20\d{2}|19\d{2})\b", pool_text_blob or ""):
            gaps.append(
                {
                    "gap_type": "missing_temporal_relation",
                    "targets": ["temporal_cue"],
                    "detail": "question_has_temporal_cue_but_no_temporal_evidence",
                }
            )

    # Multi-hop bridge: ≥2 entities grounded in disjoint evidence, no bridging pick yet
    if len(state.terms) >= 2 and len(state.selected) >= 1:
        per_term_docs: dict[str, set[str]] = {
            t.lower(): set() for t in state.terms
        }
        for s in state.selected:
            text = (s.get("text") or "").lower()
            for t in state.terms:
                if t.lower() in text:
                    per_term_docs[t.lower()].add(s["id"])
        grounded = [t for t, docs in per_term_docs.items() if docs]
        if len(grounded) >= 2:
            has_bridge = False
            for s in state.selected:
                text = (s.get("text") or "").lower()
                hit = sum(1 for t in grounded if t in text)
                if hit >= 2:
                    has_bridge = True
                    break
            if not has_bridge and "bridge" not in state.unsatisfiable_terms:
                gaps.append(
                    {
                        "gap_type": "missing_multi_hop_bridge",
                        "targets": grounded[:4],
                        "detail": "entities_grounded_separately_no_bridging_evidence",
                    }
                )

    # Additional evidence for multi-aspect questions — soft cap
    need_extra = max(3, min(8, 2 + len(state.terms)))
    if _question_wants_multi(state.question, state.terms):
        if len(state.selected) < need_extra and "additional" not in state.unsatisfiable_terms:
            gaps.append(
                {
                    "gap_type": "missing_additional_evidence",
                    "targets": uncovered[:4] or state.terms[:4],
                    "detail": "multi_aspect_question_under_covered",
                }
            )

    # Fact/event gap: have entities but very few distinct event fingerprints
    if (
        state.grounded_entities
        and len(state.grounded_events) < 2
        and len(state.selected) >= 1
        and len(state.selected) < 6
        and "fact" not in state.unsatisfiable_terms
    ):
        gaps.append(
            {
                "gap_type": "missing_fact_event",
                "targets": list(state.grounded_entities)[:4],
                "detail": "entities_present_but_sparse_event_support",
            }
        )

    # Dedup by gap_type keeping first
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for g in gaps:
        if g["gap_type"] in seen:
            continue
        seen.add(g["gap_type"])
        uniq.append(g)
    return uniq


def score_for_gap(
    item: dict[str, Any],
    state: EvidenceState,
    gap: dict[str, Any],
    *,
    lambda_div: float = 0.06,
    beta_ent: float = 0.12,
    gamma_gap: float = 0.18,
    delta_temp: float = 0.08,
) -> dict[str, float]:
    dense = _score(item)
    if dense == float("-inf"):
        dense = 0.0
    complementarity = marginal_diversity(item, state.selected)
    entity_consistency = query_entity_coverage(item, state.terms, state.selected)
    text = item.get("text") or ""
    targets = [str(t).lower() for t in (gap.get("targets") or [])]
    gap_hits = 0
    for t in targets:
        if t == "temporal_cue":
            continue
        if t and t in text.lower():
            gap_hits += 1
    gap_relevance = gap_hits / max(1, len([t for t in targets if t != "temporal_cue"]))
    if gap["gap_type"] == "missing_temporal_relation":
        temporal_usefulness = 1.0 if _mentions_temporal(text) else 0.0
    else:
        temporal_usefulness = 0.35 if _mentions_temporal(text) else 0.0

    # Bridge bonus: covers ≥2 already-grounded or target entities
    bridge = 0.0
    if gap["gap_type"] == "missing_multi_hop_bridge":
        hit = sum(1 for t in targets if t in text.lower())
        bridge = min(1.0, hit / 2.0)

    total = (
        dense
        + beta_ent * entity_consistency
        + lambda_div * complementarity
        + gamma_gap * (gap_relevance + 0.5 * bridge)
        + delta_temp * temporal_usefulness
    )
    return {
        "dense": dense,
        "entity_consistency": entity_consistency,
        "complementarity": complementarity,
        "gap_relevance": gap_relevance,
        "temporal_usefulness": temporal_usefulness,
        "bridge": bridge,
        "total": total,
    }


def pick_anchor(
    pool: list[dict[str, Any]], state: EvidenceState, n_anchors: int = 2
) -> list[dict[str, Any]]:
    """Initial anchors: prefer high-dense items covering query terms."""
    remaining = [
        x for x in ranked(pool) if x["id"] not in state.selected_ids
    ]
    if not remaining:
        return []
    scored = []
    for item in remaining:
        dense = _score(item)
        cov = query_entity_coverage(item, state.terms, state.selected)
        scored.append((item, dense + 0.25 * cov))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored[:n_anchors]]


def target_search(
    pool: list[dict[str, Any]],
    state: EvidenceState,
    gap: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, float] | None, int | None]:
    remaining = [x for x in ranked(pool) if x["id"] not in state.selected_ids]
    if not remaining:
        return None, None, None
    pool_index = {item["id"]: i + 1 for i, item in enumerate(ranked(pool))}
    best = None
    best_sc = None
    for item in remaining:
        sc = score_for_gap(item, state, gap)
        if best is None or sc["total"] > best_sc["total"]:  # type: ignore[index]
            best, best_sc = item, sc
    assert best is not None and best_sc is not None
    return best, best_sc, pool_index.get(best["id"])


def run_gap_directed_agent(
    pool: list[dict[str, Any]],
    question: str,
    *,
    max_seeds: int = 60,
    max_steps: int = 80,
    n_anchors: int = 2,
    category: int | None = None,
    gold_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Return (seeds, meta, trajectory). Trajectory records every action."""
    terms = query_terms(question)
    state = EvidenceState(question=question, terms=terms, category=category)
    trajectory: list[dict[str, Any]] = []
    gold_ids = gold_ids or set()
    step = 0
    stopped_by = "max_steps"
    gap_type_counts: dict[str, int] = {g: 0 for g in GAP_TYPES}
    gap_gold_hits: dict[str, int] = {g: 0 for g in GAP_TYPES}
    gap_attempts: dict[str, int] = {g: 0 for g in GAP_TYPES}
    pool_ranked = ranked(pool)
    pool_index = {item["id"]: i + 1 for i, item in enumerate(pool_ranked)}
    pool_text_blob = "\n".join(str(x.get("text") or "") for x in pool_ranked)

    def _log(action: str, **kwargs: Any) -> None:
        nonlocal step
        step += 1
        row = {"step": step, "action": action, **kwargs}
        trajectory.append(row)

    def _gaps() -> list[dict[str, Any]]:
        return detect_gaps(state, pool_text_blob=pool_text_blob)

    # ANCHOR phase
    anchors = pick_anchor(pool, state, n_anchors=n_anchors)
    for item in anchors:
        if len(state.selected) >= max_seeds:
            break
        update_state_from_pick(state, item)
        state.selected.append(item)
        _log(
            "ANCHOR",
            selected_evidence_id=item["id"],
            detected_gap=None,
            gap_type=None,
            gap_before=0,
            gap_after=len(_gaps()),
            candidate_rank=pool_index.get(item["id"]),
            dense_score=_score(item),
            is_gold=item["id"] in gold_ids,
        )

    stagnant = 0
    while len(state.selected) < max_seeds and step < max_steps:
        gaps_before = _gaps()
        state.unresolved_gaps = gaps_before
        _log(
            "GAP_DETECT",
            selected_evidence_id=None,
            detected_gap=[g["gap_type"] for g in gaps_before],
            gap_type=gaps_before[0]["gap_type"] if gaps_before else None,
            gap_before=len(gaps_before),
            gap_after=len(gaps_before),
            candidate_rank=None,
            dense_score=None,
            is_gold=None,
            gaps=gaps_before,
        )
        for g in gaps_before:
            gap_type_counts[g["gap_type"]] = gap_type_counts.get(g["gap_type"], 0) + 1

        if not gaps_before:
            stopped_by = "no_gaps"
            _log(
                "STOP",
                selected_evidence_id=None,
                detected_gap=[],
                gap_type=None,
                gap_before=0,
                gap_after=0,
                candidate_rank=None,
                dense_score=None,
                is_gold=None,
                reason="no_gaps",
            )
            break

        gap = gaps_before[0]
        pick, sc, rank = target_search(pool, state, gap)
        if pick is None:
            stopped_by = "no_candidates"
            _log(
                "STOP",
                selected_evidence_id=None,
                detected_gap=[g["gap_type"] for g in gaps_before],
                gap_type=gap["gap_type"],
                gap_before=len(gaps_before),
                gap_after=len(gaps_before),
                candidate_rank=None,
                dense_score=None,
                is_gold=None,
                reason="no_candidates",
            )
            break

        # Only entity / temporal / bridge can be unsatisfiable in-pool.
        # additional / fact gaps always accept the best remaining pick.
        unsat = False
        if gap["gap_type"] == "missing_entity" and (sc or {}).get("gap_relevance", 0) <= 1e-9:
            for t in gap.get("targets") or []:
                state.unsatisfiable_terms.add(str(t).lower())
            unsat = True
        elif gap["gap_type"] == "missing_temporal_relation" and (sc or {}).get(
            "temporal_usefulness", 0
        ) <= 1e-9:
            state.unsatisfiable_terms.add("temporal")
            unsat = True
        elif gap["gap_type"] == "missing_multi_hop_bridge" and (sc or {}).get(
            "bridge", 0
        ) <= 1e-9:
            state.unsatisfiable_terms.add("bridge")
            unsat = True

        if unsat:
            stagnant += 1
            if stagnant >= 3 and len(state.selected) >= max(4, n_anchors):
                stopped_by = "unsatisfiable_gaps"
                _log(
                    "STOP",
                    selected_evidence_id=None,
                    detected_gap=[g["gap_type"] for g in gaps_before],
                    gap_type=gap["gap_type"],
                    gap_before=len(gaps_before),
                    gap_after=len(_gaps()),
                    candidate_rank=None,
                    dense_score=None,
                    is_gold=None,
                    reason="unsatisfiable_gaps",
                )
                break
            continue

        stagnant = 0
        gap_attempts[gap["gap_type"]] = gap_attempts.get(gap["gap_type"], 0) + 1
        is_gold = pick["id"] in gold_ids
        if is_gold:
            gap_gold_hits[gap["gap_type"]] = gap_gold_hits.get(gap["gap_type"], 0) + 1

        update_state_from_pick(state, pick)
        state.selected.append(pick)
        gaps_after = _gaps()
        _log(
            "TARGET_SEARCH",
            selected_evidence_id=pick["id"],
            detected_gap=gap["gap_type"],
            gap_type=gap["gap_type"],
            gap_before=len(gaps_before),
            gap_after=len(gaps_after),
            candidate_rank=rank,
            dense_score=sc["dense"] if sc else None,
            score_components=sc,
            is_gold=is_gold,
            gap_detail=gap.get("detail"),
        )
    else:
        final_gaps = _gaps()
        if len(state.selected) >= max_seeds:
            stopped_by = "max_seeds"
            reason = "max_seeds"
        else:
            stopped_by = "max_steps"
            reason = "max_steps"
        _log(
            "STOP",
            selected_evidence_id=None,
            detected_gap=[g["gap_type"] for g in final_gaps],
            gap_type=None,
            gap_before=len(final_gaps),
            gap_after=len(final_gaps),
            candidate_rank=None,
            dense_score=None,
            is_gold=None,
            reason=reason,
        )

    # Fill remaining budget to K=max_seeds so Gold Recall @60 is comparable;
    # gap-directed order still dominates low budgets.
    n_gap_seeds = len(state.selected)
    fill_gap = {
        "gap_type": "missing_additional_evidence",
        "targets": state.terms[:4],
        "detail": "budget_fill",
    }
    while len(state.selected) < max_seeds:
        pick, sc, rank = target_search(pool, state, fill_gap)
        if pick is None:
            break
        update_state_from_pick(state, pick)
        state.selected.append(pick)
        _log(
            "TARGET_SEARCH",
            selected_evidence_id=pick["id"],
            detected_gap="budget_fill",
            gap_type="budget_fill",
            gap_before=0,
            gap_after=0,
            candidate_rank=rank,
            dense_score=sc["dense"] if sc else None,
            score_components=sc,
            is_gold=pick["id"] in gold_ids,
            gap_detail="budget_fill",
        )
    if len(state.selected) >= max_seeds and stopped_by == "max_steps":
        stopped_by = "max_seeds"

    # Gap closure: non-fill TARGET_SEARCH steps that reduced gap count
    target_steps = [
        t
        for t in trajectory
        if t["action"] == "TARGET_SEARCH" and t.get("gap_type") != "budget_fill"
    ]
    closures = sum(
        1 for t in target_steps if (t.get("gap_after") or 0) < (t.get("gap_before") or 0)
    )
    closure_rate = closures / len(target_steps) if target_steps else 0.0

    recovery = {}
    for gt in GAP_TYPES:
        att = gap_attempts.get(gt, 0)
        recovery[gt] = (gap_gold_hits.get(gt, 0) / att) if att else None

    meta = {
        "mode": "gap_directed",
        "n_seeds": len(state.selected),
        "n_gap_directed_seeds": n_gap_seeds,
        "n_budget_fill": max(0, len(state.selected) - n_gap_seeds),
        "n_steps": len(trajectory),
        "stopped_by": stopped_by,
        "early_stop": stopped_by in ("no_gaps", "unsatisfiable_gaps"),
        "query_terms": terms,
        "grounded_entities": sorted(state.grounded_entities),
        "temporal_hits": state.temporal_hits,
        "gap_type_counts": gap_type_counts,
        "gap_attempts": gap_attempts,
        "gap_gold_hits": gap_gold_hits,
        "gap_gold_recovery_rate": recovery,
        "gap_closure_rate": closure_rate,
        "n_target_search": len(target_steps),
        "seed_ids": [s["id"] for s in state.selected],
        "action_counts": {
            a: sum(1 for t in trajectory if t["action"] == a)
            for a in ("ANCHOR", "GAP_DETECT", "TARGET_SEARCH", "STOP")
        },
    }
    return state.selected, meta, trajectory
