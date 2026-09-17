"""CoD (Chain on Demand): query-conditioned greedy seed allocation.

score(e | S, q) = dense_score(e, q)
                + lambda * marginal_diversity(e, S)
                + beta * query_entity_coverage(e, q, S)

Diversity is an additive complementarity term (1 - max text Jaccard to S).
Entity coverage uses only strings already in the question.
"""

from __future__ import annotations

import re
from typing import Any

from .scoring import extract_entities, ranked, text_jaccard, _score

_CONTENT_STOP = {
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "did",
    "does",
    "do",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "for",
    "with",
    "from",
    "about",
    "to",
    "of",
    "and",
    "or",
    "but",
    "that",
    "this",
    "his",
    "her",
    "their",
    "they",
    "she",
    "he",
    "it",
    "been",
    "being",
    "many",
    "much",
    "some",
    "any",
    "into",
    "over",
    "after",
    "before",
    "during",
}


def query_terms(question: str) -> list[str]:
    """Explicit entities + content keywords already in the question."""
    terms: list[str] = []
    seen: set[str] = set()
    for name in extract_entities(question):
        key = name.lower()
        if key not in seen:
            terms.append(name)
            seen.add(key)
    for tok in re.findall(r"[a-zA-Z][a-zA-Z']{3,}", question):
        low = tok.lower()
        if low in _CONTENT_STOP or low in seen:
            continue
        terms.append(tok)
        seen.add(low)
    return terms


def _mentions(text: str, term: str) -> bool:
    return term.lower() in (text or "").lower()


def covered_terms(items: list[dict[str, Any]], terms: list[str]) -> set[str]:
    covered: set[str] = set()
    for item in items:
        text = item.get("text") or ""
        for term in terms:
            if _mentions(text, term):
                covered.add(term.lower())
    return covered


def marginal_diversity(item: dict[str, Any], selected: list[dict[str, Any]]) -> float:
    """Complementarity vs selected set: 1 - max text Jaccard. Empty S → 1.0."""
    if not selected:
        return 1.0
    sims = [text_jaccard(item.get("text") or "", s.get("text") or "") for s in selected]
    return 1.0 - max(sims)


def query_entity_coverage(
    item: dict[str, Any],
    terms: list[str],
    selected: list[dict[str, Any]],
) -> float:
    """Fraction of still-uncovered query terms that this evidence mentions."""
    if not terms:
        return 0.0
    already = covered_terms(selected, terms)
    uncovered = [t for t in terms if t.lower() not in already]
    if not uncovered:
        return 0.0
    text = item.get("text") or ""
    hits = sum(1 for t in uncovered if _mentions(text, t))
    return hits / len(uncovered)


def conditioned_score(
    item: dict[str, Any],
    selected: list[dict[str, Any]],
    terms: list[str],
    *,
    lambda_div: float,
    beta_ent: float,
) -> dict[str, float]:
    dense = _score(item)
    if dense == float("-inf"):
        dense = 0.0
    div = marginal_diversity(item, selected)
    cov = query_entity_coverage(item, terms, selected)
    total = dense + lambda_div * div + beta_ent * cov
    return {
        "dense": dense,
        "diversity": div,
        "entity_coverage": cov,
        "total": total,
    }


def _replacement_log(
    cand: dict[str, Any],
    victim: dict[str, Any],
    pool_index: dict[str, int],
    cand_sc: dict[str, float],
    vic_sc: dict[str, float],
) -> dict[str, Any]:
    return {
        "in_id": cand["id"],
        "out_id": victim["id"],
        "in_dense_rank": pool_index.get(cand["id"]),
        "out_dense_rank": pool_index.get(victim["id"]),
        "in_dense": cand_sc["dense"],
        "out_dense": vic_sc["dense"],
        "in_total": cand_sc["total"],
        "out_total": vic_sc["total"],
        "in_diversity": cand_sc["diversity"],
        "in_entity_coverage": cand_sc["entity_coverage"],
    }


def select_with_seed_protection(
    pool: list[dict[str, Any]],
    question: str,
    k: int,
    *,
    lambda_div: float,
    beta_ent: float,
    protect_prefix: int,
    max_replacements: int,
    replace_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Start from dense top-k; replace a seed only if a later candidate scores higher."""
    pool_ranked = ranked(pool)
    if not pool_ranked:
        return [], {"mode": "seed_protected", "k": k, "pool_size": 0}

    terms = query_terms(question)
    baseline = pool_ranked[:k]
    selected = list(baseline)
    pool_index = {item["id"]: i + 1 for i, item in enumerate(pool_ranked)}
    protected_ids = {item["id"] for item in baseline[: min(protect_prefix, len(baseline))]}
    replacements: list[dict[str, Any]] = []
    remaining = [item for item in pool_ranked[k:]]

    while len(replacements) < max_replacements and remaining:
        replaceable = [s for s in selected if s["id"] not in protected_ids]
        if not replaceable:
            break

        best_move = None
        best_gain = float("-inf")
        replaceable_scored = []
        for victim in replaceable:
            others = [s for s in selected if s["id"] != victim["id"]]
            vic_sc = conditioned_score(
                victim, others, terms, lambda_div=lambda_div, beta_ent=beta_ent
            )
            replaceable_scored.append((victim, others, vic_sc))
        replaceable_scored.sort(key=lambda x: x[2]["total"])
        victim, others, vic_sc = replaceable_scored[0]

        for cand in remaining:
            cand_vs = conditioned_score(
                cand, others, terms, lambda_div=lambda_div, beta_ent=beta_ent
            )
            gain = cand_vs["total"] - vic_sc["total"]
            if gain >= replace_margin and gain > best_gain:
                best_gain = gain
                best_move = (cand, victim, cand_vs, vic_sc)

        if best_move is None:
            break
        cand, victim, cand_vs, vic_sc = best_move
        selected = [s if s["id"] != victim["id"] else cand for s in selected]
        remaining = [x for x in remaining if x["id"] != cand["id"]]
        protected_ids.add(cand["id"])
        replacements.append(_replacement_log(cand, victim, pool_index, cand_vs, vic_sc))

    seed_ids = {s["id"] for s in selected}
    baseline_ids = {s["id"] for s in baseline}
    return selected, {
        "mode": "query_conditioned_seed_protected",
        "k": k,
        "pool_size": len(pool_ranked),
        "lambda_div": lambda_div,
        "beta_ent": beta_ent,
        "protect_prefix": protect_prefix,
        "max_replacements": max_replacements,
        "replace_margin": replace_margin,
        "query_terms": terms,
        "n_replacements": len(replacements),
        "replacements": replacements,
        "n_promoted": len(seed_ids - baseline_ids),
        "n_displaced": len(baseline_ids - seed_ids),
        "overlap_with_baseline_topk": len(seed_ids & baseline_ids),
        "promoted_dense_ranks": [r["in_dense_rank"] for r in replacements],
        "displaced_dense_ranks": [r["out_dense_rank"] for r in replacements],
    }


def select_greedy_unprotected(
    pool: list[dict[str, Any]],
    question: str,
    k: int,
    *,
    lambda_div: float,
    beta_ent: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedy fill of k slots from the pool; no baseline seed protection."""
    pool_ranked = ranked(pool)
    if not pool_ranked:
        return [], {"mode": "greedy_unprotected", "k": k, "pool_size": 0}

    terms = query_terms(question)
    remaining = list(pool_ranked)
    selected: list[dict[str, Any]] = []
    pick_log: list[dict[str, Any]] = []
    pool_index = {item["id"]: i + 1 for i, item in enumerate(pool_ranked)}

    while len(selected) < k and remaining:
        best = None
        best_sc = None
        for cand in remaining:
            sc = conditioned_score(
                cand, selected, terms, lambda_div=lambda_div, beta_ent=beta_ent
            )
            if best is None or sc["total"] > best_sc["total"]:  # type: ignore[index]
                best = cand
                best_sc = sc
        assert best is not None and best_sc is not None
        selected.append(best)
        remaining = [x for x in remaining if x["id"] != best["id"]]
        pick_log.append(
            {
                "id": best["id"],
                "dense_rank": pool_index[best["id"]],
                **best_sc,
            }
        )

    baseline_ids = {s["id"] for s in pool_ranked[:k]}
    seed_ids = {s["id"] for s in selected}
    return selected, {
        "mode": "query_conditioned_greedy_unprotected",
        "k": k,
        "pool_size": len(pool_ranked),
        "lambda_div": lambda_div,
        "beta_ent": beta_ent,
        "query_terms": terms,
        "n_replacements": len(seed_ids - baseline_ids),
        "n_promoted": len(seed_ids - baseline_ids),
        "n_displaced": len(baseline_ids - seed_ids),
        "overlap_with_baseline_topk": len(seed_ids & baseline_ids),
        "promoted_dense_ranks": sorted(
            pool_index[i] for i in seed_ids - baseline_ids
        ),
        "displaced_dense_ranks": sorted(
            pool_index[i] for i in baseline_ids - seed_ids
        ),
        "first_picks": pick_log[:8],
    }
