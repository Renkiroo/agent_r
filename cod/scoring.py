"""Shared scoring helpers for CoD and the Gap-Directed agent.

Dense score, ranking, text overlap, and question-entity extraction.
Shared by CoD and the Gap-Directed agent.
"""

from __future__ import annotations

import re
from typing import Any

from .store import entry_position

_SESSION_RE = re.compile(r"session_(\d+)")
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
_STOP_NAMES = {
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Why",
    "How",
    "The",
    "A",
    "An",
    "In",
    "On",
    "At",
    "For",
    "With",
    "From",
    "About",
    "Does",
    "Did",
    "Is",
    "Are",
    "Was",
    "Were",
    "Has",
    "Have",
    "Had",
    "Do",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
}


def dense_score(item: dict[str, Any]) -> float:
    val = item.get("score")
    if val is None:
        return float("-inf")
    return float(val)


_score = dense_score


def ranked(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=dense_score, reverse=True)


def session_key(item: dict[str, Any]) -> int:
    if item.get("session_index") is not None:
        return int(item["session_index"])
    eid = str(item.get("id") or "")
    match = _SESSION_RE.search(eid)
    if match:
        return int(match.group(1))
    return entry_position(eid)[0]


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (text or "").lower()))


def text_jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def extract_entities(question: str) -> list[str]:
    found: list[str] = []
    for match in _NAME_RE.finditer(question):
        name = match.group(1)
        if name in _STOP_NAMES:
            continue
        if name.lower() not in {x.lower() for x in found}:
            found.append(name)
    return found
