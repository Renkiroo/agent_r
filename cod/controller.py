from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from .prompts import CONTROLLER_SYSTEM_PROMPT, CONTROLLER_USER_PROMPT
from .schema import AgentState, ControllerDecision


def format_evidence(evidence: list[dict[str, Any]], include_timestamp: bool = True) -> str:
    """Render retrieved utterances with explicit speaker metadata.

    Speaker comes from RawMemoryEntry metadata already present on each hit; this
    function only presents it. It does not extract speakers from text.
    """
    if not evidence:
        return "(no evidence)"
    lines = []
    for item in evidence:
        speaker = item.get("speaker") or "Unknown"
        text = item.get("text", "")
        if include_timestamp:
            timestamp = item.get("timestamp") or "unknown time"
            lines.append(f"[{timestamp}] {speaker}: {text}")
        else:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def parse_controller_decision(content: str) -> ControllerDecision:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content.strip(), re.DOTALL)
    raw = match.group(1) if match else content.strip()
    payload = json.loads(raw)
    status = payload.get("status")
    if status not in {"sufficient", "need_more_evidence"}:
        raise ValueError(f"Unsupported controller status: {status!r}")
    query = payload.get("query")
    if status == "need_more_evidence" and not query:
        raise ValueError("need_more_evidence requires a non-empty query")
    return ControllerDecision(
        status=status,
        query=query,
        reason=payload.get("reason"),
        expand_entry_id=payload.get("expand_entry_id"),
    )


def merge_evidence(
    current: list[dict[str, Any]], additions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = {item["id"]: item for item in current}
    for item in additions:
        existing = merged.get(item["id"])
        if existing is None:
            merged[item["id"]] = item
        elif existing.get("score") is None and item.get("score") is not None:
            merged[item["id"]] = item
    return list(merged.values())


class IterativeMemoryController:
    """Frozen LLM controller for iterative dense retrieval (IMR)."""

    def __init__(
        self,
        client: Any,
        model: str,
        memory_store: Any,
        top_k: int = 10,
        max_steps: int = 3,
        expand_window: int = 2,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.client = client
        self.model = model
        self.memory_store = memory_store
        self.top_k = top_k
        self.max_steps = max_steps
        self.expand_window = expand_window
        self.temperature = temperature
        self.max_tokens = max_tokens

    def decide(
        self, question: str, evidence: list[dict[str, Any]]
    ) -> tuple[ControllerDecision, dict[str, int]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CONTROLLER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": CONTROLLER_USER_PROMPT.format(
                        question=question,
                        evidence=format_evidence(evidence),
                    ),
                },
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = getattr(response, "usage", None)
        token_usage = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
        return parse_controller_decision(response.choices[0].message.content), token_usage

    def run(self, question: str, conversation_id: str) -> dict[str, Any]:
        state = AgentState(
            original_query=question,
            retrieval_history=[],
            current_evidence=[],
            actions=[],
        )
        retrieval_query = question
        controller_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        stop_reason = "max_steps"

        for step in range(self.max_steps):
            state.step = step
            hits = self.memory_store.search_memory(
                conversation_id, retrieval_query, self.top_k
            )
            state.current_evidence = merge_evidence(state.current_evidence, hits)
            action = {
                "step": step,
                "action": "search_memory",
                "retrieval_query": retrieval_query,
                "returned_ids": [item["id"] for item in hits],
                "results": hits,
            }
            state.retrieval_history.append(action)
            state.actions.append(action)

            decision, usage = self.decide(question, state.current_evidence)
            for key in controller_tokens:
                controller_tokens[key] += usage[key]
            action["decision"] = asdict(decision)

            if decision.expand_entry_id:
                known_ids = {item["id"] for item in state.current_evidence}
                if decision.expand_entry_id in known_ids:
                    expanded = self.memory_store.expand_context(
                        conversation_id,
                        decision.expand_entry_id,
                        self.expand_window,
                    )
                    state.current_evidence = merge_evidence(
                        state.current_evidence, expanded
                    )
                    expand_action = {
                        "step": step,
                        "action": "expand_context",
                        "entry_id": decision.expand_entry_id,
                        "window": self.expand_window,
                        "returned_ids": [item["id"] for item in expanded],
                        "results": expanded,
                    }
                    state.retrieval_history.append(expand_action)
                    state.actions.append(expand_action)

            if decision.status == "sufficient":
                stop_reason = "sufficient"
                state.actions.append({"step": step, "action": "stop"})
                break
            retrieval_query = decision.query or retrieval_query

        return {
            "state": {
                "original_query": state.original_query,
                "retrieval_history": state.retrieval_history,
                "current_evidence": state.current_evidence,
                "actions": state.actions,
                "step": state.step,
            },
            "final_evidence": state.current_evidence,
            "num_rounds": state.step + 1,
            "num_tool_calls": len(state.retrieval_history),
            "stop_reason": stop_reason,
            "controller_token_usage": controller_tokens,
            "reward": None,
        }
