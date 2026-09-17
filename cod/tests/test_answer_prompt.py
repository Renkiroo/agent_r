from __future__ import annotations

import os

import pytest
from openai import OpenAI

from cod.config import load_config
from cod.controller import format_evidence
from cod.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_PROMPT
from cod.run import generate_answer


def test_format_evidence_keeps_speaker_metadata_explicit() -> None:
    evidence = [
        {
            "id": "conv/session_1/turn_7",
            "speaker": "Joanna",
            "timestamp": "2022-08-22 10:57:00",
            "text": "I celebrated by making this delicious treat - yum!",
        }
    ]
    rendered = format_evidence(evidence, include_timestamp=True)
    assert rendered == (
        "[2022-08-22 10:57:00] Joanna: "
        "I celebrated by making this delicious treat - yum!"
    )
    assert "Joanna:" in rendered
    dense = format_evidence(evidence, include_timestamp=False)
    assert dense == "Joanna: I celebrated by making this delicious treat - yum!"


def test_answer_prompt_requires_speaker_and_premise_checks() -> None:
    lowered = ANSWER_SYSTEM_PROMPT.lower()
    assert "speaker" in lowered
    assert "premise" in lowered
    assert "mismatch" in lowered or "inconsistent" in lowered
    assert "chain-of-thought" in lowered
    assert "timestamp speaker: utterance" in ANSWER_USER_PROMPT.lower()


def _live_answer(question: str, evidence: list[dict]) -> str:
    config = load_config()
    client = OpenAI(
        api_key=config["openai_api_key"], base_url=config["openai_api_base"]
    )
    answer, _ = generate_answer(
        client,
        config["llm_model"],
        question,
        evidence,
        include_timestamp=True,
        temperature=float(config.get("temperature", 0)),
        max_tokens=int(config.get("answer_max_tokens", 512)),
    )
    return answer


@pytest.mark.skipif(
    os.environ.get("OURS1_LIVE_ANSWER_TESTS") != "1",
    reason="Set OURS1_LIVE_ANSWER_TESTS=1 to call the answer LLM",
)
def test_live_a_speaker_attribution_joanna_not_nate() -> None:
    answer = _live_answer(
        "How did Nate celebrate after sharing his book with a writers group?",
        [
            {
                "id": "x",
                "speaker": "Joanna",
                "timestamp": "2022-08-22 10:57:00",
                "text": (
                    "Thanks, Nate! It feels great knowing that people like my "
                    "writing. I celebrated by making this delicious treat - yum!"
                ),
            }
        ],
    )
    lowered = answer.lower()
    assert "joanna" in lowered
    assert "delicious treat" in lowered or "treat" in lowered
    assert "nate celebrated" not in lowered
    assert "does not say how anyone celebrated" not in lowered


@pytest.mark.skipif(
    os.environ.get("OURS1_LIVE_ANSWER_TESTS") != "1",
    reason="Set OURS1_LIVE_ANSWER_TESTS=1 to call the answer LLM",
)
def test_live_b_tim_fantasy_passion() -> None:
    answer = _live_answer(
        "What passion does Tim mention connects him with people from all over the world?",
        [
            {
                "id": "x",
                "speaker": "Tim",
                "timestamp": "2023-01-01 12:00:00",
                "text": (
                    "I love how my passion for fantasy stuff brings me closer "
                    "to people from all over the world."
                ),
            }
        ],
    )
    lowered = answer.lower()
    assert "fantasy" in lowered


@pytest.mark.skipif(
    os.environ.get("OURS1_LIVE_ANSWER_TESTS") != "1",
    reason="Set OURS1_LIVE_ANSWER_TESTS=1 to call the answer LLM",
)
def test_live_c_prized_possession_basketball() -> None:
    answer = _live_answer(
        "What does Tim have that serves as a reminder of hard work and is his prized possession?",
        [
            {
                "id": "x",
                "speaker": "Tim",
                "timestamp": "2023-01-01 12:00:00",
                "text": (
                    "this is my prized possession, a basketball signed by my "
                    "favorite player. It serves as a reminder of all the hard work."
                ),
            }
        ],
    )
    lowered = answer.lower()
    assert "basketball" in lowered


@pytest.mark.skipif(
    os.environ.get("OURS1_LIVE_ANSWER_TESTS") != "1",
    reason="Set OURS1_LIVE_ANSWER_TESTS=1 to call the answer LLM",
)
def test_live_d_rejects_coding_project_premise() -> None:
    answer = _live_answer(
        "What was the setback Tim faced in his coding project on 21 November, 2023?",
        [
            {
                "id": "x",
                "speaker": "Tim",
                "timestamp": "2023-11-21 12:00:00",
                "text": (
                    "I tried writing a story based on my experiences in the UK, "
                    "but it didn't go the way I wanted."
                ),
            }
        ],
    )
    lowered = answer.lower()
    assert "story" in lowered or "writing" in lowered
    assert "uk" in lowered
    # May note coding mismatch, but must not invent a coding setback.
    assert "bug" not in lowered
    assert "compiler" not in lowered
