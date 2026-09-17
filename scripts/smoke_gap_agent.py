#!/usr/bin/env python3
"""Smoke: Gap-Directed Agent selects evidence for one question (no LLM by default)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cod.config import load_config
from cod.expand import expand_seeds
from cod.gap_agent import run_gap_directed_agent
from cod.run import load_questions
from cod.selector import select_greedy_unprotected
from cod.store import RawMemoryStore, create_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke Gap-Directed Agent selection")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--qdrant-dir", default=None)
    parser.add_argument("--qid", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-k", type=int, default=200)
    parser.add_argument("--final-k", type=int, default=60)
    parser.add_argument("--answer", action="store_true", help="also call answer LLM")
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "smoke_gap_agent.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config, require_api=args.answer)

    questions = load_questions(config["data_path"], config["subset_path"])
    rng = random.Random(args.seed)
    item = (
        next(q for q in questions if q["question_id"] == args.qid)
        if args.qid
        else rng.choice(questions)
    )

    qdrant_dir = args.qdrant_dir or config.get("raw_store_dir")
    embedder = create_embedder(config["embed_model"], config.get("device", "cpu"))
    store = RawMemoryStore(embedder=embedder, store_dir=qdrant_dir)

    question = item["question"]
    cid = item["conversation_id"]
    pool = store.search_memory(cid, question, args.pool_k)

    cod_seeds, cod_meta = select_greedy_unprotected(
        pool, question, args.final_k, lambda_div=0.06, beta_ent=0.10
    )
    gap_seeds, gap_meta, traj = run_gap_directed_agent(
        pool, question, max_seeds=args.final_k, max_steps=80
    )

    expand_w = int(config.get("expand_window", 2))
    cod_ev = expand_seeds(store, cid, cod_seeds, expand_w)
    gap_ev = expand_seeds(store, cid, gap_seeds, expand_w)

    out = {
        "question_id": item["question_id"],
        "category": item.get("category"),
        "question": question,
        "pool_k": args.pool_k,
        "final_k": args.final_k,
        "cod": {
            "n_seeds": len(cod_seeds),
            "n_evidence_after_expand": len(cod_ev),
            "meta": {
                k: cod_meta.get(k)
                for k in ("mode", "n_promoted", "overlap_with_baseline_topk")
            },
        },
        "gap_agent": {
            "n_seeds": len(gap_seeds),
            "n_evidence_after_expand": len(gap_ev),
            "stopped_by": gap_meta.get("stopped_by"),
            "n_steps": gap_meta.get("n_steps"),
            "gap_type_counts": gap_meta.get("gap_type_counts"),
            "action_counts": gap_meta.get("action_counts"),
            "seed_ids": [s["id"] for s in gap_seeds[:10]],
            "trajectory_head": traj[:6],
        },
    }

    if args.answer:
        from openai import OpenAI

        from cod.controller import format_evidence
        from cod.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_PROMPT

        client = OpenAI(
            api_key=config["openai_api_key"], base_url=config["openai_api_base"]
        )
        ctx = format_evidence(gap_ev)
        resp = client.chat.completions.create(
            model=config["llm_model"],
            temperature=float(config.get("temperature", 0)),
            max_tokens=int(config.get("answer_max_tokens", 512)),
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": ANSWER_USER_PROMPT.format(
                        question=question, evidence=ctx
                    ),
                },
            ],
        )
        out["answer"] = (resp.choices[0].message.content or "").strip()

    store.close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[wrote] {args.output}", flush=True)


if __name__ == "__main__":
    main()
