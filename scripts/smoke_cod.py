#!/usr/bin/env python3
"""Smoke: CoD (Chain on Demand) on one LoCoMo question. Needs an API key."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from cod.config import load_config
from cod.run import load_questions
from cod.run_cod import process_question
from cod.store import RawMemoryStore, create_embedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--qid", default=None, help="optional fixed question_id")
    parser.add_argument("--qdrant-dir", default=None)
    parser.add_argument(
        "--output",
        default=str(ROOT / "results" / "smoke_cod.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    questions = load_questions(config["data_path"], config["subset_path"])
    rng = random.Random(args.seed)
    if args.qid:
        item = next(q for q in questions if q["question_id"] == args.qid)
    else:
        item = rng.choice(questions)

    qdrant_dir = args.qdrant_dir or config["raw_store_dir"]
    embedder = create_embedder(config["embed_model"], config.get("device", "cpu"))
    store = RawMemoryStore(embedder=embedder, store_dir=qdrant_dir)
    client = OpenAI(
        api_key=config["openai_api_key"], base_url=config["openai_api_base"]
    )

    print(
        f"[smoke] qid={item['question_id']} cat={item['category']} "
        f"q={item['question'][:120]}",
        flush=True,
    )
    record = process_question(
        item,
        store=store,
        client=client,
        config=config,
        pool_k=200,
        final_k=int(config["raw_dense_top_k"]),
        expand_window=int(config["expand_window"]),
        lambda_div=0.06,
        beta_ent=0.10,
    )
    store.close()

    slim = {
        "qid": record["question_id"],
        "category": record["category"],
        "question": record["question"],
        "reference": record["reference"],
        "judge_correct": record["judge_correct"],
        "answer": record["answer"],
        "gold_coverage_ratio": record["gold_coverage_ratio"],
        "n_replacements": record["selection_meta"].get("n_replacements"),
        "final_evidence_count": record["final_evidence_count"],
        "error": record.get("error"),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(slim, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(slim, ensure_ascii=False, indent=2))
    print(f"\n[wrote] {args.output}", flush=True)


if __name__ == "__main__":
    main()
