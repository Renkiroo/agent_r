"""Full LoCoMo evaluation for CoD (Chain on Demand)."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from openai import OpenAI

from .paths import resolve_args_paths
from .config import load_config
from .selector import select_greedy_unprotected
from .run import (
    estimate_evidence_tokens,
    generate_answer,
    gold_coverage,
    load_questions,
)
from .expand import DEFAULT_BETA, DEFAULT_LAMBDA, expand_seeds
from .store import RawMemoryStore, create_embedder


def judge_once(
    question: str,
    reference: str,
    answer: str,
    client: OpenAI,
    model_name: str,
) -> int:
    """Same LoCoMo judge prompt; tolerant to intermittent proxy json_object errors."""
    from llm_judge import ACCURACY_PROMPT, extract_json

    prompt = ACCURACY_PROMPT.format(
        question=question, gold_answer=reference, generated_answer=answer
    )
    prompt = prompt.rstrip() + "\n\nReturn a JSON object only."

    def _call(use_response_format: bool) -> int:
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "store": False,
        }
        if use_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        label = json.loads(extract_json(response.choices[0].message.content))["label"]
        return 1 if label == "CORRECT" else 0

    last_err: Exception | None = None
    for attempt in range(8):
        try:
            try:
                return _call(True)
            except Exception as exc:
                msg = str(exc).lower()
                if "json" in msg or "502" in msg or "response_format" in msg:
                    return _call(False)
                raise
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(min(2**attempt, 30))
    raise last_err  # type: ignore[misc]


def answer_and_judge(
    item: dict[str, Any],
    evidence: list[dict[str, Any]],
    client: OpenAI,
    config: dict[str, Any],
) -> tuple[str, int, dict[str, int], str | None]:
    error = None
    answer = ""
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    judge_correct = 0
    try:
        answer, tokens = generate_answer(
            client,
            config["llm_model"],
            item["question"],
            evidence,
            include_timestamp=True,
            temperature=float(config.get("temperature", 0)),
            max_tokens=int(config.get("answer_max_tokens", 512)),
        )
        judge_correct = int(
            judge_once(
                item["question"],
                item["reference"],
                answer,
                client,
                config["judge_model"],
            )
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        judge_correct = 0
    return answer, judge_correct, tokens, error


def pack_result(
    seeds: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    meta: dict[str, Any],
    item: dict[str, Any],
    answer: str,
    judge_correct: int,
    tokens: dict[str, int],
    error: str | None,
) -> dict[str, Any]:
    coverage = gold_coverage(item.get("gold_evidence") or [], evidence)
    return {
        "seed_ids": [s["id"] for s in seeds],
        "seed_count": len(seeds),
        "final_evidence_count": len(evidence),
        "retrieved_token_estimate": estimate_evidence_tokens(evidence),
        "gold_coverage": coverage,
        "gold_coverage_ratio": float(coverage.get("gold_coverage_ratio") or 0.0),
        "selection_meta": meta,
        "answer": answer,
        "judge_correct": judge_correct,
        "answer_token_usage": tokens,
        "error": error,
        "final_evidence": evidence,
    }


def process_question(
    item: dict[str, Any],
    store: RawMemoryStore,
    client: OpenAI,
    config: dict[str, Any],
    pool_k: int,
    final_k: int,
    expand_window: int,
    lambda_div: float,
    beta_ent: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    cid = item["conversation_id"]
    question = item["question"]
    pool = store.search_memory(cid, question, pool_k)

    seeds, meta = select_greedy_unprotected(
        pool,
        question,
        final_k,
        lambda_div=lambda_div,
        beta_ent=beta_ent,
    )
    evidence = expand_seeds(store, cid, seeds, expand_window)
    answer, judge_correct, tokens, error = answer_and_judge(
        item, evidence, client, config
    )
    result = pack_result(
        seeds, evidence, meta, item, answer, judge_correct, tokens, error
    )

    return {
        **item,
        "method": "cod",
        "pool_k": pool_k,
        "final_k": final_k,
        "expand_window": expand_window,
        "hyperparams": {
            "lambda_div": lambda_div,
            "beta_ent": beta_ent,
            "selector": "greedy_unprotected",
        },
        "retrieved_entries": result["final_evidence_count"],
        "latency": time.perf_counter() - started,
        **result,
    }


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            done[row["question_id"]] = row
    return done


def run(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    resolve_args_paths(args, "output_dir")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(config["data_path"], config["subset_path"])
    if args.limit is not None:
        questions = questions[: args.limit]

    result_path = output_dir / "results.jsonl"
    done = {} if args.no_resume else load_done(result_path)
    pending = [q for q in questions if q["question_id"] not in done]
    print(
        f"[full] total={len(questions)} done={len(done)} pending={len(pending)}",
        flush=True,
    )

    embedder = create_embedder(config["embed_model"], config.get("device", "cuda"))
    store = RawMemoryStore(
        embedder=embedder,
        store_dir=args.qdrant_dir or config["raw_store_dir"],
    )
    client = OpenAI(
        api_key=config["openai_api_key"], base_url=config["openai_api_base"]
    )
    pool_k = int(args.pool_k)
    final_k = int(args.final_k or config["raw_dense_top_k"])
    expand_window = int(config["expand_window"])

    worker = partial(
        process_question,
        store=store,
        client=client,
        config=config,
        pool_k=pool_k,
        final_k=final_k,
        expand_window=expand_window,
        lambda_div=float(args.lambda_div),
        beta_ent=float(args.beta_ent),
    )

    mode = "a" if done and not args.no_resume else "w"
    completed = len(done)
    with result_path.open(mode, encoding="utf-8") as result_file:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for record in pool.map(worker, pending):
                result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                result_file.flush()
                completed += 1
                print(
                    f"[{completed}/{len(questions)}] {record['question_id']} "
                    f"judge={record['judge_correct']} "
                    f"cov={record['gold_coverage_ratio']:.2f} "
                    f"repl={record['selection_meta'].get('n_replacements')}",
                    flush=True,
                )

    store.close()

    all_rows = load_done(result_path)
    ordered = [all_rows[q["question_id"]] for q in questions if q["question_id"] in all_rows]
    with result_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(ordered)
    summary = {
        "method": "cod",
        "pool_k": pool_k,
        "final_k": final_k,
        "lambda_div": float(args.lambda_div),
        "beta_ent": float(args.beta_ent),
        "question_count": n,
        "accuracy": sum(r["judge_correct"] for r in ordered) / n if n else 0.0,
        "errors": sum(bool(r.get("error")) for r in ordered),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full LoCoMo CoD evaluation")
    p.add_argument("--config", default="config.cod.yaml")
    p.add_argument("--qdrant-dir")
    p.add_argument(
        "--output-dir",
        default="results/cod",
    )
    p.add_argument("--pool-k", type=int, default=200)
    p.add_argument("--final-k", type=int, default=60)
    p.add_argument("--lambda-div", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--beta-ent", type=float, default=DEFAULT_BETA)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--no-resume", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        alt = Path(__file__).resolve().parents[1] / args.config
        if alt.exists():
            args.config = str(alt)
    config = load_config(args.config)
    run(args, config)


if __name__ == "__main__":
    main()
