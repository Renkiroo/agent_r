from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from llm_judge import evaluate_llm_judge

from .config import DEFAULT_CONFIG_PATH, load_config
from .controller import IterativeMemoryController, format_evidence, merge_evidence
from .prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_PROMPT
from .sde import select_seeds, selective_expand
from .store import RawMemoryStore, create_embedder


_DIA_ID_RE = re.compile(r"^D(?P<session>\d+):(?P<turn>\d+)$")


def gold_raw_entry_ids(item: dict[str, Any]) -> list[str]:
    """Map LoCoMo gold dia_ids to raw memory entry ids (diagnosis/oracle only)."""
    conversation_id = item["conversation_id"]
    mapped: list[str] = []
    for dia_id in item.get("gold_evidence_ids") or []:
        match = _DIA_ID_RE.match(str(dia_id))
        if not match:
            continue
        session = int(match.group("session"))
        turn = int(match.group("turn")) - 1
        mapped.append(f"{conversation_id}/session_{session}/turn_{turn}")
    return mapped


def load_oracle_gold_entries(
    store: RawMemoryStore, conversation_id: str, gold_ids: list[str]
) -> list[dict[str, Any]]:
    """Fetch gold utterances from the store when they fall outside dense top-k."""
    if not gold_ids:
        return []
    by_id = {entry.id: entry for entry in store._load_entries(conversation_id)}
    entries = []
    for gold_id in gold_ids:
        entry = by_id.get(gold_id)
        if entry is None:
            continue
        payload = entry.to_dict(include_embedding=False)
        payload["score"] = float("-inf")
        entries.append(payload)
    return entries


RAW_METHODS = {"raw_dense", "raw_time", "raw_temporal", "imr", "sde"}


def load_questions(data_path: str, subset_path: str) -> list[dict[str, Any]]:
    with Path(subset_path).open("r", encoding="utf-8") as handle:
        subset = json.load(handle)
    subset_by_id = {item["question_id"]: item for item in subset}
    with Path(data_path).open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    questions = []
    for sample in dataset:
        conversation_id = sample["sample_id"]
        utterance_by_id = {
            turn.get("dia_id"): {
                "id": turn.get("dia_id"),
                "speaker": turn.get("speaker"),
                "text": turn.get("text"),
                "session": session_key,
            }
            for session_key, turns in sample["conversation"].items()
            if session_key.startswith("session_")
            and not session_key.endswith("_date_time")
            and isinstance(turns, list)
            for turn in turns
            if turn.get("dia_id")
        }
        for qa_index, qa in enumerate(sample.get("qa", [])):
            question_id = f"{conversation_id}::qa_{qa_index}"
            if question_id not in subset_by_id:
                continue
            evidence_ids = qa.get("evidence", [])
            questions.append(
                {
                    "question_id": question_id,
                    "conversation_id": conversation_id,
                    "qa_index": qa_index,
                    "category": qa["category"],
                    "question": qa["question"],
                    "reference": qa.get("answer")
                    or qa.get("adversarial_answer", ""),
                    "gold_evidence_ids": evidence_ids,
                    "gold_evidence": [
                        utterance_by_id[item]
                        for item in evidence_ids
                        if item in utterance_by_id
                    ],
                    "collision_gold_evidence": subset_by_id[question_id].get(
                        "gold_evidence", []
                    ),
                }
            )
    if len(questions) != len(subset):
        raise ValueError(
            f"Mapped {len(questions)} questions, expected {len(subset)} from subset"
        )
    return questions


def retry(operation: Callable[[], Any], attempts: int = 3) -> Any:
    error = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:  # API providers expose heterogeneous errors
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise error  # type: ignore[misc]


def generate_answer(
    client: OpenAI,
    model: str,
    question: str,
    evidence: list[dict[str, Any]],
    include_timestamp: bool,
    temperature: float,
    max_tokens: int,
    seed: int | None = None,
) -> tuple[str, dict[str, int]]:
    prompt = ANSWER_USER_PROMPT.format(
        question=question,
        evidence=format_evidence(evidence, include_timestamp=include_timestamp),
    )
    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "store": False,
    }
    if seed is not None:
        create_kwargs["seed"] = seed
    response = retry(
        lambda: client.chat.completions.create(**create_kwargs)
    )
    usage = getattr(response, "usage", None)
    return response.choices[0].message.content, {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def estimate_evidence_tokens(evidence: list[dict[str, Any]]) -> int:
    return sum(max(1, len(item.get("text", "").split())) for item in evidence)


def gold_coverage(
    gold_evidence: list[dict[str, Any]], evidence: list[dict[str, Any]]
) -> dict[str, float | int]:
    retrieved_texts = [
        " ".join(str(item.get("text", "")).lower().split()) for item in evidence
    ]
    matched = 0
    for gold in gold_evidence:
        gold_text = " ".join(str(gold.get("text", "")).lower().split())
        if gold_text and any(
            gold_text in retrieved or retrieved in gold_text
            for retrieved in retrieved_texts
            if retrieved
        ):
            matched += 1
    total = len(gold_evidence)
    return {
        "gold_coverage_count": matched,
        "gold_coverage_total": total,
        "gold_coverage_ratio": matched / total if total else 0.0,
    }


def retrieve_once(
    method: str,
    store: RawMemoryStore,
    conversation_id: str,
    question: str,
    top_k: int,
    expand_window: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hits = store.search_memory(conversation_id, question, top_k)
    trajectory = [
        {
            "step": 0,
            "action": "search_memory",
            "retrieval_query": question,
            "returned_ids": [item["id"] for item in hits],
            "results": hits,
        }
    ]
    evidence = hits
    if method == "raw_temporal":
        expanded: list[dict[str, Any]] = []
        for hit in hits:
            context = store.expand_context(conversation_id, hit["id"], expand_window)
            expanded = merge_evidence(expanded, context)
        evidence = merge_evidence(hits, expanded)
        trajectory.append(
            {
                "step": 0,
                "action": "expand_context",
                "seed_ids": [item["id"] for item in hits],
                "window": expand_window,
                "returned_ids": [item["id"] for item in expanded],
                "results": expanded,
            }
        )
    return evidence, trajectory


def process_question(
    item: dict[str, Any],
    method: str,
    store: RawMemoryStore,
    client: OpenAI,
    config: dict[str, Any],
    top_k: int,
    controller: IterativeMemoryController | None,
    sde_m: int,
    sde_window: int,
    sde_strategy: str,
    sde_max_context_turns: int,
    seed_strategy: str = "score",
    rescue_core_k: int | None = None,
    rescue_k: int | None = None,
    rescue_rank_lo: int | None = None,
    rescue_rank_hi: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    error = None
    answer = ""
    evidence: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    num_rounds = 0
    num_tool_calls = 0
    stop_reason = "error"
    answer_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    controller_tokens = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    retrieval_metadata: dict[str, Any] = {}
    dense_evidence: list[dict[str, Any]] = []
    try:
        if controller is not None:
            controlled = controller.run(item["question"], item["conversation_id"])
            evidence = controlled["final_evidence"]
            trajectory = controlled["state"]["retrieval_history"]
            num_rounds = controlled["num_rounds"]
            num_tool_calls = controlled["num_tool_calls"]
            stop_reason = controlled["stop_reason"]
            controller_tokens = controlled["controller_token_usage"]
        elif method == "sde":
            dense_evidence = store.search_memory(
                item["conversation_id"], item["question"], top_k
            )
            gold_ids = (
                gold_raw_entry_ids(item) if seed_strategy == "oracle" else []
            )
            gold_entries = (
                load_oracle_gold_entries(
                    store, item["conversation_id"], gold_ids
                )
                if seed_strategy == "oracle"
                else []
            )
            seeds = select_seeds(
                dense_evidence,
                sde_m,
                strategy=seed_strategy,
                gold_entry_ids=gold_ids,
                gold_entries=gold_entries,
                rescue_core_k=rescue_core_k,
                rescue_k=rescue_k,
                rescue_rank_lo=rescue_rank_lo,
                rescue_rank_hi=rescue_rank_hi,
            )
            sde_result = selective_expand(
                store,
                item["conversation_id"],
                seeds,
                dense_candidates=len(dense_evidence),
                strategy=sde_strategy,
                window=sde_window,
                max_context_turns_per_seed=sde_max_context_turns,
                seed_selection_strategy=seed_strategy,
            )
            evidence = sde_result["evidence"]
            retrieval_metadata = sde_result["metadata"]
            if seed_strategy == "rescue":
                retrieval_metadata.update(
                    {
                        "rescue_core_k": rescue_core_k,
                        "rescue_k": rescue_k,
                        "rescue_rank_lo": rescue_rank_lo,
                        "rescue_rank_hi": rescue_rank_hi,
                    }
                )
            trajectory = [
                {
                    "step": 0,
                    "action": "search_memory",
                    "retrieval_query": item["question"],
                    "returned_ids": [entry["id"] for entry in dense_evidence],
                    "results": dense_evidence,
                },
                {
                    "step": 0,
                    "action": "selective_expand",
                    "seed_ids": [entry["id"] for entry in seeds],
                    "seed_strategy": seed_strategy,
                    "strategy": sde_strategy,
                    "window": sde_window if sde_strategy == "window" else None,
                    "max_context_turns_per_seed": (
                        sde_max_context_turns
                        if sde_strategy == "session"
                        else None
                    ),
                    "returned_ids": [entry["id"] for entry in sde_result["expanded"]],
                    "results": sde_result["expanded"],
                    "metadata": retrieval_metadata,
                },
            ]
            num_rounds = 1
            num_tool_calls = len(trajectory)
            stop_reason = "single_pass"
        else:
            evidence, trajectory = retrieve_once(
                method,
                store,
                item["conversation_id"],
                item["question"],
                top_k,
                int(config["expand_window"]),
            )
            num_rounds = 1
            num_tool_calls = len(trajectory)
            stop_reason = "single_pass"
            if trajectory:
                dense_evidence = trajectory[0].get("results", [])

        answer, answer_tokens = generate_answer(
            client,
            config["llm_model"],
            item["question"],
            evidence,
            include_timestamp=method != "raw_dense",
            temperature=float(config.get("temperature", 0)),
            max_tokens=int(config.get("answer_max_tokens", 512)),
        )
        judge_correct = retry(
            lambda: evaluate_llm_judge(
                item["question"],
                item["reference"],
                answer,
                client_obj=client,
                model_name=config["judge_model"],
            )
        )
    except Exception as exc:
        stop_reason = "error"
        judge_correct = 0
        error = f"{type(exc).__name__}: {exc}"

    coverage = gold_coverage(item.get("gold_evidence", []), evidence)
    dense_coverage = gold_coverage(
        item.get("gold_evidence", []), dense_evidence or evidence
    )
    return {
        **item,
        "method": method,
        "answer": answer,
        "prediction": answer,
        "final_evidence": evidence,
        "retrieved_evidence": evidence,
        "trajectory": trajectory,
        "judge_correct": int(judge_correct),
        "num_rounds": num_rounds,
        "num_tool_calls": num_tool_calls,
        "stop_reason": stop_reason,
        "retrieved_entries": len(evidence),
        "retrieved_token_estimate": estimate_evidence_tokens(evidence),
        "dense_gold_coverage": dense_coverage,
        "gold_coverage": coverage,
        "sde": retrieval_metadata if method == "sde" else None,
        "retrieval_metadata": retrieval_metadata,
        "answer_token_usage": answer_tokens,
        "controller_token_usage": controller_tokens,
        "latency": time.perf_counter() - started,
        "reward": None,
        "error": error,
    }


def run_raw(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    seed_strategy = getattr(args, "seed_strategy", "score")
    rescue_core_k = getattr(args, "rescue_core_k", None)
    rescue_k = getattr(args, "rescue_k", None)
    rescue_rank_lo = getattr(args, "rescue_rank_lo", None)
    rescue_rank_hi = getattr(args, "rescue_rank_hi", None)
    if args.method == "sde":
        expand_tag = (
            "session"
            if args.sde_strategy == "session"
            else f"w{args.sde_window}"
        )
        if seed_strategy == "rescue":
            if None in (rescue_core_k, rescue_k, rescue_rank_lo, rescue_rank_hi):
                raise ValueError(
                    "rescue seed strategy requires --rescue-core-k --rescue-k "
                    "--rescue-rank-lo --rescue-rank-hi"
                )
            expected_m = int(rescue_core_k) + int(rescue_k)
            if args.sde_m != expected_m:
                # Keep CLI m aligned with core+rescue budget for logging.
                args.sde_m = expected_m
            variant = (
                f"sde_rescue_m{args.sde_m}_c{rescue_core_k}"
                f"_r{rescue_rank_lo}-{rescue_rank_hi}_{expand_tag}"
            )
            results_root = Path(config["results_root"])
        else:
            seed_tag = {
                "score": "score",
                "session_diverse": "diverse",
                "oracle": "oracle",
            }.get(seed_strategy, seed_strategy)
            variant = f"sde_{seed_tag}_m{args.sde_m}_{expand_tag}"
            results_root = Path(config["results_root"])
        default_output_dir = results_root / variant
    else:
        variant = args.method
        default_output_dir = Path(config["results_root"]) / args.method
    output_dir = Path(args.output_dir or default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(config["data_path"], config["subset_path"])
    if args.limit is not None:
        questions = questions[: args.limit]

    embedder = create_embedder(config["embed_model"], config.get("device", "cuda"))
    store = RawMemoryStore(
        embedder=embedder,
        store_dir=args.qdrant_dir or config["raw_store_dir"],
    )
    client = OpenAI(
        api_key=config["openai_api_key"], base_url=config["openai_api_base"]
    )
    top_k = int(
        args.top_k
        or (
            config["imr_top_k"]
            if args.method == "imr"
            else (
                config["sde_dense_top_k"]
                if args.method == "sde"
                else config["raw_dense_top_k"]
            )
        )
    )
    controller = None
    if args.method == "imr":
        controller = IterativeMemoryController(
            client=client,
            model=config["llm_model"],
            memory_store=store,
            top_k=top_k,
            max_steps=int(config["max_steps"]),
            expand_window=int(config["expand_window"]),
            temperature=float(config.get("temperature", 0)),
            max_tokens=int(config.get("controller_max_tokens", 256)),
        )

    result_path = output_dir / "results.jsonl"
    trajectory_path = output_dir / "trajectory.jsonl"
    results = []
    worker = partial(
        process_question,
        method=args.method,
        store=store,
        client=client,
        config=config,
        top_k=top_k,
        controller=controller,
        sde_m=args.sde_m,
        sde_window=args.sde_window,
        sde_strategy=args.sde_strategy,
        sde_max_context_turns=int(config["sde_max_context_turns_per_seed"]),
        seed_strategy=seed_strategy,
        rescue_core_k=rescue_core_k,
        rescue_k=rescue_k,
        rescue_rank_lo=rescue_rank_lo,
        rescue_rank_hi=rescue_rank_hi,
    )
    executor = ThreadPoolExecutor(max_workers=args.workers)
    with result_path.open("w", encoding="utf-8") as result_file, trajectory_path.open(
        "w", encoding="utf-8"
    ) as trajectory_file:
        for index, record in enumerate(executor.map(worker, questions), start=1):
            result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            trajectory_file.write(
                json.dumps(
                    {
                        "question_id": record["question_id"],
                        "query": record["question"],
                        "trajectory": record["trajectory"],
                        "final_evidence": record["final_evidence"],
                        "answer": record["answer"],
                        "latency": record["latency"],
                        "num_rounds": record["num_rounds"],
                        "num_tool_calls": record["num_tool_calls"],
                        "reward": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            result_file.flush()
            trajectory_file.flush()
            results.append(record)
            print(
                f"[{index}/{len(questions)}] {record['question_id']} "
                f"correct={record['judge_correct']} rounds={record['num_rounds']}",
                flush=True,
            )
    executor.shutdown(wait=True)
    store.close()

    count = len(results)
    gold_matched = sum(
        int(item["gold_coverage"]["gold_coverage_count"]) for item in results
    )
    gold_total = sum(
        int(item["gold_coverage"]["gold_coverage_total"]) for item in results
    )
    dense_gold_matched = sum(
        int(item["dense_gold_coverage"]["gold_coverage_count"]) for item in results
    )
    summary = {
        "method": args.method,
        "representation": "raw",
        "llm_model": config["llm_model"],
        "judge_model": config["judge_model"],
        "embed_model": config["embed_model"],
        "question_count": count,
        "accuracy": sum(item["judge_correct"] for item in results) / count if count else 0,
        "avg_rounds": sum(item["num_rounds"] for item in results) / count if count else 0,
        "avg_retrieved_entries": (
            sum(item["retrieved_entries"] for item in results) / count if count else 0
        ),
        "avg_retrieved_tokens": (
            sum(item["retrieved_token_estimate"] for item in results) / count
            if count
            else 0
        ),
        "total_answer_tokens": sum(
            item["answer_token_usage"]["total_tokens"] for item in results
        ),
        "avg_answer_tokens": (
            sum(item["answer_token_usage"]["total_tokens"] for item in results) / count
            if count
            else 0
        ),
        "total_controller_tokens": sum(
            item["controller_token_usage"]["total_tokens"] for item in results
        ),
        "avg_controller_tokens": (
            sum(item["controller_token_usage"]["total_tokens"] for item in results)
            / count
            if count
            else 0
        ),
        "avg_latency": sum(item["latency"] for item in results) / count if count else 0,
        "errors": sum(bool(item["error"]) for item in results),
        "top_k": top_k,
        "gold_coverage": gold_matched / gold_total if gold_total else 0,
        "dense_gold_coverage": (
            dense_gold_matched / gold_total if gold_total else 0
        ),
        "max_steps": int(config["max_steps"]) if args.method == "imr" else 1,
        "expand_window": int(config["expand_window"]),
    }
    if args.method == "sde":
        summary.update(
            {
                "sde_m": args.sde_m,
                "sde_strategy": args.sde_strategy,
                "seed_strategy": seed_strategy,
                "rescue_core_k": rescue_core_k,
                "rescue_k": rescue_k,
                "rescue_rank_lo": rescue_rank_lo,
                "rescue_rank_hi": rescue_rank_hi,
                "sde_window": (
                    args.sde_window if args.sde_strategy == "window" else None
                ),
                "sde_max_context_turns_per_seed": int(
                    config["sde_max_context_turns_per_seed"]
                ),
                "avg_seeds": (
                    sum(item["sde"].get("num_seeds", 0) for item in results) / count
                    if count
                    else 0
                ),
                "avg_seed_sessions": (
                    sum(item["sde"].get("num_seed_sessions", 0) for item in results)
                    / count
                    if count
                    else 0
                ),
                "avg_seed_diversity": (
                    sum(item["sde"].get("seed_diversity", 0.0) for item in results)
                    / count
                    if count
                    else 0
                ),
                "variant": variant,
            }
        )
        baseline_path = (
            Path(config["results_root"]) / "raw_temporal" / "results.jsonl"
        )
        if baseline_path.exists():
            baseline = {
                item["question_id"]: item
                for item in (
                    json.loads(line)
                    for line in baseline_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
            paired = {"wins": 0, "losses": 0, "ties": 0}
            for item in results:
                base = baseline.get(item["question_id"])
                if not base:
                    continue
                difference = int(item["judge_correct"]) - int(base["judge_correct"])
                paired[
                    "wins" if difference > 0 else "losses" if difference < 0 else "ties"
                ] += 1
            summary["paired_vs_raw_temporal"] = paired

        metadata_keys = (
            "data_path",
            "subset_path",
            "raw_store_dir",
            "results_root",
            "device",
            "embed_model",
            "llm_model",
            "judge_model",
            "sde_dense_top_k",
            "sde_max_context_turns_per_seed",
            "temperature",
            "answer_max_tokens",
        )
        metadata = {
            "method": "sde",
            "variant": variant,
            "parameters": {
                "sde_m": args.sde_m,
                "sde_strategy": args.sde_strategy,
                "seed_strategy": seed_strategy,
                "rescue_core_k": rescue_core_k,
                "rescue_k": rescue_k,
                "rescue_rank_lo": rescue_rank_lo,
                "rescue_rank_hi": rescue_rank_hi,
                "sde_window": (
                    args.sde_window if args.sde_strategy == "window" else None
                ),
                "workers": args.workers,
                "top_k": top_k,
            },
            "config": {key: config.get(key) for key in metadata_keys},
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LoCoMo baselines (raw-dense / raw-temporal / SDE / IMR)")
    parser.add_argument(
        "--method",
        required=True,
        choices=["raw_dense", "raw_time", "raw_temporal", "imr", "sde"],
        help="Retrieval baseline. CoD and Gap-Directed Agent have dedicated runners.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--qdrant-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--limit", type=int, help="Development-only question limit")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sde-m", type=int, default=5)
    parser.add_argument("--sde-window", type=int, default=4)
    parser.add_argument(
        "--sde-strategy", choices=["window", "session"], default="window"
    )
    parser.add_argument(
        "--seed-strategy",
        choices=["score", "session_diverse", "oracle", "rescue"],
        default="score",
        help="SDE seed selection only. oracle is diagnosis-only.",
    )
    parser.add_argument("--rescue-core-k", type=int)
    parser.add_argument("--rescue-k", type=int)
    parser.add_argument("--rescue-rank-lo", type=int)
    parser.add_argument("--rescue-rank-hi", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    summary = run_raw(args, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
