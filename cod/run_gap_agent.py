"""Gap-Directed Agent selection-only runner + budget curves (no LLM)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .analyze_gap_agent import write_report
from .config import load_config
from .expand import expand_seeds
from .gap_agent import run_gap_directed_agent
from .paths import resolve_args_paths
from .run import gold_coverage, gold_raw_entry_ids, load_questions
from .scoring import ranked
from .selector import select_greedy_unprotected
from .store import RawMemoryStore, create_embedder

BUDGETS = [10, 20, 30, 40, 50, 60]


def evaluate_seeds(
    store: Any,
    item: dict[str, Any],
    seeds: list[dict[str, Any]],
    expand_window: int,
) -> dict[str, Any]:
    evidence = expand_seeds(store, item["conversation_id"], seeds, expand_window)
    coverage = gold_coverage(item.get("gold_evidence") or [], evidence)
    return {
        "gold_coverage": coverage,
        "final_evidence_count": len(evidence),
        "seed_count": len(seeds),
    }


def budget_curve_for_ordered_seeds(
    store: Any,
    item: dict[str, Any],
    ordered_seeds: list[dict[str, Any]],
    expand_window: int,
    budgets: list[int] = BUDGETS,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for budget in budgets:
        prefix = ordered_seeds[: min(budget, len(ordered_seeds))]
        if not prefix:
            out[str(budget)] = {
                "gold_coverage": {
                    "gold_coverage_count": 0,
                    "gold_coverage_total": len(item.get("gold_evidence") or []),
                    "gold_coverage_ratio": 0.0,
                },
                "final_evidence_count": 0,
                "seed_count": 0,
            }
            continue
        out[str(budget)] = evaluate_seeds(store, item, prefix, expand_window)
    return out


def _final_cov(curve: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return curve.get("60") or curve.get(str(max(BUDGETS))) or {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    resolve_args_paths(args, "output_dir", "runs_dir")
    config = load_config(args.config, require_api=False)
    questions = load_questions(config["data_path"], config["subset_path"])
    if args.limit is not None:
        questions = questions[: args.limit]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    embedder = create_embedder(config["embed_model"], config.get("device", "cpu"))
    store = RawMemoryStore(
        embedder=embedder, store_dir=args.qdrant_dir or config["raw_store_dir"]
    )
    expand_window = int(config["expand_window"])

    per_q_path = out_dir / "per_question.jsonl"
    traj_path = out_dir / "trajectories.jsonl"
    traj_runs = runs_dir / "trajectories.jsonl"

    records: list[dict[str, Any]] = []
    with per_q_path.open("w", encoding="utf-8") as pq, traj_path.open(
        "w", encoding="utf-8"
    ) as tj, traj_runs.open("w", encoding="utf-8") as tj2:
        for i, item in enumerate(questions, start=1):
            qid = item["question_id"]
            pool = store.search_memory(
                item["conversation_id"], item["question"], args.pool_k
            )
            pool_ranked = ranked(pool)
            gold_ids = set(gold_raw_entry_ids(item))

            dense_seeds = pool_ranked[:60]
            dense_curve = budget_curve_for_ordered_seeds(
                store, item, dense_seeds, expand_window
            )

            cod_seeds, cod_meta = select_greedy_unprotected(
                pool,
                item["question"],
                60,
                lambda_div=float(args.lambda_div),
                beta_ent=float(args.beta_ent),
            )
            cod_curve = budget_curve_for_ordered_seeds(
                store, item, cod_seeds, expand_window
            )

            gap_seeds, gap_meta, traj = run_gap_directed_agent(
                pool,
                item["question"],
                max_seeds=60,
                max_steps=int(args.max_steps),
                n_anchors=int(args.n_anchors),
                category=int(item.get("category") or 0),
                gold_ids=gold_ids,
            )
            gap_curve = budget_curve_for_ordered_seeds(
                store, item, gap_seeds, expand_window
            )

            row = {
                "question_id": qid,
                "category": item.get("category"),
                "gold_raw_ids": sorted(gold_ids),
                "dense": {
                    "seed_ids": [s["id"] for s in dense_seeds],
                    "budget_curve": dense_curve,
                    "final": _final_cov(dense_curve),
                },
                "cod": {
                    "seed_ids": [s["id"] for s in cod_seeds],
                    "meta": {
                        k: cod_meta.get(k)
                        for k in ("mode", "n_promoted", "overlap_with_baseline_topk")
                    },
                    "budget_curve": cod_curve,
                    "final": _final_cov(cod_curve),
                },
                "gap_agent": {
                    "seed_ids": [s["id"] for s in gap_seeds],
                    "meta": gap_meta,
                    "budget_curve": gap_curve,
                    "final": _final_cov(gap_curve),
                },
            }
            records.append(row)
            pq.write(json.dumps(row, ensure_ascii=False) + "\n")
            traj_row = {
                "question_id": qid,
                "category": item.get("category"),
                "trajectory": traj,
                "meta": gap_meta,
            }
            line = json.dumps(traj_row, ensure_ascii=False) + "\n"
            tj.write(line)
            tj2.write(line)
            pq.flush()
            tj.flush()
            tj2.flush()

            gap_cov = row["gap_agent"]["final"]["gold_coverage"]
            cod_cov = row["cod"]["final"]["gold_coverage"]
            print(
                f"[{i}/{len(questions)}] {qid} "
                f"CoD={float(cod_cov.get('gold_coverage_ratio') or 0):.2f} "
                f"Gap={float(gap_cov.get('gold_coverage_ratio') or 0):.2f} "
                f"steps={gap_meta['n_steps']} stop={gap_meta['stopped_by']}",
                flush=True,
            )

    store.close()
    write_report(per_q_path, out_dir, runs_dir)
    return {"records": records, "out_dir": str(out_dir), "runs_dir": str(runs_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gap-Directed Agent selection-only (vs CoD / dense)"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--qdrant-dir")
    parser.add_argument("--pool-k", type=int, default=200)
    parser.add_argument("--lambda-div", type=float, default=0.06)
    parser.add_argument("--beta-ent", type=float, default=0.10)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--n-anchors", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default="results/gap_agent")
    parser.add_argument("--runs-dir", default="results/gap_agent/runs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        alt = Path(__file__).resolve().parents[1] / args.config
        if alt.exists():
            args.config = str(alt)
    run(args)


if __name__ == "__main__":
    main()
