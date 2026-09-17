"""Summarize Gap-Directed Agent vs CoD / dense (selection-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BUDGETS = [10, 20, 30, 40, 50, 60]
METHODS = ("dense", "cod", "gap_agent")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _cov(final_block: dict[str, Any]) -> dict[str, Any]:
    return final_block.get("gold_coverage") or {}


def _count(c: dict[str, Any]) -> int:
    return int(c.get("gold_coverage_count") or 0)


def _total(c: dict[str, Any]) -> int:
    return int(c.get("gold_coverage_total") or 0)


def _ratio(c: dict[str, Any]) -> float:
    return float(c.get("gold_coverage_ratio") or 0.0)


def aggregate_final(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    matched = sum(_count(_cov(r[method]["final"])) for r in rows)
    tot = sum(_total(_cov(r[method]["final"])) for r in rows)
    full = sum(1 for r in rows if _ratio(_cov(r[method]["final"])) == 1.0)
    n = len(rows) or 1
    return {
        "n": len(rows),
        "gold_evidence_recall": matched / tot if tot else 0.0,
        "full_gold_coverage": full / n,
        "avg_gold_retrieved": matched / n,
        "retrieval_failure": n - full,
    }


def aggregate_budget(
    rows: list[dict[str, Any]], method: str, budget: int
) -> dict[str, Any]:
    key = str(budget)
    matched = tot = full = 0
    for row in rows:
        block = (row[method].get("budget_curve") or {}).get(key) or {}
        cov = block.get("gold_coverage") or {}
        matched += _count(cov)
        tot += _total(cov)
        full += int(_ratio(cov) == 1.0)
    n = len(rows) or 1
    return {
        "budget": budget,
        "gold_evidence_recall": matched / tot if tot else 0.0,
        "full_gold_coverage": full / n,
        "avg_gold_retrieved": matched / n,
        "retrieval_failure": n - full,
    }


def write_report(per_q_path: Path, out_dir: Path, runs_dir: Path | None = None) -> dict[str, Any]:
    rows = load_rows(per_q_path)
    summary = {
        "n": len(rows),
        "methods": {m: aggregate_final(rows, m) for m in METHODS},
        "budget_curves": {
            m: [aggregate_budget(rows, m, b) for b in BUDGETS] for m in METHODS
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Gap-Directed Agent (selection-only)",
        "",
        f"Questions: **{len(rows)}**",
        "",
        "| Method | Gold recall@60 | Full-gold@60 |",
        "|--------|----------------|--------------|",
    ]
    labels = {"dense": "Dense Top-K", "cod": "CoD (one-shot)", "gap_agent": "Gap-Directed Agent"}
    for key, label in labels.items():
        stats = summary["methods"][key]
        lines.append(
            f"| {label} | {stats['gold_evidence_recall']*100:.1f}% | "
            f"{stats['full_gold_coverage']*100:.1f}% |"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if runs_dir is not None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary["methods"], ensure_ascii=False, indent=2), flush=True)
    return summary
