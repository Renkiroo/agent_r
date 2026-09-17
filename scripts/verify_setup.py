#!/usr/bin/env python3
"""Verify the standalone CoD / Gap-Directed package."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_PATHS = [
    "data/locomo10.json",
    "data/locomo_100.json",
    "data/locomo_full.json",
    "qdrant_raw/manifest.json",
    "cod/gap_agent.py",
    "cod/selector.py",
    "cod/controller.py",
    "llm_judge.py",
]

REQUIRED_MODULES = [
    "numpy",
    "yaml",
    "qdrant_client",
    "sentence_transformers",
    "openai",
]


def check_paths() -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_PATHS:
        if not (ROOT / rel).exists():
            errors.append(f"missing path: {rel}")
    return errors


def check_imports() -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            errors.append(f"missing package: {name}")
    return errors


def check_cod_import() -> list[str]:
    errors: list[str] = []
    try:
        from cod.config import PROJECT_ROOT, load_config

        if PROJECT_ROOT != ROOT:
            errors.append(f"PROJECT_ROOT mismatch: {PROJECT_ROOT} != {ROOT}")
        cfg = load_config(require_api=False)
        for key in ("data_path", "raw_store_dir", "subset_path"):
            val = cfg.get(key)
            if not val or not Path(val).exists():
                errors.append(f"config path not found: {key}={val}")
    except Exception as exc:
        errors.append(f"cod import failed: {type(exc).__name__}: {exc}")
    return errors


def run_selection_smoke() -> list[str]:
    errors: list[str] = []
    try:
        from cod.config import load_config
        from cod.gap_agent import run_gap_directed_agent
        from cod.run import load_questions
        from cod.store import RawMemoryStore, create_embedder

        config = load_config(require_api=False)
        questions = load_questions(config["data_path"], config["subset_path"])
        item = questions[0]
        device = config.get("device", "cpu")
        embedder = create_embedder(config["embed_model"], device)
        store = RawMemoryStore(embedder=embedder, store_dir=config["raw_store_dir"])
        pool = store.search_memory(item["conversation_id"], item["question"], 50)
        seeds, meta, _ = run_gap_directed_agent(
            pool, item["question"], max_seeds=10, max_steps=20
        )
        store.close()
        if not seeds:
            errors.append("gap agent returned empty seeds")
        if not meta.get("stopped_by"):
            errors.append("gap agent missing stopped_by in meta")
    except Exception as exc:
        errors.append(f"selection smoke failed: {type(exc).__name__}: {exc}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CoD / Gap-Directed setup")
    parser.add_argument(
        "--smoke-selection",
        action="store_true",
        help="run one gap-agent selection (loads embedder + qdrant)",
    )
    args = parser.parse_args()

    all_errors: list[str] = []
    all_errors.extend(check_paths())
    all_errors.extend(check_imports())
    all_errors.extend(check_cod_import())
    if args.smoke_selection:
        all_errors.extend(run_selection_smoke())

    if all_errors:
        print("VERIFY FAILED:")
        for err in all_errors:
            print(f"  - {err}")
        sys.exit(1)

    print("VERIFY OK")
    print(f"  root: {ROOT}")
    if args.smoke_selection:
        print("  selection smoke: passed")


if __name__ == "__main__":
    main()
