"""Project-root path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
BASELINES_DIR = RESULTS_DIR / "baselines"
QDRANT_DIR = PROJECT_ROOT / "qdrant_raw"

LOCOMO10 = DATA_DIR / "locomo10.json"
LOCOMO_100 = DATA_DIR / "locomo_100.json"
LOCOMO_FULL = DATA_DIR / "locomo_full.json"
COLLISION = DATA_DIR / "topic_address_collision_locomo.json"


def project_path(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))


def data_path(*parts: str) -> str:
    return str(DATA_DIR.joinpath(*parts))


def results_path(*parts: str) -> str:
    return str(RESULTS_DIR.joinpath(*parts))


def runs_path(*parts: str) -> str:
    return str(RUNS_DIR.joinpath(*parts))


def baseline_path(*parts: str) -> str:
    return str(BASELINES_DIR.joinpath(*parts))


def resolve_cli_path(path: str | Path | None, *fallback: str) -> Path:
    """Resolve CLI path relative to project root when not absolute."""
    if path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
    else:
        candidate = PROJECT_ROOT.joinpath(*fallback)
    return candidate.resolve()


def resolve_args_paths(args: Any, *fields: str) -> None:
    """Resolve relative CLI paths against project root."""
    for field in fields:
        value = getattr(args, field, None)
        if value:
            setattr(args, field, str(resolve_cli_path(value)))


def resolve_config_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    for base in (Path.cwd(), PROJECT_ROOT):
        joined = (base / candidate).resolve()
        if joined.exists():
            return joined
    return (PROJECT_ROOT / candidate).resolve()
