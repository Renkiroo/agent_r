from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config.local.yaml"

INHERITED_KEYS = (
    "openai_api_key",
    "openai_api_base",
    "llm_model",
    "judge_model",
    "embed_model",
)

_PATH_KEYS = (
    "data_path",
    "subset_path",
    "collision_path",
    "raw_store_dir",
    "results_root",
)

_LOCAL_FALLBACKS: dict[str, list[str]] = {
    "data_path": ["data/locomo10.json"],
    "subset_path": ["data/locomo_100.json", "data/locomo_full.json"],
    "collision_path": ["data/topic_address_collision_locomo.json"],
    "raw_store_dir": ["qdrant_raw"],
    "results_root": ["results/runs"],
}


def _resolve_path(
    value: str | None, root: Path, fallbacks: list[str]
) -> str | None:
    candidates: list[Path] = []
    if value:
        raw = Path(value).expanduser()
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.insert(0, root / value)
    for fb in fallbacks:
        candidates.append(root / fb)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    if value and not Path(value).is_absolute():
        return str((root / value).resolve())
    return value


def resolve_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Map remote/missing paths to local project-relative files when possible."""
    for key in _PATH_KEYS:
        if key in config or key in _LOCAL_FALLBACKS:
            config[key] = _resolve_path(
                config.get(key),
                PROJECT_ROOT,
                _LOCAL_FALLBACKS.get(key, []),
            )
    return config


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    env_map = {
        "openai_api_key": "OPENAI_API_KEY",
        "openai_api_base": "OPENAI_API_BASE",
        "llm_model": "LLM_MODEL",
        "judge_model": "JUDGE_MODEL",
        "embed_model": "EMBED_MODEL",
    }
    for key, env_name in env_map.items():
        env_val = os.environ.get(env_name)
        if env_val:
            config[key] = env_val
    return config


def load_config(
    path: str | Path | None = DEFAULT_CONFIG_PATH,
    *,
    require_api: bool = True,
) -> dict[str, Any]:
    """Load config, merge optional config.local.yaml, resolve local paths."""
    config_path = Path(path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if LOCAL_CONFIG_PATH.is_file() and config_path != LOCAL_CONFIG_PATH.resolve():
        with LOCAL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            local = yaml.safe_load(handle) or {}
        for key, value in local.items():
            if value is not None:
                config[key] = value

    ems_path = config.get("ems_config_path")
    if ems_path:
        with Path(ems_path).expanduser().open("r", encoding="utf-8") as handle:
            ems_config = yaml.safe_load(handle) or {}
        for key in INHERITED_KEYS:
            if config.get(key) in (None, ""):
                config[key] = ems_config.get(key)

    config = _apply_env_overrides(config)
    config = resolve_config_paths(config)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)

    if require_api:
        missing = [key for key in INHERITED_KEYS if not config.get(key)]
        if missing:
            raise ValueError(
                f"Missing required configuration values: {missing}. "
                "Set them in config.local.yaml or environment variables."
            )
    return config
