"""Runtime helpers to read model parameters from Skill_MAS model_config.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_MODEL_CONFIG = Path(__file__).resolve().parents[1] / "skill_mas" / "model_config.json"

_API_KEY_PLACEHOLDERS = frozenset(
    {
        "",
        "your_api_key_here",
        "replace_with_your_api_key",
        "replace_me",
    }
)


def load_model_config() -> dict[str, Any]:
    payload = json.loads(_MODEL_CONFIG.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolve_api_key(config_value: Any = None) -> str | None:
    """Resolve an API key from config or standard environment variables."""
    val = str(config_value).strip() if config_value is not None else ""
    if val and val.lower() not in _API_KEY_PLACEHOLDERS and not val.lower().startswith("your_"):
        return val
    for env_name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "DEEPSEEK_API_KEY"):
        env_val = os.environ.get(env_name, "").strip()
        if env_val:
            return env_val
    return None


def resolve_base_url(config_value: Any = None) -> str | None:
    """Resolve an OpenAI-compatible base URL from config or environment variables."""
    val = str(config_value).strip() if config_value is not None else ""
    if val and not val.lower().startswith("your_"):
        return val
    for env_name in ("OPENAI_API_BASE", "OPENAI_BASE_URL"):
        env_val = os.environ.get(env_name, "").strip()
        if env_val:
            return env_val
    return None


def model_runtime_params(model_name: str) -> dict[str, Any]:
    cfg = load_model_config()
    row = cfg.get(str(model_name), {})
    if not isinstance(row, dict):
        return {}
    allowed = ("api_key", "base_url", "temperature", "reasoning_effort", "max_tokens")
    out = {k: row[k] for k in allowed if k in row and row[k] is not None}
    resolved_key = resolve_api_key(row.get("api_key"))
    if resolved_key:
        out["api_key"] = resolved_key
    resolved_base = resolve_base_url(row.get("base_url"))
    if resolved_base:
        out["base_url"] = resolved_base
    return out


def apply_model_runtime_params(model_name: str, llm_args: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(llm_args or {})
    out.update(model_runtime_params(model_name))
    return out
