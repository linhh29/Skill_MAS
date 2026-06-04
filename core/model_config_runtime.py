"""Runtime helpers to read model parameters from Skill_MAS model_config.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MODEL_CONFIG = Path(__file__).resolve().parents[1] / "skill_mas" / "model_config.json"


def load_model_config() -> dict[str, Any]:
    payload = json.loads(_MODEL_CONFIG.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def model_runtime_params(model_name: str) -> dict[str, Any]:
    cfg = load_model_config()
    row = cfg.get(str(model_name), {})
    if not isinstance(row, dict):
        return {}
    allowed = ("api_key", "base_url", "temperature", "reasoning_effort", "max_tokens")
    return {k: row[k] for k in allowed if k in row and row[k] is not None}


def apply_model_runtime_params(model_name: str, llm_args: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(llm_args or {})
    out.update(model_runtime_params(model_name))
    return out
