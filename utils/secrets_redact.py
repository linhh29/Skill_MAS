"""Strip API keys and auth headers from structures persisted to logs / JSON exports.

Kept in Skill_MAS for exports that read VitaBench JSON without importing ``vita``.
Logic mirrors ``vitabench_single/src/vita/utils/secrets_redact.py``.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

_DROP_TOP_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "secret_key",
        "access_token",
        "refresh_token",
        "openai_api_key",
        "anthropic_api_key",
    }
)

_REDACT_HEADER_NAMES: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "x-auth-token",
        "cookie",
    }
)


def _drop_key(key: str) -> bool:
    return key.strip().lower() in _DROP_TOP_KEYS


def _redact_headers(headers: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in headers.items():
        lk = str(k).strip().lower()
        if lk in _REDACT_HEADER_NAMES:
            out[k] = "<redacted>"
        elif isinstance(v, dict):
            out[k] = redact_secrets_for_export(v)
        elif isinstance(v, list):
            out[k] = redact_secrets_for_export(v)
        else:
            out[k] = v
    return out


def redact_secrets_for_export(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if _drop_key(ks):
                continue
            lk = ks.strip().lower()
            if lk == "headers" and isinstance(v, dict):
                out[k] = _redact_headers(v)
            elif isinstance(v, dict):
                out[k] = redact_secrets_for_export(v)
            elif isinstance(v, list):
                out[k] = redact_secrets_for_export(v)
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [redact_secrets_for_export(x) for x in obj]
    return obj


def redact_llm_args_for_export(llm_args: Optional[dict]) -> Optional[dict]:
    if llm_args is None:
        return None
    return redact_secrets_for_export(copy.deepcopy(llm_args))


def redact_vita_results_top_info(data: dict[str, Any]) -> dict[str, Any]:
    """Redact ``info.user_info`` / ``info.agent_info`` ``llm_args`` in a VitaBench Results dict."""
    o = copy.deepcopy(data)
    info = o.get("info")
    if isinstance(info, dict):
        for role in ("user_info", "agent_info"):
            block = info.get(role)
            if isinstance(block, dict) and isinstance(block.get("llm_args"), dict):
                block["llm_args"] = redact_llm_args_for_export(block["llm_args"])
    return o
