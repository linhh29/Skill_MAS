"""Resolve API keys from env placeholders in model_config.json."""

from __future__ import annotations

import os
import re

_ENV_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def resolve_secret(value: str | None, *, default_env: str = "OPENAI_API_KEY") -> str | None:
    """
    Resolve a config secret string.

    Supports:
    - ``${VAR_NAME}`` → ``os.environ[VAR_NAME]``
    - ``env:VAR_NAME`` → ``os.environ[VAR_NAME]``
    - empty / missing → ``os.environ[default_env]``
    - literal value (discouraged for publication; kept for local overrides)
    """
    if value is None:
        raw = ""
    else:
        raw = str(value).strip()
    if not raw:
        env = (os.environ.get(default_env) or "").strip()
        return env or None
    m = _ENV_PLACEHOLDER.match(raw)
    if m:
        env = (os.environ.get(m.group(1)) or "").strip()
        return env or None
    if raw.startswith("env:"):
        env = (os.environ.get(raw[4:].strip()) or "").strip()
        return env or None
    if raw.lower().startswith("your_") and "key" in raw.lower():
        env = (os.environ.get(default_env) or "").strip()
        return env or None
    return raw
