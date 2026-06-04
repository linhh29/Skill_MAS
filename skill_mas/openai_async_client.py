"""Async OpenAI client for Skill_MAS code planner."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import httpx
from openai import AsyncOpenAI

# --- Why ``httpx.ReadError`` / mid-response failures happen (no code fix here; environment/Payload) ---
#
# - **TCP/TLS path**: HPC / NAT / firewall / proxy may RST idle sockets or drop long transfers;
#   ``ReadError`` means the HTTP layer expected more bytes on the body and the connection ended.
# - **Payload shape**: contrastive sends a very large JSON body (trajectories + embedded schema); long
#   requests + long responses increase time on the wire → more exposure to flaky links.
# - **Stack**: ``AsyncOpenAI`` uses **httpx/httpcore**, not ``urllib3``; behavior differs from Vita's
#   ``requests`` path (curl can succeed while Python fails on the same host).
# - **Mitigations** (ops): stable egress, proxy if required, reduce concurrency, smaller reflection
#   batches, or run from a node with reliable WAN — not application retries.
#
# Optional timeouts (seconds); tune if the link is slow but stable.
_DEFAULT_READ_TIMEOUT = float(os.environ.get("SKILL_MAS_OPENAI_READ_TIMEOUT", "600"))
_DEFAULT_CONNECT_TIMEOUT = float(os.environ.get("SKILL_MAS_OPENAI_CONNECT_TIMEOUT", "120"))

MODEL_CONFIG_PATH = Path(__file__).resolve().parent / "model_config.json"


def _resolve_api_key(raw: str | None) -> str | None:
    try:
        from Skill_MAS.utils.secrets_resolve import resolve_secret
    except ImportError:
        from utils.secrets_resolve import resolve_secret  # type: ignore[no-redef]
    return resolve_secret(raw)

# ``pipeline.evolve`` merges Vita ``models.get(model_id)`` into optimizer args; that dict often
# contains display-only keys (``name``, ``description``, …) that are **not** OpenAI API parameters.
# Only pass through keys accepted by ``AsyncOpenAI.chat.completions.create``.
_CHAT_COMPLETION_ALLOWED_OPTIONAL_KEYS = frozenset(
    {
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "n",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning_effort",
        "logit_bias",
        "user",
        "metadata",
        "modalities",
        "prediction",
        "audio",
        "service_tier",
        "store",
        "web_search_options",
        # DashScope / Qwen-compatible extensions (merged into JSON body by the SDK).
        "extra_body",
    }
)


def chat_completion_extras_from_optimizer_llm_args(args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Filter optimizer/runtime kwargs into arguments accepted by ``chat.completions.create``."""
    if not args:
        return {}
    out: dict[str, Any] = {}
    for k, v in dict(args).items():
        if k not in _CHAT_COMPLETION_ALLOWED_OPTIONAL_KEYS or v is None:
            continue
        out[str(k)] = v
    return out


def structured_json_candidate_text(message: Any) -> str:
    """
    Prefer ``message.content``; for some DashScope/Qwen reasoning responses the structured
    JSON may appear only under ``reasoning_content`` / ``reasoning`` when ``content`` is empty.
    """
    content = (getattr(message, "content", None) or "").strip()
    if content:
        return content
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for key in ("reasoning_content", "reasoning"):
            v = extra.get(key)
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("{") or s.startswith("["):
                    return s
    for key in ("reasoning_content", "reasoning"):
        v = getattr(message, key, None)
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("{") or s.startswith("["):
                return s
    return content


def _load_model_config() -> dict[str, Any]:
    if not MODEL_CONFIG_PATH.is_file():
        return {}
    data = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_pricing_table(path: Optional[Path] = None) -> dict[str, Any]:
    """
    Skill_MAS-local pricing loader.
    Uses model_config.json and filters out meta keys like "_defaults".
    """
    p = path or MODEL_CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"pricing file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"pricing file must contain a JSON object: {p}")
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def normalize_usage_tokens(usage: Mapping[str, Any] | None) -> tuple[int, int, int]:
    """Return (prompt_tokens, completion_tokens, total_tokens)."""
    if not usage:
        return 0, 0, 0
    u = dict(usage)
    prompt = int(u.get("prompt_token_count", u.get("prompt_tokens", u.get("input_tokens", 0))) or 0)
    output = int(
        u.get("candidates_token_count", u.get("completion_tokens", u.get("output_tokens", 0))) or 0
    )
    total = int(u.get("total_token_count", u.get("total_tokens", prompt + output)) or 0)
    return prompt, output, total


def _row_to_per_1m(row: Any) -> tuple[float, float] | None:
    if not isinstance(row, dict):
        return None
    if "input_per_1m" in row and "output_per_1m" in row:
        return float(row["input_per_1m"]), float(row["output_per_1m"])
    if "input_per_1k" in row and "output_per_1k" in row:
        return float(row["input_per_1k"]) * 1000.0, float(row["output_per_1k"]) * 1000.0
    return None


def lookup_model_rates(table: Mapping[str, Any], model: str) -> tuple[float, float, str]:
    """
    Returns (input_usd_per_1m_tokens, output_usd_per_1m_tokens, resolved_key).
    Strict lookup only: no default row.
    """
    key = (model or "").strip()
    if key and key in table:
        rates = _row_to_per_1m(table[key])
        if rates:
            return rates[0], rates[1], key
    lower_map = {str(k).lower(): k for k in table if not str(k).startswith("_")}
    lk = key.lower()
    if lk and lk in lower_map:
        k = lower_map[lk]
        rates = _row_to_per_1m(table[k])
        if rates:
            return rates[0], rates[1], str(k)
    allowed = sorted(str(k) for k in table if not str(k).startswith("_"))
    raise ValueError(f"Unknown pricing model {model!r}. Available keys: {allowed}")


def estimate_cost_usd(
    *,
    model: str,
    usage: Mapping[str, Any] | None,
    table: Optional[Mapping[str, Any]] = None,
) -> tuple[float, int, int, str]:
    """Returns (cost_usd, prompt_tokens, output_tokens, pricing_key_used)."""
    prompt, output, _ = normalize_usage_tokens(usage)
    t = dict(table) if table is not None else load_pricing_table()
    inp_1m, out_1m, pkey = lookup_model_rates(t, model)
    cost = (inp_1m * prompt + out_1m * output) / 1_000_000.0
    return float(cost), prompt, output, pkey


def _usage_dict_from_chat_completion(response: Any) -> dict[str, Any]:
    """
    Build token counts for pricing from ``chat.completion`` ``usage``.

    Dashscope / DeepSeek / Qwen often include ``completion_tokens_details.reasoning_tokens``.
    That field is a *breakdown* of output: billable ``completion_tokens`` already matches
    ``total_tokens - prompt_tokens`` when the gateway is consistent (see raw API JSON).
    Do **not** add ``reasoning_tokens`` again or output will be double-counted.

    If ``prompt_tokens + completion_tokens != total_tokens``, reconcile using ``total_tokens``
    (some gateways omit overlap correctly only at top level).
    """
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    pt = int(getattr(u, "prompt_tokens", None) or 0)
    ct = int(getattr(u, "completion_tokens", None) or 0)
    tt = int(getattr(u, "total_tokens", None) or 0)
    if tt > 0 and pt + ct != tt:
        ct = max(tt - pt, 0)
    elif tt == 0 and (pt or ct):
        tt = pt + ct
    return {"input_tokens": pt, "output_tokens": ct, "total_tokens": tt}


def enrich_usage_with_cost(
    usage: MutableMapping[str, Any] | None,
    *,
    model: str,
    table: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return a new usage dict with estimated_cost_usd and pricing metadata."""
    u = dict(usage or {})
    cost, _p, _o, pkey = estimate_cost_usd(model=model, usage=u, table=table)
    u["estimated_cost_usd"] = round(cost, 10)
    u["pricing_model_key"] = pkey
    return u


class AsyncOpenAIClient:
    """OpenAI-compatible async client for generic Skill_MAS LLM calls."""

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.model_config = _load_model_config()
        row = self.model_config.get(self.model, {}) if isinstance(self.model_config, dict) else {}

        raw_key = api_key or (row.get("api_key") if isinstance(row, dict) else None)
        self.api_key = _resolve_api_key(str(raw_key) if raw_key is not None else None)
        self.base_url = base_url or (row.get("base_url") if isinstance(row, dict) else None)
        self.reasoning_effort = row.get("reasoning_effort") if isinstance(row, dict) else None
        self.temperature = row.get("temperature") if isinstance(row, dict) else None
        self.max_tokens = row.get("max_tokens") if isinstance(row, dict) else None
        eb = row.get("extra_body") if isinstance(row, dict) else None
        self.extra_body: dict[str, Any] | None = eb if isinstance(eb, dict) else None

        _timeout = httpx.Timeout(
            _DEFAULT_READ_TIMEOUT,
            connect=_DEFAULT_CONNECT_TIMEOUT,
            read=_DEFAULT_READ_TIMEOUT,
            write=_DEFAULT_CONNECT_TIMEOUT,
        )
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=_timeout,
            max_retries=5,
        )
        has_model_pricing = isinstance(row, dict) and (
            ("input_per_1m" in row and "output_per_1m" in row)
            or ("input_per_1k" in row and "output_per_1k" in row)
        )
        if not has_model_pricing:
            raise ValueError(
                f"Model {self.model!r} must include pricing fields in {MODEL_CONFIG_PATH}. "
                "Require input/output per-1m (or per-1k) rates."
            )
        self.pricing = {self.model: row}


    async def generate(
        self,
        *,
        user_prompt: str,
        system_prompt: str = "",
        **create_kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        call_kw = dict(create_kwargs)
        merged_extra: dict[str, Any] = {}
        if self.extra_body:
            merged_extra.update(self.extra_body)
        call_extra = call_kw.pop("extra_body", None)
        if isinstance(call_extra, dict):
            merged_extra.update(call_extra)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}),
            **call_kw,
        }
        if merged_extra:
            kwargs["extra_body"] = merged_extra
        response = await self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        text = structured_json_candidate_text(msg)
        usage = _usage_dict_from_chat_completion(response)
        usage = enrich_usage_with_cost(usage, model=self.model, table=self.pricing)
        return text, usage

    async def aclose(self) -> None:
        closer = getattr(self.client, "close", None)
        if closer is None:
            return
        res = closer()
        if hasattr(res, "__await__"):
            await res  # type: ignore[misc]

