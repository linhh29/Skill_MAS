"""Shrink VitaBench trajectory blobs before sending to reflection / optimizer LLMs.

VitaBench embeds long ``result_preview`` strings (tool returns) and mirrors them in
``# [SkillMAS internal tool trace ...]`` message text. The evaluator also formats tool
calls as ``name(k=repr(v))``, so huge simulator kwargs such as ``w={...}`` (user profile /
world state) appear **without** the ``result_preview=`` prefix — those must be folded too.

Replacing these blob patterns cuts prompt tokens without touching other benchmarks.
"""

from __future__ import annotations

import re
from typing import Any

MEANINGFUL_RETURN_PLACEHOLDER = "【Meaningful return】"
NO_RETURN_PLACEHOLDER = "【No return】"

# Lines like ``    result_preview=...`` from evaluator_traj / message serialization.
_RESULT_PREVIEW_LINE_RE = re.compile(
    r"(^[\ \t]*result_preview=)([^\n]*)",
    flags=re.MULTILINE,
)

# ``some_tool(w={...})`` / ``foo(database=[...])`` from VitaBench window formatting (repr).
_KWARG_REPR_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)=(\{|\[)")

# JSON serialized tool arguments: ``"w": {...},``
_JSON_KW_RE = re.compile(r'"([A-Za-z_][\w]*)"\s*:\s*(\{|\[)')


def _placeholder_for_preview(preview: str | None) -> str:
    if preview is None:
        return NO_RETURN_PLACEHOLDER
    s = str(preview).strip()
    if not s:
        return NO_RETURN_PLACEHOLDER
    low = s.lower()
    if low in ("none", "null", "undefined"):
        return NO_RETURN_PLACEHOLDER
    if s in ("{}", "[]"):
        return NO_RETURN_PLACEHOLDER
    return MEANINGFUL_RETURN_PLACEHOLDER


def placeholder_for_result_preview(preview: str | None) -> str:
    """Same as internal folding — exported for trajectory sanitizers."""

    return _placeholder_for_preview(preview)


def _skip_python_string_literal(text: str, i: int) -> int | None:
    """Skip a Python repr single- or double-quoted string starting at ``i``."""
    if i >= len(text):
        return None
    quote = text[i]
    if quote not in "'\"":
        return None
    i += 1
    n = len(text)
    while i < n:
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote:
            return i + 1
        i += 1
    return None


def _skip_json_string_literal(text: str, i: int) -> int | None:
    """Skip a JSON double-quoted string starting at ``i``."""
    if i >= len(text) or text[i] != '"':
        return None
    i += 1
    n = len(text)
    while i < n:
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == '"':
            return i + 1
        i += 1
    return None


def _match_balanced_collection_pyrepr(text: str, open_idx: int) -> int | None:
    """``open_idx`` points to ``{`` or ``[`` (Python repr). Returns index past the closer."""
    if open_idx >= len(text) or text[open_idx] not in "[{":
        return None
    pairs = {"[": "]", "{": "}"}
    stack = [text[open_idx]]
    j = open_idx + 1
    n = len(text)
    while j < n and stack:
        c = text[j]
        if c in "'\"":
            nxt = _skip_python_string_literal(text, j)
            if nxt is None:
                return None
            j = nxt
            continue
        if c in "[{":
            stack.append(c)
        elif c in "]}":
            if not stack:
                return None
            op = stack[-1]
            exp = pairs[op]
            if c != exp:
                return None
            stack.pop()
        j += 1
    if stack:
        return None
    return j


def _match_balanced_collection_json(text: str, open_idx: int) -> int | None:
    """Same as pyrepr but JSON string rules (only ``"`` strings)."""
    if open_idx >= len(text) or text[open_idx] not in "[{":
        return None
    pairs = {"[": "]", "{": "}"}
    stack = [text[open_idx]]
    j = open_idx + 1
    n = len(text)
    while j < n and stack:
        c = text[j]
        if c == '"':
            nxt = _skip_json_string_literal(text, j)
            if nxt is None:
                return None
            j = nxt
            continue
        if c in "[{":
            stack.append(c)
        elif c in "]}":
            if not stack:
                return None
            op = stack[-1]
            exp = pairs[op]
            if c != exp:
                return None
            stack.pop()
        j += 1
    if stack:
        return None
    return j


def _collapse_large_kwarg_reprs(text: str, min_chars: int = 1200) -> str:
    """Fold ``name={...}`` / ``name=[...]`` Python repr blobs (e.g. ``w={...}``)."""
    out: list[str] = []
    i = 0
    while True:
        m = _KWARG_REPR_RE.search(text, i)
        if not m:
            out.append(text[i:])
            break
        open_idx = m.start(2)
        end = _match_balanced_collection_pyrepr(text, open_idx)
        if end is None:
            out.append(text[i : m.start() + 1])
            i = m.start() + 1
            continue
        val_len = end - open_idx
        out.append(text[i : m.start()])
        key = m.group(1)
        if val_len >= min_chars:
            out.append(f"{key}={MEANINGFUL_RETURN_PLACEHOLDER}")
        else:
            out.append(text[m.start() : end])
        i = end
    return "".join(out)


def _collapse_large_json_kw_values(text: str, min_chars: int = 1200) -> str:
    """Fold large JSON object/array values after ASCII-looking keys (tool ``arguments`` dumps)."""
    out: list[str] = []
    i = 0
    while True:
        m = _JSON_KW_RE.search(text, i)
        if not m:
            out.append(text[i:])
            break
        open_idx = m.start(2)
        end = _match_balanced_collection_json(text, open_idx)
        if end is None:
            out.append(text[i : m.start() + 1])
            i = m.start() + 1
            continue
        val_len = end - open_idx
        out.append(text[i : m.start()])
        key = m.group(1)
        if val_len >= min_chars:
            # Preserve valid JSON: quoted key + colon + placeholder string (quoted).
            out.append(f'"{key}": "{MEANINGFUL_RETURN_PLACEHOLDER}"')
        else:
            out.append(text[m.start() : end])
        i = end
    return "".join(out)


def _compress_tool_trace_text(text: str) -> str:
    def _repl(m: re.Match[str]) -> str:
        return m.group(1) + _placeholder_for_preview(m.group(2))

    t = _RESULT_PREVIEW_LINE_RE.sub(_repl, text)
    t = _collapse_large_kwarg_reprs(t)
    t = _collapse_large_json_kw_values(t)
    return t


def compress_vitabench_raw_for_llm(obj: Any) -> Any:
    """Recursively replace ``result_preview`` fields and matching lines inside strings."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "result_preview":
                if v is None:
                    out[k] = NO_RETURN_PLACEHOLDER
                elif isinstance(v, str):
                    out[k] = _placeholder_for_preview(v)
                elif isinstance(v, (dict, list)) and not v:
                    out[k] = NO_RETURN_PLACEHOLDER
                else:
                    out[k] = MEANINGFUL_RETURN_PLACEHOLDER
            else:
                out[k] = compress_vitabench_raw_for_llm(v)
        return out
    if isinstance(obj, list):
        return [compress_vitabench_raw_for_llm(x) for x in obj]
    if isinstance(obj, str):
        return _compress_tool_trace_text(obj)
    return obj


def prepare_trajectory_raw_for_llm(bench_backend: str, raw_result: Any) -> Any:
    """VitaBench-only trajectory shrinking for LLM payloads; other backends unchanged."""
    bb = (bench_backend or "").strip().lower()
    if bb != "vitabench":
        return raw_result
    return compress_vitabench_raw_for_llm(raw_result)
