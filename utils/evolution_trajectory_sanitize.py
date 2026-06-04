"""Unified trajectory view for Skill-MAS evolution (contrastive / bank LLM payloads).

**Build stages in this view:** each item has ``stage``, ``stage_name``, ``elapsed_sec``, and a
**subset** of ``parsed_json``: Stage 1 full JSON; Stage 2 ``reasoning`` plus slim ``sub_agents``
(``role_name`` / ``tool_context`` only, no long prompts); Stage 3 ``reasoning`` plus a note that
``mas_code`` is omitted here **in favor of** top-level ``mas_code`` (``forward_async`` slice).
``raw_response`` is still omitted. On-disk traces unchanged. ``mas_code`` is extracted from raw
stages before slicing.

**Tool-return placeholders (only where applicable):**
- **VitaBench:** ``compress_vitabench_raw_for_llm`` folds ``result_preview``, huge
  ``name={...}`` / ``name=[...]`` repr blobs (e.g. ``w={...}``), and JSON argument dumps,
  including strings inside ``evaluation`` / ``window_evaluations``.
- **BCP (BrowseComp):** known bulk search keys in ``workflow_state`` (e.g. long
  ``search_results``) may be replaced by short placeholders via
  ``collapse_bulk_tool_like_payloads``.

**DRB (evolution payload only):** same head/tail folding as HLEMath on each ``out_*`` (per sub-agent)
and ``final_output``. Disk ``workflow_state.json`` is unchanged.

**HLEMath (evolution payload only):** in ``workflow_state``, each ``out_*`` and ``final_output``
string is head/tail folded (64 words per side by default, middle ``【Intermediate Reasoning】``).
``task_info`` / ``task_text`` stay full. Disk ``workflow_state.json`` is unchanged.

"""

from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

from .vitabench_llm_compress import (
    MEANINGFUL_RETURN_PLACEHOLDER,
    NO_RETURN_PLACEHOLDER,
    compress_vitabench_raw_for_llm,
    placeholder_for_result_preview,
)

SCHEMA_EVOLUTION_VIEW = "skill_mas_evolution_trajectory_view_v1"

# HLEMath / DRB: shorten per-agent outputs (workflow_state ``out_*``) for contrastive / bank LLMs only.
_EVOLUTION_WS_HEAD_WORDS = 64
_EVOLUTION_WS_TAIL_WORDS = 64
_EVOLUTION_WS_INTERMEDIATE_MARKER = "【Intermediate Reasoning】"


def _truncate_workflow_outputs_head_tail(s: str) -> str:
    """Keep leading/trailing words; fold the middle for each ``out_*`` / ``final_output`` string."""
    if not isinstance(s, str) or not s.strip():
        return s
    h = max(0, int(_EVOLUTION_WS_HEAD_WORDS))
    t = max(0, int(_EVOLUTION_WS_TAIL_WORDS))
    mk = _EVOLUTION_WS_INTERMEDIATE_MARKER
    words = s.split()
    cap = h + t
    if cap <= 0:
        return s
    if len(words) == 1 and len(s) > 6_000:
        side = 900
        return f"{s[:side]}\n{mk}\n{s[-side:]}"
    if len(words) <= cap:
        return s
    n = len(words)
    if h + t > n:
        ha = min(h, max(1, n // 2))
        ta = min(t, max(0, n - ha))
        head = " ".join(words[:ha]) if ha else ""
        tail = " ".join(words[n - ta:]) if ta else ""
    else:
        head = " ".join(words[:h]) if h else ""
        tail = " ".join(words[-t:]) if t else ""
    if tail:
        return f"{head}\n{mk}\n{tail}"
    return f"{head}\n{mk}"


def _mas_code_forward_async_only(mas_code: str | None) -> str | None:
    """Keep codegen from ``async def forward_async`` onward (runtime topology); omit verbose ``__init__``."""
    if not isinstance(mas_code, str) or not mas_code.strip():
        return mas_code
    m = re.search(r"\basync\s+def\s+forward_async\s*\(", mas_code)
    if m is None:
        m = re.search(r"\bdef\s+forward_async\s*\(", mas_code)
    if m is None:
        return mas_code
    line_start = mas_code.rfind("\n", 0, m.start()) + 1
    return mas_code[line_start:]


# Bulk search / retrieval payloads (BrowseComp-style); avoid generic keys like "content".
_BULK_TOOLISH_KEYS = frozenset(
    {
        "result_preview",
        "search_results",
        "search_result",
        "documents",
        "snippets",
        "hits",
        "pages",
        "page_content",
        "raw_html",
        "raw_content",
        "observation",
    }
)


def _collapse_bulk_text(parent_key: str, value: str) -> str:
    """Replace only known bulk search/tool blobs with placeholders — no length-based truncation."""
    lk = parent_key.lower()
    if lk in _BULK_TOOLISH_KEYS and len(value) > 2_000:
        return MEANINGFUL_RETURN_PLACEHOLDER if value.strip() else NO_RETURN_PLACEHOLDER
    return value


def collapse_bulk_tool_like_payloads(obj: Any, *, parent_key: str = "") -> Any:
    """Replace huge search/tool-like strings; recurse dicts/lists."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            ks = str(k)
            if ks == "result_preview":
                out[ks] = placeholder_for_result_preview(v) if isinstance(v, str) else (
                    MEANINGFUL_RETURN_PLACEHOLDER if v not in (None, [], {}) else NO_RETURN_PLACEHOLDER
                )
                continue
            out[ks] = collapse_bulk_tool_like_payloads(v, parent_key=ks)
        return out
    if isinstance(obj, list):
        return [collapse_bulk_tool_like_payloads(x, parent_key=parent_key) for x in obj]
    if isinstance(obj, str):
        return _collapse_bulk_text(parent_key, obj)
    return obj


def _find_task_dict(tasks: list[Any], task_id: str) -> dict[str, Any] | None:
    for t in tasks:
        if isinstance(t, dict) and str(t.get("id")) == str(task_id):
            return t
    return None


def _task_description_vita(task: dict[str, Any]) -> str:
    return str(
        task.get("instructions")
        or task.get("instruction")
        or task.get("task_text")
        or task.get("description")
        or ""
    )


def _slim_assistant_tool_calls(tool_calls: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tool_calls, list):
        return None
    slim: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name", "?")
        args = tc.get("arguments")
        arg_s = json.dumps(args, ensure_ascii=False) if args is not None else ""
        if arg_s:
            arg_s = compress_vitabench_raw_for_llm(arg_s)
        slim.append({"name": name, "arguments": arg_s})
    return slim or None


def _slim_vita_message(m: dict[str, Any]) -> dict[str, Any]:
    role = m.get("role")
    content = m.get("content")
    if isinstance(content, str):
        content = compress_vitabench_raw_for_llm(content)
    row: dict[str, Any] = {"role": role, "content": content}
    tc = _slim_assistant_tool_calls(m.get("tool_calls"))
    if tc:
        row["tool_calls"] = tc
    return row


def _get_raw_build_traces_vita(messages: list[Any]) -> list[Any]:
    """Un-trimmed build_stage_traces from the last assistant message (for mas_code / raw extract)."""
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        raw = m.get("raw_data")
        if not isinstance(raw, dict):
            continue
        sm = raw.get("skill_mas")
        if not isinstance(sm, dict):
            continue
        stages = sm.get("build_stage_traces")
        if isinstance(stages, list) and stages:
            return stages
    return []


def _extract_build_traces_vita(messages: list[Any]) -> list[dict[str, Any]]:
    return _trim_build_stage_traces(_get_raw_build_traces_vita(messages))


def _extract_mas_code_from_stages(stages: list[dict[str, Any]]) -> str | None:
    for st in reversed(stages):
        if not isinstance(st, dict):
            continue
        pj = st.get("parsed_json")
        if isinstance(pj, dict):
            mc = pj.get("mas_code")
            if isinstance(mc, str) and mc.strip():
                return mc
    return None


def _infer_build_stage_number(st: dict[str, Any]) -> int | None:
    s = st.get("stage")
    if isinstance(s, int):
        return s
    if isinstance(s, float):
        return int(s)
    if isinstance(s, str) and s.strip().isdigit():
        return int(s.strip())
    sn = str(st.get("stage_name") or "").strip()
    low = sn.lower()
    if low.startswith("stage "):
        tail = sn[6:].strip()
        if tail.isdigit():
            return int(tail)
    return None


def _evolution_parsed_json_subset(st: dict[str, Any], stage_num: int | None) -> dict[str, Any] | None:
    """Stage 1–2 reasoning/structure for reflection; Stage 3 reasoning only (no mas_code duplicate)."""
    pj = st.get("parsed_json")
    if not isinstance(pj, dict):
        return None

    if stage_num == 3:
        out: dict[str, Any] = {
            "note": "mas_code omitted here; see top-level raw_result.mas_code (forward_async slice).",
        }
        if pj.get("reasoning") is not None:
            out["reasoning"] = pj.get("reasoning")
        return out

    if stage_num == 1:
        return copy.deepcopy(pj)

    if stage_num == 2:
        slim: dict[str, Any] = {}
        if pj.get("reasoning") is not None:
            slim["reasoning"] = pj.get("reasoning")
        sub = pj.get("sub_agents")
        if isinstance(sub, list):
            slim_agents: list[dict[str, Any]] = []
            for sa in sub:
                if isinstance(sa, dict):
                    slim_agents.append(
                        {
                            "role_name": sa.get("role_name"),
                            "tool_context": sa.get("tool_context"),
                        }
                    )
            slim["sub_agents"] = slim_agents
        return slim

    if pj.get("reasoning") is not None:
        return {"reasoning": pj.get("reasoning")}
    return None


def _trim_build_stage_traces(stages: Any) -> list[dict[str, Any]]:
    """Build-stage rows for evolution LLMs: metadata + slim ``parsed_json`` per stage; no ``raw_response``."""
    if not isinstance(stages, list):
        return []
    out: list[dict[str, Any]] = []
    with_pj = 0
    for st in stages:
        if not isinstance(st, dict):
            continue
        snum = _infer_build_stage_number(st)
        entry: dict[str, Any] = {
            "stage": st.get("stage"),
            "stage_name": st.get("stage_name"),
            "elapsed_sec": st.get("elapsed_sec"),
        }
        pj_out = _evolution_parsed_json_subset(st, snum)
        if pj_out is not None:
            entry["parsed_json"] = pj_out
            with_pj += 1
        out.append(entry)
    if with_pj:
        _log.debug(
            "evolution_trajectory_sanitize: build stages include slim parsed_json for %s row(s)",
            with_pj,
        )
    return out


def _sanitize_vitabench(raw: dict[str, Any], task_id: str, record: Any) -> dict[str, Any]:
    tasks = raw.get("tasks") if isinstance(raw.get("tasks"), list) else []
    task_dict = _find_task_dict(tasks, task_id)
    task_description = _task_description_vita(task_dict) if task_dict else ""

    sims = raw.get("simulations")
    scoped: list[Any] = []
    if isinstance(sims, list):
        scoped = [x for x in sims if isinstance(x, dict) and str(x.get("task_id")) == str(task_id)]
    sim = scoped[0] if scoped else None

    build_stage_traces: list[dict[str, Any]] = []
    runtime_messages: list[dict[str, Any]] = []
    evaluation: dict[str, Any] = {}
    mas_code: str | None = None

    if isinstance(sim, dict):
        msgs = sim.get("messages")
        if isinstance(msgs, list):
            raw_stages = _get_raw_build_traces_vita(msgs)
            mas_code = _extract_mas_code_from_stages(raw_stages)
            build_stage_traces = _trim_build_stage_traces(raw_stages)
            runtime_messages = [_slim_vita_message(m) for m in msgs if isinstance(m, dict)]
        evaluation = copy.deepcopy(sim.get("reward_info") or {})
        term = sim.get("termination_reason")
        if term is not None:
            evaluation["termination_reason"] = term if isinstance(term, str) else str(term)
        evaluation = compress_vitabench_raw_for_llm(evaluation)

    return {
        "schema": SCHEMA_EVOLUTION_VIEW,
        "bench_backend": "vitabench",
        "task_id": str(task_id),
        "task_description": task_description,
        "score_record": {"score": float(record.score), "score_source": record.score_source},
        "skill_mas_build_stages": build_stage_traces,
        "mas_code": _mas_code_forward_async_only(mas_code),
        "runtime_messages": runtime_messages,
        "evaluation": evaluation,
    }


def _resolve_bundle_trace_json(bundle: dict[str, Any], task_id: str) -> Path | None:
    ptd = Path(bundle.get("process_trace_dir") or "")
    if not ptd.is_dir():
        return None
    tid = str(task_id).strip()
    candidates: list[Path] = []
    if tid.isdigit():
        candidates.append(ptd / f"{int(tid)}.json")
        candidates.append(ptd / f"{tid}.json")
    else:
        candidates.append(ptd / f"{tid}.json")
    for c in candidates:
        if c.is_file():
            return c
    sub = ptd / "sample_trace_json"
    if sub.is_dir():
        for p in sorted(sub.glob("*.json")):
            base = p.name
            if tid.isdigit() and (f"_{int(tid)}_" in base or base.startswith(f"{int(tid)}__")):
                return p
            if tid in base:
                return p
    return None


def _resolve_workflow_state_path(bundle: dict[str, Any], task_id: str) -> Path | None:
    ptd = Path(bundle.get("process_trace_dir") or "")
    if not ptd.is_dir():
        return None
    logs_root = ptd / "sample_logs"
    if not logs_root.is_dir():
        return None
    for d in sorted(logs_root.iterdir()):
        if not d.is_dir():
            continue
        ws = d / "workflow_state.json"
        if ws.is_file():
            return ws
    return None


def _slim_workflow_state(
    ws: dict[str, Any],
    *,
    fold_bulk_placeholders: bool,
    bench_backend: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in ws.items():
        ks = str(k)
        if ks.startswith("usage_"):
            continue
        if ks.startswith("out_") or ks in ("final_output", "task_info", "task_text"):
            out[ks] = v
    if "final_output" not in out and "final_output" in ws:
        out["final_output"] = ws["final_output"]
    bb = (bench_backend or "").strip().lower()
    if bb in ("hlemath", "drb"):
        slim: dict[str, Any] = {}
        for k, v in out.items():
            ks = str(k)
            if (ks.startswith("out_") or ks == "final_output") and isinstance(v, str):
                slim[k] = _truncate_workflow_outputs_head_tail(v)
            else:
                slim[k] = v
        out = slim
    if fold_bulk_placeholders:
        return collapse_bulk_tool_like_payloads(out)
    return out


def _sanitize_bundle_trace_json(trace: dict[str, Any], record: Any) -> dict[str, Any]:
    prompt = str(trace.get("prompt") or "")
    raw_bt = trace.get("build_stage_traces")
    if not isinstance(raw_bt, list):
        raw_bt = []
    mas_code = (
        trace.get("mas_code")
        if isinstance(trace.get("mas_code"), str)
        else _extract_mas_code_from_stages(raw_bt)
    )
    mas_code = _mas_code_forward_async_only(mas_code)
    stages = _trim_build_stage_traces(raw_bt)
    usage_totals = trace.get("usage_totals")

    return {
        "schema": SCHEMA_EVOLUTION_VIEW,
        "bench_backend": getattr(record, "bench_backend", ""),
        "task_id": str(getattr(record, "task_id", "")),
        "task_description": prompt,
        "score_record": {"score": float(record.score), "score_source": record.score_source},
        "skill_mas_build_stages": stages,
        "mas_code": mas_code,
        "usage_totals": usage_totals if isinstance(usage_totals, dict) else None,
        "runtime_execution": {},
    }


def _sanitize_from_bundle_path(bundle_path: Path, record: Any) -> dict[str, Any] | None:
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(bundle, dict):
        return None

    trace_path = _resolve_bundle_trace_json(bundle, str(getattr(record, "task_id", "")))
    trace_obj: dict[str, Any] | None = None
    if trace_path and trace_path.is_file():
        try:
            trace_obj = json.loads(trace_path.read_text(encoding="utf-8"))
        except Exception:
            trace_obj = None
    if isinstance(trace_obj, dict):
        slim = _sanitize_bundle_trace_json(trace_obj, record)
        bb = (getattr(record, "bench_backend", None) or "").strip().lower()
        fold_bulk = bb == "bcp"
        ws_path = _resolve_workflow_state_path(bundle, str(getattr(record, "task_id", "")))
        if ws_path and ws_path.is_file():
            try:
                ws = json.loads(ws_path.read_text(encoding="utf-8"))
                if isinstance(ws, dict):
                    slim.setdefault("runtime_execution", {})
                    slim["runtime_execution"]["workflow_state"] = _slim_workflow_state(
                        ws, fold_bulk_placeholders=fold_bulk, bench_backend=bb
                    )
            except Exception:
                pass
        slim["bundle_paths"] = {
            "process_trace": str(trace_path) if trace_path else None,
            "workflow_state": str(ws_path) if ws_path else None,
        }
        return slim

    return {
        "schema": SCHEMA_EVOLUTION_VIEW,
        "bench_backend": getattr(record, "bench_backend", ""),
        "task_id": str(getattr(record, "task_id", "")),
        "task_description": "",
        "score_record": {"score": float(record.score), "score_source": record.score_source},
        "skill_mas_build_stages": [],
        "mas_code": None,
        "runtime_execution": {},
        "note": "process trace JSON not found under process_trace_dir; bundle metadata only.",
        "bundle_summary": {k: bundle.get(k) for k in ("race_model_tag", "raw_jsonl", "process_trace_dir") if k in bundle},
    }


def sanitize_raw_result_for_evolution(record: Any, raw_result: Any) -> Any:
    """Return a compact trajectory view for evolution LLMs.

    ``record`` must expose ``bench_backend``, ``task_id``, ``score``, ``score_source``,
    ``raw_result_path`` (for bundle backends).
    """
    bb = (getattr(record, "bench_backend", None) or "").strip().lower()
    tid = str(getattr(record, "task_id", ""))

    if bb == "vitabench" and isinstance(raw_result, dict):
        scoped = raw_result
        sims = scoped.get("simulations")
        if isinstance(sims, list):
            filtered = [x for x in sims if isinstance(x, dict) and str(x.get("task_id")) == tid]
            scoped = dict(scoped)
            scoped["simulations"] = filtered
        return _sanitize_vitabench(scoped, tid, record)

    if bb in ("drb", "hlemath", "bcp"):
        raw_path = Path(str(getattr(record, "raw_result_path", "") or ""))
        if raw_path.is_file():
            hit = _sanitize_from_bundle_path(raw_path, record)
            if hit is not None:
                return hit
        if isinstance(raw_result, dict):
            if bb == "bcp":
                return collapse_bulk_tool_like_payloads(copy.deepcopy(raw_result))
            return copy.deepcopy(raw_result)

    if isinstance(raw_result, (dict, list)):
        if bb == "bcp":
            return collapse_bulk_tool_like_payloads(copy.deepcopy(raw_result))
        return copy.deepcopy(raw_result)
    return raw_result
