"""Summarize simulation results for skill optimization."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .config import (
    OPTIMIZER_INTERNAL_TRACE_ENTRY_CHARS,
    OPTIMIZER_INTERNAL_TRACE_MAX_ITEMS,
    OPTIMIZER_ROUTER_PREVIEW_LONG,
    OPTIMIZER_ROUTER_PREVIEW_SHORT,
    OPTIMIZER_RUBRIC_LINE_MAX,
    OPTIMIZER_WORST_TASKS_FULL_ROUTER,
    OPTIMIZER_WORST_TASK_IDS_COUNT,
)


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return str(x)


def _compact_rubrics(ri: dict[str, Any]) -> list[dict[str, Any]] | None:
    nlr = ri.get("nl_rubrics")
    if not isinstance(nlr, list) or not nlr:
        return None
    out: list[dict[str, Any]] = []
    for item in nlr:
        if not isinstance(item, dict):
            continue
        rub = item.get("nl_rubric") or ""
        rub_s = str(rub)[:OPTIMIZER_RUBRIC_LINE_MAX]
        jus = item.get("justification") or ""
        out.append(
            {
                "met": item.get("met"),
                "nl_rubric": rub_s,
                "justification": str(jus)[:OPTIMIZER_RUBRIC_LINE_MAX],
            }
        )
    return out or None


def _summarize_internal_trace(traces: Any) -> dict[str, Any] | None:
    if not isinstance(traces, list) or not traces:
        return None
    names: list[str] = []
    for t in traces:
        if isinstance(t, dict):
            n = t.get("name") or t.get("tool") or t.get("function")
            if n:
                names.append(str(n))
        elif isinstance(t, str):
            names.append(t[:32])
    cnt = Counter(names)
    tail = traces[-OPTIMIZER_INTERNAL_TRACE_MAX_ITEMS :]
    tail_s: list[str] = []
    for t in tail:
        s = json.dumps(t, ensure_ascii=False, default=str) if not isinstance(t, str) else t
        tail_s.append(s[:OPTIMIZER_INTERNAL_TRACE_ENTRY_CHARS])
    return {
        "total_entries": len(traces),
        "tool_name_counts": dict(cnt.most_common(16)),
        "tail_preview": tail_s,
    }


def summarize_run(results_path: Path) -> dict[str, Any]:
    data = load_results(results_path)
    sims = data.get("simulations") or []
    rows: list[dict[str, Any]] = []
    rewards: list[float] = []

    for sim in sims:
        rid = sim.get("task_id")
        ri = sim.get("reward_info") or {}
        if not isinstance(ri, dict):
            ri = {}
        reward = ri.get("reward")
        rv = float(reward) if reward is not None else 0.0
        rewards.append(rv)

        sm: dict[str, Any] = {}
        msgs = sim.get("messages") or []
        for m in reversed(msgs):
            if not isinstance(m, dict):
                continue
            if m.get("role") != "assistant":
                continue
            raw = m.get("raw_data")
            if isinstance(raw, dict):
                sm = raw.get("skill_mas") or {}
                if sm:
                    break

        rr_full = sm.get("router_reasoning") or ""
        term = sim.get("termination_reason")
        if term is not None and not isinstance(term, str):
            term = str(term)

        rb = ri.get("reward_breakdown")
        rb_safe = _json_safe(rb) if rb is not None else None

        rows.append(
            {
                "task_id": rid,
                "reward": reward,
                "termination_reason": term,
                "reward_rubrics_compact": _compact_rubrics(ri),
                "reward_breakdown": rb_safe,
                "selected_skill": sm.get("selected_skill"),
                "router_reasoning": "",
                "num_stages": sm.get("num_stages"),
                "used_tools": sm.get("used_tools"),
                "internal_tool_trace_summary": _summarize_internal_trace(sm.get("internal_tool_trace")),
            }
        )
        rows[-1]["_router_full"] = rr_full

    n = len(rows)
    med = float(statistics.median(rewards)) if rewards else 0.0
    worst_ids: list[Any] = []
    sorted_idx: list[int] = []
    if rows:
        sorted_idx = sorted(
            range(n),
            key=lambda i: (float(rows[i].get("reward") or 0), str(rows[i].get("task_id"))),
        )
        worst_ids = [rows[i].get("task_id") for i in sorted_idx[:OPTIMIZER_WORST_TASK_IDS_COUNT]]

    bottom_long = set(sorted_idx[:OPTIMIZER_WORST_TASKS_FULL_ROUTER]) if sorted_idx else set()
    for i, row in enumerate(rows):
        rv = float(row.get("reward") or 0)
        rr_full = str(row.pop("_router_full", "") or "")
        long_preview = i in bottom_long or rv <= med
        cap = OPTIMIZER_ROUTER_PREVIEW_LONG if long_preview else OPTIMIZER_ROUTER_PREVIEW_SHORT
        row["router_reasoning"] = rr_full[:cap]

    avg = sum(rewards) / n if rewards else 0.0
    aggregate: dict[str, Any] = {
        "num_tasks": n,
        "min_reward": min(rewards) if rewards else None,
        "max_reward": max(rewards) if rewards else None,
        "median_reward": med if rewards else None,
    }
    if n > 1:
        aggregate["reward_std"] = float(statistics.pstdev(rewards))
    else:
        aggregate["reward_std"] = 0.0
    aggregate["worst_task_ids"] = worst_ids

    breakdown_acc: dict[str, list[float]] = {}
    for row in rows:
        rb = row.get("reward_breakdown")
        if not isinstance(rb, dict):
            continue
        for k, v in rb.items():
            try:
                breakdown_acc.setdefault(str(k), []).append(float(v))
            except (TypeError, ValueError):
                continue
    reward_breakdown_avg = {
        k: (sum(vs) / len(vs) if vs else 0.0) for k, vs in sorted(breakdown_acc.items())
    }

    pass_k = {"1": avg}
    pass_at_n = {"1": avg}
    average_at_n = {"1": avg}

    return {
        "avg_reward": avg,
        "aggregate": aggregate,
        "agent_metrics": {
            "average_reward": avg,
            "reward_breakdown": reward_breakdown_avg,
            "pass_hat_k": pass_k,
            "pass_at_n": pass_at_n,
            "average_at_n": average_at_n,
        },
        "per_task": rows,
        "raw_path": str(results_path),
    }


def parse_race_result_txt(path: Path) -> dict[str, float]:
    """Parse ``race_result.txt`` key-value lines into dimension → score."""
    out: dict[str, float] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, _, rest = line.partition(":")
        key = k.strip()
        try:
            out[key] = float(rest.strip())
        except ValueError:
            continue
    return out


def summarize_drb_round(bundle_path: Path) -> dict[str, Any]:
    """Build optimizer-shaped summary from a DRB round bundle JSON + RACE + process traces."""
    bundle = load_results(bundle_path)
    race_txt = Path(bundle.get("race_result_txt") or "")
    race_scores = parse_race_result_txt(race_txt)
    overall = 0.0
    for _k in ("Overall Score", "overall", "Overall"):
        if _k in race_scores:
            overall = float(race_scores[_k])
            break

    ptd = Path(bundle.get("process_trace_dir") or "")
    task_ids: list[int] = list(bundle.get("task_ids") or [])
    rows: list[dict[str, Any]] = []
    for tid in task_ids:
        tp = ptd / f"{tid}.json"
        if not tp.is_file():
            rows.append(
                {
                    "task_id": tid,
                    "reward": None,
                    "termination_reason": None,
                    "reward_rubrics_compact": None,
                    "reward_breakdown": None,
                    "selected_skill": None,
                    "router_reasoning": "",
                    "num_stages": None,
                    "used_tools": None,
                    "internal_tool_trace_summary": None,
                    "selected_skills": None,
                    "workflow_stages": None,
                    "usage_totals": None,
                    "final_article_preview": "",
                    "gate_failed_final": None,
                }
            )
            continue
        tr = json.loads(tp.read_text(encoding="utf-8"))
        wf = tr.get("workflow_stages")
        if isinstance(wf, list):
            num_stages = len(wf)
        else:
            num_stages = None
        rows.append(
            {
                "task_id": tid,
                "reward": None,
                "termination_reason": "gate_failed" if tr.get("gate_failed_final") else None,
                "reward_rubrics_compact": None,
                "reward_breakdown": None,
                "selected_skill": (tr.get("selected_skills") or [None])[0]
                if isinstance(tr.get("selected_skills"), list) and tr.get("selected_skills")
                else None,
                "router_reasoning": str(tr.get("router_reasoning") or "")[
                    : OPTIMIZER_ROUTER_PREVIEW_LONG
                ],
                "num_stages": num_stages,
                "used_tools": None,
                "internal_tool_trace_summary": _summarize_internal_trace(tr.get("steps")),
                "selected_skills": tr.get("selected_skills"),
                "workflow_stages": wf,
                "usage_totals": tr.get("usage_totals"),
                "final_article_preview": tr.get("final_article_preview"),
                "gate_failed_final": tr.get("gate_failed_final"),
            }
        )

    n = len(rows)
    med = overall
    worst_ids: list[Any] = []
    sorted_idx: list[int] = []
    if rows:
        sorted_idx = sorted(
            range(n),
            key=lambda i: (
                1 if rows[i].get("gate_failed_final") else 0,
                -float(rows[i].get("reward") or 0),
                str(rows[i].get("task_id")),
            ),
            reverse=True,
        )
        worst_ids = [rows[i].get("task_id") for i in sorted_idx[:OPTIMIZER_WORST_TASK_IDS_COUNT]]

    aggregate: dict[str, Any] = {
        "num_tasks": n,
        "min_reward": overall if n else None,
        "max_reward": overall if n else None,
        "median_reward": med if n else None,
        "reward_std": 0.0,
        "worst_task_ids": worst_ids,
        "race_dimensions": {k: v for k, v in race_scores.items() if k != "Overall Score"},
        "race_result_path": str(race_txt.resolve()) if race_txt.is_file() else str(race_txt),
        "bench_backend": "drb",
    }

    pass_k = {"1": overall}
    pass_at_n = {"1": overall}
    average_at_n = {"1": overall}

    return {
        "avg_reward": overall,
        "aggregate": aggregate,
        "agent_metrics": {
            "average_reward": overall,
            "reward_breakdown": {},
            "pass_hat_k": pass_k,
            "pass_at_n": pass_at_n,
            "average_at_n": average_at_n,
        },
        "per_task": rows,
        "raw_path": str(bundle_path.resolve()),
        "bench_backend": "drb",
    }
