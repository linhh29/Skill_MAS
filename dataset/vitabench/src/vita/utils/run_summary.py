"""
Write a compact summary JSON next to the main simulation file (same directory),
similar in spirit to Skill_MAS artifacts vitabench summary_*.json.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger

from vita.data_model.simulation import Results, RunConfig, SimulationRun
from vita.metrics.agent_metrics import AgentMetrics
from vita.skill_mas_paths import (
    default_skill_mas_init_skill_path,
    default_skill_mas_workspace_dir,
    is_skill_mas_style_agent,
    skill_mas_results_model_label,
)
from vita.utils.utils import maskill_skip_vitabench_trace_export


def _extract_skill_mas_summary(sim: SimulationRun) -> Optional[dict[str, Any]]:
    """Pull Skill-MAS fields from the last assistant message that carries them."""
    for msg in reversed(sim.messages):
        raw = getattr(msg, "raw_data", None)
        if not raw or not isinstance(raw, dict):
            continue
        sm = raw.get("skill_mas")
        if not sm or not isinstance(sm, dict):
            continue
        return {
            "schema_version": sm.get("schema_version"),
            "routing_mode": sm.get("routing_mode"),
            "skill_dir": sm.get("skill_dir"),
            "used_tools": sm.get("used_tools"),
            "init_skill_path": sm.get("init_skill_path"),
            "class_name": sm.get("class_name"),
            "success": sm.get("success"),
            "failure_stage": sm.get("failure_stage"),
            "failure_reason": sm.get("failure_reason"),
            "generation_attempts_used": sm.get("generation_attempts_used"),
            "execution_attempts_used": sm.get("execution_attempts_used"),
            "workflow_state_keys": sm.get("workflow_state_keys"),
        }
    return None


def _compact_rubrics(sim: SimulationRun) -> list[dict[str, Any]]:
    ri = sim.reward_info
    if not ri or not ri.nl_rubrics:
        return []
    out = []
    for x in ri.nl_rubrics:
        out.append(
            {
                "met": x.met,
                "nl_rubric": x.nl_rubric,
                "justification": x.justification,
            }
        )
    return out


def build_run_summary_dict(
    config: RunConfig,
    results: Results,
    metrics: AgentMetrics,
    simulation_basename: str,
) -> dict[str, Any]:
    """Structured summary for one completed benchmark run."""
    df = results.to_df()
    if df.empty or "reward" not in df.columns:
        task_means = pd.Series(dtype=float)
        num_tasks = 0
    else:
        task_means = df.groupby("task_id")["reward"].mean()
        num_tasks = int(df["task_id"].nunique())

    if len(task_means) > 0:
        min_reward = float(task_means.min())
        max_reward = float(task_means.max())
        med = float(task_means.median())
        std = float(task_means.std()) if len(task_means) > 1 else 0.0
        worst = [str(x) for x in task_means.nsmallest(min(5, len(task_means))).index.tolist()]
    else:
        min_reward = max_reward = med = std = 0.0
        worst = []

    agent_metrics: dict[str, Any] = {
        "average_reward": metrics.avg_reward,
        "reward_breakdown": dict(metrics.avg_reward_breakdown or {}),
        "pass_hat_k": {str(k): v for k, v in metrics.pass_hat_ks.items()},
        "pass_at_n": {str(k): v for k, v in (metrics.pass_at_n or {}).items()},
        "average_at_n": {str(k): v for k, v in (metrics.average_at_n or {}).items()},
        "avg_agent_cost": metrics.avg_agent_cost,
        "total_agent_cost": metrics.total_agent_cost,
        "avg_user_cost": metrics.avg_user_cost,
        "total_user_cost": metrics.total_user_cost,
        "total_duration_sec": metrics.total_duration,
    }
    if metrics.all_types_metrics:
        agent_metrics["all_types_metrics"] = metrics.all_types_metrics

    per_task: list[dict[str, Any]] = []
    for sim in results.simulations:
        ri = sim.reward_info
        reward = float(ri.reward) if ri else 0.0
        row: dict[str, Any] = {
            "task_id": sim.task_id,
            "trial": sim.trial,
            "reward": reward,
            "termination_reason": str(sim.termination_reason),
            "reward_rubrics_compact": _compact_rubrics(sim),
            "reward_breakdown": dict(ri.reward_breakdown) if ri and ri.reward_breakdown else {},
            "duration": sim.duration,
            "agent_cost": sim.agent_cost,
            "user_cost": sim.user_cost,
        }
        sm = _extract_skill_mas_summary(sim)
        if sm:
            row.update(sm)
        per_task.append(row)

    effective_skill_dir = config.skill_mas_dir or (
        str(default_skill_mas_workspace_dir()) if is_skill_mas_style_agent(config.agent) else None
    )

    skill_mas_eval = None
    if is_skill_mas_style_agent(config.agent) and effective_skill_dir:
        first_sm = next(
            (
                x for x in per_task
                if isinstance(x.get("routing_mode"), str)
            ),
            {},
        )
        skill_mas_eval = {
            "schema_version": "vita_run_summary_skill_mas/1",
            "skill_mas_dir": effective_skill_dir,
            "init_skill_md": str(default_skill_mas_init_skill_path()),
            "routing_mode": first_sm.get("routing_mode"),
        }
        if config.planner_llm:
            skill_mas_eval["planner_llm"] = config.planner_llm

    return {
        "schema_version": "vita_run_summary/2",
        "simulation_file": simulation_basename,
        "timestamp": results.timestamp,
        "run": {
            "domain": config.domain,
            "agent": config.agent,
            "agent_llm": config.llm_agent,
            "user": config.user,
            "user_llm": config.llm_user,
            "evaluator_llm": config.llm_evaluator,
            "evaluation_type": config.evaluation_type,
            "skill_mas_dir": effective_skill_dir,
            "num_tasks_requested": config.num_tasks,
            "task_ids": config.task_ids,
        },
        "skill_mas_eval": skill_mas_eval,
        "avg_reward": metrics.avg_reward,
        "aggregate": {
            "num_tasks": num_tasks,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "median_reward": med,
            "reward_std": std,
            "worst_task_ids": worst,
        },
        "agent_metrics": agent_metrics,
        "per_task": per_task,
    }


def write_run_summary(
    save_path: Path,
    config: RunConfig,
    results: Results,
    metrics: AgentMetrics,
) -> Optional[Path]:
    """
    Write ``<stem>_summary.json`` beside the main simulation JSON (same directory).
    """
    save_path = Path(save_path)
    out = save_path.parent / f"{save_path.stem}_summary.json"
    try:
        payload = build_run_summary_dict(
            config,
            results,
            metrics,
            simulation_basename=save_path.name,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"[run_summary] wrote {out}")
        return out
    except Exception as e:
        logger.warning(f"[run_summary] failed to write summary: {e!r}")
        return None


def _sanitize_component(name: str) -> str:
    import re
    s = (name or "").split("/")[-1]
    out = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return out or "unknown"


def _skill_suffix(skill_dir: str | None) -> str:
    default = "default"
    path = str(Path(skill_dir or str(default_skill_mas_workspace_dir())).resolve())
    mark = "/skills/"
    if mark in path:
        rel = path.split(mark, 1)[1].strip("/")
        return rel if rel else default
    return _sanitize_component(Path(path).name)


def _default_skill_mas_trace_out_dir(config: RunConfig) -> Path:
    """Directory under ``results/<model>/skill_mas_process_traces/<skill_suffix>/`` (same as shell manifest)."""
    src_root = Path(__file__).resolve().parents[3]
    model_safe = _sanitize_component(
        skill_mas_results_model_label(
            agent=config.agent,
            llm_agent=config.llm_agent,
            planner_llm=config.planner_llm,
        )
    )
    suffix = _skill_suffix(config.skill_mas_dir)
    if config.agent == "preload_agent":
        suffix = f"preload_agent/{suffix}"
    return src_root / "results" / model_safe / "skill_mas_process_traces" / suffix


def _resolve_skill_mas_trace_out_dir(config: RunConfig) -> Path:
    raw = config.skill_mas_trace_out_dir
    if raw is not None and str(raw).strip():
        return Path(str(raw)).expanduser().resolve()
    return _default_skill_mas_trace_out_dir(config)


def _skill_mas_emit_disk_artifacts(config: RunConfig) -> bool:
    """``sample_logs`` / shell manifest: honor explicit trace dir; else respect MASKILL_SKIP…."""
    raw = config.skill_mas_trace_out_dir
    if raw is not None and str(raw).strip():
        return True
    return not maskill_skip_vitabench_trace_export()


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return repr(value)


def _extract_sub_agent_logs(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Match deep_research_bench / hlemath: one entry per ``out_<agent>`` key in workflow state."""
    logs: list[dict[str, Any]] = []
    for key, value in (state or {}).items():
        if not str(key).startswith("out_"):
            continue
        agent_name = str(key)[4:]
        logs.append(
            {
                "agent_name": agent_name,
                "output_key": key,
                "output": str(value or ""),
                "usage_key": f"usage_{agent_name}",
                "usage": _to_jsonable((state or {}).get(f"usage_{agent_name}", {})),
            }
        )
    logs.sort(key=lambda x: x["agent_name"])
    return logs


def _skill_mas_sample_subdir_for_sim(sim: SimulationRun, index: int) -> str:
    """
    Filesystem-safe folder name per simulation (task_id can be long or contain punctuation).
    Prefix index keeps a stable sort order like BrowseComp ``sample_logs/0/``, ``1/``, …
    """
    import hashlib

    tid = str(sim.task_id)
    base = _sanitize_component(tid)
    if len(base) > 100:
        base = hashlib.sha256(tid.encode("utf-8")).hexdigest()[:20]
    tr = sim.trial
    tr_part = f"_t{int(tr)}" if tr is not None else ""
    return f"{index:04d}__{base}{tr_part}"


def _first_user_text(sim: SimulationRun) -> str:
    from vita.data_model.message import UserMessage

    for msg in sim.messages or []:
        if isinstance(msg, UserMessage) and getattr(msg, "content", None):
            return str(msg.content)
    return ""


def write_skill_mas_sample_logs(
    save_path: Path,
    config: RunConfig,
    results: Results,
) -> Optional[Path]:
    """
    Per-sample artifacts under ``sample_logs/<subdir>/`` (forward_code.py, workflow_state.json, …),
    and one trajectory JSON per sample under ``sample_trace_json/<subdir>.json``,
    aligned with BrowseComp-Plus / hlemath / DRB ``skill_mas`` runners.
    """
    if not is_skill_mas_style_agent(config.agent):
        return None
    if not _skill_mas_emit_disk_artifacts(config):
        logger.info("[run_summary] skip skill_mas sample_logs (MASKILL_SKIP_VITABENCH_TRACE_EXPORT)")
        return None
    try:
        out_dir = _resolve_skill_mas_trace_out_dir(config)
        sample_root = out_dir / "sample_logs"
        sample_root.mkdir(parents=True, exist_ok=True)
        trace_json_root = out_dir / "sample_trace_json"
        trace_json_root.mkdir(parents=True, exist_ok=True)

        for idx, sim in enumerate(results.simulations):
            full_sm: dict[str, Any] = {}
            for msg in reversed(sim.messages or []):
                raw = getattr(msg, "raw_data", None)
                if isinstance(raw, dict) and isinstance(raw.get("skill_mas"), dict):
                    full_sm = raw.get("skill_mas") or {}
                    break
            mas_code = str(full_sm.get("mas_code") or "").strip()
            if not mas_code:
                continue
            sub = _skill_mas_sample_subdir_for_sim(sim, idx)
            sample_dir = sample_root / sub
            sample_dir.mkdir(parents=True, exist_ok=True)
            state = full_sm.get("workflow_state") or {}
            if not isinstance(state, dict):
                state = {}
            sub_agent_logs = _extract_sub_agent_logs(state)
            (sample_dir / "forward_code.py").write_text(mas_code, encoding="utf-8")
            (sample_dir / "workflow_state.json").write_text(
                json.dumps(_to_jsonable(state), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (sample_dir / "build_stage_traces.json").write_text(
                json.dumps(full_sm.get("build_stage_traces") or [], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (sample_dir / "sub_agent_outputs.json").write_text(
                json.dumps(sub_agent_logs, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if full_sm.get("internal_tool_trace"):
                (sample_dir / "vita_internal_tool_trace.json").write_text(
                    json.dumps(_to_jsonable(full_sm.get("internal_tool_trace")), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            trace_name = f"{sub}.json"
            trace_payload = {
                "schema_version": "vitabench_skill_mas_trace/1",
                "task_id": sim.task_id,
                "trial": sim.trial,
                "simulation_id": sim.id,
                "task_text_preview": _first_user_text(sim)[:8000],
                "init_skill_path": full_sm.get("init_skill_path"),
                "class_name": full_sm.get("class_name"),
                "success": full_sm.get("success"),
                "failure_stage": full_sm.get("failure_stage"),
                "failure_reason": full_sm.get("failure_reason"),
                "generation_attempts_used": full_sm.get("generation_attempts_used"),
                "execution_attempts_used": full_sm.get("execution_attempts_used"),
                "retry_events": _to_jsonable(full_sm.get("retry_events")),
                "normalized_sub_agents": full_sm.get("normalized_sub_agents"),
                "workflow_state_keys": full_sm.get("workflow_state_keys"),
                "workflow_state": _to_jsonable(state),
                "sub_agents": sub_agent_logs,
                "final_output": full_sm.get("final_output"),
                "used_tools": full_sm.get("used_tools"),
            }
            (trace_json_root / trace_name).write_text(
                json.dumps(trace_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        logger.info(f"[run_summary] wrote skill_mas sample_logs under {sample_root}")
        return sample_root
    except Exception as e:
        logger.warning(f"[run_summary] failed writing skill_mas sample_logs: {e!r}")
        return None


def write_skill_mas_shell_manifest(
    save_path: Path,
    config: RunConfig,
    results: Results,
    summary_path: Optional[Path] = None,
) -> Optional[Path]:
    if not is_skill_mas_style_agent(config.agent):
        return None
    if not _skill_mas_emit_disk_artifacts(config):
        logger.info("[run_summary] skip skill_mas shell manifest (MASKILL_SKIP_VITABENCH_TRACE_EXPORT)")
        return None
    try:
        out_dir = _resolve_skill_mas_trace_out_dir(config)
        out_dir.mkdir(parents=True, exist_ok=True)

        run_stem = Path(save_path).stem
        out_path = out_dir / f"{run_stem}_shell_manifest.json"

        per_task: list[dict[str, Any]] = []
        for idx, sim in enumerate(results.simulations):
            sm = _extract_skill_mas_summary(sim) or {}
            full_sm: dict[str, Any] = {}
            for msg in reversed(sim.messages):
                raw = getattr(msg, "raw_data", None)
                if isinstance(raw, dict) and isinstance(raw.get("skill_mas"), dict):
                    full_sm = raw.get("skill_mas") or {}
                    break
            mas_code = str(full_sm.get("mas_code") or "").strip()
            sample_subdir = _skill_mas_sample_subdir_for_sim(sim, idx)
            sample_log_dir = (
                str((out_dir / "sample_logs" / sample_subdir).resolve()) if mas_code else None
            )
            per_task.append(
                {
                    "task_id": sim.task_id,
                    "trial": sim.trial,
                    "termination_reason": str(sim.termination_reason),
                    "sample_log_dir": sample_log_dir,
                    "skill_mas": full_sm if full_sm else sm,
                }
            )

        payload = {
            "schema_version": "vitabench_shell_manifest/3",
            "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z"),
            "skill_mas_dir": str(Path(config.skill_mas_dir or str(default_skill_mas_workspace_dir())).resolve()),
            "init_skill_path": str(default_skill_mas_init_skill_path()),
            "cli": {
                "domain": config.domain,
                "agent": config.agent,
                "agent_llm": config.llm_agent,
                "user": config.user,
                "user_llm": config.llm_user,
                "evaluator_llm": config.llm_evaluator,
                "num_tasks": config.num_tasks,
                "max_steps": config.max_steps,
                "max_concurrency": config.max_concurrency,
            },
            "artifacts": {
                "simulation_json": str(Path(save_path).resolve()),
                "summary_json": str(Path(summary_path).resolve()) if summary_path else None,
                "sample_logs_dir": str((out_dir / "sample_logs").resolve()),
                "sample_trace_json_dir": str((out_dir / "sample_trace_json").resolve()),
            },
            "per_task": per_task,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[run_summary] wrote skill_mas shell manifest {out_path}")
        return out_path
    except Exception as e:
        logger.warning(f"[run_summary] failed writing skill_mas shell manifest: {e!r}")
        return None
