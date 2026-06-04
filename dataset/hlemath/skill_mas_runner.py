"""HLEMATH Skill-MAS runner backed by Skill_MAS build pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from Skill_MAS.skill_mas.build import (
    MASRunWithRetryResult,
    ThreeStageBuildArtifacts,
    make_planner_call_fn,
    run_mas_pipeline_with_retries,
    make_text_call_fn,
    merge_usage_totals,
)
from Skill_MAS.skill_mas.process_trace_layout import sample_trace_json_dir, skill_mas_sample_log_subdir


@dataclass
class HleMathTask:
    id: int
    prompt: str
    language: str = "en"


def _preview(text: str, max_chars: int = 1200) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "...(truncated)"


def _extract_boxed_answer(text: str) -> str:
    s = (text or "").strip()
    m = re.search(r"\\boxed\{([^}]+)\}", s)
    if m:
        return m.group(1).strip()
    return ""


def _collect_usage_totals(state: dict[str, Any]) -> dict[str, Any]:
    usage_totals: dict[str, Any] = {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    usage_items: list[tuple[str, dict[str, Any]]] = []
    for k, v in (state or {}).items():
        if str(k).startswith("usage_") and isinstance(v, dict):
            usage_items.append((str(k), v))
    final_usage = next((v for k, v in usage_items if k == "usage_final"), None)
    final_is_alias = False
    if isinstance(final_usage, dict):
        for k, v in usage_items:
            if k != "usage_final" and v == final_usage:
                final_is_alias = True
                break
    for k, v in usage_items:
        if k == "usage_final" and final_is_alias:
            continue
        merge_usage_totals(usage_totals, v)
    return usage_totals


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return repr(value)


def _extract_sub_agent_logs(state: dict[str, Any]) -> list[dict[str, Any]]:
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


def _stage_traces_to_jsonable(artifacts: ThreeStageBuildArtifacts) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in artifacts.stage_traces:
        out.append(
            {
                "stage": s.stage,
                "stage_name": s.stage_name,
                "elapsed_sec": round(float(s.elapsed_sec), 6),
                "prompt": s.prompt,
                "raw_response": s.raw_response,
                "parsed_json": _to_jsonable(s.parsed_json),
            }
        )
    return out


def _empty_artifacts() -> ThreeStageBuildArtifacts:
    return ThreeStageBuildArtifacts(
        mas_code="",
        stage_traces=[],
        stage1={},
        stage2={},
        stage3={},
        normalized_sub_agents=[],
    )


def _print_runtime_trace(
    *,
    task_id: int,
    artifacts: ThreeStageBuildArtifacts,
    state: dict[str, Any],
    final_output: str,
    usage_acc: dict[str, Any],
    sub_agent_logs: list[dict[str, Any]],
) -> None:
    print(f"\n[hlemath-skill-mas] ===== task={task_id} build trace =====", flush=True)
    for s in artifacts.stage_traces:
        print(
            f"[hlemath-skill-mas] task={task_id} {s.stage_name} elapsed={s.elapsed_sec:.2f}s",
            flush=True,
        )
        print(f"[hlemath-skill-mas] {s.stage_name} prompt:\n{s.prompt}\n", flush=True)
        print(f"[hlemath-skill-mas] {s.stage_name} raw_response:\n{s.raw_response}\n", flush=True)
        print(
            "[hlemath-skill-mas] "
            f"{s.stage_name} parsed_json:\n{json.dumps(_to_jsonable(s.parsed_json), ensure_ascii=False, indent=2)}\n",
            flush=True,
        )

    print(f"[hlemath-skill-mas] ===== task={task_id} generated MAS code =====", flush=True)
    print(artifacts.mas_code, flush=True)
    print(f"[hlemath-skill-mas] ===== task={task_id} workflow outputs =====", flush=True)
    for item in sub_agent_logs:
        print(f"[hlemath-skill-mas] sub-agent={item['agent_name']} output:", flush=True)
        print(str(item.get("output", "")), flush=True)
        print(
            "[hlemath-skill-mas] "
            f"sub-agent={item['agent_name']} usage={json.dumps(_to_jsonable(item.get('usage', {})), ensure_ascii=False)}",
            flush=True,
        )
    print(
        "[hlemath-skill-mas] "
        f"workflow_state={json.dumps(_to_jsonable(state), ensure_ascii=False, indent=2)}",
        flush=True,
    )
    print(f"[hlemath-skill-mas] final_output:\n{final_output}", flush=True)
    print(
        "[hlemath-skill-mas] "
        f"usage_totals={json.dumps(_to_jsonable(usage_acc), ensure_ascii=False)}",
        flush=True,
    )


async def run_hlemath_skill_mas_on_task(
    task: HleMathTask,
    *,
    init_skill_path: str | Path,
    client: Any,
    process_trace_dir: Path | None = None,
    trace_stem: str | None = None,
    quiet: bool = False,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    init_skill_path = Path(init_skill_path).resolve()
    class_name = f"HLEMathTask{task.id}_MASWorkflow"
    # Build-process robustness for HLEMath:
    # request structured JSON outputs for Stage-1/2/3 planning while keeping a
    # backend-compat fallback inside make_planner_call_fn.
    planner_call_fn = make_planner_call_fn(
        client,
        generation_kwargs={
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        },
    )
    text_call_fn = make_text_call_fn(client)

    run_result: MASRunWithRetryResult = await run_mas_pipeline_with_retries(
        task_text=task.prompt,
        class_name=class_name,
        init_skill_path=init_skill_path,
        planner_call_fn=planner_call_fn,
        text_call_fn=text_call_fn,
        dataset_name="hlemath",
        max_generation_attempts=5,
        max_execution_attempts=3,
        tool_call_fn=None,
    )
    artifacts = run_result.artifacts or _empty_artifacts()
    mas_code = run_result.mas_code or artifacts.mas_code
    state = run_result.state or {}
    final_output = str(run_result.final_output or "").strip()
    usage_acc = _collect_usage_totals(state)
    sub_agent_logs = _extract_sub_agent_logs(state)
    now_iso = datetime.now(timezone.utc).isoformat()

    trace_payload = {
        "schema_version": "hlemath_skill_mas_trace/3",
        "generated_at_utc": now_iso,
        "task_id": task.id,
        "prompt": task.prompt,
        "init_skill_path": str(init_skill_path),
        "class_name": class_name,
        "success": bool(run_result.success),
        "failure_stage": run_result.failure_stage,
        "failure_reason": run_result.failure_reason,
        "generation_attempts_used": run_result.generation_attempts_used,
        "execution_attempts_used": run_result.execution_attempts_used,
        "retry_events": _to_jsonable(run_result.retry_events),
        "usage_totals": usage_acc,
        "mas_code_chars": len(mas_code),
        "build_stage_traces": _stage_traces_to_jsonable(artifacts),
        "normalized_sub_agents": _to_jsonable(artifacts.normalized_sub_agents),
        "state_keys": sorted(list((state or {}).keys())),
        "workflow_state": _to_jsonable(state),
        "sub_agents": sub_agent_logs,
        "final_output": final_output,
        "extracted_boxed_answer": _extract_boxed_answer(final_output),
    }
    if process_trace_dir is not None:
        process_trace_dir.mkdir(parents=True, exist_ok=True)
        stem = trace_stem or skill_mas_sample_log_subdir(task.id, task.id)
        sample_dir = process_trace_dir / "sample_logs" / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        trace_json_root = sample_trace_json_dir(process_trace_dir)
        trace_json_root.mkdir(parents=True, exist_ok=True)
        (sample_dir / "forward_code.py").write_text(mas_code, encoding="utf-8")
        (sample_dir / "workflow_state.json").write_text(
            json.dumps(_to_jsonable(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (sample_dir / "build_stage_traces.json").write_text(
            json.dumps(_stage_traces_to_jsonable(artifacts), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (sample_dir / "sub_agent_outputs.json").write_text(
            json.dumps(sub_agent_logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (trace_json_root / f"{stem}.json").write_text(
            json.dumps(trace_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if not quiet:
        _print_runtime_trace(
            task_id=task.id,
            artifacts=artifacts,
            state=state,
            final_output=final_output,
            usage_acc=usage_acc,
            sub_agent_logs=sub_agent_logs,
        )
        if not run_result.success:
            print(
                f"[hlemath-skill-mas] task={task.id} FAILED stage={run_result.failure_stage} "
                f"reason={run_result.failure_reason}",
                flush=True,
            )

    return final_output, usage_acc, trace_payload
