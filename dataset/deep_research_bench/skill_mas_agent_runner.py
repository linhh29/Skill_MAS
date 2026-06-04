"""Generate DRB raw_data with a fixed four-phase Skill-MAS agent path."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from Skill_MAS.skill_mas.build import (
    MASRunWithRetryResult,
    ThreeStageBuildArtifacts,
    make_planner_call_fn,
    make_text_call_fn,
    merge_usage_totals,
    run_mas_pipeline_with_retries,
)

try:
    from .drb_runtime import DRBArticle, enrich_usage_with_cost, load_drb_tasks, load_pricing_table, write_drb_articles
    from .evaluate import run_drb_evaluation
    from .run_single_agent import (
        _evaluate_one_sample_async,
        _load_jsonl_rows,
        load_existing_output,
        load_existing_traces,
    )
    from Skill_MAS.skill_mas.openai_async_client import AsyncOpenAIClient
    from Skill_MAS.skill_mas.process_trace_layout import (
        iter_per_sample_trace_json_files,
        sample_trace_json_dir,
        skill_mas_sample_log_subdir,
        skill_mas_trace_suffix_parts,
        skill_workspace_from_init_path,
    )
except ImportError:
    from drb_runtime import DRBArticle, enrich_usage_with_cost, load_drb_tasks, load_pricing_table, write_drb_articles
    from evaluate import run_drb_evaluation
    from run_single_agent import (
        _evaluate_one_sample_async,
        _load_jsonl_rows,
        load_existing_output,
        load_existing_traces,
    )
    from Skill_MAS.skill_mas.openai_async_client import AsyncOpenAIClient
    from Skill_MAS.skill_mas.process_trace_layout import (
        iter_per_sample_trace_json_files,
        sample_trace_json_dir,
        skill_mas_sample_log_subdir,
        skill_mas_trace_suffix_parts,
        skill_workspace_from_init_path,
    )


@dataclass
class DRBTask:
    id: int
    prompt: str
    language: str = "en"


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


async def run_skill_mas_on_task_async(
    task: Any,
    *,
    skill_dir: str | Path,
    client: Any,
    dry_run: bool,
    skill_name: list[str] | None = None,
    event_writer: Any | None = None,
    process_trace_dir: Path | None = None,
    trace_stem: str | None = None,
    io_lock: threading.Lock | None = None,
    quiet: bool = False,
    model_name: Optional[str] = None,
    include_trace_payload: bool = False,
) -> tuple[DRBArticle, dict[str, Any]] | tuple[DRBArticle, dict[str, Any], dict[str, Any]]:
    del skill_name, event_writer, dry_run  # compatibility
    lock = io_lock or threading.Lock()
    skill_dir = Path(skill_dir).resolve()
    init_skill_path = skill_dir if skill_dir.is_file() else (skill_dir / "SKILL.md")
    if not init_skill_path.is_file():
        raise FileNotFoundError(f"Skill_MAS init skill not found: {init_skill_path}")

    task_obj = DRBTask(id=int(task.id), prompt=str(task.prompt), language=str(getattr(task, "language", "en")))
    class_name = f"DRBTask{task_obj.id}_MASWorkflow"
    planner_call_fn = make_planner_call_fn(client)
    text_call_fn = make_text_call_fn(client)

    run_result: MASRunWithRetryResult = await run_mas_pipeline_with_retries(
        task_text=task_obj.prompt,
        class_name=class_name,
        init_skill_path=init_skill_path,
        planner_call_fn=planner_call_fn,
        text_call_fn=text_call_fn,
        dataset_name="drb",
        max_generation_attempts=5,
        max_execution_attempts=3,
        tool_call_fn=None,
    )
    artifacts = run_result.artifacts or _empty_artifacts()
    mas_code = run_result.mas_code or artifacts.mas_code
    state = run_result.state or {}
    final_output = str(run_result.final_output or "").strip()
    resolved_model = (
        str(getattr(client, "model", "") or "").strip()
        or str(getattr(client, "model_name", "") or "").strip()
        or str(model_name or "").strip()
        or "qwen3.5-plus"
    )
    usage_acc = enrich_usage_with_cost(
        _collect_usage_totals(state),
        model=resolved_model,
        table=load_pricing_table(),
    )
    sub_agent_logs = _extract_sub_agent_logs(state)
    now_iso = datetime.now(timezone.utc).isoformat()

    payload = {
        "schema_version": "drb_skill_mas_trace/3",
        "generated_at_utc": now_iso,
        "task_id": task_obj.id,
        "prompt": task_obj.prompt,
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
        "article": final_output,
        "final_output": final_output,
    }
    if process_trace_dir is not None:
        process_trace_dir.mkdir(parents=True, exist_ok=True)
        stem = trace_stem or skill_mas_sample_log_subdir(task_obj.id, task_obj.id)
        sample_dir = process_trace_dir / "sample_logs" / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        trace_json_root = sample_trace_json_dir(process_trace_dir)
        trace_json_root.mkdir(parents=True, exist_ok=True)

        def _write_trace_files_sync() -> None:
            with lock:
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
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        await asyncio.to_thread(_write_trace_files_sync)
    if not quiet and not run_result.success:
        print(
            f"[drb-skill-mas] task={task_obj.id} FAILED stage={run_result.failure_stage} "
            f"reason={run_result.failure_reason}",
            flush=True,
        )
    base = (DRBArticle(id=task_obj.id, prompt=task_obj.prompt, article=final_output), usage_acc)
    if include_trace_payload:
        return base[0], base[1], payload
    return base


def run_skill_mas_on_task(
    task: Any,
    *,
    skill_dir: str | Path,
    client: Any,
    dry_run: bool,
    skill_name: list[str] | None = None,
    event_writer: Any | None = None,
    process_trace_dir: Path | None = None,
    trace_stem: str | None = None,
    io_lock: threading.Lock | None = None,
    quiet: bool = False,
    model_name: Optional[str] = None,
    include_trace_payload: bool = False,
) -> tuple[DRBArticle, dict[str, Any]] | tuple[DRBArticle, dict[str, Any], dict[str, Any]]:
    """Sync entrypoint for CLI/scripts: one-shot ``asyncio.run`` around the async pipeline."""
    return asyncio.run(
        run_skill_mas_on_task_async(
            task,
            skill_dir=skill_dir,
            client=client,
            dry_run=dry_run,
            skill_name=skill_name,
            event_writer=event_writer,
            process_trace_dir=process_trace_dir,
            trace_stem=trace_stem,
            io_lock=io_lock,
            quiet=quiet,
            model_name=model_name,
            include_trace_payload=include_trace_payload,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill-MAS DRB multi-stage generator.")
    parser.add_argument("--query_file", default="data/drb_test.jsonl")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--model_name", default=os.environ.get("SKILL_MAS_AGENT_MODEL_NAME", "gpt-4o"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=1,
        help="Max concurrent Skill-MAS tasks (single asyncio loop + semaphore).",
    )
    parser.add_argument("--provider", choices=["auto", "openai"], default="auto")
    from Skill_MAS.utils.paths import INIT_SKILL_DIR

    parser.add_argument("--skill_dir", default=str(INIT_SKILL_DIR))
    parser.add_argument("--process_trace_dir", default=None)
    parser.add_argument("--skip_race", action="store_true", help="Skip RACE evaluation after generation.")
    parser.add_argument("--race_model_tag", default=None, help="Optional model tag used by deepresearch_bench_race.")
    parser.add_argument("--race_output_dir", default=None, help="Directory for RACE outputs (default: <repo>/results).")
    parser.add_argument("--race_max_workers", type=int, default=10, help="Max workers for RACE scoring.")
    parser.add_argument("--cost_output_file", default=None)
    parser.add_argument(
        "--eval_per_sample",
        action="store_true",
        help="After async Skill-MAS generation, run RACE judge in the same asyncio.run as generation (--judge_max_concurrent).",
    )
    parser.add_argument(
        "--judge_max_concurrent",
        type=int,
        default=8,
        help="Max concurrent judge LLM calls within the single judge event loop.",
    )
    args = parser.parse_args()

    query_path = Path(args.query_file).resolve()
    tasks = load_drb_tasks(query_path)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    bench_dir = Path(__file__).resolve().parent
    output_path = Path(args.output_file).resolve() if args.output_file else None
    if output_path is None and args.skip_race:
        raise ValueError("--output_file is required when --skip_race is enabled.")
    skill_dir_resolved = str(Path(args.skill_dir).resolve())
    skill_workspace = skill_workspace_from_init_path(Path(args.skill_dir))
    skill_trace_suffix_parts = skill_mas_trace_suffix_parts(skill_workspace)
    skill_trace_suffix_joined = "/".join(skill_trace_suffix_parts)
    process_trace_dir = (
        Path(args.process_trace_dir).resolve()
        if args.process_trace_dir
        else (bench_dir / "results" / args.model_name / "skill_mas_process_traces").joinpath(
            *skill_trace_suffix_parts
        ).resolve()
    )
    process_trace_dir.mkdir(parents=True, exist_ok=True)
    task_rows = _load_jsonl_rows(query_path)
    rubric_by_prompt = {str(r.get("prompt")): r.get("rubric", {}) for r in task_rows if r.get("prompt")}
    reference_by_prompt = {
        str(r.get("prompt")): str(r.get("reference_article", ""))
        for r in task_rows
        if r.get("prompt")
    }
    judge_model = os.environ.get("DRB_RACE_MODEL", args.model_name)
    judge_client = AsyncOpenAIClient(model=judge_model) if args.eval_per_sample else None
    pending_judges: list[tuple[Any, str, dict[str, Any], str]] = []
    trace_existing, trace_totals = ({}, {"prompt_tokens": 0.0, "output_tokens": 0.0, "total_tokens": 0.0, "estimated_cost_usd": 0.0})
    if not args.overwrite:
        trace_existing, trace_totals = load_existing_traces(
            process_trace_dir,
            require_judged=bool(args.eval_per_sample),
        )
    file_existing = {} if args.overwrite or output_path is None else load_existing_output(output_path)
    existing = dict(trace_existing)
    existing.update(file_existing)
    max_concurrent = max(1, int(args.max_concurrent))
    io_lock = threading.Lock()
    print(
        f"[drb-skill-mas] init model={args.model_name} tasks_total={len(tasks)} "
        f"max_concurrent={max_concurrent} trace_dir={process_trace_dir}",
        flush=True,
    )
    if args.process_trace_dir:
        print(f"[drb-skill-mas] skill_dir={skill_dir_resolved}", flush=True)
    else:
        print(
            f"[drb-skill-mas] skill_dir={skill_dir_resolved} skill_trace_suffix={skill_trace_suffix_joined}",
            flush=True,
        )
    if not args.overwrite and existing:
        print(f"[drb-skill-mas] resume detected completed={len(existing)} from existing outputs/traces", flush=True)

    def _write_trace_payload(trace_stem: str, payload: dict[str, Any]) -> None:
        with io_lock:
            tdir = sample_trace_json_dir(process_trace_dir)
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / f"{trace_stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def _write_trace_generated(
        *,
        trace_stem: str,
        base_payload: dict[str, Any],
        status: str = "generated",
        race_eval: dict[str, Any] | None = None,
    ) -> None:
        payload = dict(base_payload)
        payload["status"] = status
        payload["model"] = args.model_name
        payload["query_file"] = str(query_path)
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        if race_eval is not None:
            payload["race_eval"] = race_eval
        _write_trace_payload(trace_stem, payload)

    task_order_index = {t.id: i for i, t in enumerate(tasks)}

    generated: Dict[int, DRBArticle] = dict(existing)
    total_prompt_tokens = int(trace_totals["prompt_tokens"])
    total_output_tokens = int(trace_totals["output_tokens"])
    total_tokens = int(trace_totals["total_tokens"])
    total_estimated_cost_usd = float(trace_totals["estimated_cost_usd"])
    pending_tasks = [task for task in tasks if task.id not in generated]
    print(f"[drb-skill-mas] pending={len(pending_tasks)}", flush=True)

    async def _async_generate_and_judge() -> None:
        nonlocal total_prompt_tokens, total_output_tokens, total_tokens, total_estimated_cost_usd
        gen_client: AsyncOpenAIClient | None = None
        try:
            if pending_tasks:
                gen_client = AsyncOpenAIClient(model=args.model_name)
                sem_gen = asyncio.Semaphore(max_concurrent)
                n_pt = len(pending_tasks)

                async def _mas_one(task: Any, ordinal: int) -> None:
                    nonlocal total_prompt_tokens, total_output_tokens, total_tokens, total_estimated_cost_usd
                    async with sem_gen:
                        t0 = time.perf_counter()
                        print(
                            f"[drb-skill-mas] start {ordinal}/{n_pt} id={task.id}",
                            flush=True,
                        )
                        order_idx = task_order_index[task.id]
                        trace_stem = skill_mas_sample_log_subdir(order_idx, task.id)
                        row, usage, trace_payload = await run_skill_mas_on_task_async(
                            task,
                            skill_dir=args.skill_dir,
                            client=gen_client,
                            dry_run=False,
                            process_trace_dir=process_trace_dir,
                            trace_stem=trace_stem,
                            io_lock=io_lock,
                            model_name=args.model_name,
                            include_trace_payload=True,
                        )
                        generated[task.id] = row
                        total_prompt_tokens += int(usage.get("prompt_token_count", usage.get("input_tokens", 0)) or 0)
                        total_output_tokens += int(usage.get("candidates_token_count", usage.get("output_tokens", 0)) or 0)
                        total_tokens += int(usage.get("total_token_count", usage.get("total_tokens", 0)) or 0)
                        total_estimated_cost_usd += float(usage.get("estimated_cost_usd", 0.0) or 0.0)
                        if args.eval_per_sample:
                            pending_judges.append((task, row.article, trace_payload, trace_stem))
                        else:
                            await asyncio.to_thread(
                                _write_trace_generated,
                                trace_stem=trace_stem,
                                base_payload=trace_payload,
                                status="generated",
                            )
                        if output_path is not None:
                            await asyncio.to_thread(
                                write_drb_articles,
                                [generated[k] for k in sorted(generated.keys())],
                                output_path,
                            )
                        elapsed = time.perf_counter() - t0
                        print(
                            f"[drb-skill-mas] done id={task.id} elapsed={elapsed:.1f}s "
                            f"cost={float(usage.get('estimated_cost_usd', 0.0) or 0.0):.6f}",
                            flush=True,
                        )

                await asyncio.gather(*[_mas_one(t, i + 1) for i, t in enumerate(pending_tasks)])
                await gen_client.aclose()
                gen_client = None

            if judge_client is not None:
                if pending_judges:
                    n_j = len(pending_judges)
                    sem_j = asyncio.Semaphore(max(1, int(args.judge_max_concurrent)))
                    done_lock = asyncio.Lock()
                    completed = 0

                    async def _judge_one(
                        ordinal: int,
                        task_obj: Any,
                        article_text: str,
                        trace_payload: dict[str, Any],
                        trace_stem: str,
                    ) -> None:
                        nonlocal completed
                        tid = int(task_obj.id)
                        print(
                            f"[drb-skill-mas] judge start {ordinal}/{n_j} id={task_obj.id}",
                            flush=True,
                        )
                        ref = reference_by_prompt.get(task_obj.prompt, "")
                        rubric = rubric_by_prompt.get(task_obj.prompt, {})
                        if not (ref and isinstance(rubric, dict) and rubric):
                            ev: dict[str, Any] = {"ok": False, "error": "missing_reference_or_rubric"}
                            judge_elapsed = 0.0
                        else:
                            t0 = time.perf_counter()
                            async with sem_j:
                                ev = await _evaluate_one_sample_async(
                                    task=task_obj,
                                    article=article_text,
                                    reference_article=ref,
                                    criteria_data=rubric,
                                    judge_client=judge_client,
                                )
                            judge_elapsed = time.perf_counter() - t0
                        await asyncio.to_thread(
                            _write_trace_generated,
                            trace_stem=trace_stem,
                            base_payload=trace_payload,
                            status="generated",
                            race_eval=ev,
                        )
                        async with done_lock:
                            completed += 1
                            cur = completed
                        if ev.get("ok"):
                            print(
                                f"[drb-skill-mas] judge done {cur}/{n_j} "
                                f"id={tid} score={float(ev.get('overall_score', 0.0)):.4f} elapsed={judge_elapsed:.1f}s",
                                flush=True,
                            )
                        else:
                            print(
                                f"[drb-skill-mas] judge failed {cur}/{n_j} "
                                f"id={tid} elapsed={judge_elapsed:.1f}s err={ev.get('error')}",
                                flush=True,
                            )

                    print(
                        f"[drb-skill-mas] judge pipeline (single event loop) total={n_j} "
                        f"max_concurrent={max(1, int(args.judge_max_concurrent))}",
                        flush=True,
                    )
                    await asyncio.gather(
                        *[
                            _judge_one(
                                i + 1,
                                task_obj,
                                article_text,
                                trace_payload,
                                trace_stem,
                            )
                            for i, (task_obj, article_text, trace_payload, trace_stem) in enumerate(pending_judges)
                        ]
                    )
                await judge_client.aclose()
        finally:
            if gen_client is not None:
                await gen_client.aclose()

    if pending_tasks or judge_client is not None:
        asyncio.run(_async_generate_and_judge())

    final_rows = [generated[k] for k in sorted(generated.keys())]
    if output_path is not None:
        write_drb_articles(final_rows, output_path)
        print(f"[drb-skill-mas] saved {len(final_rows)} rows to {output_path}", flush=True)
    else:
        print(f"[drb-skill-mas] generated {len(final_rows)} rows (no raw jsonl persisted)", flush=True)
    cost_output_path = Path(args.cost_output_file).resolve() if args.cost_output_file else (process_trace_dir / "generation_cost.json")
    cost_output_path.parent.mkdir(parents=True, exist_ok=True)
    cost_payload = {
        "schema_version": "skill_mas_cost_summary/1",
        "model_name": args.model_name,
        "num_rows": len(final_rows),
        "prompt_tokens": total_prompt_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_total_cost_usd_from_model_pricing_json": round(total_estimated_cost_usd, 10),
        "pricing_json_note": "USD per 1M tokens from Skill_MAS/skill_mas/model_config.json; sums per-sample usage.",
    }
    cost_output_path.write_text(json.dumps(cost_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[drb-skill-mas] saved cost summary to {cost_output_path}", flush=True)

    if not args.skip_race:
        race_output_dir = (
            Path(args.race_output_dir).resolve()
            if args.race_output_dir
            else (process_trace_dir / "race").resolve()
        )
        race_output_dir.mkdir(parents=True, exist_ok=True)
        model_tag = (args.race_model_tag or args.model_name).strip()
        if not model_tag:
            raise ValueError("race model tag is empty; set --race_model_tag or use a valid --output_file name.")
        if args.eval_per_sample:
            raw_results = []
            for trace_file in iter_per_sample_trace_json_files(process_trace_dir):
                obj = json.loads(trace_file.read_text(encoding="utf-8"))
                ev = obj.get("race_eval")
                if not isinstance(ev, dict):
                    continue
                task_id = int(obj.get("task_id", -1))
                prompt = str(obj.get("prompt", ""))
                if ev.get("ok"):
                    raw_results.append(
                        {
                            "id": task_id,
                            "prompt": prompt,
                            "comprehensiveness": float(ev.get("comprehensiveness", 0.0)),
                            "insight": float(ev.get("insight", 0.0)),
                            "instruction_following": float(ev.get("instruction_following", 0.0)),
                            "readability": float(ev.get("readability", 0.0)),
                            "overall_score": float(ev.get("overall_score", 0.0)),
                        }
                    )
                else:
                    raw_results.append({"id": task_id, "prompt": prompt, "error": ev.get("error", "judge_failed")})
            raw_results.sort(key=lambda x: x.get("id", 10**9))
            raw_path = race_output_dir / "raw_results.jsonl"
            with raw_path.open("w", encoding="utf-8") as f:
                for row in raw_results:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            ok_rows = [r for r in raw_results if "error" not in r]
            if ok_rows:
                comp = sum(float(r.get("comprehensiveness", 0.0)) for r in ok_rows) / len(ok_rows)
                insi = sum(float(r.get("insight", 0.0)) for r in ok_rows) / len(ok_rows)
                inst = sum(float(r.get("instruction_following", 0.0)) for r in ok_rows) / len(ok_rows)
                read = sum(float(r.get("readability", 0.0)) for r in ok_rows) / len(ok_rows)
                overall = sum(float(r.get("overall_score", 0.0)) for r in ok_rows) / len(ok_rows)
            else:
                comp = insi = inst = read = overall = 0.0
            (race_output_dir / "race_result.txt").write_text(
                "\n".join(
                    [
                        f"Comprehensiveness: {comp:.4f}",
                        f"Insight: {insi:.4f}",
                        f"Instruction Following: {inst:.4f}",
                        f"Readability: {read:.4f}",
                        f"Overall Score: {overall:.4f}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[drb-skill-mas] saved per-sample RACE results to {race_output_dir.resolve()}", flush=True)
        else:
            print("[drb-skill-mas] start RACE evaluation", flush=True)
            run_drb_evaluation(
                bench_dir=bench_dir,
                model_name=model_tag,
                raw_data_dir=(output_path.parent.resolve() if output_path is not None else process_trace_dir),
                output_dir=race_output_dir,
                query_file=query_path,
                process_trace_dir=process_trace_dir,
                use_model_subdir=False,
                max_workers=max(1, int(args.race_max_workers)),
                limit=args.limit,
                skip_cleaning=False,
                cleaned_data_dir=None,
                force=True,
            )
            print(f"[drb-skill-mas] saved RACE results to {race_output_dir.resolve()}", flush=True)

    manifest = {
        "schema_version": "skill_mas_run_manifest/1",
        "mode": "skill_mas",
        "model": args.model_name,
        "skill_dir": skill_dir_resolved,
        "skill_workspace": str(skill_workspace.resolve()),
        "skill_trace_suffix": skill_trace_suffix_joined,
        "dataset": str(query_path),
        "max_concurrency": int(max_concurrent),
        "num_tasks": len(tasks),
        "process_trace_dir": str(process_trace_dir.resolve()),
        "run_results": {
            "generation_cost_json": str(cost_output_path.resolve()),
            "raw_output_jsonl": str(output_path.resolve()) if output_path is not None else None,
            "race_result_dir": str(race_output_dir.resolve()) if not args.skip_race else None,
            "sample_trace_json_dir": str(sample_trace_json_dir(process_trace_dir).resolve()),
        },
    }
    (process_trace_dir / "_skill_mas_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[drb-skill-mas] saved manifest to {process_trace_dir / '_skill_mas_run_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

