#!/usr/bin/env python3
"""Evaluate Skill-MAS on BrowseComp JSONL via Skill_MAS `build.run_mas_pipeline_with_retries` (init SKILL.md path)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from Skill_MAS.utils.paths import BCP_ROOT, INIT_SKILL_DIR, ensure_sys_path

ensure_sys_path(include_dataset=True, include_bcp=True)
_BENCH_ROOT = BCP_ROOT

from bcp_io import load_jsonl
from browsecomp_retrieval_tool import make_browsecomp_retrieval_tool_fn
from judge import judge_answer
from openai_client import AsyncOpenAIClient
from retrieval import BM25Retriever
from skill_mas_runner import BrowseCompTask, run_browsecomp_skill_mas_on_task
from Skill_MAS.skill_mas.process_trace_layout import (
    sample_trace_json_dir,
    skill_mas_sample_log_subdir,
    skill_mas_trace_suffix_parts,
    skill_workspace_from_init_path,
)


def _default_preload_planner_llm() -> str:
    env = (os.environ.get("BROWSECOMP_SKILL_MAS_PLANNER_LLM") or "").strip()
    if env:
        return env
    env2 = (os.environ.get("VITABENCH_SKILL_MAS_PLANNER_LLM") or "").strip()
    if env2:
        return env2
    return "deepseek-v4-flash"


def _skill_mas_results_model_label(*, agent_llm: str, planner_llm: str | None) -> str:
    """Top-level ``results/<model>/`` folder; preload runs use planner, not executor."""
    if planner_llm:
        return planner_llm.split("/")[-1]
    return agent_llm.split("/")[-1]


def _skill_mas_trace_suffix_parts(skill_workspace: Path, planner_llm: str | None) -> list[str]:
    parts = skill_mas_trace_suffix_parts(skill_workspace)
    if planner_llm:
        return ["preload_agent", *parts]
    return parts


def _load_existing_scored_sample(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    required = {"index", "score", "extracted", "usage", "prediction"}
    if not isinstance(data, dict) or not required.issubset(data.keys()):
        return None
    return data


def main() -> None:
    default_init = INIT_SKILL_DIR / "SKILL.md"
    env_init = (os.environ.get("SKILL_MAS_INIT_SKILL") or "").strip()
    p = argparse.ArgumentParser(description="BrowseComp Skill-MAS evaluation")
    p.add_argument(
        "--jsonl",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "browsecomp_plus_validate.jsonl",
    )
    p.add_argument(
        "--init-skill",
        type=Path,
        default=Path(env_init) if env_init else default_init,
        help="Path to root SKILL.md for build.generate_mas_artifacts. Default: SKILL_MAS_INIT_SKILL or Skill_MAS/init_skill/SKILL.md",
    )
    p.add_argument("--agent-llm", default=os.environ.get("AGENT_MODEL", "qwen3.5-plus"))
    p.add_argument(
        "--planner-llm",
        default=None,
        help="Planner LLM for Skill-MAS build stages (preload mode). Saves under results/<planner-llm>/...",
    )
    p.add_argument("--max-concurrency", type=int, default=50, help="Max concurrent samples (capped at 50)")
    p.add_argument("--max-problems", type=int, default=0, help="0 = all lines")
    p.add_argument("--indices", type=str, default="", help="Comma-separated line indices (default: all)")
    p.add_argument("--judge-llm", default="deepseek-v4-flash")
    p.add_argument("--judge-timeout-s", type=float, default=120.0)
    p.add_argument("--per-sample-timeout-s", type=float, default=600.0)
    p.add_argument(
        "--index-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "scripts_build_index" / "indexes" / "bm25"),
        help="BM25 index path (same as single-agent; recorded for runs / future tool wiring)",
    )
    p.add_argument("--retrieval-topk", type=int, default=5)
    p.add_argument("--doc-max-tokens", type=int, default=512)
    p.add_argument("--max-retrieval-rounds", type=int, default=10)
    p.add_argument("--out", type=Path, default=None, help="Write JSON summary path")
    args = p.parse_args()

    init_skill = Path(args.init_skill).resolve()
    if not init_skill.is_file():
        raise FileNotFoundError(f"--init-skill must be an existing SKILL.md file: {init_skill}")

    rows = load_jsonl(Path(args.jsonl).resolve())
    if args.indices.strip():
        want = {int(x.strip()) for x in args.indices.split(",") if x.strip()}
        indices = sorted(want)
        for i in indices:
            if i < 0 or i >= len(rows):
                raise IndexError(f"index {i} out of range [0,{len(rows)})")
    else:
        indices = list(range(len(rows)))
        if args.max_problems and args.max_problems > 0:
            indices = indices[: args.max_problems]

    planner_llm = (args.planner_llm or "").strip() or None
    preload_mode = bool(planner_llm)

    agent_llm_client = AsyncOpenAIClient(model=args.agent_llm)
    planner_llm_client = AsyncOpenAIClient(model=planner_llm) if preload_mode else agent_llm_client
    client = agent_llm_client
    chat_client = agent_llm_client.client
    reasoning_effort = agent_llm_client.reasoning_effort
    retriever = BM25Retriever(index_path=args.index_path)
    judge_llm_client = AsyncOpenAIClient(model=args.judge_llm)
    skill_workspace = skill_workspace_from_init_path(init_skill)
    results_model = _skill_mas_results_model_label(agent_llm=args.agent_llm, planner_llm=planner_llm)
    trace_suffix = _skill_mas_trace_suffix_parts(skill_workspace, planner_llm)
    trace_root = (Path(__file__).resolve().parent / "results" / results_model / "skill_mas_process_traces").joinpath(
        *trace_suffix
    )
    trace_root.mkdir(parents=True, exist_ok=True)
    scored_root = trace_root / "scored_samples"
    scored_root.mkdir(parents=True, exist_ok=True)

    async def _run_all() -> tuple[list[dict], int, float]:
        sem = asyncio.Semaphore(max(1, min(50, int(args.max_concurrency))))
        progress_lock = asyncio.Lock()
        done = 0
        total = len(indices)

        async def _one(idx: int) -> dict:
            nonlocal done
            scored_path = scored_root / f"{idx}.json"
            existing = _load_existing_scored_sample(scored_path)
            if existing is not None:
                async with progress_lock:
                    done += 1
                    print(f"[browsecomp-skill-mas] resume hit {done}/{total} index={idx} score={existing.get('score')}", flush=True)
                return existing

            row = rows[idx]
            q = str(row.get("query", ""))
            gold = str(row.get("answer", ""))
            row_tid = row.get("id", idx)
            trace_stem = skill_mas_sample_log_subdir(idx, row_tid)
            t0 = time.perf_counter()
            print(f"[browsecomp-skill-mas] start index={idx}", flush=True)
            try:
                task = BrowseCompTask(id=int(idx), question=q)
                retrieval_sessions: list[dict] = []
                tool_fn = make_browsecomp_retrieval_tool_fn(
                    chat_client=chat_client,
                    model=args.agent_llm,
                    retriever=retriever,
                    retrieval_topk=int(args.retrieval_topk),
                    doc_max_tokens=int(args.doc_max_tokens),
                    max_rounds=int(args.max_retrieval_rounds),
                    reasoning_effort=reasoning_effort,
                    full_task_question=q,
                    pricing=agent_llm_client.pricing,
                    retrieval_sessions_out=retrieval_sessions,
                )

                async with sem:
                    try:
                        pred_text, usage, trace_payload = await asyncio.wait_for(
                            run_browsecomp_skill_mas_on_task(
                                task,
                                init_skill_path=init_skill,
                                client=client,
                                planner_client=planner_llm_client if preload_mode else None,
                                process_trace_dir=trace_root,
                                trace_stem=trace_stem,
                                quiet=True,
                                tool_call_fn=tool_fn,
                                retrieval_sessions_out=retrieval_sessions,
                            ),
                            timeout=max(1.0, float(args.per_sample_timeout_s)),
                        )
                    except asyncio.TimeoutError:
                        raise RuntimeError(
                            f"Skill-MAS pipeline exceeded per-sample timeout ({args.per_sample_timeout_s}s)"
                        ) from None
                judge = await asyncio.wait_for(
                    judge_answer(
                        judge_client=judge_llm_client,
                        question=q,
                        response=pred_text,
                        correct_answer=gold,
                    ),
                    timeout=max(1.0, float(args.judge_timeout_s)),
                )
                sc = int(judge.get("score", 0))
                extracted = str(judge.get("extracted_final_answer", ""))
                sample = {
                    "index": idx,
                    "id": row.get("id", idx),
                    "query": q,
                    "expected_answer": gold,
                    "prediction": pred_text,
                    "score": sc,
                    "extracted": extracted,
                    "usage": usage,
                    "skill_trace_schema": trace_payload.get("schema_version"),
                    "judge": judge,
                    "prediction_preview": pred_text[:500],
                    "model": args.agent_llm,
                }
                if preload_mode:
                    sample["planner_llm"] = planner_llm
                    sample["results_model"] = results_model
            except Exception as e:
                sample = {
                    "index": idx,
                    "id": row.get("id", idx),
                    "query": q,
                    "expected_answer": gold,
                    "prediction": "",
                    "score": 0,
                    "extracted": "",
                    "usage": {},
                    "judge": {
                        "extracted_final_answer": "None",
                        "reasoning": f"Exception during generation/judge: {e}",
                        "correct": "no",
                        "confidence": 100,
                        "score": 0,
                    },
                    "prediction_preview": "",
                    "model": args.agent_llm,
                    "error": str(e),
                }
            scored_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
            elapsed = time.perf_counter() - t0
            async with progress_lock:
                done += 1
                print(
                    f"[browsecomp-skill-mas] done {done}/{total} index={idx} "
                    f"score={sample.get('score', 0)} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            return sample

        try:
            out = await asyncio.gather(*[_one(i) for i in indices])
            out.sort(key=lambda x: int(x["index"]))
            total_score = sum(int(x["score"]) for x in out)
            total_cost = sum(float((x.get("usage") or {}).get("estimated_cost_usd", 0.0)) for x in out)
            total_cost += sum(
                float(((x.get("judge") or {}).get("usage_totals") or {}).get("estimated_cost_usd", 0.0))
                for x in out
            )
            return out, total_score, total_cost
        finally:
            await agent_llm_client.aclose()
            if preload_mode and planner_llm_client is not agent_llm_client:
                await planner_llm_client.aclose()
            await judge_llm_client.aclose()

    results, total_score, total_cost = asyncio.run(_run_all())
    n = len(indices) or 1
    summary = {
        "jsonl": str(Path(args.jsonl).resolve()),
        "init_skill_path": str(init_skill),
        "agent_llm": args.agent_llm,
        "planner_llm": planner_llm,
        "results_model": results_model,
        "mode": "preload_agent" if preload_mode else "skill_mas",
        "judge_llm": args.judge_llm,
        "per_sample_timeout_s": float(args.per_sample_timeout_s),
        "index_path": args.index_path,
        "retrieval_topk": int(args.retrieval_topk),
        "doc_max_tokens": int(args.doc_max_tokens),
        "max_retrieval_rounds": int(args.max_retrieval_rounds),
        "num_evaluated": len(indices),
        "accuracy": total_score / n,
        "total_score": total_score,
        "estimated_total_cost_usd_from_model_pricing_json": round(total_cost, 10),
        "results": results,
    }
    out = args.out if args.out is not None else trace_root / "browsecomp_skill_mas_eval_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (trace_root / "generation_cost.json").write_text(
        json.dumps(
            {
                "model": results_model,
                "executor_llm": args.agent_llm,
                "planner_llm": planner_llm,
                "num_tasks": len(indices),
                "estimated_total_cost_usd_from_model_pricing_json": round(total_cost, 10),
                "pricing_json_note": "USD per 1M tokens from Skill_MAS/skill_mas/model_config.json via Skill_MAS enrich_usage_with_cost; sums workflow usage + judge.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (trace_root / "_skill_mas_run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "skill_mas_run_manifest/2",
                "mode": "preload_agent" if preload_mode else "skill_mas",
                "model": results_model,
                "executor_llm": args.agent_llm,
                "planner_llm": planner_llm,
                "judge_model": args.judge_llm,
                "dataset": str(Path(args.jsonl).resolve()),
                "init_skill_path": str(init_skill),
                "dataset_name": "bcp",
                "pipeline": "Skill_MAS.build.run_mas_pipeline_with_retries",
                "max_concurrency": int(max(1, min(50, int(args.max_concurrency)))),
                "per_sample_timeout_s": float(args.per_sample_timeout_s),
                "index_path": args.index_path,
                "retrieval_topk": int(args.retrieval_topk),
                "doc_max_tokens": int(args.doc_max_tokens),
                "max_retrieval_rounds": int(args.max_retrieval_rounds),
                "process_trace_dir": str(trace_root.resolve()),
                "run_results": {
                    "summary_json": str(out.resolve()),
                    "generation_cost_json": str((trace_root / "generation_cost.json").resolve()),
                    "scored_samples_dir": str(scored_root.resolve()),
                    "sample_trace_json_dir": str(sample_trace_json_dir(trace_root).resolve()),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"accuracy": summary["accuracy"], "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
