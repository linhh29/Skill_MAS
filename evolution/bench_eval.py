"""Benchmark evaluation adapters reusing native benchmark implementations."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

from Skill_MAS.utils.paths import ensure_sys_path

ensure_sys_path(include_dataset=True, include_bcp=True)

from browsecomp_retrieval_tool import make_browsecomp_retrieval_tool_fn
from bcp_io import load_jsonl
from deep_research_bench.drb_runtime import DRBArticle, DRBTask, load_drb_tasks, write_drb_articles
from deep_research_bench.evaluate import run_drb_evaluation
from deep_research_bench.skill_mas_agent_runner import run_skill_mas_on_task, run_skill_mas_on_task_async
from hlemath.score import HLEMATHScorer
from judge import judge_answer
from retrieval import BM25Retriever
from score import BrowseCompScorer
from Skill_MAS.skill_mas.openai_async_client import AsyncOpenAIClient
from skill_mas_runner import BrowseCompTask, run_browsecomp_skill_mas_on_task

from ..utils.config import DRB_BENCH_ROOT, ROUND_BENCH_ROLLOUTS_DIRNAME, print_traces_enabled
from ..utils.llm_cost import (
    _skill_mas_pricing_table,
    add_usage_totals,
    empty_usage_totals,
    pricing_reference,
    strict_resolve_pricing_model_key,
)
from ..utils.run_log_export import export_drb_evaluation_logs
from ..core.model_config_runtime import model_runtime_params

_drb_br_resume_lock = threading.Lock()
_drb_br_resume_count = 0


def reset_drb_bench_rollout_resume_counter() -> None:
    """Call before a batch of parallel ``run_drb_evaluation_round`` invocations."""
    global _drb_br_resume_count
    with _drb_br_resume_lock:
        _drb_br_resume_count = 0


def take_drb_bench_rollout_resume_count() -> int:
    """Return how many DRB rollouts reused bench_rollouts on-disk cache since last reset, then clear."""
    global _drb_br_resume_count
    with _drb_br_resume_lock:
        n = _drb_br_resume_count
        _drb_br_resume_count = 0
        return n


def _drb_bench_rollout_resume_note() -> None:
    global _drb_br_resume_count
    with _drb_br_resume_lock:
        _drb_br_resume_count += 1


def _sanitize_model_tag(raw: str) -> str:
    s = raw.strip().replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    s = s.strip("._-") or "maskill_eval"
    return s[:80]


def _drb_cached_rollout_complete(bundle_path: Path, expected_task_ids: list[int]) -> bool:
    """
    True when a previous run left a bundle plus RACE outputs on disk so we can skip
    Skill-MAS + RACE regeneration (bench_rollouts reuse).
    """
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    got = sorted(int(x) for x in (bundle.get("task_ids") or []))
    if got != sorted(int(x) for x in expected_task_ids):
        return False
    race_txt = Path(bundle.get("race_result_txt") or "")
    if not race_txt.is_file():
        return False
    raw_results = race_txt.parent / "raw_results.jsonl"
    if not raw_results.is_file() or raw_results.stat().st_size == 0:
        return False
    for key in ("raw_jsonl", "query_subset_jsonl", "process_trace_dir"):
        p = Path(bundle.get(key) or "")
        if key == "process_trace_dir":
            if not p.is_dir():
                return False
        elif not p.is_file():
            return False
    return True


def _usage_rows_from_drb_bundle(bundle_path: Path, expected_task_ids: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rebuild aggregate_usage / per_task rows from cached process-trace JSON (nested or flat)."""
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    ptd = Path(bundle.get("process_trace_dir") or "")
    aggregate_usage = empty_usage_totals()
    per_task_rows: list[dict[str, Any]] = []
    want = {int(x) for x in expected_task_ids}
    files: list[Path] = []
    nested = ptd / "sample_trace_json"
    if nested.is_dir():
        files.extend(sorted(f for f in nested.glob("*.json") if not f.name.startswith("_")))
    if not files:
        files.extend(sorted(f for f in ptd.glob("*.json") if not f.name.startswith("_")))
    seen: set[int] = set()
    for path in files:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tid_val = row.get("task_id", row.get("id"))
        if tid_val is None:
            continue
        try:
            tid_i = int(tid_val)
        except (TypeError, ValueError):
            continue
        if tid_i not in want:
            continue
        u = dict(row.get("usage_totals") or {})
        if not u:
            continue
        seen.add(tid_i)
        add_usage_totals(aggregate_usage, u)
        per_task_rows.append({"task_id": tid_i, **u})
    return aggregate_usage, per_task_rows


def _artifact_round_dir(runs_dir: Path, bench_id: str, run_id: str, round_idx: int) -> Path:
    """Per-round runs root (``…/runs/<bench>/<run>/round_XX``); aligns with vitabench artifact layout."""
    p = runs_dir / bench_id / run_id / f"round_{round_idx:02d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _bench_rollouts_dir(round_prefix: Path) -> Path:
    """Holds per-round native benchmark workspaces so ``round_XX/`` matches VitaBench top-level layout."""
    p = round_prefix / ROUND_BENCH_ROLLOUTS_DIRNAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_drb_bench_layout() -> None:
    if not DRB_BENCH_ROOT.is_dir():
        raise FileNotFoundError(
            f"Expected DeepResearchBench at {DRB_BENCH_ROOT} (clone deep_research_bench next to Ant)."
        )


def ensure_hlemath_jsonl_readable(jsonl_path: Path) -> None:
    p = Path(jsonl_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"HLEMATH JSONL not found: {p}")
    first = next((ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()), "")
    if first.startswith("version https://git-lfs.github.com"):
        raise RuntimeError(f"{p} is a Git LFS pointer; fetch real JSONL first.")


def ensure_bcp_jsonl_readable(jsonl_path: Path) -> None:
    p = Path(jsonl_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"BrowseComp JSONL not found: {p}")
    first = ""
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                first = line.strip()
                break
    if not first:
        raise RuntimeError(f"BrowseComp JSONL is empty: {p}")


def build_drb_client(agent_llm: str) -> AsyncOpenAIClient:
    if not os.environ.get("OPENAI_API_KEY"):
        params = model_runtime_params(agent_llm)
        if params.get("api_key"):
            os.environ["OPENAI_API_KEY"] = str(params["api_key"])
        if params.get("base_url") and not os.environ.get("OPENAI_API_BASE"):
            os.environ["OPENAI_API_BASE"] = str(params["base_url"])
    if os.environ.get("OPENAI_API_KEY"):
        return AsyncOpenAIClient(model=agent_llm)
    raise ValueError("DRB skill-mas requires OPENAI_API_KEY in the environment (OpenAI-compatible endpoint).")


def run_bcp_evaluation_round(
    *,
    bench_id: str,
    run_id: str,
    round_idx: int,
    task_ids: list[str],
    jsonl_path: Path,
    agent_llm: str,
    max_concurrency: int,
    runs_dir: Path,
    log_root: Path,
    skills_evolution_dir: Path,
    trajectory_tag: str = "",
    judge_llm: str | None = None,
    judge_timeout_s: float = 120.0,
    bcp_index_path: str | Path | None = None,
    bcp_retrieval_topk: int = 5,
    bcp_doc_max_tokens: int = 512,
    bcp_max_retrieval_rounds: int = 10,
) -> tuple[Path, Path, dict[str, Any]]:
    del log_root
    suffix = f"_{trajectory_tag}" if trajectory_tag.strip() else ""
    round_prefix = _artifact_round_dir(runs_dir, bench_id, run_id, round_idx)
    br = _bench_rollouts_dir(round_prefix)
    root = br / f"bcp_eval_r{round_idx:02d}{suffix}"
    if root.exists():
        shutil.rmtree(root)
    process_trace_dir = root / "process_traces"
    process_trace_dir.mkdir(parents=True, exist_ok=True)
    skill_round = skills_evolution_dir / bench_id / run_id / f"round_{round_idx:02d}"
    skill_md = skill_round / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"Skill workspace missing SKILL.md: {skill_round}")
    os.environ["SKILL_MAS_DIR"] = str(skill_round.resolve())
    rows = load_jsonl(Path(jsonl_path).resolve())
    row_by_id = {str(r.get("id")): r for r in rows if r.get("id") is not None}
    missing = [tid for tid in task_ids if tid not in row_by_id]
    if missing:
        raise RuntimeError(f"BrowseComp validation file missing ids: {missing[:10]}")
    ordered_ids = [str(x) for x in task_ids]
    scorer = BrowseCompScorer()
    client = build_drb_client(agent_llm)
    idx_default = _REPO / "BrowseComp-Plus" / "scripts_build_index" / "indexes" / "bm25"
    index_resolved = Path(bcp_index_path).resolve() if bcp_index_path else idx_default
    retriever = BM25Retriever(index_path=str(index_resolved))
    chat_client = client.client
    judge_model = (judge_llm or "").strip()
    use_judge = bool(judge_model)
    judge_client: AsyncOpenAIClient | None = build_drb_client(judge_model) if use_judge else None
    aggregate_usage = empty_usage_totals()
    per_task_rows: list[dict[str, Any]] = []
    per_task_scores: dict[str, float] = {}
    per_task_judges: dict[str, Any] = {}

    async def _one(
        tid: str,
    ) -> tuple[str, float, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        row = row_by_id[tid]
        q = str(row.get("query", ""))
        task = BrowseCompTask(id=int(tid), question=q)
        tool_fn = make_browsecomp_retrieval_tool_fn(
            chat_client=chat_client,
            model=agent_llm,
            retriever=retriever,
            retrieval_topk=int(bcp_retrieval_topk),
            doc_max_tokens=int(bcp_doc_max_tokens),
            max_rounds=int(bcp_max_retrieval_rounds),
            reasoning_effort=client.reasoning_effort,
            full_task_question=q,
            pricing=client.pricing,
            retrieval_sessions_out=None,
        )
        pred, usage, _trace_payload = await run_browsecomp_skill_mas_on_task(
            task,
            init_skill_path=skill_md,
            client=client,
            process_trace_dir=process_trace_dir,
            quiet=True,
            tool_call_fn=tool_fn,
            retrieval_sessions_out=None,
        )
        if use_judge and judge_client is not None:
            try:
                judge = await asyncio.wait_for(
                    judge_answer(
                        judge_client=judge_client,
                        question=str(row.get("query", "")),
                        response=pred,
                        correct_answer=str(row.get("answer", "")),
                    ),
                    timeout=max(1.0, float(judge_timeout_s)),
                )
            except asyncio.TimeoutError:
                judge = {
                    "extracted_final_answer": "None",
                    "reasoning": "Timeout in judge.",
                    "correct": "no",
                    "confidence": 100,
                    "score": 0,
                    "usage_totals": {},
                }
            except Exception as e:
                judge = {
                    "extracted_final_answer": "None",
                    "reasoning": f"Exception during judge: {e}",
                    "correct": "no",
                    "confidence": 100,
                    "score": 0,
                    "usage_totals": {},
                }
            score_i = int(judge.get("score", 0))
            ju = dict(judge.get("usage_totals") or {})
            return tid, float(score_i), dict(usage), ju, judge
        score_i, _extracted = scorer.calculate_score(str(row.get("answer", "")), pred)
        return tid, float(score_i), dict(usage), {}, None

    async def _run_all() -> list[tuple[str, float, dict[str, Any], dict[str, Any], dict[str, Any] | None]]:
        sem = asyncio.Semaphore(max(1, int(max_concurrency)))

        async def _guarded(
            tid: str,
        ) -> tuple[str, float, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
            async with sem:
                return await _one(tid)

        try:
            return await asyncio.gather(*[_guarded(tid) for tid in ordered_ids])
        finally:
            # Close on the same loop that owns the httpx pool; a second asyncio.run() after
            # run() exits would use a new loop and trigger "Event loop is closed" on teardown.
            await client.aclose()
            if judge_client is not None:
                await judge_client.aclose()

    rows_result = asyncio.run(_run_all())
    for tid, score_f, usage, ju, judge_payload in rows_result:
        per_task_scores[tid] = score_f
        add_usage_totals(aggregate_usage, usage)
        add_usage_totals(aggregate_usage, ju)
        row_out: dict[str, Any] = {"task_id": tid, **usage}
        if ju:
            row_out["judge_usage_totals"] = ju
        per_task_rows.append(row_out)
        if judge_payload is not None:
            per_task_judges[tid] = judge_payload

    bundle = {
        "bench_backend": "bcp",
        "round_idx": round_idx,
        "bench_id": bench_id,
        "run_id": run_id,
        "task_ids": ordered_ids,
        "jsonl_file": str(Path(jsonl_path).resolve()),
        "process_trace_dir": str(process_trace_dir.resolve()),
        "merged_skill_dir": str(skill_round.resolve()),
        "per_task_scores": per_task_scores,
        "judge_llm": judge_model if use_judge else None,
        "per_task_judges": per_task_judges if use_judge else {},
        "bcp_retrieval": {
            "index_path": str(index_resolved),
            "retrieval_topk": int(bcp_retrieval_topk),
            "doc_max_tokens": int(bcp_doc_max_tokens),
            "max_retrieval_rounds": int(bcp_max_retrieval_rounds),
        },
    }
    bundle_path = br / f"bcp_bundle_r{round_idx:02d}{suffix}.json"
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_section: dict[str, Any] = {
        "phase": "bcp_skill_mas_rollout",
        "description": (
            "Skill-MAS generation + BrowseComp LLM-judge scoring."
            if use_judge
            else "Skill-MAS generation + BrowseComp exact-match scoring."
        ),
        "model": agent_llm,
        "judge_llm": judge_model if use_judge else None,
        "aggregate_usage": aggregate_usage,
        "per_task": per_task_rows,
        "pricing_reference": pricing_reference(),
    }
    return bundle_path, root, eval_section


def run_hlemath_evaluation_round(
    *,
    bench_id: str,
    run_id: str,
    round_idx: int,
    task_ids: list[str],
    jsonl_path: Path,
    agent_llm: str,
    max_concurrency: int,
    runs_dir: Path,
    log_root: Path,
    skills_evolution_dir: Path,
    trajectory_tag: str = "",
) -> tuple[Path, Path, dict[str, Any]]:
    del log_root, max_concurrency
    strict_resolve_pricing_model_key(agent_llm, _skill_mas_pricing_table())
    suffix = f"_{_sanitize_model_tag(trajectory_tag)}" if trajectory_tag.strip() else ""
    round_prefix = _artifact_round_dir(runs_dir, bench_id, run_id, round_idx)
    br = _bench_rollouts_dir(round_prefix)
    root = br / f"hlemath_eval_r{round_idx:02d}{suffix}"
    if root.exists():
        shutil.rmtree(root)
    process_trace_dir = root / "process_traces"
    process_trace_dir.mkdir(parents=True, exist_ok=True)
    skill_round = skills_evolution_dir / bench_id / run_id / f"round_{round_idx:02d}"
    if not (skill_round / "SKILL.md").is_file():
        raise FileNotFoundError(f"Skill workspace missing SKILL.md: {skill_round}")
    os.environ["SKILL_MAS_DIR"] = str(skill_round.resolve())
    rows: list[dict[str, Any]] = []
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith("version https://git-lfs.github.com"):
                raise RuntimeError(f"{jsonl_path} is Git LFS pointer; use real JSONL (git lfs pull).")
            rows.append(json.loads(raw))
    id_set = {int(x) for x in task_ids}
    missing = sorted(id_set - set(range(len(rows))))
    if missing:
        raise RuntimeError(f"HLEMATH jsonl missing indices: {missing[:10]}")
    ordered_tasks = sorted(id_set)

    async def _async_round() -> tuple[Path, Path, dict[str, Any]]:
        aggregate_usage = empty_usage_totals()
        per_task_rows: list[dict[str, Any]] = []
        per_task_scores: dict[str, float] = {}
        scorer = HLEMATHScorer()
        client = build_drb_client(agent_llm)
        try:
            for idx in ordered_tasks:
                from openai import BadRequestError

                row = rows[idx]
                task = DRBTask(id=int(idx), prompt=str(row.get("question", "")), language="en")
                gold = str(row.get("answer", ""))
                try:
                    art, usage = await run_skill_mas_on_task_async(
                        task,
                        skill_dir=skill_round.resolve(),
                        client=client,
                        dry_run=False,
                        process_trace_dir=process_trace_dir,
                        io_lock=None,
                        quiet=True,
                        model_name=agent_llm,
                    )
                except BadRequestError:
                    art = DRBArticle(id=int(idx), prompt=task.prompt, article="")
                    usage = dict(empty_usage_totals())
                score_f = float(scorer.calculate_score(gold, art.article or "")[0])
                per_task_scores[str(idx)] = score_f
                add_usage_totals(aggregate_usage, usage)
                per_task_rows.append({"task_id": idx, **usage})

            bundle = {
                "bench_backend": "hlemath",
                "round_idx": round_idx,
                "bench_id": bench_id,
                "run_id": run_id,
                "task_ids": [str(x) for x in ordered_tasks],
                "jsonl_file": str(Path(jsonl_path).resolve()),
                "process_trace_dir": str(process_trace_dir.resolve()),
                "merged_skill_dir": str(skill_round.resolve()),
                "per_task_scores": per_task_scores,
            }
            bundle_path = br / f"hlemath_bundle_r{round_idx:02d}{suffix}.json"
            bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
            eval_section: dict[str, Any] = {
                "phase": "hlemath_skill_mas_rollout",
                "description": "Skill-MAS generation + HLEMATH sympy/boxed scoring.",
                "model": agent_llm,
                "aggregate_usage": aggregate_usage,
                "per_task": per_task_rows,
                "pricing_reference": pricing_reference(),
            }
            return bundle_path, root, eval_section
        finally:
            await client.aclose()

    return asyncio.run(_async_round())


def run_drb_evaluation_round(
    *,
    bench_id: str,
    run_id: str,
    round_idx: int,
    task_ids: list[int],
    drb_bench_root: Path,
    agent_llm: str,
    max_concurrency: int,
    race_max_workers: int,
    runs_dir: Path,
    log_root: Path,
    skills_evolution_dir: Path,
    merged_workspaces_dir: Path,
    trajectory_tag: str = "",
    export_logs: bool = True,
    drb_query_jsonl: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    del merged_workspaces_dir
    if not drb_bench_root.is_dir():
        raise FileNotFoundError(f"DeepResearchBench not found: {drb_bench_root}")
    query_file = Path(drb_query_jsonl).resolve() if drb_query_jsonl else drb_bench_root / "data" / "prompt_data" / "query.jsonl"
    if not query_file.is_file():
        raise FileNotFoundError(f"Missing DRB query file: {query_file}")
    strict_resolve_pricing_model_key(agent_llm, _skill_mas_pricing_table())
    suffix = f"_{_sanitize_model_tag(trajectory_tag)}" if trajectory_tag.strip() else ""
    round_prefix = _artifact_round_dir(runs_dir, bench_id, run_id, round_idx)
    br = _bench_rollouts_dir(round_prefix)
    bundle_path = br / f"drb_bundle_r{round_idx:02d}{suffix}.json"
    root = br / f"drb_eval_r{round_idx:02d}{suffix}"

    skill_round = skills_evolution_dir / bench_id / run_id / f"round_{round_idx:02d}"
    if not (skill_round / "SKILL.md").is_file():
        raise FileNotFoundError(f"Skill workspace missing SKILL.md under {skill_round}")
    os.environ["SKILL_MAS_DIR"] = str(skill_round.resolve())
    target_id_set = {int(x) for x in task_ids}
    tasks = [t for t in load_drb_tasks(query_file) if int(t.id) in target_id_set]
    if len(tasks) != len(task_ids):
        raise RuntimeError("DRB query file missing task ids.")
    tid_list = [int(t.id) for t in tasks]

    if bundle_path.is_file() and _drb_cached_rollout_complete(bundle_path, tid_list):
        aggregate_usage, per_task_rows = _usage_rows_from_drb_bundle(bundle_path, tid_list)
        eval_section_cache: dict[str, Any] = {
            "phase": "drb_skill_mas_rollout",
            "description": "Skill-MAS per-task usage totals (reused cached bench_rollouts).",
            "model": agent_llm,
            "aggregate_usage": aggregate_usage,
            "per_task": per_task_rows,
            "pricing_reference": pricing_reference(),
        }
        log_round_cache = (
            export_drb_evaluation_logs(
                bundle_path=bundle_path,
                bench_id=bench_id,
                run_id=run_id,
                round_idx=round_idx,
                log_root=log_root,
                merged_skill_dir=skill_round.resolve(),
            )
            if export_logs
            else root
        )
        _drb_bench_rollout_resume_note()
        return bundle_path, log_round_cache, eval_section_cache

    if root.exists():
        shutil.rmtree(root)
    raw_data_dir = root / "raw_data"
    cleaned_data_dir = root / "cleaned_data"
    process_trace_dir = root / "process_traces"
    race_out = root / "race_out"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    cleaned_data_dir.mkdir(parents=True, exist_ok=True)
    process_trace_dir.mkdir(parents=True, exist_ok=True)
    del max_concurrency
    trace_out = print_traces_enabled()

    async def _generate_rollout() -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
        client = build_drb_client(agent_llm)
        aggregate_usage = empty_usage_totals()
        per_task_rows: list[dict[str, Any]] = []
        articles: list[Any] = []
        try:
            for t in tasks:
                from openai import BadRequestError

                try:
                    art, usage = await run_skill_mas_on_task_async(
                        t,
                        skill_dir=skill_round.resolve(),
                        client=client,
                        dry_run=False,
                        process_trace_dir=process_trace_dir,
                        io_lock=None,
                        quiet=not trace_out,
                    )
                except BadRequestError as exc:
                    brief = str(exc).split("\n", 1)[0][:400]
                    art = DRBArticle(id=int(t.id), prompt=t.prompt, article=f"[SKILL_MAS_RUN_FAILED]\n\n{brief}")
                    usage = dict(empty_usage_totals())
                if not isinstance(getattr(art, "article", None), str) or not str(art.article).strip():
                    art = DRBArticle(
                        id=int(t.id),
                        prompt=t.prompt,
                        article="[SKILL_MAS_RUN_FAILED]\n\nempty article returned by run_skill_mas_on_task",
                    )
                articles.append(art)
                u = dict(usage)
                add_usage_totals(aggregate_usage, u)
                per_task_rows.append({"task_id": int(t.id), **u})
            return articles, aggregate_usage, per_task_rows
        finally:
            await client.aclose()

    articles, aggregate_usage, per_task_rows = asyncio.run(_generate_rollout())

    tag = _sanitize_model_tag(f"{bench_id}_{run_id}_r{round_idx:02d}{suffix}")
    raw_jsonl = raw_data_dir / f"{tag}.jsonl"
    write_drb_articles(articles, raw_jsonl)
    # Keep full query rows (rubric/reference_article/etc.) for the new async evaluator path.
    query_subset_path = root / f"query_r{round_idx:02d}.jsonl"
    row_by_id: dict[int, dict[str, Any]] = {}
    with query_file.open("r", encoding="utf-8") as f:
        for ln in f:
            raw = ln.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                continue
            rid = obj.get("id")
            if rid is None:
                continue
            try:
                rid_i = int(rid)
            except Exception:
                continue
            if rid_i in target_id_set:
                row_by_id[rid_i] = obj
    with query_subset_path.open("w", encoding="utf-8") as f:
        for t in tasks:
            rid = int(t.id)
            row = row_by_id.get(rid)
            if row is None:
                # Fallback keeps compatibility with legacy query layouts.
                row = {"id": rid, "prompt": t.prompt, "language": t.language}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    run_drb_evaluation(
        bench_dir=drb_bench_root.resolve(),
        model_name=tag,
        raw_data_dir=raw_data_dir.resolve(),
        output_dir=race_out.resolve(),
        query_file=query_subset_path.resolve(),
        process_trace_dir=process_trace_dir.resolve(),
        max_workers=race_max_workers,
        limit=None,
        skip_cleaning=False,
        cleaned_data_dir=cleaned_data_dir.resolve(),
        force=True,
    )
    race_result_txt = race_out / "race" / tag / "race_result.txt"
    bundle = {
        "bench_backend": "drb",
        "round_idx": round_idx,
        "bench_id": bench_id,
        "run_id": run_id,
        "task_ids": [int(t.id) for t in tasks],
        "race_model_tag": tag,
        "raw_jsonl": str(raw_jsonl.resolve()),
        "query_subset_jsonl": str(query_subset_path.resolve()),
        "race_result_txt": str(race_result_txt.resolve()),
        "process_trace_dir": str(process_trace_dir.resolve()),
        "merged_skill_dir": str(skill_round.resolve()),
        "drb_bench_root": str(drb_bench_root.resolve()),
    }
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    log_round = (
        export_drb_evaluation_logs(
            bundle_path=bundle_path,
            bench_id=bench_id,
            run_id=run_id,
            round_idx=round_idx,
            log_root=log_root,
            merged_skill_dir=skill_round.resolve(),
        )
        if export_logs
        else root
    )
    eval_section: dict[str, Any] = {
        "phase": "drb_skill_mas_rollout",
        "description": "Skill-MAS per-task usage totals.",
        "model": agent_llm,
        "aggregate_usage": aggregate_usage,
        "per_task": per_task_rows,
        "pricing_reference": pricing_reference(),
    }
    return bundle_path, log_round, eval_section
