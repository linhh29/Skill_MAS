"""Skill-MAS evolution pipeline: multi-trajectory rollout + single-SKILL optimization."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from Skill_MAS.utils.paths import ensure_sys_path

ensure_sys_path(include_vita=True, include_dataset=True)

from ..utils.config import resolve_init_skill_root
from ..evolution.assemble_select import compute_round_score, finalize_best_round, update_round_scoreboard
from ..evolution.bank_optimizer import run_bank_evolution_step_async
from ..evolution.contrastive_reflect import run_local_contrastive_reflection_async
from ..evolution.rollout_multi import run_multi_trajectory_rollout
from ..evolution.agent_patch import apply_skill_mas_patches
from ..evolution.bench_eval import (
    ensure_bcp_jsonl_readable,
    ensure_drb_bench_layout,
    ensure_hlemath_jsonl_readable,
)
from ..utils.run_log_export import export_comprehensive_round_logs, export_round_artifact_index
from ..utils.llm_cost import (
    PricingModelError,
    add_usage_totals,
    build_round_cost_document,
    empty_usage_totals,
    finalize_round_cost_artifacts,
    record_pricing_error,
    validate_evolve_config_models,
)
from ..utils.config import (
    BENCH_SEGMENT_VITABENCH,
    DEFAULT_AGENT_LLM,
    DEFAULT_DRB_RACE_MAX_WORKERS,
    DEFAULT_EVALUATOR_LLM,
    DEFAULT_EVOLVE_ROUNDS,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_STEPS,
    DEFAULT_OPTIMIZER_LLM,
    DEFAULT_RUN_ID,
    DEFAULT_USER_LLM,
    DEFAULT_VAL_SIZE,
    DRB_BENCH_ROOT,
    EVOLVE_K_TRAJECTORIES,
    BROWSECOMP_BENCH_ROOT,
    VITA_SRC,
    allocate_run_id,
    log_root_for,
    resolve_run_id,
    runs_dir,
    skills_evolution_dir,
)
from .dataset_split import load_split_file
from .resume import compute_resume_start
from .model_config_runtime import apply_model_runtime_params
from .task_select import (
    browsecomp_validate_ids,
    drb_validate_ids,
    hlemath_validate_ids,
    vitabench_validate_ids,
)


def _ensure_vita_path() -> None:
    if not VITA_SRC.is_dir():
        raise FileNotFoundError(f"Vita src not found: {VITA_SRC}")


async def _run_optimizer_phases_async(
    *,
    by_task: dict[str, Any],
    round_root: Path,
    cur_skill_root: Path,
    opt_model: str,
    opt_args: dict[str, Any],
    r: int,
    rounds: int,
    bb: str,
    bench_id: str,
    domain: str,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Contrastive reflection then bank SKILL rewrite in one event loop (single asyncio.run from evolve)."""
    contrastive_out = await run_local_contrastive_reflection_async(
        by_task=by_task,
        round_root=round_root,
        optimizer_model=opt_model,
        optimizer_llm_args=opt_args,
    )
    _, contrastive_reports, _contrastive_usage = contrastive_out
    bank_meta, bank_usage = await run_bank_evolution_step_async(
        skill_round_dir=cur_skill_root,
        by_task=by_task,
        round_idx=r,
        total_rounds=rounds,
        optimizer_model=opt_model,
        optimizer_llm_args=opt_args,
        bench_backend=bb,
        bench_id=bench_id,
        domain=domain,
        contrastive_reports=contrastive_reports,
    )
    return contrastive_out, (bank_meta, bank_usage)


def _load_model_defaults() -> dict[str, Any]:
    """Load model defaults from Vita when available."""
    try:
        from vita.config import models as vita_models  # type: ignore[import-not-found]

        if isinstance(vita_models, dict):
            return vita_models
    except Exception:
        pass
    return {}


def init_round_zero(
    bench_id: str,
    run_id: str,
    init_skill_root: Path,
    *,
    bench_backend: str,
    agent_llm: str | None = None,
) -> Path:
    base = skills_evolution_dir(bench_backend, agent_llm) / bench_id / run_id / "round_00"
    if base.exists():
        shutil.rmtree(base)
    src_skill = init_skill_root / "SKILL.md"
    if not src_skill.is_file():
        raise FileNotFoundError(f"Missing init SKILL.md: {src_skill}")
    base.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_skill, base / "SKILL.md")
    (base / "bank_meta.json").write_text(
        json.dumps(
            {
                "init_skill_root": str(init_skill_root.resolve()),
                "version": 2,
                "history": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return base


def _aggregate_eval_sections(eval_sections: list[dict[str, Any]]) -> dict[str, Any]:
    agg = empty_usage_totals()
    for sec in eval_sections:
        if "aggregate_usage_model_pricing_json" in sec:
            add_usage_totals(agg, sec.get("aggregate_usage_model_pricing_json"))
        elif "aggregate_usage" in sec:
            add_usage_totals(agg, sec.get("aggregate_usage"))
    return {
        "phase": "multi_trajectory_rollout",
        "description": "Aggregated usage across all trajectories in this round.",
        "aggregate_usage_model_pricing_json": agg,
        "num_sections": len(eval_sections),
    }


def _carry_to_next_round(
    *,
    bench_backend: str,
    bench_id: str,
    run_id: str,
    round_idx: int,
    agent_llm: str | None = None,
) -> None:
    cur = skills_evolution_dir(bench_backend, agent_llm) / bench_id / run_id / f"round_{round_idx:02d}"
    nxt = skills_evolution_dir(bench_backend, agent_llm) / bench_id / run_id / f"round_{round_idx + 1:02d}"
    if nxt.exists():
        shutil.rmtree(nxt)
    nxt.mkdir(parents=True, exist_ok=True)
    for fn in ("SKILL.md", "bank_meta.json"):
        p = cur / fn
        if p.is_file():
            shutil.copy2(p, nxt / fn)


def evolve(
    *,
    bench_id: str,
    domain: str,
    task_set_name: str = "",
    val_size: int = DEFAULT_VAL_SIZE,
    rounds: int = DEFAULT_EVOLVE_ROUNDS,
    init_skill_root: Path | None = None,
    split_file: Path | None = None,
    split_seed: int = 0,
    agent_llm: str = DEFAULT_AGENT_LLM,
    user_llm: str = DEFAULT_USER_LLM,
    evaluator_llm: str = DEFAULT_EVALUATOR_LLM,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    language: str = DEFAULT_LANGUAGE,
    optimizer_llm: str = DEFAULT_OPTIMIZER_LLM,
    run_id: str = DEFAULT_RUN_ID,
    fresh: bool = False,
    bench_backend: str = "vitabench",
    drb_bench_root: Path | None = None,
    drb_race_max_workers: int = DEFAULT_DRB_RACE_MAX_WORKERS,
    k_trajectories: int = EVOLVE_K_TRAJECTORIES,
    hlemath_jsonl: Path | None = None,
    bcp_jsonl: Path | None = None,
    jsonl_path: Path | None = None,
    max_problems: int = 0,
    bcp_judge_llm: str | None = None,
    bcp_judge_timeout_s: float = 120.0,
    bcp_index_path: Path | str | None = None,
    bcp_retrieval_topk: int = 5,
    bcp_doc_max_tokens: int = 512,
    bcp_max_retrieval_rounds: int = 10,
) -> str:
    bb = (bench_backend or "vitabench").strip().lower()
    if jsonl_path is None:
        raise ValueError("--jsonl is required for all backends.")
    unified_jsonl = Path(jsonl_path).resolve()

    if bb == "vitabench":
        _ensure_vita_path()
        # Force Vita to use caller-provided validation JSON/JSONL for cross-domain task loading.
        os.environ["VITA_CROSS_DOMAIN_TASKS_PATH"] = str(unified_jsonl)
        apply_skill_mas_patches()
    elif bb == "drb":
        ensure_drb_bench_layout()
    elif bb == "hlemath":
        ensure_hlemath_jsonl_readable(unified_jsonl)
    elif bb == "bcp":
        ensure_bcp_jsonl_readable(unified_jsonl)
        if not BROWSECOMP_BENCH_ROOT.is_dir():
            raise FileNotFoundError(f"BrowseComp-Plus repo not found: {BROWSECOMP_BENCH_ROOT}")
    else:
        raise ValueError(
            f"Unknown bench_backend={bench_backend!r}; "
            "use 'vitabench', 'drb', 'hlemath', or 'bcp'."
        )

    models = _load_model_defaults()

    run_id = resolve_run_id(bench_id, run_id, fresh=fresh, bench_backend=bb, agent_llm=agent_llm)
    init_root = resolve_init_skill_root(init_skill_root)

    resolved_hlemath_jsonl = unified_jsonl
    resolved_bcp_jsonl = unified_jsonl
    resolved_drb_query_jsonl = unified_jsonl
    resolved_vita_validate_file = unified_jsonl

    if split_file is not None:
        sp = load_split_file(Path(split_file))
        task_ids = [str(x) for x in (sp.get("ids") or sp.get("val_ids") or [])]
        if not task_ids:
            raise RuntimeError(f"No ids/val_ids in split file {split_file}")
        if bb == "hlemath":
            jf = sp.get("jsonl_file")
            if jf:
                resolved_hlemath_jsonl = Path(str(jf)).resolve()
        elif bb == "bcp":
            jf = sp.get("jsonl_file")
            if jf:
                resolved_bcp_jsonl = Path(str(jf)).resolve()
    elif bb == "vitabench":
        task_ids = vitabench_validate_ids(resolved_vita_validate_file)
        if not task_ids:
            raise RuntimeError("No task ids in dataset/vitabench/data/vita_validate.json")
    elif bb == "drb":
        task_ids = drb_validate_ids(resolved_drb_query_jsonl)
        if not task_ids:
            raise RuntimeError("No task ids in deep_research_bench/data/drb_validate.jsonl")
    elif bb == "hlemath":
        task_ids = hlemath_validate_ids(resolved_hlemath_jsonl)
        if not task_ids:
            raise RuntimeError(f"No task ids in {resolved_hlemath_jsonl}")
    elif bb == "bcp":
        task_ids = browsecomp_validate_ids(resolved_bcp_jsonl)
        if not task_ids:
            raise RuntimeError(f"No task ids in {resolved_bcp_jsonl}")
    else:
        raise RuntimeError(f"Unhandled bench_backend={bb!r} for split loading")

    if int(max_problems or 0) > 0:
        task_ids = task_ids[: int(max_problems)]
        if not task_ids:
            raise RuntimeError(f"No task ids selected after max_problems={max_problems}.")

    task_ids_int = [int(x) for x in task_ids] if bb == "drb" else []

    start_r, need_init, status = compute_resume_start(
        bench_id, run_id, rounds, bench_backend=bb, agent_llm=agent_llm
    )
    if status == "complete":
        print(f"Evolve already complete for bench_id={bench_id} run_id={run_id} (all {rounds} eval rounds done).")
        return run_id

    if need_init:
        init_round_zero(
            bench_id,
            run_id,
            init_skill_root=init_root,
            bench_backend=bb,
            agent_llm=agent_llm,
        )
        print(f"Starting new run: bench_id={bench_id} run_id={run_id} init_skill={init_root}")
    else:
        print(f"Resuming bench_id={bench_id} run_id={run_id} from eval round index r={start_r}")

    opt_model = optimizer_llm or agent_llm
    opt_args: dict[str, Any] = apply_model_runtime_params(opt_model, dict(models.get(opt_model, {})))

    llm_a = apply_model_runtime_params(agent_llm, dict(models.get(agent_llm, {})))
    llm_u = apply_model_runtime_params(user_llm, dict(models.get(user_llm, {})))
    llm_e = apply_model_runtime_params(evaluator_llm, dict(models.get(evaluator_llm, {})))

    try:
        validate_evolve_config_models(
            agent_llm=agent_llm,
            user_llm=user_llm,
            evaluator_llm=evaluator_llm,
            optimizer_llm=opt_model,
            judge_llm=bcp_judge_llm if bb == "bcp" else None,
        )
    except PricingModelError as e:
        record_pricing_error(
            runs_dir=runs_dir(bb, agent_llm),
            bench_id=bench_id,
            run_id=run_id,
            exc=e,
            round_idx=None,
            log_round=None,
            extra={"phase": "config_validation"},
        )
        raise

    for r in range(start_r, rounds):
        log_round = log_root_for(bb, agent_llm) / bench_id / run_id / f"round_{r:02d}"
        log_round.mkdir(parents=True, exist_ok=True)
        opt_sections: list[dict[str, Any]] = []
        eval_section: dict[str, Any] = {}
        try:
            if bb == "vitabench":
                current_task_ids = task_ids
            elif bb == "drb":
                current_task_ids = [str(x) for x in task_ids_int]
            else:
                current_task_ids = task_ids

            by_task, eval_sections, round_root = run_multi_trajectory_rollout(
                bench_backend=bb,
                bench_id=bench_id,
                run_id=run_id,
                round_idx=r,
                task_ids=current_task_ids,
                k_trajectories=k_trajectories,
                task_set_name=task_set_name,
                domain=domain,
                agent_llm=agent_llm,
                user_llm=user_llm,
                evaluator_llm=evaluator_llm,
                llm_args_agent=llm_a,
                llm_args_user=llm_u,
                llm_args_evaluator=llm_e,
                max_steps=max_steps,
                max_concurrency=max_concurrency,
                language=language,
                runs_dir=runs_dir(bb, agent_llm),
                log_root=log_root_for(bb, agent_llm),
                drb_bench_root=Path(drb_bench_root) if drb_bench_root else DRB_BENCH_ROOT,
                drb_query_jsonl=resolved_drb_query_jsonl,
                drb_race_max_workers=drb_race_max_workers,
                hlemath_jsonl=resolved_hlemath_jsonl,
                bcp_jsonl=resolved_bcp_jsonl,
                bcp_judge_llm=bcp_judge_llm,
                bcp_judge_timeout_s=bcp_judge_timeout_s,
                bcp_index_path=bcp_index_path,
                bcp_retrieval_topk=bcp_retrieval_topk,
                bcp_doc_max_tokens=bcp_doc_max_tokens,
                bcp_max_retrieval_rounds=bcp_max_retrieval_rounds,
            )
            eval_section = _aggregate_eval_sections(eval_sections)

            cur_skill_root = skills_evolution_dir(bb, agent_llm) / bench_id / run_id / f"round_{r:02d}"
            contrastive_out, bank_pair = asyncio.run(
                _run_optimizer_phases_async(
                    by_task=by_task,
                    round_root=round_root,
                    cur_skill_root=cur_skill_root,
                    opt_model=opt_model,
                    opt_args=opt_args,
                    r=r,
                    rounds=rounds,
                    bb=bb,
                    bench_id=bench_id,
                    domain=domain,
                )
            )
            _, _, contrastive_usage = contrastive_out
            bank_meta, bank_usage = bank_pair
            opt_sections.extend(contrastive_usage)
            opt_sections.extend(bank_usage)

            round_score = compute_round_score(by_task)
            aggregate = {
                "num_tasks": len(by_task),
                "k_trajectories": int(max(1, k_trajectories)),
                "bank_evolution": bank_meta,
            }
            score_payload = update_round_scoreboard(
                runs_root=runs_dir(bb, agent_llm) / bench_id / run_id,
                round_idx=r,
                round_score=round_score,
                skill_path=cur_skill_root / "SKILL.md",
                skill_round_path=cur_skill_root,
            )
            summary = {
                "round_idx": r,
                "avg_reward": round_score,
                "aggregate": aggregate,
                "scoreboard": score_payload,
                "bank_meta": bank_meta,
                "raw_path": str(round_root),
            }
            export_round_artifact_index(
                log_round_dir=log_round,
                round_idx=r,
                trajectories_dir=round_root / "trajectories",
                aspects_dir=round_root / "aspects",
                contrastive_dir=round_root / "contrastive",
                skill_round_dir=cur_skill_root,
                bank_meta_file=cur_skill_root / "bank_meta.json",
                knee_images_dir=cur_skill_root / "knee_images",
            )
            export_comprehensive_round_logs(
                log_round_dir=log_round,
                round_root=round_root,
                skill_round_dir=cur_skill_root,
                bench_backend=bb,
                by_task=by_task,
            )
            if r < rounds - 1:
                _carry_to_next_round(
                    bench_backend=bb,
                    bench_id=bench_id,
                    run_id=run_id,
                    round_idx=r,
                    agent_llm=agent_llm,
                )

            round_cost = build_round_cost_document(
                bench_backend=bb,
                bench_id=bench_id,
                run_id=run_id,
                round_idx=r,
                eval_section=eval_section,
                optimizer_sections=opt_sections,
                extra_notes=None,
            )
            summary["llm_cost"] = {
                "round_total_estimated_cost_usd": round_cost["round_total_estimated_cost_usd"],
                "cost_detail_filename": f"evolve_llm_cost_r{r:02d}.json",
                "cumulative_summary_filename": "evolve_llm_cost_cumulative.json",
                "pricing_note": "USD estimates from Skill_MAS/skill_mas/model_config.json (per 1M prompt/output tokens).",
            }

            out_meta = runs_dir(bb, agent_llm) / bench_id / run_id / f"summary_r{r:02d}.json"
            out_meta.parent.mkdir(parents=True, exist_ok=True)
            out_meta.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            (log_round / "summary_metrics.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            finalize_round_cost_artifacts(
                runs_dir=runs_dir(bb, agent_llm),
                log_round=log_round,
                bench_id=bench_id,
                run_id=run_id,
                round_idx=r,
                payload=round_cost,
            )
            print(
                f"[Skill_MAS] round {r} LLM cost estimate (USD, Skill_MAS/skill_mas/model_config.json): "
                f"{round_cost['round_total_estimated_cost_usd']}",
                flush=True,
            )
        except PricingModelError as e:
            record_pricing_error(
                runs_dir=runs_dir(bb, agent_llm),
                bench_id=bench_id,
                run_id=run_id,
                exc=e,
                round_idx=r,
                log_round=log_round,
                extra={"phase": "evolve_round"},
            )
            raise

    finalize_best_round(runs_dir(bb, agent_llm) / bench_id / run_id)
    return run_id


def snapshot_baseline_only(
    bench_id: str,
    init_skill_root: Path | None,
    domain: str,
    run_id: str = DEFAULT_RUN_ID,
    bench_backend: str = BENCH_SEGMENT_VITABENCH,
    agent_llm: str | None = None,
) -> str:
    _ = domain
    bb = (bench_backend or BENCH_SEGMENT_VITABENCH).strip().lower()
    if bb == "drb":
        if not DRB_BENCH_ROOT.is_dir():
            raise FileNotFoundError(
                f"Expected DeepResearchBench at {DRB_BENCH_ROOT} (clone deep_research_bench next to Ant)."
            )
    elif bb == "vitabench":
        _ensure_vita_path()
    elif bb == "hlemath":
        pass
    elif bb == "bcp":
        if not BROWSECOMP_BENCH_ROOT.is_dir():
            raise FileNotFoundError(f"BrowseComp-Plus not found: {BROWSECOMP_BENCH_ROOT}")
    else:
        _ensure_vita_path()
    run_id = allocate_run_id(bench_id, run_id, bench_backend=bb, agent_llm=agent_llm)
    root = resolve_init_skill_root(init_skill_root)
    init_round_zero(bench_id, run_id, init_skill_root=root, bench_backend=bb, agent_llm=agent_llm)
    return run_id

