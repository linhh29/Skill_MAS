"""Step 2: trajectory reflection synthesis from full n*k trajectories."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from Skill_MAS.skill_mas.openai_async_client import (
    AsyncOpenAIClient,
    chat_completion_extras_from_optimizer_llm_args,
)

from ..utils.config import (
    EVOLVE_MAX_REFLECTION_CASES_PER_ROUND,
    ROUND_CONTRASTIVE_DIRNAME,
    ROUND_PATCH_POOL_FILENAME,
    SCHEMA_CONTRASTIVE_REPORT,
    SCHEMA_DOMAIN_PATCH,
)
from ..utils.evolution_trajectory_sanitize import sanitize_raw_result_for_evolution
from ..utils.llm_cost import optimizer_call_report
from .elbow_selection import compute_reflection_task_selection
from .prompts import (
    SYS_CONTRASTIVE_PHASE1,
    SYS_CONTRASTIVE_PHASE2,
    user_contrastive_phase1,
    user_contrastive_phase2,
)
from .schemas import DomainPatch, TrajectoryRecord


def _contrastive_json_object_response_format() -> dict[str, Any]:
    """
    OpenAI-compatible JSON object mode (DeepSeek / DashScope: use json_object, not json_schema).
    Prompts in ``prompts.py`` already require the word 'json' and include field examples.
    """
    return {"type": "json_object"}


def _strip_json_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _record_payload(rec: TrajectoryRecord) -> dict[str, Any]:
    raw_path = Path(rec.raw_result_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing raw_result_path for trajectory payload: {raw_path}")
    try:
        raw_result: Any = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse raw trajectory json: {raw_path}") from e
    if rec.bench_backend == "vitabench" and isinstance(raw_result, dict):
        sims = raw_result.get("simulations")
        if isinstance(sims, list):
            task_id_s = str(rec.task_id)
            scoped = [x for x in sims if isinstance(x, dict) and str(x.get("task_id")) == task_id_s]
            raw_result = dict(raw_result)
            raw_result["simulations"] = scoped
    raw_result = sanitize_raw_result_for_evolution(rec, raw_result)
    return {
        "task_id": rec.task_id,
        "trajectory_tag": rec.trajectory_tag,
        "trajectory_idx": rec.trajectory_idx,
        "score": rec.score,
        "score_source": rec.score_source,
        "log_path": rec.log_path,
        "raw_result_path": rec.raw_result_path,
        "raw_result": raw_result,
        "phase_snapshots": [
            {"phase": p.phase, "instruction": p.instruction, "output_preview": p.output_preview}
            for p in rec.phase_snapshots
        ],
    }


def _safe_task_filename(task_id: str) -> str:
    return str(task_id).replace("/", "_").replace("\\", "_")


def _select_reflection_task_ids(
    *,
    by_task: dict[str, list[TrajectoryRecord]],
    max_cases: int,
) -> list[str]:
    """Pick task ids via per-task priority (uncertainty+difficulty) then second-diff elbow."""
    task_rows: list[tuple[str, list[float]]] = []
    for task_id, rows in sorted(by_task.items(), key=lambda kv: str(kv[0])):
        if len(rows) < 2:
            # Contrastive step compares rollouts; need at least two samples.
            continue
        task_rows.append((str(task_id), [float(r.score) for r in rows]))
    selected, _ = compute_reflection_task_selection(
        task_rows,
        max_reflection_cases=max_cases,
        sensitivity=1.0,
    )
    return selected


async def run_local_contrastive_reflection_async(
    *,
    by_task: dict[str, list[TrajectoryRecord]],
    round_root: Path,
    optimizer_model: str,
    optimizer_llm_args: dict[str, Any],
) -> tuple[list[DomainPatch], list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate candidate patch pool from full n*k trajectories (async OpenAI client).

    Two-phase Step2:
    - Phase 1: one LLM call per selected task with ALL rollouts for that task (parallelized).
    - Phase 2: one LLM call; each task block = original task text + **full** Phase-1 JSON (all schema fields; no mas_code).
    Returns ``reports`` for Stage3 (bank): Phase2 only — ``phase2_response``, ``cross_sample_synthesis``, ``meta_analysis``
    (no Phase-1 payloads; Phase-1 remains on disk under ``phase1_*.json`` / ``summary.json``).
    """
    contrastive_dir = round_root / ROUND_CONTRASTIVE_DIRNAME
    contrastive_dir.mkdir(parents=True, exist_ok=True)
    patch_pool_path = round_root / ROUND_PATCH_POOL_FILENAME

    patches: list[DomainPatch] = []
    reports: list[dict[str, Any]] = []
    usage_reports: list[dict[str, Any]] = []
    max_cases = int(max(1, EVOLVE_MAX_REFLECTION_CASES_PER_ROUND))
    selected_task_ids = _select_reflection_task_ids(
        by_task=by_task,
        max_cases=max_cases,
    )
    selection_mode = "second_diff_elbow"
    if not selected_task_ids:
        patch_pool_path.write_text("", encoding="utf-8")
        return patches, reports, usage_reports

    client = AsyncOpenAIClient(model=optimizer_model)
    extras = chat_completion_extras_from_optimizer_llm_args(optimizer_llm_args)

    async def _phase1_one(tid: str) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]], int, int]:
        rows = sorted(by_task.get(tid, []), key=lambda x: x.score)
        traj_payloads = [_record_payload(r) for r in rows]
        wrap = {"task_id": tid, "num_rollouts": len(rows), "trajectories": traj_payloads}
        payload_str = json.dumps(wrap, ensure_ascii=False, indent=2)
        ic = len(payload_str)
        et = max(1, ic // 4)
        user = user_contrastive_phase1(
            task_id=tid,
            num_rollouts=len(rows),
            trajectories_payload=payload_str,
            input_char_count=ic,
            estimated_input_tokens=et,
        )
        text, usage = await client.generate(
            system_prompt=SYS_CONTRASTIVE_PHASE1,
            user_prompt=user,
            **extras,
            response_format=_contrastive_json_object_response_format(),
        )
        obj = json.loads(_strip_json_fence(text))
        return tid, obj, dict(usage or {}), traj_payloads, ic, et

    try:
        phase1_out = await asyncio.gather(*[_phase1_one(tid) for tid in selected_task_ids])

        phase1_by_task: dict[str, dict[str, Any]] = {}
        traj_payload_by_task: dict[str, list[dict[str, Any]]] = {}
        phase1_chars: dict[str, dict[str, int]] = {}

        for tid, obj, usage, traj_payloads, ic, et in phase1_out:
            usage_reports.append(
                optimizer_call_report(
                    phase="contrastive_reflection_phase1",
                    model=optimizer_model,
                    usage=usage,
                )
            )
            phase1_by_task[tid] = obj
            traj_payload_by_task[tid] = traj_payloads
            phase1_chars[tid] = {"input_char_count": ic, "estimated_input_tokens": et}
            fn = _safe_task_filename(tid)
            (contrastive_dir / f"phase1_{fn}.json").write_text(
                json.dumps(obj, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        phase2_blocks: list[str] = []
        for tid in selected_task_ids:
            obj = phase1_by_task[tid]
            traj_payloads = traj_payload_by_task[tid]
            rep_payload = traj_payloads[-1]
            rr = rep_payload.get("raw_result")
            if isinstance(rr, dict):
                tdesc = str(rr.get("task_description") or "").strip()
            else:
                tdesc = ""
            phase1_json_str = json.dumps(obj, ensure_ascii=False, indent=2)
            phase2_blocks.append(
                f"=== TASK task_id={tid} ===\n"
                f"[Original task / instruction]\n{tdesc}\n\n"
                f"[Phase-1 complete structured output (JSON; all schema fields)]\n"
                f"{phase1_json_str}\n"
                f"---\n"
            )

        phase2_str = "\n".join(phase2_blocks)
        p2_ic = len(phase2_str)
        p2_et = max(1, p2_ic // 4)
        user2 = user_contrastive_phase2(
            selected_task_ids=selected_task_ids,
            per_task_blocks=phase2_str,
            input_char_count=p2_ic,
            estimated_input_tokens=p2_et,
        )
        text2, usage2 = await client.generate(
            system_prompt=SYS_CONTRASTIVE_PHASE2,
            user_prompt=user2,
            **extras,
            response_format=_contrastive_json_object_response_format(),
        )
        usage_reports.append(
            optimizer_call_report(
                phase="contrastive_reflection_phase2",
                model=optimizer_model,
                usage=dict(usage2 or {}),
            )
        )
        payload2 = json.loads(_strip_json_fence(text2))
        synthesis = payload2.get("cross_sample_synthesis") or {}
        meta = payload2.get("meta_analysis") or {}

        (contrastive_dir / "phase2_cross.json").write_text(
            json.dumps(payload2, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for tid in selected_task_ids:
            item = phase1_by_task[tid]
            task_id = str(item.get("task_id") or "").strip()
            if not task_id:
                raise ValueError("Phase1 response missing task_id")
            if task_id != str(tid).strip():
                raise ValueError(
                    f"Phase1 task_id mismatch: expected {tid!r} (this rollout batch), got {task_id!r}"
                )
            rows = sorted(by_task.get(task_id, []), key=lambda x: x.score)
            if len(rows) < 2:
                raise ValueError(f"task_id={task_id} has <2 trajectories in by_task")
            low = rows[0]
            high = rows[-1]
            cp = item.get("candidate_patch") or {}
            phase_s = str(cp.get("target_phase") or cp.get("phase") or "Phase 1").strip()
            constraint_s = str(cp.get("constraint_rule") or cp.get("constraint") or "").strip()
            impl = str(cp.get("implementation_mechanism") or "").strip()
            impact = str(cp.get("expected_impact") or "").strip()
            rationale_s = str(cp.get("rationale") or "").strip()
            rationale_parts = [p for p in (impl, impact) if p]
            rationale_final = "\n".join(rationale_parts) if rationale_parts else rationale_s
            patch = DomainPatch(
                schema=SCHEMA_DOMAIN_PATCH,
                task_id=task_id,
                phase=phase_s or "Phase 1",
                constraint=constraint_s,
                rationale=rationale_final,
                source_gap=float(high.score) - float(low.score),
                source_high_traj=high.trajectory_tag,
                source_low_traj=low.trajectory_tag,
            )
            if not patch.constraint:
                raise ValueError(
                    f"Phase1 returned empty candidate_patch.constraint_rule for task_id={task_id}"
                )
            rollout_refs = [
                {
                    "trajectory_tag": r.trajectory_tag,
                    "trajectory_idx": r.trajectory_idx,
                    "score": r.score,
                    "score_source": r.score_source,
                    "log_path": r.log_path,
                    "raw_result_path": r.raw_result_path,
                }
                for r in rows
            ]
            task_audit = {
                "task_id": task_id,
                "rollout_refs": rollout_refs,
                "candidate_patch": patch.to_dict(),
            }
            patches.append(patch)
            fn = _safe_task_filename(task_id)
            (contrastive_dir / f"{fn}.json").write_text(
                json.dumps(task_audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        reports = [
            {
                "schema": SCHEMA_CONTRASTIVE_REPORT,
                "task_id": "__step2_phase2_aggregate__",
                "reflection_mode": "two_phase",
                "selection_mode": selection_mode,
                "selected_task_ids": selected_task_ids,
                "cross_sample_synthesis": synthesis,
                "meta_analysis": meta,
                "phase2_response": payload2,
            }
        ]

        (contrastive_dir / "summary.json").write_text(
            json.dumps(
                {
                    "reflection_mode": "two_phase",
                    "selected_task_ids": selected_task_ids,
                    "phase1_per_task_input": phase1_chars,
                    "phase2": {
                        "input_char_count": p2_ic,
                        "estimated_input_tokens": p2_et,
                    },
                    "cross_sample_synthesis": synthesis,
                    "meta_analysis": meta,
                    "phase1_narrative_summaries": {
                        tid: str(phase1_by_task[tid].get("narrative_summary") or "")
                        for tid in selected_task_ids
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        raise RuntimeError(
            f"Two-phase contrastive reflection failed for task_ids={selected_task_ids}"
        ) from e

    patch_pool_path.write_text(
        "\n".join(json.dumps(p.to_dict(), ensure_ascii=False) for p in patches) + ("\n" if patches else ""),
        encoding="utf-8",
    )
    return patches, reports, usage_reports


def run_local_contrastive_reflection(
    *,
    by_task: dict[str, list[TrajectoryRecord]],
    round_root: Path,
    optimizer_model: str,
    optimizer_llm_args: dict[str, Any],
) -> tuple[list[DomainPatch], list[dict[str, Any]], list[dict[str, Any]]]:
    """Sync wrapper for callers outside an async event loop."""
    return asyncio.run(
        run_local_contrastive_reflection_async(
            by_task=by_task,
            round_root=round_root,
            optimizer_model=optimizer_model,
            optimizer_llm_args=optimizer_llm_args,
        )
    )
