"""LLM-based optimization for single-file 3-stage SKILL.md."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vita.agent.skill_mas_runtime import parse_skill_file

from Skill_MAS.skill_mas.openai_async_client import (
    AsyncOpenAIClient,
    chat_completion_extras_from_optimizer_llm_args,
)

from ..evolution.schemas import TrajectoryRecord
from ..utils.config import EVOLVE_MAX_REFLECTION_CASES_PER_ROUND
from ..utils.llm_cost import optimizer_call_report
from .elbow_selection import (
    compute_priority_scores,
    compute_reflection_task_selection,
    second_diff_elbow_detail,
)
from .prompts import SYS_SKILL_WRITER, user_bank_optimizer_prompt


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    m = re.match(r"^```(?:markdown|md)?\s*\n([\s\S]*?)\n```\s*$", t)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def _write_knee_artifacts(skill_round_dir: Path, by_task: dict[str, list[TrajectoryRecord]], round_idx: int) -> None:
    task_rows: list[tuple[str, list[float]]] = []
    for task_id, rows in sorted(by_task.items(), key=lambda kv: str(kv[0])):
        if not rows:
            continue
        task_rows.append((str(task_id), [float(r.score) for r in rows]))

    ranked: list[tuple[str, float]] = []
    if task_rows:
        samples_scores = [s for _, s in task_rows]
        priorities = compute_priority_scores(samples_scores)
        ranked = sorted(
            zip([t for t, _ in task_rows], priorities),
            key=lambda x: x[1],
            reverse=True,
        )

    knee_dir = skill_round_dir / "knee_images"
    knee_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "round_idx": int(round_idx),
        "method": "second_diff_elbow",
        "priority_metric": "uncertainty_std_mean_difficulty_blend",
        "num_tasks": len(ranked),
        "priorities_desc": [{"task_id": t, "priority": float(p)} for t, p in ranked],
    }
    if ranked:
        values = [p for _, p in ranked]
        detail = second_diff_elbow_detail(values, sensitivity=1.0)
        payload.update({k: v for k, v in detail.items() if k != "n"})
        sdi = detail.get("second_diff_argmax_index")
        if sdi is not None:
            kidx = min(int(sdi) + 1, len(ranked) - 1)
        else:
            kidx = min(max(int(detail.get("selected_count") or 1) - 1, 0), len(ranked) - 1)
        payload["knee_index"] = int(kidx)
        payload["knee_task_id"] = ranked[kidx][0]
        pri_at = float(ranked[kidx][1])
        payload["knee_priority"] = pri_at
        payload["knee_gap"] = pri_at
    else:
        payload["knee_index"] = None
        payload["knee_task_id"] = None
        payload["knee_priority"] = None
        payload["knee_gap"] = None
    (knee_dir / f"knee_r{round_idx:02d}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _, priority_report = compute_reflection_task_selection(
        task_rows,
        max_reflection_cases=int(max(1, EVOLVE_MAX_REFLECTION_CASES_PER_ROUND)),
        sensitivity=1.0,
    )
    priority_payload = {"round_idx": int(round_idx), **priority_report}
    (knee_dir / f"priority_selection_r{round_idx:02d}.json").write_text(
        json.dumps(priority_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pdf_path = knee_dir / f"knee_r{round_idx:02d}.pdf"
    fig, ax = plt.subplots(figsize=(10, 4))
    if not ranked:
        ax.text(0.05, 0.5, "No tasks with trajectory scores this round", fontsize=12)
        ax.set_axis_off()
    else:
        ys = [p for _, p in ranked]
        xs = list(range(len(ys)))
        ax.plot(xs, ys, color="#2563eb", linewidth=2)
        kidx = int(payload["knee_index"]) if payload["knee_index"] is not None else 0
        ax.scatter([kidx], [ys[kidx]], color="#dc2626", s=35, zorder=3)
        ax.set_xlabel("Task rank (descending priority)")
        ax.set_ylabel("Task priority (uncertainty + difficulty)")
        ax.set_title(
            f"round={round_idx} second_diff_elbow rank={kidx} task={payload['knee_task_id']} "
            f"(selected_count={payload.get('selected_count', '')})"
        )
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)


async def run_bank_evolution_step_async(
    *,
    skill_round_dir: Path,
    by_task: dict[str, list[TrajectoryRecord]],
    round_idx: int,
    total_rounds: int,
    optimizer_model: str,
    optimizer_llm_args: dict[str, Any],
    bench_backend: str,
    bench_id: str,
    domain: str,
    contrastive_reports: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Optimize round skill file and overwrite ``round_xx/SKILL.md`` (async OpenAI client)."""
    usage_reports: list[dict[str, Any]] = []
    selected_task_ids = [str(tid) for tid in list(by_task.keys())[:16]]
    bench_hint = f"backend={bench_backend} bench_id={bench_id} domain={domain}"
    current_skill = skill_round_dir / "SKILL.md"
    if not current_skill.is_file():
        raise FileNotFoundError(f"Missing round skill file: {current_skill}")
    if contrastive_reports is not None and not isinstance(contrastive_reports, list):
        raise TypeError("contrastive_reports must be a list[dict] or None")
    if contrastive_reports:
        for i, item in enumerate(contrastive_reports):
            if not isinstance(item, dict):
                raise TypeError(f"contrastive_reports[{i}] must be a dict")
            if not str(item.get("task_id") or "").strip():
                raise ValueError(f"contrastive_reports[{i}] missing non-empty task_id")
            if "cross_sample_synthesis" not in item:
                raise ValueError(
                    f"contrastive_reports[{i}] missing required Step2 field: cross_sample_synthesis"
                )
    user = user_bank_optimizer_prompt(
        round_idx=round_idx,
        total_rounds=total_rounds,
        bench_hint=bench_hint,
        current_skill_md=current_skill.read_text(encoding="utf-8"),
        step2_reflection_summary=json.dumps(
            {
                "num_reports": len(contrastive_reports or []),
                "reports": contrastive_reports or [],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    client = AsyncOpenAIClient(model=optimizer_model)
    extras = chat_completion_extras_from_optimizer_llm_args(optimizer_llm_args)
    md_text, usage = await client.generate(
        system_prompt=SYS_SKILL_WRITER,
        user_prompt=user,
        **extras,
    )
    usage_reports.append(
        optimizer_call_report(
            phase="bank_optimizer_three_stage",
            model=optimizer_model,
            usage=dict(usage or {}),
        )
    )
    md = _strip_fences(md_text or "")
    tmp_md = skill_round_dir / "_optimizer_tmp_skill.md"
    tmp_md.write_text(md, encoding="utf-8")
    spec = parse_skill_file(tmp_md)
    if spec.validation_issues:
        raise ValueError(f"Invalid generated SKILL.md: {spec.validation_issues}")
    current_skill.write_text(md, encoding="utf-8")

    meta_path = skill_round_dir / "bank_meta.json"
    step_meta: dict[str, Any] = {
        "round_idx": round_idx,
        "updated_skill_path": str(current_skill),
        "updated_skill_name": str(spec.name),
        "selected_task_ids": selected_task_ids,
    }
    meta_existing: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta_existing = loaded
        except Exception:
            meta_existing = {}
    history: list[Any] = list(meta_existing.get("history", []))
    history.append(step_meta)
    meta_out = dict(meta_existing)
    meta_out["history"] = history
    meta_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_knee_artifacts(skill_round_dir, by_task, round_idx)

    return step_meta, usage_reports


def run_bank_evolution_step(
    *,
    skill_round_dir: Path,
    by_task: dict[str, list[TrajectoryRecord]],
    round_idx: int,
    total_rounds: int,
    optimizer_model: str,
    optimizer_llm_args: dict[str, Any],
    bench_backend: str,
    bench_id: str,
    domain: str,
    contrastive_reports: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Sync wrapper for callers outside an async event loop."""
    return asyncio.run(
        run_bank_evolution_step_async(
            skill_round_dir=skill_round_dir,
            by_task=by_task,
            round_idx=round_idx,
            total_rounds=total_rounds,
            optimizer_model=optimizer_model,
            optimizer_llm_args=optimizer_llm_args,
            bench_backend=bench_backend,
            bench_id=bench_id,
            domain=domain,
            contrastive_reports=contrastive_reports,
        )
    )
