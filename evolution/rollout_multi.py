"""Step 1: multi-trajectory rollout with phase-aspect snapshots."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from Skill_MAS.utils.paths import ensure_sys_path

ensure_sys_path(include_vita=True, include_dataset=True)

from ..core.model_config_runtime import model_runtime_params
from ..utils.config import (
    DEFAULT_LANGUAGE,
    EVOLVE_K_TRAJECTORIES,
    ROUND_ASPECTS_DIRNAME,
    ROUND_BENCH_ROLLOUTS_DIRNAME,
    ROUND_TRAJECTORIES_DIRNAME,
    SCHEMA_TRAJECTORY_RECORD,
    merged_workspaces_dir,
    skills_evolution_dir,
)
from .bench_eval import (
    _drb_cached_rollout_complete,
    _sanitize_model_tag,
    reset_drb_bench_rollout_resume_counter,
    run_bcp_evaluation_round,
    run_drb_evaluation_round,
    run_hlemath_evaluation_round,
    take_drb_bench_rollout_resume_count,
)
from ..utils.llm_cost import vita_rollout_cost_report
from ..utils.traces import summarize_drb_round, summarize_run
from .schemas import PhaseSnapshot, TrajectoryRecord


def _round_root(*, runs_dir: Path, bench_id: str, run_id: str, round_idx: int) -> Path:
    return runs_dir / bench_id / run_id / f"round_{round_idx:02d}"


def _extract_vita_phase_snapshots(results_path: Path, task_id: str) -> list[PhaseSnapshot]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    sims = list(data.get("simulations") or [])
    if not sims:
        return []
    sim = None
    for row in sims:
        if str(row.get("task_id")) == str(task_id):
            sim = row
            break
    if sim is None:
        return []
    msgs = list(sim.get("messages") or [])
    skill_mas: dict[str, Any] = {}
    for m in reversed(msgs):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        raw = m.get("raw_data")
        if isinstance(raw, dict) and isinstance(raw.get("skill_mas"), dict):
            skill_mas = dict(raw["skill_mas"])
            break

    phases: list[PhaseSnapshot] = []
    # Prefer explicit build-stage traces when present.
    stages = skill_mas.get("build_stage_traces")
    if isinstance(stages, list) and stages:
        for i, st in enumerate(stages, 1):
            if not isinstance(st, dict):
                continue
            phase_name = str(st.get("stage_name") or st.get("phase") or f"Phase {i}")
            instruction = str(st.get("prompt") or st.get("instruction") or "")[:2000]
            output_preview = (
                str(st.get("parsed_json") or st.get("raw_response") or st.get("output") or st.get("result") or "")[
                    :2000
                ]
            )
            phases.append(
                PhaseSnapshot(
                    phase=phase_name,
                    instruction=instruction,
                    output_preview=output_preview,
                )
            )
    return phases


def _extract_drb_phase_snapshots(bundle_path: Path, task_id: str) -> list[PhaseSnapshot]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    ptd = Path(bundle.get("process_trace_dir") or "")
    trace_file = ptd / f"{int(task_id)}.json"
    if not trace_file.is_file():
        return []
    tr = json.loads(trace_file.read_text(encoding="utf-8"))
    wf = tr.get("build_stage_traces")
    out: list[PhaseSnapshot] = []
    if isinstance(wf, list):
        for i, st in enumerate(wf, 1):
            if not isinstance(st, dict):
                continue
            phase_name = str(st.get("stage_name") or st.get("phase") or f"Phase {i}")
            instruction = str(st.get("prompt") or st.get("instruction") or "")
            output_preview = (
                str(st.get("parsed_json") or st.get("raw_response") or st.get("output") or st.get("result") or "")[
                    :2000
                ]
            )
            out.append(
                PhaseSnapshot(
                    phase=phase_name,
                    instruction=instruction[:2000],
                    output_preview=output_preview,
                )
            )
    return out


def _vita_nl_assertion_score(per_task_row: dict[str, Any]) -> float:
    """Continuous score from NL assertions: satisfied_ratio in [0,1]."""
    rubrics = per_task_row.get("reward_rubrics_compact")
    if isinstance(rubrics, list) and rubrics:
        total = 0
        met = 0
        for item in rubrics:
            if not isinstance(item, dict):
                continue
            if "met" not in item:
                continue
            total += 1
            if bool(item.get("met")):
                met += 1
        if total > 0:
            return float(met) / float(total)
    # Fallback to raw reward if assertions are unavailable.
    return float(per_task_row.get("reward") or 0.0)


def _try_extract_drb_task_score(bundle_path: Path, task_id: str) -> float | None:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    race_txt = Path(bundle.get("race_result_txt") or "")
    race_root = race_txt.parent if race_txt.parent.is_dir() else race_txt
    if not race_root.exists():
        return None

    tid_int = int(task_id)
    for p in race_root.rglob("*.jsonl"):
        try:
            for ln in p.read_text(encoding="utf-8").splitlines():
                item = json.loads(ln)
                rid = item.get("id", item.get("task_id"))
                if rid is None or int(rid) != tid_int:
                    continue
                for key in ("score", "final_score", "overall_score", "race_score", "overall"):
                    if key in item:
                        return float(item[key])
        except Exception:
            continue
    for p in race_root.rglob("*.json"):
        try:
            item = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(item, list):
            seq = item
        else:
            seq = [item]
        for row in seq:
            if not isinstance(row, dict):
                continue
            rid = row.get("id", row.get("task_id"))
            if rid is None:
                continue
            if int(rid) != tid_int:
                continue
            for key in ("score", "final_score", "overall_score", "race_score", "overall"):
                if key in row:
                    return float(row[key])
    return None


def _try_extract_hlemath_task_score(bundle_path: Path, task_id: str) -> float | None:
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    pts = bundle.get("per_task_scores") or {}
    if str(task_id) in pts:
        return float(pts[str(task_id)])
    return None


def _try_extract_bcp_task_score(bundle_path: Path, task_id: str) -> float | None:
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    pts = bundle.get("per_task_scores") or {}
    if str(task_id) in pts:
        return float(pts[str(task_id)])
    return None


def _load_trajectory_record(path: Path) -> TrajectoryRecord | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    snaps_raw = payload.get("phase_snapshots") or []
    snaps: list[PhaseSnapshot] = []
    if isinstance(snaps_raw, list):
        for item in snaps_raw:
            if not isinstance(item, dict):
                continue
            snaps.append(
                PhaseSnapshot(
                    phase=str(item.get("phase") or ""),
                    instruction=str(item.get("instruction") or ""),
                    output_preview=str(item.get("output_preview") or ""),
                )
            )
    try:
        return TrajectoryRecord(
            schema=str(payload.get("schema") or ""),
            bench_backend=str(payload.get("bench_backend") or ""),
            round_idx=int(payload.get("round_idx")),
            task_id=str(payload.get("task_id") or ""),
            trajectory_idx=int(payload.get("trajectory_idx")),
            trajectory_tag=str(payload.get("trajectory_tag") or ""),
            score=float(payload.get("score")),
            score_source=str(payload.get("score_source") or ""),
            log_path=str(payload.get("log_path") or ""),
            raw_result_path=str(payload.get("raw_result_path") or ""),
            phase_snapshots=snaps,
            metadata=dict(payload.get("metadata") or {}),
        )
    except Exception:
        return None


def _bundle_path_for_rollout_slot(
    bb: str,
    bench_rollouts: Path,
    round_idx: int,
    k_idx: int,
    tid_s: str,
) -> Path:
    """Matches ``bench_eval`` naming: ``*_bundle_rXX_<sanitized(traj_KK_task_id)>.json``."""
    tag = f"traj_{int(k_idx):02d}_task_{tid_s}"
    suf = f"_{_sanitize_model_tag(tag)}"
    if bb == "drb":
        return bench_rollouts / f"drb_bundle_r{round_idx:02d}{suf}.json"
    if bb == "hlemath":
        return bench_rollouts / f"hlemath_bundle_r{round_idx:02d}{suf}.json"
    if bb == "bcp":
        return bench_rollouts / f"bcp_bundle_r{round_idx:02d}{suf}.json"
    raise ValueError(f"unsupported bench_backend for bundle slot path: {bb!r}")


def _hlemath_or_bcp_bundle_rollout_complete(bundle_path: Path, tid_s: str) -> bool:
    """Bundle JSON + ``per_task_scores`` + ``process_traces`` dir (last artifacts written for that job)."""
    if not bundle_path.is_file():
        return False
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    pts = bundle.get("per_task_scores") or {}
    if str(tid_s) not in pts:
        return False
    try:
        float(pts[str(tid_s)])
    except (TypeError, ValueError):
        return False
    ptd = Path(bundle.get("process_trace_dir") or "")
    return ptd.is_dir()


def _trajectory_record_from_bundle_resume(
    bb: str,
    bundle_path: Path,
    tid_s: str,
    k_idx: int,
    round_idx: int,
    rollout_agent_temperature: float,
) -> TrajectoryRecord | None:
    """Rebuild a trajectory record when ``trajectories/<tid>/traj_*.json`` is missing but bench bundle is complete."""
    if bb == "drb":
        score = _try_extract_drb_task_score(bundle_path, tid_s)
        src = "drb_race_per_task"
    elif bb == "hlemath":
        score = _try_extract_hlemath_task_score(bundle_path, tid_s)
        src = "hlemath_sympy"
    elif bb == "bcp":
        score = _try_extract_bcp_task_score(bundle_path, tid_s)
        try:
            bundle_resume = json.loads(bundle_path.read_text(encoding="utf-8"))
            src = (
                "bcp_llm_judge"
                if (bundle_resume.get("judge_llm") or "").strip()
                else "bcp_exact_match"
            )
        except Exception:
            src = "bcp_exact_match"
    else:
        return None
    if score is None:
        return None
    snaps = _extract_drb_phase_snapshots(bundle_path, tid_s)
    return TrajectoryRecord(
        schema=SCHEMA_TRAJECTORY_RECORD,
        bench_backend=bb,
        round_idx=round_idx,
        task_id=tid_s,
        trajectory_idx=k_idx,
        trajectory_tag=f"task_{tid_s}_traj_{k_idx:02d}",
        score=float(score),
        score_source=src,
        log_path=str(bundle_path.resolve()),
        raw_result_path=str(bundle_path.resolve()),
        phase_snapshots=snaps,
        metadata={"temperature": rollout_agent_temperature},
    )


def _vitabench_shell_manifest_path(
    *,
    bench_rollouts: Path,
    bench_id: str,
    run_id: str,
    round_idx: int,
    k_idx: int,
) -> Path:
    """Final Skill-MAS artifact for a VitaBench trajectory (written after ``run_domain``)."""
    traj_tag = f"traj_{int(k_idx):02d}"
    stem = f"mas_evolve_{bench_id}_{run_id}_r{round_idx:02d}_{traj_tag}"
    return (
        bench_rollouts
        / f"vitabench_eval_r{round_idx:02d}_{traj_tag}"
        / "process_traces"
        / f"{stem}_shell_manifest.json"
    )


def _persist_non_vita_trajectory_record(record: TrajectoryRecord, traj_root: Path, asp_root: Path) -> None:
    tid_s = str(record.task_id)
    tdir = traj_root / tid_s
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"traj_{int(record.trajectory_idx):02d}.json").write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    atdir = asp_root / tid_s / f"traj_{int(record.trajectory_idx):02d}"
    atdir.mkdir(parents=True, exist_ok=True)
    for i, snap in enumerate(record.phase_snapshots, 1):
        (atdir / f"phase_{i}.json").write_text(
            json.dumps(
                {"phase": snap.phase, "instruction": snap.instruction, "output_preview": snap.output_preview},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _persist_vita_trajectory_artifacts(
    *,
    record: TrajectoryRecord,
    traj_root: Path,
    asp_root: Path,
) -> None:
    tid_s = str(record.task_id)
    tdir = traj_root / tid_s
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"traj_{int(record.trajectory_idx):02d}.json").write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    atdir = asp_root / tid_s / f"traj_{int(record.trajectory_idx):02d}"
    atdir.mkdir(parents=True, exist_ok=True)
    for i, snap in enumerate(record.phase_snapshots, 1):
        (atdir / f"phase_{i}.json").write_text(
            json.dumps(
                {
                    "phase": snap.phase,
                    "instruction": snap.instruction,
                    "output_preview": snap.output_preview,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def _rebuild_vita_records_without_rollout(
    *,
    raw_path: Path,
    normalized_task_ids: list[str],
    k_idx: int,
    bench_id: str,
    run_id: str,
    round_idx: int,
    rollout_agent_temperature: float,
    bench_rollouts_eval_dir: Path,
    vita_process_trace_dir: Path,
) -> dict[str, TrajectoryRecord] | None:
    """Rebuild per-task records from an existing simulation JSON (manifest present, traj JSON optional)."""
    try:
        summary = summarize_run(raw_path)
    except Exception:
        return None
    per_task = {
        str(row.get("task_id")): row
        for row in list(summary.get("per_task") or [])
        if isinstance(row, dict)
    }
    out: dict[str, TrajectoryRecord] = {}
    for tid_s in normalized_task_ids:
        row = per_task.get(str(tid_s)) or {}
        score = _vita_nl_assertion_score(row)
        phase_snapshots = _extract_vita_phase_snapshots(raw_path, str(tid_s))
        tag = f"task_{tid_s}_traj_{k_idx:02d}"
        out[tid_s] = TrajectoryRecord(
            schema=SCHEMA_TRAJECTORY_RECORD,
            bench_backend="vitabench",
            round_idx=round_idx,
            task_id=str(tid_s),
            trajectory_idx=k_idx,
            trajectory_tag=tag,
            score=score,
            score_source="vitabench_nl_assertion_ratio",
            log_path=str(raw_path),
            raw_result_path=str(raw_path),
            phase_snapshots=phase_snapshots,
            metadata={
                "temperature": rollout_agent_temperature,
                "process_trace_dir": str(vita_process_trace_dir.resolve()),
                "bench_rollouts_eval_dir": str(bench_rollouts_eval_dir.resolve()),
            },
        )
    return out


def _try_resume_records_for_trajectory(
    *,
    traj_root: Path,
    normalized_task_ids: list[str],
    trajectory_idx: int,
) -> dict[str, TrajectoryRecord] | None:
    out: dict[str, TrajectoryRecord] = {}
    for tid_s in normalized_task_ids:
        p = traj_root / tid_s / f"traj_{trajectory_idx:02d}.json"
        if not p.is_file():
            return None
        rec = _load_trajectory_record(p)
        if rec is None:
            return None
        if rec.task_id != tid_s or int(rec.trajectory_idx) != int(trajectory_idx):
            return None
        out[tid_s] = rec
    return out


def _drb_eval_section_from_bundle(bundle_path: Path) -> dict[str, Any]:
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    ptd = Path(bundle.get("process_trace_dir") or "")
    if not ptd.is_dir():
        return {}
    agg: dict[str, Any] = {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    per_task: list[dict[str, Any]] = []
    trace_files: list[Path] = []
    nested = ptd / "sample_trace_json"
    if nested.is_dir():
        trace_files.extend(sorted(f for f in nested.glob("*.json") if not f.name.startswith("_")))
    if not trace_files:
        trace_files.extend(sorted(ptd.glob("*.json")))
    for p in trace_files:
        if p.name.startswith("_"):
            continue
        if not nested.is_dir() and not p.stem.isdigit():
            continue
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        u = dict(row.get("usage_totals") or {})
        if not u:
            continue
        tid_val = row.get("task_id", row.get("id"))
        if tid_val is None:
            if not p.stem.isdigit():
                continue
            tid_out = int(p.stem)
        else:
            try:
                tid_out = int(tid_val)
            except (TypeError, ValueError):
                continue
        pt = int(u.get("prompt_tokens", 0) or 0)
        ot = int(u.get("output_tokens", 0) or 0)
        tt = int(u.get("total_tokens", pt + ot) or 0)
        cc = float(u.get("estimated_cost_usd", 0.0) or 0.0)
        agg["prompt_tokens"] += pt
        agg["output_tokens"] += ot
        agg["total_tokens"] += tt
        agg["estimated_cost_usd"] += cc
        per_task.append(
            {
                "task_id": tid_out,
                "prompt_tokens": pt,
                "output_tokens": ot,
                "total_tokens": tt,
                "estimated_cost_usd": cc,
            }
        )
    if not per_task:
        return {}
    return {
        "phase": "drb_skill_mas_rollout",
        "description": "Resumed from existing DRB trajectory bundle/process traces.",
        "aggregate_usage": agg,
        "per_task": per_task,
        "notes": ["Trajectory resumed from artifacts; no new DRB rollout executed."],
    }


def run_multi_trajectory_rollout(
    *,
    bench_backend: str,
    bench_id: str,
    run_id: str,
    round_idx: int,
    task_ids: list[str],
    k_trajectories: int = EVOLVE_K_TRAJECTORIES,
    task_set_name: str = "",
    domain: str,
    agent_llm: str,
    user_llm: str,
    evaluator_llm: str,
    llm_args_agent: dict[str, Any] | None,
    llm_args_user: dict[str, Any] | None,
    llm_args_evaluator: dict[str, Any] | None,
    max_steps: int,
    max_concurrency: int,
    language: str = DEFAULT_LANGUAGE,
    runs_dir: Path,
    log_root: Path,
    drb_bench_root: Path | None = None,
    drb_query_jsonl: Path | None = None,
    drb_race_max_workers: int = 1,
    hlemath_jsonl: Path | None = None,
    bcp_jsonl: Path | None = None,
    bcp_judge_llm: str | None = None,
    bcp_judge_timeout_s: float = 120.0,
    bcp_index_path: str | Path | None = None,
    bcp_retrieval_topk: int = 5,
    bcp_doc_max_tokens: int = 512,
    bcp_max_retrieval_rounds: int = 10,
) -> tuple[dict[str, list[TrajectoryRecord]], list[dict[str, Any]], Path]:
    """Execute Step 1 and persist trajectory/aspect artifacts for this round."""
    bb = (bench_backend or "vitabench").strip().lower()
    bcp_judge_model = (bcp_judge_llm or "").strip() or None
    round_root = _round_root(runs_dir=runs_dir, bench_id=bench_id, run_id=run_id, round_idx=round_idx)
    traj_root = round_root / ROUND_TRAJECTORIES_DIRNAME
    asp_root = round_root / ROUND_ASPECTS_DIRNAME
    traj_root.mkdir(parents=True, exist_ok=True)
    asp_root.mkdir(parents=True, exist_ok=True)

    skill_root = skills_evolution_dir(bb, agent_llm) / bench_id / run_id / f"round_{round_idx:02d}"
    if not (skill_root / "SKILL.md").is_file():
        raise FileNotFoundError(f"Skill workspace missing SKILL.md: {skill_root}")
    merged = skill_root.resolve()
    os.environ["SKILL_MAS_DIR"] = str(merged)
    os.environ.setdefault("SKILL_MAS_MAX_TOOL_ROUNDS", "10")
    # VitaBench: avoid filling dataset/vitabench/results/<model>/skill_mas_process_traces/ during evolve
    # and skip mirror_simulation_to_results into dataset/vitabench/results/.
    # Per-task Skill-MAS artifacts are written under round_XX/bench_rollouts/ via RunConfig.skill_mas_trace_out_dir.
    # Set MASKILL_SKIP_VITABENCH_TRACE_EXPORT=0 before launch to restore legacy export + mirror under dataset/vitabench/results/.
    if bb == "vitabench" and "MASKILL_SKIP_VITABENCH_TRACE_EXPORT" not in os.environ:
        os.environ["MASKILL_SKIP_VITABENCH_TRACE_EXPORT"] = "1"

    by_task: dict[str, list[TrajectoryRecord]] = {}
    eval_sections: list[dict[str, Any]] = []

    normalized_task_ids = [str(t) for t in task_ids]
    for tid_s in normalized_task_ids:
        by_task.setdefault(tid_s, [])

    rollout_agent_temperature = float(model_runtime_params(agent_llm).get("temperature", 1.0))

    if bb == "vitabench":
        from vita.config import models
        from vita.data_model.simulation import RunConfig
        from vita.run import run_domain
        from vita.utils.utils import DATA_DIR

        llm_a_base = dict(llm_args_agent if llm_args_agent is not None else models.get(agent_llm, {}))
        llm_u_base = dict(llm_args_user if llm_args_user is not None else models.get(user_llm, {}))
        llm_e_base = dict(
            llm_args_evaluator if llm_args_evaluator is not None else models.get(evaluator_llm, {})
        )
        ts_for_cfg = task_set_name if task_set_name.strip() else None
        bench_rollouts_v = round_root / ROUND_BENCH_ROLLOUTS_DIRNAME
        pending: list[int] = []

        # Resume when bench_rollouts marks a trajectory complete (shell manifest = last artifact).
        for k in range(int(max(1, k_trajectories))):
            manifest = _vitabench_shell_manifest_path(
                bench_rollouts=bench_rollouts_v,
                bench_id=bench_id,
                run_id=run_id,
                round_idx=round_idx,
                k_idx=k,
            )
            if manifest.is_file():
                resumed = _try_resume_records_for_trajectory(
                    traj_root=traj_root,
                    normalized_task_ids=normalized_task_ids,
                    trajectory_idx=k,
                )
                if resumed is not None:
                    for tid_s in normalized_task_ids:
                        by_task[tid_s].append(resumed[tid_s])
                    first = resumed.get(normalized_task_ids[0]) if normalized_task_ids else None
                    if first:
                        try:
                            rp = Path(first.raw_result_path)
                            if rp.is_file():
                                eval_sections.append(
                                    vita_rollout_cost_report(
                                        rp,
                                        agent_llm=agent_llm,
                                        user_llm=user_llm,
                                        evaluator_llm=evaluator_llm,
                                    )
                                )
                        except Exception:
                            pass
                    continue

                traj_tag = f"traj_{k:02d}"
                save_name = f"mas_evolve_{bench_id}_{run_id}_r{round_idx:02d}_{traj_tag}.json"
                raw_path = DATA_DIR / "simulations" / save_name
                br_eval = bench_rollouts_v / f"vitabench_eval_r{round_idx:02d}_{traj_tag}"
                vita_pt = (br_eval / "process_traces").resolve()
                rebuilt = _rebuild_vita_records_without_rollout(
                    raw_path=raw_path,
                    normalized_task_ids=normalized_task_ids,
                    k_idx=k,
                    bench_id=bench_id,
                    run_id=run_id,
                    round_idx=round_idx,
                    rollout_agent_temperature=rollout_agent_temperature,
                    bench_rollouts_eval_dir=br_eval,
                    vita_process_trace_dir=vita_pt,
                )
                if rebuilt is None:
                    pending.append(k)
                    continue
                for tid_s in normalized_task_ids:
                    rec = rebuilt[tid_s]
                    by_task[tid_s].append(rec)
                    _persist_vita_trajectory_artifacts(record=rec, traj_root=traj_root, asp_root=asp_root)
                try:
                    if raw_path.is_file():
                        eval_sections.append(
                            vita_rollout_cost_report(
                                raw_path,
                                agent_llm=agent_llm,
                                user_llm=user_llm,
                                evaluator_llm=evaluator_llm,
                            )
                        )
                except Exception:
                    pass
                continue

            pending.append(k)

        n_vita_traj = int(max(1, k_trajectories))
        print(
            f"[Skill_MAS] vitabench rollout resume (bench_rollouts shell manifest): "
            f"{n_vita_traj - len(pending)}/{n_vita_traj} trajectories complete; "
            f"pending_trajectories={len(pending)}.",
            flush=True,
        )

        per_traj_max_concurrency = 1
        if pending:
            per_traj_max_concurrency = max(1, int(max_concurrency) // max(1, len(pending)))

        def _run_single_vita_job(k_idx: int) -> tuple[list[TrajectoryRecord], dict[str, Any]]:
            llm_a = dict(llm_a_base)
            llm_u = dict(llm_u_base)
            llm_e = dict(llm_e_base)

            traj_tag = f"traj_{k_idx:02d}"
            save_name = f"mas_evolve_{bench_id}_{run_id}_r{round_idx:02d}_{traj_tag}.json"
            raw_path = DATA_DIR / "simulations" / save_name
            if raw_path.exists():
                raw_path.unlink()

            br_eval = round_root / ROUND_BENCH_ROLLOUTS_DIRNAME / f"vitabench_eval_r{round_idx:02d}_{traj_tag}"
            vita_process_trace_dir = (br_eval / "process_traces").resolve()
            vita_process_trace_dir.mkdir(parents=True, exist_ok=True)

            cfg = RunConfig(
                domain=domain,
                task_set_name=ts_for_cfg,
                task_ids=normalized_task_ids,
                num_tasks=None,
                agent="skill_mas_agent",
                llm_agent=agent_llm,
                llm_args_agent=llm_a,
                user="static_input_user",
                llm_user=user_llm,
                llm_args_user=llm_u,
                num_trials=1,
                max_steps=max_steps,
                max_errors=10,
                save_to=save_name,
                max_concurrency=per_traj_max_concurrency,
                seed=300 + k_idx,
                log_level="INFO",
                enable_think=False,
                language=language,
                llm_evaluator=evaluator_llm,
                llm_args_evaluator=llm_e,
                skill_mas_trace_out_dir=str(vita_process_trace_dir),
            )
            run_domain(cfg)
            summary = summarize_run(raw_path)
            per_task = {
                str(row.get("task_id")): row
                for row in list(summary.get("per_task") or [])
                if isinstance(row, dict)
            }
            records: list[TrajectoryRecord] = []
            for tid_s in normalized_task_ids:
                row = per_task.get(str(tid_s)) or {}
                score = _vita_nl_assertion_score(row)
                phase_snapshots = _extract_vita_phase_snapshots(raw_path, str(tid_s))
                tag = f"task_{tid_s}_traj_{k_idx:02d}"
                record = TrajectoryRecord(
                    schema=SCHEMA_TRAJECTORY_RECORD,
                    bench_backend=bb,
                    round_idx=round_idx,
                    task_id=str(tid_s),
                    trajectory_idx=k_idx,
                    trajectory_tag=tag,
                    score=score,
                    score_source="vitabench_nl_assertion_ratio",
                    log_path=str(raw_path),
                    raw_result_path=str(raw_path),
                    phase_snapshots=phase_snapshots,
                    metadata={
                        "temperature": rollout_agent_temperature,
                        "process_trace_dir": str(vita_process_trace_dir),
                        "bench_rollouts_eval_dir": str(br_eval.resolve()),
                    },
                )
                records.append(record)
            section = vita_rollout_cost_report(
                raw_path,
                agent_llm=agent_llm,
                user_llm=user_llm,
                evaluator_llm=evaluator_llm,
            )
            return records, section

        if pending:
            print(
                f"[Skill_MAS] vitabench rollout fanout: pending_jobs={len(pending)} "
                f"global_max_concurrency={max(1, int(max_concurrency))} "
                f"per_trajectory_max_concurrency={per_traj_max_concurrency}",
                flush=True,
            )
            async def _gather_vita_jobs() -> list[tuple[list[TrajectoryRecord], dict[str, Any]]]:
                sem = asyncio.Semaphore(max(1, int(max_concurrency)))

                async def _one(k_idx: int) -> tuple[list[TrajectoryRecord], dict[str, Any]]:
                    async with sem:
                        return await asyncio.to_thread(_run_single_vita_job, k_idx)

                return await asyncio.gather(*[_one(k) for k in pending])

            for records, section in asyncio.run(_gather_vita_jobs()):
                eval_sections.append(section)
                for record in records:
                    tid_s = str(record.task_id)
                    by_task[tid_s].append(record)
                    tdir = traj_root / tid_s
                    tdir.mkdir(parents=True, exist_ok=True)
                    (tdir / f"traj_{int(record.trajectory_idx):02d}.json").write_text(
                        json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    atdir = asp_root / tid_s / f"traj_{int(record.trajectory_idx):02d}"
                    atdir.mkdir(parents=True, exist_ok=True)
                    for i, snap in enumerate(record.phase_snapshots, 1):
                        (atdir / f"phase_{i}.json").write_text(
                            json.dumps(
                                {
                                    "phase": snap.phase,
                                    "instruction": snap.instruction,
                                    "output_preview": snap.output_preview,
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
        return by_task, eval_sections, round_root

    if bb in ("hlemath", "bcp", "drb"):
        bench_rollouts = round_root / ROUND_BENCH_ROLLOUTS_DIRNAME
        pending: list[tuple[int, str]] = []
        reused_slots = 0
        total_slots = max(1, int(k_trajectories)) * len(normalized_task_ids)

        for k in range(int(max(1, k_trajectories))):
            for tid_s in normalized_task_ids:
                bp = _bundle_path_for_rollout_slot(bb, bench_rollouts, round_idx, k, tid_s)
                slot_ok = False
                if bb == "drb":
                    try:
                        tid_i = int(tid_s)
                    except ValueError:
                        tid_i = -1
                    slot_ok = bp.is_file() and tid_i >= 0 and _drb_cached_rollout_complete(bp, [tid_i])
                elif bb in ("hlemath", "bcp"):
                    slot_ok = _hlemath_or_bcp_bundle_rollout_complete(bp, tid_s)

                if not slot_ok:
                    pending.append((k, tid_s))
                    continue

                rec_path = traj_root / tid_s / f"traj_{k:02d}.json"
                rec = _trajectory_record_from_bundle_resume(
                    bb,
                    bp,
                    tid_s,
                    k,
                    round_idx,
                    rollout_agent_temperature,
                )
                if rec is None and rec_path.is_file():
                    loaded = _load_trajectory_record(rec_path)
                    if loaded is not None and loaded.task_id == tid_s and int(loaded.trajectory_idx) == int(k):
                        rec = loaded
                if rec is None:
                    pending.append((k, tid_s))
                    continue

                by_task[tid_s].append(rec)
                resumed_eval = _drb_eval_section_from_bundle(Path(rec.log_path))
                if resumed_eval:
                    eval_sections.append(resumed_eval)
                _persist_non_vita_trajectory_record(rec, traj_root, asp_root)
                reused_slots += 1

        print(
            f"[Skill_MAS] {bb} rollout resume (bench_rollouts bundle+RACE/HLEMATH/BCP artifacts): "
            f"reused {reused_slots}/{total_slots} slots; pending_jobs={len(pending)}.",
            flush=True,
        )

        def _persist_record(_tid_s: str, record: TrajectoryRecord) -> None:
            _persist_non_vita_trajectory_record(record, traj_root, asp_root)

        def _run_single_non_vita_job(k_idx: int, tid_s: str) -> tuple[str, TrajectoryRecord, dict[str, Any]]:
            traj_tag = f"traj_{k_idx:02d}"
            if bb == "hlemath":
                if hlemath_jsonl is None:
                    raise ValueError("hlemath rollout requires hlemath_jsonl path")
                bundle_path, _log_round, eval_section = run_hlemath_evaluation_round(
                    bench_id=bench_id,
                    run_id=run_id,
                    round_idx=round_idx,
                    task_ids=[tid_s],
                    jsonl_path=Path(hlemath_jsonl).resolve(),
                    agent_llm=agent_llm,
                    max_concurrency=1,
                    runs_dir=runs_dir,
                    log_root=log_root,
                    skills_evolution_dir=skills_evolution_dir(bb, agent_llm),
                    trajectory_tag=f"{traj_tag}_task_{tid_s}",
                )
                maybe_score = _try_extract_hlemath_task_score(bundle_path, tid_s)
                if maybe_score is None:
                    raise RuntimeError(f"HLEMATH rollout missing score for task_id={tid_s} bundle={bundle_path}")
                phase_snapshots = _extract_drb_phase_snapshots(bundle_path, tid_s)
                record = TrajectoryRecord(
                    schema=SCHEMA_TRAJECTORY_RECORD,
                    bench_backend=bb,
                    round_idx=round_idx,
                    task_id=tid_s,
                    trajectory_idx=k_idx,
                    trajectory_tag=f"task_{tid_s}_traj_{k_idx:02d}",
                    score=float(maybe_score),
                    score_source="hlemath_sympy",
                    log_path=str(bundle_path),
                    raw_result_path=str(bundle_path),
                    phase_snapshots=phase_snapshots,
                    metadata={"temperature": rollout_agent_temperature},
                )
                return tid_s, record, eval_section

            if bb == "bcp":
                if bcp_jsonl is None:
                    raise ValueError("bcp rollout requires bcp_jsonl path")
                bundle_path, _log_round, eval_section = run_bcp_evaluation_round(
                    bench_id=bench_id,
                    run_id=run_id,
                    round_idx=round_idx,
                    task_ids=[tid_s],
                    jsonl_path=Path(bcp_jsonl).resolve(),
                    agent_llm=agent_llm,
                    max_concurrency=1,
                    runs_dir=runs_dir,
                    log_root=log_root,
                    skills_evolution_dir=skills_evolution_dir(bb, agent_llm),
                    trajectory_tag=f"{traj_tag}_task_{tid_s}",
                    judge_llm=bcp_judge_model,
                    judge_timeout_s=float(bcp_judge_timeout_s),
                    bcp_index_path=bcp_index_path,
                    bcp_retrieval_topk=int(bcp_retrieval_topk),
                    bcp_doc_max_tokens=int(bcp_doc_max_tokens),
                    bcp_max_retrieval_rounds=int(bcp_max_retrieval_rounds),
                )
                maybe_score = _try_extract_bcp_task_score(bundle_path, tid_s)
                if maybe_score is None:
                    raise RuntimeError(f"BrowseComp rollout missing score for task_id={tid_s} bundle={bundle_path}")
                phase_snapshots = _extract_drb_phase_snapshots(bundle_path, tid_s)
                record = TrajectoryRecord(
                    schema=SCHEMA_TRAJECTORY_RECORD,
                    bench_backend=bb,
                    round_idx=round_idx,
                    task_id=tid_s,
                    trajectory_idx=k_idx,
                    trajectory_tag=f"task_{tid_s}_traj_{k_idx:02d}",
                    score=float(maybe_score),
                    score_source=("bcp_llm_judge" if bcp_judge_model else "bcp_exact_match"),
                    log_path=str(bundle_path),
                    raw_result_path=str(bundle_path),
                    phase_snapshots=phase_snapshots,
                    metadata={"temperature": rollout_agent_temperature},
                )
                return tid_s, record, eval_section

            if drb_bench_root is None:
                raise ValueError("DRB rollout requires drb_bench_root.")
            bp, _lr, es = run_drb_evaluation_round(
                bench_id=bench_id,
                run_id=run_id,
                round_idx=round_idx,
                task_ids=[int(tid_s)],
                drb_bench_root=drb_bench_root,
                agent_llm=agent_llm,
                max_concurrency=1,
                race_max_workers=max(1, int(drb_race_max_workers)),
                runs_dir=runs_dir,
                log_root=log_root,
                skills_evolution_dir=skills_evolution_dir(bb, agent_llm),
                merged_workspaces_dir=merged_workspaces_dir(bb, agent_llm),
                trajectory_tag=f"{traj_tag}_task_{tid_s}",
                export_logs=False,
                drb_query_jsonl=drb_query_jsonl,
            )
            eval_section = es
            _ = summarize_drb_round(bp)
            maybe_score = _try_extract_drb_task_score(bp, tid_s)
            if maybe_score is None:
                raise RuntimeError(
                    "DRB rollout requires per-task RACE score; "
                    f"missing score for task_id={tid_s} trajectory_idx={k_idx} bundle={bp}"
                )
            phase_snapshots = _extract_drb_phase_snapshots(bp, tid_s)
            meta: dict[str, Any] = {"temperature": rollout_agent_temperature}
            record = TrajectoryRecord(
                schema=SCHEMA_TRAJECTORY_RECORD,
                bench_backend=bb,
                round_idx=round_idx,
                task_id=tid_s,
                trajectory_idx=k_idx,
                trajectory_tag=f"task_{tid_s}_traj_{k_idx:02d}",
                score=float(maybe_score),
                score_source="drb_race_per_task",
                log_path=str(bp),
                raw_result_path=str(bp),
                phase_snapshots=phase_snapshots,
                metadata=meta,
            )
            return tid_s, record, eval_section

        if pending:
            if bb == "drb":
                reset_drb_bench_rollout_resume_counter()
            print(
                f"[Skill_MAS] {bb} rollout fanout: pending_jobs={len(pending)} max_concurrency={max(1, int(max_concurrency))}",
                flush=True,
            )
            async def _gather_non_vita_jobs() -> list[tuple[str, TrajectoryRecord, dict[str, Any]]]:
                sem = asyncio.Semaphore(max(1, int(max_concurrency)))

                async def _one(job: tuple[int, str]) -> tuple[str, TrajectoryRecord, dict[str, Any]]:
                    k_idx, tid = job
                    async with sem:
                        return await asyncio.to_thread(_run_single_non_vita_job, k_idx, tid)

                return await asyncio.gather(*[_one(p) for p in pending])

            for tid_s, record, eval_section in asyncio.run(_gather_non_vita_jobs()):
                by_task[tid_s].append(record)
                eval_sections.append(eval_section)
                _persist_record(tid_s, record)
            if bb == "drb":
                n_br_resume = take_drb_bench_rollout_resume_count()
                print(
                    f"[Skill_MAS] DRB bench_rollouts disk cache: {n_br_resume}/{len(pending)} executed jobs "
                    f"reused existing bundle+RACE under bench_rollouts/ (remaining ran full Skill-MAS + RACE).",
                    flush=True,
                )
        return by_task, eval_sections, round_root

    for k in range(int(max(1, k_trajectories))):
        traj_tag = f"traj_{k:02d}"
        resumed = _try_resume_records_for_trajectory(
            traj_root=traj_root,
            normalized_task_ids=normalized_task_ids,
            trajectory_idx=k,
        )
        if resumed is not None:
            print(
                f"[Skill_MAS] resume round={round_idx} trajectory={traj_tag} from existing artifacts",
                flush=True,
            )
            for tid_s in normalized_task_ids:
                by_task[tid_s].append(resumed[tid_s])
            if bb == "drb":
                first = resumed.get(normalized_task_ids[0]) if normalized_task_ids else None
                if first and first.log_path:
                    resumed_eval = _drb_eval_section_from_bundle(Path(first.log_path))
                    if resumed_eval:
                        eval_sections.append(resumed_eval)
            continue
        if bb == "hlemath":
            if hlemath_jsonl is None:
                raise ValueError("hlemath rollout requires hlemath_jsonl path")
            bundle_path, _log_round, eval_section = run_hlemath_evaluation_round(
                bench_id=bench_id,
                run_id=run_id,
                round_idx=round_idx,
                task_ids=normalized_task_ids,
                jsonl_path=Path(hlemath_jsonl).resolve(),
                agent_llm=agent_llm,
                max_concurrency=max(1, int(max_concurrency)),
                runs_dir=runs_dir,
                log_root=log_root,
                skills_evolution_dir=skills_evolution_dir(bb, agent_llm),
                trajectory_tag=traj_tag,
            )
            eval_sections.append(eval_section)
            for tid_s in normalized_task_ids:
                maybe_score = _try_extract_hlemath_task_score(bundle_path, tid_s)
                if maybe_score is None:
                    raise RuntimeError(
                        f"HLEMATH rollout missing score for task_id={tid_s} bundle={bundle_path}"
                    )
                score = float(maybe_score)
                phase_snapshots = _extract_drb_phase_snapshots(bundle_path, tid_s)
                tag = f"task_{tid_s}_traj_{k:02d}"
                record = TrajectoryRecord(
                    schema=SCHEMA_TRAJECTORY_RECORD,
                    bench_backend=bb,
                    round_idx=round_idx,
                    task_id=tid_s,
                    trajectory_idx=k,
                    trajectory_tag=tag,
                    score=score,
                    score_source="hlemath_sympy",
                    log_path=str(bundle_path),
                    raw_result_path=str(bundle_path),
                    phase_snapshots=phase_snapshots,
                    metadata={"temperature": rollout_agent_temperature},
                )
                by_task[tid_s].append(record)
                tdir = traj_root / tid_s
                tdir.mkdir(parents=True, exist_ok=True)
                (tdir / f"traj_{k:02d}.json").write_text(
                    json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                atdir = asp_root / tid_s / f"traj_{k:02d}"
                atdir.mkdir(parents=True, exist_ok=True)
                for i, snap in enumerate(phase_snapshots, 1):
                    (atdir / f"phase_{i}.json").write_text(
                        json.dumps(
                            {"phase": snap.phase, "instruction": snap.instruction, "output_preview": snap.output_preview},
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
        elif bb == "bcp":
            if bcp_jsonl is None:
                raise ValueError("bcp rollout requires bcp_jsonl path")
            bundle_path, _log_round, eval_section = run_bcp_evaluation_round(
                bench_id=bench_id,
                run_id=run_id,
                round_idx=round_idx,
                task_ids=normalized_task_ids,
                jsonl_path=Path(bcp_jsonl).resolve(),
                agent_llm=agent_llm,
                max_concurrency=max(1, int(max_concurrency)),
                runs_dir=runs_dir,
                log_root=log_root,
                skills_evolution_dir=skills_evolution_dir(bb, agent_llm),
                trajectory_tag=traj_tag,
                judge_llm=bcp_judge_model,
                judge_timeout_s=float(bcp_judge_timeout_s),
                bcp_index_path=bcp_index_path,
                bcp_retrieval_topk=int(bcp_retrieval_topk),
                bcp_doc_max_tokens=int(bcp_doc_max_tokens),
                bcp_max_retrieval_rounds=int(bcp_max_retrieval_rounds),
            )
            eval_sections.append(eval_section)
            for tid_s in normalized_task_ids:
                maybe_score = _try_extract_bcp_task_score(bundle_path, tid_s)
                if maybe_score is None:
                    raise RuntimeError(
                        f"BrowseComp rollout missing score for task_id={tid_s} bundle={bundle_path}"
                    )
                score = float(maybe_score)
                phase_snapshots = _extract_drb_phase_snapshots(bundle_path, tid_s)
                tag = f"task_{tid_s}_traj_{k:02d}"
                record = TrajectoryRecord(
                    schema=SCHEMA_TRAJECTORY_RECORD,
                    bench_backend=bb,
                    round_idx=round_idx,
                    task_id=tid_s,
                    trajectory_idx=k,
                    trajectory_tag=tag,
                    score=score,
                    score_source=("bcp_llm_judge" if bcp_judge_model else "bcp_exact_match"),
                    log_path=str(bundle_path),
                    raw_result_path=str(bundle_path),
                    phase_snapshots=phase_snapshots,
                    metadata={"temperature": rollout_agent_temperature},
                )
                by_task[tid_s].append(record)
                tdir = traj_root / tid_s
                tdir.mkdir(parents=True, exist_ok=True)
                (tdir / f"traj_{k:02d}.json").write_text(
                    json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                atdir = asp_root / tid_s / f"traj_{k:02d}"
                atdir.mkdir(parents=True, exist_ok=True)
                for i, snap in enumerate(phase_snapshots, 1):
                    (atdir / f"phase_{i}.json").write_text(
                        json.dumps(
                            {"phase": snap.phase, "instruction": snap.instruction, "output_preview": snap.output_preview},
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
        elif bb == "vitabench":
            raise RuntimeError("vitabench rollout should be handled by global fanout branch.")
        elif bb == "drb":
            if drb_bench_root is None:
                raise ValueError("DRB rollout requires drb_bench_root.")
            bundle_path, _log_round, eval_section = run_drb_evaluation_round(
                bench_id=bench_id,
                run_id=run_id,
                round_idx=round_idx,
                task_ids=[int(t) for t in normalized_task_ids],
                drb_bench_root=drb_bench_root,
                agent_llm=agent_llm,
                max_concurrency=max(1, int(max_concurrency)),
                race_max_workers=int(drb_race_max_workers),
                runs_dir=runs_dir,
                log_root=log_root,
                skills_evolution_dir=skills_evolution_dir(bb, agent_llm),
                merged_workspaces_dir=merged_workspaces_dir(bb, agent_llm),
                trajectory_tag=traj_tag,
                export_logs=False,
                drb_query_jsonl=drb_query_jsonl,
            )
            _ = summarize_drb_round(bundle_path)
            eval_sections.append(eval_section)
            for tid_s in normalized_task_ids:
                maybe_score = _try_extract_drb_task_score(bundle_path, tid_s)
                if maybe_score is None:
                    raise RuntimeError(
                        "DRB rollout requires per-task RACE score; "
                        f"missing score for task_id={tid_s} in bundle={bundle_path}"
                    )
                score = float(maybe_score)
                score_source = "drb_race_per_task"
                phase_snapshots = _extract_drb_phase_snapshots(bundle_path, tid_s)
                tag = f"task_{tid_s}_traj_{k:02d}"
                record = TrajectoryRecord(
                    schema=SCHEMA_TRAJECTORY_RECORD,
                    bench_backend=bb,
                    round_idx=round_idx,
                    task_id=tid_s,
                    trajectory_idx=k,
                    trajectory_tag=tag,
                    score=score,
                    score_source=score_source,
                    log_path=str(bundle_path),
                    raw_result_path=str(bundle_path),
                    phase_snapshots=phase_snapshots,
                    metadata={"temperature": rollout_agent_temperature},
                )
                by_task[tid_s].append(record)
                tdir = traj_root / tid_s
                tdir.mkdir(parents=True, exist_ok=True)
                (tdir / f"traj_{k:02d}.json").write_text(
                    json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                atdir = asp_root / tid_s / f"traj_{k:02d}"
                atdir.mkdir(parents=True, exist_ok=True)
                for i, snap in enumerate(phase_snapshots, 1):
                    (atdir / f"phase_{i}.json").write_text(
                        json.dumps(
                            {"phase": snap.phase, "instruction": snap.instruction, "output_preview": snap.output_preview},
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
        else:
            raise ValueError(f"Unknown bench_backend for rollout: {bb!r}")

    return by_task, eval_sections, round_root

