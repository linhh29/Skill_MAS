"""Detect resume point from artifacts (summary_rXX + skills round_XX)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ..utils.config import BENCH_SEGMENT_VITABENCH, runs_dir, skills_evolution_dir

from Skill_MAS.utils.paths import ensure_sys_path

ensure_sys_path(include_dataset=True)

def _single_skill_ready(root: Path) -> bool:
    return (root / "SKILL.md").is_file()


def skills_round_ready(
    bench_id: str,
    run_id: str,
    round_idx: int,
    *,
    bench_backend: str = BENCH_SEGMENT_VITABENCH,
    agent_llm: str | None = None,
) -> bool:
    root = skills_evolution_dir(bench_backend, agent_llm) / bench_id / run_id / f"round_{round_idx:02d}"
    if not root.is_dir():
        return False
    return _single_skill_ready(root)


def find_latest_completed_eval_index(
    bench_id: str,
    run_id: str,
    *,
    bench_backend: str = BENCH_SEGMENT_VITABENCH,
    agent_llm: str | None = None,
) -> int | None:
    d = runs_dir(bench_backend, agent_llm) / bench_id / run_id
    if not d.is_dir():
        return None
    best: int | None = None
    for p in d.iterdir():
        if p.suffix != ".json" or not p.name.startswith("summary_r"):
            continue
        m = re.match(r"summary_r(\d+)\.json$", p.name)
        if not m:
            continue
        idx = int(m.group(1))
        if best is None or idx > best:
            best = idx
    return best


def compute_resume_start(
    bench_id: str,
    run_id: str,
    rounds: int,
    *,
    bench_backend: str = BENCH_SEGMENT_VITABENCH,
    agent_llm: str | None = None,
) -> tuple[int, bool, str | None]:
    """Return ``(start_r, need_init, status)``.

    ``status`` is ``None`` to continue, ``"complete"`` if nothing left to run, or we raise.
    """
    r00 = skills_evolution_dir(bench_backend, agent_llm) / bench_id / run_id / "round_00"
    latest_summary = find_latest_completed_eval_index(
        bench_id, run_id, bench_backend=bench_backend, agent_llm=agent_llm
    )

    if not r00.is_dir():
        return (0, True, None)

    if latest_summary is None:
        if not skills_round_ready(bench_id, run_id, 0, bench_backend=bench_backend, agent_llm=agent_llm):
            raise RuntimeError(f"{r00} exists but round_00 skills are incomplete; fix or remove.")
        return (0, False, None)

    if latest_summary >= rounds - 1:
        return (0, False, "complete")

    next_r = latest_summary + 1
    if not skills_round_ready(
        bench_id, run_id, next_r, bench_backend=bench_backend, agent_llm=agent_llm
    ):
        raise RuntimeError(
            f"Cannot resume: found summary_r{latest_summary:02d}.json but "
            f"artifacts/.../<dataset>_<model>/skills/.../round_{next_r:02d}/ is missing or incomplete "
            f"(optimizer step may have failed). Repair or delete partial data."
        )
    return (next_r, False, None)

