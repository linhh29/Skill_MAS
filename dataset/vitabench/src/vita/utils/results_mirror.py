"""
Mirror completed simulation JSONs into dataset/vitabench/results/ with a stable layout
(agent LLM / method / optional skill-mas suffix), aligned with drb_bridge conventions.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

from vita.data_model.simulation import RunConfig
from vita.skill_mas_paths import (
    default_skill_mas_workspace_dir,
    is_skill_mas_style_agent,
    skill_mas_results_model_label,
)
from vita.utils.utils import RESULTS_DIR, maskill_skip_vitabench_trace_export


def sanitize_component(name: str) -> str:
    """Make a single path component safe (no slashes/spaces)."""
    s = (name or "").split("/")[-1]
    out = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return out or "unknown"


def skill_mas_trace_suffix(skill_dir: str | None) -> str:
    """
    Derive suffix under skill_mas_process_traces/, same idea as drb_bridge/run.sh:
    if path contains '/skills/', use the relative part after it; else basename sanitized.
    """
    default = "default"
    path = str(Path(skill_dir or str(default_skill_mas_workspace_dir())).resolve())
    mark = "/skills/"
    if mark in path:
        rel = path.split(mark, 1)[1].strip("/")
        return rel if rel else default
    base = Path(path).name
    safe = sanitize_component(base)
    return safe if safe else default


def results_subdir_for_agent(agent: str, skill_suffix: str | None) -> Path:
    """Path segments under results/<model>/ for this agent type."""
    if agent in ("llm_agent", "one_shot_llm_agent"):
        return Path("single_agent")
    if agent in ("aorchestra_agent", "aorchestra_bridge_agent"):
        return Path("aorchestra")
    if agent == "skill_mas_agent":
        suf = skill_suffix or "default"
        parts = [p for p in suf.split("/") if p]
        return Path("skill_mas_process_traces", *parts)
    if agent == "preload_agent":
        suf = skill_suffix or "default"
        parts = ["preload_agent", *[p for p in suf.split("/") if p]]
        return Path("skill_mas_process_traces", *parts)
    return Path(sanitize_component(agent))


def mirror_simulation_to_results(src: Path, config: RunConfig) -> Optional[Path]:
    """
    Copy src into RESULTS_DIR/<agent_llm_safe>/<method_subdir>/basename(src).
    Skips mas_evolve_* filenames and missing sources.
    """
    if maskill_skip_vitabench_trace_export():
        logger.info("[results_mirror] skip copy into results/ (MASKILL_SKIP_VITABENCH_TRACE_EXPORT)")
        return None
    src = Path(src)
    if not src.is_file():
        logger.warning(f"[results_mirror] source missing, skip: {src}")
        return None
    name = src.name
    if name.startswith("mas_evolve_"):
        logger.info(f"[results_mirror] skip optimize artifact: {name}")
        return None

    model_safe = sanitize_component(
        skill_mas_results_model_label(
            agent=config.agent,
            llm_agent=config.llm_agent,
            planner_llm=config.planner_llm,
        )
    )
    skill_suffix: str | None = None
    if is_skill_mas_style_agent(config.agent):
        skill_suffix = skill_mas_trace_suffix(config.skill_mas_dir)

    sub = results_subdir_for_agent(config.agent, skill_suffix)
    dest_dir = RESULTS_DIR / model_safe / sub
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    shutil.copy2(src, dest)
    logger.info(f"[results_mirror] copied to {dest}")

    summary_src = src.parent / f"{src.stem}_summary.json"
    if summary_src.is_file():
        summary_dest = dest_dir / summary_src.name
        shutil.copy2(summary_src, summary_dest)
        logger.info(f"[results_mirror] copied summary to {summary_dest}")

    return dest
