"""Resolve Skill_MAS checkout paths for VitaBench (same conventions as hlemath / BrowseComp-Plus)."""

from __future__ import annotations

import os
from pathlib import Path


def _import_paths():
    try:
        from Skill_MAS.utils.paths import (
            INIT_SKILL_DIR,
            MODEL_CONFIG_JSON,
            PACKAGE_ROOT,
            SKILL_MAS_ROOT,
            VITABENCH_ROOT,
        )
        return PACKAGE_ROOT, SKILL_MAS_ROOT, VITABENCH_ROOT, INIT_SKILL_DIR, MODEL_CONFIG_JSON
    except ImportError:
        # Fallback when only vita src is on PYTHONPATH (e.g. isolated vita.cli)
        sm = Path(__file__).resolve().parents[4]
        pkg = sm.parent
        vb = sm / "dataset" / "vitabench"
        return pkg, sm, vb, sm / "init_skill", sm / "skill_mas" / "model_config.json"


def ant_repo_root() -> Path:
    """Parent of ``Skill_MAS`` (PYTHONPATH root for ``import Skill_MAS``)."""
    return _import_paths()[0]


def skill_mas_root() -> Path:
    """``SKILL_MAS_ROOT`` env, else ``<package>/Skill_MAS``."""
    env = (os.environ.get("SKILL_MAS_ROOT") or "").strip()
    if env:
        return Path(env).resolve()
    return _import_paths()[1]


def vitabench_root() -> Path:
    """VitaBench checkout root (``dataset/vitabench``)."""
    return _import_paths()[2]


def skill_mas_model_config_path() -> Path:
    """``Skill_MAS/skill_mas/model_config.json`` (pricing + per-model API routing)."""
    env = (os.environ.get("SKILL_MAS_MODEL_CONFIG") or "").strip()
    if env:
        return Path(env).resolve()
    return _import_paths()[4]


def default_skill_mas_init_skill_path() -> Path:
    """Init markdown: ``SKILL_MAS_INIT_SKILL`` or ``<skill_mas_root>/init_skill/SKILL.md``."""
    env = (os.environ.get("SKILL_MAS_INIT_SKILL") or "").strip()
    if env:
        return Path(env).resolve()
    return skill_mas_root() / "init_skill" / "SKILL.md"


SKILL_MAS_STYLE_AGENTS = frozenset({"skill_mas_agent", "preload_agent"})


def is_skill_mas_style_agent(agent: str | None) -> bool:
    """Agents that run Skill-MAS build + Vita tool bridge (share trace/summary helpers)."""
    return (agent or "").strip() in SKILL_MAS_STYLE_AGENTS


def preload_agent_planner_model_name(planner_llm: str | None = None) -> str:
    """Resolved planner model short name for preload_agent artifact paths."""
    env = (os.environ.get("VITABENCH_SKILL_MAS_PLANNER_LLM") or "").strip()
    name = (planner_llm or env or "deepseek-v4-flash").strip()
    return name.split("/")[-1]


def skill_mas_results_model_label(
    *,
    agent: str,
    llm_agent: str,
    planner_llm: str | None = None,
) -> str:
    """Top-level ``results/<model>/`` folder name; preload_agent uses planner, not executor."""
    if (agent or "").strip() == "preload_agent":
        return preload_agent_planner_model_name(planner_llm)
    return (llm_agent or "unknown").split("/")[-1]


def default_skill_mas_workspace_dir() -> Path:
    """
    Skill workspace for ``skill_mas_agent`` (phase banks or flat skills).
    ``SKILL_MAS_DIR`` / ``SKILL_MAS_WORKSPACE`` env, else ``<skill_mas_root>/init_skill``.
    """
    for key in ("SKILL_MAS_DIR", "SKILL_MAS_WORKSPACE"):
        env = (os.environ.get(key) or "").strip()
        if env:
            return Path(env).resolve()
    return skill_mas_root() / "init_skill"
