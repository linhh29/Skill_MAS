"""Lazy loader: invoke the AFlow VITABENCH graph from VitaBench (Ant repo layout).

``run_automatic_mas_aflow_step`` only runs the configured AFlow ``Workflow`` (instructions in).
``AutomaticMasBridgeAgent`` then chains ``SkillMASAgent._run_tool_loop`` with the same Vita prompts
as ``run_tool_bridge_for_prompt``, so tool calls execute against the task environment.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger


def _ant_repo_root() -> Path:
    from vita.skill_mas_paths import ant_repo_root

    return ant_repo_root()


def run_automatic_mas_aflow_step(task_dict: dict, exec_model_name: str | None) -> tuple[str, dict[str, Any]]:
    """
    Ensure PYTHONPATH includes ``automatic_mas/AFlow``, ``dataset/vitabench/src``, and repo root,
    then call ``benchmarks.vita_aflow_bridge.run_aflow_vita_workflow_sync``.

    Returns the graph reply text and metadata (including ``estimated_cost_usd``). Vita simulator tools are
    **not** invoked here—see ``AutomaticMasBridgeAgent._run_skill_mas_build_pipeline`` for the tool loop.
    """
    from vita.skill_mas_paths import vitabench_root

    root = _ant_repo_root()
    aflow = root / "automatic_mas" / "AFlow"
    vsrc = vitabench_root() / "src"
    for p in (aflow, vsrc, root):
        ps = str(p.resolve())
        if ps not in sys.path:
            sys.path.insert(0, ps)

    model = exec_model_name or os.environ.get("AFLOW_VITA_EXEC_MODEL") or "qwen3.5-plus"

    try:
        from benchmarks.vita_aflow_bridge import run_aflow_vita_workflow_sync  # noqa: E402
    except Exception as exc:
        logger.exception("Failed to import AFlow vita bridge: %s", exc)
        raise

    return run_aflow_vita_workflow_sync(task_dict, exec_model_name=str(model))
