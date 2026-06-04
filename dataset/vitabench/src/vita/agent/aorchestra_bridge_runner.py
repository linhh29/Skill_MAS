"""
Invoke AOrchestra MainAgent+SubAgent (``run_orchestra_qa_once``) for VitaBench ``AutomaticMas*`` bridge.

Same role as ``aflow_bridge_runner.run_automatic_mas_aflow_step`` but uses
``automatic_mas/AOrchestra`` instead of the AFlow VITABENCH ``Workflow`` graph.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Callable

from loguru import logger


def _ant_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def run_automatic_mas_aorchestra_step(
    task_dict: dict,
    exec_model_name: str | None,
    *,
    vita_tool_bridge_fn: Callable[[str], tuple[str, dict[str, Any]]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Add Ant root + ``automatic_mas/AOrchestra`` to ``sys.path``, then run instructions-only
    Orchestra QA (``vita_static_user_message_from_task``) and return (text, meta with cost).

    When ``vita_tool_bridge_fn`` is set (e.g. ``SkillMASAgent.run_tool_bridge_for_prompt``), each
    MainAgent orchestration step triggers the VitaBench env tool loop between rounds. This module is
    VitaBench-only; it passes ``vitabench_orchestra_step_bridge=True`` so ``run_orchestra_qa_once`` does not
    run that logic for HLEMATH/DRB/OrchestraQAGraph (which keep the default ``False``).
    """
    root = _ant_repo_root()
    aorch = root / "automatic_mas" / "AOrchestra"
    for p in (aorch, root):
        ps = str(p.resolve())
        if ps not in sys.path:
            sys.path.insert(0, ps)

    model = exec_model_name or os.environ.get("AO_ORCHESTRA_EXEC_MODEL") or "qwen3-max"
    raw = (os.environ.get("AO_ORCHESTRA_SUB_MODELS") or "").strip()
    sub_models = [x.strip() for x in raw.split(",") if x.strip()] or [model]
    max_attempts = int(os.environ.get("AO_ORCHESTRA_MAX_ATTEMPTS", "10"))

    try:
        from benchmark.aflow_datasets.vita_input import (  # noqa: E402
            vita_static_user_message_from_task,
        )
        from benchmark.aflow_datasets.orchestra_graph import (  # noqa: E402
            run_orchestra_qa_once,
        )
    except Exception as exc:
        logger.exception("Failed to import AOrchestra bridge: %s", exc)
        raise

    text = vita_static_user_message_from_task(task_dict)
    out, cost = asyncio.run(
        run_orchestra_qa_once(
            text,
            main_model=str(model),
            sub_models=sub_models,
            max_attempts=max_attempts,
            task_id=str(task_dict.get("id", ""))[:32] or None,
            vita_tool_bridge_fn=vita_tool_bridge_fn,
            vitabench_orchestra_step_bridge=True,
        )
    )
    meta: dict[str, Any] = {
        "estimated_cost_usd": float(cost or 0.0),
        "bridge": "aorchestra",
        "main_model": str(model),
        "sub_models": sub_models,
    }
    return (out if isinstance(out, str) else str(out)), meta
