"""PreloadAgent: strong Skill-MAS planner + lightweight executor on VitaBench.

Uses a separate (stronger) LLM for Skill-MAS Stages 1–3 and code repair, while
``--agent-llm`` (typically gpt-5.4-nano) runs sub-agent text reasoning and the
Vita environment tool bridge.

Planner model resolution (first match wins):
1. ``planner_llm`` constructor arg (from ``--planner-llm`` / ``RunConfig.planner_llm``)
2. env ``VITABENCH_SKILL_MAS_PLANNER_LLM``
3. default ``deepseek-v4-flash``
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Optional

from vita.agent.skill_mas_agent import SkillMASAgent
from vita.config import models
from vita.data_model.message import AssistantMessage


def default_preload_planner_llm() -> str:
    env = (os.environ.get("VITABENCH_SKILL_MAS_PLANNER_LLM") or "").strip()
    if env:
        return env
    return "deepseek-v4-flash"


class PreloadAgent(SkillMASAgent):
    """Skill-MAS build with a dedicated planner LLM; executor uses ``self.llm``."""

    def __init__(
        self,
        tools: list,
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        time=None,
        enable_think: bool = False,
        language: str = None,
        skill_dir: Optional[str] = None,
        *,
        vita_max_tool_rounds: Optional[int] = None,
        planner_llm: Optional[str] = None,
        planner_llm_args: Optional[dict] = None,
    ):
        resolved_planner = (planner_llm or default_preload_planner_llm()).strip()
        resolved_planner_args = (
            deepcopy(planner_llm_args)
            if planner_llm_args is not None
            else deepcopy(models.get(resolved_planner, {}))
        )
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            time=time,
            enable_think=enable_think,
            language=language,
            skill_dir=skill_dir,
            vita_max_tool_rounds=vita_max_tool_rounds,
            planner_llm=resolved_planner,
            planner_llm_args=resolved_planner_args,
        )

    def _run_skill_mas_build_pipeline(self, task_text: str) -> tuple[AssistantMessage, dict[str, Any]]:
        reply, meta = super()._run_skill_mas_build_pipeline(task_text)
        meta = dict(meta)
        meta["schema_version"] = "skill_mas_message_meta/4"
        meta["routing_mode"] = "preload_agent_skill_mas_build_py"
        meta["planner_llm"] = self._skill_mas_planner_model()
        meta["executor_llm"] = self.llm
        return reply, meta
