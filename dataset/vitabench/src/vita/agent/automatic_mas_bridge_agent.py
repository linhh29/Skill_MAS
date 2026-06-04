"""
AutomaticMasBridgeAgent — VitaBench agent that mirrors ``SkillMASAgent`` orchestration:

- Same constructor / tools / ``vita_max_tool_rounds`` as ``SkillMASAgent``.
- Replaces Skill-MAS ``build.py`` with an **upstream runner** (lazy-loaded), then runs the same **multi-round**
  Vita tool loop as ``SkillMASAgent.run_tool_bridge_for_prompt`` (``_run_tool_loop``).

Runners (same pattern as ``aflow_bridge_runner.py``):

- **AFlow**: ``vita.agent.aflow_bridge_runner.run_automatic_mas_aflow_step`` (default).
- **AOrchestra**: ``vita.agent.aorchestra_bridge_runner.run_automatic_mas_aorchestra_step``.

Pick backend with env ``VITA_AUTOMATIC_MAS_BRIDGE_BACKEND`` = ``aflow`` | ``aorchestra`` (default ``aflow``).
Registry: ``AutomaticMasBridgeAgentOrchestraAlias`` is registered as both ``aorchestra_bridge_agent`` and the
legacy short name ``aorchestra_agent`` (same behavior — AOrchestra runner without env).

Register as ``automatic_mas_bridge_agent``. Use with ``run_task(..., agent="automatic_mas_bridge_agent")`` or set
``VITA_AUTOMATIC_MAS_BRIDGE_BACKEND=aorchestra``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from loguru import logger

from vita.agent.skill_mas_agent import (
    SkillMASAgent,
    _VITA_SKILL_MAS_SYSTEM_SUFFIX,
    _VITA_SKILL_MAS_USER_FOOTER,
    _VITA_SKILL_MAS_USER_HEADER,
    _to_jsonable,
)
from vita.data_model.message import AssistantMessage
from vita.data_model.tasks import Task


def _resolve_automatic_mas_bridge_backend(agent: "AutomaticMasBridgeAgent") -> str:
    """Env ``VITA_AUTOMATIC_MAS_BRIDGE_BACKEND`` overrides optional class-level default."""
    env = (os.environ.get("VITA_AUTOMATIC_MAS_BRIDGE_BACKEND") or "").strip().lower()
    if env in ("aflow", "aorchestra"):
        return env
    forced = getattr(type(agent), "_forced_automatic_mas_bridge_backend", None)
    if forced in ("aflow", "aorchestra"):
        return forced
    return "aflow"


class AutomaticMasBridgeAgent(SkillMASAgent):
    """
    Subclass of ``SkillMASAgent`` that overrides only the **generation** stage (Skill-MAS pipeline → runner).

    Receives the full ``Task`` from ``run._run_task_internal`` so the bridge can pass stable task ids /
    metadata upstream even though the graph input text follows ``StaticInputUser`` (instructions only).
    """

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
        task: Optional[Task] = None,
    ):
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
        )
        self._task: Optional[Task] = task

    def _task_dict(self, task_text: str) -> dict[str, Any]:
        if self._task is not None:
            try:
                return self._task.model_dump(mode="json")
            except Exception:
                return _to_jsonable(self._task)
        return {"instructions": (task_text or "").strip(), "id": ""}

    def _run_skill_mas_build_pipeline(self, task_text: str) -> tuple[AssistantMessage, dict[str, Any]]:
        backend = _resolve_automatic_mas_bridge_backend(self)
        task_dict = self._task_dict(task_text)

        if backend == "aorchestra":
            from vita.agent.aorchestra_bridge_runner import run_automatic_mas_aorchestra_step

            runner_label = "AOrchestra output"
            stage_role = "aorchestra_then_vita_tool_bridge"
            routing_mode = "automatic_mas_aorchestra_then_vita_tools"
            bridge_meta_key = "aorchestra_bridge"
            text_before_key = "aorchestra_text_before_tools"
            empty_hint = "I could not produce a final answer from the AOrchestra run."
            step_name = "AOrchestra"
            try:
                final_text, bridge_meta = run_automatic_mas_aorchestra_step(
                    task_dict,
                    exec_model_name=self.llm,
                    vita_tool_bridge_fn=self.run_tool_bridge_for_prompt,
                )
            except Exception as exc:
                logger.exception("AutomaticMasBridgeAgent AOrchestra step failed: %s", exc)
                final_text = f"[AutomaticMasBridgeAgent] AOrchestra bridge failed: {exc}"
                bridge_meta = {"error": str(exc), "estimated_cost_usd": 0.0}
        else:
            from vita.agent.aflow_bridge_runner import run_automatic_mas_aflow_step

            runner_label = "AFlow workflow output"
            stage_role = "aflow_then_vita_tool_bridge"
            routing_mode = "automatic_mas_aflow_then_vita_tools"
            bridge_meta_key = "aflow_bridge"
            text_before_key = "aflow_text_before_tools"
            empty_hint = "I could not produce a final answer from the AFlow workflow."
            step_name = "AFlow"
            try:
                final_text, bridge_meta = run_automatic_mas_aflow_step(
                    task_dict,
                    exec_model_name=self.llm,
                )
            except Exception as exc:
                logger.exception("AutomaticMasBridgeAgent AFlow step failed: %s", exc)
                final_text = f"[AutomaticMasBridgeAgent] AFlow bridge failed: {exc}"
                bridge_meta = {"error": str(exc), "estimated_cost_usd": 0.0}

        upstream_text = str(final_text or "").strip()
        if not upstream_text:
            upstream_text = empty_hint

        merged = (
            f"[{runner_label} — use as guidance and complete the task via Vita environment tools]\n\n"
            f"{upstream_text}\n\n"
            "---\n\n"
            "[Original instructions]\n"
            f"{task_text.strip()}"
        )
        combined_system = self.system_prompt + "\n\n" + _VITA_SKILL_MAS_SYSTEM_SUFFIX
        constrained_prompt = _VITA_SKILL_MAS_USER_HEADER + merged + _VITA_SKILL_MAS_USER_FOOTER

        bridge_assistant: AssistantMessage
        try:
            tool_out, used_tools, bridge_assistant, internal_trace = self._run_tool_loop(
                system_prompt=combined_system,
                user_prompt=constrained_prompt,
                stage_role=stage_role,
                max_rounds=self._bridge_max_tool_rounds(),
            )
        except Exception as exc:
            logger.exception("AutomaticMasBridgeAgent Vita tool loop after %s failed: %s", step_name, exc)
            tool_out = upstream_text
            used_tools = []
            internal_trace = [{"stage_role": stage_role, "error": str(exc)}]
            bridge_assistant = AssistantMessage(role="assistant", content=tool_out)

        display = (tool_out or "").strip() or upstream_text
        upstream_cost = float(bridge_meta.get("estimated_cost_usd", 0.0) or 0.0)
        vita_cost = float(getattr(bridge_assistant, "cost", None) or 0.0)

        reply = AssistantMessage(role="assistant", content=display)
        reply.usage = dict(getattr(bridge_assistant, "usage", None) or {})
        reply.cost = upstream_cost + vita_cost

        upstream_json = _to_jsonable(bridge_meta)
        meta: dict[str, Any] = {
            "schema_version": "automatic_mas_bridge_meta/3",
            "bridge_backend": backend,
            "routing_mode": routing_mode,
            "task_id": task_dict.get("id"),
            "class_name": None,
            "success": "error" not in bridge_meta,
            "upstream_bridge": upstream_json,
            bridge_meta_key: upstream_json,
            "used_tools": sorted(set(used_tools)),
            "internal_tool_trace": internal_trace,
            "workflow_state": {},
            text_before_key: upstream_text,
            "final_output": display,
        }
        return reply, meta


class AutomaticMasBridgeAgentOrchestraAlias(AutomaticMasBridgeAgent):
    """
    Registry name ``aorchestra_bridge_agent``: same behavior as ``AutomaticMasBridgeAgent`` with default
    AOrchestra runner. Prefer env ``VITA_AUTOMATIC_MAS_BRIDGE_BACKEND=aorchestra`` on the base class instead.
    """

    _forced_automatic_mas_bridge_backend = "aorchestra"
