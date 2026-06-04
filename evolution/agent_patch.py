"""Runtime patches for SkillMASAgent (SKILL_MAS_DIR + tool rounds)."""

from __future__ import annotations

import os
from typing import Any

_INIT_DONE = False
_STAGE_DONE = False


def apply_skill_mas_patches() -> None:
    global _INIT_DONE, _STAGE_DONE
    from vita.agent import skill_mas_agent as sm

    if not _INIT_DONE:
        _orig = sm.SkillMASAgent.__init__

        def _init(
            self: Any,
            tools: Any,
            domain_policy: str,
            llm: Any = None,
            llm_args: Any = None,
            time: Any = None,
            enable_think: bool = False,
            language: Any = None,
            skill_dir: Any = None,
            *,
            vita_max_tool_rounds: Any = None,
        ) -> None:
            skill_dir = skill_dir or os.environ.get("SKILL_MAS_DIR")
            return _orig(
                self,
                tools,
                domain_policy,
                llm=llm,
                llm_args=llm_args,
                time=time,
                enable_think=enable_think,
                language=language,
                skill_dir=skill_dir,
                vita_max_tool_rounds=vita_max_tool_rounds,
            )

        sm.SkillMASAgent.__init__ = _init  # type: ignore[method-assign]
        _INIT_DONE = True

    if not _STAGE_DONE and hasattr(sm.SkillMASAgent, "_run_stage_with_tool_loop"):
        _orig_s = sm.SkillMASAgent._run_stage_with_tool_loop

        def _stage(
            self: Any,
            *,
            system_prompt: str,
            user_prompt: str,
            allow_tools: bool,
            stage_role: str = "",
            max_rounds: int = 10,
        ) -> Any:
            env_cap = os.environ.get("SKILL_MAS_MAX_TOOL_ROUNDS")
            if env_cap is not None:
                try:
                    max_rounds = int(env_cap)
                except ValueError:
                    max_rounds = 10
            else:
                max_rounds = 10
            return _orig_s(
                self,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                allow_tools=allow_tools,
                stage_role=stage_role,
                max_rounds=max_rounds,
            )

        sm.SkillMASAgent._run_stage_with_tool_loop = _stage  # type: ignore[method-assign]
        _STAGE_DONE = True

