"""SkillMASAgent: VitaBench ↔ ``Skill_MAS/skill_mas/build.py`` adapter only.

The 3-stage MAS lives in Skill_MAS; here we only wire planners and the Vita tool bridge.

Bridge behavior merges (1) ``domain_policy`` + optional VitaBench execution hints
(``_VITA_SKILL_MAS_*``), (2) ``vita_max_tool_rounds`` injected from ``run.py`` / ``max_steps``,
and (3) full tool strings to the model (truncation only in trace JSON previews).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from copy import deepcopy
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel

from vita.agent.base import LocalAgent, ValidAgentInputMessage, is_valid_agent_history_message
from vita.data_model.message import APICompatibleMessage, AssistantMessage, Message, SystemMessage, ToolMessage, UserMessage
from vita.environment.tool import Tool
from vita.skill_mas_paths import default_skill_mas_init_skill_path, default_skill_mas_workspace_dir, skill_mas_root
from vita.utils.llm_utils import generate
from vita.utils.utils import get_now, get_weekday


def default_skill_mas_dir() -> str:
    env = (os.environ.get("SKILL_MAS_DIR") or os.environ.get("SKILL_MAS_WORKSPACE") or "").strip()
    if env:
        return env
    return str(default_skill_mas_workspace_dir())


DEFAULT_SKILL_MAS_DIR = default_skill_mas_dir()

# VitaBench-only hints appended to the Skill-MAS sub-agent bridge (does not change Skill_MAS build stages).
_VITA_SKILL_MAS_SYSTEM_SUFFIX = (
    "## VitaBench tool execution (mandatory)\n"
    "- Bookings, orders, tickets, deliveries, in-store purchases, and payments MUST be executed via Vita "
    "environment tools until the simulator shows paid/confirmed (or an explicit cancel/replace flow).\n"
    "- Recommendation-only or planning-only text is NOT completion when the sub-task implies a real transaction.\n"
    "- Do not claim purchase success without tool-backed order/booking identifiers and status.\n"
    "- SUB-AGENT MODE: you are an intermediate Skill-MAS workflow node, NOT the final VitaBench turn. "
    "NEVER output '###STOP###' in this step — the outer SkillMASAgent adds it once after all sub-agents finish.\n"
)

_VITA_SKILL_MAS_USER_HEADER = (
    "**READ FIRST — VitaBench execution**\n"
    "- If this sub-task involves booking, ordering, purchasing, paying, or cancelling something in the "
    "environment, you MUST call Vita tools (including payment tools when applicable) until the action is "
    "actually reflected in tool results.\n"
    "- Once you have chosen a concrete option (store/product/time/seat/address), execute it immediately via "
    "tools; do not stop at comparison tables or suggestions.\n"
    "- Downstream rubrics often require real orders: instore may need both reservation AND meal/order steps "
    "when the task asks for a套餐/下单.\n"
    "\n---\n\n"
)

_VITA_SKILL_MAS_USER_FOOTER = (
    "\n\n---\n\nVITABENCH EXECUTION CONSTRAINTS (detail):\n"
    "- If this sub-task requires booking/ordering/purchasing/canceling/modifying any real object, "
    "you MUST call Vita tools to execute it.\n"
    "- If you have formed a concrete executable plan (selected item/store/time/address/seat/etc.), "
    "you MUST immediately execute that plan via Vita tools; planning/recommendation text alone is "
    "NOT completion.\n"
    "- For any created order/ticket/hotel/attraction/delivery, you MUST actually create the "
    "corresponding Vita order and complete payment via Vita tools when the domain requires a separate "
    "payment step; it cannot stay in an unpaid/pending state or be \"ordered on behalf\" only.\n"
    "- Do NOT stop at recommendation/search/comparison once a concrete option is chosen; continue "
    "tool calls until final executable status is reached (paid/confirmed, or explicitly cancelled "
    "with reason and alternative attempted).\n"
    "- Never claim success from reasoning alone; include concrete tool-backed results.\n"
    "- If an action fails, retry with reasonable alternatives and report failure transparently.\n"
    "- Return concise, downstream-usable output with key IDs/times/addresses/status.\n"
    "- Do NOT output '###STOP###' in this sub-task (reserved for the final user reply only).\n"
    "- Do NOT return YAML/JSON field manifests with '待填' or '执行节点' placeholders instead of real tool outcomes."
)


def _prompt_likely_requires_vita_tools(prompt: str) -> bool:
    p = (prompt or "").lower()
    keys = (
        "book",
        "order",
        "pay",
        "reserve",
        "deliver",
        "executor",
        "transaction",
        "create_",
        "mandatory tool",
        "vita environment",
        "vita tool",
        "下单",
        "预约",
        "外卖",
        "支付",
        "ticket",
        "restaurant",
        "script",
        "spa",
        "hotel",
        "洗浴",
        "汗蒸",
        "mandatory tool execution",
        "vita_transaction_executor",
    )
    return any(k in p for k in keys)


def _parse_vita_bridge_payload(payload: Any) -> tuple[str, bool]:
    """Return (prompt_text, require_tool_call) from bridge input."""
    if isinstance(payload, dict):
        require = bool(payload.get("require_tool_call"))
        return _normalize_vita_bridge_prompt(payload), require
    text = payload if isinstance(payload, str) else str(payload or "")
    require = "mandatory tool execution" in text.lower()
    return text, require


def _vitabench_skill_mas_max_tool_rounds() -> int:
    """Align inner tool iterations with typical ``python -m vita.cli run --max-steps`` (default 300)."""
    raw = (os.environ.get("VITABENCH_SKILL_MAS_MAX_TOOL_ROUNDS") or "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return 300


class SkillMASAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


def _truncate(text: str, max_chars: Optional[int] = None) -> str:
    """If ``max_chars`` is None, return full text (matches orchestrator tool messages)."""
    t = text or ""
    if max_chars is None or len(t) <= max_chars:
        return t
    return t[:max_chars] + "...(truncated)"


def _trace_preview(text: str, max_chars: int = 4000) -> str:
    """Short preview only for ``internal_tool_trace`` JSON; model sees full tool strings."""
    return _truncate(text, max_chars)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return repr(value)


def _aggregate_usage_from_state(state: dict[str, Any]) -> dict[str, Any]:
    acc = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    if not isinstance(state, dict):
        return acc
    for k, v in state.items():
        if not str(k).startswith("usage_") or not isinstance(v, dict):
            continue
        acc["prompt_tokens"] += int(v.get("prompt_tokens", v.get("input_tokens", 0)) or 0)
        acc["completion_tokens"] += int(v.get("completion_tokens", v.get("output_tokens", 0)) or 0)
        acc["total_tokens"] += int(
            v.get("total_tokens", (v.get("prompt_tokens", 0) or 0) + (v.get("completion_tokens", 0) or 0)) or 0
        )
        acc["estimated_cost_usd"] += float(v.get("estimated_cost_usd", 0.0) or 0.0)
    return acc


def _normalize_vita_bridge_prompt(payload: Any) -> str:
    """
    Normalize Vita tool-bridge input into a stable prompt string.

    Backward compatible:
    - str input: passthrough
    - dict input: render deterministic sections (role/instruction/plan/task)
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        require = bool(payload.get("require_tool_call"))
        role = str(payload.get("role_name", "") or "").strip()
        instruction = str(payload.get("role_instruction", "") or "").strip()
        action_plan = str(payload.get("action_plan", "") or "").strip()
        task_input = str(payload.get("task_input", payload.get("prompt", "")) or "").strip()
        parts: list[str] = []
        parts.append(
            "[VITA TOOL RUNTIME — IMPORTANT]\n"
            "Vita environment tools are connected in this execution step via the runner bridge."
        )
        parts.append(
            "[INTERMEDIATE SUB-AGENT — NO STOP TOKEN]\n"
            "Do NOT output '###STOP###' in this step. You are an internal workflow node; "
            "only the outer final reply may end the VitaBench session."
        )
        if role:
            parts.append(f"[Sub-agent role]\n{role}")
        if instruction:
            parts.append(f"[Sub-task instruction]\n{instruction}")
        if action_plan:
            parts.append(f"[Pre-computed action plan — READ THIS FIRST before using tools]\n{action_plan}")
        if task_input:
            parts.append(f"[Task input]\n{task_input}")
        if require:
            parts.append(
                "[MANDATORY TOOL EXECUTION]\n"
                "You MUST call at least one Vita environment tool (search / create / pay / book) "
                "before finishing this sub-task."
            )
        if parts:
            return "\n\n".join(parts)
        return str(payload)
    return str(payload or "")


class SkillMASAgent(LocalAgent[SkillMASAgentState]):
    def __init__(
        self,
        tools: list[Tool],
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
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.planner_llm = (planner_llm or "").strip() or None
        self.planner_llm_args = deepcopy(planner_llm_args) if planner_llm_args is not None else None
        self.time = (time + " " + get_weekday(time, language)) if time is not None else get_now("%Y-%m-%d %H:%M:%S")
        self.enable_think = enable_think
        _dir = (skill_dir or os.environ.get("SKILL_MAS_DIR") or "").strip()
        self.skill_dir = _dir or default_skill_mas_dir()
        # Injected by VitaBench ``run._run_task_internal`` from ``max_steps``; env fallback for ad-hoc use.
        self._vita_max_tool_rounds: Optional[int] = vita_max_tool_rounds

    def _skill_mas_planner_model(self) -> str:
        if self.planner_llm:
            return self.planner_llm
        if self.llm is None:
            raise ValueError("LLM is not set")
        return self.llm

    def _skill_mas_planner_llm_args(self) -> dict:
        if self.planner_llm_args is not None:
            return self.planner_llm_args
        return self.llm_args

    def set_seed(self, seed: int) -> None:
        """Apply VitaBench run seed to ``generate`` (same pattern as ``LLMAgent``); avoids ``BaseAgent`` warnings."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed")
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed
        if self.planner_llm_args is not None:
            self.planner_llm_args["seed"] = seed

    @property
    def system_prompt(self) -> str:
        return self.domain_policy.format(time=self.time)

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> SkillMASAgentState:
        message_history = message_history or []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return SkillMASAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def _ensure_stop_token(self, reply: AssistantMessage) -> None:
        if reply.is_tool_call():
            return
        text = reply.content or ""
        if self.STOP_TOKEN not in text:
            reply.content = text + "\n\n" + self.STOP_TOKEN

    def _bridge_max_tool_rounds(self) -> int:
        if self._vita_max_tool_rounds is not None:
            return max(1, int(self._vita_max_tool_rounds))
        return _vitabench_skill_mas_max_tool_rounds()

    def _dispatch_assistant_tool_calls(
        self,
        assistant: AssistantMessage,
        *,
        tool_map: dict[str, Tool],
        used_tools: list[str],
        trace: list[dict[str, Any]],
        stage_role: str,
    ) -> list[ToolMessage]:
        tool_msgs: list[ToolMessage] = []
        round_calls: list[dict[str, Any]] = []
        for tc in assistant.tool_calls or []:
            used_tools.append(tc.name)
            tool_obj = tool_map.get(tc.name)
            if tool_obj is None:
                err = f"Tool '{tc.name}' not found."
                tool_msgs.append(
                    ToolMessage(
                        id=tc.id or "",
                        name=tc.name,
                        role="tool",
                        requestor="assistant",
                        content=err,
                        error=True,
                    )
                )
                round_calls.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result_preview": _trace_preview(err),
                        "error": True,
                    }
                )
                continue
            try:
                result = tool_obj(**(tc.arguments or {}))
                body = str(result)
                tool_msgs.append(
                    ToolMessage(
                        id=tc.id or "",
                        name=tc.name,
                        role="tool",
                        requestor="assistant",
                        content=body,
                        error=False,
                    )
                )
                round_calls.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result_preview": _trace_preview(body),
                        "error": False,
                    }
                )
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                tool_msgs.append(
                    ToolMessage(
                        id=tc.id or "",
                        name=tc.name,
                        role="tool",
                        requestor="assistant",
                        content=err,
                        error=True,
                    )
                )
                round_calls.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result_preview": _trace_preview(err),
                        "error": True,
                    }
                )
        if round_calls:
            trace.append({"stage_role": stage_role, "tool_round": round_calls})
        return tool_msgs

    def _run_tool_loop(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stage_role: str,
        max_rounds: Optional[int] = None,
        require_tool_call: bool = False,
    ) -> tuple[str, list[str], AssistantMessage, list[dict[str, Any]]]:
        msgs: list[APICompatibleMessage] = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]
        tool_map = {t.name: t for t in self.tools}
        used_tools: list[str] = []
        trace: list[dict[str, Any]] = []
        assistant: Optional[AssistantMessage] = None

        rounds = max_rounds if max_rounds is not None else self._bridge_max_tool_rounds()
        has_tools = bool(self.tools)
        must_use_tools = bool(require_tool_call or _prompt_likely_requires_vita_tools(user_prompt))

        for round_idx in range(rounds):
            tool_choice: Optional[str] = None
            if has_tools and round_idx == 0 and require_tool_call:
                tool_choice = "required"
            assistant = generate(
                model=self.llm,
                tools=self.tools,
                messages=msgs,
                tool_choice=tool_choice,
                enable_think=self.enable_think,
                **self.llm_args,
            )
            if assistant is None:
                assistant = AssistantMessage(role="assistant", content="")
            msgs.append(assistant)
            if not assistant.is_tool_call():
                break

            tool_msgs = self._dispatch_assistant_tool_calls(
                assistant,
                tool_map=tool_map,
                used_tools=used_tools,
                trace=trace,
                stage_role=stage_role,
            )
            msgs.extend(tool_msgs)

        if not used_tools and must_use_tools and has_tools:
            print(f"[{stage_role}] zero tool calls; forcing required-tool retry", flush=True)
            msgs.append(
                UserMessage(
                    role="user",
                    content=(
                        "You stopped without calling any Vita environment tools. "
                        "Tools ARE connected in this step. "
                        "Empty upstream lists or 'missing fields' are NOT valid reasons to abstain — "
                        "use GLOBAL TASK CONTEXT and search the environment yourself. "
                        "You MUST call at least one relevant tool (search / create order or booking / pay) "
                        "before providing your final answer."
                    ),
                )
            )
            retry_rounds = min(8, max(1, rounds))
            for retry_idx in range(retry_rounds):
                tool_choice = "required" if retry_idx == 0 or not used_tools else None
                assistant = generate(
                    model=self.llm,
                    tools=self.tools,
                    messages=msgs,
                    tool_choice=tool_choice,
                    enable_think=self.enable_think,
                    **self.llm_args,
                )
                if assistant is None:
                    assistant = AssistantMessage(role="assistant", content="")
                msgs.append(assistant)
                if not assistant.is_tool_call():
                    continue
                tool_msgs = self._dispatch_assistant_tool_calls(
                    assistant,
                    tool_map=tool_map,
                    used_tools=used_tools,
                    trace=trace,
                    stage_role=stage_role,
                )
                msgs.extend(tool_msgs)
                if used_tools:
                    break

        if assistant is None:
            assistant = AssistantMessage(role="assistant", content="")
        out = assistant.content or ""
        if not out.strip():
            summary = generate(
                model=self.llm,
                tools=[],
                messages=msgs + [UserMessage(role="user", content="Summarize key tool outcomes and conclusion briefly.")],
                enable_think=self.enable_think,
                **self.llm_args,
            )
            if summary is None:
                summary = AssistantMessage(role="assistant", content="")
            out = (summary.content or "").strip()
            if out:
                assistant = summary
        return out, used_tools, assistant, trace

    def run_tool_bridge_for_prompt(
        self, prompt: str, *, max_tool_rounds: Optional[int] = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Same inner bridge as Skill_MAS ``vita_tool`` / ``tool_call_fn`` (see ``sub_agent.py``): Vita domain
        policy + execution hints, then multi-step tool loop against this agent's environment tools.

        ``prompt`` is typically a merged sub-task string (role + instruction + task input). Used by AFlow
        VitaBench operators and external callers that already constructed this agent with ``environment.get_tools()``.

        ``max_tool_rounds``: when set, caps LLM+tool iterations for this call only (defaults to agent limit).
        """
        combined_system = self.system_prompt + "\n\n" + _VITA_SKILL_MAS_SYSTEM_SUFFIX
        constrained_prompt = _VITA_SKILL_MAS_USER_HEADER + (prompt or "") + _VITA_SKILL_MAS_USER_FOOTER
        cap = self._bridge_max_tool_rounds()
        if max_tool_rounds is not None:
            cap = min(cap, max(1, int(max_tool_rounds)))
        out_text, _stage_tools, assistant, _stage_trace = self._run_tool_loop(
            system_prompt=combined_system,
            user_prompt=constrained_prompt,
            stage_role="vita_tool_bridge",
            max_rounds=cap,
        )
        usage = dict(assistant.usage or {})
        usage["estimated_cost_usd"] = float(assistant.cost or 0.0)
        return out_text or "", usage

    def _run_skill_mas_build_pipeline(self, task_text: str) -> tuple[AssistantMessage, dict[str, Any]]:
        from Skill_MAS.skill_mas.build import run_mas_pipeline_with_retries

        used_tools_all: list[str] = []
        used_tools_by_role: dict[str, list[str]] = {}
        internal_trace_all: list[dict[str, Any]] = []

        async def planner_call_fn(prompt: str) -> str:
            assistant = generate(
                model=self._skill_mas_planner_model(),
                tools=[],
                messages=[
                    SystemMessage(role="system", content="You are a MAS planner. Output JSON only."),
                    UserMessage(role="user", content=prompt),
                ],
                enable_think=self.enable_think,
                **self._skill_mas_planner_llm_args(),
            )
            if assistant is None:
                return ""
            return assistant.content or ""

        async def text_call_fn(prompt: str) -> tuple[str, dict[str, Any]]:
            # VitaBench-specific: keep reasoning phases pure text generation.
            # Remind planner that execution phase has connected Vita tools.
            planning_prompt = (
                _VITA_SKILL_MAS_USER_HEADER
                + (prompt or "")
                + "\n\n[Phase 1 planning only — intermediate sub-agent; execution phase WILL have Vita tools "
                "connected via the runner bridge. Do not conclude tools are unavailable. "
                "Do NOT output '###STOP###' — reserved for the outer final reply only.]\n"
            )
            assistant = generate(
                model=self.llm,
                tools=[],
                messages=[
                    SystemMessage(role="system", content=self.system_prompt + "\n\n" + _VITA_SKILL_MAS_SYSTEM_SUFFIX),
                    UserMessage(role="user", content=planning_prompt),
                ],
                enable_think=self.enable_think,
                **self.llm_args,
            )
            if assistant is None:
                return "", {"estimated_cost_usd": 0.0}
            usage = dict(assistant.usage or {})
            usage["estimated_cost_usd"] = float(assistant.cost or 0.0)
            return assistant.content or "", usage

        async def tool_call_fn(prompt_payload: Any) -> tuple[str, dict[str, Any]]:
            prompt, require_tool_call = _parse_vita_bridge_payload(prompt_payload)
            stage_role = "skill_mas_tool_bridge"
            if isinstance(prompt_payload, dict):
                role_name = str(prompt_payload.get("role_name") or "").strip()
                if role_name:
                    stage_role = role_name
            combined_system = self.system_prompt + "\n\n" + _VITA_SKILL_MAS_SYSTEM_SUFFIX
            constrained_prompt = _VITA_SKILL_MAS_USER_HEADER + prompt + _VITA_SKILL_MAS_USER_FOOTER
            out_text, stage_tools, assistant, stage_trace = self._run_tool_loop(
                system_prompt=combined_system,
                user_prompt=constrained_prompt,
                stage_role=stage_role,
                max_rounds=self._bridge_max_tool_rounds(),
                require_tool_call=require_tool_call,
            )
            used_tools_all.extend(stage_tools)
            used_tools_by_role.setdefault(stage_role, []).extend(stage_tools)
            internal_trace_all.extend(stage_trace)
            usage = dict(assistant.usage or {})
            usage["estimated_cost_usd"] = float(assistant.cost or 0.0)
            return out_text, usage

        init_skill = default_skill_mas_init_skill_path()
        class_name = f"VitaTask_{hashlib.sha1(task_text.encode('utf-8')).hexdigest()[:10]}_MASWorkflow"
        coro = run_mas_pipeline_with_retries(
            task_text=task_text,
            class_name=class_name,
            init_skill_path=init_skill,
            planner_call_fn=planner_call_fn,
            text_call_fn=text_call_fn,
            tool_call_fn=tool_call_fn,
            dataset_name="vita",
            max_generation_attempts=5,
            max_execution_attempts=3,
        )
        try:
            run_result = asyncio.run(coro)
        except RuntimeError as exc:
            if "asyncio.run() cannot be called" not in str(exc):
                raise
            loop = asyncio.new_event_loop()
            try:
                run_result = loop.run_until_complete(coro)
            finally:
                loop.close()

        final_text = str(getattr(run_result, "final_output", "") or "").strip()
        if not final_text:
            final_text = "I could not produce a final answer from Skill-MAS build pipeline."
        reply = AssistantMessage(role="assistant", content=final_text)
        agg_usage = _aggregate_usage_from_state(getattr(run_result, "state", {}) or {})
        reply.usage = {
            "prompt_tokens": int(agg_usage["prompt_tokens"]),
            "completion_tokens": int(agg_usage["completion_tokens"]),
            "total_tokens": int(agg_usage["total_tokens"]),
        }
        reply.cost = float(agg_usage["estimated_cost_usd"])
        artifacts = getattr(run_result, "artifacts", None)
        build_stage_traces: list[dict[str, Any]] = []
        normalized_sub_agents: list[dict[str, Any]] = []
        if artifacts is not None:
            build_stage_traces = [
                {
                    "stage": int(getattr(s, "stage", 0) or 0),
                    "stage_name": str(getattr(s, "stage_name", "")),
                    "elapsed_sec": float(getattr(s, "elapsed_sec", 0.0) or 0.0),
                    "prompt": str(getattr(s, "prompt", "")),
                    "raw_response": str(getattr(s, "raw_response", "")),
                    "parsed_json": _to_jsonable(getattr(s, "parsed_json", {})),
                }
                for s in (getattr(artifacts, "stage_traces", None) or [])
            ]
            normalized_sub_agents = _to_jsonable(getattr(artifacts, "normalized_sub_agents", []) or [])
        meta = {
            "schema_version": "skill_mas_message_meta/3",
            "routing_mode": "skill_mas_build_py",
            "executor_llm": self.llm,
            "skill_dir": self.skill_dir,
            "init_skill_path": str(init_skill.resolve()),
            "class_name": class_name,
            "success": bool(getattr(run_result, "success", False)),
            "failure_stage": getattr(run_result, "failure_stage", None),
            "failure_reason": getattr(run_result, "failure_reason", None),
            "generation_attempts_used": int(getattr(run_result, "generation_attempts_used", 0) or 0),
            "execution_attempts_used": int(getattr(run_result, "execution_attempts_used", 0) or 0),
            "used_tools": sorted(set(used_tools_all)),
            "used_tools_by_role": {k: sorted(set(v)) for k, v in used_tools_by_role.items()},
            "internal_tool_trace": internal_trace_all,
            "retry_events": getattr(run_result, "retry_events", []) or [],
            "workflow_state_keys": sorted(list((getattr(run_result, "state", {}) or {}).keys())),
            "mas_code": str(getattr(run_result, "mas_code", "") or ""),
            "build_stage_traces": build_stage_traces,
            "normalized_sub_agents": normalized_sub_agents,
            "workflow_state": _to_jsonable(getattr(run_result, "state", {}) or {}),
            "final_output": final_text,
        }
        if self.planner_llm and self.planner_llm != self.llm:
            meta["planner_llm"] = self.planner_llm
        return reply, meta

    def generate_next_message(self, message: ValidAgentInputMessage, state: SkillMASAgentState) -> tuple[AssistantMessage, SkillMASAgentState]:
        task_text: Optional[str] = None
        if isinstance(message, UserMessage) and getattr(message, "content", None):
            task_text = message.content
        if not task_text or not str(task_text).strip():
            messages = state.system_messages + state.messages + [message]
            assistant = generate(
                model=self.llm,
                tools=self.tools,
                messages=messages,
                enable_think=self.enable_think,
                **self.llm_args,
            )
            state.messages.append(message)
            state.messages.append(assistant)
            self._ensure_stop_token(assistant)
            return assistant, state

        reply, meta = self._run_skill_mas_build_pipeline(str(task_text).strip())
        self._ensure_stop_token(reply)
        state.messages.append(message)
        state.messages.append(reply)
        raw = reply.raw_data or {}
        raw["skill_mas"] = meta
        reply.raw_data = raw
        return reply, state

