import json
from typing import Optional, List

from loguru import logger  # type: ignore[import-untyped]
from pydantic import BaseModel

from vita.agent.llm_agent import LLMAgent
from vita.agent.base import is_valid_agent_history_message
from vita.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
)
from vita.environment.tool import Tool
from vita.utils.llm_utils import generate


class PlannerDecision(BaseModel):
    """
    由 Planner 输出的结构化决策结果。
    """

    todo_summary: str
    selected_subtask: str
    next_role: str  # "executor"、"fixer" 或 "planner"
    executor_instruction: Optional[str] = None
    fixer_instruction: Optional[str] = None
    planner_message: Optional[str] = None  # 当无需 Executor/Fixer 时，直接用于回复用户的内容
    should_stop: bool = False  # 当所有子任务完成、无需继续对话时设为 True


class PlannerExecutorFixerAgentState(BaseModel):
    """
    Multi-agent 的整体状态：
    - messages: 对 orchestrator 可见的对话历史（User/Agent/Tool）
    - todo_summary: 当前 Planner 维护的 TODO 列表摘要（文本形式）
    """

    system_messages: List[SystemMessage]
    messages: List[APICompatibleMessage]
    todo_summary: str = ""


class PlannerExecutorFixerAgent(LLMAgent):
    """
    一个 MAS 壳 agent：对 VitaBench 来说是单个 Agent，
    但在 `generate_next_message` 内部显式调用 Planner / Executor / Fixer 三套逻辑。

    - 外部接口：保持与 `LLMAgent` 完全一致（构造函数签名、get_init_state、generate_next_message）。
    - 内部实现：每一轮至少两次 LLM 调用：
        1）Planner：读取当前对话和 TODO，生成新的规划与下一步决策；
        2）Executor 或 Fixer：根据 Planner 决策实际调用工具或回复用户。
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        time=None,
        enable_think: bool = False,
        language: str = None,
    ):
        # 调用父类初始化，保证与 run.py 的构造逻辑兼容
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            time=time,
            enable_think=enable_think,
            language=language,
        )

    # ------- Planner / Executor / Fixer 专用 system prompt --------

    def _planner_system_message(self, todo_summary: str) -> SystemMessage:
        content = (
            "你是 Planner Agent，负责理解用户需求、拆解任务并维护 TODO 列表。\n"
            "【重要限制】你主要是一个「任务规划器」，默认情况下**不要直接回答用户的问题，也不要直接与用户对话**，\n"
            "而是把你的决策通过 JSON 告诉 Executor/Fixer，由它们来与用户交互和调用工具。\n"
            "- 绝大多数情况下，你的输出都是给内部 Executor/Fixer 看的，而不是直接给用户看的；\n"
            "- 不要在 JSON 之外追加安慰用户/道歉/解释等自然语言文本；\n"
            "- 只有在你确认【当前轮次不需要调用任何工具，也不需要 Fixer 介入，只需要给用户一个简单的总结性回复】时，\n"
            "  才可以在 planner_message 字段中填入对用户的最终回复内容，并将 next_role 设为 \"planner\"，此时 Executor/Fixer 不会再单独生成回复。\n"
            "- 默认情况下，你和整个多智能体系统的目标是：**从理解需求 → 方案规划 → 细节确认 → 实际完成（包括帮用户完成下单/预订等具体操作）**，而不是只给建议。\n"
            "  当用户的 instructions 中包含下单、预订、购买等意图时，你需要在 TODO 中安排完整链路：\n"
            "  1）先通过对话向用户确认商品/门店/时间/人数等关键信息；\n"
            "  2）在用户确认「信息无误，可以下单」之后，将 next_role 设为 \"executor\"，并在 executor_instruction 中明确写出「现在可以调用对应下单/预订类工具，按上述已确认信息代为完成下单」；\n"
            "  3）确保 Executor 实际调用对应 tool 完成下单，而不是只停留在推荐层面。\n"
            "注意：你【只负责规划】，不允许直接决定具体的工具调用 JSON，也不允许输出如下结构："
            "{\"name\": \"tool_xxx\", \"arguments\": {...}}、{\"tool_id\": ...} 等。\n"
            "你应该用自然语言描述应该调用什么类型的工具、需要哪些关键信息，由 Executor 根据你的说明去真正调用工具。\n"
            "请严格输出一个 JSON，对应模型 PlannerDecision：\n"
            "{\n"
            '  \"todo_summary\": \"当前 TODO 列表的自然语言摘要，包含每个子任务的状态\",\n'
            '  \"selected_subtask\": \"本轮需要重点推进的子任务（简明扼要）\",\n'
            '  \"next_role\": \"executor、fixer 或 planner 三者之一\",\n'
            '  \"executor_instruction\": \"如果 next_role=executor，则给 Executor 的详细指令；否则为 null\",\n'
            '  \"fixer_instruction\": \"如果 next_role=fixer，则给 Fixer 的详细指令；否则为 null\",\n'
            '  \"planner_message\": \"只有当 next_role=planner 且不需要调用任何工具/修复流程时，才在这里填入准备直接给用户的最终自然语言回复；其它情况下必须为 null 或空字符串\",\n'
            '  \"should_stop\": \"布尔值；只有在【已经向用户明确确认过是否还有其他需求，且用户清楚回复没有更多需求、可以结束】的前提下，才可以设为 true；'
            '否则必须为 false（包括你只是主观觉得任务完成、但还没和用户做结束确认的情况）\"\n'
            "}\n"
            "注意：\n"
            "- **无论任何情况，只要你有输出，就必须严格是一个合法的 JSON 对象（上面定义的 PlannerDecision），绝不能输出纯文本、数字、数组或其它任意形式**；\n"
            "- 只能输出一个合法的 JSON 对象，不能包含多余文本或注释；\n"
            "- todo_summary 可以简短，但要覆盖所有关键子任务和状态；\n"
            "- 严禁在 JSON 中出现字段名 name / arguments / tool_id / function 等与工具 JSON 结构相关的字段；"
            "如果需要调用工具，请在 executor_instruction 中用自然语言说明应该调用哪一类工具、用哪些关键参数即可。\n"
            "- 当你发现之前的执行路径存在问题、用户不满意、或环境结果异常时，应该把 next_role 设为 \"fixer\"。\n"
            "- 当你认为主要任务已经完成，但【还没有和用户做结束确认】时，必须让 next_role=executor，并在 executor_instruction 中明确要求 Executor 向用户发起一次确认，"
            "例如：解释已经完成了哪些内容，然后问「请确认是否还有其他需求，如果没有，我将结束本次服务」。此时 should_stop 必须为 false。\n"
            f"当前已有 TODO 摘要（可以在此基础上更新）：{todo_summary or '（暂无，需从头规划）'}\n"
        )
        return SystemMessage(role="system", content=content)

    def _executor_system_message(
        self, decision: PlannerDecision
    ) -> SystemMessage:
        content = (
            "你是 Executor Agent，负责根据 Planner 的规划执行当前子任务。\n"
            "请遵循以下要求：\n"
            f"- 当前子任务: {decision.selected_subtask}\n"
            f"- Planner 给你的执行指令: {decision.executor_instruction or ''}\n"
            "- 你可以调用提供的工具（tools），也可以在信息足够时直接给出自然语言回复；\n"
            "- 工具调用要尽量准确、避免无意义和重复调用；\n"
            "- 对用户的回复要清晰，解释你做了什么以及下一步计划。\n"
        )
        return SystemMessage(role="system", content=content)

    def _fixer_system_message(self, decision: PlannerDecision) -> SystemMessage:
        content = (
            "你是 Fixer Agent，负责在当前方案出现问题时进行诊断和修复。\n"
            "请遵循以下要求：\n"
            f"- Planner 判断当前需要修复，给你的信息: {decision.fixer_instruction or ''}\n"
            "- 阅读最近几轮对话和工具调用结果，找出问题原因（工具选择错误/参数错误/计划不合理等）；\n"
            "- 可以在必要时调用新的工具，或提出新的子任务/步骤建议；\n"
            "- 对用户的回复要说明你发现的问题、如何修复，以及后续会怎么做。\n"
        )
        return SystemMessage(role="system", content=content)

    # ------- State 接口，与 LLMAgent 保持兼容 --------

    def get_init_state(
        self, message_history: Optional[List[Message]] = None
    ) -> PlannerExecutorFixerAgentState:
        if message_history is None:
            message_history = []
        assert all(
            is_valid_agent_history_message(m) for m in message_history
        ), "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."

        base_system = SystemMessage(role="system", content=self.system_prompt)
        return PlannerExecutorFixerAgentState(
            system_messages=[base_system],
            messages=message_history,
            todo_summary="",
        )

    # ------- 核心：一轮内部 MAS 协作流程 --------

    def _run_planner(
        self,
        state: PlannerExecutorFixerAgentState,
        last_input: Message,
    ) -> tuple[PlannerDecision, AssistantMessage, SystemMessage]:
        """
        运行 Planner 子 agent，返回：
        - decision: 结构化 PlannerDecision
        - planner_reply: 原始 LLM 回复（会写进 raw_data）
        - planner_system: 当轮使用的 system prompt

        为了保证 JSON 可靠性，这里最多重试若干次（默认 5 次）；
        只有在始终无法解析为合法 PlannerDecision 时，才退回到默认 executor 决策。
        """
        planner_system = self._planner_system_message(state.todo_summary)
        messages: List[APICompatibleMessage] = [planner_system] + state.messages

        max_attempts = 5
        last_reply: Optional[AssistantMessage] = None
        last_raw: str = ""

        # Qwen json_schema 格式，严格约束输出结构
        planner_json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "planner_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "todo_summary": {
                            "type": "string",
                            "description": "当前 TODO 列表的自然语言摘要，包含每个子任务的状态",
                        },
                        "selected_subtask": {
                            "type": "string",
                            "description": "本轮需要重点推进的子任务（简明扼要）",
                        },
                        "next_role": {
                            "type": "string",
                            "description": "\"executor\"、\"fixer\" 或 \"planner\" 三者之一",
                        },
                        "executor_instruction": {
                            "type": "string",
                            "description": "当 next_role=executor 时给 Executor 的详细指令；否则可以是空字符串",
                        },
                        "fixer_instruction": {
                            "type": "string",
                            "description": "当 next_role=fixer 时给 Fixer 的详细指令；否则可以是空字符串",
                        },
                        "planner_message": {
                            "type": "string",
                            "description": "当 next_role=planner 且无需工具/修复时，用于直接回复用户的自然语言内容；其它情况下应为空字符串",
                        },
                        "should_stop": {
                            "type": "boolean",
                            "description": "只有在已与用户确认无任何剩余需求且可以结束时为 true，否则为 false",
                        },
                    },
                    "required": [
                        "todo_summary",
                        "selected_subtask",
                        "next_role",
                        "executor_instruction",
                        "fixer_instruction",
                        "planner_message",
                        "should_stop",
                    ],
                    "additionalProperties": False,
                },
            },
        }

        for attempt in range(1, max_attempts + 1):
            planner_reply: AssistantMessage = generate(
                model=self.llm,
                tools=[],  # Planner 不直接调用工具，只做规划
                messages=messages,
                enable_think=self.enable_think,
                # 使用 Qwen json_schema 严格约束输出结构
                response_format=planner_json_schema,
                **self.llm_args,
            )
            last_reply = planner_reply
            raw = planner_reply.content or ""
            last_raw = raw
            logger.debug(
                f"[MAS][Planner] attempt={attempt}/{max_attempts} reply content: {raw!r}"
            )

            # 提前过滤明显不合法的情况（空串、只有数字/数组等）
            if not raw.strip():
                logger.warning(
                    f"[MAS][Planner] empty content on attempt {attempt}, retrying..."
                )
                continue

            json_str = raw
            if "{" in raw and "}" in raw:
                json_str = raw[raw.find("{") : raw.rfind("}") + 1]
            try:
                data = json.loads(json_str)
                # 要求是一个对象，而不是数组/数字
                if not isinstance(data, dict):
                    raise ValueError("PlannerDecision JSON must be an object")
                decision = PlannerDecision(**data)
                state.todo_summary = decision.todo_summary
                logger.debug(
                    f"[MAS][Planner] parsed decision successfully on attempt {attempt}: "
                    f"{decision.model_dump()!r}"
                )
                return decision, planner_reply, planner_system
            except Exception as e:
                logger.warning(
                    f"[MAS][Planner] failed to parse JSON decision on attempt "
                    f"{attempt}/{max_attempts}, error={e!r}, raw_reply={raw!r}"
                )
                # 尝试下一次
                continue

        # 多次尝试后仍然失败：退回到默认 executor 决策
        logger.warning(
            "[MAS][Planner] all attempts to get valid JSON decision failed, "
            "fallback to default executor decision."
        )
        fallback_raw = last_raw or ""
        user_instruction = ""
        # 如果有最近一条来自用户的自然语言输入，用它来指导 Executor
        try:
            if isinstance(last_input, Message) and getattr(last_input, "content", None):
                user_instruction = str(last_input.content)
        except Exception:
            user_instruction = ""
        decision = PlannerDecision(
            todo_summary=state.todo_summary or "根据对话历史为用户完成当前需求。",
            selected_subtask="根据当前用户消息完成需求。",
            next_role="executor",
            executor_instruction=(
                user_instruction
                or fallback_raw
                or "根据当前对话为用户完成需求。"
            ),
            should_stop=False,
        )
        state.todo_summary = decision.todo_summary

        # 如果连一次有效回复都没有，就构造一个最简 AssistantMessage 以便记录 raw_data
        if last_reply is None:
            last_reply = AssistantMessage(role="assistant", content=fallback_raw or None)

        return decision, last_reply, planner_system

    def _run_executor(
        self,
        state: PlannerExecutorFixerAgentState,
        decision: PlannerDecision,
        last_input: Message,
    ) -> tuple[AssistantMessage, SystemMessage]:
        logger.debug(
            f"[MAS][Executor] executing subtask={decision.selected_subtask!r} "
            f"with instruction={decision.executor_instruction!r}"
        )
        executor_system = self._executor_system_message(decision)
        messages: List[APICompatibleMessage] = (
            [executor_system] + state.messages + [last_input]
        )
        reply: AssistantMessage = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            enable_think=self.enable_think,
            **self.llm_args,
        )
        logger.debug(f"[MAS][Executor] reply content: {reply.content!r}")
        return reply, executor_system

    def _run_fixer(
        self,
        state: PlannerExecutorFixerAgentState,
        decision: PlannerDecision,
        last_input: Message,
    ) -> tuple[AssistantMessage, SystemMessage]:
        logger.debug(
            f"[MAS][Fixer] fixing with instruction={decision.fixer_instruction!r}"
        )
        fixer_system = self._fixer_system_message(decision)
        messages: List[APICompatibleMessage] = (
            [fixer_system] + state.messages + [last_input]
        )
        reply: AssistantMessage = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            enable_think=self.enable_think,
            **self.llm_args,
        )
        logger.debug(f"[MAS][Fixer] reply content: {reply.content!r}")
        return reply, fixer_system

    def generate_next_message(
        self,
        message: Message,
        state: PlannerExecutorFixerAgentState,
    ) -> tuple[AssistantMessage, PlannerExecutorFixerAgentState]:
        """
        一轮 MAS 交互：
        1. 把来自 User / Tool 的输入写入公共历史；
        2. 调用 Planner 生成结构化决策（含 TODO / 子任务 / 是否需要 Fixer）；
        3. 根据决策调用 Executor 或 Fixer 真正输出对外的 Assistant 消息。
        4. 将 Planner / Executor / Fixer 的内部过程写入 assistant.raw_data，便于后续分析。
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
            last_input: Message = message.tool_messages[-1]
        else:
            state.messages.append(message)
            last_input = message

        decision, planner_reply, planner_system = self._run_planner(
            state, last_input
        )

        next_role = (decision.next_role or "executor").lower()
        # 特殊分支：Planner 认为无需调用 Executor/Fixer，仅需给用户一个简单总结性回复
        if next_role == "planner" and (decision.planner_message or "").strip():
            assistant = AssistantMessage(
                role="assistant",
                content=decision.planner_message,
            )
            # 为了 raw_data 一致性，这里构造一个虚拟的 role_system（Planner 此轮自身的说明）
            role_system = SystemMessage(
                role="system",
                content="Planner decided no further tool calls or fixes are needed; "
                "directly replying to user with planner_message.",
            )
            mas_role = "planner"
        elif next_role == "fixer":
            assistant, role_system = self._run_fixer(state, decision, last_input)
            mas_role = "fixer"
        else:
            assistant, role_system = self._run_executor(state, decision, last_input)
            mas_role = "executor"

        # 如果 Planner 判断所有任务已经完成，则在最终回复中注入 STOP_TOKEN，
        # 让 orchestrator 以 TerminationReason.AGENT_STOP 正常结束对话。
        if decision.should_stop and not assistant.is_tool_call():
            stop_token = self.STOP_TOKEN
            content = assistant.content or ""
            if stop_token not in (content or ""):
                if content:
                    content = content.rstrip() + "\n\n" + stop_token
                else:
                    content = stop_token
                assistant.content = content

        # 将完整 MAS 内部过程记录到 raw_data 里，写入 simulation JSON
        raw = assistant.raw_data or {}
        raw.update(
            {
                "mas_role": mas_role,
                "planner_decision": decision.model_dump(),
                "planner_raw_reply": planner_reply.content,
                "planner_system_prompt": planner_system.content,
                "role_system_prompt": role_system.content,
            }
        )
        assistant.raw_data = raw

        state.messages.append(assistant)
        return assistant, state

