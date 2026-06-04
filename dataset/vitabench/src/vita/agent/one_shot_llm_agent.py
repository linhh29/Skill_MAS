from copy import deepcopy
from typing import List, Optional

from loguru import logger

from vita.agent.llm_agent import LLMAgent, LLMAgentState
from vita.agent.base import ValidAgentInputMessage
from vita.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
)
from vita.environment.tool import Tool
from vita.utils.llm_utils import generate


class OneShotLLMAgent(LLMAgent):
    """
    单轮 LLMAgent：
    - 整个任务中只进行一次 LLM 调用；
    - 不调用任何工具（tools 列表会被忽略）；
    - 在第一次回复中自动追加 STOP_TOKEN，保证 orchestrator 立刻结束对话。

    这样配合 `StaticInputUser`：
    - 用户：一次性把任务 instructions 全部说完；
    - Agent：一次性给出完整解决方案，不再多轮澄清或 follow-up。
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
        # 复用父类初始化逻辑
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            time=time,
            enable_think=enable_think,
            language=language,
        )
        # 在整个 Simulation 中，只允许一次真正的 LLM 回复
        self._has_replied = False

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        # 直接复用父类的初始化逻辑
        return super().get_init_state(message_history=message_history)

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        - 第一次被调用时：读取完整历史 + 当前用户输入，一次性调用 LLM，生成最终回复；
        - 后续如果仍被调用：直接返回只包含 STOP_TOKEN 的空回复，避免再多轮交互。
        """
        # 先把最新一条消息写入 history
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.messages

        if not self._has_replied:
            if self.llm is None:
                raise ValueError("LLM is not set")

            # 只调用一次 LLM，且不使用任何工具
            assistant_message = generate(
                model=self.llm,
                tools=[],  # 禁用工具调用，确保一次性思考&回答
                messages=messages,
                enable_think=self.enable_think,
                **self.llm_args,
            )

            # 强制在内容中追加 STOP_TOKEN，保证 orchestrator 立刻结束
            content = assistant_message.content or ""
            stop_token = self.STOP_TOKEN
            if stop_token not in content:
                content = (content.rstrip() + "\n\n" + stop_token).lstrip()
                assistant_message.content = content

            logger.debug("[OneShotLLMAgent] First and only LLM reply generated.")
            self._has_replied = True
            state.messages.append(assistant_message)
            return assistant_message, state

        # 如果（按理说不该发生）还有后续调用，直接返回一个只含 STOP_TOKEN 的消息
        logger.warning(
            "[OneShotLLMAgent] generate_next_message called after first reply; "
            "returning STOP_TOKEN-only message."
        )
        assistant_message = AssistantMessage(
            role="assistant",
            content=self.STOP_TOKEN,
        )
        state.messages.append(assistant_message)
        return assistant_message, state

