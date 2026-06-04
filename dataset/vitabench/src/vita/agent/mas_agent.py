"""
MASAgent: BaseAgent that delegates to DeepResearchAgent MAS.

Interface matches VitaBench BaseAgent (get_init_state, generate_next_message).
Each user message is treated as a full task; MAS is run once and the final
answer is returned as a single AssistantMessage (no per-turn tool use).
"""
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from vita.agent.base import ValidAgentInputMessage, is_valid_agent_history_message
from vita.agent.llm_agent import LLMAgent, LLMAgentState
from vita.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from vita.environment.tool import Tool


def _run_mas_sync(query: str, config_path: Optional[str] = None, **kwargs) -> str:
    """Call DeepResearchAgent MAS; add Ant root to path if needed."""
    try:
        from mas_agent_bridge import run_mas_sync as _run
        return _run(query, config_path=config_path, **kwargs)
    except ImportError:
        pass
    ant_root = Path(__file__).resolve()
    for _ in range(5):
        ant_root = ant_root.parent
        if ant_root.name == "Ant" or (ant_root / "mas_agent_bridge").exists():
            break
    if str(ant_root) not in sys.path:
        sys.path.insert(0, str(ant_root))
    from mas_agent_bridge import run_mas_sync as _run
    return _run(query, config_path=config_path, **kwargs)


class MASAgent(LLMAgent):
    """
    Agent that uses DeepResearchAgent MAS internally.
    Same constructor as LLMAgent so VitaBench run.py can instantiate it;
    generate_next_message runs MAS on the user task and returns the final answer.
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
        mas_config_path: Optional[str] = None,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm or "placeholder",
            llm_args=llm_args or {},
            time=time,
            enable_think=enable_think,
            language=language,
        )
        self._mas_config_path = mas_config_path

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content="MAS agent.")],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        task_text: Optional[str] = None
        if isinstance(message, UserMessage) and getattr(message, "content", None):
            task_text = message.content
        if hasattr(message, "content") and getattr(message, "content", None):
            task_text = getattr(message, "content", None) or task_text
        if not task_text or not str(task_text).strip():
            return super().generate_next_message(message, state)
        task_text = str(task_text).strip()
        trace_path = None
        trace_dir = os.environ.get("TRACE_DIR")
        if trace_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_path = os.path.join(trace_dir, f"run_{ts}_{uuid4().hex[:8]}.json")
        try:
            result = _run_mas_sync(
                task_text,
                config_path=self._mas_config_path,
                model_id=self.llm,
                trace_path=trace_path,
                vitabench_tools=self.tools,
            )
        except Exception as e:
            result = f"[MAS error] {e!r}"
        if result is None:
            result = ""
        reply = AssistantMessage(role="assistant", content=(result if isinstance(result, str) else str(result)))
        state.messages.append(message)
        state.messages.append(reply)
        return reply, state
