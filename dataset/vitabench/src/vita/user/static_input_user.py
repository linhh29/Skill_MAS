from typing import Optional, Tuple

from loguru import logger

from vita.data_model.message import (
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from vita.environment.tool import Tool
from vita.user.base import (
    OUT_OF_SCOPE,
    STOP,
    TRANSFER,
    BaseUser,
    UserState,
    ValidUserInputMessage,
    is_valid_user_history_message,
)


class StaticInputUser(BaseUser):
    """
    A simple user implementation that does NOT call any LLM.

    Behavior:
    - On its first turn, it sends the full task instructions as a single user message.
    - On subsequent turns, it sends a STOP message so the orchestrator can terminate.

    This avoids multi-turn LLM-based user simulation and instead injects the
    benchmark's user instructions directly to the agent in one shot.
    """

    def __init__(
        self,
        tools: Optional[list[Tool]] = None,
        instructions: Optional[str] = None,
        persona: Optional[str] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        language: str = None,
    ):
        # We intentionally ignore llm / llm_args – no LLM calls are made here.
        super().__init__(instructions=instructions, llm=None, llm_args=None)
        self.tools = tools
        self.persona = persona
        self.language = language

    @property
    def system_prompt(self) -> str:
        """
        A minimal system prompt, mainly for compatibility with UserState.
        The real behavior is hard-coded in generate_next_message.
        """
        persona = self.persona or ""
        instructions = self.instructions or ""
        return (
            "You are a static benchmark user. "
            "You do NOT ask follow-up questions. "
            "You only provide the full goal description once, then stop.\n\n"
            f"Persona: {persona}\n"
            f"User goal / instructions: {instructions}\n"
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        """
        Initialize the user state with a single system message.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_user_history_message(m) for m in message_history), (
            "Invalid user message history. User messages must be of type "
            "UserMessage, AssistantMessage, or ToolMessage to User."
        )

        user_state = UserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        return user_state

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """
        Reuse the same STOP/TRANSFER/OUT_OF_SCOPE tokens as other users.
        """
        if message.is_tool_call():
            return False
        assert message.content is not None
        return (
            STOP in message.content
            or TRANSFER in message.content
            or OUT_OF_SCOPE in message.content
        )

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        return self._generate_next_message(message, state)

    def _generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """
        First time the agent talks to the user:
        - Return the full task instructions as a single user message.

        On later turns:
        - Immediately return a STOP message so the conversation ends.
        """
        # Record incoming assistant/tool message into the history
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Check whether we've already produced a UserMessage before
        has_user_message = any(isinstance(m, UserMessage) for m in state.messages)

        if not has_user_message:
            # First user turn: send the full instructions once
            content = self.instructions or ""
            logger.debug(f"[StaticInputUser] First turn, sending instructions: {content!r}")
        else:
            # Subsequent turns: immediately send STOP to end the dialogue
            content = STOP
            logger.debug("[StaticInputUser] Subsequent turn, sending STOP.")

        user_message = UserMessage(
            role="user",
            content=content,
        )

        state.messages.append(user_message)
        return user_message, state

    def set_seed(self, seed: int):
        """
        Override BaseUser.set_seed to be a no-op.

        StaticInputUser does not call any LLM, so seeding is irrelevant.

        This avoids raising `LLM is not set` when the orchestrator tries to
        propagate the global seed to both agent and user.
        """
        logger.debug(f"[StaticInputUser] Ignoring set_seed({seed}) since no LLM is used.")

