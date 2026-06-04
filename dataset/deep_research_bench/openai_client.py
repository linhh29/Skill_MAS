"""Re-export Skill_MAS ``AsyncOpenAIClient`` (async API only; no sync wrapper)."""

from __future__ import annotations

from Skill_MAS.skill_mas.openai_async_client import AsyncOpenAIClient  # type: ignore[import-not-found]

__all__ = ["AsyncOpenAIClient"]
