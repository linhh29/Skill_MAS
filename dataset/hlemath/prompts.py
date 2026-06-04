"""Prompt templates for HLEMATH-specific generation."""

from __future__ import annotations


HLEMATH_SYSTEM_PROMPT = (
    "You are an IMO-level math problem solver. "
    "Focus on correctness, concise derivation, and strict final-answer formatting."
)


def build_single_agent_user_prompt(question: str) -> str:
    return (
        "Solve the following math problem.\n"
        "The last line must be exactly one final answer in the form \\\\boxed{...}.\n\n"
        f"Problem:\n{question.strip()}\n"
    )
