"""Prompt templates for BrowseComp-Plus single-agent evaluation only."""


from __future__ import annotations


def build_single_agent_user_prompt(query: str, contexts: list[str]) -> str:
    joined = "\n\n".join(contexts) if contexts else "(no retrieved context)"
    return (
        "Answer the question using only the provided context snippets.\n"
        "Output only the final answer text (no chain-of-thought, no bullet list).\n\n"
        f"Question:\n{query.strip()}\n\n"
        f"Retrieved context:\n{joined}\n"
    )
