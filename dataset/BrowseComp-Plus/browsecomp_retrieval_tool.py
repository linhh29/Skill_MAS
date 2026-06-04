"""Bind BrowseComp BM25 multi-round retrieval to Skill_MAS ``tool_call_fn(SubAgentRequest)``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from retrieval import BM25Retriever
from retrieval_flow import multi_round_retrieve_contexts


def _format_evidence_text(
    contexts: list[str],
    *,
    trajectory: list[dict[str, Any]] | None = None,
    evidence_scope: str = "last_round",
) -> str:
    """Join snippets; optionally append a human-readable multi-round summary.

    Note: ``doc_max_tokens`` in ``multi_round_retrieve_contexts`` is enforced via
    ``_truncate_text_by_tokens``, which truncates by **whitespace-separated words**
    (historical naming), not neural tokenizer units.
    """
    if not contexts:
        base = "(No documents retrieved from the corpus.)"
    else:
        base = (
            "Retrieved evidence (ground answers in these snippets only):\n\n"
            + "\n\n---\n\n".join(contexts)
        )
    if not trajectory:
        return base
    lines = [
        "",
        "--- Multi-round retrieval trace (controller-guided BM25) ---",
    ]
    for step in trajectory:
        r = step.get("round", "?")
        q = str(step.get("query", "")).strip().replace("\n", " ")
        # if len(q) > 160:
        #     q = q[:157] + "..."
        ids = step.get("retrieved_docids") or []
        dec = step.get("decision") or {}
        act = str(dec.get("action", "")).strip()
        n_new = len(step.get("new_docids_first_seen_this_round") or [])
        lines.append(
            f"Round {r}: planner_action={act!r} | bm25_query={q!r} | "
            f"bm25_hits={len(ids)} | new_distinct_snippets={n_new}"
        )
    scope_label = "accumulated distinct snippets" if evidence_scope == "accumulated" else "last BM25 round only"
    lines.append(f"Evidence snippets shown above ({scope_label}): {len(contexts)}")
    return base + "\n".join(lines)


def make_browsecomp_retrieval_tool_fn(
    *,
    chat_client: Any,
    model: str,
    retriever: BM25Retriever,
    retrieval_topk: int,
    doc_max_tokens: int,
    max_rounds: int,
    reasoning_effort: str | None,
    full_task_question: str,
    pricing: Mapping[str, Any] | None = None,
    retrieval_sessions_out: list[dict[str, Any]] | None = None,
) -> Callable[[Any], Any]:
    """
    Build an async ``tool_call_fn`` for ``template.sub_agent.SubAgentRequest.execute_async``:
    ``async (req) -> (text, usage)``.

    ``multi_round_retrieve_contexts`` runs multiple planner-guided BM25 rounds. For Skill_MAS,
    we return a bounded accumulated evidence view to improve robustness when relevant snippets
    appear in earlier rounds.
    """
    from Skill_MAS.skill_mas.openai_async_client import enrich_usage_with_cost

    async def tool_call_fn(req: Any) -> tuple[str, dict[str, Any]]:
        tc = getattr(req, "tool_context", None) or {}
        if not isinstance(tc, dict):
            tc = {}
        mode = str(tc.get("execution_mode", "multi_turn_search")).strip().lower()
        if mode not in ("multi_turn_search",):
            raise RuntimeError(
                "BrowseComp retrieval tool expected tool_context.execution_mode="
                f"'multi_turn_search', got {tc.get('execution_mode')!r}"
            )

        user_prompt = str(getattr(req, "user_prompt", "") or "").strip()
        role_inst = str(getattr(req, "role_instruction", "") or "").strip()
        seed = str(tc.get("seed_question") or "").strip() or str(full_task_question or "").strip() or user_prompt

        dc = tc.get("delegation_context")
        if dc is None or (isinstance(dc, str) and not str(dc).strip()):
            parts = []
            if role_inst:
                parts.append(f"[Role constraints]\n{role_inst}")
            if user_prompt:
                parts.append(f"[Sub-task input]\n{user_prompt}")
            delegation_context = "\n\n".join(parts) if parts else None
        else:
            delegation_context = str(dc)

        iq = tc.get("initial_bm25_query")
        initial_bm25_query = str(iq).strip() if iq else None

        evidence_policy = "accumulated"
        max_final_snippets = max(int(retrieval_topk), min(3, int(max_rounds)) * int(retrieval_topk))
        contexts, trajectory, planner_usage = await multi_round_retrieve_contexts(
            client=chat_client,
            model=model,
            question=seed,
            retriever=retriever,
            retrieval_topk=int(retrieval_topk),
            doc_max_tokens=int(doc_max_tokens),
            max_rounds=int(max_rounds),
            reasoning_effort=reasoning_effort,
            delegation_context=delegation_context,
            initial_bm25_query=initial_bm25_query,
            evidence_policy=evidence_policy,
            max_final_snippets=max_final_snippets,
        )
        text = _format_evidence_text(contexts, trajectory=trajectory, evidence_scope=evidence_policy)
        prompt_t = int(planner_usage.get("prompt_tokens", 0) or 0)
        comp_t = int(planner_usage.get("completion_tokens", 0) or 0)
        usage: dict[str, Any] = {
            "input_tokens": prompt_t,
            "output_tokens": comp_t,
            "total_tokens": prompt_t + comp_t,
            "retrieval_rounds_done": len(trajectory),
        }
        usage = enrich_usage_with_cost(usage, model=model, table=pricing)

        if retrieval_sessions_out is not None:
            retrieval_sessions_out.append(
                {
                    "role_name": str(getattr(req, "role_name", "") or ""),
                    "seed_question": seed,
                    "initial_bm25_query": initial_bm25_query,
                    "delegation_context_preview": (delegation_context or ""),
                    "settings": {
                        "retrieval_topk": int(retrieval_topk),
                        "doc_max_tokens": int(doc_max_tokens),
                        "max_rounds": int(max_rounds),
                    },
                    "single_agent_alignment": (
                        "Skill_MAS mode: contexts are bounded accumulated snippets across rounds; "
                        "configured to preserve early-round evidence and reduce brittle last-round drift."
                    ),
                    "evidence_policy": evidence_policy,
                    "max_final_snippets": max_final_snippets,
                    "trajectory": trajectory,
                    "last_round_snippet_count": len((trajectory[-1].get("retrieved_docids") if trajectory else []) or []),
                    "evidence_snippet_count": len(contexts),
                    "planner_token_usage": planner_usage,
                }
            )

        return text, usage

    return tool_call_fn
