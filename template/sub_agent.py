"""Unified sub-agent executor for all dataset runners."""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass, replace
from typing import Any, Callable


_BM25_QUERY_GEN_INSTRUCTION = (
    "You are a search-query generator for a keyword-based retrieval system (BM25). "
    "Your ONLY job is to produce a fluent, natural-language search query sentence — you must NOT "
    "attempt to answer the question or provide analysis.\n\n"
    "Guidelines:\n"
    "- Write a complete, grammatical sentence that reads like a search request, e.g. "
    "\"Find the Spanish TV series whose creator was born in 1973\" or "
    "\"Identify the actress who debuted on television in 1995\".\n"
    "- Embed the most discriminative concrete clues (names, places, dates, titles) from "
    "the information below directly into the sentence, not as a keyword list.\n"
    "- Keep the sentence under ~50 words. Do not add explanations, commentary, or formatting.\n"
    "- Output ONLY the search query sentence, nothing else."
)


def build_workflow_subagent_prompt(
    role_instruction: str,
    base_user_prompt: str,
    task_info: str,
    upstream_outputs: dict[str, str] | None = None,
) -> str:
    """
    Canonical prompt layout for generated MAS workflows (``prompt_override`` passed to
    ``SubAgentRequest.execute_async``). Centralizes section headings so templates only wire topology.
    """
    prompt = f"--- ROLE INSTRUCTION ---\n{role_instruction.strip()}\n\n"
    prompt += f"--- GLOBAL TASK CONTEXT ---\n{task_info.strip()}\n\n"
    if upstream_outputs:
        prompt += "--- UPSTREAM DEPENDENCIES DATA ---\n"
        for key, value in upstream_outputs.items():
            prompt += f"[{key.upper()}]:\n{value}\n\n"
    prompt += f"--- YOUR CURRENT TASK ---\n{base_user_prompt.strip()}"
    return prompt


def _extract_section(prompt_text: str, section_name: str) -> str:
    """Extract the content between --- SECTION_NAME --- and the next --- or end of text."""
    pattern = (
        r'---\s*' + re.escape(section_name) + r'\s*---\s*\n'
        + r'(.*?)'
        + r'(?:\n---\s*[A-Z][A-Z\s]*---|$)'
    )
    match = re.search(pattern, prompt_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _merge_usage_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Sum overlapping numeric usage fields (tokens, estimated cost); keep other keys from both."""
    out: dict[str, Any] = dict(a or {})
    bb = dict(b or {})
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "estimated_cost_usd",
    ):
        if key not in bb:
            continue
        av = out.get(key, 0) or 0
        bv = bb[key] = bb.get(key, 0) or 0
        try:
            out[key] = float(av) + float(bv) if key == "estimated_cost_usd" else int(av) + int(bv)
        except (TypeError, ValueError):
            out[key] = bb[key]
    for k, v in bb.items():
        if k not in out:
            out[k] = v
    return out


@dataclass(frozen=True)
class SubAgentRequest:
    role_name: str
    role_instruction: str
    user_prompt: str
    tool_context: dict[str, Any] | None = None

    async def execute_async(
        self,
        *,
        text_call_fn: Callable[[str], tuple[str, dict[str, Any]]] | Callable[[str], Any],
        tool_call_fn: Callable[["SubAgentRequest"], tuple[str, dict[str, Any]]] | Callable[["SubAgentRequest"], Any] | None = None,
        prompt_override: str | None = None,
    ) -> "SubAgentResult":
        """
        Async variant of ``execute``.

        Accepts both sync and async ``text_call_fn``/``tool_call_fn``.
        """
        active_user_prompt = self.user_prompt if prompt_override is None else prompt_override
        request_payload = {
            "role_name": self.role_name,
            "role_instruction": self.role_instruction,
            "user_prompt": active_user_prompt,
            "tool_context": self.tool_context or {},
        }

        # ``tool_call_fn`` is dataset-supplied, e.g. BrowseComp BM25 returns raw evidence text.
        # Then ``text_call_fn`` synthesizes the sub-agent deliverable (answer / title / JSON) from that evidence.
        tc = self.tool_context or {}
        if str(tc.get("execution_mode", "")).strip().lower() == "multi_turn_search" and tool_call_fn is not None:
            # ── Phase 1: LLM-guided query generation ──
            # Collect all available context: original task, role instruction, upstream data,
            # and the current sub-task. Feed these into text_call_fn with a specialised system
            # instruction that tells the LLM to ONLY produce BM25 search keywords — not answer
            # the question. This replaces the old approach of mechanically appending upstream
            # data to the original task; the LLM can now distil the most discriminative
            # keywords for each retrieval node.
            original_task = _extract_section(active_user_prompt, "GLOBAL TASK CONTEXT")
            upstream_data = _extract_section(active_user_prompt, "UPSTREAM DEPENDENCIES DATA")
            current_task = _extract_section(active_user_prompt, "YOUR CURRENT TASK")

            query_gen_prompt = _BM25_QUERY_GEN_INSTRUCTION + "\n\n---\n\n"
            if original_task:
                query_gen_prompt += f"[Original question]\n{original_task}\n\n"
            if self.role_instruction:
                query_gen_prompt += f"[Role / sub-agent purpose]\n{self.role_instruction}\n\n"
            if upstream_data:
                query_gen_prompt += f"[Upstream agent outputs]\n{upstream_data}\n\n"
            if current_task:
                query_gen_prompt += f"[Current sub-task]\n{current_task}\n\n"
            # Fallback: if no structured sections were found, include the full prompt
            if not (original_task or upstream_data or current_task):
                query_gen_prompt += f"[Full task input]\n{active_user_prompt}\n\n"
            query_gen_prompt += "Generate the BM25 search query now."

            query_ret = text_call_fn(query_gen_prompt)
            if inspect.isawaitable(query_ret):
                generated_query, query_usage = await query_ret
            else:
                generated_query, query_usage = await asyncio.to_thread(text_call_fn, query_gen_prompt)

            generated_query = str(generated_query or "").strip()
            # If LLM failed to produce a useful query, fall back to original task
            if not generated_query or len(generated_query) < 3:
                generated_query = (original_task or active_user_prompt)
            query_usage = query_usage or {}

            print(f"  [BM25 query for '{self.role_name}'] {generated_query}")

            # ── Phase 2: BM25 retrieval with LLM-generated seed ──
            enriched_tc = dict(tc)
            enriched_tc["initial_bm25_query"] = generated_query
            req_for_tool = replace(self, user_prompt=active_user_prompt, tool_context=enriched_tc)
            tool_ret = tool_call_fn(req_for_tool)
            if inspect.isawaitable(tool_ret):
                evidence_text, tool_usage = await tool_ret
            else:
                evidence_text, tool_usage = await asyncio.to_thread(tool_call_fn, req_for_tool)
            tool_usage = tool_usage or {}

            # merged = _merge_usage_dicts(query_usage, tool_usage)
            # print(f"SubAgentResult: {evidence_text}")
            # return SubAgentResult(
            #     text=str(evidence_text or "").strip(),
            #     usage=merged,
            #     request_payload=request_payload,
            # )

            # ── Phase 3: Answer synthesis ──
            synthesis_prompt = (
                f"[Sub-agent role]\n{self.role_name}\n\n"
                f"[Sub-task instruction]\n{self.role_instruction}\n\n"
                f"[Retrieved evidence]\n{evidence_text}\n\n"
                f"[Task input]\n{active_user_prompt}\n\n"
                "Using the retrieved evidence as grounding, answer the initial question."
            )
            text_ret = text_call_fn(synthesis_prompt)
            if inspect.isawaitable(text_ret):
                text, llm_usage = await text_ret
            else:
                text, llm_usage = await asyncio.to_thread(text_call_fn, synthesis_prompt)

            merged = _merge_usage_dicts(
                _merge_usage_dicts(query_usage, tool_usage),
                llm_usage or {},
            )
            print(f"SubAgentResult: {text}")
            return SubAgentResult(
                text=str(text or "").strip(),
                usage=merged,
                request_payload=request_payload,
            )
        elif str(tc.get("execution_mode", "")).strip().lower() == "vita_tool" and tool_call_fn is not None:
            # ── VitaBench two-phase execution ──
            # Phase 1 (Reasoning): Use text_call_fn (no tools) to first understand the sub-task,
            # extract key constraints, and formulate a structured action plan. This prevents the
            # tool-loop LLM from being overwhelmed by the full context and helps it start with a
            # clear understanding of what this node needs to accomplish.
            # Phase 2 (Execution): Pass the original prompt augmented with the reasoning plan
            # into tool_call_fn, which runs the multi-step LLM+tool interaction loop.

            # ── Phase 1: Reasoning / task comprehension ──
            reasoning_instruction = (
                "You are a task comprehension assistant. Your job is to carefully read the "
                "sub-agent role, instruction, task context, and upstream data below, then "
                "produce a **structured action plan** for this specific sub-task node.\n\n"
                "Your plan MUST include:\n"
                "1. **Task understanding**: What is this sub-task node responsible for? What is "
                "its scope (what it should and should NOT do)?\n"
                "2. **Key constraints**: List every concrete constraint explicitly stated or "
                "strongly implied by the task context and upstream data (dates, quantities, "
                "addresses, preferences, budget limits, etc.).\n"
                "3. **Required actions**: What tool actions must be completed to fulfill this "
                "sub-task? List them in the order they should be executed.\n"
                "4. **Critical parameters**: Extract concrete values needed for tool calls "
                "from the upstream data and task context.\n"
                "5. **Potential pitfalls**: Identify any assumptions or implicit requirements "
                "in the task that could lead to mistakes if overlooked.\n\n"
                "Output a concise, structured plan. Do NOT attempt to execute any actions — "
                "just analyze and plan. This plan will be provided to the execution agent to "
                "help it understand the task before it starts using tools."
            )
            reasoning_prompt = (
                f"{reasoning_instruction}\n\n"
                f"---\n\n"
                f"[Sub-agent role]\n{self.role_name}\n\n"
                f"[Sub-task instruction]\n{self.role_instruction}\n\n"
                f"[Task input]\n{active_user_prompt}\n\n"
                "Produce your structured action plan now."
            )

            reasoning_ret = text_call_fn(reasoning_prompt)
            if inspect.isawaitable(reasoning_ret):
                reasoning_text, reasoning_usage = await reasoning_ret
            else:
                reasoning_text, reasoning_usage = await asyncio.to_thread(text_call_fn, reasoning_prompt)
            reasoning_text = str(reasoning_text or "").strip()
            reasoning_usage = reasoning_usage or {}

            print(f"  [Reasoning plan for '{self.role_name}'] {reasoning_text[:200]}...")

            # ── Phase 2: Execution with augmented context ──
            # Prepend the reasoning plan as an "ACTION PLAN" section so the tool-loop LLM
            # starts with a clear understanding of what to do, rather than having to figure
            # it out from scratch amidst the full context.
            merged_prompt = (
                f"[Sub-agent role]\n{self.role_name}\n\n"
                f"[Sub-task instruction]\n{self.role_instruction}\n\n"
                f"[Pre-computed action plan — READ THIS FIRST before using tools]\n{reasoning_text}\n\n"
                f"[Task input]\n{active_user_prompt}"
            )
            # Prefer a structured bridge payload when downstream adapters support it;
            # fall back to legacy plain-string prompt for compatibility.
            bridge_payload = {
                "role_name": self.role_name,
                "role_instruction": self.role_instruction,
                "action_plan": reasoning_text,
                "task_input": active_user_prompt,
                "prompt": merged_prompt,
            }
            payload_for_sync_call: Any = bridge_payload
            try:
                text_ret = tool_call_fn(bridge_payload)
            except TypeError:
                payload_for_sync_call = merged_prompt
                text_ret = tool_call_fn(merged_prompt)
            if inspect.isawaitable(text_ret):
                text, tool_usage = await text_ret
            else:
                text, tool_usage = await asyncio.to_thread(tool_call_fn, payload_for_sync_call)
            tool_usage = tool_usage or {}

            merged = _merge_usage_dicts(reasoning_usage, tool_usage)
            print(f"SubAgentResult: {text}")
            return SubAgentResult(
                text=text,
                usage=merged,
                request_payload=request_payload,
            )

        merged_prompt = (
            f"[Sub-agent role]\n{self.role_name}\n\n"
            f"[Sub-task instruction]\n{self.role_instruction}\n\n"
            f"[Task input]\n{active_user_prompt}"
        )
        text_ret = text_call_fn(merged_prompt)
        if inspect.isawaitable(text_ret):
            text, usage = await text_ret
        else:
            text, usage = await asyncio.to_thread(text_call_fn, merged_prompt)
        print(f"SubAgentResult: {text}")
        return SubAgentResult(
            text=text,
            usage=usage or {},
            request_payload=request_payload,
        )


@dataclass(frozen=True)
class SubAgentResult:
    text: str
    usage: dict[str, Any]
    request_payload: dict[str, Any]