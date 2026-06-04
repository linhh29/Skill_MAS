"""Shared fixed-phase Skill-MAS pipeline builder/executor."""

from __future__ import annotations

import asyncio
import ast
import inspect
import json
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

_SKILL_MAS_PKG_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE_ROOT = _SKILL_MAS_PKG_ROOT.parent
for _p in (_PACKAGE_ROOT, _SKILL_MAS_PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from template.mas_code_template import prepend_fixed_imports, stage3_llm_reference_bundle

@dataclass(frozen=True)
class RouteDecision:
    selected_skill_name: str
    reasoning: str
    raw_response: str
    usage: dict[str, Any]


@dataclass(frozen=True)
class PhaseExecutionResult:
    text: str
    usage: dict[str, Any]
    meta: dict[str, Any]


@dataclass(frozen=True)
class PhasePipelineResult:
    prior_outputs: dict[str, str]
    per_phase: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    usage_totals: dict[str, Any]


@dataclass(frozen=True)
class SubTaskSpec:
    role_name: str
    role_instruction: str
    user_prompt: str = ""
    tool_context: dict[str, Any] | None = None


@dataclass(frozen=True)
class BuildStageTrace:
    stage: int
    stage_name: str
    prompt: str
    raw_response: str
    parsed_json: dict[str, Any]
    elapsed_sec: float


@dataclass(frozen=True)
class ThreeStageBuildArtifacts:
    mas_code: str
    stage_traces: list[BuildStageTrace]
    stage1: dict[str, Any]
    stage2: dict[str, Any]
    stage3: dict[str, Any]
    normalized_sub_agents: list[dict[str, Any]]


@dataclass(frozen=True)
class MASRunWithRetryResult:
    success: bool
    final_output: str
    state: dict[str, Any]
    mas_code: str
    artifacts: ThreeStageBuildArtifacts | None
    generation_attempts_used: int
    execution_attempts_used: int
    failure_stage: str | None
    failure_reason: str | None
    retry_events: list[dict[str, Any]]


AsyncTextCallFn = Callable[[str], Awaitable[tuple[str, dict[str, Any]]]]
SyncTextCallFn = Callable[[str], tuple[str, dict[str, Any]]]
AsyncToolCallFn = Callable[[Any], Awaitable[tuple[str, dict[str, Any]]]]
SyncToolCallFn = Callable[[Any], tuple[str, dict[str, Any]]]

TOOL_EXECUTION_MODES = ("llm_only", "multi_turn_search", "vita_tool")

# When Stage-2 omits user_prompt, use a short embed-safe default (full task is in GLOBAL TASK CONTEXT at runtime).
_RUNTIME_USER_PROMPT_FALLBACK = (
    "Read GLOBAL TASK CONTEXT for the full task, constraints, and required output format. "
    "Use UPSTREAM DEPENDENCIES DATA when present. Execute only the sub-task described in ROLE INSTRUCTION."
)


def _strip_leading_import_noise(mas_code: str) -> str:
    """
    Remove leading import / __future__ lines from model-emitted ``mas_code`` so that
    ``prepend_fixed_imports`` does not duplicate import blocks (a common cause of SyntaxError).
    """
    lines = (mas_code or "").splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        if s.startswith("from __future__"):
            i += 1
            continue
        if s.startswith("from typing"):
            i += 1
            continue
        if s.startswith("import "):
            i += 1
            continue
        if "template.sub_agent" in s and s.startswith("from "):
            i += 1
            continue
        break
    return "\n".join(lines[i:]).lstrip("\n")


def _normalize_dataset_name(dataset_name: str | None) -> str:
    s = (dataset_name or "").strip().lower()
    if s in {"hle", "hlemath"}:
        return "hlemath"
    if s in {"drb", "drb_bridge"}:
        return "drb"
    if s in {"bcp", "browsecomp"}:
        return "bcp"
    if s in {"vita", "vitabench"}:
        return "vita"
    raise ValueError(
        "Unsupported dataset_name. Expected one of: "
        "hlemath|hle, drb|drb_bridge, bcp|browsecomp, vita|vitabench; "
        f"got {dataset_name!r}"
    )


def _dataset_final_requirement_text(normalized_dataset: str) -> str:
    """
    What the *benchmark / evaluator* ultimately expects for this dataset.
    Fed into Stage 1 so the JSON \"goal\" and decomposition align with downstream grading.
    Summaries are aligned with dataset-specific runner prompts (e.g. BCP single-agent prompts).
    """
    d = (normalized_dataset or "").strip().lower()

    # hard_constraints = "When you design the whole multi-agent system, you must consider the final required output format of the dataset. For example, when it ask for a comprehensive report, your final node/agent should output a comprehensive report. Even when you use a validator or checker, you should output the final answer in the required format after the validation or checking."
    if d == "bcp":
        return (
            "BrowseComp-Plus (open-domain QA with retrieval): behave as a careful assistant; "
            "answers must be grounded only in the provided retrieval snippets, not in unstated facts. "
            "The required deliverable is a single final answer string only (no chain-of-thought, no bullet list)."
        )
    if d == "hlemath":
        return (
            "HLE/Math: IMO-level problem solving with emphasis on correctness and concise derivation. "
            "The final line of the solution must be exactly one answer in the form \\boxed{...} "
            "(strict final-answer formatting as in the single-agent user prompt)."
        )
    if d == "drb":
        return (
            "Deep Research Bench: produce a long-form research article that fully addresses the task prompt. "
            "Please provide a comprehensive research report."
        )
    if d == "vita":
        return (
            "[IMPORTANT] VitaBench grading contract (mandatory for decomposition): scoring is based on **verified tool "
            "calls and simulator state** (orders, reservations, payments), not on natural-language plans or "
            '"audit passed" summaries alone. '
            "When the user asks to buy, book, order, pay, or reserve something, you MUST decompose into "
            "sub-tasks whose **success_criteria** imply completing the transaction via environment tools "
            "(e.g. appropriate create_*/pay_*/book_* style actions as required by the scenario), not stopping "
            "at search/list/price-check/information-only steps unless the task is explicitly lookup-only. "
            "Never assign a sub-agent instructions that forbid booking or purchasing when the task requires a "
            "completed transaction. Names like *_search only are discouraged when the rubric expects a "
            "completed purchase or reservation — prefer sub-task intents that match **done** actions."
        )
    raise ValueError(
        "Dataset not specified or generic: infer the intended deliverable and evaluation style from "
        "[TASK_TEXT] alone; prefer concise outputs that match typical grading for this task type."
    )


# --- HLEMATH-only build quality helpers (must not affect other datasets) ---

_HLEMATH_BANNED_NODE_RE = re.compile(
    r"literature|known_result|web_search|multi_turn|bm25|"
    r"retrieve_evidence|citation_search|open_domain|search_agent",
    re.IGNORECASE,
)
_HLEMATH_VERIFY_NODE_RE = re.compile(
    r"verify|validation|validate|check|reconcile|consistency",
    re.IGNORECASE,
)
_HLEMATH_FINAL_ROLE_RE = re.compile(
    r"final|format|boxed|answer|finalize|reconcile",
    re.IGNORECASE,
)


def _hlemath_topology_templates_block() -> str:
    return (
        "[HLEMATH_TOPOLOGY_TEMPLATES — pick exactly ONE]\n"
        "1) interpret_compute_verify_finalize (default): parse givens/constraints → core derivation "
        "→ independent verification → single \\boxed{...} final line.\n"
        "2) dual_solver_reconcile_finalize: two independent solution attempts → reconcile conflicts "
        "→ verify → \\boxed{...}.\n"
        "3) classify_enumerate_verify_finalize: classify cases → enumerate/count with explicit bounds "
        "→ verify completeness → \\boxed{...}.\n"
        "Forbidden node capabilities for HLEMATH (llm_only): literature recall, web/search retrieval, "
        "citation lookup, or 'known results' databases — they add hallucination risk without tools.\n"
        "Prefer 3–4 high-signal nodes; avoid chains longer than 5 nodes.\n"
    )


def _hlemath_stage1_build_constraints() -> list[str]:
    return [
        "Select one topology template from [HLEMATH_TOPOLOGY_TEMPLATES] and reflect it in sub_tasks + dependencies.",
        "Do not hardcode unstated assumptions in decomposition (e.g., silently resolving ambiguous symbols).",
        "If ambiguity exists, include explicit disambiguation/reconciliation capability instead of fixing one interpretation too early.",
        "Keep decomposition compact (typically 3–4 nodes, never more than 5).",
        "Include at least one verification/reconciliation node before the final formatting node.",
        "Do not design retrieval/literature/search nodes — HLEMATH runs llm_only.",
    ]


def _hlemath_stage2_build_constraints() -> list[str]:
    return (
        "[HLEMATH_AGENT_PROTOCOL]\n"
        "- All inter-agent deliverables: plain text only (no JSON/dict/list payloads, no markdown code fences).\n"
        "- INTERMEDIATE agents: derivation artifacts only (givens, key steps, intermediate values, consistency check). "
        "Must NOT output the final \\boxed{...} answer.\n"
        "- FINAL agent: must re-derive from GLOBAL TASK CONTEXT + upstream; do not merely summarize upstream. "
        "Recompute critical steps; reconcile conflicting upstream values with explicit calculation. "
        "Final line must be exactly one \\boxed{...}.\n"
        "- Forbid unstated mathematical assumptions; mark unavoidable ones with ASSUMPTION: and justify.\n"
        "- Avoid shallow one-line outputs at every node.\n"
    )


def _hlemath_validate_stage1_decomposition(stage1: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    raw = stage1.get("sub_tasks")
    items: list[dict[str, Any]] = []
    if isinstance(raw, list):
        items = [x for x in raw if isinstance(x, dict)]
    elif isinstance(raw, dict):
        items = [raw]
    n = len(items)
    if n < 2:
        issues.append("sub_tasks must contain at least 2 nodes for a meaningful MAS pipeline.")
    if n > 5:
        issues.append(f"sub_tasks has {n} nodes; HLEMATH prefers 3–4 (max 5) to avoid context dilution.")
    has_verify = False
    for item in items:
        name = str(item.get("name") or "")
        desc = str(item.get("description") or "")
        blob = f"{name} {desc}"
        if _HLEMATH_BANNED_NODE_RE.search(blob):
            issues.append(
                f"node '{name}' suggests retrieval/literature/search, which is unsuitable for llm_only HLEMATH."
            )
        if _HLEMATH_VERIFY_NODE_RE.search(blob):
            has_verify = True
    if n >= 3 and not has_verify:
        issues.append(
            "decomposition lacks an explicit verification/reconciliation node before final answer formatting."
        )
    return len(issues) == 0, issues


def _build_hlemath_stage1_repair_prompt(
    *,
    original_prompt: str,
    stage1_json: dict[str, Any],
    issues: list[str],
) -> str:
    return (
        "You are revising Stage-1 HLEMATH task decomposition only (not solving the math problem).\n"
        "Fix the structural issues below while preserving the original task intent.\n"
        "Requirements:\n"
        "- Return ONE valid JSON object matching [OUTPUT_JSON_SCHEMA] from ORIGINAL_STAGE_PROMPT.\n"
        "- Pick one template from [HLEMATH_TOPOLOGY_TEMPLATES].\n"
        "- 3–4 nodes preferred; max 5; include verification before finalize.\n"
        "- No retrieval/literature/search nodes.\n"
        "- sub_tasks describe capabilities/processes only, not solution content.\n\n"
        f"[ISSUES_TO_FIX]\n" + "\n".join(f"- {x}" for x in issues) + "\n\n"
        f"[PREVIOUS_STAGE1_JSON]\n{json.dumps(stage1_json, ensure_ascii=False, indent=2)}\n\n"
        f"[ORIGINAL_STAGE_PROMPT]\n{original_prompt}\n\n"
        "Output revised Stage-1 JSON now."
    )


def _hlemath_validate_stage2_sub_agents(
    subs: list[SubTaskSpec],
    *,
    final_idx: int,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for idx, spec in enumerate(subs):
        rn = (spec.role_name or "").strip().lower()
        ri = (spec.role_instruction or "").lower()
        is_final = idx == final_idx
        if not is_final:
            if re.search(r"\\boxed\s*\{", ri) or "final line must be" in ri and "boxed" in ri:
                issues.append(
                    f"intermediate agent '{spec.role_name}' instructs a final \\boxed{{...}} output."
                )
            if _HLEMATH_FINAL_ROLE_RE.search(rn) and not _HLEMATH_VERIFY_NODE_RE.search(rn):
                issues.append(
                    f"intermediate agent '{spec.role_name}' looks like a final-role name; "
                    "rename or narrow its scope to non-final derivation."
                )
    return len(issues) == 0, issues


def _hlemath_role_looks_like_final(role_name: str) -> bool:
    rn = (role_name or "").strip().lower()
    if not rn:
        return False
    if _HLEMATH_VERIFY_NODE_RE.search(rn):
        return False
    return bool(_HLEMATH_FINAL_ROLE_RE.search(rn))


def _hlemath_validate_mas_code_behavior(
    mas_code: str,
    *,
    sub_agent_role_names: list[str],
    preferred_final_role_name: str | None,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    fo_lines = [
        ln
        for ln in mas_code.splitlines()
        if re.search(r'state\["final_output"\]\s*=', ln)
        and not ln.strip().startswith("#")
    ]
    agent_ref_lines = [
        ln for ln in fo_lines if re.search(r'state\["out_\w+"\]', ln)
    ]
    if len(agent_ref_lines) != 1:
        issues.append(
            f"expected exactly one agent-referenced final_output assignment, found {len(agent_ref_lines)}."
        )
    final_role = (preferred_final_role_name or "").strip()
    if final_role:
        expected = f'state["out_{final_role}"]'
        if agent_ref_lines and expected not in agent_ref_lines[0]:
            issues.append(
                f"final_output should reference {expected} (preferred final agent for HLEMATH)."
            )
        if expected not in mas_code:
            issues.append(f"mas_code never writes {expected}; final agent may be missing from workflow.")
    for role in sub_agent_role_names:
        if f'out_{role}' in mas_code or f'"out_{role}"' in mas_code:
            continue
        # Role keys in generated code may be sanitized; only warn on preferred final
        if role == final_role:
            issues.append(f"preferred final role '{role}' has no out_* state key in mas_code.")
    return len(issues) == 0, issues


def _allowed_tool_modes_for_dataset(dataset_name: str) -> set[str]:
    if dataset_name in {"hlemath", "drb"}:
        return {"llm_only"}
    if dataset_name == "bcp":
        return {"multi_turn_search"}
    if dataset_name == "vita":
        return {"vita_tool"}
    raise ValueError(
        "Unsupported dataset_name. Expected one of: "
        "hlemath|hle, drb|drb_bridge, bcp|browsecomp, vita|vitabench; "
        f"got {dataset_name!r}"
    )


def _default_execution_mode_for_dataset(allowed: set[str]) -> str:
    """When the model omits execution_mode or picks one illegal for this dataset, fall back legally."""
    if len(allowed) == 1:
        return next(iter(allowed))
    if "llm_only" in allowed:
        return "llm_only"
    return sorted(allowed)[0]


def _sanitize_tool_context(tool_context: dict[str, Any] | None, *, dataset_name: str) -> dict[str, Any]:
    tc = dict(tool_context or {})
    raw_mode = str(tc.get("execution_mode", "")).strip().lower()
    allowed = _allowed_tool_modes_for_dataset(dataset_name)
    valid = set(TOOL_EXECUTION_MODES)
    mode = raw_mode if raw_mode in valid else None
    if mode is None:
        mode = _default_execution_mode_for_dataset(allowed)
    elif mode not in allowed:
        mode = _default_execution_mode_for_dataset(allowed)
    tc["execution_mode"] = mode
    return tc


def _enforce_dataset_tool_modes_in_mas_code(mas_code: str, *, dataset_name: str) -> str:
    """
    Stage 3 codegen may ignore Stage-2-normalized ``tool_context`` and copy template placeholders
    (e.g. ``llm_only`` from ``template_text``). Rewrite ``SubAgentRequest`` literals to match the
    dataset's allowed modes (same rules as ``_sanitize_tool_context``).
    """
    try:
        nd = _normalize_dataset_name(dataset_name)
    except ValueError:
        return mas_code
    allowed = _allowed_tool_modes_for_dataset(nd)
    if len(allowed) != 1:
        return mas_code
    only = next(iter(allowed))
    out = mas_code
    if only == "vita_tool":
        for wrong in ("llm_only", "multi_turn_search"):
            out = out.replace(f'"execution_mode": "{wrong}"', '"execution_mode": "vita_tool"')
            out = out.replace(f"'execution_mode': '{wrong}'", "'execution_mode': 'vita_tool'")
        return out
    if only == "llm_only":
        for wrong in ("vita_tool", "multi_turn_search"):
            out = out.replace(f'"execution_mode": "{wrong}"', '"execution_mode": "llm_only"')
            out = out.replace(f"'execution_mode': '{wrong}'", "'execution_mode': 'llm_only'")
        return out
    if only == "multi_turn_search":
        for wrong in ("llm_only", "vita_tool"):
            out = out.replace(f'"execution_mode": "{wrong}"', '"execution_mode": "multi_turn_search"')
            out = out.replace(f"'execution_mode': '{wrong}'", "'execution_mode': 'multi_turn_search'")
        return out
    return out


def _sanitize_final_output_in_mas_code(
    mas_code: str,
    *,
    sub_agent_role_names: list[str],
    preferred_final_role_name: str | None = None,
) -> str:
    """
    Post-process Stage-3 ``mas_code`` to enforce P0 correctness rules:

    1. Remove **all** hardcoded string-literal assignments to ``state["final_output"]``
       (especially ``"UNANSWERABLE: ..."``, but also any other static string).
       The final output MUST come from an executed sub-agent, not a hardcoded fallback.

    2. When multiple assignments to ``state["final_output"]`` exist, keep only the
       **last** one that references a ``state["out_..."]`` key (i.e. a sub-agent output).
       Remove all earlier assignments, especially UNANSWERABLE fallbacks.

    3. If no ``state["final_output"] = state["out_..."]`` assignment exists at all,
       insert one that references the **last** sub-agent defined in ``__init__``.
       This handles the common case where Stage-3 omits or botches the final assignment.

    4. Replace f-string concatenation assignments (``f"...{state.get(...)}..."``)
       with a direct sub-agent output reference, preferring the final/merge agent.

    Parameters
    ----------
    mas_code:
        Generated Python source from Stage 3.
    sub_agent_role_names:
        Ordered list of role names from ``normalized_sub_agents`` — used to determine
        the intended final agent when no valid assignment is found.
    """
    lines = mas_code.splitlines()
    out_lines: list[str] = []
    removed_unanswerable = 0
    removed_hardcoded = 0
    removed_f_string = 0
    kept_agent_ref_lines: list[str] = []

    # Pattern for state["final_output"] = ...
    fo_assign_re = re.compile(r'^(\s*)state\["final_output"\]\s*=\s*(.+)$')
    # Identify value types
    unanswerable_re = re.compile(r'^\s*["\']UNANSWERABLE', re.IGNORECASE)
    hardcoded_str_re = re.compile(r'^\s*["\'][^"\']*["\']\s*$')
    fstring_re = re.compile(r'^\s*f["\']')
    agent_ref_re = re.compile(r'^\s*state\["out_(\w+)"\]')
    # Also match state.get patterns that are used in f-string contexts
    state_get_re = re.compile(r'^\s*state\.get\(')

    for line in lines:
        m = fo_assign_re.match(line)
        if m is None:
            out_lines.append(line)
            continue

        indent = m.group(1)
        value = m.group(2).strip()

        # Case 1: UNANSWERABLE hardcoded string → remove entirely
        if unanswerable_re.match(value):
            removed_unanswerable += 1
            # Drop a comment instead so the code structure stays valid
            out_lines.append(f"{indent}# [P0-SANITIZE] removed UNANSWERABLE hardcoded fallback")
            continue

        # Case 2: Other hardcoded string literal (not UNANSWERABLE, but still not agent output)
        if hardcoded_str_re.match(value) and not agent_ref_re.match(value):
            removed_hardcoded += 1
            out_lines.append(f"{indent}# [P0-SANITIZE] removed hardcoded string final_output")
            continue

        # Case 3: f-string concatenation → remove (will be replaced with agent ref later)
        if fstring_re.match(value):
            removed_f_string += 1
            out_lines.append(f"{indent}# [P0-SANITIZE] removed f-string concatenation final_output")
            continue

        # Case 4: agent reference — keep, and track it
        if agent_ref_re.match(value):
            kept_agent_ref_lines.append(line)
            out_lines.append(line)
            continue

        # Case 5: intermediate variable (e.g. final_summary, final_answer)
        # These may or may not be valid. Keep them for now — if no agent_ref
        # exists, we will insert one later.
        out_lines.append(line)
        # Track as a "kept" line so we can decide later
        kept_agent_ref_lines.append(line)
        continue

    # After first pass: check if any agent_ref assignment survived.
    # Use a relaxed pattern (no ^ anchor) since kept_agent_ref_lines store full lines,
    # not just the RHS value.
    _agent_ref_relaxed = re.compile(r'state\["out_\w+"\]')
    has_agent_ref = any(_agent_ref_relaxed.search(l) for l in kept_agent_ref_lines)

    if not has_agent_ref and sub_agent_role_names:
        # No valid state["final_output"] = state["out_XXX"] found.
        # Insert one referencing the last sub-agent (which _with_hard_instruction_constraints
        # marked as is_final=True, meaning it should produce the complete final answer).
        final_role = (preferred_final_role_name or "").strip() or sub_agent_role_names[-1]
        # Find the indentation of the last forward_async section
        # Look for "return state" line to determine where to insert
        insert_idx = None
        for i, line in enumerate(out_lines):
            if re.match(r'\s*return\s+state\s*', line):
                insert_idx = i
                break

        if insert_idx is not None:
            # Find indentation from surrounding context
            indent_match = re.match(r'^(\s*)', out_lines[insert_idx])
            indent = indent_match.group(1) if indent_match else "        "
            # Insert the final_output assignment just before "return state"
            insert_line = f'{indent}state["final_output"] = state["out_{final_role}"]'
            out_lines.insert(insert_idx, insert_line)
            # Also add usage_final if it doesn't exist
            usage_line = f'{indent}state["usage_final"] = state["usage_{final_role}"]'
            out_lines.insert(insert_idx + 1, usage_line)
            print(
                f"[P0-SANITIZE] inserted final_output assignment: "
                f'state["final_output"] = state["out_{final_role}"]',
                flush=True,
            )
        else:
            # Fallback: append at the very end of forward_async
            # Find the end of forward_async method
            for i in range(len(out_lines) - 1, -1, -1):
                if "return state" in out_lines[i] or "return dict" in out_lines[i]:
                    insert_idx = i
                    break
            if insert_idx is not None:
                indent_match = re.match(r'^(\s*)', out_lines[insert_idx])
                indent = indent_match.group(1) if indent_match else "        "
                out_lines.insert(insert_idx, f'{indent}state["final_output"] = state["out_{final_role}"]')
                out_lines.insert(insert_idx + 1, f'{indent}state["usage_final"] = state["usage_{final_role}"]')
            else:
                print("[P0-SANITIZE] WARNING: could not find return statement to insert final_output", flush=True)

    # Also: if there are multiple agent_ref assignments, keep only the last one
    # (this handles cases where both UNANSWERABLE and agent_ref were present,
    #  and we already removed the UNANSWERABLE ones above)
    _agent_ref_relaxed = re.compile(r'state\["out_\w+"\]')
    agent_ref_indices = [i for i, line in enumerate(out_lines) if fo_assign_re.match(line) and _agent_ref_relaxed.search(line)]
    if len(agent_ref_indices) > 1:
        # Keep only the last one
        for idx in agent_ref_indices[:-1]:
            indent_match = re.match(r'^(\s*)', out_lines[idx])
            indent = indent_match.group(1) if indent_match else "        "
            out_lines[idx] = f"{indent}# [P0-SANITIZE] removed duplicate final_output assignment (kept last one)"

    sanitized = "\n".join(out_lines)
    if removed_unanswerable or removed_hardcoded or removed_f_string:
        print(
            f"[P0-SANITIZE] removed: {removed_unanswerable} UNANSWERABLE, "
            f"{removed_hardcoded} hardcoded strings, {removed_f_string} f-string concatenations",
            flush=True,
        )
    return sanitized


def _enforce_final_output_role_in_mas_code(
    mas_code: str,
    *,
    final_role_name: str | None,
) -> str:
    """
    Force ``state["final_output"]`` / ``state["usage_final"]`` to reference one chosen role.

    Used only for VitaBench, where wrong final-role binding can silently drop completed branches.
    """
    role = (final_role_name or "").strip()
    if not role:
        return mas_code
    role_ref = f'state["out_{role}"]'
    if role_ref not in mas_code:
        # Do not force an impossible role reference when stage-3 omitted this output key.
        return mas_code

    lines = mas_code.splitlines()
    fo_assign_re = re.compile(r'^(\s*)state\["final_output"\]\s*=')
    uf_assign_re = re.compile(r'^(\s*)state\["usage_final"\]\s*=')
    return_idx = None
    base_indent = "        "
    new_lines: list[str] = []
    for i, line in enumerate(lines):
        if return_idx is None and re.match(r"^\s*return\s+state\s*$", line):
            return_idx = i
            m = re.match(r"^(\s*)", line)
            if m:
                base_indent = m.group(1)
        if fo_assign_re.match(line):
            indent = re.match(r"^(\s*)", line).group(1)
            new_lines.append(f"{indent}# [VITA-FINAL-ENFORCE] replaced final_output assignment")
            continue
        if uf_assign_re.match(line):
            indent = re.match(r"^(\s*)", line).group(1)
            new_lines.append(f"{indent}# [VITA-FINAL-ENFORCE] replaced usage_final assignment")
            continue
        new_lines.append(line)

    if return_idx is None:
        return mas_code

    # return_idx is for original lines; relocate in rewritten list by first matching return state.
    insert_idx = None
    for i, line in enumerate(new_lines):
        if re.match(r"^\s*return\s+state\s*$", line):
            insert_idx = i
            break
    if insert_idx is None:
        return mas_code

    forced_lines = [
        f'{base_indent}state["final_output"] = state["out_{role}"]',
        f'{base_indent}state["usage_final"] = state.get("usage_{role}", state.get("usage_final", {{}}))',
    ]
    new_lines[insert_idx:insert_idx] = forced_lines
    return "\n".join(new_lines)


def make_planner_call_fn(
    llm_client: Any,
    *,
    system_prompt: str = "You are a MAS planner. Output JSON only.",
    generation_kwargs: dict[str, Any] | None = None,
) -> Callable[[str], Awaitable[str]]:
    """
    Build a sync ``planner_call_fn`` from a generic async LLM client.

    The client is expected to expose:
      async generate(user_prompt=..., system_prompt=...) -> (text, usage)
    """

    gen_fn = getattr(llm_client, "generate")
    is_async_generate = inspect.iscoroutinefunction(gen_fn)

    planner_kwargs = dict(generation_kwargs or {})

    def _is_unsupported_generation_kw_error(err: Exception) -> bool:
        msg = str(err).lower()
        return any(
            tok in msg
            for tok in (
                "unsupported",
                "unknown",
                "unexpected keyword",
                "invalid parameter",
                "response_format",
            )
        )

    async def _planner_call_async(prompt: str) -> str:
        async def _call_once(with_kwargs: bool) -> tuple[str, dict[str, Any]]:
            call_kwargs = dict(planner_kwargs) if with_kwargs else {}
            if is_async_generate:
                return await gen_fn(user_prompt=prompt, system_prompt=system_prompt, **call_kwargs)
            return await asyncio.to_thread(
                gen_fn,
                user_prompt=prompt,
                system_prompt=system_prompt,
                **call_kwargs,
            )

        try:
            text, _usage = await _call_once(with_kwargs=bool(planner_kwargs))
        except Exception as e:
            if not planner_kwargs or not _is_unsupported_generation_kw_error(e):
                raise
            print("[planner_call_fn] generation_kwargs unsupported by backend; fallback to default kwargs.", flush=True)
            text, _usage = await _call_once(with_kwargs=False)
        return text

    return _planner_call_async


def make_text_call_fn(
    llm_client: Any,
    *,
    system_prompt: str = "",
) -> Callable[[str], Awaitable[tuple[str, dict[str, Any]]]]:
    """Build async text_call_fn from a generic async LLM client."""

    gen_fn = getattr(llm_client, "generate")
    is_async_generate = inspect.iscoroutinefunction(gen_fn)

    async def _text_call_async(prompt: str) -> tuple[str, dict[str, Any]]:
        if is_async_generate:
            return await gen_fn(user_prompt=prompt, system_prompt=system_prompt)
        return await asyncio.to_thread(
            gen_fn,
            user_prompt=prompt,
            system_prompt=system_prompt,
        )

    return _text_call_async


def merge_usage_totals(acc: dict[str, Any], usage: dict[str, Any]) -> None:
    acc["prompt_tokens"] = int(acc.get("prompt_tokens", 0)) + int(
        usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    )
    acc["output_tokens"] = int(acc.get("output_tokens", 0)) + int(usage.get("output_tokens", 0) or 0)
    acc["total_tokens"] = int(acc.get("total_tokens", 0)) + int(
        usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("output_tokens", 0)) or 0
    )
    acc["estimated_cost_usd"] = float(acc.get("estimated_cost_usd", 0.0)) + float(
        usage.get("estimated_cost_usd", 0.0) or 0.0
    )


def run_fixed_phase_pipeline(
    *,
    phase_indices: list[int],
    banks: dict[int, list[Any]],
    route_phase_fn: Callable[[int, list[Any], dict[str, str]], RouteDecision],
    build_role_instruction_fn: Callable[[int, str], tuple[str, str]],
    build_user_prompt_fn: Callable[[int, str, dict[str, str]], str],
    execute_phase_fn: Callable[[int, str, str, Any, str], PhaseExecutionResult],
    usage_merge_fn: Callable[[dict[str, Any], dict[str, Any]], None] = merge_usage_totals,
    initial_prior_outputs: dict[str, str] | None = None,
    initial_per_phase: list[dict[str, Any]] | None = None,
    initial_steps: list[dict[str, Any]] | None = None,
    initial_usage_totals: dict[str, Any] | None = None,
) -> PhasePipelineResult:
    prior_outputs: dict[str, str] = dict(initial_prior_outputs or {})
    per_phase: list[dict[str, Any]] = list(initial_per_phase or [])
    steps: list[dict[str, Any]] = list(initial_steps or [])
    usage_totals: dict[str, Any] = dict(
        initial_usage_totals
        or {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    )

    for phase_idx in phase_indices:
        specs = banks.get(phase_idx) or []
        if not specs:
            raise RuntimeError(f"phase_{phase_idx} bank empty")
        route = route_phase_fn(phase_idx, specs, prior_outputs)
        usage_merge_fn(usage_totals, route.usage)
        steps.append(
            {
                "stage": f"skill_router_phase_{phase_idx}",
                "request": {"phase": phase_idx, "candidates": [s.name for s in specs]},
                "response": {
                    "selected_skills": [route.selected_skill_name],
                    "reasoning": route.reasoning,
                    "raw_response": route.raw_response,
                },
                "usage": route.usage,
            }
        )
        selected = next((s for s in specs if s.name == route.selected_skill_name), specs[0])
        role, instruction = build_role_instruction_fn(phase_idx, selected.name)
        user_prompt = build_user_prompt_fn(phase_idx, role, prior_outputs)
        stage = execute_phase_fn(phase_idx, role, instruction, selected, user_prompt)
        usage_merge_fn(usage_totals, stage.usage)
        steps.append(
            {
                "stage": role,
                "active_skill": selected.name,
                "request": {"instruction": instruction, "user_prompt": user_prompt},
                "response": stage.text,
                "usage": stage.usage,
            }
        )
        prior_outputs[role] = stage.text
        per_phase.append(
            {
                "phase": phase_idx,
                "role": role,
                "selected_skill": selected.name,
                "selected_skill_path": str(selected.path.resolve()),
                "router_reasoning": route.reasoning,
                "candidate_skills": [s.name for s in specs],
                "router_raw_response": route.raw_response,
                **(stage.meta or {}),
            }
        )
    return PhasePipelineResult(
        prior_outputs=prior_outputs,
        per_phase=per_phase,
        steps=steps,
        usage_totals=usage_totals,
    )


async def run_fixed_phase_pipeline_async(
    *,
    phase_indices: list[int],
    banks: dict[int, list[Any]],
    route_phase_fn: Callable[[int, list[Any], dict[str, str]], RouteDecision],
    build_role_instruction_fn: Callable[[int, str], tuple[str, str]],
    build_user_prompt_fn: Callable[[int, str, dict[str, str]], str],
    execute_phase_fn: Callable[[int, str, str, Any, str], Awaitable[PhaseExecutionResult]],
    usage_merge_fn: Callable[[dict[str, Any], dict[str, Any]], None] = merge_usage_totals,
    initial_prior_outputs: dict[str, str] | None = None,
    initial_per_phase: list[dict[str, Any]] | None = None,
    initial_steps: list[dict[str, Any]] | None = None,
    initial_usage_totals: dict[str, Any] | None = None,
) -> PhasePipelineResult:
    """Same as ``run_fixed_phase_pipeline`` but awaits async ``execute_phase_fn``."""
    prior_outputs: dict[str, str] = dict(initial_prior_outputs or {})
    per_phase: list[dict[str, Any]] = list(initial_per_phase or [])
    steps: list[dict[str, Any]] = list(initial_steps or [])
    usage_totals: dict[str, Any] = dict(
        initial_usage_totals
        or {
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
    )

    for phase_idx in phase_indices:
        specs = banks.get(phase_idx) or []
        if not specs:
            raise RuntimeError(f"phase_{phase_idx} bank empty")
        route = route_phase_fn(phase_idx, specs, prior_outputs)
        usage_merge_fn(usage_totals, route.usage)
        steps.append(
            {
                "stage": f"skill_router_phase_{phase_idx}",
                "request": {"phase": phase_idx, "candidates": [s.name for s in specs]},
                "response": {
                    "selected_skills": [route.selected_skill_name],
                    "reasoning": route.reasoning,
                    "raw_response": route.raw_response,
                },
                "usage": route.usage,
            }
        )
        selected = next((s for s in specs if s.name == route.selected_skill_name), specs[0])
        role, instruction = build_role_instruction_fn(phase_idx, selected.name)
        user_prompt = build_user_prompt_fn(phase_idx, role, prior_outputs)
        stage = await execute_phase_fn(phase_idx, role, instruction, selected, user_prompt)
        usage_merge_fn(usage_totals, stage.usage)
        steps.append(
            {
                "stage": role,
                "active_skill": selected.name,
                "request": {"instruction": instruction, "user_prompt": user_prompt},
                "response": stage.text,
                "usage": stage.usage,
            }
        )
        prior_outputs[role] = stage.text
        per_phase.append(
            {
                "phase": phase_idx,
                "role": role,
                "selected_skill": selected.name,
                "selected_skill_path": str(selected.path.resolve()),
                "router_reasoning": route.reasoning,
                "candidate_skills": [s.name for s in specs],
                "router_raw_response": route.raw_response,
                **(stage.meta or {}),
            }
        )
    return PhasePipelineResult(
        prior_outputs=prior_outputs,
        per_phase=per_phase,
        steps=steps,
        usage_totals=usage_totals,
    )


def _load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _enumerate_json_dicts(text: str) -> list[dict[str, Any]]:
    """Scan left-to-right and parse each complete top-level JSON object (non-overlapping)."""
    raw = text or ""
    decoder = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    i = 0
    n = len(raw)
    while i < n:
        if raw[i] != "{":
            i += 1
            continue
        try:
            obj, consumed = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i += consumed
    return out


def _stage_hint_from_label(stage_name: str) -> int | None:
    m = re.search(r"Stage\s*(\d+)", stage_name, flags=re.I)
    return int(m.group(1)) if m else None


def _is_tool_context_fragment(obj: dict[str, Any]) -> bool:
    """Small JSON blobs that appear before the real Stage 2 wrapper (examples / tool_context)."""
    ks = set(obj.keys())
    return ks <= {"execution_mode"} or ks == {"execution_mode", "reasoning"}


def _extract_best_json_dict_for_stage(text: str, *, stage_hint: int | None) -> dict[str, Any] | None:
    """Prefer schema-shaped dicts over incidental inner JSON (e.g. {{\"execution_mode\": ...}} only)."""
    cands = _enumerate_json_dicts(text)
    if not cands:
        return None
    if stage_hint is None:
        return cands[0]
    if stage_hint == 1:
        scored: list[tuple[int, dict[str, Any]]] = []
        for obj in cands:
            st = obj.get("sub_tasks")
            score = 0
            if isinstance(st, list) and len(st) > 0:
                score += 100 + len(st)
            elif isinstance(st, list):
                score += 10
            if obj.get("goal"):
                score += 5
            if isinstance(st, dict):
                score += 50
            n = str(obj.get("name", "")).strip()
            d = str(obj.get("description", "")).strip()
            if n and d:
                score += 25
            scored.append((score, obj))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]
    if stage_hint == 2:
        pool = [o for o in cands if not _is_tool_context_fragment(o)] or cands
        best: dict[str, Any] | None = None
        best_score = -1
        for obj in pool:
            sa = obj.get("sub_agents")
            score = 0
            if isinstance(sa, list) and len(sa) > 0:
                score = 100 + len(sa)
            elif isinstance(sa, dict):
                score = 40
            rn = str(obj.get("role_name", "")).strip()
            ri = str(obj.get("role_instruction", "")).strip()
            if rn and ri:
                score = max(score, 80)
            if score > best_score:
                best_score = score
                best = obj
        if best is None and pool:
            return max(pool, key=lambda o: len(o.keys()))
        return best
    if stage_hint == 3:
        best: dict[str, Any] | None = None
        best_len = -1
        for obj in cands:
            mc = obj.get("mas_code")
            if isinstance(mc, str) and mc.strip():
                if len(mc) > best_len:
                    best_len = len(mc)
                    best = obj
        return best
    return cands[0]


def _extract_first_json_dict(text: str) -> dict[str, Any] | None:
    return _extract_best_json_dict_for_stage(text, stage_hint=None)


def _coerce_stage1_subtasks(payload: dict[str, Any]) -> dict[str, Any]:
    """If Stage 1 omits sub_tasks[] but returns a single task at top level, normalize."""
    raw_st = payload.get("sub_tasks")
    if isinstance(raw_st, dict):
        return {**payload, "sub_tasks": [raw_st]}
    if isinstance(raw_st, list) and len(raw_st) > 0:
        return payload
    n = str(payload.get("name", "")).strip()
    d = str(payload.get("description", "")).strip()
    if n and d:
        return {**payload, "sub_tasks": [{"name": n, "description": d}]}
    return payload


def _unwrap_stage2_agent_item(item: dict[str, Any]) -> dict[str, Any]:
    """Turn {\"role_id\": {\"role_name\": \"...\", ...}} into the inner agent dict."""
    if str(item.get("role_name", "")).strip():
        return item
    for v in item.values():
        if isinstance(v, dict) and str(v.get("role_name", "")).strip():
            return v
    if len(item) == 1:
        inner = next(iter(item.values()))
        if isinstance(inner, dict):
            return inner
    return item


def _extract_skill_stages(skill_text: str) -> dict[int, str]:
    """Extract stage module bodies from headings: ## 1., ## 2., ## 3."""
    matches = list(re.finditer(r"^##\s+([123])\.[^\n]*\n", skill_text, flags=re.M))
    out: dict[int, str] = {}
    for i, m in enumerate(matches):
        stage = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(skill_text)
        out[stage] = skill_text[start:end].strip()
    missing = [s for s in (1, 2, 3) if s not in out]
    if missing:
        raise RuntimeError(f"init skill missing stage modules: {missing}")
    return out


def _mas_build_contract_general_constraints() -> str:
    """Parser + cross-stage payload rules—task *how-to* stays in init SKILL.md."""
    return (
        "[GENERAL_CONSTRAINTS]\n"
        "- Single JSON object only (no markdown fences, no surrounding prose).\n"
        "- The runner parses the first top-level `{...}` object from your reply.\n"
        "- Key names must match [OUTPUT_JSON_SCHEMA]; prior-stage outputs are embedded verbatim into "
        "later [STAGE_INPUT_JSON] blobs.\n"
        "- Fields such as ``dataset_final_requirements`` encode **benchmark grading contracts** "
        "(evaluator-facing), not optional narrative hints.\n"
        "- Cross-Stage Execution Safety: Stage 2 ``role_instruction`` / ``user_prompt`` strings and Stage 3 "
        "``mas_code`` are ultimately embedded into **executable Python**. Any content that breaks Python string "
        "literals or syntax (unescaped quotes, unmatched parentheses, duplicate import blocks, invalid dict "
        "keys) will cause **total workflow failure** at runtime. Design all text so it survives embedding as "
        "Python source.\n\n"
    )


def _mas_build_contract_specific_constraints(stage: int) -> str:
    if stage == 1:
        return (
            "[STAGE_SPECIFIC_CONSTRAINTS]\n\n"
            "Stage 1: Task Decomposition — Engineering Constraints. "
            "- Node Count Limits: the number of decomposed sub-tasks must be kept within a reasonable range "
            "(typically 3-6 for most tasks), because over-fragmentation leads to context window explosion and "
            "information dilution, while under-fragmentation defeats the purpose of a MAS. "
            "- Strict Node Naming Convention: all sub-task IDs must be globally unique and formatted strictly "
            "in lowercase snake_case (e.g., data_analyzer). This ID serves as the absolute identifier across "
            "all stages, and modifying the spelling or casing in later stages is strictly forbidden.\n"
            "- Decomposition Principle — Describe WHAT, Not HOW: you are defining the structural blueprint of "
            "the multi-agent system, NOT solving the user's problem. Each sub-task should describe a capability "
            "or processing stage (e.g., 'retrieve_evidence', 'analyze_constraints', 'synthesize_answer'), not "
            "attempt to answer the question or provide solution content. The actual solving happens later when "
            "the generated agents execute at runtime.\n"
            "- Topology Awareness: consider whether the task benefits from a simple sequential chain, or needs "
            "a fan-out/fan-in pattern (multiple independent analyses merged at the end), or an iterative loop "
            "(e.g., retrieve → analyze → refine → retrieve again). Express this via the dependency structure. "
            "For tasks requiring trial-and-error (e.g., open-domain search), design at least one iterative or "
            "fallback-capable node rather than a single one-shot attempt.\n"
            "- Information Preservation: ensure each sub-task's description is self-contained enough that a "
            "downstream agent can understand the context without needing the full original query. Avoid designs "
            "where intermediate outputs become too thin (e.g., a 'filter' node that outputs only null/empty — "
            "design instead for nodes that always produce structured, actionable output even when uncertain.\n\n"
        )
    if stage == 2:
        return (
            "[STAGE_SPECIFIC_CONSTRAINTS]\n\n"
            "Stage 2: Agent Engineering — Engineering Constraints. "
            "- Absolute 1-to-1 Node Mapping: every sub-agent generated in this stage must strictly correspond "
            "to a sub-task defined in Stage 1; the role_name must perfectly match the node ID from Stage 1. "
            "Do not omit any nodes, and do not hallucinate new agents that were not planned. "
            "- Strict Boundary Enforcement: agent instructions and user prompts must NOT contain any routing "
            "or scheduling logic. Agents should never be instructed to \"pass the result to the reviewer "
            'agent.\" Agents must only focus on completing their specific sub-task. The system framework '
            "will handle the data routing.\n"
            "- You Are Engineering Prompts, Not Solving Problems: your output is a set of agent specifications "
            "(role_instruction + user_prompt) that will be used to call an LLM at runtime. Do NOT attempt to "
            "answer the user's question yourself. Your role_instruction defines WHAT the agent should do and "
            "its behavioral boundaries. The user_prompt must be **short** (see Runtime Task Context below); "
            "the full task is always injected at runtime in ``GLOBAL TASK CONTEXT``, so you do not need to "
            "duplicate the entire problem inside ``user_prompt``.\n"
            "- Runtime Task Context (Critical): at execution time, every sub-agent prompt is assembled with a "
            "``GLOBAL TASK CONTEXT`` section that already contains the **full original task_text**. Therefore "
            "your ``user_prompt`` MUST stay **short and Python-embeddable** (plain ASCII where possible): "
            "instruct the agent to read ``GLOBAL TASK CONTEXT`` for the full problem, citations, LaTeX, and "
            "output format — do **NOT** paste the entire task, code fences, or long LaTeX into ``user_prompt`` "
            "because Stage 3 copies these strings into ``SubAgentRequest(...)`` inside generated Python and "
            "unescaped ``\\\"`` or ``\"\"\"`` substrings will cause **SyntaxError** and zero score.\n"
            "- Avoid Empty/Null Output Designs: do NOT design agents whose natural output could be empty, null, "
            "or a simple 'not found' message. For example, instead of a 'filter_candidates' agent that might "
            "output null when no candidates match, design an 'analyze_and_rank' agent that always produces a "
            "structured ranking or analysis. Every agent should produce actionable, substantive output.\n"
            "- Execution Mode Reasoning: choose tool_context.execution_mode based on the actual capabilities "
            "needed. For tasks requiring information retrieval, use 'multi_turn_search' so the agent can "
            "perform multi-round searches. For pure reasoning/analysis, use 'llm_only'. Do not default all "
            "agents to 'llm_only' — the system's effectiveness depends on correctly routing tool-capable "
            "agents to the right execution mode.\n"
            "- Upstream-to-Retrieval Bridge (Critical for multi_turn_search agents): if a sub-agent with "
            "execution_mode='multi_turn_search' depends on an upstream agent, the upstream agent's output "
            "must contain actionable retrieval guidance — not just a re-formatted restatement of the original "
            "query constraints. The multi_turn_search agent uses a BM25 (lexical keyword) retrieval engine "
            "under the hood; its planner receives the upstream output as 'Delegation / planning context'. "
            "If the upstream output only contains vague abstractions like 'amphibian' or 'early 1990s', the "
            "BM25 planner cannot generate precise search queries. Instead, the upstream agent should produce: "
            "(a) concrete candidate entity names, specific search keywords, or named entities that BM25 can "
            "match directly; (b) a 'suggested_search_queries' list of 3-5 distinct BM25-ready queries; "
            "(c) any disambiguation clues (e.g., 'the amphibian reference likely means frog/newt/salamander "
            "in a company name'). If the upstream agent is a pure constraint extractor and cannot supply "
            "such concrete terms, then DO NOT place it before a multi_turn_search agent — instead, make "
            "the multi_turn_search agent the FIRST node in the pipeline so it receives the full original "
            "query directly and can derive its own search strategy.\n\n"
        )
    if stage == 3:
        return (
            "[STAGE_SPECIFIC_CONSTRAINTS]\n\n"
            "Stage 3: Workflow Orchestration & Code Generation — Engineering Constraints. "
            "- Template Immutability: you must strictly build upon the provided MASWorkflowTemplate; do not "
            "modify the function signatures (arguments or return types) of __init__, forward_async, or "
            "build_workflow_subagent_prompt; only inject code between the designated <START> and <END> markers. "
            "- No Third-Party Orchestration Frameworks: try not to import other libraries as much as possible; "
            "if you truly need to import external libraries, do not import or attempt to use external MAS "
            "frameworks like langgraph, autogen, or crewai; use only Python standard libraries. "
            "- No Duplicate Imports: do **NOT** emit ``from __future__ import annotations``, ``from typing ...``, "
            "or ``from template.sub_agent ...`` — the runner prepends a fixed import block. Emit only the "
            "workflow class body. Duplicate imports cause invalid Python and immediate failure.\n"
            "- Syntactically Valid Python: before returning ``mas_code``, mentally verify matching "
            "parentheses/brackets/braces, valid string literals, and valid dict keys in ``upstream_outputs`` "
            "(use simple snake_case string keys only, e.g. ``\"upstream_a\"`` — never use task text fragments "
            "or LaTeX as dict keys).\n"
            "- String Literal Safety for SubAgentRequest fields: when embedding ``role_instruction`` and "
            "``user_prompt`` into Python, prefer **triple-single-quoted** raw strings ``r'''...'''`` or "
            "``'''...'''`` and ensure the chosen delimiter never appears unescaped inside the text. If you "
            "use double-quoted strings, every internal ``\"`` must be escaped as ``\\\"``. Never embed "
            "``\"\"\"`` inside triple-double-quoted strings.\n"
            "- Safe State Dictionary Access: when populating the upstream_outputs dictionary in forward_async, "
            "the values must only reference keys in the state dictionary that have ALREADY been executed and "
            "populated; do not attempt to access a node's output before it has been awaited, and do not "
            "hallucinate variable names. "
            "- Force Sequential Await (No Complex Concurrency): even if the DAG topology contains logically "
            "parallel branches, you must generate the code using sequential await calls; do not use "
            "asyncio.gather() or complex threading logic; the DAG dataflow is perfectly maintained by passing "
            "the correct variables via upstream_outputs without needing concurrent execution logic. "
            "- Mandatory Final State Assignment (Critical — Single-Source Rule): "
            'forward_async must contain **exactly ONE** assignment to state["final_output"], and it must be '
            'of the form ``state["final_output"] = state["out_<FINAL_AGENT_ROLE_NAME>"]`` — a direct reference '
            "to the output of the last/topological-sink sub-agent stored in the state dict. "
            "**This rule has three strict sub-rules:** "
            "(1) **No hardcoded string literals**: do NOT assign a fixed string like ``\"UNANSWERABLE\"`` or "
            '``\"EVIDENCE_INSUFFICIENT\"`` to ``state["final_output"]``. The final answer must always come '
            "from an agent execution, never from a hardcoded fallback string. If retrieval or analysis fails, "
            "the final agent itself must still produce its best-effort answer — a hardcoded ``UNANSWERABLE`` "
            "string completely bypasses the agent and guarantees a zero score. "
            "(2) **No f-string concatenation or intermediate variable**: do NOT use ``f\"Task summary: {...}\"`` "
            "or ``final_summary = ...; state[\"final_output\"] = final_summary`` patterns. The final_output "
            "must be exactly ``state[\"out_<ROLE>\"]`` so the runtime can reliably extract the agent's direct "
            "output without parsing wrapper text, JSON envelopes, or concatenated summaries. "
            "(3) **No duplicate assignments**: do NOT write two ``state[\"final_output\"] = ...`` lines (e.g. "
            "one fallback and one real assignment). Python executes sequentially; the last line wins, making "
            "the intent ambiguous and the code fragile. Write exactly one assignment. "
            'Additionally, assign total token usage to state["usage_final"] (typically '
            'state["usage_<FINAL_AGENT_ROLE_NAME>"]). The system relies on these exact keys to evaluate '
            "the success of the MAS. "
            "- Strict Bounding for Loops and Ensembles (Crucial): if you design advanced topologies like "
            "iterative correction loops (e.g., while not approved:) or ensemble generation (calling the same "
            "agent multiple times), or other advanced topology, you MUST hardcode a strict upper bound limit. "
            "Any loop must be explicitly capped at a maximum of 3 iterations (e.g., using a counter "
            "if attempts >= 3: break or for _ in range(3):). Infinite loops are strictly prohibited and will "
            "cause the system to crash. "
            "- Safe State Dictionary Access and Loop Variable Handling: when populating upstream_outputs or "
            "updating variables within a loop, values must only reference keys in the state dictionary that "
            "have ALREADY been executed; inside loops, ensure you correctly overwrite or append to state keys "
            '(e.g., state["current_code_draft"]) so the next iteration receives the updated data, and this '
            "requirment is the same for ensemble topology or other advanced topologies.\n"
            "- Output Fallback Logic: if a retrieval or tool-based agent produces an empty, null, or otherwise "
            "unusable output, the downstream agent should NOT propagate that null downstream. Instead, design "
            "the code to detect empty upstream outputs and either (a) re-execute the upstream agent with a "
            "modified prompt, or (b) provide a default fallback value with a clear marker. This prevents "
            "cascading null/empty failures through the pipeline.\n"
            "- Final Agent Output Guarantee: the last agent in the workflow MUST produce the complete, "
            "final answer in the exact format required by the benchmark (see dataset_final_requirements in "
            "Stage 1). Do NOT let the final agent produce a summary, meta-commentary, or anything other than "
            "the direct answer. If the upstream data is insufficient, the final agent should still attempt to "
            "produce its best answer based on available information rather than outputting null or empty.\n\n"
        )
    raise ValueError(f"Invalid stage: {stage}")


def _mas_build_contract_prefix(stage: int) -> str:
    """
    MAS-pipeline-only instructions prepended to each stage: general + stage-specific blocks.

    Keep ``SKILL.md`` generic; put JSON/codegen contracts here.
    """
    if stage not in (1, 2, 3):
        raise ValueError(f"Invalid stage: {stage}")
    return (
        f"[MAS_BUILD_CONTRACT — Stage {stage}]\n\n"
        f"{_mas_build_contract_general_constraints()}"
        f"{_mas_build_contract_specific_constraints(stage)}"
    )


def _stage_prompt(
    stage_skill: str,
    stage_input: dict[str, Any],
    output_schema: str,
    *,
    stage_index: int | None = None,
) -> str:
    """
    Minimal prompt contract:
    - optional MAS-build prefix: ``[GENERAL_CONSTRAINTS]`` + ``[STAGE_SPECIFIC_CONSTRAINTS]``
      (see ``_mas_build_contract_prefix``)
    - stage skill text (from SKILL.md sections ## 1. / ## 2. / ## 3.)
    - stage input json
    - required output schema
    """
    prefix = _mas_build_contract_prefix(stage_index) if stage_index is not None else ""
    body = (
        f"{stage_skill}\n\n"
        f"[STAGE_INPUT_JSON]\n{json.dumps(stage_input, ensure_ascii=False, indent=2)}\n\n"
        f"[OUTPUT_JSON_SCHEMA]\n{output_schema}\n"
    )
    return f"{prefix}{body}"


def _normalize_sub_tasks(payload: dict[str, Any]) -> list[SubTaskSpec]:
    raw = payload.get("sub_agents")
    # Some models return one agent dict instead of `"sub_agents": [ {...}, ... ]`, or omit `sub_agents`
    # and flatten `role_*` onto the root object.
    if isinstance(raw, dict):
        raw = [raw]
    elif not isinstance(raw, list):
        rn = str(payload.get("role_name", "")).strip()
        ri = str(payload.get("role_instruction", "")).strip()
        if rn and ri:
            raw = [
                {
                    "role_name": payload.get("role_name"),
                    "role_instruction": payload.get("role_instruction"),
                    "user_prompt": payload.get("user_prompt", ""),
                    "tool_context": payload.get("tool_context"),
                }
            ]
        else:
            return []
    out: list[SubTaskSpec] = []
    for raw_item in raw:
        if not isinstance(raw_item, dict):
            continue
        item = _unwrap_stage2_agent_item(raw_item)
        role_name = str(item.get("role_name", "")).strip()
        role_instruction = str(item.get("role_instruction", "")).strip()
        if not role_name or not role_instruction:
            continue
        user_prompt = str(item.get("user_prompt", "")).strip()
        tool_context = item.get("tool_context")
        if tool_context is not None and not isinstance(tool_context, dict):
            tool_context = None
        out.append(
            SubTaskSpec(
                role_name=role_name,
                role_instruction=role_instruction,
                user_prompt=user_prompt,
                tool_context=tool_context,
            )
        )
    return out


def _fallback_sub_agents_from_stage1(
    stage1_payload: dict[str, Any],
    *,
    dataset_name: str,
) -> list[SubTaskSpec]:
    """
    Deterministic fallback when Stage 2 output is malformed/empty.
    Uses Stage 1 decomposition to build a minimal executable agent list.
    """
    raw = stage1_payload.get("sub_tasks")
    items: list[dict[str, Any]] = []
    if isinstance(raw, list):
        items = [x for x in raw if isinstance(x, dict)]
    elif isinstance(raw, dict):
        items = [raw]
    if not items:
        return []

    out: list[SubTaskSpec] = []
    for idx, item in enumerate(items, start=1):
        raw_name = str(item.get("name") or "").strip().lower()
        role_name = re.sub(r"[^a-z0-9_]+", "_", raw_name).strip("_") or f"stage1_node_{idx}"
        description = str(item.get("description") or "").strip()
        role_instruction = (
            f"You are the '{role_name}' sub-agent in a generated MAS workflow. "
            "Complete ONLY this sub-task and provide concise, actionable output for downstream nodes.\n\n"
            f"Sub-task description: {description or 'Follow Stage-1 decomposition intent.'}\n\n"
            "Hard constraints:\n"
            "- Do not output null/empty placeholders.\n"
            "- Keep output deterministic and directly usable by later agents.\n"
            "- Follow the required final format only if you are the final node."
        )
        out.append(
            SubTaskSpec(
                role_name=role_name,
                role_instruction=role_instruction,
                user_prompt=_RUNTIME_USER_PROMPT_FALLBACK,
                tool_context=_sanitize_tool_context({"execution_mode": "llm_only"}, dataset_name=dataset_name),
            )
        )
    return out


def _safe_identifier(raw: str, *, prefix: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw or "")).strip("_")
    if not s:
        s = prefix
    if s[0].isdigit():
        s = f"{prefix}_{s}"
    return s


def _is_mas_code_compilable(mas_code: str) -> tuple[bool, str]:
    try:
        ast.parse(mas_code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:  # pragma: no cover - defensive
        return False, f"{type(e).__name__}: {e}"


def _build_deterministic_sequential_workflow_code(
    *,
    class_name: str,
    subs: list[SubTaskSpec],
) -> str:
    """
    Emit a guaranteed-compilable sequential workflow from normalized sub-agents.
    This avoids hard failures when free-form Stage 3 code generation is unstable.
    """
    if not subs:
        raise RuntimeError("cannot build deterministic workflow without sub-agents")

    lines: list[str] = []
    lines.append(f"class {class_name}:")
    lines.append("    def __init__(")
    lines.append("        self,")
    lines.append("        *,")
    lines.append('        text_call_fn: Callable[[str], tuple[str, dict[str, Any]]],')
    lines.append('        tool_call_fn: Callable[[SubAgentRequest], tuple[str, dict[str, Any]]] | None = None,')
    lines.append("    ):")
    lines.append("        self.text_call_fn = text_call_fn")
    lines.append("        self.tool_call_fn = tool_call_fn")
    lines.append("")

    role_to_attr: list[tuple[str, str]] = []
    for i, spec in enumerate(subs, start=1):
        attr = _safe_identifier(spec.role_name, prefix=f"agent_{i}")
        role_key = _safe_identifier(spec.role_name, prefix=f"node_{i}").lower()
        role_to_attr.append((role_key, attr))
        lines.append(f"        self.{attr} = SubAgentRequest(")
        lines.append(f"            role_name={spec.role_name!r},")
        lines.append(f"            role_instruction={spec.role_instruction!r},")
        lines.append(f"            user_prompt={spec.user_prompt!r},")
        lines.append(f"            tool_context={dict(spec.tool_context or {})!r},")
        lines.append("        )")
        lines.append("")

    lines.append("    async def forward_async(self, task_info: str) -> dict[str, Any]:")
    lines.append('        state: dict[str, Any] = {"task_info": task_info}')
    lines.append("")

    prev_role_key: str | None = None
    for idx, (role_key, attr) in enumerate(role_to_attr, start=1):
        out_key = f"out_{role_key}"
        usage_key = f"usage_{role_key}"
        lines.append(f"        out_{idx} = await self.{attr}.execute_async(")
        lines.append("            text_call_fn=self.text_call_fn,")
        lines.append("            tool_call_fn=self.tool_call_fn,")
        lines.append("            prompt_override=build_workflow_subagent_prompt(")
        lines.append(f"                role_instruction=self.{attr}.role_instruction,")
        lines.append(f"                base_user_prompt=self.{attr}.user_prompt,")
        lines.append('                task_info=state["task_info"],')
        if prev_role_key is None:
            lines.append("                upstream_outputs=None,")
        else:
            lines.append("                upstream_outputs={")
            lines.append(f'                    "{prev_role_key}_output": state["out_{prev_role_key}"],')
            lines.append("                },")
        lines.append("            ),")
        lines.append("        )")
        lines.append(f'        state["{out_key}"] = out_{idx}.text')
        lines.append(f'        state["{usage_key}"] = out_{idx}.usage')
        lines.append("")
        prev_role_key = role_key

    assert prev_role_key is not None
    lines.append(f'        state["final_output"] = state["out_{prev_role_key}"]')
    lines.append(f'        state["usage_final"] = state.get("usage_{prev_role_key}", {{}})')
    lines.append("        return state")
    return "\n".join(lines)


def _extract_user_prompt_from_stage2_raw(raw_text: str, role_name: str) -> str:
    if not raw_text.strip() or not role_name.strip():
        return ""
    role_re = re.escape(role_name)
    m = re.search(
        rf'"role_name"\s*:\s*"{role_re}".*?"user_prompt"\s*:\s*"((?:\\.|[^"\\])*)"',
        raw_text,
        flags=re.S,
    )
    if not m:
        return ""
    encoded = m.group(1)
    try:
        return json.loads(f'"{encoded}"')
    except Exception:
        return encoded.replace('\\"', '"').replace("\\n", "\n")


def _with_hard_instruction_constraints(
    role_instruction: str,
    *,
    is_final: bool,
    dataset_name: str,
    role_name: str = "",
) -> str:
    base = (role_instruction or "").strip()
    if is_final:
        # The final agent is responsible for producing the complete answer.
        # It should NOT be constrained to "keep output concise" or "only do this sub-task"
        # because its sub-task IS the final deliverable.
        rules = [
            "HARD CONSTRAINTS:",
            # "- You are the FINAL agent in this workflow. Your output IS the final answer delivered to the user.",
            # "- Produce a COMPLETE, fully-formed answer — do not output summaries, meta-commentary, or placeholders.",
            # "- Ensure your output strictly matches the format required by the task (e.g., a single answer string, or \\boxed{...} for math).",
            # "- If upstream information is incomplete or uncertain, still produce your best answer based on all available information.",
            # "- Do NOT output null, empty, or 'not found' messages — always provide a substantive answer.",
        ]
        if dataset_name == "bcp":
            rules.extend(
                [
                    "- For retrieval-grounded tasks, avoid deterministic refusal sentinel strings (e.g., NO_VALID_EVIDENCE_FOUND / UNABLE_TO_DETERMINE) unless the task explicitly requires those exact tokens.",
                    "- If evidence is partial, provide the best evidence-supported candidate answer rather than terminating with a failure token.",
                ]
            )
        if dataset_name == "hlemath":
            rules.extend(
                [
                    "- You are the FINAL HLEMATH agent. Re-derive the solution from GLOBAL TASK CONTEXT and UPSTREAM "
                    "DEPENDENCIES; do not merely paraphrase or trust upstream without recomputing critical steps.",
                    "- Independently recompute key equations and numeric values; if upstream candidates conflict, "
                    "resolve with explicit calculation and output exactly one final value.",
                    "- Do NOT give a brief or one-line response. Explain how upstream inputs help or mislead your "
                    "sub-task (what you use, what you reject, and why), then work through your sub-task in depth "
                    "with explicit steps, intermediate values, and checks—not a high-level summary.",
                    "- Your output before the final line must contain substantive step-by-step work for your "
                    "sub-task; never output only \\boxed{...} without derivation.",
                    "- Output plain text only (no JSON/YAML/dict wrappers, no markdown code fences).",
                    "- The last line must be exactly one \\boxed{...} answer.",
                    "- Do not rely on unstated assumptions; list unavoidable assumptions with ASSUMPTION: before the boxed line.",
                ]
            )
    else:
        # Intermediate agents should be concise and focused on their specific sub-task.
        rules = [
            "HARD CONSTRAINTS:",
            # "- You must complete ONLY this sub-task; do not solve downstream sub-tasks.",
            # "- Keep output concise and directly usable by the next sub-agent.",
            # "- Do not add unrelated analysis, checklists, or meta commentary.",
            # "- Always produce substantive, structured output — never output null, empty, or 'not found' messages.",
        ]
        if dataset_name == "hlemath":
            rules.extend(
                [
                    "- You are an INTERMEDIATE HLEMATH agent. Do NOT output the final \\boxed{...} answer.",
                    "- Do NOT give a brief, one-line, or placeholder response. You must thoroughly complete "
                    "your assigned sub-task, not defer the real work to downstream agents.",
                    "- First explain how GLOBAL TASK CONTEXT and UPSTREAM DEPENDENCIES help your sub-task: "
                    "which givens/constraints you adopt, which upstream claims you trust or challenge, and why.",
                    "- Then solve your sub-task in depth: explicit reasoning, key equations, computed intermediate "
                    "values, case splits if needed, and at least one consistency check—never only a conclusion.",
                    "- Hand off plain text only (no JSON/dict/list schemas, no markdown code fences).",
                    "- Do not introduce mathematical assumptions not grounded in TASK CONTEXT or UPSTREAM DATA.",
                    "- If an assumption is unavoidable, mark it with ASSUMPTION: and explain why.",
                ]
            )
            if _hlemath_role_looks_like_final(role_name):
                rules.append(
                    "- Despite the role name, you are NOT the final agent; leave final formatting and \\boxed{...} to the downstream final node."
                )
    return f"{base}\n\n" + "\n".join(rules)


def _choose_final_sub_agent_index_for_dataset(
    subs: list[SubTaskSpec],
    *,
    dataset_name: str,
) -> int:
    """
    Choose which sub-agent should receive FINAL-agent constraints.

    Important:
    - Keep default behavior (last sub-agent) for all non-Vita datasets.
    - For VitaBench only, prefer explicit aggregation/synthesis roles when present.
      This prevents accidentally forcing a branch executor (e.g. book/order node)
      to become the final answer node.
    """
    if not subs:
        return -1
    if dataset_name == "hlemath":
        for i, spec in enumerate(subs):
            rn = (spec.role_name or "").strip().lower()
            if any(
                tok in rn
                for tok in (
                    "final",
                    "format",
                    "boxed",
                    "answer",
                    "finalize",
                )
            ):
                return i
        return len(subs) - 1
    if dataset_name != "vita":
        return len(subs) - 1

    # VitaBench-specific heuristic: prefer explicit "final aggregation" style roles.
    role_name_priority_tokens = (
        "final",
        "report",
        "summary",
        "synth",
        "aggregate",
        "merge",
        "compilation",
        "finalize",
    )
    for i, spec in enumerate(subs):
        rn = (spec.role_name or "").strip().lower()
        if any(tok in rn for tok in role_name_priority_tokens):
            return i

    # Fallback: preserve previous behavior.
    return len(subs) - 1


async def _call_stage_json(
    llm_call_fn: Callable[[str], Awaitable[str]],
    prompt: str,
    *,
    stage_name: str,
) -> dict[str, Any]:
    obj, _raw, _elapsed = await _call_stage_json_with_meta(
        llm_call_fn,
        prompt,
        stage_name=stage_name,
    )
    return obj


async def _call_stage_json_with_meta(
    llm_call_fn: Callable[[str], Awaitable[str]],
    prompt: str,
    *,
    stage_name: str,
) -> tuple[dict[str, Any], str, float]:
    def _is_structurally_valid_for_stage(obj: dict[str, Any], hint: int | None) -> bool:
        if hint == 1:
            st = _coerce_stage1_subtasks(obj).get("sub_tasks")
            return isinstance(st, list) and len(st) > 0
        if hint == 2:
            return len(_normalize_sub_tasks(obj)) > 0
        if hint == 3:
            mc = obj.get("mas_code")
            return isinstance(mc, str) and bool(mc.strip())
        return True

    def _build_json_repair_prompt(
        *,
        stage_label: str,
        original_prompt: str,
        malformed_response: str,
    ) -> str:
        max_chars = 40000
        clipped = (malformed_response or "")[:max_chars]
        if len(malformed_response or "") > max_chars:
            clipped += "\n...[truncated]"
        return (
            "You are a strict JSON repair assistant for a MAS build stage.\n"
            "Rewrite the malformed assistant response into ONE valid JSON object only.\n"
            "Requirements:\n"
            "- Follow [OUTPUT_JSON_SCHEMA] from ORIGINAL_STAGE_PROMPT exactly.\n"
            "- Do not add markdown fences or explanations.\n"
            "- Keep original intent/content; only repair JSON structure/escaping.\n"
            "- Return a single top-level JSON object.\n\n"
            f"[STAGE]\n{stage_label}\n\n"
            f"[ORIGINAL_STAGE_PROMPT]\n{original_prompt}\n\n"
            f"[MALFORMED_ASSISTANT_RESPONSE]\n{clipped}\n\n"
            "Output JSON now."
        )

    print(f"\n[{stage_name}] prompt_chars={len(prompt)}", flush=True)
    t0 = time.perf_counter()
    raw = await llm_call_fn(prompt)
    elapsed = time.perf_counter() - t0
    print(f"[{stage_name}] llm_elapsed_sec={elapsed:.2f}", flush=True)
    preview = (raw or "").strip().replace("\n", " ")
    if len(preview) > 300:
        preview = preview[:300] + "...(truncated)"
    print(f"[{stage_name}] raw_response_preview={preview}", flush=True)
    hint = _stage_hint_from_label(stage_name)
    obj = _extract_best_json_dict_for_stage(raw, stage_hint=hint)
    if isinstance(obj, dict) and _is_structurally_valid_for_stage(obj, hint):
        print(f"[{stage_name}] parsed_keys={sorted(obj.keys())}", flush=True)
        return obj, raw, elapsed

    # One repair attempt for malformed / structurally invalid stage JSON.
    repair_prompt = _build_json_repair_prompt(
        stage_label=stage_name,
        original_prompt=prompt,
        malformed_response=raw,
    )
    print(f"[{stage_name}] attempting json_repair_retry=1", flush=True)
    t1 = time.perf_counter()
    repaired_raw = await llm_call_fn(repair_prompt)
    elapsed += time.perf_counter() - t1
    repaired_obj = _extract_best_json_dict_for_stage(repaired_raw, stage_hint=hint)
    if not isinstance(repaired_obj, dict):
        raise RuntimeError(f"{stage_name} returned no valid JSON object.")
    if not _is_structurally_valid_for_stage(repaired_obj, hint):
        raise RuntimeError(f"{stage_name} returned structurally invalid JSON payload.")
    print(f"[{stage_name}] parsed_keys={sorted(repaired_obj.keys())}", flush=True)
    print(f"[{stage_name}] json_repair_retry=1 succeeded", flush=True)
    return repaired_obj, repaired_raw, elapsed


async def run_three_stage_skill_build(
    *,
    task_text: str,
    init_skill_text: str,
    template_text: str,
    llm_call_fn: Callable[[str], Awaitable[str]],
    class_name: str,
    dataset_name: str = "unknown",
) -> str:
    """
    Execute exactly 3 stages from init_skill and return final MAS code.
    """
    artifacts = await run_three_stage_skill_build_with_trace(
        task_text=task_text,
        init_skill_text=init_skill_text,
        template_text=template_text,
        llm_call_fn=llm_call_fn,
        class_name=class_name,
        dataset_name=dataset_name,
    )
    return artifacts.mas_code


async def run_three_stage_skill_build_with_trace(
    *,
    task_text: str,
    init_skill_text: str,
    template_text: str,
    llm_call_fn: Callable[[str], Awaitable[str]],
    class_name: str,
    dataset_name: str = "unknown",
) -> ThreeStageBuildArtifacts:
    """Execute three-stage build and return full stage traces and mas_code."""
    stages = _extract_skill_stages(init_skill_text)
    stage_traces: list[BuildStageTrace] = []

    normalized_dataset = _normalize_dataset_name(dataset_name)
    dataset_final_requirements = _dataset_final_requirement_text(normalized_dataset)

    s1_schema = (
        '{ "goal": "...", "sub_tasks": [{"name":"...", "description":"..."}], '
        '"constraints": ["..."], "dependencies": [{"from":"...","to":"...","type":"..."}], '
        '"success_criteria": ["..."], '
        '"reasoning": "detailed rationale: how you interpreted the task, why this decomposition/dependencies, and how ambiguity/assumption risks are controlled. '
        'IMPORTANT: sub_tasks descriptions must describe the CAPABILITY/PROCESS each node provides, NOT attempt '
        'to answer the user question or provide solution content." }'
    )
    s1_constraints: list[str] = []
    s1_extra_input: dict[str, Any] = {}
    if normalized_dataset == "hlemath":
        s1_constraints.extend(_hlemath_stage1_build_constraints())
        s1_extra_input["hlemath_topology_templates"] = _hlemath_topology_templates_block()
    s1_prompt = _stage_prompt(
        stages[1],
        {
            "task_text": task_text,
            "dataset": normalized_dataset,
            "dataset_final_requirements": dataset_final_requirements,
            "constraints": s1_constraints,
            **s1_extra_input,
        },
        s1_schema,
        stage_index=1,
    )
    s1, s1_raw, s1_elapsed = await _call_stage_json_with_meta(llm_call_fn, s1_prompt, stage_name="Stage 1")
    s1 = _coerce_stage1_subtasks(s1)
    if normalized_dataset == "hlemath":
        s1_ok, s1_issues = _hlemath_validate_stage1_decomposition(s1)
        if not s1_ok:
            print(
                f"[Stage 1][HLEMATH] decomposition issues ({len(s1_issues)}); attempting one structural repair",
                flush=True,
            )
            for issue in s1_issues:
                print(f"[Stage 1][HLEMATH]   - {issue}", flush=True)
            repair_prompt = _build_hlemath_stage1_repair_prompt(
                original_prompt=s1_prompt,
                stage1_json=s1,
                issues=s1_issues,
            )
            t_repair = time.perf_counter()
            repaired_raw = await llm_call_fn(repair_prompt)
            s1_elapsed += time.perf_counter() - t_repair
            repaired_obj = _extract_best_json_dict_for_stage(repaired_raw, stage_hint=1)
            if isinstance(repaired_obj, dict):
                repaired_obj = _coerce_stage1_subtasks(repaired_obj)
                repaired_ok, repaired_issues = _hlemath_validate_stage1_decomposition(repaired_obj)
                if repaired_ok:
                    s1 = repaired_obj
                    s1_raw = repaired_raw
                    print("[Stage 1][HLEMATH] structural repair succeeded", flush=True)
                else:
                    print(
                        "[Stage 1][HLEMATH] structural repair still has issues; keeping original Stage 1",
                        flush=True,
                    )
                    for issue in repaired_issues:
                        print(f"[Stage 1][HLEMATH]   - {issue}", flush=True)
            else:
                print("[Stage 1][HLEMATH] structural repair returned invalid JSON; keeping original Stage 1", flush=True)
    stage_traces.append(
        BuildStageTrace(
            stage=1,
            stage_name="Stage 1",
            prompt=s1_prompt,
            raw_response=s1_raw,
            parsed_json=s1,
            elapsed_sec=s1_elapsed,
        )
    )
    s1_subs = s1.get("sub_tasks")
    s1_sub_count = len(s1_subs) if isinstance(s1_subs, list) else 0
    print(f"[Stage 1] sub_tasks_count={s1_sub_count}", flush=True)

    allowed_modes = sorted(_allowed_tool_modes_for_dataset(normalized_dataset))
    s2_constraints = [
        f"Only choose execution_mode from {allowed_modes} for this dataset.",
        "Set tool_context.execution_mode explicitly for every sub-agent.",
    ]
    if normalized_dataset == "vita":
        s2_constraints.append(
            'For VitaBench, every sub-agent must use {"execution_mode":"vita_tool"} only. '
            "Lookup-only or reasoning-only steps still run in the Vita simulator with environment tools; never use llm_only."
        )
        s2_constraints.append(
            "For VitaBench multi-chain plans, explicitly include a final aggregation/synthesis role "
            "(e.g. final_report/final_summary/final_aggregator) that combines all required transaction outcomes. "
            "Do not end the workflow on a single branch action node (like only book/order/verify for one branch)."
        )
    elif normalized_dataset == "bcp":
        s2_constraints.append(
            'For BrowseComp (bcp), every sub-agent must use tool_context {"execution_mode":"multi_turn_search"}. '
            "Do not use llm_only for any node, including planning/validation/final synthesis nodes."
        )
    elif normalized_dataset == "hlemath":
        s2_constraints.append(_hlemath_stage2_build_constraints())
    else:
        s2_constraints.append(
            'For sub-tasks that do not use tools, set tool_context to {"execution_mode":"llm_only"}; do not use null for tool_context.'
        )
    s2_constraints.append(
        "The dataset_final_requirements below defines what the benchmark evaluator expects as the final deliverable. "
        "Ensure the LAST agent (the one that produces final_output) generates output in exactly that format. "
        "Intermediate agents should produce structured outputs that directly support the final agent in meeting this requirement."
    )
    s2_schema = (
        '{ "sub_agents": ['
        '{"role_name":"...", "role_instruction":"...", "user_prompt":"...", '
        '"tool_context": {"execution_mode":"llm_only|multi_turn_search|vita_tool", "...":"..."} }'
        '], "reasoning": "brief rationale: role split, ordering, and execution_mode choices vs Stage 1. '
        'IMPORTANT: role_instruction defines the agent\'s role and behavioral boundaries; user_prompt provides '
        'the concrete task input. Neither should contain solution content or attempt to answer the question." }'
    )
    s2_prompt = _stage_prompt(
        stages[2],
        {
            "task_text": task_text,
            "stage1": s1,
            "dataset": normalized_dataset,
            "dataset_final_requirements": dataset_final_requirements,
            "tool_context_execution_modes": list(TOOL_EXECUTION_MODES),
            "allowed_modes_for_dataset": allowed_modes,
            "constraints": s2_constraints,
        },
        s2_schema,
        stage_index=2,
    )
    s2, s2_raw, s2_elapsed = await _call_stage_json_with_meta(llm_call_fn, s2_prompt, stage_name="Stage 2")
    stage_traces.append(
        BuildStageTrace(
            stage=2,
            stage_name="Stage 2",
            prompt=s2_prompt,
            raw_response=s2_raw,
            parsed_json=s2,
            elapsed_sec=s2_elapsed,
        )
    )
    subs = _normalize_sub_tasks(s2)
    if not subs:
        print("[Stage 2] fallback: deriving sub_agents from Stage 1 decomposition", flush=True)
        subs = _fallback_sub_agents_from_stage1(s1, dataset_name=normalized_dataset)
    if not subs:
        raise RuntimeError("Stage 2 returned invalid/empty sub_agents.")
    final_sub_agent_idx = _choose_final_sub_agent_index_for_dataset(
        subs,
        dataset_name=normalized_dataset,
    )
    if normalized_dataset == "hlemath":
        s2_ok, s2_issues = _hlemath_validate_stage2_sub_agents(
            subs,
            final_idx=final_sub_agent_idx,
        )
        if not s2_ok:
            print(
                f"[Stage 2][HLEMATH] agent-protocol warnings ({len(s2_issues)}); applying hard constraints",
                flush=True,
            )
            for issue in s2_issues:
                print(f"[Stage 2][HLEMATH]   - {issue}", flush=True)
    constrained_subs: list[SubTaskSpec] = []
    for idx, spec in enumerate(subs):
        raw_user_prompt = spec.user_prompt or _extract_user_prompt_from_stage2_raw(s2_raw, spec.role_name)
        # Keep user_prompt short and embed-safe: full task is always in GLOBAL TASK CONTEXT at runtime
        # (see template.sub_agent.build_workflow_subagent_prompt). Never substitute raw task_text here —
        # pasting LaTeX / JSON / quotes into SubAgentRequest literals breaks Stage-3 Python codegen.
        if not raw_user_prompt or len(raw_user_prompt.strip()) < 10:
            raw_user_prompt = _RUNTIME_USER_PROMPT_FALLBACK
        is_final_agent = (idx == final_sub_agent_idx)
        constrained_subs.append(
            SubTaskSpec(
                role_name=spec.role_name,
                role_instruction=_with_hard_instruction_constraints(
                    spec.role_instruction,
                    is_final=is_final_agent,
                    dataset_name=normalized_dataset,
                    role_name=spec.role_name,
                ),
                user_prompt=raw_user_prompt,
                tool_context=_sanitize_tool_context(spec.tool_context, dataset_name=normalized_dataset),
            )
        )
    subs = constrained_subs
    print(f"[Stage 2] sub_agents_count={len(subs)} roles={[s.role_name for s in subs]}", flush=True)
    preferred_final_role_name = None
    if 0 <= final_sub_agent_idx < len(subs):
        preferred_final_role_name = subs[final_sub_agent_idx].role_name

    s3_schema = (
        '{ "mas_code": "full python code string", '
        '"reasoning": "brief rationale: topology, dataflow, and how you kept SubAgentRequest string literals '
        "PYTHON-SAFE (delimiters/escaping, no duplicate imports) before emitting mas_code\" }"
    )
    normalized_sub_agents = [s.__dict__ for s in subs]
    s3_input = {
        "task_text": task_text,
        "stage1": s1,
        "stage2": {"sub_agents": normalized_sub_agents},
        "target_class_name": class_name,
        "template_text": template_text,
        "dataset": normalized_dataset,
        "dataset_final_requirements": dataset_final_requirements,
    }
    s3_prompt = _stage_prompt(stages[3], s3_input, s3_schema, stage_index=3)
    s3, s3_raw, s3_elapsed = await _call_stage_json_with_meta(llm_call_fn, s3_prompt, stage_name="Stage 3")
    stage_traces.append(
        BuildStageTrace(
            stage=3,
            stage_name="Stage 3",
            prompt=s3_prompt,
            raw_response=s3_raw,
            parsed_json=s3,
            elapsed_sec=s3_elapsed,
        )
    )
    mas_code = s3.get("mas_code")
    if not isinstance(mas_code, str) or not mas_code.strip():
        raise RuntimeError("Stage 3 returned empty mas_code.")
    mas_code = mas_code.strip()
    mas_code = _strip_leading_import_noise(mas_code)
    mas_code = _enforce_dataset_tool_modes_in_mas_code(mas_code, dataset_name=normalized_dataset)
    # P0: sanitize final_output assignments — remove UNANSWERABLE/hardcoded fallbacks,
    # f-string concatenations, and ensure a single agent-reference assignment exists.
    sub_agent_role_names = [s.role_name for s in subs]
    mas_code = _sanitize_final_output_in_mas_code(
        mas_code,
        sub_agent_role_names=sub_agent_role_names,
        preferred_final_role_name=preferred_final_role_name,
    )
    if normalized_dataset == "vita":
        mas_code = _enforce_final_output_role_in_mas_code(
            mas_code,
            final_role_name=preferred_final_role_name,
        )
    if normalized_dataset == "hlemath":
        mas_code = _enforce_final_output_role_in_mas_code(
            mas_code,
            final_role_name=preferred_final_role_name,
        )
    mas_code = prepend_fixed_imports(mas_code)
    compilable, compile_err = _is_mas_code_compilable(mas_code)
    use_deterministic_fallback = not compilable
    if compilable and normalized_dataset == "hlemath":
        behavior_ok, behavior_issues = _hlemath_validate_mas_code_behavior(
            mas_code,
            sub_agent_role_names=sub_agent_role_names,
            preferred_final_role_name=preferred_final_role_name,
        )
        if not behavior_ok:
            use_deterministic_fallback = True
            print(
                f"[Stage 3][HLEMATH] mas_code behavioral check failed ({len(behavior_issues)}); "
                "switching to deterministic sequential workflow code.",
                flush=True,
            )
            for issue in behavior_issues:
                print(f"[Stage 3][HLEMATH]   - {issue}", flush=True)
    if use_deterministic_fallback:
        if not compilable:
            print(
                f"[Stage 3] fallback: generated mas_code not compilable ({compile_err}); "
                "switching to deterministic sequential workflow code.",
                flush=True,
            )
        mas_code = prepend_fixed_imports(
            _build_deterministic_sequential_workflow_code(class_name=class_name, subs=subs)
        )
    print(f"[Stage 3] mas_code_chars={len(mas_code)}", flush=True)
    return ThreeStageBuildArtifacts(
        mas_code=mas_code,
        stage_traces=stage_traces,
        stage1=s1,
        stage2=s2,
        stage3=s3,
        normalized_sub_agents=normalized_sub_agents,
    )


async def run_generated_workflow(
    *,
    mas_code: str,
    class_name: str,
    task_text: str,
    text_call_fn: AsyncTextCallFn | SyncTextCallFn,
    tool_call_fn: AsyncToolCallFn | SyncToolCallFn | None = None,
) -> dict[str, Any]:
    """
    Exec generated MAS code and run forward().
    Returns workflow state dict.
    """
    # Ensure generated code can import template modules regardless of launch path.
    for p in (_PACKAGE_ROOT, _SKILL_MAS_PKG_ROOT):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


    ns: dict[str, Any] = {}
    exec(mas_code, ns, ns)
    if class_name not in ns:
        raise RuntimeError(f"Generated code did not define class {class_name}.")
    cls = ns[class_name]
    workflow = cls(text_call_fn=text_call_fn, tool_call_fn=tool_call_fn)
    out = workflow.forward_async(task_text)
    if asyncio.iscoroutine(out):
        out = await out
    if not isinstance(out, dict):
        raise RuntimeError("Workflow forward() must return a dict state.")
    return out


def _extract_predicted_answer(text: str) -> str:
    s = (text or "").strip()
    m = re.search(r"\\boxed\{([^}]+)\}", s)
    if m:
        return m.group(1).strip()
    m2 = re.findall(r"-?\d+(?:\.\d+)?", s)
    if m2:
        return m2[-1].strip()
    return s


async def generate_mas_code_for_task(
    *,
    task_text: str,
    class_name: str = "GeneratedMASWorkflow",
    init_skill_path: str | Path | None = None,
    planner_call_fn: Callable[[str], Awaitable[str]] | None = None,
    dataset_name: str = "unknown",
) -> str:
    """Generate MAS code by running init_skill stage-1/2/3 sequentially."""
    if planner_call_fn is None:
        raise RuntimeError("planner_call_fn is required for three-stage build.")
    template_text = stage3_llm_reference_bundle()
    _skill_text = ""
    if init_skill_path is not None and Path(init_skill_path).is_file():
        _skill_text = _load_text(init_skill_path)
    if not _skill_text:
        raise RuntimeError(f"init skill file not found or empty: {init_skill_path}")
    return await run_three_stage_skill_build(
        task_text=task_text,
        init_skill_text=_skill_text,
        template_text=template_text,
        llm_call_fn=planner_call_fn,
        class_name=class_name,
        dataset_name=dataset_name,
    )


async def generate_mas_artifacts_for_task(
    *,
    task_text: str,
    class_name: str = "GeneratedMASWorkflow",
    init_skill_path: str | Path | None = None,
    planner_call_fn: Callable[[str], Awaitable[str]] | None = None,
    dataset_name: str = "unknown",
) -> ThreeStageBuildArtifacts:
    """Generate MAS code with full stage traces for logging/debugging."""
    if planner_call_fn is None:
        raise RuntimeError("planner_call_fn is required for three-stage build.")
    template_text = stage3_llm_reference_bundle()
    _skill_text = ""
    if init_skill_path is not None and Path(init_skill_path).is_file():
        _skill_text = _load_text(init_skill_path)
    if not _skill_text:
        raise RuntimeError(f"init skill file not found or empty: {init_skill_path}")
    return await run_three_stage_skill_build_with_trace(
        task_text=task_text,
        init_skill_text=_skill_text,
        template_text=template_text,
        llm_call_fn=planner_call_fn,
        class_name=class_name,
        dataset_name=dataset_name,
    )


async def generate_mas_artifacts_with_retries(
    *,
    task_text: str,
    class_name: str = "GeneratedMASWorkflow",
    init_skill_path: str | Path | None = None,
    planner_call_fn: Callable[[str], Awaitable[str]] | None = None,
    max_attempts: int = 5,
    dataset_name: str = "unknown",
) -> tuple[ThreeStageBuildArtifacts | None, int, list[dict[str, Any]]]:
    """Generate MAS artifacts with retry budget for stage/build failures."""
    attempts = max(1, int(max_attempts))
    events: list[dict[str, Any]] = []
    last_error = ""
    for i in range(1, attempts + 1):
        try:
            artifacts = await generate_mas_artifacts_for_task(
                task_text=task_text,
                class_name=class_name,
                init_skill_path=init_skill_path,
                planner_call_fn=planner_call_fn,
                dataset_name=dataset_name,
            )
            if i > 1:
                events.append(
                    {
                        "type": "generation_retry_success",
                        "attempt": i,
                        "message": "generation succeeded after retries",
                    }
                )
            return artifacts, i, events
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            events.append(
                {
                    "type": "generation_retry_error",
                    "attempt": i,
                    "message": last_error,
                }
            )
            print(f"[MAS generation retry {i}/{attempts}] {last_error}", flush=True)
    events.append(
        {
            "type": "generation_retry_exhausted",
            "attempt": attempts,
            "message": last_error or "unknown generation failure",
        }
    )
    return None, attempts, events


async def _repair_mas_code_once(
    *,
    broken_code: str,
    class_name: str,
    task_text: str,
    error_message: str,
    planner_call_fn: Callable[[str], Awaitable[str]],
) -> str:
    repair_prompt = (
        "You are a Python code fixer for a generated MAS workflow. "
        "Return JSON only with key 'mas_code'.\n\n"
        f"[TASK]\n{task_text}\n\n"
        f"[TARGET_CLASS_NAME]\n{class_name}\n\n"
        f"[ERROR]\n{error_message}\n\n"
        "[BROKEN_MAS_CODE]\n"
        f"{broken_code}\n\n"
        "Constraints:\n"
        "- Keep the same class name.\n"
        "- Keep async workflow executable.\n"
        "- forward_async must return a dict and include final_output.\n"
        "- Do NOT include from __future__, from typing, or from template.sub_agent imports "
        "(the runner prepends them).\n"
        "- SubAgentRequest string fields must use Python-safe quoting (prefer triple-single quotes '''...''').\n"
        "- Return valid Python code only in mas_code.\n"
        '- Output JSON format: {"mas_code":"..."}\n'
    )
    raw = await planner_call_fn(repair_prompt)
    obj = _extract_best_json_dict_for_stage(raw, stage_hint=3)
    if not isinstance(obj, dict):
        raise RuntimeError("repair step returned non-JSON content")
    fixed = obj.get("mas_code")
    if not isinstance(fixed, str) or not fixed.strip():
        raise RuntimeError("repair step returned empty mas_code")
    return fixed


async def run_generated_workflow_with_retries(
    *,
    mas_code: str,
    class_name: str,
    task_text: str,
    text_call_fn: AsyncTextCallFn | SyncTextCallFn,
    planner_call_fn: Callable[[str], Awaitable[str]],
    tool_call_fn: AsyncToolCallFn | SyncToolCallFn | None = None,
    max_attempts: int = 3,
    dataset_name: str = "unknown",
) -> tuple[bool, dict[str, Any], str, int, list[dict[str, Any]], str | None]:
    """
    Execute generated MAS with retry budget.
    On execution failure, tries to repair mas_code and rerun.
    """
    attempts = max(1, int(max_attempts))
    code = _enforce_dataset_tool_modes_in_mas_code(mas_code.strip(), dataset_name=dataset_name)
    events: list[dict[str, Any]] = []
    last_error: str | None = None
    for i in range(1, attempts + 1):
        try:
            state = await run_generated_workflow(
                mas_code=code,
                class_name=class_name,
                task_text=task_text,
                text_call_fn=text_call_fn,
                tool_call_fn=tool_call_fn,
            )
            if i > 1:
                events.append(
                    {
                        "type": "execution_retry_success",
                        "attempt": i,
                        "message": "execution succeeded after code repair retries",
                    }
                )
            return True, state, code, i, events, None
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            events.append(
                {
                    "type": "execution_retry_error",
                    "attempt": i,
                    "message": last_error,
                }
            )
            print(f"[MAS execution retry {i}/{attempts}] {last_error}", flush=True)
            if i >= attempts:
                break
            try:
                code = await _repair_mas_code_once(
                    broken_code=code,
                    class_name=class_name,
                    task_text=task_text,
                    error_message=last_error,
                    planner_call_fn=planner_call_fn,
                )
                code = _enforce_dataset_tool_modes_in_mas_code(code.strip(), dataset_name=dataset_name)
                code = _strip_leading_import_noise(code)
                code = prepend_fixed_imports(code)
                events.append(
                    {
                        "type": "execution_repair_generated",
                        "attempt": i,
                        "message": "generated repaired MAS code",
                    }
                )
            except Exception as repair_err:
                repair_msg = f"{type(repair_err).__name__}: {repair_err}"
                events.append(
                    {
                        "type": "execution_repair_error",
                        "attempt": i,
                        "message": repair_msg,
                    }
                )
                print(f"[MAS repair retry {i}/{attempts}] {repair_msg}", flush=True)
    events.append(
        {
            "type": "execution_retry_exhausted",
            "attempt": attempts,
            "message": last_error or "unknown execution failure",
        }
    )
    return False, {}, code, attempts, events, last_error


async def run_mas_pipeline_with_retries(
    *,
    task_text: str,
    class_name: str,
    init_skill_path: str | Path,
    planner_call_fn: Callable[[str], Awaitable[str]],
    text_call_fn: AsyncTextCallFn | SyncTextCallFn,
    tool_call_fn: AsyncToolCallFn | SyncToolCallFn | None = None,
    max_generation_attempts: int = 5,
    max_execution_attempts: int = 3,
    dataset_name: str = "unknown",
) -> MASRunWithRetryResult:
    """
    End-to-end MAS build + execution with retry policies.
    Never raises for expected build/execution failures.
    """
    retry_events: list[dict[str, Any]] = []
    artifacts, gen_attempts, gen_events = await generate_mas_artifacts_with_retries(
        task_text=task_text,
        class_name=class_name,
        init_skill_path=init_skill_path,
        planner_call_fn=planner_call_fn,
        max_attempts=max_generation_attempts,
        dataset_name=dataset_name,
    )
    retry_events.extend(gen_events)
    if artifacts is None:
        reason = ""
        for ev in reversed(retry_events):
            if ev.get("type") in {"generation_retry_error", "generation_retry_exhausted"}:
                reason = str(ev.get("message") or "")
                break
        return MASRunWithRetryResult(
            success=False,
            final_output="",
            state={},
            mas_code="",
            artifacts=None,
            generation_attempts_used=gen_attempts,
            execution_attempts_used=0,
            failure_stage="generation",
            failure_reason=reason or "generation failed after retries",
            retry_events=retry_events,
        )

    ok, state, final_code, exec_attempts, exec_events, exec_err = await run_generated_workflow_with_retries(
        mas_code=artifacts.mas_code,
        class_name=class_name,
        task_text=task_text,
        text_call_fn=text_call_fn,
        planner_call_fn=planner_call_fn,
        tool_call_fn=tool_call_fn,
        max_attempts=max_execution_attempts,
        dataset_name=dataset_name,
    )
    retry_events.extend(exec_events)
    final_output = ""
    if isinstance(state, dict):
        final_output = str(state.get("final_output", "") or "")
    return MASRunWithRetryResult(
        success=ok,
        final_output=final_output,
        state=state if isinstance(state, dict) else {},
        mas_code=final_code,
        artifacts=artifacts,
        generation_attempts_used=gen_attempts,
        execution_attempts_used=exec_attempts,
        failure_stage=None if ok else "execution",
        failure_reason=None if ok else (exec_err or "execution failed after retries"),
        retry_events=retry_events,
    )




async def _main() -> None:
    # Default quick test: use the first HLEMath sample.
    from Skill_MAS.utils.paths import HLEMATH_ROOT, INIT_SKILL_DIR

    hlemath_jsonl = HLEMATH_ROOT / "data" / "hlemath_test.jsonl"
    init_skill = INIT_SKILL_DIR / "SKILL.md"
    model = "qwen3.5-plus"

    with hlemath_jsonl.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    if not first:
        raise RuntimeError(f"No sample found in {hlemath_jsonl}")
    row = json.loads(first)
    task_text = str(row.get("question", "")).strip()
    if not task_text:
        raise RuntimeError("First sample has empty question")

    from openai_async_client import AsyncOpenAIClient

    client = AsyncOpenAIClient(model=model)
    planner_call_fn = make_planner_call_fn(client)

    class_name = f"HLEMathTask{row.get('id', '0')}_MASWorkflow"
    code = await generate_mas_code_for_task(
        task_text=task_text,
        class_name=class_name,
        init_skill_path=init_skill,
        planner_call_fn=planner_call_fn,
    )
    print(code)

    text_call_fn = make_text_call_fn(client)
    state = await run_generated_workflow(
        mas_code=code,
        class_name=class_name,
        task_text=task_text,
        text_call_fn=text_call_fn,
        tool_call_fn=None,
    )
    final_output = str(state.get("final_output", ""))
    pred = _extract_predicted_answer(final_output)
    gold = str(row.get("answer", "")).strip()
    print("\n===== WORKFLOW RESULT =====")
    print(f"final_output: {final_output}")
    print(f"predicted_answer: {pred}")
    print(f"gold_answer: {gold}")
    print(f"match: {pred == gold}")


if __name__ == "__main__":
    asyncio.run(_main())

