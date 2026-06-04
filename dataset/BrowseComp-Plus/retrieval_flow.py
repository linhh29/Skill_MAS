"""Shared multi-round BM25 retrieval with LLM-guided query refinement.

Avoids repeating near-duplicate queries across rounds by tracking normalized queries,
similarity checks, and cumulative evidence summaries for the planner. The returned
snippet list is from the **last BM25 round only** (see ``multi_round_retrieve_contexts``).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from retrieval import BM25Retriever

_MODEL_CONFIG_JSON: dict[str, Any] | None = None


def _skill_mas_model_config() -> dict[str, Any]:
    """Lazy-load ``<repo>/Skill_MAS/skill_mas/model_config.json`` for DashScope ``extra_body`` etc."""
    global _MODEL_CONFIG_JSON
    if _MODEL_CONFIG_JSON is not None:
        return _MODEL_CONFIG_JSON
    p = Path(__file__).resolve().parents[2] / "skill_mas" / "model_config.json"
    if not p.is_file():
        _MODEL_CONFIG_JSON = {}
        return _MODEL_CONFIG_JSON
    raw = json.loads(p.read_text(encoding="utf-8"))
    _MODEL_CONFIG_JSON = raw if isinstance(raw, dict) else {}
    return _MODEL_CONFIG_JSON


def _extra_body_for_model(model: str) -> dict[str, Any] | None:
    """Return OpenAI-SDK ``extra_body`` dict for ``model`` when defined in Skill_MAS JSON."""
    mc = _skill_mas_model_config()
    key = (model or "").strip()
    row: Any = None
    if key and key in mc and isinstance(mc[key], dict):
        row = mc[key]
    elif key:
        lower_map = {
            str(k).lower(): v
            for k, v in mc.items()
            if isinstance(v, dict) and not str(k).startswith("_")
        }
        row = lower_map.get(key.lower())
    if not isinstance(row, dict):
        return None
    eb = row.get("extra_body")
    return dict(eb) if isinstance(eb, dict) else None


def _tokenize(q: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", q or "") if len(t) > 2]


def _normalize_query(q: str) -> str:
    return " ".join(_tokenize(q))


def _jaccard_tokens(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def queries_too_similar(a: str, b: str, *, thresh: float = 0.72) -> bool:
    """Return True if two queries cover essentially the same keywords."""
    na, nb = _normalize_query(a), _normalize_query(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Short-query containment / prefix overlap often repeats 'townlet restored ...'
    if len(na) > 24 and len(nb) > 24:
        if na in nb or nb in na:
            return True
    return _jaccard_tokens(a, b) >= thresh


def dedupe_queries(queries: list[str], *, max_queries: int = 12) -> list[str]:
    """Keep order; drop empties and near-duplicates."""
    out: list[str] = []
    for q in queries:
        qq = str(q).strip()
        if not qq:
            continue
        if any(queries_too_similar(qq, prev) for prev in out):
            continue
        out.append(qq[:4096])
        if len(out) >= max_queries:
            break
    return out


def heuristic_fallback_queries(question: str, *, max_q: int = 5) -> list[str]:
    """Cheap BM25 seeds when delegation JSON omits retrieval_queries."""
    q = (question or "").strip()
    if not q:
        return []
    words = _tokenize(q)
    bag = " ".join(words[:22])
    chunks = [q[: min(len(q), 512)], bag]
    if len(q) > 512:
        chunks.append(q[512 : min(len(q), 1536)])
    return dedupe_queries([c for c in chunks if str(c).strip()], max_queries=max_q)


def gather_bm25_contexts_from_queries(
    *,
    retriever: BM25Retriever,
    queries: list[str],
    retrieval_topk: int,
    doc_max_tokens: int,
    max_snippets: int = 56,
    max_chars_total: int = 32000,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Execute BM25 for multiple queries with doc-level dedupe (Skill-MAS execution stage)."""
    seen_docids: set[str] = set()
    contexts: list[str] = []
    trajectory: list[dict[str, Any]] = []
    char_budget = int(max_chars_total)

    for query in queries:
        if not query.strip():
            continue
        docs = retriever.search(query, max(1, int(retrieval_topk)))
        round_docids: list[str] = []
        for d in docs:
            if d.docid in seen_docids:
                continue
            if len(contexts) >= max_snippets:
                break
            seen_docids.add(d.docid)
            block = f"[docid={d.docid}] {_truncate_text_by_tokens(d.text, int(doc_max_tokens))}"
            if char_budget <= 0:
                break
            if len(block) > char_budget:
                block = block[: char_budget]
            char_budget -= len(block)
            contexts.append(block)
            round_docids.append(d.docid)
        trajectory.append(
            {
                "query": query,
                "retrieved_docids": round_docids,
            }
        )
        if len(contexts) >= max_snippets or char_budget <= 0:
            break

    return contexts, trajectory, sorted(seen_docids)


def _truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    toks = (text or "").split()
    if max_tokens <= 0 or len(toks) <= max_tokens:
        return text or ""
    return " ".join(toks[:max_tokens])


def _brief_snippets_from_docs(docs: list[Any], *, per_doc_words: int = 24, max_docs: int = 6) -> str:
    lines: list[str] = []
    for d in docs[:max_docs]:
        tid = getattr(d, "docid", "") or ""
        snippet = _truncate_text_by_tokens(getattr(d, "text", "") or "", per_doc_words)
        lines.append(f"- {tid}: {snippet}")
    return "\n".join(lines)


def _cumulative_evidence_digest(
    contexts: list[str],
    *,
    max_bullets: int = 12,
    words_per_bullet: int = 14,
) -> str:
    """Compress collected snippets so the planner synthesizes gaps, not repeats."""
    if not contexts:
        return "(no evidence yet)"
    bullets: list[str] = []
    for block in contexts[-max_bullets:]:
        line = _truncate_text_by_tokens(block.replace("\n", " "), words_per_bullet)
        bullets.append("- " + line)
    return "\n".join(bullets)


def _delegation_block(delegation_context: str | None, *, max_chars: int = 8000) -> str:
    if not delegation_context or not str(delegation_context).strip():
        return ""
    dc = str(delegation_context).strip()[:max_chars]
    return (
        "\n\nDelegation / planning context (hints only; each new search_query must still be "
        "distinct from prior queries and from trivial paraphrases):\n"
        f"{dc}\n"
    )


def _multi_round_controller_prompt(
    *,
    seed: str,
    delegation_context: str | None,
    query: str,
    trajectory: list[dict[str, Any]],
    rounds: int,
    r: int,
    cumulative: str,
    brief_top: str,
) -> str:
    executed_queries = list(dict.fromkeys([str(s.get("query", "")).strip() for s in trajectory] + [query]))
    executed_queries = [x for x in executed_queries if x][-12:]
    prior_queries_block = "\n".join(f"{i + 1}. {qx}" for i, qx in enumerate(executed_queries))

    return (
        "You improve lexical (BM25) retrieval across rounds. Stay concise.\n"
        "Rules:\n"
        "- Output JSON only with keys: action, search_query, reason.\n"
        "- action is \"search\" or \"final\".\n"
        "- If action is \"search\", search_query MUST be a NEW query distinct from ALL prior queries "
        "(different entities, synonyms, facets, spellings, dates, OR complementary gaps).\n"
        "- Do NOT repeat or trivially shorten previous queries (avoid overlapping keyword bundles).\n"
        "- Base the next query on the QUESTION plus WHAT IS STILL MISSING given Evidence digest.\n"
        "- Use action=\"final\" when additional retrieval is unlikely to change the answer materially "
        "or when snippets already cover all constraints.\n\n"
        f"Question:\n{seed.strip()}\n"
        f"{_delegation_block(delegation_context)}"
        f"\nRound: {r}/{rounds}\n"
        f"Current BM25 query:\n{query}\n\n"
        f"Queries already used (do NOT repeat):\n{prior_queries_block}\n\n"
        f"Evidence digest so far:\n{cumulative}\n\n"
        f"Top snippets this round:\n{brief_top}\n"
    )


def _parse_controller_decision(raw: str) -> dict[str, Any]:
    s = (raw or "").strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m_action = re.search(r'"?action"?\s*:\s*"?(search|final)"?', s, re.I)
    m_query = re.search(r'"?search_query"?\s*:\s*"((?:[^"\\]|\\.)*)"', s, re.I)
    sq = ""
    if m_query:
        sq = bytes(m_query.group(1), "utf-8").decode("unicode_escape") if "\\" in m_query.group(1) else m_query.group(1)
    return {
        "action": (m_action.group(1).lower() if m_action else "final"),
        "search_query": sq.strip(),
        "reason": "",
    }


async def multi_round_retrieve_contexts(
    *,
    client: Any,
    model: str,
    question: str,
    retriever: BM25Retriever,
    retrieval_topk: int,
    doc_max_tokens: int,
    max_rounds: int,
    reasoning_effort: str | None = None,
    delegation_context: str | None = None,
    initial_bm25_query: str | None = None,
    evidence_policy: str = "last_round",
    max_final_snippets: int | None = None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    """BM25 rounds guided by JSON planner; skips redundant next-query repeats.

    Returns ``(contexts, trajectory, planner_usage_tokens)`` where:

    - ``contexts``:
      - ``evidence_policy="last_round"``: snippets from the last BM25 round.
      - ``evidence_policy="accumulated"``: de-duplicated snippets first seen across all rounds.
      In both modes, ``max_final_snippets`` can cap output size.
    - ``trajectory``: one entry per round with ``new_docids_first_seen_this_round`` / planner ``decision``.
    - Planner usage dict has ``prompt_tokens`` and ``completion_tokens`` summed over controller calls.
    """
    seen_docids: set[str] = set()
    accumulated_contexts: list[str] = []
    last_round_contexts: list[str] = []
    trajectory: list[dict[str, Any]] = []
    queries_used_norm: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    seed = (question or "").strip()
    iq = (initial_bm25_query or "").strip()
    query = iq[:4096] if iq else seed[:4096]
    rounds = max(1, int(max_rounds))

    extra_kw: dict[str, Any] = {}
    if reasoning_effort:
        extra_kw["reasoning_effort"] = reasoning_effort
    _eb = _extra_body_for_model(model)
    if _eb:
        extra_kw["extra_body"] = _eb

    def register_query_used(q: str) -> None:
        n = _normalize_query(q)
        if n and n not in queries_used_norm:
            queries_used_norm.append(n)

    register_query_used(query)

    for r in range(1, rounds + 1):
        docs = await asyncio.to_thread(retriever.search, query, max(1, int(retrieval_topk)))
        round_docids = [d.docid for d in docs]
        last_round_contexts = [
            f"[docid={d.docid}] {_truncate_text_by_tokens(d.text, int(doc_max_tokens))}"
            for d in docs
        ]
        new_docids_first_seen_this_round: list[str] = []
        for d in docs:
            if d.docid in seen_docids:
                continue
            seen_docids.add(d.docid)
            snippet = f"[docid={d.docid}] {_truncate_text_by_tokens(d.text, int(doc_max_tokens))}"
            accumulated_contexts.append(snippet)
            new_docids_first_seen_this_round.append(str(d.docid))

        brief_top = _brief_snippets_from_docs(docs)
        cumulative = _cumulative_evidence_digest(accumulated_contexts)

        controller_prompt = _multi_round_controller_prompt(
            seed=seed,
            delegation_context=delegation_context,
            query=query,
            trajectory=trajectory,
            rounds=rounds,
            r=r,
            cumulative=cumulative,
            brief_top=brief_top,
        )

        ctrl = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a retrieval planner. Output JSON only."},
                {"role": "user", "content": controller_prompt},
            ],
            response_format={"type": "json_object"},
            **extra_kw,
        )
        if getattr(ctrl, "usage", None) is not None:
            prompt_tokens += int(ctrl.usage.prompt_tokens or 0)
            completion_tokens += int(ctrl.usage.completion_tokens or 0)
        ctrl_text = ctrl.choices[0].message.content or ""
        decision = _parse_controller_decision(ctrl_text)
        action = str(decision.get("action", "final")).strip().lower()
        next_query = str(decision.get("search_query", "")).strip()
        reason = str(decision.get("reason", "")).strip()

        trajectory.append(
            {
                "round": r,
                "query": query,
                "retrieved_docids": round_docids,
                "new_docids_first_seen_this_round": new_docids_first_seen_this_round,
                "decision": {"action": action, "search_query": next_query, "reason": reason},
                "queries_used_norm_snapshot": list(queries_used_norm),
            }
        )

        if action != "search" or not next_query:
            break

        rejected = False
        reject_reason = ""
        nq_norm = _normalize_query(next_query)
        if not nq_norm:
            rejected = True
            reject_reason = "empty_search_query"
        elif nq_norm in queries_used_norm:
            rejected = True
            reject_reason = "duplicate_normalized"

        if rejected:
            trajectory[-1]["decision"]["rejected_next_query"] = next_query
            trajectory[-1]["decision"]["reject_reason"] = reject_reason
            break

        query = next_query[:4096]
        register_query_used(query)

    chosen_contexts = accumulated_contexts if str(evidence_policy).strip().lower() == "accumulated" else last_round_contexts
    if isinstance(max_final_snippets, int) and max_final_snippets > 0:
        chosen_contexts = chosen_contexts[:max_final_snippets]
    retr_raw = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    return chosen_contexts, trajectory, retr_raw


def multi_round_retrieve_contexts_sync(
    *,
    sync_client: Any,
    model: str,
    question: str,
    retriever: BM25Retriever,
    retrieval_topk: int,
    doc_max_tokens: int,
    max_rounds: int,
    reasoning_effort: str | None = None,
    delegation_context: str | None = None,
    initial_bm25_query: str | None = None,
    evidence_policy: str = "last_round",
    max_final_snippets: int | None = None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    """Same semantics as ``multi_round_retrieve_contexts`` but blocking OpenAI client + sync BM25.

    One retrieval session = multiple BM25 rounds, each followed by one planner LLM call.
    """
    seen_docids: set[str] = set()
    accumulated_contexts: list[str] = []
    last_round_contexts: list[str] = []
    trajectory: list[dict[str, Any]] = []
    queries_used_norm: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    seed = (question or "").strip()
    iq = (initial_bm25_query or "").strip()
    query = iq[:4096] if iq else seed[:4096]
    rounds = max(1, int(max_rounds))

    extra_kw: dict[str, Any] = {}
    if reasoning_effort:
        extra_kw["reasoning_effort"] = reasoning_effort
    _eb = _extra_body_for_model(model)
    if _eb:
        extra_kw["extra_body"] = _eb

    def register_query_used(q: str) -> None:
        n = _normalize_query(q)
        if n and n not in queries_used_norm:
            queries_used_norm.append(n)

    register_query_used(query)

    for r in range(1, rounds + 1):
        docs = retriever.search(query, max(1, int(retrieval_topk)))
        round_docids = [d.docid for d in docs]
        last_round_contexts = [
            f"[docid={d.docid}] {_truncate_text_by_tokens(d.text, int(doc_max_tokens))}"
            for d in docs
        ]
        new_docids_first_seen_this_round: list[str] = []
        for d in docs:
            if d.docid in seen_docids:
                continue
            seen_docids.add(d.docid)
            snippet = f"[docid={d.docid}] {_truncate_text_by_tokens(d.text, int(doc_max_tokens))}"
            accumulated_contexts.append(snippet)
            new_docids_first_seen_this_round.append(str(d.docid))

        brief_top = _brief_snippets_from_docs(docs)
        cumulative = _cumulative_evidence_digest(accumulated_contexts)

        controller_prompt = _multi_round_controller_prompt(
            seed=seed,
            delegation_context=delegation_context,
            query=query,
            trajectory=trajectory,
            rounds=rounds,
            r=r,
            cumulative=cumulative,
            brief_top=brief_top,
        )

        ctrl = sync_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a retrieval planner. Output JSON only."},
                {"role": "user", "content": controller_prompt},
            ],
            response_format={"type": "json_object"},
            **extra_kw,
        )
        if getattr(ctrl, "usage", None) is not None:
            prompt_tokens += int(ctrl.usage.prompt_tokens or 0)
            completion_tokens += int(ctrl.usage.completion_tokens or 0)

        ctrl_text = ctrl.choices[0].message.content or ""
        decision = _parse_controller_decision(ctrl_text)
        action = str(decision.get("action", "final")).strip().lower()
        next_query = str(decision.get("search_query", "")).strip()
        reason = str(decision.get("reason", "")).strip()

        trajectory.append(
            {
                "round": r,
                "query": query,
                "retrieved_docids": round_docids,
                "new_docids_first_seen_this_round": new_docids_first_seen_this_round,
                "decision": {"action": action, "search_query": next_query, "reason": reason},
                "queries_used_norm_snapshot": list(queries_used_norm),
            }
        )

        if action != "search" or not next_query:
            break

        rejected = False
        reject_reason = ""
        nq_norm = _normalize_query(next_query)
        if not nq_norm:
            rejected = True
            reject_reason = "empty_search_query"
        elif nq_norm in queries_used_norm:
            rejected = True
            reject_reason = "duplicate_normalized"

        if rejected:
            trajectory[-1]["decision"]["rejected_next_query"] = next_query
            trajectory[-1]["decision"]["reject_reason"] = reject_reason
            break

        query = next_query[:4096]
        register_query_used(query)

    chosen_contexts = accumulated_contexts if str(evidence_policy).strip().lower() == "accumulated" else last_round_contexts
    if isinstance(max_final_snippets, int) and max_final_snippets > 0:
        chosen_contexts = chosen_contexts[:max_final_snippets]
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    return chosen_contexts, trajectory, usage
