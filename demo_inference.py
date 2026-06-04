#!/usr/bin/env python3
"""Run Skill-MAS build + inference on a single question with any skill file."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL_MAS_HOME = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "vitabench_single" / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "vitabench_single" / "src"))

from Skill_MAS.core.model_config_runtime import apply_model_runtime_params, model_runtime_params
from Skill_MAS.skill_mas.build import (
    _extract_predicted_answer,
    make_planner_call_fn,
    make_text_call_fn,
    run_mas_pipeline_with_retries,
)
from Skill_MAS.skill_mas.openai_async_client import AsyncOpenAIClient
from Skill_MAS.utils.config import BROWSECOMP_BENCH_ROOT, DEFAULT_AGENT_LLM


def _infer_dataset_from_skill(skill_path: Path) -> str | None:
    name = skill_path.name.lower()
    stem = skill_path.stem.lower()
    parent = skill_path.parent.name.lower()
    token = stem if stem not in {"skill", "skill.md"} else parent
    mapping = {
        "vitabench": "vita",
        "vita": "vita",
        "drb": "drb",
        "deepresearch": "drb",
        "hlemath": "hlemath",
        "hle": "hlemath",
        "bcp": "bcp",
        "browsecomp": "bcp",
    }
    for key, dataset in mapping.items():
        if key in token or key in parent:
            return dataset
    if parent == "init_skill":
        return "hlemath"
    return None


def _resolve_skill_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    if p.is_dir():
        candidate = p / "SKILL.md"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"No SKILL.md under directory: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"Skill file not found: {p}")
    return p


def _apply_model_env(model: str) -> None:
    params = model_runtime_params(model)
    api_key = params.get("api_key")
    base_url = params.get("base_url")
    if api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(api_key)
    if base_url and not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = str(base_url)
    for env_name, key in (
        ("SKILL_MAS_AGENT_TEMPERATURE", "temperature"),
        ("SKILL_MAS_AGENT_REASONING_EFFORT", "reasoning_effort"),
        ("SKILL_MAS_AGENT_MAX_TOKENS", "max_tokens"),
    ):
        if key in params and params[key] is not None and not os.environ.get(env_name):
            os.environ[env_name] = str(params[key])


def _class_name_for_question(question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]
    return f"DemoMASWorkflow_{digest}"


def _build_bcp_tool_call_fn(
    *,
    client: AsyncOpenAIClient,
    question: str,
    index_path: Path | None,
    retrieval_topk: int,
    doc_max_tokens: int,
    max_retrieval_rounds: int,
):
    bcp_root = BROWSECOMP_BENCH_ROOT
    if not bcp_root.is_dir():
        raise FileNotFoundError(
            f"BrowseComp-Plus repo not found at {bcp_root}. "
            "Clone it next to the repository root or choose a non-bcp skill/dataset."
        )
    if str(bcp_root) not in sys.path:
        sys.path.insert(0, str(bcp_root))

    from browsecomp_retrieval_tool import make_browsecomp_retrieval_tool_fn
    from retrieval import BM25Retriever

    idx_default = bcp_root / "scripts_build_index" / "indexes" / "bm25"
    index_resolved = index_path.resolve() if index_path else idx_default
    retriever = BM25Retriever(index_path=str(index_resolved))
    return make_browsecomp_retrieval_tool_fn(
        chat_client=client.client,
        model=client.model,
        retriever=retriever,
        retrieval_topk=int(retrieval_topk),
        doc_max_tokens=int(doc_max_tokens),
        max_rounds=int(max_retrieval_rounds),
        reasoning_effort=client.reasoning_effort,
        full_task_question=question,
        pricing=client.pricing,
        retrieval_sessions_out=None,
    )


async def _run(args: argparse.Namespace) -> int:
    skill_path = _resolve_skill_path(args.skill)
    question = (args.question or "").strip()
    if not question:
        raise ValueError("Question must not be empty.")

    dataset = (args.dataset or "").strip().lower() or _infer_dataset_from_skill(skill_path)
    if not dataset:
        raise ValueError(
            "Could not infer dataset from skill path. Pass --dataset "
            "(hlemath|drb|bcp|vita)."
        )

    model = (args.model or DEFAULT_AGENT_LLM).strip()
    _apply_model_env(model)

    class_name = _class_name_for_question(question)
    client = AsyncOpenAIClient(model=model)
    runtime = apply_model_runtime_params(model, {})
    planner_kwargs = {
        k: runtime[k]
        for k in ("temperature", "max_tokens", "reasoning_effort", "extra_body")
        if k in runtime and runtime[k] is not None
    }

    planner_call_fn = make_planner_call_fn(client, generation_kwargs=planner_kwargs)
    text_call_fn = make_text_call_fn(client)

    tool_call_fn = None
    if dataset in {"bcp", "browsecomp"}:
        tool_call_fn = _build_bcp_tool_call_fn(
            client=client,
            question=question,
            index_path=Path(args.bcp_index_path).expanduser() if args.bcp_index_path else None,
            retrieval_topk=args.bcp_retrieval_topk,
            doc_max_tokens=args.bcp_doc_max_tokens,
            max_retrieval_rounds=args.bcp_max_retrieval_rounds,
        )
    elif dataset in {"vita", "vitabench"}:
        print(
            "[demo_inference] vita dataset selected: tool-backed VitaBench simulation is not "
            "wired in this demo script. Use hlemath/drb/bcp skills for standalone inference, "
            "or run full benchmark evolution via run_vita.sh.",
            file=sys.stderr,
        )
        return 2

    print(f"[demo_inference] skill={skill_path}", flush=True)
    print(f"[demo_inference] dataset={dataset} model={model}", flush=True)
    print(f"[demo_inference] question={question[:200]}{'...' if len(question) > 200 else ''}", flush=True)

    try:
        result = await run_mas_pipeline_with_retries(
            task_text=question,
            class_name=class_name,
            init_skill_path=skill_path,
            planner_call_fn=planner_call_fn,
            text_call_fn=text_call_fn,
            tool_call_fn=tool_call_fn,
            max_generation_attempts=args.max_generation_attempts,
            max_execution_attempts=args.max_execution_attempts,
            dataset_name=dataset,
        )
    finally:
        await client.aclose()

    if args.save_mas_code and result.mas_code:
        out_path = Path(args.save_mas_code).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.mas_code, encoding="utf-8")
        print(f"[demo_inference] saved MAS code -> {out_path}", flush=True)

    if args.verbose and result.artifacts is not None:
        traces = [
            {
                "stage": t.stage,
                "stage_name": t.stage_name,
                "elapsed_sec": t.elapsed_sec,
                "parsed_json": t.parsed_json,
            }
            for t in result.artifacts.stage_traces
        ]
        print(json.dumps({"build_stages": traces}, ensure_ascii=False, indent=2), flush=True)

    if not result.success:
        print(f"[demo_inference] failed stage={result.failure_stage} reason={result.failure_reason}", file=sys.stderr)
        return 1

    final_output = result.final_output.strip()
    predicted = _extract_predicted_answer(final_output) if dataset in {"hlemath", "hle"} else final_output

    print("\n===== ANSWER =====")
    print(predicted if predicted else final_output)
    if predicted and predicted != final_output:
        print("\n===== FULL FINAL OUTPUT =====")
        print(final_output)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and run a Skill-MAS workflow from a skill file and one input question.",
    )
    parser.add_argument(
        "--skill",
        required=True,
        help="Path to SKILL.md or optimized_skill/*.md (e.g. Skill_MAS/init_skill/SKILL.md)",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Input question / task prompt for the generated MAS.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_AGENT_LLM,
        help=f"Agent/planner model id (default: {DEFAULT_AGENT_LLM})",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Dataset profile: hlemath|drb|bcp|vita. Auto-inferred from skill filename when omitted.",
    )
    parser.add_argument(
        "--bcp-index-path",
        default="",
        help="BrowseComp-Plus BM25 index directory (bcp dataset only).",
    )
    parser.add_argument(
        "--bcp-retrieval-topk",
        type=int,
        default=5,
        help="BM25 hits per retrieval round (bcp only).",
    )
    parser.add_argument(
        "--bcp-doc-max-tokens",
        type=int,
        default=512,
        help="Per-snippet token budget for retrieved docs (bcp only).",
    )
    parser.add_argument(
        "--bcp-max-retrieval-rounds",
        type=int,
        default=10,
        help="Max planner-guided retrieval rounds (bcp only).",
    )
    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=5,
        help="Retry budget for three-stage MAS code generation.",
    )
    parser.add_argument(
        "--max-execution-attempts",
        type=int,
        default=3,
        help="Retry budget for executing generated MAS code.",
    )
    parser.add_argument(
        "--save-mas-code",
        default="",
        help="Optional path to write the generated MAS Python code.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print parsed JSON from each build stage.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
