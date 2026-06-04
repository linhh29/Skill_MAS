#!/usr/bin/env python3
"""Evaluate Skill-MAS on HLEMATH JSONL via Skill_MAS build pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from Skill_MAS.utils.paths import (
    DATASET_ROOT,
    HLEMATH_ROOT,
    INIT_SKILL_DIR,
    PACKAGE_ROOT,
    ensure_sys_path,
)

ensure_sys_path(include_dataset=True)
_BENCH_ROOT = HLEMATH_ROOT

from hlemath.openai_client import AsyncOpenAIClient
from hlemath.skill_mas_runner import HleMathTask, run_hlemath_skill_mas_on_task
from hlemath.score import HLEMATHScorer
from Skill_MAS.skill_mas.process_trace_layout import (
    sample_trace_json_dir,
    skill_mas_sample_log_subdir,
    skill_mas_trace_suffix_parts,
    skill_workspace_from_init_path,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                raise ValueError(f"Invalid JSONL: empty line at {path}:{lineno}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{lineno}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL row type at {path}:{lineno}: expected object")
            rows.append(row)
    return rows


def _load_existing_scored_sample(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    required = {"index", "score", "extracted", "usage", "prediction"}
    if not isinstance(data, dict) or not required.issubset(data.keys()):
        return None
    return data


def main() -> None:
    p = argparse.ArgumentParser(description="HLEMATH Skill-MAS evaluation")
    p.add_argument(
        "--jsonl",
        type=Path,
        default=_BENCH_ROOT / "data" / "hlemath_validate.jsonl",
        help="Path to hlemath_validate.jsonl or hlemath_test.jsonl",
    )
    p.add_argument(
        "--init-skill",
        type=Path,
        default=None,
        help="Init skill markdown path. Default: SKILL_MAS_INIT_SKILL or Skill_MAS/init_skill/SKILL.md",
    )
    p.add_argument("--agent-llm", default=os.environ.get("AGENT_MODEL", "qwen3.5-plus"))
    p.add_argument("--max-concurrency", type=int, default=50, help="Max concurrent samples (capped at 50)")
    p.add_argument("--max-problems", type=int, default=0, help="0 = all lines")
    p.add_argument("--indices", type=str, default="", help="Comma-separated line indices (default: all)")
    p.add_argument("--out", type=Path, default=None, help="Write JSON summary path")
    args = p.parse_args()

    init_skill = args.init_skill
    if init_skill is None:
        env = (os.environ.get("SKILL_MAS_INIT_SKILL") or "").strip()
        init_skill = Path(env) if env else INIT_SKILL_DIR / "SKILL.md"
    init_skill = Path(init_skill).resolve()
    if not init_skill.is_file():
        raise FileNotFoundError(f"init skill file not found: {init_skill}")

    rows = _load_jsonl(Path(args.jsonl).resolve())
    if args.indices.strip():
        want = {int(x.strip()) for x in args.indices.split(",") if x.strip()}
        indices = sorted(want)
        for i in indices:
            if i < 0 or i >= len(rows):
                raise IndexError(f"index {i} out of range [0,{len(rows)})")
    else:
        indices = list(range(len(rows)))
        if args.max_problems and args.max_problems > 0:
            indices = indices[: args.max_problems]

    client = AsyncOpenAIClient(model=args.agent_llm)
    scorer = HLEMATHScorer()
    skill_workspace = skill_workspace_from_init_path(init_skill)
    trace_suffix = skill_mas_trace_suffix_parts(skill_workspace)
    trace_root = (
        _BENCH_ROOT / "results" / args.agent_llm / "skill_mas_process_traces"
    ).joinpath(*trace_suffix)
    trace_root.mkdir(parents=True, exist_ok=True)
    scored_root = trace_root / "scored_samples"
    scored_root.mkdir(parents=True, exist_ok=True)

    async def _run_all() -> tuple[list[dict], int, float]:
        sem = asyncio.Semaphore(max(1, min(50, int(args.max_concurrency))))
        progress_lock = asyncio.Lock()
        done = 0
        total = len(indices)

        async def _one(idx: int) -> dict:
            nonlocal done
            scored_path = scored_root / f"{idx}.json"
            existing = _load_existing_scored_sample(scored_path)
            if existing is not None:
                async with progress_lock:
                    done += 1
                    print(
                        f"[hlemath-skill-mas] resume hit {done}/{total} index={idx} "
                        f"score={existing.get('score')}",
                        flush=True,
                    )
                return existing

            row = rows[idx]
            q = str(row.get("question", ""))
            gold = str(row.get("answer", ""))
            task = HleMathTask(id=int(idx), prompt=q, language="en")
            row_tid = row.get("id", idx)
            trace_stem = skill_mas_sample_log_subdir(idx, row_tid)
            t0 = time.perf_counter()
            print(f"[hlemath-skill-mas] start index={idx}", flush=True)

            async with sem:
                try:
                    pred_text, usage, _trace = await run_hlemath_skill_mas_on_task(
                        task,
                        init_skill_path=init_skill,
                        client=client,
                        process_trace_dir=trace_root,
                        trace_stem=trace_stem,
                        quiet=False,
                    )
                except Exception as e:
                    pred_text = ""
                    usage = {}
                    _trace = {"failure_stage": "runner", "failure_reason": f"{type(e).__name__}: {e}"}
            sc, extracted = scorer.calculate_score(gold, pred_text)
            sample = {
                "index": idx,
                "question": q,
                "expected_answer": gold,
                "prediction": pred_text,
                "score": sc,
                "extracted": extracted,
                "usage": usage,
                "prediction_preview": pred_text[:500],
                "model": args.agent_llm,
                "trace_json": str((sample_trace_json_dir(trace_root) / f"{trace_stem}.json").resolve()),
                "sample_log_dir": str((trace_root / "sample_logs" / trace_stem).resolve()),
                "failure_stage": (_trace or {}).get("failure_stage"),
                "failure_reason": (_trace or {}).get("failure_reason"),
            }
            scored_path.write_text(
                json.dumps(sample, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            elapsed = time.perf_counter() - t0
            async with progress_lock:
                done += 1
                print(
                    f"[hlemath-skill-mas] done {done}/{total} index={idx} "
                    f"score={sc} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            return sample

        try:
            out = await asyncio.gather(*[_one(i) for i in indices])
            out.sort(key=lambda x: int(x["index"]))
            total_score = sum(int(x["score"]) for x in out)
            total_cost = sum(float((x.get("usage") or {}).get("estimated_cost_usd", 0.0)) for x in out)
            return out, total_score, total_cost
        finally:
            await client.aclose()

    results, total_score, total_cost = asyncio.run(_run_all())

    n = len(indices) or 1
    summary = {
        "jsonl": str(Path(args.jsonl).resolve()),
        "init_skill": str(init_skill),
        "agent_llm": args.agent_llm,
        "num_evaluated": len(indices),
        "accuracy": total_score / n,
        "total_score": total_score,
        "estimated_total_cost_usd_from_model_pricing_json": round(total_cost, 10),
        "results": results,
    }
    out = args.out
    if out is None:
        out = trace_root / "hlemath_skill_mas_eval_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (trace_root / "generation_cost.json").write_text(
        json.dumps(
            {
                "model": args.agent_llm,
                "num_tasks": len(indices),
                "estimated_total_cost_usd_from_model_pricing_json": round(total_cost, 10),
                "pricing_json_note": "USD per 1M tokens from Skill_MAS/skill_mas/model_config.json via Skill_MAS.skill_mas.openai_async_client.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (trace_root / "_skill_mas_run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "skill_mas_run_manifest/1",
                "mode": "skill_mas",
                "model": args.agent_llm,
                "dataset": str(Path(args.jsonl).resolve()),
                "max_concurrency": int(max(1, min(50, int(args.max_concurrency)))),
                "process_trace_dir": str(trace_root.resolve()),
                "run_results": {
                    "summary_json": str(out.resolve()),
                    "generation_cost_json": str((trace_root / "generation_cost.json").resolve()),
                    "scored_samples_dir": str(scored_root.resolve()),
                    "sample_logs_dir": str((trace_root / "sample_logs").resolve()),
                    "sample_trace_json_dir": str(sample_trace_json_dir(trace_root).resolve()),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"accuracy": summary["accuracy"], "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
