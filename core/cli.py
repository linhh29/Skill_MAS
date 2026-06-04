"""CLI for Skill_MAS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "vitabench_single" / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "vitabench_single" / "src"))

from ..utils.config import (
    DEFAULT_AGENT_LLM,
    DEFAULT_BENCH_ID,
    DEFAULT_DOMAIN,
    DEFAULT_DRB_RACE_MAX_WORKERS,
    DEFAULT_EVALUATOR_LLM,
    DEFAULT_EVOLVE_ROUNDS,
    DEFAULT_FRESH,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_STEPS,
    DEFAULT_OPTIMIZER_LLM,
    DEFAULT_RUN_ID,
    DEFAULT_USER_LLM,
    DEFAULT_VAL_SIZE,
    EVOLVE_K_TRAJECTORIES,
    log_root_for,
    results_dataset_dir,
)
from .dataset_split import load_split_file
from .pipeline import evolve, snapshot_baseline_only
from .task_select import (
    browsecomp_validate_ids,
    drb_validate_ids,
    hlemath_validate_ids,
    vitabench_validate_ids,
)


def cmd_list_val(args: argparse.Namespace) -> None:
    bb = (args.bench_backend or "vitabench").strip().lower()
    if not getattr(args, "jsonl", "").strip():
        raise ValueError("--jsonl is required for all backends.")
    unified_jsonl = Path(args.jsonl).resolve()
    if getattr(args, "split_file", "").strip():
        sp = load_split_file(Path(args.split_file.strip()).resolve())
        ids = [str(x) for x in (sp.get("ids") or sp.get("val_ids") or [])]
        if int(getattr(args, "max_problems", 0) or 0) > 0:
            ids = ids[: int(args.max_problems)]
        print(f"split_file={args.split_file} count={len(ids)}")
        for i in ids:
            print(i)
        return
    if bb == "drb":
        ids = drb_validate_ids(unified_jsonl)
        if int(getattr(args, "max_problems", 0) or 0) > 0:
            ids = ids[: int(args.max_problems)]
        src = str(unified_jsonl)
        print(f"bench_backend=drb source={src} count={len(ids)}")
        for i in ids:
            print(i)
        return
    if bb == "hlemath":
        jp = unified_jsonl
        ids = hlemath_validate_ids(jp)
        if int(getattr(args, "max_problems", 0) or 0) > 0:
            ids = ids[: int(args.max_problems)]
        print(f"bench_backend=hlemath source={jp} count={len(ids)}")
        for i in ids:
            print(i)
        return
    if bb == "bcp":
        ids = browsecomp_validate_ids(unified_jsonl)
        if int(getattr(args, "max_problems", 0) or 0) > 0:
            ids = ids[: int(args.max_problems)]
        src = str(unified_jsonl)
        print(f"bench_backend=bcp source={src} count={len(ids)}")
        for i in ids:
            print(i)
        return
    ids = vitabench_validate_ids(unified_jsonl)
    src = str(unified_jsonl)
    if int(getattr(args, "max_problems", 0) or 0) > 0:
        limit = int(args.max_problems)
        ids = ids[:limit]
    print("bench_backend=vitabench source={} count={}".format(src, len(ids)))
    for i in ids:
        print(i)


def cmd_split_build(args: argparse.Namespace) -> None:
    del args
    print("split-build is deprecated.")
    print("Validation data is read directly from benchmark folders:")
    print("- vitabench_single/data/vita_validate.json")
    print("- deep_research_bench/data/drb_validate.jsonl")
    print("- hlemath/data/hlemath_validate.jsonl")
    print("- BrowseComp-Plus/data/browsecomp_plus_validate.jsonl")


def cmd_init(args: argparse.Namespace) -> None:
    init_root = (
        Path(args.init_skill_root.strip()).resolve()
        if args.init_skill_root.strip()
        else None
    )
    bb = (args.bench_backend or "vitabench").strip().lower()
    rid = snapshot_baseline_only(
        args.bench_id,
        init_root,
        args.domain,
        run_id=args.run_id,
        bench_backend=bb,
        agent_llm=args.agent_llm,
    )
    ds = results_dataset_dir(bb, args.agent_llm)
    print(f"run_id={rid}")
    print(
        f"skills under Skill_MAS/results/{ds}/artifacts/skills/{args.bench_id}/{rid}/round_00/"
    )


def cmd_evolve(args: argparse.Namespace) -> None:
    if not getattr(args, "jsonl", "").strip():
        raise ValueError("--jsonl is required for all backends.")
    init_root = (
        Path(args.init_skill_root.strip()).resolve()
        if args.init_skill_root.strip()
        else None
    )
    split_path = (
        Path(args.split_file.strip()).resolve()
        if args.split_file.strip()
        else None
    )
    _bcp_judge = (getattr(args, "judge_llm", "") or "").strip()
    _bcp_idx = (getattr(args, "bcp_index_path", "") or "").strip()
    rid = evolve(
        bench_id=args.bench_id,
        domain=args.domain,
        task_set_name=args.task_set_name,
        val_size=args.val_size,
        rounds=args.rounds,
        init_skill_root=init_root,
        split_file=split_path,
        split_seed=args.split_seed,
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        evaluator_llm=args.evaluator_llm,
        max_steps=args.max_steps,
        max_concurrency=args.max_concurrency,
        language=args.language,
        optimizer_llm=args.optimizer_llm,
        run_id=args.run_id,
        fresh=args.fresh,
        bench_backend=args.bench_backend,
        drb_bench_root=Path(args.drb_bench_root).resolve()
        if args.drb_bench_root.strip()
        else None,
        drb_race_max_workers=args.drb_race_max_workers,
        k_trajectories=args.k_trajectories,
        hlemath_jsonl=None,
        bcp_jsonl=None,
        jsonl_path=Path(args.jsonl).resolve(),
        max_problems=int(getattr(args, "max_problems", 0) or 0),
        bcp_judge_llm=(_bcp_judge if _bcp_judge else None),
        bcp_judge_timeout_s=float(getattr(args, "judge_timeout_s", 120.0)),
        bcp_index_path=Path(_bcp_idx).resolve() if _bcp_idx else None,
        bcp_retrieval_topk=int(getattr(args, "bcp_retrieval_topk", 5)),
        bcp_doc_max_tokens=int(getattr(args, "bcp_doc_max_tokens", 512)),
        bcp_max_retrieval_rounds=int(getattr(args, "bcp_max_retrieval_rounds", 10)),
    )
    bb = (args.bench_backend or "vitabench").strip().lower()
    ds = results_dataset_dir(bb, args.agent_llm)
    print(f"run_id={rid}")
    print(f"artifacts/skills: .../results/{ds}/artifacts/skills/{args.bench_id}/{rid}/")
    print(f"summaries: .../results/{ds}/artifacts/runs/{args.bench_id}/{rid}/")
    print(f"logs: {log_root_for(bb, args.agent_llm) / args.bench_id / rid}/")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="Skill_MAS",
        description="Skill-MAS evolution (single SKILL.md, built-in validation datasets)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-val", help="Print validation task IDs (split file or last-N)")
    p_list.add_argument("--domain", default=DEFAULT_DOMAIN)
    p_list.add_argument("--task-set-name", default="")
    p_list.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    p_list.add_argument("--language", default=DEFAULT_LANGUAGE)
    p_list.add_argument(
        "--split-file",
        default="",
        help="If set, print ids from this JSON (overrides default validation data files)",
    )
    p_list.add_argument(
        "--bench-backend",
        default="vitabench",
        choices=("vitabench", "drb", "hlemath", "bcp"),
        help="Task source backend",
    )
    p_list.add_argument(
        "--jsonl",
        default="",
        help="Unified dataset file path override (all backends).",
    )
    p_list.add_argument(
        "--max-problems",
        type=int,
        default=0,
        help="Limit number of validation problems loaded (0 = all).",
    )
    p_list.set_defaults(func=cmd_list_val)

    p_split = sub.add_parser(
        "split-build",
        help="Deprecated. Show built-in validation data locations.",
    )
    p_split.add_argument("--domain", default=DEFAULT_DOMAIN)
    p_split.add_argument("--task-set-name", default="")
    p_split.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    p_split.add_argument("--seed", type=int, default=0)
    p_split.add_argument("--language", default=DEFAULT_LANGUAGE)
    p_split.add_argument(
        "--force",
        action="store_true",
        help="Overwrite split file even if an equivalent one exists",
    )
    p_split.add_argument(
        "--bench-backend",
        default="vitabench",
        choices=("vitabench", "drb", "hlemath", "bcp"),
    )
    p_split.add_argument(
        "--hlemath-jsonl",
        default="",
        help="hlemath: path to hlemath_validate.jsonl (default: hlemath/data/hlemath_validate.jsonl)",
    )
    p_split.set_defaults(func=cmd_split_build)

    p_init = sub.add_parser("init-run", help="Copy init SKILL.md to round_00 only")
    p_init.add_argument("--bench-id", default=DEFAULT_BENCH_ID)
    p_init.add_argument("--domain", default=DEFAULT_DOMAIN)
    p_init.add_argument(
        "--init-skill-root",
        default="",
        help="Directory containing SKILL.md (default: Skill_MAS/init_skill)",
    )
    p_init.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        metavar="ID",
        help="Logical run folder under this bench (default from config; colliding name → default_2, …)",
    )
    p_init.add_argument(
        "--bench-backend",
        default="vitabench",
        choices=("vitabench", "drb", "hlemath", "bcp"),
        help="Artifact subtree under Skill_MAS/results/<dataset>_<agent_model>/",
    )
    p_init.add_argument(
        "--agent-llm",
        default=DEFAULT_AGENT_LLM,
        help="Rollout/agent model id; names results folder Skill_MAS/results/<backend>_<tag>/",
    )
    p_init.set_defaults(func=cmd_init)

    p_e = sub.add_parser("evolve", help="Run evaluation + bank optimization rounds")
    p_e.add_argument("--bench-id", default=DEFAULT_BENCH_ID)
    p_e.add_argument("--domain", default=DEFAULT_DOMAIN)
    p_e.add_argument("--task-set-name", default="")
    p_e.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    p_e.add_argument("--split-seed", type=int, default=0, dest="split_seed")
    p_e.add_argument(
        "--split-file",
        default="",
        help="Optional JSON with val_ids; when omitted, use backend built-in validation data.",
    )
    p_e.add_argument(
        "--init-skill-root",
        default="",
        help="Directory containing SKILL.md for round_00 (default: Skill_MAS/init_skill)",
    )
    p_e.add_argument("--rounds", type=int, default=DEFAULT_EVOLVE_ROUNDS)
    p_e.add_argument("--agent-llm", default=DEFAULT_AGENT_LLM)
    p_e.add_argument("--user-llm", default=DEFAULT_USER_LLM)
    p_e.add_argument("--evaluator-llm", default=DEFAULT_EVALUATOR_LLM)
    p_e.add_argument("--optimizer-llm", default=DEFAULT_OPTIMIZER_LLM)
    p_e.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p_e.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    p_e.add_argument("--language", default=DEFAULT_LANGUAGE)
    p_e.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        metavar="ID",
        help="Logical run folder (default from config). Reuses existing tree unless --fresh.",
    )
    p_e.add_argument(
        "--fresh",
        action="store_true",
        default=DEFAULT_FRESH,
        help="New run under <run-id>_<n> (n = max existing + 1 across artifacts); start from round_00.",
    )
    p_e.add_argument(
        "--bench-backend",
        default="vitabench",
        choices=("vitabench", "drb", "hlemath", "bcp"),
        help="vitabench | drb+RACE | hlemath JSONL | bcp JSONL",
    )
    p_e.add_argument(
        "--drb-bench-root",
        default="",
        help="DRB: DeepResearchBench repo root (default: deep_research_bench next to Ant)",
    )
    p_e.add_argument(
        "--drb-race-max-workers",
        type=int,
        default=DEFAULT_DRB_RACE_MAX_WORKERS,
        help="DRB: parallel workers for deepresearch_bench_race.py",
    )
    p_e.add_argument(
        "--jsonl",
        default="",
        help="Unified dataset file path override (all backends).",
    )
    p_e.add_argument(
        "--max-problems",
        type=int,
        default=0,
        help="Limit number of validation problems loaded (0 = all).",
    )
    p_e.add_argument(
        "--k-trajectories",
        type=int,
        default=EVOLVE_K_TRAJECTORIES,
        help="Step1: number of trajectories per sample",
    )
    p_e.add_argument(
        "--judge-llm",
        default="",
        metavar="MODEL",
        help="bcp only: LLM-as-judge model id (BrowseComp-Plus judge.py); omit for exact-match scoring.",
    )
    p_e.add_argument(
        "--judge-timeout-s",
        type=float,
        default=120.0,
        help="bcp only: per-task timeout for judge_answer (seconds).",
    )
    p_e.add_argument(
        "--bcp-index-path",
        default="",
        metavar="PATH",
        help="bcp only: BM25 index directory (default: BrowseComp-Plus/scripts_build_index/indexes/bm25).",
    )
    p_e.add_argument(
        "--bcp-retrieval-topk",
        type=int,
        default=5,
        help="bcp only: BM25 hits per query round (same as BrowseComp-Plus run_eval).",
    )
    p_e.add_argument(
        "--bcp-doc-max-tokens",
        type=int,
        default=512,
        help="bcp only: per-snippet truncation budget for retrieved docs.",
    )
    p_e.add_argument(
        "--bcp-max-retrieval-rounds",
        type=int,
        default=10,
        help="bcp only: max planner-guided retrieval rounds per sub-agent tool call.",
    )
    p_e.set_defaults(func=cmd_evolve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
