"""Paths and defaults for Skill_MAS (repo-root relative)."""

from __future__ import annotations

import os
import re
from pathlib import Path


def print_traces_enabled() -> bool:
    return os.environ.get("MASKILL_PRINT_TRACES", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


from Skill_MAS.utils.paths import (
    BCP_ROOT,
    DRB_ROOT,
    HLEMATH_ROOT,
    INIT_SKILL_DIR,
    PACKAGE_ROOT,
    REPO_ROOT,
    RESULTS_ROOT,
    SKILL_MAS_HOME,
    SKILL_MAS_ROOT,
    VITA_SRC,
    VITABENCH_ROOT,
)

# Re-export for modules that import from config
DATASET_ROOT = SKILL_MAS_ROOT / "dataset"
BENCH_SEGMENT_VITABENCH = "vitabench"
BENCH_SEGMENT_DRB = "drb"
BENCH_SEGMENT_HLEMATH = "hlemath"
BENCH_SEGMENT_BCP = "bcp"


def sanitize_agent_llm_for_path(raw: str | None) -> str:
    """Stable filesystem segment from an LLM model id (dataset dir is ``{bench}_{tag}``)."""
    s = (raw or "").strip().lower()
    if not s:
        s = str(DEFAULT_AGENT_LLM).strip().lower()
    s = s.replace("-", "")
    s = re.sub(r"[^a-z0-9._]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    if not s:
        s = "model"
    return s[:100]


def results_dataset_dir(bench_backend: str, agent_llm: str | None = None) -> str:
    """Top-level folder under ``Skill_MAS/results/``: ``{dataset}_{model_tag}`` (e.g. ``vitabench_qwen3.5plus``)."""
    return f"{bench_segment(bench_backend)}_{sanitize_agent_llm_for_path(agent_llm)}"


def bench_segment(bench_backend: str) -> str:
    bb = (bench_backend or BENCH_SEGMENT_VITABENCH).strip().lower()
    mapping = {
        BENCH_SEGMENT_VITABENCH: BENCH_SEGMENT_VITABENCH,
        BENCH_SEGMENT_DRB: BENCH_SEGMENT_DRB,
        BENCH_SEGMENT_HLEMATH: BENCH_SEGMENT_HLEMATH,
        BENCH_SEGMENT_BCP: BENCH_SEGMENT_BCP,
    }
    return mapping.get(bb, BENCH_SEGMENT_VITABENCH)


def artifacts_root(bench_backend: str, agent_llm: str | None = None) -> Path:
    return RESULTS_ROOT / results_dataset_dir(bench_backend, agent_llm) / "artifacts"


def runs_dir(bench_backend: str, agent_llm: str | None = None) -> Path:
    return artifacts_root(bench_backend, agent_llm) / "runs"


def skills_evolution_dir(bench_backend: str, agent_llm: str | None = None) -> Path:
    return artifacts_root(bench_backend, agent_llm) / "skills"


def merged_workspaces_dir(bench_backend: str, agent_llm: str | None = None) -> Path:
    return artifacts_root(bench_backend, agent_llm) / "merged_skill_workspaces"


def optimizer_validate_dir(bench_backend: str, agent_llm: str | None = None) -> Path:
    return artifacts_root(bench_backend, agent_llm) / "optimizer_validate"


def log_root_for(bench_backend: str, agent_llm: str | None = None) -> Path:
    return RESULTS_ROOT / results_dataset_dir(bench_backend, agent_llm) / "log"


DEFAULT_BENCH_ID = "skill_mas_agent"
DEFAULT_RUN_ID = "exp1"
DEFAULT_VAL_SIZE = 32
DEFAULT_EVOLVE_ROUNDS = 10
DEFAULT_AGENT_LLM = "qwen3.5-plus"
DEFAULT_OPTIMIZER_LLM = "qwen3.5-plus"
DEFAULT_MAX_CONCURRENCY = 16
DEFAULT_FRESH = False

DEFAULT_DOMAIN = "delivery,instore,ota"
DEFAULT_USER_LLM = "qwen3.5-plus"
DEFAULT_EVALUATOR_LLM = "qwen3-max"
DEFAULT_MAX_STEPS = 300
DEFAULT_LANGUAGE = "chinese"
os.environ["DRB_RACE_MODEL"] = DEFAULT_EVALUATOR_LLM

OPTIMIZER_SUMMARY_JSON_MAX_CHARS = 12000
OPTIMIZER_WORST_TASK_IDS_COUNT = 5
OPTIMIZER_WORST_TASKS_FULL_ROUTER = 5
OPTIMIZER_ROUTER_PREVIEW_SHORT = 400
OPTIMIZER_ROUTER_PREVIEW_LONG = 2000
OPTIMIZER_RUBRIC_LINE_MAX = 300
OPTIMIZER_INTERNAL_TRACE_MAX_ITEMS = 8
OPTIMIZER_INTERNAL_TRACE_ENTRY_CHARS = 120

DRB_BENCH_ROOT = DRB_ROOT
DRB_VALIDATE_JSONL = DRB_BENCH_ROOT / "data" / "drb_validate.jsonl"
DEFAULT_DRB_RACE_MAX_WORKERS = 16

HLEMATH_JSONL_DEFAULT = HLEMATH_ROOT / "data" / "hlemath_validate.jsonl"
BROWSECOMP_VALIDATE_JSONL = BCP_ROOT / "data" / "browsecomp_plus_validate.jsonl"
BROWSECOMP_BENCH_ROOT = BCP_ROOT
VITA_VALIDATE_JSON = VITABENCH_ROOT / "data" / "vita_validate.json"

EVOLVE_K_TRAJECTORIES = 5
EVOLVE_MAX_REFLECTION_CASES_PER_ROUND = 128

ROUND_TRAJECTORIES_DIRNAME = "trajectories"
ROUND_ASPECTS_DIRNAME = "aspects"
ROUND_CONTRASTIVE_DIRNAME = "contrastive"
# Native benchmark runner output (DRB/HLEMATH/BCP); keeps round_XX top-level aligned with VitaBench.
ROUND_BENCH_ROLLOUTS_DIRNAME = "bench_rollouts"
ROUND_PATCH_POOL_FILENAME = "patch_pool.jsonl"

SCHEMA_TRAJECTORY_RECORD = "skill_mas_trajectory_record_v1"
SCHEMA_CONTRASTIVE_REPORT = "skill_mas_contrastive_report_v1"
SCHEMA_DOMAIN_PATCH = "skill_mas_domain_patch_v1"
SCHEMA_ROUND_ARTIFACT_INDEX = "skill_mas_round_artifact_index_v1"
SCHEMA_WINNER = "skill_mas_best_round_selection_v1"


def sanitize_run_id(raw: str) -> str:
    s = raw.strip().replace("\\", "_").replace("/", "_")
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    s = s.strip("._-")
    if not s:
        s = DEFAULT_RUN_ID
    return s[:120]


def allocate_run_id(
    bench_id: str,
    preferred: str | None = None,
    *,
    bench_backend: str = BENCH_SEGMENT_VITABENCH,
    agent_llm: str | None = None,
) -> str:
    base = sanitize_run_id(preferred) if preferred and preferred.strip() else DEFAULT_RUN_ID
    rid = base
    n = 2
    while _run_id_paths_exist(bench_id, rid, bench_backend, agent_llm=agent_llm):
        rid = f"{base}_{n}"
        n += 1
    return rid


def _run_id_names_under_bench(bench_id: str, bench_backend: str, agent_llm: str | None = None) -> set[str]:
    out: set[str] = set()
    for root in (
        skills_evolution_dir(bench_backend, agent_llm),
        log_root_for(bench_backend, agent_llm),
        runs_dir(bench_backend, agent_llm),
        merged_workspaces_dir(bench_backend, agent_llm),
        optimizer_validate_dir(bench_backend, agent_llm),
    ):
        d = root / bench_id
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.is_dir():
                out.add(p.name)
    return out


def _next_fresh_run_id(
    bench_id: str, base: str, bench_backend: str, agent_llm: str | None = None
) -> str:
    names = _run_id_names_under_bench(bench_id, bench_backend, agent_llm)
    pat = re.compile(rf"^{re.escape(base)}_(\d+)$")
    max_n = 0
    for name in names:
        m = pat.match(name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return sanitize_run_id(f"{base}_{max_n + 1}")


def resolve_run_id(
    bench_id: str,
    preferred: str | None,
    *,
    fresh: bool,
    bench_backend: str = BENCH_SEGMENT_VITABENCH,
    agent_llm: str | None = None,
) -> str:
    if fresh:
        base = (
            sanitize_run_id(preferred)
            if preferred and preferred.strip()
            else sanitize_run_id(DEFAULT_RUN_ID)
        )
        return _next_fresh_run_id(bench_id, base, bench_backend, agent_llm)
    base = sanitize_run_id(preferred) if preferred and preferred.strip() else DEFAULT_RUN_ID
    run_root = skills_evolution_dir(bench_backend, agent_llm) / bench_id / base
    if run_root.is_dir() and (run_root / "round_00").is_dir():
        return base
    return allocate_run_id(bench_id, preferred, bench_backend=bench_backend, agent_llm=agent_llm)


def _run_id_paths_exist(
    bench_id: str, run_id: str, bench_backend: str, agent_llm: str | None = None
) -> bool:
    return (
        (skills_evolution_dir(bench_backend, agent_llm) / bench_id / run_id).exists()
        or (log_root_for(bench_backend, agent_llm) / bench_id / run_id).exists()
        or (runs_dir(bench_backend, agent_llm) / bench_id / run_id).exists()
        or (merged_workspaces_dir(bench_backend, agent_llm) / bench_id / run_id).exists()
        or (optimizer_validate_dir(bench_backend, agent_llm) / bench_id / run_id).exists()
    )


def resolve_init_skill_root(explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.resolve()
        if p.is_file() and p.name == "SKILL.md":
            return p.parent
        if not p.is_dir():
            raise FileNotFoundError(p)
        if not (p / "SKILL.md").is_file():
            raise FileNotFoundError(f"Expected SKILL.md under init skill root: {p}")
        return p
    if (INIT_SKILL_DIR / "SKILL.md").is_file():
        return INIT_SKILL_DIR.resolve()
    raise FileNotFoundError(
        f"Expected SKILL.md under {INIT_SKILL_DIR}. "
        "Pass --init-skill-root /path/to/dir containing SKILL.md."
    )
