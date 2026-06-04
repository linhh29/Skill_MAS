"""Canonical filesystem paths for Skill_MAS and nested benchmarks."""

from __future__ import annotations

from pathlib import Path

# Skill_MAS/utils/paths.py -> parents[1] == Skill_MAS package root
SKILL_MAS_ROOT = Path(__file__).resolve().parents[1]
# Parent of Skill_MAS (e.g. arxiv_code/) — add to PYTHONPATH for ``import Skill_MAS``
PACKAGE_ROOT = SKILL_MAS_ROOT.parent

DATASET_ROOT = SKILL_MAS_ROOT / "dataset"
VITABENCH_ROOT = DATASET_ROOT / "vitabench"
VITA_SRC = VITABENCH_ROOT / "src"
HLEMATH_ROOT = DATASET_ROOT / "hlemath"
DRB_ROOT = DATASET_ROOT / "deep_research_bench"
BCP_ROOT = DATASET_ROOT / "BrowseComp-Plus"

MODEL_CONFIG_JSON = SKILL_MAS_ROOT / "skill_mas" / "model_config.json"
INIT_SKILL_DIR = SKILL_MAS_ROOT / "init_skill"
RESULTS_ROOT = SKILL_MAS_ROOT / "results"

# Back-compat alias used by older modules
REPO_ROOT = PACKAGE_ROOT
SKILL_MAS_HOME = SKILL_MAS_ROOT


def ensure_sys_path(*, include_vita: bool = False, include_dataset: bool = False, include_bcp: bool = False) -> None:
    """Insert canonical roots onto ``sys.path`` (idempotent)."""
    import sys

    roots = [PACKAGE_ROOT, SKILL_MAS_ROOT]
    if include_dataset:
        roots.append(DATASET_ROOT)
    if include_vita:
        roots.append(VITA_SRC)
    if include_bcp:
        roots.append(BCP_ROOT)
    for root in roots:
        s = str(root)
        if s not in sys.path:
            sys.path.insert(0, s)
