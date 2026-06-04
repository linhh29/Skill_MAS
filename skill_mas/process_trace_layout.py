"""
Disk layout for Skill-MAS process traces aligned with dataset/vitabench
(``results/<model>/skill_mas_process_traces/<skill_suffix>/sample_logs/0000__<task_id>_t0/``).

Per-sample MAS trajectory JSON files live under ``sample_trace_json/`` (parallel to ``sample_logs/``).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Collects ``0000__<task>_t0.json`` files next to ``sample_logs/`` (not loose in trace root).
SAMPLE_TRACE_JSON_DIRNAME = "sample_trace_json"


def sample_trace_json_dir(process_trace_dir: Path | str) -> Path:
    return Path(process_trace_dir).resolve() / SAMPLE_TRACE_JSON_DIRNAME


def iter_per_sample_trace_json_files(process_trace_dir: Path | str):
    """
    Yield per-sample ``*.json`` trace payloads.

    Prefers ``<process_trace_dir>/sample_trace_json/*.json`` when that folder has traces;
    otherwise falls back to legacy flat layout (root ``*.json``).
    """
    root = Path(process_trace_dir).resolve()
    nested = root / SAMPLE_TRACE_JSON_DIRNAME
    if nested.is_dir():
        nested_files = [f for f in sorted(nested.glob("*.json")) if not f.name.startswith("_")]
        if nested_files:
            yield from nested_files
            return
    for f in sorted(root.glob("*.json")):
        if f.name.startswith("_"):
            continue
        yield f


def sanitize_component(name: str) -> str:
    s = (name or "").split("/")[-1]
    out = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return out or "unknown"


def skill_workspace_from_init_path(init_path: Path | str) -> Path:
    """Skill workspace directory: parent of ``SKILL.md`` when init path is a file, else the path itself."""
    p = Path(init_path).resolve()
    if p.is_file():
        return p.parent
    return p


def skill_mas_trace_suffix_parts(workspace: Path | str) -> tuple[str, ...]:
    """
    Path segments under ``skill_mas_process_traces/``, matching
    ``dataset/vitabench`` ``results_mirror.skill_mas_trace_suffix`` + multi-segment dirs.
    """
    default = ("default",)
    path = str(Path(workspace).resolve())
    mark = "/skills/"
    if mark in path:
        rel = path.split(mark, 1)[1].strip("/")
        if not rel:
            return default
        return tuple(p for p in rel.split("/") if p)
    base = Path(path).name
    safe = sanitize_component(base)
    return (safe if safe else "default",)


def skill_mas_sample_log_subdir(
    index: int,
    task_id: str | int,
    *,
    trial: int | None = 0,
) -> str:
    """One folder name under ``sample_logs/`` (vitabench ``_skill_mas_sample_subdir_for_sim``)."""
    tid = str(task_id)
    base = sanitize_component(tid)
    if len(base) > 100:
        base = hashlib.sha256(tid.encode("utf-8")).hexdigest()[:20]
    tr_part = f"_t{int(trial)}" if trial is not None else ""
    return f"{int(index):04d}__{base}{tr_part}"
