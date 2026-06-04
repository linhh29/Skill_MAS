#!/usr/bin/env python3
"""
Merge RACE per-sample metrics from multiple process-trace run directories.

For each sample id, keep the row from the source whose overall_score is highest
(ties: earlier --dirs entry wins). Writes aggregated race_result.txt matching
run_single_agent / skill_mas_agent_runner format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent

# Edit in file; use `python merge.py --dirs ...` to override.
DEFAULT_MERGE_DIRS: list[Path] = [
    _REPO_ROOT / "results/qwen3.5-plus/skill_mas_process_traces_0.4360",
    _REPO_ROOT / "results/qwen3.5-plus/skill_mas_process_traces",
]

DEFAULT_MERGE_OUT_DIR = _REPO_ROOT / "results/qwen3.5-plus/merged_max_overall/drb_test"


def _has_non_manifest_json(data_dir: Path) -> bool:
    from Skill_MAS.skill_mas.process_trace_layout import SAMPLE_TRACE_JSON_DIRNAME

    nested = data_dir / SAMPLE_TRACE_JSON_DIRNAME
    if nested.is_dir():
        for f in nested.glob("*.json"):
            if not f.name.startswith("_"):
                return True
    for f in data_dir.glob("*.json"):
        if not f.name.startswith("_"):
            return True
    return False


def resolve_dataset_dir(root: Path, dataset: str) -> Path:
    """Accept either .../run_root or .../run_root/<dataset> (e.g. drb_test)."""
    root = root.resolve()
    nested = root / dataset
    if (nested / "race" / "raw_results.jsonl").is_file():
        return nested
    if (root / "race" / "raw_results.jsonl").is_file():
        return root
    if _has_non_manifest_json(nested):
        return nested
    if _has_non_manifest_json(root):
        return root
    raise FileNotFoundError(
        f"Cannot resolve dataset dir under {root} (tried {dataset!r}): "
        "need race/raw_results.jsonl or per-task *.json traces."
    )


def row_from_race_eval(task_id: int, prompt: str, ev: dict[str, Any]) -> dict[str, Any] | None:
    if not ev.get("ok"):
        return {"id": task_id, "prompt": prompt, "error": ev.get("error", "judge_failed")}
    return {
        "id": task_id,
        "prompt": prompt,
        "comprehensiveness": float(ev.get("comprehensiveness", 0.0)),
        "insight": float(ev.get("insight", 0.0)),
        "instruction_following": float(ev.get("instruction_following", 0.0)),
        "readability": float(ev.get("readability", 0.0)),
        "overall_score": float(ev.get("overall_score", 0.0)),
    }


def load_from_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = int(obj["id"])
            out[tid] = obj
    return out


def load_from_trace_jsons(data_dir: Path) -> dict[int, dict[str, Any]]:
    from Skill_MAS.skill_mas.process_trace_layout import iter_per_sample_trace_json_files

    out: dict[int, dict[str, Any]] = {}
    for trace_file in iter_per_sample_trace_json_files(data_dir):
        try:
            obj = json.loads(trace_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            tid = int(obj.get("task_id"))
        except Exception:
            continue
        prompt = str(obj.get("prompt", "") or "")
        ev = obj.get("race_eval")
        if not isinstance(ev, dict):
            continue
        row = row_from_race_eval(tid, prompt, ev)
        if row is not None:
            out[tid] = row
    return out


def load_race_rows(data_dir: Path) -> dict[int, dict[str, Any]]:
    raw_path = data_dir / "race" / "raw_results.jsonl"
    if raw_path.is_file():
        return load_from_jsonl(raw_path)
    return load_from_trace_jsons(data_dir)


def overall_score(row: dict[str, Any]) -> float | None:
    if "error" in row:
        return None
    try:
        return float(row.get("overall_score", 0.0))
    except (TypeError, ValueError):
        return None


def merge_max_overall(
    sources: list[tuple[str, dict[int, dict[str, Any]]]],
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    """Pick best row per id by overall_score; ties favor earlier source in the list."""
    all_ids: set[int] = set()
    for _, rows in sources:
        all_ids |= set(rows.keys())

    merged: dict[int, dict[str, Any]] = {}
    winners: dict[int, str] = {}

    for tid in sorted(all_ids):
        best_row: dict[str, Any] | None = None
        best_score: float | None = None
        best_label = ""
        for label, rows in sources:
            if tid not in rows:
                continue
            row = rows[tid]
            s = overall_score(row)
            if s is None:
                continue
            if best_score is None or s > best_score:
                best_score = s
                best_row = dict(row)
                best_label = label
        if best_row is not None:
            merged[tid] = best_row
            winners[tid] = best_label

    return merged, winners


def write_race_summary(rows: dict[int, dict[str, Any]], path: Path) -> None:
    ok = [r for r in rows.values() if "error" not in r]
    if ok:
        comp = sum(float(r["comprehensiveness"]) for r in ok) / len(ok)
        insi = sum(float(r["insight"]) for r in ok) / len(ok)
        inst = sum(float(r["instruction_following"]) for r in ok) / len(ok)
        read = sum(float(r["readability"]) for r in ok) / len(ok)
        overall = sum(float(r["overall_score"]) for r in ok) / len(ok)
    else:
        comp = insi = inst = read = overall = 0.0
    path.write_text(
        "\n".join(
            [
                f"Comprehensiveness: {comp:.4f}",
                f"Insight: {insi:.4f}",
                f"Instruction Following: {inst:.4f}",
                f"Readability: {read:.4f}",
                f"Overall Score: {overall:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dirs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_MERGE_DIRS),
        help="Process trace roots; defaults to DEFAULT_MERGE_DIRS in this file.",
    )
    p.add_argument(
        "--dataset",
        default="drb_test",
        help="Dataset subdirectory when race files live under <root>/<dataset>/... (default: drb_test).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_MERGE_OUT_DIR,
        help="Output directory (raw_results.jsonl, race_result.txt, merge_manifest.json).",
    )
    args = p.parse_args()

    dir_list: list[Path] = list(args.dirs) if args.dirs else list(DEFAULT_MERGE_DIRS)

    sources: list[tuple[str, dict[int, dict[str, Any]]]] = []
    for d in dir_list:
        root = d.expanduser()
        data_dir = resolve_dataset_dir(root, args.dataset)
        label = str(root)
        rows = load_race_rows(data_dir)
        sources.append((label, rows))

    merged, winners = merge_max_overall(sources)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for tid in sorted(merged):
            f.write(json.dumps(merged[tid], ensure_ascii=False) + "\n")

    write_race_summary(merged, out_dir / "race_result.txt")

    per_id: dict[str, Any] = {}
    for tid in sorted(merged):
        per_id[str(tid)] = {
            "winner": winners[tid],
            "overall_score": float(merged[tid].get("overall_score", 0.0)) if "error" not in merged[tid] else None,
            "candidates": {
                label: (overall_score(rows[tid]) if tid in rows else None) for label, rows in sources
            },
        }

    manifest = {
        "dataset": args.dataset,
        "sources": [s[0] for s in sources],
        "num_samples": len(merged),
        "per_id": per_id,
    }
    (out_dir / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    n_ok = sum(1 for r in merged.values() if "error" not in r)
    print(f"[merge] wrote {len(merged)} rows ({n_ok} scored) to {raw_path}")
    print(f"[merge] summary -> {out_dir / 'race_result.txt'}")


if __name__ == "__main__":
    main()
