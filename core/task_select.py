"""Task id selection for Skill-MAS validation datasets."""

from __future__ import annotations

import json
from pathlib import Path

def _load_jsonl_ids(path: Path) -> list[str]:
    out: list[str] = []
    p = Path(path).resolve()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            task_id = row.get("id")
            if task_id is None:
                continue
            out.append(str(task_id))
    return out


def vitabench_validate_ids(validate_file: Path) -> list[str]:
    p = Path(validate_file).resolve()
    if p.suffix.lower() == ".jsonl":
        return _load_jsonl_ids(p)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [str(x) for x in payload.get("ids", [])]
    if isinstance(payload, list):
        out: list[str] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            task_id = row.get("id")
            if task_id is not None:
                out.append(str(task_id))
        return out
    return []


def drb_validate_ids(jsonl_file: Path) -> list[str]:
    p = Path(jsonl_file).resolve()
    return _load_jsonl_ids(p)


def browsecomp_validate_ids(jsonl_file: Path) -> list[str]:
    p = Path(jsonl_file).resolve()
    return _load_jsonl_ids(p)


def hlemath_validate_ids(jsonl_file: Path) -> list[str]:
    p = Path(jsonl_file).resolve()
    out: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                out.append(str(idx))
    return out


def val_ids_from_split(split_file: Path) -> list[str]:
    data = json.loads(Path(split_file).read_text(encoding="utf-8"))
    if data.get("ids") is not None:
        return [str(x) for x in data["ids"]]
    return [str(x) for x in (data.get("val_ids") or [])]


def test_ids_from_split(split_file: Path) -> list[str]:
    data = json.loads(Path(split_file).read_text(encoding="utf-8"))
    if data.get("ids") is not None:
        return [str(x) for x in data["ids"]]
    return [str(x) for x in (data.get("test_ids") or [])]

