"""Shared JSONL helpers for BrowseComp-Plus runners (not Skill_MAS-specific)."""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
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


def jsonl_stem_key(path: Path) -> str:
    return path.stem.replace("/", "_").replace("\\", "_")
