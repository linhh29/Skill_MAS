"""DRB runtime helpers hosted inside deep_research_bench.

This module is the in-repo replacement for former drb_bridge primitives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional
from Skill_MAS.skill_mas.openai_async_client import (
    enrich_usage_with_cost as _skillmas_enrich_usage_with_cost,
    estimate_cost_usd as _skillmas_estimate_cost_usd,
    load_pricing_table as _skillmas_load_pricing_table,
    normalize_usage_tokens as _skillmas_normalize_usage_tokens,
)


@dataclass(frozen=True)
class DRBTask:
    id: int
    prompt: str
    language: str
    topic: str = ""


@dataclass(frozen=True)
class DRBArticle:
    id: int
    prompt: str
    article: str


def validate_article(item: DRBArticle) -> None:
    if not isinstance(item.id, int):
        raise ValueError(f"id must be int, got {type(item.id)}")
    if not isinstance(item.prompt, str) or not item.prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(item.article, str) or not item.article.strip():
        raise ValueError("article must be a non-empty string")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_drb_tasks(query_file: str | Path) -> list[DRBTask]:
    rows = _read_jsonl(Path(query_file))
    out: list[DRBTask] = []
    for row in rows:
        out.append(
            DRBTask(
                id=int(row["id"]),
                prompt=str(row["prompt"]),
                language=str(row.get("language", "")),
                topic=str(row.get("topic", "")),
            )
        )
    return out


def write_drb_articles(articles: Iterable[DRBArticle], output_file: str | Path) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in articles:
            validate_article(item)
            f.write(
                json.dumps(
                    {"id": item.id, "prompt": item.prompt, "article": item.article},
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_pricing_table(path: Optional[Path] = None) -> dict[str, Any]:
    from Skill_MAS.utils.paths import MODEL_CONFIG_JSON

    p = path or MODEL_CONFIG_JSON
    return _skillmas_load_pricing_table(p)


def normalize_usage_tokens(usage: Mapping[str, Any] | None) -> tuple[int, int, int]:
    return _skillmas_normalize_usage_tokens(usage)


def _row_to_per_1m(row: Any) -> tuple[float, float] | None:
    if not isinstance(row, dict):
        return None
    if "input_per_1m" in row and "output_per_1m" in row:
        return float(row["input_per_1m"]), float(row["output_per_1m"])
    if "input_per_1k" in row and "output_per_1k" in row:
        return float(row["input_per_1k"]) * 1000.0, float(row["output_per_1k"]) * 1000.0
    return None


def estimate_cost_usd(
    *, model: str, usage: Mapping[str, Any] | None, table: Optional[Mapping[str, Any]] = None
) -> tuple[float, int, int, str]:
    return _skillmas_estimate_cost_usd(model=model, usage=usage, table=table)


def enrich_usage_with_cost(
    usage: MutableMapping[str, Any] | None, *, model: str, table: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    return _skillmas_enrich_usage_with_cost(usage, model=model, table=table)
