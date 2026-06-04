"""Persisted validation / test splits for Skill-MAS (seeded shuffle).

Files live under ``Skill_MAS/data/``:
  - ``*_validate.json`` — validation ids (and metadata)
  - ``*_test.json`` — held-out test ids (and metadata)

Naming matches ``vita_*`` / ``drb_*`` with ``_s{seed}_v{val_size}_`` infix so Vita/DRB runners can resolve the default test file without shell changes.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ..utils.config import DEFAULT_LANGUAGE

_REPO = Path(__file__).resolve().parents[2]
_DATA_ROOT = _REPO / "Skill_MAS" / "data"


def split_root() -> Path:
    return _DATA_ROOT


def _vita_base_name(task_set_name: str, language: str | None, *, seed: int, val_size: int) -> str:
    safe_set = task_set_name.replace("/", "_").replace(",", "-")[:80]
    safe_lang = (language or DEFAULT_LANGUAGE).replace("/", "_")[:32]
    return f"vita_{safe_set}_{safe_lang}_s{int(seed)}_v{int(val_size)}"


def vita_split_validate_path(task_set_name: str, language: str | None, *, seed: int = 0, val_size: int = 32) -> Path:
    return _DATA_ROOT / f"{_vita_base_name(task_set_name, language, seed=seed, val_size=val_size)}_validate.json"


def vita_split_test_path(task_set_name: str, language: str | None, *, seed: int = 0, val_size: int = 32) -> Path:
    return _DATA_ROOT / f"{_vita_base_name(task_set_name, language, seed=seed, val_size=val_size)}_test.json"


def _drb_base_name(query_file: Path, *, seed: int, val_size: int) -> str:
    stem = Path(query_file).stem[:80]
    return f"drb_{stem}_s{int(seed)}_v{int(val_size)}"


def drb_split_validate_path(query_file: Path, *, seed: int = 0, val_size: int = 32) -> Path:
    return _DATA_ROOT / f"{_drb_base_name(query_file, seed=seed, val_size=val_size)}_validate.json"


def drb_split_test_path(query_file: Path, *, seed: int = 0, val_size: int = 32) -> Path:
    return _DATA_ROOT / f"{_drb_base_name(query_file, seed=seed, val_size=val_size)}_test.json"


def build_vita_split(
    *,
    task_set_name: str,
    language: str | None,
    seed: int = 0,
    val_size: int = 32,
    force: bool = False,
) -> Path:
    """Shuffle all task ids; write ``*_validate.json`` and ``*_test.json``. Returns validate path."""
    import sys

    if str(_REPO / "vitabench_single" / "src") not in sys.path:
        sys.path.insert(0, str(_REPO / "vitabench_single" / "src"))
    from vita.run import load_tasks

    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    path_val = vita_split_validate_path(task_set_name, language, seed=seed, val_size=val_size)
    path_test = vita_split_test_path(task_set_name, language, seed=seed, val_size=val_size)
    tasks = load_tasks(task_set_name=task_set_name, language=language)
    if not tasks:
        raise RuntimeError(f"No tasks for task_set_name={task_set_name!r} language={language!r}")
    ids = [str(t.id) for t in tasks]
    rng = random.Random(int(seed))
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_val = min(int(val_size), len(shuffled))
    val_ids = shuffled[:n_val]
    test_ids = shuffled[n_val:]

    def _payload(*, split: str, id_list: list[str]) -> dict[str, Any]:
        peer = path_test.name if split == "validate" else path_val.name
        return {
            "split": split,
            "backend": "vitabench",
            "seed": int(seed),
            "val_size": int(val_size),
            "task_set_name": task_set_name,
            "language": language,
            "ids": id_list,
            "total_in_source": len(ids),
            "peer_split_file": peer,
        }

    pv = _payload(split="validate", id_list=val_ids)
    pt = _payload(split="test", id_list=test_ids)

    def _matches() -> bool:
        if not path_val.is_file() or not path_test.is_file():
            return False
        try:
            ev = json.loads(path_val.read_text(encoding="utf-8"))
            et = json.loads(path_test.read_text(encoding="utf-8"))
        except Exception:
            return False
        return ev.get("ids") == val_ids and et.get("ids") == test_ids

    if not force and _matches():
        return path_val

    path_val.write_text(json.dumps(pv, ensure_ascii=False, indent=2), encoding="utf-8")
    path_test.write_text(json.dumps(pt, ensure_ascii=False, indent=2), encoding="utf-8")
    return path_val


def build_drb_split(
    *,
    query_file: Path,
    seed: int = 0,
    val_size: int = 32,
    force: bool = False,
) -> Path:
    from deep_research_bench.drb_runtime import load_drb_tasks

    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    path_val = drb_split_validate_path(query_file, seed=seed, val_size=val_size)
    path_test = drb_split_test_path(query_file, seed=seed, val_size=val_size)
    tasks = load_drb_tasks(query_file)
    if not tasks:
        raise RuntimeError(f"No DRB tasks in {query_file}")
    ids = [int(t.id) for t in tasks]
    rng = random.Random(int(seed))
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_val = min(int(val_size), len(shuffled))
    val_ids = [str(x) for x in shuffled[:n_val]]
    test_ids = [str(x) for x in shuffled[n_val:]]

    def _payload(*, split: str, id_list: list[str]) -> dict[str, Any]:
        peer = path_test.name if split == "validate" else path_val.name
        return {
            "split": split,
            "backend": "drb",
            "seed": int(seed),
            "val_size": int(val_size),
            "query_file": str(Path(query_file).resolve()),
            "ids": id_list,
            "total_in_source": len(ids),
            "peer_split_file": peer,
        }

    pv = _payload(split="validate", id_list=val_ids)
    pt = _payload(split="test", id_list=test_ids)

    def _matches() -> bool:
        if not path_val.is_file() or not path_test.is_file():
            return False
        try:
            ev = json.loads(path_val.read_text(encoding="utf-8"))
            et = json.loads(path_test.read_text(encoding="utf-8"))
        except Exception:
            return False
        return ev.get("ids") == val_ids and et.get("ids") == test_ids

    if not force and _matches():
        return path_val

    path_val.write_text(json.dumps(pv, ensure_ascii=False, indent=2), encoding="utf-8")
    path_test.write_text(json.dumps(pt, ensure_ascii=False, indent=2), encoding="utf-8")
    return path_val


def load_split_file(path: Path) -> dict[str, Any]:
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Split file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _hlemath_jsonl_line_count(jsonl_path: Path) -> int:
    n = 0
    first = True
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            if first:
                first = False
                if raw.startswith("version ") and "git-lfs" in raw:
                    raise RuntimeError(
                        f"{jsonl_path} is a Git LFS pointer. Run `git lfs pull` or use real JSONL."
                    )
            n += 1
    return n


def _hlemath_base_name(jsonl_path: Path, *, seed: int, val_size: int) -> str:
    stem = Path(jsonl_path).stem[:80]
    return f"hlemath_{stem}_s{int(seed)}_v{int(val_size)}"


def hlemath_split_validate_path(jsonl_path: Path, *, seed: int = 0, val_size: int = 32) -> Path:
    return _DATA_ROOT / f"{_hlemath_base_name(jsonl_path, seed=seed, val_size=val_size)}_validate.json"


def hlemath_split_test_path(jsonl_path: Path, *, seed: int = 0, val_size: int = 32) -> Path:
    return _DATA_ROOT / f"{_hlemath_base_name(jsonl_path, seed=seed, val_size=val_size)}_test.json"


def build_hlemath_split(
    *,
    jsonl_path: Path,
    seed: int = 0,
    val_size: int = 32,
    force: bool = False,
) -> Path:
    """Shuffle line indices ``0..n-1``; write validate + test JSON. Returns validate path."""
    _DATA_ROOT.mkdir(parents=True, exist_ok=True)
    path_val = hlemath_split_validate_path(jsonl_path, seed=seed, val_size=val_size)
    path_test = hlemath_split_test_path(jsonl_path, seed=seed, val_size=val_size)
    n_lines = _hlemath_jsonl_line_count(Path(jsonl_path).resolve())
    if n_lines <= 0:
        raise RuntimeError(f"No JSONL rows in {jsonl_path}")
    ids = [str(i) for i in range(n_lines)]
    rng = random.Random(int(seed))
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_val = min(int(val_size), len(shuffled))
    val_ids = shuffled[:n_val]
    test_ids = shuffled[n_val:]

    def _payload(*, split: str, id_list: list[str]) -> dict[str, Any]:
        peer = path_test.name if split == "validate" else path_val.name
        return {
            "split": split,
            "backend": "hlemath",
            "seed": int(seed),
            "val_size": int(val_size),
            "jsonl_file": str(Path(jsonl_path).resolve()),
            "ids": id_list,
            "total_in_source": n_lines,
            "peer_split_file": peer,
        }

    pv = _payload(split="validate", id_list=val_ids)
    pt = _payload(split="test", id_list=test_ids)

    def _matches() -> bool:
        if not path_val.is_file() or not path_test.is_file():
            return False
        try:
            ev = json.loads(path_val.read_text(encoding="utf-8"))
            et = json.loads(path_test.read_text(encoding="utf-8"))
        except Exception:
            return False
        return ev.get("ids") == val_ids and et.get("ids") == test_ids

    if not force and _matches():
        return path_val

    path_val.write_text(json.dumps(pv, ensure_ascii=False, indent=2), encoding="utf-8")
    path_test.write_text(json.dumps(pt, ensure_ascii=False, indent=2), encoding="utf-8")
    return path_val


