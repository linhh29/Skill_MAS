"""Skill_MAS phase-bank layout helpers used by Skill_MAS and BrowseComp-Plus runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAS_PHASE_COUNT = 4


@dataclass(frozen=True)
class PhaseSkillSpec:
    """One SKILL.md unit discovered under ``phase_<k>/``."""

    name: str
    body: str
    path: Path


def is_phase_bank_layout(skill_dir: str | Path) -> bool:
    """Return True when directory follows phase_1..phase_N bank layout."""
    root = Path(skill_dir).resolve()
    if (root / "SKILL.md").is_file():
        return False
    return root.is_dir() and (root / "phase_1").is_dir()


def discover_phase_skill_banks(
    skill_dir: str | Path,
    *,
    skip_invalid: bool = True,
    max_phases: int | None = None,
) -> tuple[dict[int, list[PhaseSkillSpec]], list[str]]:
    """
    Load SKILL.md files under ``phase_1`` .. ``phase_<N>``.

    Each ``phase_k`` directory is scanned recursively for ``SKILL.md``.
    Skill ``name`` is derived from the path relative to ``phase_k`` (directory names),
    or the phase folder name when SKILL.md sits directly under ``phase_k``.
    """
    root = Path(skill_dir).resolve()
    limit = int(max_phases) if max_phases is not None else MAS_PHASE_COUNT
    issues: list[str] = []
    banks: dict[int, list[PhaseSkillSpec]] = {}

    for i in range(1, limit + 1):
        pdir = root / f"phase_{i}"
        if not pdir.is_dir():
            issues.append(f"missing directory: {pdir}")
            banks[i] = []
            continue
        specs: list[PhaseSkillSpec] = []
        for md in sorted(pdir.rglob("SKILL.md")):
            try:
                body = md.read_text(encoding="utf-8")
            except OSError as exc:
                msg = f"read failed {md}: {exc}"
                issues.append(msg)
                if not skip_invalid:
                    raise
                continue
            rel = md.parent.relative_to(pdir)
            if str(rel) == ".":
                name = md.parent.name
            else:
                name = "__".join(rel.parts) if rel.parts else md.parent.name
            specs.append(PhaseSkillSpec(name=name, body=body, path=md.resolve()))
        if not specs:
            issues.append(f"phase_{i}: no SKILL.md found under {pdir}")
        banks[i] = specs

    return banks, issues


def phase_bank_snapshot(
    skill_dir: str | Path,
    *,
    banks: dict[int, list[PhaseSkillSpec]],
    validation_issues: list[str],
) -> dict[str, Any]:
    """JSON-friendly summary for process traces."""
    root = Path(skill_dir).resolve()
    phases_out: dict[str, list[dict[str, str]]] = {}
    for k in sorted(banks.keys()):
        phases_out[str(k)] = [
            {"name": s.name, "path": str(s.path.resolve())} for s in banks[k]
        ]
    return {
        "skill_root": str(root),
        "mas_phase_count": MAS_PHASE_COUNT,
        "phases": phases_out,
        "validation_issues": list(validation_issues),
    }


__all__ = [
    "MAS_PHASE_COUNT",
    "PhaseSkillSpec",
    "discover_phase_skill_banks",
    "is_phase_bank_layout",
    "phase_bank_snapshot",
]

