"""Structured payload types for Skill-MAS evolution data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PhaseSnapshot:
    phase: str
    instruction: str = ""
    output_preview: str = ""


@dataclass
class TrajectoryRecord:
    schema: str
    bench_backend: str
    round_idx: int
    task_id: str
    trajectory_idx: int
    trajectory_tag: str
    score: float
    score_source: str
    log_path: str
    raw_result_path: str
    phase_snapshots: list[PhaseSnapshot] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["phase_snapshots"] = [asdict(x) for x in self.phase_snapshots]
        return out


@dataclass
class DomainPatch:
    schema: str
    task_id: str
    phase: str
    constraint: str
    rationale: str
    source_gap: float
    source_high_traj: str
    source_low_traj: str
    frequency: int = 1
    status: str = "candidate"

    def key(self) -> tuple[str, str]:
        return (self.phase.strip().lower(), self.constraint.strip().lower())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

