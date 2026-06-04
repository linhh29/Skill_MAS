"""Step 4: assemble completed protocol and select best round."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.config import (
    SCHEMA_WINNER,
)
from .schemas import TrajectoryRecord


def compute_round_score(by_task: dict[str, list[TrajectoryRecord]]) -> float:
    """Mean trajectory score per task, then arithmetic mean across tasks.

    For each task id, aggregate **all** trajectories with the average of ``TrajectoryRecord.score``,
    then ``round_score`` is the average of those per-task means (equal weight per task).
    """
    vals: list[float] = []
    for rows in by_task.values():
        if not rows:
            continue
        vals.append(sum(float(r.score) for r in rows) / len(rows))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def update_round_scoreboard(
    *,
    runs_root: Path,
    round_idx: int,
    round_score: float,
    skill_path: Path,
    skill_round_path: Path,
) -> dict[str, Any]:
    scoreboard = runs_root / "round_scores.json"
    payload: dict[str, Any] = {"rounds": []}
    if scoreboard.is_file():
        payload = json.loads(scoreboard.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {"rounds": []}
    rounds = [r for r in (payload.get("rounds") or []) if int(r.get("round_idx", -1)) != int(round_idx)]
    rounds.append(
        {
            "round_idx": round_idx,
            "round_score": round_score,
            "skill_path": str(skill_path),
            "skill_round_path": str(skill_round_path),
        }
    )
    rounds.sort(key=lambda x: int(x["round_idx"]))
    payload["rounds"] = rounds
    scoreboard.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def finalize_best_round(runs_root: Path) -> dict[str, Any]:
    runs_root.mkdir(parents=True, exist_ok=True)

    def _round_skill_complexity(skill_round_path_str: str) -> int:
        p = Path(skill_round_path_str or "")
        if not p.is_dir():
            return 10**9
        return 1 if (p / "SKILL.md").is_file() else 10**9

    def _round_stability_risk(round_idx: int) -> int:
        _ = round_idx
        return 0

    scoreboard = runs_root / "round_scores.json"
    if not scoreboard.is_file():
        winner = {
            "schema": SCHEMA_WINNER,
            "best_round_idx": 0,
            "best_round_score": 0.0,
            "skill_path": "",
            "skill_round_path": "",
        }
        (runs_root / "winner.json").write_text(json.dumps(winner, ensure_ascii=False, indent=2), encoding="utf-8")
        return winner
    payload = json.loads(scoreboard.read_text(encoding="utf-8"))
    rounds = list(payload.get("rounds") or [])
    if not rounds:
        winner = {
            "schema": SCHEMA_WINNER,
            "best_round_idx": 0,
            "best_round_score": 0.0,
            "skill_path": "",
            "skill_round_path": "",
        }
    else:
        # Selection rule:
        # 1) Higher score first
        # 2) In ties, prefer lower phase-bank complexity
        # 3) If still tied, prefer lower stability risk
        def _rank_key(row: dict[str, Any]) -> tuple[float, int, int]:
            ridx = int(row.get("round_idx", 0))
            round_path = str(row.get("skill_round_path") or row.get("hooks_path") or "")
            complexity = _round_skill_complexity(round_path)
            stability_risk = _round_stability_risk(ridx)
            return (
                float(row.get("round_score", 0.0) or 0.0),
                -complexity,
                -stability_risk,
            )

        best = max(rounds, key=_rank_key)
        best_round_idx = int(best.get("round_idx", 0))
        best_round_path = str(best.get("skill_round_path") or best.get("hooks_path") or "")
        best_complexity = _round_skill_complexity(best_round_path)
        best_stability_risk = _round_stability_risk(best_round_idx)
        winner = {
            "schema": SCHEMA_WINNER,
            "best_round_idx": best_round_idx,
            "best_round_score": float(best.get("round_score", 0.0) or 0.0),
            "skill_path": str(best.get("skill_path") or ""),
            "skill_round_path": best_round_path,
            "tie_break": {
                "phase_bank_complexity": best_complexity,
                "stability_risk": best_stability_risk,
            },
        }
    (runs_root / "winner.json").write_text(json.dumps(winner, ensure_ascii=False, indent=2), encoding="utf-8")
    return winner

