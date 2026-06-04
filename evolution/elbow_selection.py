"""Elbow selection via second finite differences (Method C)."""

from __future__ import annotations

import math
from typing import Any


def _population_std(xs: list[float]) -> float:
    """Match ``numpy.std(row, ddof=0)`` for trajectory scores in one task."""
    n = len(xs)
    if n <= 1:
        return 0.0
    mean = sum(xs) / n
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / n)


def _normalize_minmax_1d(values: list[float]) -> list[float]:
    if not values:
        return []
    min_s = min(values)
    max_s = max(values)
    if max_s - min_s < 1e-8:
        return [0.5] * len(values)
    return [(v - min_s) / (max_s - min_s) for v in values]


def _priority_vectors(samples_scores: list[list[float]]) -> dict[str, list[float]]:
    """Raw + min-max normalized uncertainty/difficulty and blended priority per row."""
    uncertainties: list[float] = []
    difficulties: list[float] = []
    for row in samples_scores:
        if not row:
            uncertainties.append(0.0)
            difficulties.append(0.0)
            continue
        mean = sum(row) / len(row)
        difficulties.append(-mean)
        uncertainties.append(_population_std(row))

    u_norm = _normalize_minmax_1d(uncertainties)
    d_norm = _normalize_minmax_1d(difficulties)
    priorities = [(u_norm[i] + d_norm[i]) / 2.0 for i in range(len(uncertainties))]
    return {
        "uncertainties_raw": uncertainties,
        "difficulties_raw": difficulties,
        "uncertainties_normalized": u_norm,
        "difficulties_normalized": d_norm,
        "priorities": priorities,
    }


def compute_priority_scores(samples_scores: list[list[float]]) -> list[float]:
    """
    Per-task priority in ``[0, 1]``: blend uncertainty (std across trajectories)
    and difficulty (negative mean score — lower mean ⇒ harder ⇒ larger after norm).

    Aligns with::

        uncertainties = np.std(samples_scores, axis=1)
        difficulties = -np.mean(samples_scores, axis=1)
        uncertainties_norm = normalize(uncertainties)
        difficulties_norm = normalize(difficulties)
        priorities = (uncertainties_norm + difficulties_norm) / 2

    ``samples_scores[i]`` is the list of trajectory-level scores for task ``i``
    (same convention as one row of a ``(n_samples, n_trajectories)`` matrix).
    """
    if not samples_scores:
        return []
    return _priority_vectors(samples_scores)["priorities"]


def compute_reflection_task_selection(
    task_rows: list[tuple[str, list[float]]],
    *,
    max_reflection_cases: int,
    sensitivity: float = 1.0,
) -> tuple[list[str], dict[str, Any]]:
    """
    Stable sorted ``task_rows`` by ``task_id``, then compute per-task uncertainty /
    difficulty (raw + normalized) and priority, apply the same second-diff elbow +
    cap as Step 2 reflection selection.

    Returns selected task ids (priority-descending, truncated by elbow and
    ``max_reflection_cases``) and a JSON-serializable report for offline analysis.
    """
    rows = sorted(task_rows, key=lambda kv: str(kv[0]))
    if not rows:
        return [], {
            "schema": "skill_mas_priority_selection_v1",
            "selection_mode": "second_diff_elbow",
            "priority_definition": {
                "uncertainty": "population_std_of_trajectory_scores",
                "difficulty": "negative_mean_trajectory_score",
                "blend": "(minmax_norm(u)+minmax_norm(d))/2 within_round",
            },
            "normalization": "min_max_across_tasks_in_round",
            "sensitivity": float(sensitivity),
            "max_reflection_cases_cap": int(max(1, max_reflection_cases)),
            "tasks": [],
            "ranking_descending_priority": [],
            "elbow_method_detail": second_diff_elbow_detail([], sensitivity=sensitivity),
            "reflection_selected_task_ids": [],
            "reflection_selected_count": 0,
            "reflection_selected_tasks": [],
        }

    samples_scores = [s for _, s in rows]
    task_ids = [str(t) for t, _ in rows]
    vec = _priority_vectors(samples_scores)

    tasks_out: list[dict[str, Any]] = []
    for i, tid in enumerate(task_ids):
        sc = samples_scores[i]
        mean = sum(sc) / len(sc) if sc else 0.0
        tasks_out.append(
            {
                "task_id": tid,
                "num_trajectories": len(sc),
                "mean_score": float(mean),
                "trajectory_scores": [float(x) for x in sc],
                "uncertainty_raw": float(vec["uncertainties_raw"][i]),
                "difficulty_raw": float(vec["difficulties_raw"][i]),
                "uncertainty_normalized": float(vec["uncertainties_normalized"][i]),
                "difficulty_normalized": float(vec["difficulties_normalized"][i]),
                "priority": float(vec["priorities"][i]),
            }
        )

    ranked_indices = sorted(
        range(len(tasks_out)),
        key=lambda i: tasks_out[i]["priority"],
        reverse=True,
    )
    ranking_desc: list[dict[str, Any]] = []
    for rank, idx in enumerate(ranked_indices):
        entry = dict(tasks_out[idx])
        entry["rank"] = int(rank)
        ranking_desc.append(entry)

    prio_desc = [tasks_out[i]["priority"] for i in ranked_indices]
    elbow_detail = second_diff_elbow_detail(prio_desc, sensitivity=sensitivity)
    elbow_k = adaptive_elbow_count(prio_desc, sensitivity=sensitivity)
    cap = int(max(1, max_reflection_cases))
    elbow_k = min(elbow_k, cap, len(ranked_indices))

    selected_ids = [str(ranking_desc[r]["task_id"]) for r in range(elbow_k)]
    selected_slice = ranking_desc[:elbow_k]

    report: dict[str, Any] = {
        "schema": "skill_mas_priority_selection_v1",
        "selection_mode": "second_diff_elbow",
        "priority_definition": {
            "uncertainty": "population_std_of_trajectory_scores",
            "difficulty": "negative_mean_trajectory_score",
            "blend": "(minmax_norm(u)+minmax_norm(d))/2 within_round",
        },
        "normalization": "min_max_across_tasks_in_round",
        "sensitivity": float(sensitivity),
        "max_reflection_cases_cap": cap,
        "num_tasks": len(tasks_out),
        "tasks": tasks_out,
        "ranking_descending_priority": ranking_desc,
        "elbow_method_detail": elbow_detail,
        "reflection_selected_task_ids": selected_ids,
        "reflection_selected_count": int(elbow_k),
        "reflection_selected_tasks": selected_slice,
    }
    return selected_ids, report


def adaptive_elbow_count(sorted_scores_desc: list[float], sensitivity: float = 1.0) -> int:
    """
    Given scores sorted in **descending** order (largest first), estimate how many
    leading entries lie before the elbow using second differences.

    Mirrors::

        diffs = np.diff(sorted_scores)
        second_diffs = np.diff(diffs)
        elbow_idx = np.argmax(np.abs(second_diffs)) + 1
        count = int(elbow_idx * sensitivity)

    Returns a count in ``[1, len(sorted_scores_desc)]`` (or 0 if input empty).
    """
    scores = sorted_scores_desc
    n = len(scores)
    if n == 0:
        return 0
    if n <= 2:
        return n

    diffs = [scores[i] - scores[i + 1] for i in range(n - 1)]
    second_diffs = [diffs[i] - diffs[i + 1] for i in range(len(diffs) - 1)]
    if not second_diffs:
        return n

    argmax_i = max(range(len(second_diffs)), key=lambda i: abs(second_diffs[i]))
    elbow_idx = argmax_i + 1
    count = int(elbow_idx * float(sensitivity))
    count = max(1, min(n, count))
    return count


def adaptive_elbow_selection(scores: list[float], sensitivity: float = 1.0) -> list[int]:
    """
    Select indices of the largest scores using the elbow count on the descending curve.

    Equivalent to ``np.argsort(scores)[::-1][:count]`` with ``count`` from
    :func:`adaptive_elbow_count` applied to the sorted values.
    """
    n = len(scores)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    sorted_vals = [scores[i] for i in order]
    k = adaptive_elbow_count(sorted_vals, sensitivity=sensitivity)
    return order[:k]


def second_diff_elbow_detail(sorted_scores_desc: list[float], sensitivity: float = 1.0) -> dict:
    """
    Same cut as :func:`adaptive_elbow_count`, plus intermediates for logging/plots.

    ``sorted_scores_desc`` must be descending.
    """
    scores = sorted_scores_desc
    n = len(scores)
    out: dict = {
        "n": n,
        "diffs": [],
        "second_diffs": [],
        "second_diff_argmax_index": None,
        "elbow_idx_before_sensitivity": None,
        "sensitivity": float(sensitivity),
        "selected_count": 0,
    }
    if n == 0:
        return out
    if n <= 2:
        out["selected_count"] = n
        return out

    diffs = [scores[i] - scores[i + 1] for i in range(n - 1)]
    second_diffs = [diffs[i] - diffs[i + 1] for i in range(len(diffs) - 1)]
    out["diffs"] = diffs
    out["second_diffs"] = second_diffs

    if not second_diffs:
        out["selected_count"] = n
        return out

    argmax_i = max(range(len(second_diffs)), key=lambda i: abs(second_diffs[i]))
    elbow_idx = argmax_i + 1
    out["second_diff_argmax_index"] = int(argmax_i)
    out["elbow_idx_before_sensitivity"] = int(elbow_idx)
    count = int(elbow_idx * float(sensitivity))
    count = max(1, min(n, count))
    out["selected_count"] = int(count)
    return out
