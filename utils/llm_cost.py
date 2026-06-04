"""LLM cost aggregation using ``Skill_MAS/skill_mas/model_config.json`` (USD per 1M tokens)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Skill_MAS.skill_mas.openai_async_client import (
    _row_to_per_1m,
    enrich_usage_with_cost,
    load_pricing_table,
    normalize_usage_tokens,
)

# Canonical location: Skill_MAS/utils/llm_cost.py
_SKILL_MAS_ROOT = Path(__file__).resolve().parents[1]
MODEL_PRICING_JSON = _SKILL_MAS_ROOT / "skill_mas" / "model_config.json"


class PricingModelError(ValueError):
    """Raised when a model id cannot be matched to ``Skill_MAS/model_pricing.json`` (strict, no silent default)."""

    def __init__(
        self,
        message: str,
        *,
        model: str,
        reason: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.reason = reason
        self.context = context or {}


def _skill_mas_pricing_table() -> dict[str, Any]:
    """Load the pricing table shipped with Skill_MAS."""
    return load_pricing_table(MODEL_PRICING_JSON)


def strict_resolve_pricing_model_key(model: str, table: dict[str, Any]) -> str:
    """
    Resolve ``model`` to a JSON key with valid rates. Does **not** fall back to the ``default`` row
    when the name is unknown — that avoids silent wrong USD totals. Add the model to ``model_config.json``.
    """
    key = (model or "").strip()
    if not key:
        raise PricingModelError(
            "Skill_MAS pricing: empty model id for a usage record.",
            model="",
            reason="empty_model_id",
        )
    if key in table:
        rates = _row_to_per_1m(table[key])
        if rates:
            return key
    lower_map = {str(k).lower(): k for k in table if not str(k).startswith("_")}
    lk = key.lower()
    if lk in lower_map:
        k = lower_map[lk]
        if _row_to_per_1m(table[k]):
            return str(k)
    allowed = sorted(str(k) for k in table if not str(k).startswith("_"))
    raise PricingModelError(
        f"Skill_MAS pricing: model {key!r} is not listed in {MODEL_PRICING_JSON}. "
        f"Add a row for this id (or an alias). Keys in file: {allowed}",
        model=key,
        reason="model_not_in_pricing_json",
        context={"allowed_keys": allowed, "pricing_file": str(MODEL_PRICING_JSON.resolve())},
    )


def enrich_usage_with_cost_strict(
    usage: dict[str, Any] | None,
    *,
    model: str,
    table: dict[str, Any],
) -> dict[str, Any]:
    resolved = strict_resolve_pricing_model_key(model, table)
    return enrich_usage_with_cost(dict(usage or {}), model=resolved, table=table)


def validate_evolve_config_models(
    *,
    agent_llm: str,
    user_llm: str,
    evaluator_llm: str,
    optimizer_llm: str,
    judge_llm: str | None = None,
    table: dict[str, Any] | None = None,
) -> None:
    """Ensure CLI/config models exist in pricing config before any API spend."""
    t = table if table is not None else _skill_mas_pricing_table()
    seen: list[tuple[str, str]] = [
        ("agent_llm", agent_llm),
        ("user_llm", user_llm),
        ("evaluator_llm", evaluator_llm),
        ("optimizer_llm", optimizer_llm),
    ]
    j = (judge_llm or "").strip()
    if j:
        seen.append(("judge_llm", j))
    for role, mid in seen:
        m = (mid or "").strip()
        if not m:
            raise PricingModelError(
                f"Skill_MAS pricing: {role} is empty; set a model id present in {MODEL_PRICING_JSON}.",
                model="",
                reason=f"empty_{role}",
                context={"role": role},
            )
        try:
            strict_resolve_pricing_model_key(m, t)
        except PricingModelError as e:
            raise PricingModelError(
                str(e),
                model=e.model,
                reason=e.reason,
                context={**e.context, "role": role},
            ) from e


def record_pricing_error(
    *,
    runs_dir: Path,
    bench_id: str,
    run_id: str,
    exc: BaseException,
    round_idx: int | None = None,
    log_round: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``pricing_error.json`` under the run (and copy under ``log_round`` if given)."""
    root = runs_dir / bench_id / run_id
    root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "skill_mas_pricing_error_v1",
        "error": True,
        "estimated_cost_usd_recorded": None,
        "note": "No valid USD cost recorded — fix model_pricing.json and rerun.",
        "pricing_file": str(MODEL_PRICING_JSON.resolve()),
        "round_idx": round_idx,
    }
    if extra:
        payload.update(extra)
    if isinstance(exc, PricingModelError):
        payload["reason"] = exc.reason
        payload["model"] = exc.model
        payload["message"] = str(exc)
        payload.update(exc.context)
    else:
        payload["reason"] = "unexpected_error"
        payload["model"] = ""
        payload["message"] = repr(exc)
    path = root / "pricing_error.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if log_round is not None:
        try:
            log_round.mkdir(parents=True, exist_ok=True)
            (log_round / "pricing_error.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
    return path


def pricing_reference() -> dict[str, Any]:
    return {
        "pricing_file": str(MODEL_PRICING_JSON.resolve()),
        "currency": "USD",
        "unit": "per_1M_prompt_and_output_tokens",
        "note": "Strict: every billed model id must appear in this file (no silent default row for unknown names).",
    }


def empty_usage_totals() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def add_usage_totals(dst: dict[str, Any], src: Any) -> None:
    """Merge token counts and add ``estimated_cost_usd`` (DRB-style dicts)."""
    s = dict(src or {})
    dst["prompt_tokens"] = int(dst.get("prompt_tokens", 0) or 0) + int(
        s.get("prompt_tokens", 0) or 0
    )
    out = int(s.get("output_tokens", s.get("completion_tokens", 0)) or 0)
    dst["output_tokens"] = int(dst.get("output_tokens", 0) or 0) + out
    dst["total_tokens"] = int(dst.get("total_tokens", 0) or 0) + int(
        s.get("total_tokens", 0) or 0
    )
    dst["estimated_cost_usd"] = float(dst.get("estimated_cost_usd", 0.0) or 0.0) + float(
        s.get("estimated_cost_usd", 0.0) or 0.0
    )


def _resolve_vita_message_model(
    m: dict[str, Any],
    *,
    agent_llm: str,
    user_llm: str,
    evaluator_llm: str,
) -> str:
    raw = m.get("raw_data")
    if isinstance(raw, dict):
        api_m = raw.get("model")
        if isinstance(api_m, str) and api_m.strip():
            return api_m.strip()
    role = m.get("role")
    if role == "user":
        return user_llm
    if role == "assistant":
        return agent_llm
    return agent_llm


def vita_rollout_cost_report(
    results_path: Path,
    *,
    agent_llm: str,
    user_llm: str,
    evaluator_llm: str,
    table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Walk VitaBench simulation JSON and sum per-message usage with ``model_pricing.json``.

    Model id per message: API ``raw_data.model`` when present, else role →
    user / assistant (agent). Evaluator calls are usually tagged with ``model`` in ``raw_data``.
    """
    t = table if table is not None else _skill_mas_pricing_table()
    data = json.loads(results_path.read_text(encoding="utf-8"))
    sims = data.get("simulations") or []
    grand = empty_usage_totals()
    per_sim: list[dict[str, Any]] = []

    for sim in sims:
        acc = empty_usage_totals()
        msgs = sim.get("messages") or []
        per_msg: list[dict[str, Any]] = []
        for m in msgs:
            if not isinstance(m, dict):
                continue
            usage = m.get("usage")
            if not usage:
                continue
            mid = _resolve_vita_message_model(
                m, agent_llm=agent_llm, user_llm=user_llm, evaluator_llm=evaluator_llm
            )
            try:
                enriched = enrich_usage_with_cost_strict(dict(usage), model=mid, table=t)
            except PricingModelError as e:
                raise PricingModelError(
                    f"{e} (task_id={sim.get('task_id')!r}, message_role={m.get('role')!r}, "
                    f"resolved_model={mid!r})",
                    model=e.model,
                    reason=e.reason,
                    context={
                        **e.context,
                        "task_id": sim.get("task_id"),
                        "message_role": m.get("role"),
                        "resolved_model": mid,
                    },
                ) from e
            add_usage_totals(acc, enriched)
            p, o, _ = normalize_usage_tokens(enriched)
            per_msg.append(
                {
                    "role": m.get("role"),
                    "model_resolved": mid,
                    "pricing_model_key": enriched.get("pricing_model_key"),
                    "prompt_tokens": p,
                    "output_tokens": o,
                    "estimated_cost_usd": enriched.get("estimated_cost_usd", 0.0),
                }
            )
        add_usage_totals(grand, acc)
        per_sim.append(
            {
                "task_id": sim.get("task_id"),
                "usage_model_pricing_json": acc,
                "vita_reported_agent_cost": sim.get("agent_cost"),
                "vita_reported_user_cost": sim.get("user_cost"),
                "per_message_llm": per_msg,
            }
        )

    return {
        "phase": "vitabench_skill_mas_rollout",
        "description": "VitaBench trajectories: each LLM turn with usage, priced via model_pricing.json.",
        "models_config": {
            "agent_llm": agent_llm,
            "user_llm": user_llm,
            "evaluator_llm": evaluator_llm,
        },
        "aggregate_usage_model_pricing_json": grand,
        "per_simulation": per_sim,
        "notes": [
            "Per message: prefer API ``raw_data.model`` for pricing key; if missing, "
            "``user`` → user_llm and ``assistant`` → agent_llm (evaluator LLM calls are usually tagged by the API).",
            "Compare ``vita_reported_*_cost`` (Vita models.yaml) vs ``usage_model_pricing_json`` (this file).",
        ],
        "pricing_reference": pricing_reference(),
    }


def optimizer_call_report(
    *,
    phase: str,
    model: str,
    usage: dict[str, Any] | None,
    table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single optimizer LLM call (contrastive/bank), repriced via ``model_config.json``."""
    t = table if table is not None else _skill_mas_pricing_table()
    enriched = enrich_usage_with_cost_strict(dict(usage or {}), model=model, table=t)
    p, o, tot = normalize_usage_tokens(enriched)
    return {
        "phase": phase,
        "model": model,
        "usage": enriched,
        "prompt_tokens": p,
        "output_tokens": o,
        "total_tokens": tot,
        "estimated_cost_usd": float(enriched.get("estimated_cost_usd", 0.0) or 0.0),
        "pricing_model_key": enriched.get("pricing_model_key"),
    }


def build_round_cost_document(
    *,
    bench_backend: str,
    bench_id: str,
    run_id: str,
    round_idx: int,
    eval_section: dict[str, Any],
    optimizer_sections: list[dict[str, Any]],
    extra_notes: list[str] | None = None,
) -> dict[str, Any]:
    sections = [s for s in [eval_section, *optimizer_sections] if s]
    total = 0.0
    for s in sections:
        if "aggregate_usage_model_pricing_json" in s:
            total += float(
                (s.get("aggregate_usage_model_pricing_json") or {}).get(
                    "estimated_cost_usd", 0.0
                )
                or 0.0
            )
        elif "aggregate_usage" in s:
            total += float((s.get("aggregate_usage") or {}).get("estimated_cost_usd", 0.0) or 0.0)
        elif (
            isinstance(s.get("usage"), dict)
            and "model" in s
            and "estimated_cost_usd" in s
        ):
            # ``optimizer_call_report`` (contrastive_reflection_*, bank_optimizer_*, …) — not
            # ``optimizer_*`` phase prefix; the old ``startswith("optimizer_")`` branch missed these.
            total += float(s.get("estimated_cost_usd", 0.0) or 0.0)
        elif str(s.get("phase", "")).startswith("optimizer_"):
            total += float(s.get("estimated_cost_usd", 0.0) or 0.0)

    out: dict[str, Any] = {
        "schema": "skill_mas_evolve_llm_cost_round_v1",
        "bench_backend": bench_backend,
        "bench_id": bench_id,
        "run_id": run_id,
        "round_idx": round_idx,
        "pricing_reference": pricing_reference(),
        "sections": sections,
        "round_total_estimated_cost_usd": round(total, 10),
    }
    if extra_notes:
        out["notes"] = extra_notes
    return out


def finalize_round_cost_artifacts(
    *,
    runs_dir: Path,
    log_round: Path,
    bench_id: str,
    run_id: str,
    round_idx: int,
    payload: dict[str, Any],
) -> Path:
    """Write ``evolve_llm_cost_rXX.json`` under artifacts runs + log round; update cumulative + manifest."""
    import shutil

    name = f"evolve_llm_cost_r{round_idx:02d}.json"
    dest_run = runs_dir / bench_id / run_id / name
    dest_run.parent.mkdir(parents=True, exist_ok=True)
    dest_run.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    dest_log = log_round / name
    shutil.copy2(dest_run, dest_log)
    cum_path = runs_dir / bench_id / run_id / "evolve_llm_cost_cumulative.json"
    merge_cumulative_summary(
        cum_path,
        round_idx=round_idx,
        round_file_name=name,
        round_total_usd=float(payload.get("round_total_estimated_cost_usd", 0.0) or 0.0),
    )
    man = log_round / "manifest.json"
    if man.is_file():
        m = json.loads(man.read_text(encoding="utf-8"))
        m["evolve_llm_cost_json"] = str(dest_log.resolve())
        m["round_total_estimated_cost_usd"] = payload.get("round_total_estimated_cost_usd")
        man.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest_run


def merge_cumulative_summary(
    path: Path,
    *,
    round_idx: int,
    round_file_name: str,
    round_total_usd: float,
) -> None:
    """Append one round to ``evolve_llm_cost_cumulative.json`` under the run folder."""
    prev: dict[str, Any] = {"rounds": [], "cumulative_estimated_cost_usd": 0.0}
    if path.is_file():
        prev = json.loads(path.read_text(encoding="utf-8"))
    rounds = [x for x in (prev.get("rounds") or []) if int(x.get("round_idx", -1)) != int(round_idx)]
    rounds.append(
        {
            "round_idx": round_idx,
            "cost_file": round_file_name,
            "round_total_estimated_cost_usd": round_total_usd,
        }
    )
    cum = sum(float(x.get("round_total_estimated_cost_usd", 0.0) or 0.0) for x in rounds)
    payload = {
        "schema": "skill_mas_evolve_llm_cost_cumulative_v1",
        "rounds": rounds,
        "cumulative_estimated_cost_usd": round(cum, 10),
        "pricing_reference": pricing_reference(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
