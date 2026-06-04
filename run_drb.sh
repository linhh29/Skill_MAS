#!/usr/bin/env bash
# Skill_MAS DRB runner (single SKILL.md three-stage optimization + RACE).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_MAS_ROOT="${SCRIPT_DIR}"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BENCH_ID="skill_mas_agent"
RUN_ID="exp1"
DOMAIN="deepresearch"

MODEL=$1
AGENT_LLM=$MODEL
OPTIMIZER_LLM=$MODEL

ROUNDS="10"
K_TRAJ="5"
MAX_PROBLEMS="0"
MAX_CONCURRENCY=$2

DRB_BENCH_ROOT="${SKILL_MAS_ROOT}/dataset/deep_research_bench"
JSONL="${DRB_BENCH_ROOT}/data/drb_validate.jsonl"
DRB_RACE_MAX_WORKERS=$2
DRB_RACE_MODEL="gemini-3.1-flash-lite-preview"

read_model_param() {
  local model="$1"
  local key="$2"
  python -m Skill_MAS.utils.model_config_param --model "$model" --key "$key"
}

cd -- "${PACKAGE_ROOT}"
export PYTHONPATH="${PACKAGE_ROOT}:${SKILL_MAS_ROOT}/dataset:${SKILL_MAS_ROOT}/dataset/vitabench/src"

export MASKILL_PRINT_TRACES="0"
export LOGURU_LEVEL="ERROR"
export VITA_SUPPRESS_MODEL_LIST="1"
export DRB_RACE_MODEL="${DRB_RACE_MODEL}"
export OPENAI_API_KEY="$(read_model_param "${AGENT_LLM}" "api_key")"
export OPENAI_API_BASE="$(read_model_param "${AGENT_LLM}" "base_url")"
export SKILL_MAS_AGENT_TEMPERATURE="$(read_model_param "${AGENT_LLM}" "temperature")"
export SKILL_MAS_AGENT_REASONING_EFFORT="$(read_model_param "${AGENT_LLM}" "reasoning_effort")"
export SKILL_MAS_AGENT_MAX_TOKENS="$(read_model_param "${AGENT_LLM}" "max_tokens")"
export DRB_RACE_API_KEY="$(read_model_param "${DRB_RACE_MODEL}" "api_key")"
export DRB_RACE_API_BASE="$(read_model_param "${DRB_RACE_MODEL}" "base_url")"

python -u -m Skill_MAS evolve \
  --bench-backend drb \
  --bench-id "${BENCH_ID}" \
  --run-id "${RUN_ID}" \
  --domain "${DOMAIN}" \
  --jsonl "${JSONL}" \
  --max-problems "${MAX_PROBLEMS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --rounds "${ROUNDS}" \
  --k-trajectories "${K_TRAJ}" \
  --agent-llm "${AGENT_LLM}" \
  --optimizer-llm "${OPTIMIZER_LLM}" \
  --drb-bench-root "${DRB_BENCH_ROOT}" \
  --drb-race-max-workers "${DRB_RACE_MAX_WORKERS}"
