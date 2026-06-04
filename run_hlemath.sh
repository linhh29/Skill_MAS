#!/usr/bin/env bash
# Skill_MAS evolution on HLEMATH (JSONL + sympy grading). Requires real JSONL (not Git LFS pointer).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_MAS_ROOT="${SCRIPT_DIR}"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BENCH_ID="skill_mas_agent"
RUN_ID="exp1"

ROUNDS="10"
K_TRAJ="5"
MAX_PROBLEMS="0"

MODEL=$1
AGENT_LLM=$MODEL
OPTIMIZER_LLM=$MODEL
MAX_CONCURRENCY=$2

JSONL="${SKILL_MAS_ROOT}/dataset/hlemath/data/hlemath_validate.jsonl"

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
export OPENAI_API_KEY="$(read_model_param "${AGENT_LLM}" "api_key")"
export OPENAI_API_BASE="$(read_model_param "${AGENT_LLM}" "base_url")"
export SKILL_MAS_AGENT_TEMPERATURE="$(read_model_param "${AGENT_LLM}" "temperature")"
export SKILL_MAS_AGENT_REASONING_EFFORT="$(read_model_param "${AGENT_LLM}" "reasoning_effort")"
export SKILL_MAS_AGENT_MAX_TOKENS="$(read_model_param "${AGENT_LLM}" "max_tokens")"

EXTRA=(--bench-backend hlemath --bench-id "${BENCH_ID}" --run-id "${RUN_ID}" --rounds "${ROUNDS}" --k-trajectories "${K_TRAJ}" --agent-llm "${AGENT_LLM}" --optimizer-llm "${OPTIMIZER_LLM}")
EXTRA+=(--jsonl "${JSONL}")
EXTRA+=(--max-problems "${MAX_PROBLEMS}")
EXTRA+=(--max-concurrency "${MAX_CONCURRENCY}")

python -u -m Skill_MAS evolve "${EXTRA[@]}"
