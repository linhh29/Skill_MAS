#!/usr/bin/env bash
# Skill_MAS VitaBench runner (single SKILL.md three-stage optimization).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_MAS_ROOT="${SCRIPT_DIR}"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BENCH_ID="skill_mas_agent"
RUN_ID="exp1"
DOMAIN="delivery,instore,ota"
TASK_SET_NAME=""
JSONL="${SKILL_MAS_ROOT}/dataset/vitabench/data/vita_validate.json"

ROUNDS="10"
K_TRAJ="5"
MAX_PROBLEMS="0"

MODEL=$1
AGENT_LLM=$MODEL
USER_LLM=$MODEL
EVALUATOR_LLM="deepseek-v4-flash"
OPTIMIZER_LLM=$MODEL
MAX_STEPS="300"
MAX_CONCURRENCY=$2
LANGUAGE="chinese"

read_model_param() {
  local model="$1"
  local key="$2"
  python -m Skill_MAS.utils.model_config_param --model "$model" --key "$key"
}

cd -- "${PACKAGE_ROOT}"
export PYTHONPATH="${PACKAGE_ROOT}:${SKILL_MAS_ROOT}/dataset/vitabench/src:${SKILL_MAS_ROOT}/dataset"

export MASKILL_PRINT_TRACES="1"
export VITA_CROSS_DOMAIN_TASKS_PATH="${JSONL}"
export OPENAI_API_KEY="$(read_model_param "${AGENT_LLM}" "api_key")"
export OPENAI_API_BASE="$(read_model_param "${AGENT_LLM}" "base_url")"
export SKILL_MAS_AGENT_TEMPERATURE="$(read_model_param "${AGENT_LLM}" "temperature")"
export SKILL_MAS_AGENT_REASONING_EFFORT="$(read_model_param "${AGENT_LLM}" "reasoning_effort")"
export SKILL_MAS_AGENT_MAX_TOKENS="$(read_model_param "${AGENT_LLM}" "max_tokens")"
export SKILL_MAS_USER_TEMPERATURE="$(read_model_param "${USER_LLM}" "temperature")"
export SKILL_MAS_USER_REASONING_EFFORT="$(read_model_param "${USER_LLM}" "reasoning_effort")"
export SKILL_MAS_USER_MAX_TOKENS="$(read_model_param "${USER_LLM}" "max_tokens")"
export SKILL_MAS_EVALUATOR_TEMPERATURE="$(read_model_param "${EVALUATOR_LLM}" "temperature")"
export SKILL_MAS_EVALUATOR_REASONING_EFFORT="$(read_model_param "${EVALUATOR_LLM}" "reasoning_effort")"
export SKILL_MAS_EVALUATOR_MAX_TOKENS="$(read_model_param "${EVALUATOR_LLM}" "max_tokens")"

EXTRA_ARGS=()
if [[ -n "${TASK_SET_NAME}" ]]; then
  EXTRA_ARGS+=(--task-set-name "${TASK_SET_NAME}")
fi

python -u -m Skill_MAS evolve \
  --bench-backend vitabench \
  --bench-id "${BENCH_ID}" \
  --run-id "${RUN_ID}" \
  --domain "${DOMAIN}" \
  --jsonl "${JSONL}" \
  --max-problems "${MAX_PROBLEMS}" \
  --rounds "${ROUNDS}" \
  --k-trajectories "${K_TRAJ}" \
  --agent-llm "${AGENT_LLM}" \
  --user-llm "${USER_LLM}" \
  --evaluator-llm "${EVALUATOR_LLM}" \
  --optimizer-llm "${OPTIMIZER_LLM}" \
  --max-steps "${MAX_STEPS}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --language "${LANGUAGE}" \
  "${EXTRA_ARGS[@]}"
