#!/usr/bin/env bash
# Skill_MAS VitaBench runner (single SKILL.md three-stage optimization).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ----------------------------
# Core experiment identifiers
# ----------------------------
BENCH_ID="skill_mas_agent"
RUN_ID="exp1"
DOMAIN="delivery,instore,ota"
TASK_SET_NAME=""
JSONL="${REPO_ROOT}/Skill_MAS/dataset/vitabench/data/vita_validate.json"

# ----------------------------
# Evolution controls
# ----------------------------
ROUNDS="10"
K_TRAJ="5"
MAX_PROBLEMS="0"

# ----------------------------
# Runtime / model settings
# ----------------------------
MODEL=$1
AGENT_LLM=$MODEL
USER_LLM=$MODEL
EVALUATOR_LLM="gemini-3.1-flash-lite-preview"
OPTIMIZER_LLM=$MODEL
MAX_STEPS="300"
MAX_CONCURRENCY=$2
LANGUAGE="chinese"
MODEL_CONFIG="${REPO_ROOT}/Skill_MAS/skill_mas/model_config.json"

read_model_param() {
  local model="$1"
  local key="$2"
  python -m Skill_MAS.utils.model_config_param --model "$model" --key "$key"
}

cd -- "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/Skill_MAS/dataset/vitabench/src:${REPO_ROOT}/Skill_MAS/dataset"

# Default during evolve: no Skill-MAS mirror under dataset/vitabench/results/.../skill_mas_process_traces/
# (rollout_multi sets MASKILL_SKIP_VITABENCH_TRACE_EXPORT). To export there anyway: export MASKILL_SKIP_VITABENCH_TRACE_EXPORT=0

# Print full traces in pipeline logs
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
