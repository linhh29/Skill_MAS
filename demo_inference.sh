#!/usr/bin/env bash
# Skill-MAS single-question inference demo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_MAS_ROOT="${SCRIPT_DIR}"
PACKAGE_ROOT="$(cd "${SKILL_MAS_ROOT}/.." && pwd)"

MODEL="${1:-qwen3.5-plus}"
SKILL_PATH="${2:-${SKILL_MAS_ROOT}/init_skill/SKILL.md}"
QUESTION="${3:-What is 17 + 28? Give the final answer in \\\\boxed{...} form.}"

cd -- "${PACKAGE_ROOT}"
export PYTHONPATH="${PACKAGE_ROOT}:${SKILL_MAS_ROOT}/dataset:${SKILL_MAS_ROOT}/dataset/vitabench/src"

read_model_param() {
  local model="$1"
  local key="$2"
  python -m Skill_MAS.utils.model_config_param --model "$model" --key "$key"
}

export OPENAI_API_KEY="$(read_model_param "${MODEL}" "api_key")"
export OPENAI_API_BASE="$(read_model_param "${MODEL}" "base_url")"
export SKILL_MAS_AGENT_TEMPERATURE="$(read_model_param "${MODEL}" "temperature")"
export SKILL_MAS_AGENT_REASONING_EFFORT="$(read_model_param "${MODEL}" "reasoning_effort")"
export SKILL_MAS_AGENT_MAX_TOKENS="$(read_model_param "${MODEL}" "max_tokens")"

python -u "${SCRIPT_DIR}/demo_inference.py" \
  --model "${MODEL}" \
  --skill "${SKILL_PATH}" \
  --question "${QUESTION}"
