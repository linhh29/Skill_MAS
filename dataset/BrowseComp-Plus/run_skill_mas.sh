#!/usr/bin/env bash
# Skill_MAS evolution on BrowseComp-Plus (single SKILL.md optimization).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_MAS_ROOT="${SCRIPT_DIR}"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BCP_ROOT="${SKILL_MAS_ROOT}/dataset/BrowseComp-Plus"

BENCH_ID="skill_mas_agent"
RUN_ID="exp1"
DOMAIN="browsecomp"

ROUNDS="10"
K_TRAJ="5"
MAX_PROBLEMS="0"

MODEL=$1
AGENT_LLM=$MODEL
OPTIMIZER_LLM=$MODEL
JUDGE_LLM="gemini-3.1-flash-lite-preview"
JUDGE_TIMEOUT_S="1200"
MAX_CONCURRENCY=$2

JSONL="${BCP_ROOT}/data/browsecomp_plus_validate.jsonl"
INDEX_PATH="${BCP_ROOT}/scripts_build_index/indexes/bm25"
RETRIEVAL_TOPK="5"
DOC_MAX_TOKENS="512"
MAX_RETRIEVAL_ROUNDS="10"

read_model_param() {
  local model="$1"
  local key="$2"
  python -m Skill_MAS.utils.model_config_param --model "$model" --key "$key"
}

cd -- "${PACKAGE_ROOT}"
export PYTHONPATH="${PACKAGE_ROOT}:${SKILL_MAS_ROOT}/dataset:${BCP_ROOT}:${SKILL_MAS_ROOT}/dataset/vitabench/src"

export MASKILL_PRINT_TRACES="0"
export LOGURU_LEVEL="ERROR"
export VITA_SUPPRESS_MODEL_LIST="1"
export OPENAI_API_KEY="$(read_model_param "${AGENT_LLM}" "api_key")"
export OPENAI_API_BASE="$(read_model_param "${AGENT_LLM}" "base_url")"
export SKILL_MAS_AGENT_TEMPERATURE="$(read_model_param "${AGENT_LLM}" "temperature")"
export SKILL_MAS_AGENT_REASONING_EFFORT="$(read_model_param "${AGENT_LLM}" "reasoning_effort")"
export SKILL_MAS_AGENT_MAX_TOKENS="$(read_model_param "${AGENT_LLM}" "max_tokens")"

python -u -m Skill_MAS evolve \
  --bench-backend bcp \
  --bench-id "${BENCH_ID}" \
  --run-id "${RUN_ID}" \
  --domain "${DOMAIN}" \
  --jsonl "${JSONL}" \
  --max-problems "${MAX_PROBLEMS}" \
  --rounds "${ROUNDS}" \
  --k-trajectories "${K_TRAJ}" \
  --agent-llm "${AGENT_LLM}" \
  --optimizer-llm "${OPTIMIZER_LLM}" \
  --judge-llm "${JUDGE_LLM}" \
  --judge-timeout-s "${JUDGE_TIMEOUT_S}" \
  --bcp-index-path "${INDEX_PATH}" \
  --bcp-retrieval-topk "${RETRIEVAL_TOPK}" \
  --bcp-doc-max-tokens "${DOC_MAX_TOKENS}" \
  --bcp-max-retrieval-rounds "${MAX_RETRIEVAL_ROUNDS}" \
  --max-concurrency "${MAX_CONCURRENCY}"
