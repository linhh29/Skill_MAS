#!/usr/bin/env bash
# BrowseComp-Plus Skill-MAS evaluation (same entry as hlemath: Skill_MAS build.run_mas_pipeline_with_retries, dataset bcp).
# Uses Skill_MAS/skill_mas/model_config.json for per-model pricing and API routing (see BrowseComp-Plus/openai_client.py).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_MAS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PACKAGE_ROOT="$(cd "${SKILL_MAS_ROOT}/.." && pwd)"
BCP_ROOT="${SCRIPT_DIR}"

skill_path=$1

export PYTHONPATH="${PACKAGE_ROOT}:${SKILL_MAS_ROOT}/dataset:${BCP_ROOT}"
export SKILL_MAS_INIT_SKILL="${SKILL_MAS_ROOT}/${skill_path}"

# deepseek-v4-flash qwen3.5-plus
AGENT_LLM=$2
JUDGE_LLM="gemini-3.1-flash-lite-preview"
JSONL="${BCP_ROOT}/data/browsecomp_plus_test.jsonl"
INDEX_PATH="${BCP_ROOT}/scripts_build_index/indexes/bm25"
MAX_CONCURRENCY=$3
MAX_PROBLEMS="0"
INDICES=""
RETRIEVAL_TOPK="5"
DOC_MAX_TOKENS="512"
MAX_RETRIEVAL_ROUNDS="10"
PER_SAMPLE_TIMEOUT_S="7200"
JUDGE_TIMEOUT_S="120"

cd "${BCP_ROOT}"
python run_eval.py \
  --jsonl "${JSONL}" \
  --init-skill "${SKILL_MAS_INIT_SKILL}" \
  --agent-llm "${AGENT_LLM}" \
  --index-path "${INDEX_PATH}" \
  --retrieval-topk "${RETRIEVAL_TOPK}" \
  --doc-max-tokens "${DOC_MAX_TOKENS}" \
  --max-retrieval-rounds "${MAX_RETRIEVAL_ROUNDS}" \
  --per-sample-timeout-s "${PER_SAMPLE_TIMEOUT_S}" \
  --judge-llm "${JUDGE_LLM}" \
  --judge-timeout-s "${JUDGE_TIMEOUT_S}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --max-problems "${MAX_PROBLEMS}" \
  ${INDICES:+--indices "${INDICES}"}
