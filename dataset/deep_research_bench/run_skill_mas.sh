#!/usr/bin/env bash
# DRB Skill-MAS generation + RACE evaluation thin wrapper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_MAS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PACKAGE_ROOT="$(cd "${SKILL_MAS_ROOT}/.." && pwd)"

export PYTHONPATH="${PACKAGE_ROOT}:${SKILL_MAS_ROOT}/dataset"

skill_path=$1
AGENT_LLM=$2
RACE_JUDGE_LLM="gemini-3.1-flash-lite-preview"
JSONL="${SCRIPT_DIR}/data/drb_test.jsonl"
SKILL_MAS_DIR="${SKILL_MAS_ROOT}/${skill_path}"
max_concurrent_tasks=$3
MAX_CONCURRENCY=${max_concurrent_tasks}
JUDGE_MAX_CONCURRENCY=${max_concurrent_tasks}
MAX_PROBLEMS="0"

export DRB_RACE_MODEL="${RACE_JUDGE_LLM}"

cd "${PACKAGE_ROOT}"
python -m deep_research_bench.run_skill_mas \
  --query_file "${JSONL}" \
  --model_name "${AGENT_LLM}" \
  --skill_dir "${SKILL_MAS_DIR}" \
  --eval_per_sample \
  --max_concurrent "${MAX_CONCURRENCY}" \
  --judge_max_concurrent "${JUDGE_MAX_CONCURRENCY}" \
  $( [[ "${MAX_PROBLEMS}" != "0" ]] && echo "--limit ${MAX_PROBLEMS}" )
