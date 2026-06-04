#!/usr/bin/env bash
# HLEMATH skill_mas evaluation.
# Data: hlemath/data/hlemath_validate.jsonl — if still LFS pointers, run: git lfs pull

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_MAS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PACKAGE_ROOT="$(cd "${SKILL_MAS_ROOT}/.." && pwd)"

skill_path=$1
export PYTHONPATH="${PACKAGE_ROOT}:${SKILL_MAS_ROOT}/dataset"
export SKILL_MAS_INIT_SKILL="${SKILL_MAS_ROOT}/${skill_path}"

AGENT_LLM=$2
max_concurrent_tasks=$3
JSONL="${SCRIPT_DIR}/data/hlemath_test.jsonl"

cd "${PACKAGE_ROOT}"
python -m hlemath.run_eval \
  --jsonl "${JSONL}" \
  --init-skill "${SKILL_MAS_INIT_SKILL}" \
  --agent-llm "${AGENT_LLM}" \
  --max-concurrency ${max_concurrent_tasks} \
  --max-problems "0"
