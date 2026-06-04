#!/usr/bin/env bash
# VitaBench Skill-MAS evaluation.
# API routing and pricing: Skill_MAS/skill_mas/model_config.json (same as hlemath / BrowseComp-Plus).
# Skill workspace: SKILL_MAS_DIR (phase banks under init_skill); init SKILL.md: SKILL_MAS_INIT_SKILL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VITABENCH_ROOT="${SCRIPT_DIR}"
SKILL_MAS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PACKAGE_ROOT="$(cd "${SKILL_MAS_ROOT}/.." && pwd)"
VITABENCH_SRC="${VITABENCH_ROOT}/src"

skill_path=$1
export PYTHONPATH="${PACKAGE_ROOT}:${SKILL_MAS_ROOT}/dataset:${VITABENCH_SRC}"
export SKILL_MAS_ROOT="${SKILL_MAS_ROOT}"
export SKILL_MAS_INIT_SKILL="${SKILL_MAS_ROOT}/${skill_path}"
export SKILL_MAS_DIR="$(dirname "${SKILL_MAS_INIT_SKILL}")"
export VITA_CROSS_DOMAIN_TASKS_PATH="${VITABENCH_ROOT}/data/vita_test.json"

# qwen3-next-80b-a3b-instruct deepseek-v4-flash qwen3-max qwen3.5-plus
model=$2
max_concurrent_tasks=$3
AGENT_MODEL=${model}
USER_MODEL=${model}
EVALUATOR_MODEL="gemini-3.1-flash-lite-preview"

# skill_mas_agent — Skill_MAS 三阶段仍在 Skill_MAS/skill_mas/build.py；此处仅 VitaBench 调度。
cd "${VITABENCH_ROOT}"
python -m vita.cli run \
  --domain delivery,instore,ota \
  --agent skill_mas_agent \
  --agent-llm "${AGENT_MODEL}" \
  --user static_input_user \
  --user-llm "${USER_MODEL}" \
  --evaluator-llm "${EVALUATOR_MODEL}" \
  --skill-mas-dir "${SKILL_MAS_DIR}" \
  --num-tasks 200 \
  --max-steps 300 \
  --max-concurrency "${max_concurrent_tasks}"
