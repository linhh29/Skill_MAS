# Skill-MAS

Skill-MAS evolves a **single meta-agent skill file** (`SKILL.md`) that instructs an LLM to design and orchestrate a Multi-Agent System (MAS) in three stages: task decomposition, agent engineering, and workflow orchestration. The evolved skill is evaluated on multiple benchmarks; trajectories from each round drive contrastive reflection and skill rewriting.

**Quick navigation**

1. **[Running Evolution](#1-running-evolution)** — multi-round skill optimization on benchmark validation sets  
2. **[Direct Inference with Existing Skills](#2-direct-inference-with-existing-skills)** — single-question build + inference with `init_skill/` or `optimized_skill/` via `demo_inference.py`

---

## Architecture Overview

```
Skill_MAS/
├── core/                 # CLI entry, evolution pipeline, resume, task selection
├── evolution/            # Rollout, contrastive reflection, skill optimizer, bench adapters
├── skill_mas/            # 3-stage MAS builder, async LLM client, trace layout
├── template/             # Generated MAS code templates and SubAgent runtime
├── utils/                # Paths, cost tracking, logging, redaction
├── init_skill/           # Initial (pre-evolution) SKILL.md
├── optimized_skill/      # Pre-evolved skills per benchmark (ready to test)
├── skill_mas/model_config.json   # Model endpoints, pricing, runtime params
├── run_*.sh              # Convenience wrappers for each benchmark
├── demo_inference.py     # Single-question build + inference demo
├── demo_inference.sh     # Shell wrapper for demo_inference.py
└── results/              # Generated at runtime (not shipped)
```

### End-to-end flow

Each evolution **round** executes the following loop:

1. **Multi-trajectory rollout** (`evolution/rollout_multi.py`)  
   Run `k` trajectories per validation task. The agent reads the current round's `SKILL.md`, generates MAS code via three build stages (`skill_mas/build.py`), executes it, and records scores plus phase-level traces.

2. **Contrastive reflection** (`evolution/contrastive_reflect.py`)  
   Compare high- vs low-scoring trajectories and synthesize structured improvement signals.

3. **Skill bank optimization** (`evolution/bank_optimizer.py`)  
   An optimizer LLM rewrites the single `SKILL.md` using reflection reports and round statistics.

4. **Round selection** (`evolution/assemble_select.py`)  
   Track per-round scores; after all rounds, select the best skill snapshot.

### Benchmark backends

| Backend | CLI flag | Validation data (default) | Runner script |
|---------|----------|---------------------------|---------------|
| VitaBench | `--bench-backend vitabench` | `vitabench_single/data/vita_validate.json` | `run_vita.sh` |
| Deep Research Bench | `--bench-backend drb` | `deep_research_bench/data/drb_validate.jsonl` | `run_drb.sh` |
| HLEMath | `--bench-backend hlemath` | `hlemath/data/hlemath_validate.jsonl` | `run_hlemath.sh` |
| BrowseComp-Plus | `--bench-backend bcp` | `BrowseComp-Plus/data/browsecomp_plus_validate.jsonl` | `run_bcp.sh` |

### Runtime artifacts

Results are written under:

```
Skill_MAS/results/{backend}_{model_tag}/
├── artifacts/
│   ├── skills/{bench_id}/{run_id}/round_XX/SKILL.md   # skill snapshots per round
│   └── runs/{bench_id}/{run_id}/summary_rXX.json      # round metrics
└── log/{bench_id}/{run_id}/round_XX/                  # traces, exports
```

---

## Prerequisites

1. **Repository layout** — Place benchmark dependencies as sibling directories of `Skill_MAS/` (e.g. `vitabench_single/`, `deep_research_bench/`, `hlemath/`, `BrowseComp-Plus/`).

2. **Python environment** — Install dependencies required by Skill-MAS and the chosen benchmark backend.

3. **Model configuration** — Edit `skill_mas/model_config.json` and fill in `api_key` and `base_url` for each model you plan to use. Do **not** commit real credentials.

4. **Environment variables** — Shell runners read model params from `model_config.json` and export `OPENAI_API_KEY`, `OPENAI_API_BASE`, and role-specific `SKILL_MAS_*` settings automatically.

---

## 1. Running Evolution

### Quick start (shell scripts)

Each `run_*.sh` script takes two positional arguments: **agent model id** and **max concurrency**.

```bash
# From repository root
cd /path/to/repo

# VitaBench (cross-domain: delivery, instore, ota)
bash Skill_MAS/run_vita.sh <model_id> <max_concurrency>

# Deep Research Bench
bash Skill_MAS/run_drb.sh <model_id> <max_concurrency>

# HLEMath
bash Skill_MAS/run_hlemath.sh <model_id> <max_concurrency>

# BrowseComp-Plus
bash Skill_MAS/run_bcp.sh <model_id> <max_concurrency>
```

Default evolution settings in the scripts:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ROUNDS` | `10` | Number of evolve rounds |
| `K_TRAJ` | `5` | Trajectories sampled per task per round |
| `MAX_PROBLEMS` | `0` | Validation subset size (`0` = all tasks) |
| `RUN_ID` | `exp1` | Logical run name under `results/` |

Edit the variables at the top of each script to change rounds, task limits, evaluator/judge models, etc.

### CLI (direct invocation)

```bash
export PYTHONPATH="/path/to/repo:/path/to/repo/vitabench_single/src"

python -m Skill_MAS evolve \
  --bench-backend vitabench \
  --bench-id skill_mas_agent \
  --run-id exp1 \
  --domain "delivery,instore,ota" \
  --jsonl vitabench_single/data/vita_validate.json \
  --rounds 10 \
  --k-trajectories 5 \
  --agent-llm <model_id> \
  --user-llm <model_id> \
  --evaluator-llm <evaluator_model_id> \
  --optimizer-llm <model_id> \
  --max-concurrency 16 \
  --max-steps 300 \
  --language chinese
```

Other useful subcommands:

```bash
# List validation task IDs for a backend
python -m Skill_MAS list-val --bench-backend vitabench --jsonl vitabench_single/data/vita_validate.json

# Seed round_00 only (copy init skill, no rollout)
python -m Skill_MAS init-run --bench-backend vitabench --bench-id skill_mas_agent --run-id exp1
```

### Resume and fresh runs

- **Resume**: Re-run the same command with the same `--run-id`. The pipeline detects completed rounds via `summary_rXX.json` and continues from the next round.
- **Fresh run**: Add `--fresh` to allocate a new run directory (`exp1_2`, `exp1_3`, …) and restart from `round_00`.

### Evolution round internals (summary)

```
round_r/
  ├── trajectories/     # per-task, per-trajectory records
  ├── aspects/          # phase-level snapshots
  └── contrastive/      # reflection reports

skills/.../round_r/
  ├── SKILL.md          # skill used in this round (rewritten after r-1)
  ├── bank_meta.json    # optimization history
  └── knee_images/      # task-priority elbow plots
```

---

## 2. Direct Inference with Existing Skills

Two directories ship ready-to-use skill files—you do **not** need to re-run evolution to try them on a single question:

| Path | Role |
|------|------|
| `init_skill/SKILL.md` | Initial meta-agent skill (pre-evolution baseline) |
| `optimized_skill/vitabench.md` | Evolved skill for VitaBench |
| `optimized_skill/drb.md` | Evolved skill for Deep Research Bench |
| `optimized_skill/hlemath.md` | Evolved skill for HLEMath |
| `optimized_skill/bcp.md` | Evolved skill for BrowseComp-Plus |

Use `demo_inference.py` to load any of these paths, run the Skill-MAS three-stage build (decomposition → agent engineering → orchestration), execute the generated MAS, and print the answer.

### Shell wrapper

From the repository root (after configuring `skill_mas/model_config.json`):

```bash
# Usage: bash Skill_MAS/demo_inference.sh <model_id> <skill_path> "<question>"

bash Skill_MAS/demo_inference.sh qwen3.5-plus \
  Skill_MAS/init_skill/SKILL.md \
  "What is 17 + 28? Give the final answer in \\boxed{...} form."

bash Skill_MAS/demo_inference.sh qwen3.5-plus \
  Skill_MAS/optimized_skill/hlemath.md \
  "Find the number of positive integers n such that n^2 + 3n + 2 is divisible by n + 1."

bash Skill_MAS/demo_inference.sh qwen3.5-plus \
  Skill_MAS/optimized_skill/bcp.md \
  "Who received the Nobel Prize in Physics in 2020?"
```

### Python CLI

```bash
export PYTHONPATH="/path/to/repo:/path/to/repo/vitabench_single/src"

python Skill_MAS/demo_inference.py \
  --model qwen3.5-plus \
  --skill Skill_MAS/optimized_skill/drb.md \
  --question "Summarize recent progress in retrieval-augmented generation." \
  --verbose

python Skill_MAS/demo_inference.py \
  --model qwen3.5-plus \
  --skill Skill_MAS/init_skill/SKILL.md \
  --question "Your task prompt here." \
  --dataset hlemath \
  --save-mas-code /tmp/demo_mas.py
```

### Arguments

| Flag | Description |
|------|-------------|
| `--skill` | Path to `SKILL.md` or `optimized_skill/*.md` (required) |
| `--question` | Input task / question (required) |
| `--model` | Agent model id from `model_config.json` (default: `qwen3.5-plus`) |
| `--dataset` | `hlemath` \| `drb` \| `bcp` \| `vita`; auto-inferred from skill filename when omitted |
| `--save-mas-code` | Optional path to save generated MAS Python code |
| `--verbose` | Print parsed JSON from each build stage |

BrowseComp-Plus skills (`bcp`) additionally accept `--bcp-index-path`, `--bcp-retrieval-topk`, `--bcp-doc-max-tokens`, and `--bcp-max-retrieval-rounds` (defaults match `run_bcp.sh`). The `BrowseComp-Plus/` repo and BM25 index must be present.



