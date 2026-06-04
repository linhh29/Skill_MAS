# Skill-MAS

Skill-MAS evolves a **single meta-agent skill file** (`SKILL.md`) that instructs an LLM to design and orchestrate a Multi-Agent System (MAS) in three stages: task decomposition, agent engineering, and workflow orchestration. The evolved skill is evaluated on multiple benchmarks; trajectories from each round drive contrastive reflection and skill rewriting.

**Jump to what you need**

1. **[Running Evolution](#1-running-evolution)** — multi-round skill optimization on benchmark validation sets  
2. **[Direct Inference with Existing Skills](#2-direct-inference-with-existing-skills)** — single-question build + inference with `init_skill/` or `optimized_skill/` via `demo_inference.py`

---

## 🏗️ Architecture Overview

```
Skill_MAS/
├── core/                 # CLI entry, evolution pipeline, resume, task selection
├── evolution/            # Rollout, contrastive reflection, skill optimizer, bench adapters
├── skill_mas/            # 3-stage MAS builder, async LLM client, model_config.json(.example)
├── template/             # Generated MAS code templates and SubAgent runtime
├── utils/                # Paths, cost tracking, logging, redaction
├── init_skill/           # Initial (pre-evolution) SKILL.md
├── optimized_skill/      # Pre-evolved skills per benchmark (ready to test)
├── dataset/              # Bundled benchmark code + data
│   ├── vitabench/        # VitaBench (src/vita, data/)
│   ├── deep_research_bench/
│   ├── hlemath/
│   └── BrowseComp-Plus/
├── run_*.sh              # Evolution wrappers (run from arxiv_code/ parent)
├── demo_inference.py     # Single-question build + inference demo
├── demo_inference.sh     # Shell wrapper for demo_inference.py
└── results/              # Generated at runtime (not shipped)
```

### How a round works

Each evolution **round** goes through this loop:

1. **Multi-trajectory rollout** (`evolution/rollout_multi.py`)  
   We run `k` trajectories per validation task. The agent reads the current round's `SKILL.md`, generates MAS code via three build stages (`skill_mas/build.py`), executes it, and records scores plus phase-level traces.

2. **Contrastive reflection** (`evolution/contrastive_reflect.py`)  
   We compare high- vs low-scoring trajectories and synthesize structured improvement signals.

3. **Skill bank optimization** (`evolution/bank_optimizer.py`)  
   An optimizer LLM rewrites the single `SKILL.md` using reflection reports and round statistics.

4. **Round selection** (`evolution/assemble_select.py`)  
   We track per-round scores; after all rounds, the best skill snapshot is selected.

### Benchmark backends

| Backend | CLI flag | Validation data (default) | Evolution script | Per-bench eval script |
|---------|----------|---------------------------|------------------|------------------------|
| VitaBench | `--bench-backend vitabench` | `dataset/vitabench/data/vita_validate.json` | `run_vita.sh` | `dataset/vitabench/run_skill_mas.sh` |
| Deep Research Bench | `--bench-backend drb` | `dataset/deep_research_bench/data/drb_validate.jsonl` | `run_drb.sh` | `dataset/deep_research_bench/run_skill_mas.sh` |
| HLEMath | `--bench-backend hlemath` | `dataset/hlemath/data/hlemath_validate.jsonl` | `run_hlemath.sh` | `dataset/hlemath/run_skill_mas.sh` |
| BrowseComp-Plus | `--bench-backend bcp` | `dataset/BrowseComp-Plus/data/browsecomp_plus_validate.jsonl` | `run_bcp.sh` | `dataset/BrowseComp-Plus/run_skill_mas.sh` |

### Where results land

Results are written under:

```
Skill_MAS/results/{backend}_{model_tag}/
├── artifacts/
│   ├── skills/{bench_id}/{run_id}/round_XX/SKILL.md   # skill snapshots per round
│   └── runs/{bench_id}/{run_id}/summary_rXX.json      # round metrics
└── log/{bench_id}/{run_id}/round_XX/                  # traces, exports
```

---

## ⚙️ Prerequisites

Before you run anything, make sure you've got these set up:

1. **Repository layout** — Check out this repo so that `Skill_MAS/` lives under a parent directory (e.g. `arxiv_code/`). All benchmarks are vendored under `Skill_MAS/dataset/`; you do **not** need sibling copies of `vitabench_single/`, `hlemath/`, etc.

2. **Python environment** — Install dependencies for Skill-MAS and the benchmark you use (see `dataset/vitabench/requirements.txt` and each benchmark's README).



4. **PYTHONPATH** — Run commands from the **parent of `Skill_MAS/`** (e.g. `arxiv_code/`) so `import Skill_MAS` and `import vita` resolve correctly. The shell scripts set this automatically.

---

## 🔄 1. Running Evolution

### Quick start (shell scripts)

Run from the directory **above** `Skill_MAS/` (e.g. `arxiv_code/`). Each `run_*.sh` takes two positional arguments: **agent model id** and **max concurrency**.

```bash
cd /path/to/arxiv_code

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

Tweak the variables at the top of each script if you want to change rounds, task limits, evaluator/judge models, etc.

### CLI (direct invocation)

```bash
cd /path/to/arxiv_code
export OPENAI_API_KEY="your-key"
export PYTHONPATH="$(pwd):$(pwd)/Skill_MAS/dataset:$(pwd)/Skill_MAS/dataset/vitabench/src"

python -m Skill_MAS evolve \
  --bench-backend vitabench \
  --bench-id skill_mas_agent \
  --run-id exp1 \
  --domain "delivery,instore,ota" \
  --jsonl Skill_MAS/dataset/vitabench/data/vita_validate.json \
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

A couple of other subcommands I find handy:

```bash
# List validation task IDs for a backend
python -m Skill_MAS list-val \
  --bench-backend vitabench \
  --jsonl Skill_MAS/dataset/vitabench/data/vita_validate.json

# Seed round_00 only (copy init skill, no rollout)
python -m Skill_MAS init-run --bench-backend vitabench --bench-id skill_mas_agent --run-id exp1
```

### Resume and fresh runs

- **Resume**: Re-run the same command with the same `--run-id`. The pipeline detects completed rounds via `summary_rXX.json` and picks up from the next round.
- **Fresh run**: Add `--fresh` to allocate a new run directory (`exp1_2`, `exp1_3`, …) and restart from `round_00`.

### What gets saved each round

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

### Evaluating a fixed skill (no evolution)

If you just want to run a benchmark with an existing skill — no multi-round evolution — use the per-bench scripts under `dataset/`. The first argument is a path **relative to `Skill_MAS/`** pointing at the skill file (e.g. `init_skill/SKILL.md` or `optimized_skill/hlemath.md`).

```bash
cd /path/to/arxiv_code
export OPENAI_API_KEY="your-key"

bash Skill_MAS/dataset/hlemath/run_skill_mas.sh init_skill/SKILL.md <model_id> <max_concurrency>
bash Skill_MAS/dataset/deep_research_bench/run_skill_mas.sh optimized_skill/drb.md <model_id> <max_concurrency>
bash Skill_MAS/dataset/BrowseComp-Plus/run_skill_mas.sh optimized_skill/bcp.md <model_id> <max_concurrency>
bash Skill_MAS/dataset/vitabench/run_skill_mas.sh optimized_skill/vitabench.md <model_id> <max_concurrency>
```

---

## 🚀 2. Direct Inference with Existing Skills

We ship two directories of ready-to-use skill files — you do **not** need to re-run evolution to try them on a single question:

| Path | Role |
|------|------|
| `init_skill/SKILL.md` | Initial meta-agent skill (pre-evolution baseline) |
| `optimized_skill/vitabench.md` | Evolved skill for VitaBench |
| `optimized_skill/drb.md` | Evolved skill for Deep Research Bench |
| `optimized_skill/hlemath.md` | Evolved skill for HLEMath |
| `optimized_skill/bcp.md` | Evolved skill for BrowseComp-Plus |

`demo_inference.py` loads any of these paths, runs the Skill-MAS three-stage build (decomposition → agent engineering → orchestration), executes the generated MAS, and prints the answer.

Supported standalone datasets in the demo: **hlemath**, **drb**, **bcp**. VitaBench needs the full simulator (`run_vita.sh` or `dataset/vitabench/run_skill_mas.sh`).

### Shell wrapper

From the parent of `Skill_MAS/` (after configuring `model_config.json` and `OPENAI_API_KEY`):

```bash
cd /path/to/arxiv_code

# Usage: bash Skill_MAS/demo_inference.sh <model_id> <skill_path> "<question>"

bash Skill_MAS/demo_inference.sh qwen3.5-plus \
  Skill_MAS/init_skill/SKILL.md \
  "What is 17 + 28? Give the final answer in \\boxed{...} form."

bash Skill_MAS/demo_inference.sh qwen3.5-plus \
  Skill_MAS/optimized_skill/hlemath.md \
  "Find the number of positive integers n such that n^2 + 3n + 2 is divisible by n + 1."

```

### Python CLI

```bash
cd /path/to/arxiv_code
export OPENAI_API_KEY="your-key"
export PYTHONPATH="$(pwd):$(pwd)/Skill_MAS/dataset:$(pwd)/Skill_MAS/dataset/vitabench/src"

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

BrowseComp-Plus skills (`bcp`) additionally accept `--bcp-index-path`, `--bcp-retrieval-topk`, `--bcp-doc-max-tokens`, and `--bcp-max-retrieval-rounds` (defaults match `run_bcp.sh`). The BM25 index under `dataset/BrowseComp-Plus/scripts_build_index/indexes/bm25` must be available (see that benchmark's README for download steps).

For BrowseComp-Plus, please refer to the official repo for the data preprocess. We don't provide a plain text data file used in our paper for preventing the data contamination.
