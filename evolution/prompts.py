"""Centralized prompts for Skill-MAS evolution and review."""

from __future__ import annotations

SYS_CONTRASTIVE_PHASE1 = """You are the diagnosis agent for Skill_MAS Step 2 (Trajectory Reflection Synthesis).
Your task is to analyze trajectories for ONE task at a time, produce that sample's contrastive diagnosis,
and output structured JSON (including narrative_summary for a later cross-sample call). This call does not
include other tasks — synthesize only within the provided task_id."""

SYS_CONTRASTIVE_PHASE2 = """You are the diagnosis agent for Skill_MAS Step 2 (Trajectory Reflection Synthesis).
Your task is to synthesize cross-sample findings for Step3 optimization. For each task you receive the original
problem text plus the COMPLETE Phase-1 structured JSON (score_statistics, trajectory_grouping, contrastive_diagnosis,
candidate_patch, narrative_summary, etc.). You do not receive raw per-agent trajectory dumps or mas_code.
Output strict JSON: cross_sample_synthesis and meta_analysis only."""


def user_contrastive_phase1(
    *,
    task_id: str,
    num_rollouts: int,
    trajectories_payload: str,
    input_char_count: int,
    estimated_input_tokens: int,
) -> str:
    return f"""[Step2: Contrastive Trajectory Analysis — Phase 1 (Intra-sample only)]

  === SELECTION CONTEXT (this call) ===
  task_id: {task_id}
  rollouts for this task: {num_rollouts}
  Input statistics:
  - Character count: {input_char_count:,}
  - Estimated tokens: ~{estimated_input_tokens:,}

  Trajectory indices: the JSON field trajectories[] is sorted by score ascending. Use 0-based indices into
  trajectories[] for high_performing_indices / low_performing_indices.

  Selection rationale: This sample was selected for HIGH PRIORITY due to:
  1. **High cross-trajectory volatility**: Large score variance across rollouts indicates unstable/inconsistent policy behavior
  2. **High intrinsic difficulty**: Low average scores suggest systematic capability gaps

  Your mission: Diagnose WHY this sample is volatile and/or difficult, and propose an actionable fix for this task.

  === INPUT DATA ===
  Below are ALL trajectories (this task, all k rollouts):

  {trajectories_payload}

  === ANALYSIS FRAMEWORK ===

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LEVEL 1: INTRA-SAMPLE CONTRASTIVE ANALYSIS (Per-Sample Deep Dive)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  For THIS sample, perform contrastive analysis by comparing its trajectories:

  Step 1.1: Quantify the divergence
  - Report score distribution: min, max, mean, std, gap (max-min)
  - Identify which trajectories are "high-performing" vs "low-performing" (use the median score as the threshold)
  - Calculate volatility metrics if applicable

  Step 1.2: Contrastive diagnosis (CRITICAL - be specific!)
  Answer these questions by comparing concrete trajectory behaviors:

  A. **Divergence points**: Where do high-score and low-score trajectories START to diverge?
     - Identify the specific step/phase/decision where paths split
     - What different choices/actions/strategies do they take?
     - Is the divergence due to: exploration randomness, reasoning errors, constraint violations, or something else?

  B. **Success factors**: Why do high-score trajectories succeed?
     - What specific behaviors/strategies/patterns lead to success?
     - Are there critical decisions that high-score trajectories get right?
     - Is success due to: correct reasoning, better exploration, constraint adherence, or luck?

  C. **Failure modes**: Why do low-score trajectories fail?
     - What specific errors/mistakes/violations occur?
     - Are failures due to: wrong reasoning, premature termination, constraint violations, inefficient search, or capability gaps?
     - Are failures recoverable (wrong path but could backtrack) or fundamental (missing capability)?

  D. **Volatility root cause**: If this sample has high uncertainty, what causes it?
     - Is the task inherently ambiguous (multiple valid interpretations)?
     - Is the policy's decision-making unstable at critical junctions?
     - Are there stochastic elements (exploration, sampling) that amplify variance?

  E. **Difficulty root cause**: If this sample has low average score, what makes it hard?
     - Does it require capabilities the policy lacks (complex reasoning, long-term planning, domain knowledge)?
     - Does it have tight constraints that are easy to violate?
     - Is it a boundary case that exposes edge-case weaknesses?

  Step 1.3: Propose a targeted patch
  Based on your diagnosis, design ONE concrete, actionable fix for this sample:
  - **Target phase**: Which phase of the process should be modified? (e.g., "Phase 2: Planning", "Phase 3: Execution")
  - **Constraint/Rule**: State a concise, executable constraint or rule (e.g., "Before taking action X, verify condition Y", "When encountering situation Z, apply strategy W")
  - **Mechanism**: How should this be implemented? (prompt engineering, constraint checking, search strategy change, etc.)
  - **Expected impact**: What specific failure mode will this fix address? How will it reduce volatility or improve success rate?

  Additionally, fill narrative_summary: a standalone prose summary that complements the structured fields for Phase 2.
  Phase 2 receives the FULL Phase-1 JSON below (all schema fields) plus the original task text — not raw trajectory dumps.

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OUTPUT FORMAT (STRICT JSON - NO MARKDOWN, NO CODE BLOCKS)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A single object with the same fields as one element of the historical per_sample_analysis array, PLUS narrative_summary:

  {{
    "task_id": "string",
    "num_trajectories": int,
    "score_statistics": {{
      "min": float,
      "max": float,
      "mean": float,
      "std": float,
      "gap": float
    }},
    "trajectory_grouping": {{
      "high_performing_indices": [],
      "low_performing_indices": [],
      "rationale": "string"
    }},
    "contrastive_diagnosis": {{
      "divergence_points": [
        "string"
      ],
      "success_factors": [
        "string"
      ],
      "failure_modes": [
        "string"
      ],
      "volatility_root_cause": "string",
      "difficulty_root_cause": "string"
    }},
    "candidate_patch": {{
      "target_phase": "string",
      "constraint_rule": "string",
      "implementation_mechanism": "string",
      "expected_impact": "string"
    }},
    "narrative_summary": "string"
  }}

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CRITICAL REMINDERS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. **Be specific**: Avoid vague statements like "trajectory fails due to poor reasoning". Instead: "trajectory fails at step 5 because it incorrectly assumes X when the constraint requires Y"

  2. **Use evidence**: Ground every claim in concrete trajectory observations. Reference specific steps, actions, or outputs.

  3. **Think contrastively**: Always compare high vs low trajectories. The DIFFERENCE is where the insight lies.

  4. **Focus on actionability**: Every diagnosis should lead to a concrete, implementable fix. Avoid unfixable issues like "task is too hard".

  5. **Quantify when possible**: Use numbers (frequencies, percentages, counts) to support claims about patterns.

  6. **Output pure JSON**: No markdown code blocks, no extra text. Start with {{ and end with }}.

  Begin your analysis now.
  """


def user_contrastive_phase2(
    *,
    selected_task_ids: list[str],
    per_task_blocks: str,
    input_char_count: int,
    estimated_input_tokens: int,
) -> str:
    ids = ", ".join(selected_task_ids)
    return f"""[Step2: Contrastive Trajectory Analysis — Phase 2 (Cross-sample only)]

  === SELECTION CONTEXT ===
  Selected task_ids (n={len(selected_task_ids)}): {ids}
  Input statistics:
  - Character count: {input_char_count:,}
  - Estimated tokens: ~{estimated_input_tokens:,}

  Selection rationale: These samples exhibit HIGH PRIORITY due to:
  1. **High cross-trajectory volatility**: Large score variance across rollouts indicates unstable/inconsistent policy behavior
  2. **High intrinsic difficulty**: Low average scores suggest systematic capability gaps

  === INPUT DATA ===
  Phase 1 already analyzed each task's rollouts. Below, each block contains:
  (1) the original problem / instruction text, and
  (2) the COMPLETE Phase-1 structured JSON for that task — every field in the Phase-1 schema (task_id, num_trajectories,
  score_statistics, trajectory_grouping, contrastive_diagnosis, candidate_patch, narrative_summary).
  You do NOT have raw per-agent workflow dumps or mas_code; use the structured Phase-1 JSON and task text only.

  {per_task_blocks}

  === ANALYSIS FRAMEWORK ===

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LEVEL 2: CROSS-SAMPLE SYNTHESIS (Global Pattern Recognition)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  After Phase 1 analyzed each sample individually, synthesize global insights:

  Step 2.1: Identify systematic weaknesses
  - What failure modes appear across MULTIPLE samples? (frequency matters!)
  - Are there common capability gaps? (e.g., "struggles with multi-step reasoning", "poor constraint checking")
  - Are there common volatility sources? (e.g., "unstable at decision point X", "sensitive to prompt phrasing")
  - Rank weaknesses by: (severity × frequency)

  Step 2.2: Identify systematic strengths
  - What does the policy consistently do well across samples?
  - Are there patterns in successful trajectories that can be reinforced?
  - What capabilities are reliable and can be leveraged?

  Step 2.3: Prioritize fixes for Step 3
  Based on the above, propose 3-5 prioritized fixes:
  - Each fix should address a HIGH-IMPACT weakness (affects multiple samples or causes severe failures)
  - Each fix should be ACTIONABLE (clear what to change and how)
  - Rank by expected ROI: (impact × feasibility)

  Format each fix as:
  "[Priority X] <Fix description>: <Rationale> → Expected to improve <metric> on
   <affected samples>"

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  OUTPUT FORMAT (STRICT JSON - NO MARKDOWN, NO CODE BLOCKS)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  {{
    "cross_sample_synthesis": {{
      "systematic_weaknesses": [
        {{
          "weakness": "string",
          "severity": "high|medium|low",
          "frequency": "string",
          "affected_samples": [],
          "manifestation": "string"
        }}
      ],
      "systematic_strengths": [
        {{
          "strength": "string",
          "evidence": "string",
          "leverage_opportunity": "string"
        }}
      ],
      "prioritized_fixes": [
        {{
          "priority": float,
          "fix_description": "string",
          "rationale": "string",
          "implementation": "string",
          "expected_impact": "string",
          "affected_samples": int,
          "estimated_effort": "low|medium|high"
        }}
      ]
    }},
    "meta_analysis": {{
      "overall_diagnosis": "string",
      "confidence_level": "high|medium|low",
      "data_quality_notes": "string",
      "recommended_next_steps": [
        "string"
      ]
    }}
  }}

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CRITICAL REMINDERS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. **Be specific**: Tie weaknesses/strengths to task_ids and concrete themes from the summaries when possible.

  2. **Use evidence**: Ground claims in the Phase-1 structured outputs and task text — do not invent unseen trajectory detail.

  3. **Think globally**: Patterns across samples drive prioritization.

  4. **Focus on actionability**: prioritized_fixes must be implementable in Step 3.

  5. **Quantify when possible**: Use counts where summaries allow.

  6. **Output pure JSON**: No markdown code blocks, no extra text. Start with {{ and end with }}.

  Begin your synthesis now.
  """


SYS_SKILL_WRITER = """You are an expert author and optimizer for Skill-MAS three-stage SKILL.md files.
Your task is to improve the current SKILL.md based on Step2 reflection evidence while preserving a valid 3-stage structure.
"""


def user_bank_optimizer_prompt(
    *,
    round_idx: int,
    total_rounds: int,
    bench_hint: str,
    current_skill_md: str,
    step2_reflection_summary: str,
) -> str:
    return f"""## Evolution Context
- Round: {round_idx + 1} / {total_rounds} (0-based round_idx={round_idx})
- Target: Evolve a DOMAIN-AGNOSTIC, high-level Meta-Agent SKILL.md.
- Benchmark context (for understanding context only): {bench_hint}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT SKILL.md
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{current_skill_md}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP2 REFLECTION ANALYSIS (Your Core Evidence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{step2_reflection_summary}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR MISSION: ARCHITECTURAL & COGNITIVE EVOLUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are an elite AI Systems Architect. Your task is to evolve the provided SKILL.md. 
The goal is to elevate the Meta-Agent's capability to design sophisticated, robust, and generalizable Multi-Agent Systems (MAS).

CRITICAL ABSTRACTION FIREWALL (MUST READ):
The Meta-Agent operates at the ARCHITECTURAL level, not the implementation level.
1. ABSOLUTELY NO domain-specific examples (e.g., DO NOT mention delivery, bookings, math formulas, or specific APIs).
2. ABSOLUTELY NO coding/syntax details (e.g., DO NOT mention Python, AST parsing, variable names, snake_case, `await`, or state dictionaries).
3. ABSOLUTELY NO hardcoded heuristics (e.g., DO NOT say "if text contains 'and', split it").

Instead, "Actionable" means providing powerful COGNITIVE FRAMEWORKS and SYSTEM DESIGNS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
PRUNING & REFINEMENT (Before Adding)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  

Before introducing new upgrades, review the CURRENT SKILL.md against the STEP2 REFLECTION ANALYSIS:                                                                    
- IDENTIFY any existing guidance that the Step2 evidence suggests is ineffective, misleading, or counterproductive. Remove or rewrite it.
- CONDENSE overlapping or redundant points into a single, sharper formulation.
- CONSTRAINT: You may remove at most ONE existing element per stage section.  
When in doubt, keep it. Only delete when the Step2 evidence directly contradicts the guidance.                                                     
                                                                              
Then proceed to following STAGE 1-3 for new additions. 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIMIZATION PROTOCOL BY STAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Review the Step2 Reflection and extract the FUNDAMENTAL REASONING GAPS. Then, upgrade the three stages using the following high-level MAS concepts:

STAGE 1: TASK DECOMPOSITION MODULE (The "What")
Focus on elevating the planning paradigm. 
- Look for gaps in handling uncertainty, implicit constraints, or complex dependencies.
- Substantive upgrades could include: "Constraint Verification Matrices", "Assumption Elicitation", "Milestone-based Evaluation", or identifying "Critical Paths" vs. "Optional Enhancements".
- Rule: Do not dictate *how* to calculate things; dictate *how to structure the logical problem space*.

STAGE 2: AGENT ENGINEERING MODULE (The "Who")
Focus on elevating agent autonomy and interaction contracts.
- Look for gaps in agent capability limits, hallucination, or poor instruction comprehension.
- Substantive upgrades could include: Designing "Standardized Input/Output Schemas (Contracts)", implementing "Self-Correction/Reflection Prompts" within individual agents, defining "Boundary Conditions" (when an agent should stop and ask for help), or establishing "System Prompts" vs "Task Prompts" isolation.
- Rule: Do not dictate specific task execution steps; dictate *agent psychology and boundary definitions*.

STAGE 3: WORKFLOW & ORCHESTRATION MODULE (The "How")
Focus on elevating topological robustness and system resilience.
- Look for gaps in information loss between agents, infinite loops, or catastrophic pipeline failures.
- Substantive upgrades could include: Advanced Topologies (e.g., "Actor-Critic pairs for quality control", "Dynamic Routing based on confidence scores", "Hierarchical delegation"), "Context Compression" (preventing context window overflow during agent handoffs), or "Global vs. Local Memory Management".
- Rule: Do not dictate code debugging logic; dictate *dataflow architecture and topological resilience patterns*.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXECUTION & OUTPUT CONSTRAINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Evidence-Driven Abstraction: Every change must resolve a flaw found in Step2, but the solution MUST be abstracted into a universal systems-engineering principle.
2. Meaningful Depth: Do not just add adjectives. Add new sub-bullet points that introduce a concrete conceptual framework (e.g., instead of "Make dependencies clear", use "Build a Directed Acyclic Graph (DAG) mapping of logic state transitions").
3. Incremental evolution (hard limit): In **this** round, introduce at most **one** substantive conceptual upgrade per SKILL stage section (## 1, ## 2, ## 3 — each stage at most one focused improvement). Do not pile on multiple unrelated changes in a single pass.
4. Output Format: Produce ONLY the complete updated SKILL.md.


Format Requirements:
- MUST start directly with the YAML frontmatter (`---`).
- MUST preserve the exact same YAML keys and the exactly three-stage markdown structure (## 1, ## 2, ## 3).
- NO markdown code fences (```) around the entire output.
- NO preamble, NO explanations, NO summary of changes. Output raw SKILL.md text only.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEGIN YOUR EVOLUTION NOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""