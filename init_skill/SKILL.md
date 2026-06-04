---
name: unified_meta_agent_skill
description: "A foundational meta-agent skill for generating Multi-Agent Systems (MAS). It systematically drives the process from conceptual task decomposition to agent engineering, and finally to workflow orchestration."
tags:
  - meta-agent
  - task-decomposition
  - agent-engineering
  - workflow-orchestration
inputs:
  - user_query
---

## 1. Task Decomposition Module (The "What")
*Core Objective: Analyze the user query and break it down into a logical blueprint.*

- **Intent & Scope Analysis:** Understand the macro objective, identify core requirements, and define the boundaries of the task.
- **Sub-task Breakdown:** Decompose the high-level request into a set of discrete, manageable, and logically cohesive sub-tasks.
- **Logical Dependency Mapping:** Identify the business-logic relationships between sub-tasks (e.g., prerequisite, parallel, or iterative). *Note: This focuses on logical order, not system dataflow.*
- **Success Criteria:** Define clear objective outcomes for each sub-task to ensure evaluability.

## 2. Agent Engineering Module (The "Who")
*Core Objective: Design specialized sub-agents tailored for the sub-tasks defined in Stage 1.*

- **Role Profiling:** Assign a unique identity and specialized role to each sub-agent based on its target sub-task.
- **Instruction Design:** Draft precise system prompts/instructions. Define the agent's specific goals, behavioral boundaries, and output expectations.
- **Input Context Framing:** Specify what contextual information this agent requires from the user or the global task to begin its work.

## 3. Workflow & Orchestration Module (The "How")
*Core Objective: Wire the distinct agents from Stage 2 into a functional, executable Multi-Agent System (MAS).*

- **Architectural Topology:** You can design the optimal MAS architecture (e.g., Sequential Pipeline, Router-based, Hierarchical, or Blackboard) based on Stage 1's logical dependencies. For those complex but important sub-tasks, you can design localized different topology design using the instantiated agents. For example, you can define a iterative loops for those sub-tasks that need a cycle check to ensure high quality. Or you can call the same agent multiple times to generate diverse outputs and give all of thems to the following sub-agents.
- **Dataflow & State Management:** Define the exact I/O mapping. Specify how the output schema of one agent transforms into the input payload/prompt of downstream agents. Determine how global context (memory/state) is maintained.
- **Executable Generation:** Output the final orchestration logic/code structure that binds the agents, tools, and dataflow into a ready-to-run system.