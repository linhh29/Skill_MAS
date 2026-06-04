"""Template framework for generated MAS forward code."""

import re

ImportTemplate = '''
from __future__ import annotations

from typing import Any, Callable
import json
from template.sub_agent import SubAgentRequest, build_workflow_subagent_prompt
'''

MASCodeTemplate_Sequential = '''
class MASSequentialWorkflowTemplate:
    """
    Executable MAS orchestration template for Sequential (Linear) Topology.
    
    LLM Code Generation Instructions:
    1. STAGE 1 & 2 (Agent Engineering): Instantiate all required sub-agents in __init__.
    2. STAGE 3 (Topology & Dataflow): Define the exact sequential execution order in forward_async.
       Pass the direct upstream node's output via the `upstream_outputs` dictionary.
    """

    def __init__(
        self,
        *,
        text_call_fn: Callable[[str], tuple[str, dict[str, Any]]],
        tool_call_fn: Callable[[SubAgentRequest], tuple[str, dict[str, Any]]] | None = None,
    ):
        self.text_call_fn = text_call_fn
        self.tool_call_fn = tool_call_fn
        
        # <STAGE_1_AND_2_AGENT_DEFINITIONS_START>
        self.agent_a = SubAgentRequest(
            role_name="agent_a",
            role_instruction="Replace with generated instruction for sub-task A.",
            user_prompt="Replace with generated user prompt for sub-task A.",
            tool_context={"execution_mode": "llm_only"},
        )
        self.agent_b = SubAgentRequest(
            role_name="agent_b",
            role_instruction="Replace with generated instruction for sub-task B.",
            user_prompt="Replace with generated user prompt for sub-task B.",
            tool_context={"execution_mode": "llm_only"},
        )
        self.final_agent = SubAgentRequest(
            role_name="final_agent",
            role_instruction="Replace with generated synthesis instruction for the final task.",
            user_prompt="Replace with generated user prompt for the final task.",
            tool_context={"execution_mode": "llm_only"},
        )
        # <STAGE_1_AND_2_AGENT_DEFINITIONS_END>

    async def forward_async(self, task_info: str) -> dict[str, Any]:
        """
        Execute the Sequential MAS workflow topology.
        Data flow: Task -> Agent A -> Agent B -> Final Agent
        """
        state: dict[str, Any] = {"task_info": task_info}

        # <STAGE_3_TOPOLOGY_EXECUTION_START>
        
        # --- 1. Execution Node A (Start Node) ---
        out_agent_a = await self.agent_a.execute_async(
            text_call_fn=self.text_call_fn,
            tool_call_fn=self.tool_call_fn,
            prompt_override=build_workflow_subagent_prompt(
                role_instruction=self.agent_a.role_instruction,
                base_user_prompt=self.agent_a.user_prompt,
                task_info=state["task_info"],
                upstream_outputs=None
            ),
        )
        state["out_agent_a"] = out_agent_a.text
        state["usage_agent_a"] = out_agent_a.usage

        # --- 2. Execution Node B (Depends on A) ---
        out_agent_b = await self.agent_b.execute_async(
            text_call_fn=self.text_call_fn,
            tool_call_fn=self.tool_call_fn,
            prompt_override=build_workflow_subagent_prompt(
                role_instruction=self.agent_b.role_instruction,
                base_user_prompt=self.agent_b.user_prompt,
                task_info=state["task_info"],
                upstream_outputs={
                    "agent_a_output": state["out_agent_a"]
                }
            ),
        )
        state["out_agent_b"] = out_agent_b.text
        state["usage_agent_b"] = out_agent_b.usage

        # --- 3. Execution Final Node (Depends on B) ---
        out_final = await self.final_agent.execute_async(
            text_call_fn=self.text_call_fn,
            tool_call_fn=self.tool_call_fn,
            prompt_override=build_workflow_subagent_prompt(
                role_instruction=self.final_agent.role_instruction,
                base_user_prompt=self.final_agent.user_prompt,
                task_info=state["task_info"],
                upstream_outputs={
                    "agent_b_output": state["out_agent_b"]
                }
            ),
        )
        state["out_final_agent"] = out_final.text
        state["usage_final_agent"] = out_final.usage

        # --- Finalizing Output ---
        state["final_output"] = state["out_final_agent"]
        state["usage_final"] = state["usage_final_agent"]
        
        # <STAGE_3_TOPOLOGY_EXECUTION_END>
        
        return state
'''



MASCodeTemplate_MultiDependency = '''
class MASMultiDependencyWorkflowTemplate:
    """
    Executable MAS orchestration template for Multi-Dependency (DAG) Topology.
    
    LLM Code Generation Instructions:
    1. STAGE 1 & 2 (Agent Engineering): Instantiate all required sub-agents in __init__.
    2. STAGE 3 (Topology & Dataflow): Route data correctly in forward_async. 
       Even for logically parallel branches, execute them sequentially via await for simplicity.
       For nodes with multiple upstream dependencies, map all required outputs 
       using the `upstream_outputs` dictionary.
    """

    def __init__(
        self,
        *,
        text_call_fn: Callable[[str], tuple[str, dict[str, Any]]],
        tool_call_fn: Callable[[SubAgentRequest], tuple[str, dict[str, Any]]] | None = None,
    ):
        self.text_call_fn = text_call_fn
        self.tool_call_fn = tool_call_fn
        
        # <STAGE_1_AND_2_AGENT_DEFINITIONS_START>
        self.agent_a = SubAgentRequest(
            role_name="agent_a",
            role_instruction="Replace with generated instruction for root task A.",
            user_prompt="Replace with generated user prompt for root task A.",
            tool_context={"execution_mode": "llm_only"},
        )
        self.agent_b = SubAgentRequest(
            role_name="agent_b",
            role_instruction="Replace with generated instruction for branch task B.",
            user_prompt="Replace with generated user prompt for branch task B.",
            tool_context={"execution_mode": "llm_only"},
        )
        self.agent_c = SubAgentRequest(
            role_name="agent_c",
            role_instruction="Replace with generated instruction for branch task C.",
            user_prompt="Replace with generated user prompt for branch task C.",
            tool_context={"execution_mode": "llm_only"},
        )
        self.final_agent = SubAgentRequest(
            role_name="final_agent",
            role_instruction="Replace with generated synthesis instruction merging B and C.",
            user_prompt="Replace with generated user prompt for the merge task.",
            tool_context={"execution_mode": "llm_only"},
        )
        # <STAGE_1_AND_2_AGENT_DEFINITIONS_END>

    async def forward_async(self, task_info: str) -> dict[str, Any]:
        """
        Execute the Multi-Dependency MAS workflow topology.
        Data flow (Diamond Shape): 
             Task -> Agent A
             Agent A -> Agent B (Branch 1)
             Agent A -> Agent C (Branch 2)
             Agent B + Agent C -> Final Agent
        Note: Execution is sequential for simplicity, but dataflow remains DAG.
        """
        state: dict[str, Any] = {"task_info": task_info}

        # <STAGE_3_TOPOLOGY_EXECUTION_START>
        
        # --- 1. Execution Node A (Root Node) ---
        out_agent_a = await self.agent_a.execute_async(
            text_call_fn=self.text_call_fn,
            tool_call_fn=self.tool_call_fn,
            prompt_override=build_workflow_subagent_prompt(
                role_instruction=self.agent_a.role_instruction,
                base_user_prompt=self.agent_a.user_prompt,
                task_info=state["task_info"],
                upstream_outputs=None
            ),
        )
        state["out_agent_a"] = out_agent_a.text
        state["usage_agent_a"] = out_agent_a.usage

        # --- 2. Execution Node B (Branch 1, depends on A) ---
        out_agent_b = await self.agent_b.execute_async(
            text_call_fn=self.text_call_fn,
            tool_call_fn=self.tool_call_fn,
            prompt_override=build_workflow_subagent_prompt(
                role_instruction=self.agent_b.role_instruction,
                base_user_prompt=self.agent_b.user_prompt,
                task_info=state["task_info"],
                upstream_outputs={
                    "agent_a_output": state["out_agent_a"]
                }
            ),
        )
        state["out_agent_b"] = out_agent_b.text
        state["usage_agent_b"] = out_agent_b.usage

        # --- 3. Execution Node C (Branch 2, depends on A, structurally independent of B) ---
        out_agent_c = await self.agent_c.execute_async(
            text_call_fn=self.text_call_fn,
            tool_call_fn=self.tool_call_fn,
            prompt_override=build_workflow_subagent_prompt(
                role_instruction=self.agent_c.role_instruction,
                base_user_prompt=self.agent_c.user_prompt,
                task_info=state["task_info"],
                upstream_outputs={
                    "agent_a_output": state["out_agent_a"]
                }
            ),
        )
        state["out_agent_c"] = out_agent_c.text
        state["usage_agent_c"] = out_agent_c.usage

        # --- 4. Execution Final Node (Merge Node, depends on BOTH B and C) ---
        out_final = await self.final_agent.execute_async(
            text_call_fn=self.text_call_fn,
            tool_call_fn=self.tool_call_fn,
            prompt_override=build_workflow_subagent_prompt(
                role_instruction=self.final_agent.role_instruction,
                base_user_prompt=self.final_agent.user_prompt,
                task_info=state["task_info"],
                upstream_outputs={
                    "agent_b_output": state["out_agent_b"],
                    "agent_c_output": state["out_agent_c"]
                }
            ),
        )
        state["out_final_agent"] = out_final.text
        state["usage_final_agent"] = out_final.usage

        # --- Finalizing Output ---
        state["final_output"] = state["out_final_agent"]
        state["usage_final"] = state["usage_final_agent"]
        
        # <STAGE_3_TOPOLOGY_EXECUTION_END>
        
        return state
'''


def stage3_llm_reference_bundle() -> str:
    """
    Stage-3 prompt payload: both topology examples only (no import block—runner prepends that later).

    Callers (``build.py``) embed this as ``template_text`` so the planner sees Sequential and
    DAG patterns before emitting ``mas_code`` (class body only).
    """
    instructions = (
        "Stage 3 reference — read before generating ``mas_code``:\n"
        "- Do **not** output import statements or ``from __future__``; the runner prepends a fixed "
        "import block before ``exec``. Emit **only** the workflow Python (typically one class).\n"
        "- Emit **one** workflow class whose name equals ``target_class_name`` from [STAGE_INPUT_JSON]. "
        "Choose structure from **either** topology below (Sequential pipeline vs diamond DAG with merge).\n"
        "- Use ``SubAgentRequest``, ``build_workflow_subagent_prompt`` (available after runner import "
        "prefix), state keys ``out_<role>``, ``usage_<role>``, ``final_output``, ``usage_final``.\n\n"
    )
    return (
        instructions
        + "[SEQUENTIAL_TOPOLOGY_TEMPLATE]\n"
        + MASCodeTemplate_Sequential.strip()
        + "\n\n[MULTI_DEPENDENCY_TOPOLOGY_TEMPLATE]\n"
        + MASCodeTemplate_MultiDependency.strip()
    )




def prepend_fixed_imports(mas_code: str) -> str:
    """Always prepend ``ImportTemplate`` before runner ``exec`` (model output should be class-only)."""
    return ImportTemplate.strip() + "\n\n" + mas_code.replace("from __future__ import annotations", "")