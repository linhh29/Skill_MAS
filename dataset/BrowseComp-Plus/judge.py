"""LLM-as-judge for BrowseComp-Plus answers (calls Skill_MAS AsyncOpenAIClient.generate)."""

from __future__ import annotations

import json
import re
from typing import Any

from openai_client import AsyncOpenAIClient


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def _parse_judge_text(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {
            "extracted_final_answer": "None",
            "reasoning": "",
            "correct": "no",
            "confidence": 100,
            "raw_judgement": raw,
        }

    # Prefer JSON if model follows instruction.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            ext = str(obj.get("extracted_final_answer", "None"))
            reason = str(obj.get("reasoning", ""))
            corr = str(obj.get("correct", "no")).strip().lower()
            conf_raw = str(obj.get("confidence", "100"))
            conf_num = re.findall(r"\d+(?:\.\d+)?", conf_raw)
            conf = float(conf_num[0]) if conf_num else 100.0
            return {
                "extracted_final_answer": ext,
                "reasoning": reason,
                "correct": "yes" if corr == "yes" else "no",
                "confidence": conf,
                "raw_judgement": raw,
            }
    except Exception:
        pass

    # Fallback key-value parsing.
    def _extract(prefix: str) -> str:
        m = re.search(rf"(?im)^\s*{re.escape(prefix)}\s*:\s*(.*)$", raw)
        return m.group(1).strip() if m else ""

    ext = _extract("extracted_final_answer") or "None"
    corr = (_extract("correct") or "no").strip().lower()
    reason = _extract("reasoning")
    conf_raw = _extract("confidence") or "100"
    conf_num = re.findall(r"\d+(?:\.\d+)?", conf_raw)
    conf = float(conf_num[0]) if conf_num else 100.0

    return {
        "extracted_final_answer": ext,
        "reasoning": reason,
        "correct": "yes" if corr == "yes" else "no",
        "confidence": conf,
        "raw_judgement": raw,
    }


JUDGE_SYSTEM = (
    "You are a strict QA grader. "
    "Output only JSON with keys: extracted_final_answer, reasoning, correct, confidence."
)


async def judge_answer(
    *,
    judge_client: AsyncOpenAIClient,
    question: str,
    response: str,
    correct_answer: str,
) -> dict[str, Any]:
    """Score a model answer; token usage and cost come from ``judge_client.generate`` (Skill_MAS)."""
    prompt = GRADER_TEMPLATE.format(
        question=question.strip(),
        response=(response or "").strip(),
        correct_answer=(correct_answer or "").strip(),
    )
    text, usage = await judge_client.generate(
        user_prompt=prompt,
        system_prompt=JUDGE_SYSTEM,
        response_format={"type": "json_object"},
    )
    parsed = _parse_judge_text(text)
    parsed["score"] = 1 if parsed.get("correct") == "yes" else 0
    parsed["usage_totals"] = usage
    return parsed
