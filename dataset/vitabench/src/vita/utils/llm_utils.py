import copy
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger
import requests

from deep_research_bench.drb_runtime import estimate_cost_usd, load_pricing_table

from vita.config import DEFAULT_MAX_RETRIES
from vita.skill_mas_paths import skill_mas_model_config_path
from vita.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from vita.environment.tool import Tool


class DictToObject:
    """
    Convert dictionary to object with attribute access
    Usage:
    response_obj = DictToObject(response)
    print(response_obj.choices[0].message.content)  # Instead of response["choices"][0]["message"]["content"]
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            elif isinstance(value, list):
                setattr(self, key, [DictToObject(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def to_dict(self):
        """Convert object back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObject):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, DictToObject) else item for item in value]
            else:
                result[key] = value
        return result


_SKILL_MAS_MODEL_CONFIG_PATH = skill_mas_model_config_path()
_SKILL_MAS_PRICING = load_pricing_table(_SKILL_MAS_MODEL_CONFIG_PATH)


def _load_skill_mas_model_config() -> dict[str, Any]:
    p = _SKILL_MAS_MODEL_CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Skill_MAS model_config not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Skill_MAS model_config must be a JSON object: {p}")
    return data


_SKILL_MAS_MODEL_CONFIG = _load_skill_mas_model_config()


def _resolve_model_row(model: str) -> dict[str, Any]:
    key = (model or "").strip()
    if key and isinstance(_SKILL_MAS_MODEL_CONFIG.get(key), dict):
        return dict(_SKILL_MAS_MODEL_CONFIG[key])
    lower_map = {
        str(k).lower(): v
        for k, v in _SKILL_MAS_MODEL_CONFIG.items()
        if isinstance(v, dict) and not str(k).startswith("_")
    }
    row = lower_map.get(key.lower())
    if isinstance(row, dict):
        return dict(row)
    known = sorted(str(k) for k, v in _SKILL_MAS_MODEL_CONFIG.items() if isinstance(v, dict) and not str(k).startswith("_"))
    raise ValueError(f"Model {model!r} not found in Skill_MAS model_config.json. Available: {known}")


def _to_chat_completions_url(base_url: str) -> str:
    b = (base_url or "").strip().rstrip("/")
    if not b:
        return b
    if b.endswith("/chat/completions"):
        return b
    return b + "/chat/completions"


def _resolve_api_key_from_config(raw: str | None) -> str:
    try:
        from Skill_MAS.utils.secrets_resolve import resolve_secret
    except ImportError:
        pkg_root = Path(__file__).resolve().parents[5]
        import sys
        if str(pkg_root.parent) not in sys.path:
            sys.path.insert(0, str(pkg_root.parent))
        from Skill_MAS.utils.secrets_resolve import resolve_secret
    resolved = resolve_secret(str(raw) if raw is not None else None)
    return (resolved or "").strip()


def _build_transport_from_skill_mas(model: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
    row = _resolve_model_row(model)
    api_key = _resolve_api_key_from_config(row.get("api_key"))
    base_url = _to_chat_completions_url(str(row.get("base_url") or ""))
    if not api_key or not base_url:
        raise ValueError(
            f"Model {model!r} is missing api_key/base_url in Skill_MAS model_config.json."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    runtime_fields: dict[str, Any] = {}
    # Use Skill_MAS row as defaults; per-call kwargs override.
    # Exclude non-generation metadata / transport keys.
    excluded = {
        "api_key",
        "base_url",
        "input_per_1m",
        "output_per_1m",
        "input_per_1k",
        "output_per_1k",
        "_comment",
    }
    for k, v in row.items():
        if k in excluded:
            continue
        if v is not None:
            runtime_fields[k] = v
    allowed_override = set(runtime_fields.keys()) | {
        "temperature",
        "max_tokens",
        "reasoning_effort",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
        "n",
    }
    for k, v in kwargs.items():
        if k in allowed_override and v is not None:
            runtime_fields[k] = v
    return base_url, headers, runtime_fields


def get_response_cost(usage, model) -> float:
    """
    Estimated cost in USD for one completion, using Skill_MAS/skill_mas/model_config.json
    (USD per 1M prompt / completion tokens).
    Unknown model ids raise an error (strict model mapping).
    """
    if usage is None:
        return 0.0
    cost, _, _, _ = estimate_cost_usd(
        model=model,
        usage={
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        },
        table=_SKILL_MAS_PRICING,
    )
    return float(cost)


def get_response_usage(response) -> dict:
    """
    Always returns token counts (zeros if the API omitted usage).
    """
    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
    }


def _requires_thought_signature_replay(model: str | None) -> bool:
    """Only Gemini/GPT models routed through qingyuntop need raw tool_call replay."""
    name = (model or "").strip()
    if not name:
        return False
    lower = name.lower()
    if not (lower.startswith("gemini") or lower.startswith("gpt")):
        return False
    try:
        row = _resolve_model_row(name)
    except ValueError:
        return False
    base_url = str(row.get("base_url") or "").lower()
    return "qingyuntop" in base_url


def _assistant_tool_calls_for_api(
    message: AssistantMessage,
    *,
    preserve_thought_signature: bool,
) -> list[dict[str, Any]] | None:
    if not message.is_tool_call():
        return None
    if preserve_thought_signature and message.raw_data is not None:
        raw_message = message.raw_data.get("message")
        if isinstance(raw_message, dict):
            raw_tool_calls = raw_message.get("tool_calls")
            if raw_tool_calls:
                return copy.deepcopy(raw_tool_calls)
    return [
        {
            "id": tc.id,
            "name": tc.name,
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments),
            },
            "type": "function",
        }
        for tc in message.tool_calls
    ]


def format_messages(messages: list[Message], *, model: str | None = None) -> list[dict]:
    preserve_thought_signature = _requires_thought_signature_replay(model)
    messages_formatted = []
    for message in messages:
        if isinstance(message, UserMessage):
            messages_formatted.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = _assistant_tool_calls_for_api(
                message,
                preserve_thought_signature=preserve_thought_signature,
            )
            messages_formatted.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
            # add interleaved thinking content if exists
            if message.raw_data is not None and message.raw_data.get("message") is not None:
                reasoning_content = message.raw_data["message"].get("reasoning_content")
                if reasoning_content:
                    messages_formatted[-1]["reasoning_content"] = reasoning_content
        elif isinstance(message, ToolMessage):
            messages_formatted.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                    "name": message.name,
                }
            )
        elif isinstance(message, SystemMessage):
            messages_formatted.append({"role": "system", "content": message.content})
    return messages_formatted


def to_claude_think_official(messages_formatted: list[dict], messages: list[Message]) -> list[dict]:
    try:
        idx = -2 if messages_formatted[-1]["role"] == "tool" else -1
        content = [
            {
                "type": "text",
                "text": messages[idx].content
            }
        ]
        if messages[idx].raw_data["message"].get("tool_calls", []):
            content.append(
                {
                    "type": "tool_use",
                    "id": messages[idx].raw_data["message"]["tool_calls"][0]['id'],
                    "name": messages[idx].raw_data["message"]["tool_calls"][0]["function"]["name"],
                    "input": messages[idx].raw_data["message"]["tool_calls"][0]["function"]["arguments"]
                }
            )
        reasoning_content = messages[idx].raw_data["message"].get("reasoning_content", None) or messages[idx].raw_data["message"].get("reasoning", None)
        if reasoning_content:
            content.append(
                {
                    "type": "thinking",
                    "thinking": reasoning_content
                }
            )

        messages_formatted[idx]["content"] = content
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)

    return messages_formatted


def kwargs_adapter(data: dict, enable_think: False, messages: list) -> dict:
    if "claude" in data["model"]:
        if not enable_think:
            data["thinking"] = {"type": "disabled"}
        else:
            data["messages"] = to_claude_think_official(data["messages"], messages)
    return data


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    enable_think: bool = False,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        enable_think: Whether to enable think mode for the agent.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    try:
        if kwargs.get("num_retries") is None:
            kwargs["num_retries"] = DEFAULT_MAX_RETRIES
        messages_formatted = format_messages(messages, model=model)
        tools = [tool.openai_schema for tool in tools] if tools else None
        if tools and tool_choice is None:
            tool_choice = "auto"
        try:
            response_format = kwargs.get("response_format")
            base_url, headers, runtime_fields = _build_transport_from_skill_mas(model, kwargs)
            data = {
                "model": model,
                "messages": messages_formatted,
                "stream": False,
                "tools": tools,
                "tool_choice": tool_choice,
            }
            data.update(runtime_fields)
            # ``model_config.json`` stores DashScope extras as ``extra_body``; the OpenAI Python
            # client merges that dict into the JSON body root. Match that here for raw HTTP.
            _eb = data.pop("extra_body", None)
            if isinstance(_eb, dict):
                data.update(_eb)
            if response_format is not None:
                data["response_format"] = response_format
            data = kwargs_adapter(data, enable_think, messages)

            max_retries = 3
            retry_delay = 1
            for attempt in range(max_retries + 1):
                try:
                    response = requests.post(base_url, json=data, headers=headers, timeout=(10, 600))
                    print(response)
                    if response.status_code != 500:
                        response_json = response.json()
                        # Check for API errors in the response
                        if 'error' in response_json:
                            error_info = response_json['error']
                            error_msg = error_info.get('message', 'Unknown API error')
                            error_type = error_info.get('type', 'unknown_error')
                            error_code = error_info.get('code', '')
                            raise ValueError(
                                f"API returned error (type: {error_type}, code: {error_code}): {error_msg}"
                            )
                        response = response_json
                        break

                    if attempt < max_retries:
                        logger.warning(f"API returned 500 error, attempt {attempt + 1} retry, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        response.raise_for_status()

                except requests.exceptions.RequestException as e:
                    if attempt < max_retries:
                        logger.warning(f"Request exception, attempt {attempt + 1} retry, retrying in {retry_delay} seconds... Error: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise e
        except Exception as e:
            logger.error(e)
            raise e
        
        # Check if response has the expected structure
        if 'choices' not in response or len(response.get('choices', [])) == 0:
            error_msg = response.get('error', {}).get('message', 'Unknown error') if 'error' in response else 'No choices in response'
            raise ValueError(f"Invalid API response: {error_msg}. Full response: {response}")
        
        usage = get_response_usage(response)
        cost = float(get_response_cost(usage, model))
        try:
            response = response['choices'][0]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to extract response from API: {e}. Response structure: {response}")
        
        if 'message' not in response:
            raise ValueError(f"Response missing 'message' field. Response: {response}")
        
        assert response['message']['role'] == "assistant", (
            "The response should be an assistant message"
        )
        message_obj = response.get("message", {})
        content_preview = message_obj.get("content")
        reasoning_preview = message_obj.get("reasoning_content")
        logger.debug(
            "[LLM.generate] response-shape "
            f"model={model} "
            f"content_present={bool(content_preview)} "
            f"reasoning_present={bool(reasoning_preview)} "
            f"tool_calls_count={len(message_obj.get('tool_calls') or [])} "
            f"finish_reason={response.get('finish_reason')}"
        )
        content = response['message'].get('content')
        tool_calls = response['message'].get('tool_calls') or []
        tool_calls = [
            ToolCall(
                id=tool_call.get('id'),
                name=tool_call.get('function', {}).get('name'),
                arguments=json.loads(tool_call.get('function', {}).get('arguments')) if tool_call.get('function', {}).get('arguments') else {},
            )
            for tool_call in tool_calls
        ]
        tool_calls = tool_calls or None
        message = AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            cost=cost,
            usage=usage,
            raw_data={
                **response,
                "_response_diagnostics": {
                    "finish_reason": response.get("finish_reason"),
                    "content_present": bool(content),
                    "reasoning_present": bool(
                        response.get("message", {}).get("reasoning_content")
                    ),
                    "tool_calls_count": len(response.get("message", {}).get("tool_calls") or []),
                },
            },
        )
        return message
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(e)


def get_cost(messages: list[Message]) -> tuple[float, float]:
    """
    Sum LLM usage costs along the trajectory. Tool and system messages contribute 0.
    Missing per-message cost is treated as 0 so pure tool rounds do not zero out the total.
    """
    agent_cost = 0.0
    user_cost = 0.0
    for message in messages:
        if isinstance(message, (ToolMessage, SystemMessage)):
            continue
        if isinstance(message, MultiToolMessage):
            continue
        c = 0.0 if message.cost is None else float(message.cost)
        if isinstance(message, AssistantMessage):
            agent_cost += c
        elif isinstance(message, UserMessage):
            user_cost += c
        else:
            logger.debug(f"get_cost: unhandled message type {type(message)}, treating as 0 cost")
    return agent_cost, user_cost
