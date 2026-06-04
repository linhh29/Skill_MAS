import os
import yaml
from pathlib import Path
from deep_research_bench.drb_runtime import load_pricing_table

from vita.skill_mas_paths import skill_mas_model_config_path

_models_yaml_path = Path(__file__).parent / "models.yaml"
if os.environ.get("VITA_MODEL_CONFIG_PATH", None):
    _models_yaml_path = os.environ.get("VITA_MODEL_CONFIG_PATH")
_skill_mas_model_config_path = skill_mas_model_config_path()

if not os.path.exists(str(_models_yaml_path)):
    raise FileNotFoundError(
        f"Model configuration file ({_models_yaml_path}) dose not exists, you should create it first.")


def _deep_merge_dict(base_dict: dict, override_dict: dict) -> dict:
    result = base_dict.copy()

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value

    return result


def _chat_completions_url(base_url: str) -> str:
    """VitaBench ``generate()`` expects the full OpenAI-compatible chat URL."""
    b = (base_url or "").strip().rstrip("/")
    if not b:
        return b
    if b.endswith("/chat/completions"):
        return b
    return b + "/chat/completions"


def _resolve_model_row(table: dict, model_name: str) -> dict:
    """Case-insensitive lookup for one model row in Skill_MAS model_config."""
    row = table.get(model_name)
    if isinstance(row, dict):
        return dict(row)
    lk = (model_name or "").strip().lower()
    for k, v in table.items():
        if isinstance(v, dict) and str(k).lower() == lk:
            return dict(v)
    raise ValueError(f"Model {model_name!r} is not defined in Skill_MAS model_config.json")


def _apply_skill_mas_model_row(merged_config: dict, model_row: dict) -> None:
    """Overlay Skill_MAS row values by model name."""
    if not isinstance(model_row, dict):
        return
    api_key = model_row.get("api_key")
    base_url = model_row.get("base_url") or merged_config.get("base_url")
    if api_key and base_url:
        merged_config["base_url"] = _chat_completions_url(str(base_url))
        merged_config["headers"] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    # Runtime generation fields supported by Skill_MAS async client.
    for k in ("temperature", "max_tokens", "reasoning_effort"):
        if model_row.get(k) is not None:
            merged_config[k] = model_row[k]
    # Keep pricing metadata attached to per-model config for visibility.
    for k in ("input_per_1m", "output_per_1m", "input_per_1k", "output_per_1k"):
        if model_row.get(k) is not None:
            merged_config[k] = model_row[k]


try:
    model_table = load_pricing_table(_skill_mas_model_config_path)
    with open(_models_yaml_path, 'r') as f:
        models_config_yaml = yaml.load(f, Loader=yaml.FullLoader)

    default_model_config = models_config_yaml.get('default', {})

    models = {"default": default_model_config}
    for model in models_config_yaml.get('models', []):
        model_name = model['name']
        merged_config = _deep_merge_dict(default_model_config, model)
        model_row = {}
        if isinstance(model_table, dict):
            try:
                model_row = _resolve_model_row(model_table, model_name)
            except Exception:
                # Keep name visible in registry; runtime call path will fail fast if model has no Skill_MAS row.
                print(f"Warning: model {model_name!r} not found in Skill_MAS model_config.json")
        _apply_skill_mas_model_row(merged_config, model_row)
        models[model_name] = merged_config

    if os.environ.get("VITA_SUPPRESS_MODEL_LIST", "").strip().lower() not in ("1", "true", "yes", "on"):
        print(f"Available models: {list(models.keys())}")

except FileNotFoundError:
    print(f"Warning: models.yaml not found at {_models_yaml_path}")
    models = {}
except Exception as e:
    print(f"Error loading models.yaml: {e}")
    models = {}

# SIMULATION
DEFAULT_MAX_STEPS = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_ERRORS = 10
DEFAULT_SEED = 300
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_NUM_TRIALS = 1
DEFAULT_SAVE_TO = None
DEFAULT_LOG_LEVEL = "DEBUG"
DEFAULT_LANGUAGE = "chinese"
DEFAULT_EVALUATION_TYPE = "trajectory"

# LLM
DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_AGENT = "gpt-4.1"
DEFAULT_LLM_USER = "gpt-4.1"
DEFAULT_LLM_EVALUATOR = "anthropic.claude-3.7-sonnet"
