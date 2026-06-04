"""Print one model field from Skill_MAS skill_mas/model_config.json."""

from __future__ import annotations

import argparse

from Skill_MAS.core.model_config_runtime import model_runtime_params
from Skill_MAS.utils.secrets_resolve import resolve_secret


def main() -> None:
    parser = argparse.ArgumentParser(description="Read one runtime parameter from model_config.json")
    parser.add_argument("--model", required=True, help="Model name key in model_config.json")
    parser.add_argument("--key", required=True, help="Field name, e.g. api_key/base_url/temperature")
    args = parser.parse_args()

    row = model_runtime_params(args.model)
    value = row.get(args.key, "")
    if args.key == "api_key":
        value = resolve_secret(None if value is None else str(value)) or ""
    print("" if value is None else str(value))


if __name__ == "__main__":
    main()
