import os
import re
import json
from html import unescape
from typing import Optional, Dict, Any
from pathlib import Path
import requests
import logging
from openai import OpenAI
try:
    from deep_research_bench.drb_runtime import load_pricing_table
except ImportError:
    from drb_runtime import load_pricing_table


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


logging.getLogger("httpx").setLevel(logging.WARNING)

# Read API settings from environment variables
API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE = os.environ.get("OPENAI_API_BASE", "")
FACT_Model = os.environ.get("DRB_FACT_MODEL", "gpt-4o-mini")
Model = os.environ.get("DRB_RACE_MODEL", "gpt-4o")
_SKILL_MAS_ROOT = Path(__file__).resolve().parents[3]
_MODEL_CONFIG_PATH = _SKILL_MAS_ROOT / "skill_mas" / "model_config.json"
_PRICING = load_pricing_table(_MODEL_CONFIG_PATH)


def _load_model_config() -> dict[str, Any]:
    if not _MODEL_CONFIG_PATH.is_file():
        return {}
    data = json.loads(_MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}

class AIClient:
    
    def __init__(self, api_key=API_KEY, model=Model, base_url: Optional[str] = None):
        self.model_config = _load_model_config()
        row = self._row_for(model)
        try:
            from Skill_MAS.utils.secrets_resolve import resolve_secret
        except ImportError:
            resolve_secret = lambda v, **_: (v or os.environ.get("OPENAI_API_KEY"))  # type: ignore
        raw_key = api_key or (row.get("api_key") if isinstance(row, dict) else None)
        self.api_key = resolve_secret(str(raw_key) if raw_key is not None else None) or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided! Please set OPENAI_API_KEY environment variable.")
        self.base_url = (
            base_url
            or (row.get("base_url") if isinstance(row, dict) else None)
            or os.environ.get("OPENAI_API_BASE", API_BASE or None)
        )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model
        if not isinstance(row, dict) or not row:
            row = _PRICING.get(self.model, {}) if isinstance(_PRICING, dict) else {}
        self.reasoning_effort = row.get("reasoning_effort") if isinstance(row, dict) else None

    def _row_for(self, model_id: str) -> Dict[str, Any]:
        mc = self.model_config if isinstance(self.model_config, dict) else {}
        key = (model_id or "").strip()
        if key and key in mc and isinstance(mc[key], dict):
            return dict(mc[key])
        if key:
            lower_map = {
                str(k).lower(): v
                for k, v in mc.items()
                if isinstance(v, dict) and not str(k).startswith("_")
            }
            r = lower_map.get(key.lower())
            if isinstance(r, dict):
                return dict(r)
        return {}
        
    def generate(self, user_prompt: str, system_prompt: str = "", model: Optional[str] = None) -> str:
        model_to_use = model or self.model
        row = self._row_for(model_to_use)
        reasoning = row.get("reasoning_effort") if isinstance(row, dict) else None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            create_kw: Dict[str, Any] = {"model": model_to_use, "messages": messages}
            if reasoning:
                create_kw["reasoning_effort"] = reasoning
            elif self.reasoning_effort and model_to_use == self.model:
                create_kw["reasoning_effort"] = self.reasoning_effort
            eb = row.get("extra_body") if isinstance(row, dict) else None
            if isinstance(eb, dict):
                create_kw["extra_body"] = dict(eb)
            response = self.client.chat.completions.create(**create_kw)
            return response.choices[0].message.content or ""
        except Exception as e:
            raise Exception(f"Failed to generate content: {str(e)}")

def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def scrape_url(url: str) -> Dict[str, Any]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        html = resp.text or ""

        title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
        title = _strip_html(title_match.group(1)) if title_match else ""

        desc_match = re.search(
            r'(?is)<meta[^>]+name=["\']description["\'][^>]*content=["\'](.*?)["\']',
            html,
        )
        description = _strip_html(desc_match.group(1)) if desc_match else ""
        content = _strip_html(html)
        if len(content) > 20000:
            content = content[:20000]

        return {
            "url": url,
            "title": title,
            "description": description,
            "content": content,
            "publish_time": "unknown",
        }
    except Exception as e:
        logger.error(str(e))
        return {
            "url": url,
            "content": "",
            "error": str(e),
        }
    
def call_model(user_prompt: str) -> str:
    client = AIClient(model=FACT_Model)
    return client.generate(user_prompt)

if __name__ == "__main__":
    url = ""
    result = scrape_url(url)
    print(result)