from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMWikiConfig:
    wiki_root: Path
    provider_name: str
    api_key: str
    model_name: str | None


def load_config(
    wiki_root: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> "LLMWikiConfig":
    """Build config from explicit args > env vars > defaults."""
    resolved_root = Path(
        wiki_root
        or os.environ.get("LLMWIKI_ROOT", "./wiki")
    )

    resolved_provider = (
        provider
        or os.environ.get("LLMWIKI_PROVIDER", "anthropic")
    ).lower()

    resolved_model = model or os.environ.get("LLMWIKI_MODEL") or None

    # API key: generic override first, then provider-specific env vars
    api_key = os.environ.get("LLMWIKI_API_KEY", "")
    if not api_key:
        provider_key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        env_var = provider_key_map.get(resolved_provider, "")
        api_key = os.environ.get(env_var, "")

    return LLMWikiConfig(
        wiki_root=resolved_root,
        provider_name=resolved_provider,
        api_key=api_key,
        model_name=resolved_model,
    )
