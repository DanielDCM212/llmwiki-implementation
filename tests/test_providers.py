import pytest
from pathlib import Path

from llmwiki.config import LLMWikiConfig, load_config
from llmwiki.providers import get_provider
from llmwiki.providers.base import LLMProvider
from llmwiki.providers.anthropic import AnthropicProvider
from llmwiki.providers.google import GoogleGenAIProvider
from llmwiki.providers.openai import OpenAIProvider


def _cfg(provider: str, model: str | None = None) -> LLMWikiConfig:
    return LLMWikiConfig(
        wiki_root=Path("./wiki"),
        provider_name=provider,
        api_key="test-key",
        model_name=model,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_returns_anthropic():
    p = get_provider(_cfg("anthropic"))
    assert isinstance(p, AnthropicProvider)
    assert p.name == "anthropic"


def test_factory_returns_google():
    p = get_provider(_cfg("google"))
    assert isinstance(p, GoogleGenAIProvider)
    assert p.name == "google"


def test_factory_returns_openai():
    p = get_provider(_cfg("openai"))
    assert isinstance(p, OpenAIProvider)
    assert p.name == "openai"


def test_factory_raises_on_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider(_cfg("unknown"))


def test_factory_uses_explicit_model():
    p = get_provider(_cfg("anthropic", model="claude-haiku-4-5-20251001"))
    assert p.model_name == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Default models
# ---------------------------------------------------------------------------

def test_anthropic_default_model():
    p = get_provider(_cfg("anthropic"))
    assert "claude" in p.model_name


def test_google_default_model():
    p = get_provider(_cfg("google"))
    assert "gemini" in p.model_name


def test_openai_default_model():
    p = get_provider(_cfg("openai"))
    assert "gpt" in p.model_name


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider_name", ["anthropic", "google", "openai"])
def test_providers_satisfy_protocol(provider_name):
    p = get_provider(_cfg(provider_name))
    assert isinstance(p, LLMProvider)


# ---------------------------------------------------------------------------
# get_chat_model (Anthropic only — others need real network)
# ---------------------------------------------------------------------------

def test_anthropic_get_chat_model_returns_chat_model():
    from langchain_core.language_models import BaseChatModel

    p = AnthropicProvider(api_key="test-key")
    model = p.get_chat_model()

    assert isinstance(model, BaseChatModel)
