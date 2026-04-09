import os
import pytest
from pathlib import Path

from llmwiki.config import load_config


def test_defaults(monkeypatch):
    monkeypatch.delenv("LLMWIKI_ROOT", raising=False)
    monkeypatch.delenv("LLMWIKI_PROVIDER", raising=False)
    monkeypatch.delenv("LLMWIKI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    cfg = load_config()

    assert cfg.wiki_root == Path("./wiki")
    assert cfg.provider_name == "anthropic"
    assert cfg.api_key == ""
    assert cfg.model_name is None


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("LLMWIKI_ROOT", "/tmp/my-wiki")
    monkeypatch.setenv("LLMWIKI_PROVIDER", "openai")
    monkeypatch.setenv("LLMWIKI_API_KEY", "generic-key")
    monkeypatch.setenv("LLMWIKI_MODEL", "gpt-4o-mini")

    cfg = load_config()

    assert cfg.wiki_root == Path("/tmp/my-wiki")
    assert cfg.provider_name == "openai"
    assert cfg.api_key == "generic-key"
    assert cfg.model_name == "gpt-4o-mini"


def test_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("LLMWIKI_ROOT", "/env/wiki")
    monkeypatch.setenv("LLMWIKI_PROVIDER", "google")

    cfg = load_config(wiki_root="/explicit/wiki", provider="anthropic")

    assert cfg.wiki_root == Path("/explicit/wiki")
    assert cfg.provider_name == "anthropic"


def test_provider_specific_api_key_fallback(monkeypatch):
    monkeypatch.delenv("LLMWIKI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key")

    cfg = load_config(provider="anthropic")

    assert cfg.api_key == "ant-key"


def test_provider_specific_key_per_provider(monkeypatch):
    monkeypatch.delenv("LLMWIKI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    assert load_config(provider="google").api_key == "google-key"
    assert load_config(provider="openai").api_key == "oai-key"


def test_generic_key_takes_precedence_over_provider_key(monkeypatch):
    monkeypatch.setenv("LLMWIKI_API_KEY", "override")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "specific")

    cfg = load_config(provider="anthropic")

    assert cfg.api_key == "override"


def test_provider_name_lowercased(monkeypatch):
    monkeypatch.setenv("LLMWIKI_PROVIDER", "Anthropic")

    cfg = load_config()

    assert cfg.provider_name == "anthropic"
