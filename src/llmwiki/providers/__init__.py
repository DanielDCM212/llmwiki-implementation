from llmwiki.config import LLMWikiConfig
from llmwiki.providers.base import LLMProvider


def get_provider(config: LLMWikiConfig) -> LLMProvider:
    """Factory: return the provider matching config.provider_name."""
    match config.provider_name:
        case "anthropic":
            from llmwiki.providers.anthropic import AnthropicProvider
            return AnthropicProvider(
                api_key=config.api_key,
                model_name=config.model_name or "claude-sonnet-4-6",
            )
        case "google":
            from llmwiki.providers.google import GoogleGenAIProvider
            return GoogleGenAIProvider(
                api_key=config.api_key,
                model_name=config.model_name or "gemini-2.0-flash",
            )
        case "openai":
            from llmwiki.providers.openai import OpenAIProvider
            return OpenAIProvider(
                api_key=config.api_key,
                model_name=config.model_name or "gpt-4o",
            )
        case _:
            raise ValueError(
                f"Unknown provider '{config.provider_name}'. "
                "Valid options: anthropic, google, openai"
            )


__all__ = ["get_provider", "LLMProvider"]
