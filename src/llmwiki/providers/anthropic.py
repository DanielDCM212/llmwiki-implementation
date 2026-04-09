from langchain_core.language_models import BaseChatModel

_DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model_name: str = _DEFAULT_MODEL):
        self._api_key = api_key
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_chat_model(self) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self._model_name,
            api_key=self._api_key,
            max_tokens=4096,
        )
