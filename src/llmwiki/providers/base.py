from typing import Protocol, runtime_checkable
from langchain_core.language_models import BaseChatModel


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def get_chat_model(self) -> BaseChatModel: ...
