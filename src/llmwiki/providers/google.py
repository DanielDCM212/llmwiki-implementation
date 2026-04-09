from langchain_core.language_models import BaseChatModel

_DEFAULT_MODEL = "gemini-2.0-flash"


class GoogleGenAIProvider:
    name = "google"

    def __init__(self, api_key: str, model_name: str = _DEFAULT_MODEL):
        self._api_key = api_key
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_chat_model(self) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=self._model_name,
            google_api_key=self._api_key,
        )
