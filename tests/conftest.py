import pytest
from pathlib import Path
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from llmwiki.wiki import WikiManager


@pytest.fixture
def wiki(tmp_path: Path) -> WikiManager:
    wm = WikiManager(tmp_path / "wiki")
    wm.ensure_structure()
    return wm


@pytest.fixture
def fake_llm():
    """FakeListChatModel that cycles through canned responses."""
    def _make(*responses: str) -> FakeListChatModel:
        return FakeListChatModel(responses=list(responses))
    return _make
