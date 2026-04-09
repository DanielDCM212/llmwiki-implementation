from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class IngestState(TypedDict):
    source_path: str
    source_content: str
    summary: str
    entities: list[dict]       # [{name, description, category}]
    concepts: list[dict]       # [{name, description}]
    pages_written: list[str]   # rel paths of pages written/updated
    index_content: str         # current index.md text (for update node)
    messages: Annotated[list[BaseMessage], add_messages]


class QueryState(TypedDict):
    question: str
    index_content: str
    relevant_pages: list[str]
    page_contents: dict[str, str]   # {rel_path: content}
    answer: str
    save_as_page: bool
    messages: Annotated[list[BaseMessage], add_messages]


class LintState(TypedDict):
    all_pages: dict[str, str]   # {rel_path: content}
    issues: list[dict]          # [{type, page, description, suggestion}]
    messages: Annotated[list[BaseMessage], add_messages]
