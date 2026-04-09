import json
import re
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from llmwiki.prompts import (
    FIND_RELEVANT_PAGES,
    GENERATE_ANSWER_PAGE,
    SYNTHESIZE_ANSWER,
    UPDATE_INDEX,
)
from llmwiki.state import QueryState
from llmwiki.wiki import WikiManager


def _load_index_node(wiki: WikiManager):
    def _run(state: QueryState) -> dict:
        return {"index_content": wiki.read_index()}
    return _run


def _find_relevant_pages_node(llm: BaseChatModel):
    chain = FIND_RELEVANT_PAGES | llm | StrOutputParser()

    def _run(state: QueryState) -> dict:
        raw = chain.invoke({
            "question": state["question"],
            "index_content": state["index_content"],
        })
        try:
            pages = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            pages = json.loads(match.group()) if match else []
        return {"relevant_pages": pages if isinstance(pages, list) else []}
    return _run


def _load_pages_node(wiki: WikiManager):
    def _run(state: QueryState) -> dict:
        contents = {}
        for rel_path in state.get("relevant_pages", []):
            content = wiki.read_page(rel_path)
            if content:
                contents[rel_path] = content
        return {"page_contents": contents}
    return _run


def _synthesize_answer_node(llm: BaseChatModel):
    chain = SYNTHESIZE_ANSWER | llm | StrOutputParser()

    def _run(state: QueryState) -> dict:
        page_contents_text = "\n\n---\n\n".join(
            f"**{path}**\n\n{content}"
            for path, content in state.get("page_contents", {}).items()
        )
        answer = chain.invoke({
            "question": state["question"],
            "page_contents": page_contents_text or "(no relevant pages found)",
        })
        return {"answer": answer}
    return _run


def _save_answer_page_node(wiki: WikiManager, llm: BaseChatModel):
    chain = GENERATE_ANSWER_PAGE | llm | StrOutputParser()
    index_chain = UPDATE_INDEX | llm | StrOutputParser()

    def _run(state: QueryState) -> dict:
        page_content = chain.invoke({
            "question": state["question"],
            "answer": state["answer"],
        })
        slug = wiki.slugify(state["question"][:60])
        rel_path = f"concepts/{slug}.md"
        wiki.write_page(rel_path, page_content)

        updated_index = index_chain.invoke({
            "current_index": state["index_content"],
            "pages_summary": f"{rel_path} | {state['question'][:60]}",
        })
        wiki.update_index(updated_index)
        wiki.append_log("query+save", state["question"][:80])
        return {}
    return _run


def _route_save(state: QueryState) -> str:
    return "save" if state.get("save_as_page") else "end"


def build_query_graph(wiki: WikiManager, llm: BaseChatModel):
    graph = StateGraph(QueryState)

    graph.add_node("load_index", _load_index_node(wiki))
    graph.add_node("find_relevant_pages", _find_relevant_pages_node(llm))
    graph.add_node("load_pages", _load_pages_node(wiki))
    graph.add_node("synthesize_answer", _synthesize_answer_node(llm))
    graph.add_node("save_answer_page", _save_answer_page_node(wiki, llm))

    graph.set_entry_point("load_index")
    graph.add_edge("load_index", "find_relevant_pages")
    graph.add_edge("find_relevant_pages", "load_pages")
    graph.add_edge("load_pages", "synthesize_answer")
    graph.add_conditional_edges(
        "synthesize_answer",
        _route_save,
        {"save": "save_answer_page", "end": END},
    )
    graph.add_edge("save_answer_page", END)

    return graph.compile()
