import json
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph

from llmwiki.prompts import (
    EXTRACT_ENTITIES_AND_CONCEPTS,
    SUMMARIZE_SOURCE,
    UPDATE_CONCEPT_PAGE,
    UPDATE_ENTITY_PAGE,
    UPDATE_INDEX,
)
from llmwiki.state import IngestState
from llmwiki.wiki import WikiManager


def _load_source_node(wiki: WikiManager):
    def _run(state: IngestState) -> dict:
        content = wiki.read_source_file(Path(state["source_path"]))
        return {"source_content": content, "pages_written": []}
    return _run


def _summarize_node(llm: BaseChatModel):
    chain = SUMMARIZE_SOURCE | llm | StrOutputParser()

    def _run(state: IngestState) -> dict:
        summary = chain.invoke({"source_content": state["source_content"]})
        return {"summary": summary}
    return _run


def _extract_node(llm: BaseChatModel):
    chain = EXTRACT_ENTITIES_AND_CONCEPTS | llm | StrOutputParser()

    def _run(state: IngestState) -> dict:
        raw = chain.invoke({"summary": state["summary"]})
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Tolerate minor formatting issues
            import re
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group()) if match else {}
        return {
            "entities": data.get("entities", []),
            "concepts": data.get("concepts", []),
        }
    return _run


def _write_summary_node(wiki: WikiManager):
    def _run(state: IngestState) -> dict:
        source_name = Path(state["source_path"]).stem
        slug = wiki.slugify(source_name)
        rel_path = f"sources/{slug}.md"
        wiki.write_page(rel_path, state["summary"])
        written = list(state.get("pages_written", []))
        written.append(rel_path)
        return {"pages_written": written}
    return _run


def _update_entities_node(wiki: WikiManager, llm: BaseChatModel):
    chain = UPDATE_ENTITY_PAGE | llm | StrOutputParser()

    def _run(state: IngestState) -> dict:
        source_name = Path(state["source_path"]).stem
        slug = wiki.slugify(source_name)
        source_ref = f"sources/{slug}"
        written = list(state.get("pages_written", []))

        for entity in state.get("entities", []):
            name = entity.get("name", "")
            if not name:
                continue
            entity_slug = wiki.slugify(name)
            rel_path = f"entities/{entity_slug}.md"
            existing = wiki.read_page(rel_path) or ""
            updated = chain.invoke({
                "entity_name": name,
                "existing_content": existing,
                "new_info": entity.get("description", ""),
                "source_ref": source_ref,
            })
            wiki.write_page(rel_path, updated)
            written.append(rel_path)

        return {"pages_written": written}
    return _run


def _update_concepts_node(wiki: WikiManager, llm: BaseChatModel):
    chain = UPDATE_CONCEPT_PAGE | llm | StrOutputParser()

    def _run(state: IngestState) -> dict:
        source_name = Path(state["source_path"]).stem
        slug = wiki.slugify(source_name)
        source_ref = f"sources/{slug}"
        written = list(state.get("pages_written", []))

        for concept in state.get("concepts", []):
            name = concept.get("name", "")
            if not name:
                continue
            concept_slug = wiki.slugify(name)
            rel_path = f"concepts/{concept_slug}.md"
            existing = wiki.read_page(rel_path) or ""
            updated = chain.invoke({
                "concept_name": name,
                "existing_content": existing,
                "new_info": concept.get("description", ""),
                "source_ref": source_ref,
            })
            wiki.write_page(rel_path, updated)
            written.append(rel_path)

        return {"pages_written": written}
    return _run


def _update_index_node(wiki: WikiManager, llm: BaseChatModel):
    chain = UPDATE_INDEX | llm | StrOutputParser()

    def _run(state: IngestState) -> dict:
        current_index = wiki.read_index()
        written = state.get("pages_written", [])
        pages_summary = "\n".join(
            f"{p} | {Path(p).stem.replace('-', ' ').title()}"
            for p in written
        )
        updated_index = chain.invoke({
            "current_index": current_index,
            "pages_summary": pages_summary,
        })
        wiki.update_index(updated_index)
        return {"index_content": updated_index}
    return _run


def _write_log_node(wiki: WikiManager):
    def _run(state: IngestState) -> dict:
        source_name = Path(state["source_path"]).name
        n_pages = len(state.get("pages_written", []))
        wiki.append_log("ingest", f"{source_name} ({n_pages} pages updated)")
        return {}
    return _run


def build_ingest_graph(wiki: WikiManager, llm: BaseChatModel):
    graph = StateGraph(IngestState)

    graph.add_node("load_source", _load_source_node(wiki))
    graph.add_node("summarize", _summarize_node(llm))
    graph.add_node("extract", _extract_node(llm))
    graph.add_node("write_summary", _write_summary_node(wiki))
    graph.add_node("update_entities", _update_entities_node(wiki, llm))
    graph.add_node("update_concepts", _update_concepts_node(wiki, llm))
    graph.add_node("update_index", _update_index_node(wiki, llm))
    graph.add_node("write_log", _write_log_node(wiki))

    graph.set_entry_point("load_source")
    graph.add_edge("load_source", "summarize")
    graph.add_edge("summarize", "extract")
    graph.add_edge("extract", "write_summary")
    graph.add_edge("write_summary", "update_entities")
    graph.add_edge("update_entities", "update_concepts")
    graph.add_edge("update_concepts", "update_index")
    graph.add_edge("update_index", "write_log")
    graph.add_edge("write_log", END)

    return graph.compile()
