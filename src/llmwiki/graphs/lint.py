import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from llmwiki.prompts import LINT_WIKI
from llmwiki.state import LintState
from llmwiki.wiki import WikiManager


def _load_all_pages_node(wiki: WikiManager):
    def _run(state: LintState) -> dict:
        return {"all_pages": wiki.load_all_pages()}
    return _run


def _detect_issues_node(llm: BaseChatModel):
    chain = LINT_WIKI | llm | StrOutputParser()

    def _run(state: LintState) -> dict:
        pages = state.get("all_pages", {})
        if not pages:
            return {"issues": []}

        all_pages_text = "\n\n---\n\n".join(
            f"**{path}**\n\n{content}"
            for path, content in pages.items()
        )
        raw = chain.invoke({"all_pages_text": all_pages_text})
        try:
            issues = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            issues = json.loads(match.group()) if match else []
        return {"issues": issues if isinstance(issues, list) else []}
    return _run


def _report_node(wiki: WikiManager):
    def _run(state: LintState) -> dict:
        issues = state.get("issues", [])
        if not issues:
            print("Wiki lint: no issues found.")
        else:
            print(f"Wiki lint: {len(issues)} issue(s) found.\n")
            for issue in issues:
                itype = issue.get("type", "?")
                page = issue.get("page", "?")
                desc = issue.get("description", "")
                suggestion = issue.get("suggestion", "")
                print(f"[{itype}] {page}")
                print(f"  Issue: {desc}")
                if suggestion:
                    print(f"  Suggestion: {suggestion}")
                print()
        wiki.append_log("lint", f"{len(issues)} issue(s) found")
        return {}
    return _run


def build_lint_graph(wiki: WikiManager, llm: BaseChatModel):
    graph = StateGraph(LintState)

    graph.add_node("load_all_pages", _load_all_pages_node(wiki))
    graph.add_node("detect_issues", _detect_issues_node(llm))
    graph.add_node("report", _report_node(wiki))

    graph.set_entry_point("load_all_pages")
    graph.add_edge("load_all_pages", "detect_issues")
    graph.add_edge("detect_issues", "report")
    graph.add_edge("report", END)

    return graph.compile()
