"""
Graph tests using FakeListChatModel — no real LLM calls.

Each graph node that calls the LLM consumes one response from the fake model.
Responses are provided in the order the nodes fire.
"""
import json
import pytest
from pathlib import Path

from llmwiki.graphs.ingest import build_ingest_graph
from llmwiki.graphs.query import build_query_graph
from llmwiki.graphs.lint import build_lint_graph
from llmwiki.wiki import WikiManager


# ---------------------------------------------------------------------------
# Ingest graph
# ---------------------------------------------------------------------------

class TestIngestGraph:
    def test_ingest_writes_summary_page(self, wiki: WikiManager, fake_llm, tmp_path):
        source = tmp_path / "article.md"
        source.write_text("# AI Article\n\nContent about AI.", encoding="utf-8")

        llm = fake_llm(
            # summarize node
            "# AI Article\n\n## Summary\nAI article summary.\n\n## Key Points\n- Point 1\n\n## Tags\nai",
            # extract node
            json.dumps({"entities": [{"name": "Alan Turing", "description": "Computer scientist", "category": "Person"}],
                        "concepts": [{"name": "Artificial Intelligence", "description": "Simulating human intelligence"}]}),
            # update_entities node
            "# Alan Turing\n\nComputer scientist. [[sources/article]]",
            # update_concepts node
            "# Artificial Intelligence\n\nSimulating human intelligence. [[sources/article]]",
            # update_index node
            "# Wiki Index\n\n## Sources\n- [[sources/article]] — AI Article\n",
        )

        result = build_ingest_graph(wiki, llm).invoke({
            "source_path": str(source),
            "pages_written": [],
            "messages": [],
        })

        assert "sources/article.md" in result["pages_written"]
        assert wiki.read_page("sources/article.md") is not None

    def test_ingest_writes_entity_and_concept_pages(self, wiki: WikiManager, fake_llm, tmp_path):
        source = tmp_path / "paper.md"
        source.write_text("# Paper\n\nContent.", encoding="utf-8")

        llm = fake_llm(
            "# Paper\n\n## Summary\nPaper summary.",
            json.dumps({"entities": [{"name": "Bob", "description": "Researcher", "category": "Person"}],
                        "concepts": [{"name": "Gradient Descent", "description": "Optimization algo"}]}),
            "# Bob\n\nResearcher.",
            "# Gradient Descent\n\nOptimization algo.",
            "# Wiki Index\n\n## Sources\n- [[sources/paper]]",
        )

        result = build_ingest_graph(wiki, llm).invoke({
            "source_path": str(source),
            "pages_written": [],
            "messages": [],
        })

        assert "entities/bob.md" in result["pages_written"]
        assert "concepts/gradient-descent.md" in result["pages_written"]

    def test_ingest_updates_log(self, wiki: WikiManager, fake_llm, tmp_path):
        source = tmp_path / "doc.md"
        source.write_text("# Doc", encoding="utf-8")

        llm = fake_llm(
            "# Doc\n\n## Summary\nDoc summary.",
            json.dumps({"entities": [], "concepts": []}),
            "# Wiki Index",
        )

        build_ingest_graph(wiki, llm).invoke({
            "source_path": str(source),
            "pages_written": [],
            "messages": [],
        })

        log = wiki.log_path.read_text(encoding="utf-8")
        assert "ingest" in log
        assert "doc.md" in log

    def test_ingest_handles_empty_entities_and_concepts(self, wiki: WikiManager, fake_llm, tmp_path):
        source = tmp_path / "empty.md"
        source.write_text("# Empty\n\nNo entities.", encoding="utf-8")

        llm = fake_llm(
            "# Empty\n\n## Summary\nNo entities.",
            json.dumps({"entities": [], "concepts": []}),
            "# Wiki Index",
        )

        result = build_ingest_graph(wiki, llm).invoke({
            "source_path": str(source),
            "pages_written": [],
            "messages": [],
        })

        # Only the summary page — no entity/concept pages
        assert result["pages_written"] == ["sources/empty.md"]

    def test_ingest_tolerates_malformed_json_extract(self, wiki: WikiManager, fake_llm, tmp_path):
        source = tmp_path / "bad.md"
        source.write_text("# Bad JSON source", encoding="utf-8")

        llm = fake_llm(
            "# Bad\n\n## Summary\nBad.",
            "not valid json at all",   # extract node returns garbage
            "# Wiki Index",
        )

        # Should not raise — falls back to empty entities/concepts
        result = build_ingest_graph(wiki, llm).invoke({
            "source_path": str(source),
            "pages_written": [],
            "messages": [],
        })

        assert result["pages_written"] == ["sources/bad.md"]


# ---------------------------------------------------------------------------
# Query graph
# ---------------------------------------------------------------------------

class TestQueryGraph:
    def _populate_wiki(self, wiki: WikiManager):
        wiki.write_page("sources/ml.md", "# ML\n\nMachine learning basics.")
        wiki.update_index(
            "# Wiki Index\n\n## Sources\n- [[sources/ml]] — ML basics\n"
        )

    def test_query_returns_answer(self, wiki: WikiManager, fake_llm):
        self._populate_wiki(wiki)

        llm = fake_llm(
            json.dumps(["sources/ml.md"]),          # find_relevant_pages
            "Machine learning is a subset of AI.",   # synthesize_answer
        )

        result = build_query_graph(wiki, llm).invoke({
            "question": "What is ML?",
            "save_as_page": False,
            "messages": [],
        })

        assert result["answer"] == "Machine learning is a subset of AI."

    def test_query_loads_relevant_pages(self, wiki: WikiManager, fake_llm):
        self._populate_wiki(wiki)

        llm = fake_llm(
            json.dumps(["sources/ml.md"]),
            "ML answer.",
        )

        result = build_query_graph(wiki, llm).invoke({
            "question": "What is ML?",
            "save_as_page": False,
            "messages": [],
        })

        assert "sources/ml.md" in result["page_contents"]

    def test_query_with_save_writes_page(self, wiki: WikiManager, fake_llm):
        self._populate_wiki(wiki)

        llm = fake_llm(
            json.dumps(["sources/ml.md"]),
            "ML is a field of AI.",
            "# What Is ML\n\nML is a field of AI. [[sources/ml]]",  # generate_answer_page
            "# Wiki Index\n\n## Sources\n...\n## Concepts\n- [[concepts/what-is-ml]]",  # update_index
        )

        build_query_graph(wiki, llm).invoke({
            "question": "What is ML?",
            "save_as_page": True,
            "messages": [],
        })

        pages = wiki.list_pages()
        assert any("what-is-ml" in p or "what-is" in p for p in pages)

    def test_query_handles_no_relevant_pages(self, wiki: WikiManager, fake_llm):
        self._populate_wiki(wiki)

        llm = fake_llm(
            json.dumps([]),              # no relevant pages found
            "I don't have enough information to answer.",
        )

        result = build_query_graph(wiki, llm).invoke({
            "question": "What is quantum computing?",
            "save_as_page": False,
            "messages": [],
        })

        assert "answer" in result
        assert result["page_contents"] == {}

    def test_query_tolerates_malformed_page_list(self, wiki: WikiManager, fake_llm):
        self._populate_wiki(wiki)

        llm = fake_llm(
            "not valid json",            # malformed page list
            "Some answer.",
        )

        result = build_query_graph(wiki, llm).invoke({
            "question": "Anything?",
            "save_as_page": False,
            "messages": [],
        })

        assert result["answer"] == "Some answer."


# ---------------------------------------------------------------------------
# Lint graph
# ---------------------------------------------------------------------------

class TestLintGraph:
    def test_lint_reports_no_issues_on_clean_wiki(self, wiki: WikiManager, fake_llm, capsys):
        wiki.write_page("sources/clean.md", "# Clean source\n\nAll good.")

        llm = fake_llm(json.dumps([]))  # no issues

        result = build_lint_graph(wiki, llm).invoke({"messages": []})

        assert result["issues"] == []
        captured = capsys.readouterr()
        assert "no issues" in captured.out

    def test_lint_reports_found_issues(self, wiki: WikiManager, fake_llm, capsys):
        wiki.write_page("sources/a.md", "# A\n\nContent.")
        wiki.write_page("sources/b.md", "# B\n\nContradicts A.")

        issues = [
            {
                "type": "contradiction",
                "page": "sources/b.md",
                "description": "Contradicts claim in sources/a.md",
                "suggestion": "Reconcile the two pages",
            }
        ]
        llm = fake_llm(json.dumps(issues))

        result = build_lint_graph(wiki, llm).invoke({"messages": []})

        assert len(result["issues"]) == 1
        assert result["issues"][0]["type"] == "contradiction"
        captured = capsys.readouterr()
        assert "contradiction" in captured.out

    def test_lint_skips_llm_call_on_empty_wiki(self, wiki: WikiManager, fake_llm):
        # Empty wiki — no pages to lint, LLM should not be called
        llm = fake_llm()  # no responses — would raise if consumed

        result = build_lint_graph(wiki, llm).invoke({"messages": []})

        assert result["issues"] == []

    def test_lint_appends_to_log(self, wiki: WikiManager, fake_llm):
        wiki.write_page("sources/x.md", "# X")
        llm = fake_llm(json.dumps([]))

        build_lint_graph(wiki, llm).invoke({"messages": []})

        log = wiki.log_path.read_text(encoding="utf-8")
        assert "lint" in log

    def test_lint_tolerates_malformed_issues_json(self, wiki: WikiManager, fake_llm, capsys):
        wiki.write_page("sources/x.md", "# X")
        llm = fake_llm("not valid json")

        result = build_lint_graph(wiki, llm).invoke({"messages": []})

        assert result["issues"] == []
