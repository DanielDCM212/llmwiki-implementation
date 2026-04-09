"""
CLI tests using argparse argument parsing directly (no subprocess).
We patch get_provider and build_*_graph to avoid real LLM calls.
"""
import json
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llmwiki.cli import main


def _run_cli(*args: str, monkeypatch=None):
    """Run the CLI with the given args list."""
    with patch("sys.argv", ["llmwiki", *args]):
        main()


# ---------------------------------------------------------------------------
# Argument parsing / error handling
# ---------------------------------------------------------------------------

def test_missing_api_key_exits(monkeypatch, tmp_path):
    monkeypatch.delenv("LLMWIKI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("sys.argv", ["llmwiki", "--wiki-root", str(tmp_path), "query", "hello"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_unknown_command_exits(monkeypatch):
    with patch("sys.argv", ["llmwiki", "bogus"]):
        with pytest.raises(SystemExit):
            main()


# ---------------------------------------------------------------------------
# Ingest command — single file
# ---------------------------------------------------------------------------

def test_ingest_single_file_invokes_graph_once(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    source = tmp_path / "article.md"
    source.write_text("# Article", encoding="utf-8")

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"pages_written": ["sources/article.md"]}

    with patch("sys.argv", ["llmwiki", "--wiki-root", str(tmp_path / "wiki"), "ingest", str(source)]):
        with patch("llmwiki.providers.anthropic.AnthropicProvider.get_chat_model", return_value=MagicMock()):
            with patch("llmwiki.graphs.ingest.build_ingest_graph", return_value=fake_graph):
                main()

    fake_graph.invoke.assert_called_once()
    captured = capsys.readouterr()
    assert "sources/article.md" in captured.out


# ---------------------------------------------------------------------------
# Ingest command — directory
# ---------------------------------------------------------------------------

def test_ingest_directory_invokes_graph_per_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.md").write_text("# A", encoding="utf-8")
    (raw / "b.md").write_text("# B", encoding="utf-8")

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"pages_written": ["sources/a.md"]}

    with patch("sys.argv", ["llmwiki", "--wiki-root", str(tmp_path / "wiki"), "ingest", str(raw)]):
        with patch("llmwiki.providers.anthropic.AnthropicProvider.get_chat_model", return_value=MagicMock()):
            with patch("llmwiki.graphs.ingest.build_ingest_graph", return_value=fake_graph):
                main()

    assert fake_graph.invoke.call_count == 2
    captured = capsys.readouterr()
    assert "2 file(s) ingested" in captured.out


def test_ingest_directory_recurses_into_subfolders(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    raw = tmp_path / "raw"
    (raw / "papers" / "ai").mkdir(parents=True)
    (raw / "papers" / "ai" / "deep.md").write_text("# Deep", encoding="utf-8")
    (raw / "notes.md").write_text("# Notes", encoding="utf-8")

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"pages_written": []}

    with patch("sys.argv", ["llmwiki", "--wiki-root", str(tmp_path / "wiki"), "ingest", str(raw)]):
        with patch("llmwiki.providers.anthropic.AnthropicProvider.get_chat_model", return_value=MagicMock()):
            with patch("llmwiki.graphs.ingest.build_ingest_graph", return_value=fake_graph):
                main()

    assert fake_graph.invoke.call_count == 2


def test_ingest_directory_filters_by_extension(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "doc.md").write_text("# Doc", encoding="utf-8")
    (raw / "image.png").write_bytes(b"fake png")
    (raw / "data.csv").write_text("a,b,c", encoding="utf-8")

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"pages_written": []}

    with patch("sys.argv", ["llmwiki", "--wiki-root", str(tmp_path / "wiki"), "ingest", str(raw)]):
        with patch("llmwiki.providers.anthropic.AnthropicProvider.get_chat_model", return_value=MagicMock()):
            with patch("llmwiki.graphs.ingest.build_ingest_graph", return_value=fake_graph):
                main()

    # Only .md should be ingested (default extensions)
    assert fake_graph.invoke.call_count == 1


def test_ingest_directory_custom_ext_flag(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "doc.md").write_text("# Doc", encoding="utf-8")
    (raw / "note.txt").write_text("plain text", encoding="utf-8")

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"pages_written": []}

    with patch("sys.argv", ["llmwiki", "--wiki-root", str(tmp_path / "wiki"),
                             "ingest", str(raw), "--ext", ".txt"]):
        with patch("llmwiki.providers.anthropic.AnthropicProvider.get_chat_model", return_value=MagicMock()):
            with patch("llmwiki.graphs.ingest.build_ingest_graph", return_value=fake_graph):
                main()

    # Only .txt matched
    assert fake_graph.invoke.call_count == 1
    call_path = fake_graph.invoke.call_args[0][0]["source_path"]
    assert call_path.endswith("note.txt")


def test_ingest_empty_directory_exits_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "image.png").write_bytes(b"fake")  # no matching files

    fake_graph = MagicMock()

    with patch("sys.argv", ["llmwiki", "--wiki-root", str(tmp_path / "wiki"), "ingest", str(raw)]):
        with patch("llmwiki.providers.anthropic.AnthropicProvider.get_chat_model", return_value=MagicMock()):
            with patch("llmwiki.graphs.ingest.build_ingest_graph", return_value=fake_graph):
                with pytest.raises(SystemExit) as exc:
                    main()

    assert exc.value.code == 0
    fake_graph.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# Query command
# ---------------------------------------------------------------------------

def test_query_command_prints_answer(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"answer": "The answer is 42."}

    with patch("sys.argv", ["llmwiki", "--wiki-root", str(tmp_path / "wiki"), "query", "What is the answer?"]):
        with patch("llmwiki.providers.anthropic.AnthropicProvider.get_chat_model", return_value=MagicMock()):
            with patch("llmwiki.graphs.query.build_query_graph", return_value=fake_graph):
                main()

    captured = capsys.readouterr()
    assert "The answer is 42." in captured.out


def test_query_passes_save_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"answer": "Saved answer."}

    with patch("sys.argv", ["llmwiki", "--wiki-root", str(tmp_path / "wiki"), "query", "Q?", "--save"]):
        with patch("llmwiki.providers.anthropic.AnthropicProvider.get_chat_model", return_value=MagicMock()):
            with patch("llmwiki.graphs.query.build_query_graph", return_value=fake_graph):
                main()

    call_kwargs = fake_graph.invoke.call_args[0][0]
    assert call_kwargs["save_as_page"] is True


# ---------------------------------------------------------------------------
# Lint command
# ---------------------------------------------------------------------------

def test_lint_command_invokes_graph(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {}

    with patch("sys.argv", ["llmwiki", "--wiki-root", str(tmp_path / "wiki"), "lint"]):
        with patch("llmwiki.providers.anthropic.AnthropicProvider.get_chat_model", return_value=MagicMock()):
            with patch("llmwiki.graphs.lint.build_lint_graph", return_value=fake_graph):
                main()

    fake_graph.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# Provider / model flag
# ---------------------------------------------------------------------------

def test_provider_flag_is_passed_to_config(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "oai-key")

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {"answer": "ok"}

    with patch("sys.argv", ["llmwiki", "--wiki-root", str(tmp_path / "wiki"),
                             "--provider", "openai", "query", "Q?"]):
        with patch("llmwiki.providers.openai.OpenAIProvider.get_chat_model", return_value=MagicMock()):
            with patch("llmwiki.graphs.query.build_query_graph", return_value=fake_graph):
                main()

    fake_graph.invoke.assert_called_once()
