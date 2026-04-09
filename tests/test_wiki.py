import pytest
from pathlib import Path

from llmwiki.wiki import WikiManager


def test_ensure_structure_creates_dirs_and_files(tmp_path):
    wm = WikiManager(tmp_path / "wiki")
    wm.ensure_structure()

    assert wm.sources_dir.is_dir()
    assert wm.entities_dir.is_dir()
    assert wm.concepts_dir.is_dir()
    assert wm.index_path.is_file()
    assert wm.log_path.is_file()


def test_ensure_structure_is_idempotent(tmp_path):
    wm = WikiManager(tmp_path / "wiki")
    wm.ensure_structure()
    wm.ensure_structure()  # should not raise or overwrite

    assert wm.index_path.is_file()


def test_write_and_read_page(wiki: WikiManager):
    wiki.write_page("sources/test.md", "# Hello")

    assert wiki.read_page("sources/test.md") == "# Hello"


def test_read_page_returns_none_for_missing(wiki: WikiManager):
    assert wiki.read_page("sources/nonexistent.md") is None


def test_write_page_creates_parent_dirs(wiki: WikiManager):
    wiki.write_page("entities/deep/nested/page.md", "content")

    assert (wiki.root / "entities/deep/nested/page.md").is_file()


def test_read_index_returns_empty_string_when_missing(tmp_path):
    wm = WikiManager(tmp_path / "wiki")
    # don't call ensure_structure
    assert wm.read_index() == ""


def test_read_index_returns_seeded_content(wiki: WikiManager):
    content = wiki.read_index()

    assert "# Wiki Index" in content


def test_update_index_overwrites(wiki: WikiManager):
    wiki.update_index("# New Index\n\nFresh content")

    assert wiki.read_index() == "# New Index\n\nFresh content"


def test_list_pages_excludes_index_and_log(wiki: WikiManager):
    wiki.write_page("sources/a.md", "A")
    wiki.write_page("entities/b.md", "B")

    pages = wiki.list_pages()

    assert "sources/a.md" in pages
    assert "entities/b.md" in pages
    assert "index.md" not in pages
    assert "log.md" not in pages


def test_list_pages_empty_wiki(wiki: WikiManager):
    assert wiki.list_pages() == []


def test_load_all_pages(wiki: WikiManager):
    wiki.write_page("sources/s.md", "source")
    wiki.write_page("concepts/c.md", "concept")

    pages = wiki.load_all_pages()

    assert pages["sources/s.md"] == "source"
    assert pages["concepts/c.md"] == "concept"


def test_append_log_creates_entry(wiki: WikiManager):
    wiki.append_log("ingest", "article.md (5 pages updated)")

    log = wiki.log_path.read_text(encoding="utf-8")
    assert "ingest" in log
    assert "article.md (5 pages updated)" in log


def test_append_log_multiple_entries(wiki: WikiManager):
    wiki.append_log("ingest", "first.md")
    wiki.append_log("query", "second question")

    log = wiki.log_path.read_text(encoding="utf-8")
    assert "ingest" in log
    assert "query" in log


def test_read_source_file(tmp_path):
    source = tmp_path / "article.md"
    source.write_text("# Article\n\nContent here.", encoding="utf-8")

    wm = WikiManager(tmp_path / "wiki")
    content = wm.read_source_file(source)

    assert content == "# Article\n\nContent here."


@pytest.mark.parametrize("title, expected", [
    ("Hello World", "hello-world"),
    ("  Spaces  ", "spaces"),
    ("Special!@#Chars", "specialchars"),
    ("Multiple   Spaces", "multiple-spaces"),
    ("Already-Slugged", "already-slugged"),
    ("", "untitled"),
    ("MixedCase Title", "mixedcase-title"),
])
def test_slugify(title, expected):
    assert WikiManager.slugify(title) == expected
