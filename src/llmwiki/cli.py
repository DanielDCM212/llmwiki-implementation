import argparse
import sys
from pathlib import Path

from llmwiki.config import load_config
from llmwiki.providers import get_provider
from llmwiki.wiki import WikiManager

_DEFAULT_EXTENSIONS = {".md", ".txt", ".rst"}


def _collect_sources(path: str, extensions: set[str]) -> list[Path]:
    """Return a list of source files from a file path or directory (recursive)."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.is_file() and f.suffix in extensions)
    print(f"Error: '{path}' is not a file or directory.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llmwiki",
        description="LLM-maintained personal wiki",
    )
    parser.add_argument(
        "--wiki-root",
        default=None,
        metavar="PATH",
        help="Wiki directory (default: ./wiki or LLMWIKI_ROOT env var)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["anthropic", "google", "openai"],
        help="LLM provider (default: anthropic or LLMWIKI_PROVIDER env var)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help="Model name override (default: provider default or LLMWIKI_MODEL env var)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest a file or directory of source documents")
    ingest_p.add_argument("file", help="Path to a source file or directory (recursive)")
    ingest_p.add_argument(
        "--ext",
        default=",".join(sorted(_DEFAULT_EXTENSIONS)),
        metavar="EXTS",
        help="Comma-separated file extensions to include when ingesting a directory "
             f"(default: {','.join(sorted(_DEFAULT_EXTENSIONS))})",
    )

    query_p = sub.add_parser("query", help="Ask a question against the wiki")
    query_p.add_argument("question", help="The question to answer")
    query_p.add_argument(
        "--save",
        action="store_true",
        help="Save the answer as a new wiki page",
    )

    sub.add_parser("lint", help="Health-check the wiki for issues")

    args = parser.parse_args()

    config = load_config(
        wiki_root=args.wiki_root,
        provider=args.provider,
        model=args.model,
    )

    if not config.api_key:
        print(
            f"Error: No API key found for provider '{config.provider_name}'.\n"
            f"Set LLMWIKI_API_KEY or the provider-specific env var "
            f"(e.g. ANTHROPIC_API_KEY).",
            file=sys.stderr,
        )
        sys.exit(1)

    provider = get_provider(config)
    llm = provider.get_chat_model()
    wiki = WikiManager(config.wiki_root)
    wiki.ensure_structure()

    match args.command:
        case "ingest":
            from llmwiki.graphs.ingest import build_ingest_graph
            extensions = {e.strip() if e.strip().startswith(".") else f".{e.strip()}"
                          for e in args.ext.split(",")}
            sources = _collect_sources(args.file, extensions)
            if not sources:
                print(f"No files found in '{args.file}' matching extensions: {extensions}")
                sys.exit(0)

            graph = build_ingest_graph(wiki, llm)
            total_pages: list[str] = []
            for i, source in enumerate(sources, 1):
                print(f"[{i}/{len(sources)}] Ingesting '{source}' ...")
                result = graph.invoke({
                    "source_path": str(source),
                    "pages_written": [],
                    "messages": [],
                })
                pages = result.get("pages_written", [])
                total_pages.extend(pages)
                for p in pages:
                    print(f"  {p}")
            print(f"\nDone. {len(sources)} file(s) ingested, {len(total_pages)} page(s) written/updated.")

        case "query":
            from llmwiki.graphs.query import build_query_graph
            graph = build_query_graph(wiki, llm)
            result = graph.invoke({
                "question": args.question,
                "save_as_page": args.save,
                "messages": [],
            })
            print(result.get("answer", "(no answer generated)"))

        case "lint":
            from llmwiki.graphs.lint import build_lint_graph
            graph = build_lint_graph(wiki, llm)
            graph.invoke({"messages": []})
