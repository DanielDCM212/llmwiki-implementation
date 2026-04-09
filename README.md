# LLMWiki

A Python CLI that builds and maintains a persistent, LLM-generated wiki from your source documents. Drop in articles, papers, or notes — the LLM summarizes them, extracts entities and concepts, cross-references everything, and keeps the wiki current as new sources arrive.

Based on the [LLMWiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## How it works

Three operations, each backed by a LangGraph workflow:

- **Ingest** — reads a source, writes a summary page, creates/updates entity and concept pages, updates the index, and logs the event
- **Query** — reads the index to find relevant pages, loads them, and synthesizes a cited answer; optionally saves the answer as a new wiki page
- **Lint** — scans all wiki pages for contradictions, stale claims, orphan pages, and missing cross-references

The wiki is a directory of plain markdown files you can open in Obsidian, VS Code, or any editor.

## Project structure

```
src/llmwiki/
├── cli.py            # CLI entry point (ingest / query / lint)
├── config.py         # Configuration — env vars and .env file
├── wiki.py           # WikiManager — all file I/O
├── state.py          # LangGraph state schemas
├── prompts.py        # LangChain prompt templates
├── providers/
│   ├── base.py       # LLMProvider protocol
│   ├── anthropic.py  # Anthropic (default)
│   ├── google.py     # Google GenAI
│   └── openai.py     # OpenAI
└── graphs/
    ├── ingest.py     # Ingest workflow graph
    ├── query.py      # Query workflow graph
    └── lint.py       # Lint workflow graph
```

## Requirements

- Python 3.12+
- An API key for your chosen provider

## Installation

```bash
# Clone and set up a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the package
pip install -e .
```

## Configuration

Create a `.env` file in the project root (loaded automatically):

```env
ANTHROPIC_API_KEY=sk-ant-...
LLMWIKI_ROOT=./wiki
```

Or use environment variables directly. Full list:

| Variable | Default | Description |
|---|---|---|
| `LLMWIKI_ROOT` | `./wiki` | Path to the wiki directory |
| `LLMWIKI_PROVIDER` | `anthropic` | LLM provider: `anthropic`, `google`, `openai` |
| `LLMWIKI_API_KEY` | — | Generic API key (overrides provider-specific keys) |
| `LLMWIKI_MODEL` | provider default | Model name override |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GOOGLE_API_KEY` | — | Google GenAI API key |
| `OPENAI_API_KEY` | — | OpenAI API key |

Config resolution order: CLI flag → env var / `.env` → default.

## Usage

### Ingest

Add a source document or a whole folder to the wiki:

```bash
# Single file
python -m llmwiki ingest raw/article.md

# Entire directory (recursive)
python -m llmwiki ingest raw/

# Subdirectory
python -m llmwiki ingest raw/papers/2024/

# Custom file extensions (default: .md, .rst, .txt)
python -m llmwiki ingest raw/ --ext .md,.txt
```

Output:
```
[1/3] Ingesting 'raw/papers/transformer.md' ...
  sources/transformer.md
  entities/attention-mechanism.md
  concepts/self-attention.md
[2/3] Ingesting 'raw/papers/bert.md' ...
  sources/bert.md
  entities/bert.md

Done. 3 file(s) ingested, 7 page(s) written/updated.
```

Each ingested source touches multiple wiki pages:
1. Writes a summary page under `wiki/sources/`
2. Creates or updates entity pages under `wiki/entities/`
3. Creates or updates concept pages under `wiki/concepts/`
4. Updates `wiki/index.md`
5. Appends an entry to `wiki/log.md`

### Query

Ask a question against the wiki:

```bash
python -m llmwiki query "What is the attention mechanism?"

# Save the answer as a new wiki page
python -m llmwiki query "Compare BERT and GPT architectures" --save
```

The query workflow reads `index.md` to find relevant pages, loads them, and synthesizes a cited answer. With `--save`, the answer is filed as a new concept page.

### Lint

Health-check the wiki for issues:

```bash
python -m llmwiki lint
```

Reports:
- Contradictions between pages
- Stale or superseded claims
- Orphan pages (no inbound links)
- Concepts mentioned but lacking their own page
- Missing cross-references

### Global flags

```bash
python -m llmwiki --wiki-root /path/to/obsidian-vault ingest raw/article.md
python -m llmwiki --provider openai --model gpt-4o query "What did I learn about X?"
python -m llmwiki --provider google query "Summarize my notes on Y"
```

## Wiki structure

```
wiki/
├── index.md          # Catalog of all pages (updated on every ingest)
├── log.md            # Append-only timeline of ingests, queries, lints
├── sources/          # One summary page per ingested source
├── entities/         # People, organizations, products, events
└── concepts/         # Ideas, techniques, frameworks
```

`index.md` is the LLM's navigation map — it reads this first on every query to find relevant pages without loading the entire wiki. `log.md` entries follow the format `## [YYYY-MM-DD] operation | description` and are grep-friendly.

## Providers

| Provider | Default model | Package |
|---|---|---|
| `anthropic` | `claude-sonnet-4-6` | `langchain-anthropic` |
| `google` | `gemini-2.0-flash` | `langchain-google-genai` |
| `openai` | `gpt-4o` | `langchain-openai` |

All providers return a LangChain `BaseChatModel`, so the graphs are fully provider-agnostic. Switch providers with `--provider` or `LLMWIKI_PROVIDER` — no code changes needed.

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests use `FakeListChatModel` — no API key required.
