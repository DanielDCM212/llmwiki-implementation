from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


@dataclass
class IndexEntry:
    category: str
    title: str
    rel_path: str
    summary: str


class WikiManager:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.sources_dir = self.root / "sources"
        self.entities_dir = self.root / "entities"
        self.concepts_dir = self.root / "concepts"
        self.index_path = self.root / "index.md"
        self.log_path = self.root / "log.md"

    def ensure_structure(self) -> None:
        """Create wiki dirs and seed files if they don't exist."""
        for d in (self.sources_dir, self.entities_dir, self.concepts_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text(
                "# Wiki Index\n\n## Sources\n\n## Entities\n\n## Concepts\n",
                encoding="utf-8",
            )
        if not self.log_path.exists():
            self.log_path.write_text("# Wiki Log\n\n", encoding="utf-8")

    # --- Read operations ---

    def read_page(self, rel_path: str) -> str | None:
        path = self.root / rel_path
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def read_index(self) -> str:
        if not self.index_path.exists():
            return ""
        return self.index_path.read_text(encoding="utf-8")

    def list_pages(self) -> list[str]:
        """Return relative paths (as strings) of all .md pages in the wiki."""
        return [
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*.md")
            if p.name not in ("index.md", "log.md")
        ]

    def load_all_pages(self) -> dict[str, str]:
        """Return {rel_path: content} for every wiki page."""
        return {
            rel: content
            for rel in self.list_pages()
            if (content := self.read_page(rel)) is not None
        }

    # --- Write operations ---

    def write_page(self, rel_path: str, content: str) -> Path:
        """Write or overwrite a wiki page. Creates parent dirs as needed."""
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def update_index(self, content: str) -> None:
        """Overwrite index.md with the given content."""
        self.index_path.write_text(content, encoding="utf-8")

    def append_log(self, operation: str, summary: str) -> None:
        """Append a timestamped entry to log.md."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        entry = f"\n## [{date_str}] {operation} | {summary}\n"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(entry)

    # --- Source reading ---

    def read_source_file(self, path: Path) -> str:
        """Read a raw source file. Assumes plain text / markdown for v1."""
        return Path(path).read_text(encoding="utf-8")

    # --- Helpers ---

    @staticmethod
    def slugify(title: str) -> str:
        """Convert a title to a filename-safe slug."""
        slug = title.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_-]+", "-", slug)
        slug = slug.strip("-")
        return slug or "untitled"
