"""
Document Loader — loads PDF, TXT, Markdown, and HTML documents.
"""

from pathlib import Path


class DocumentLoader:
    """Loads and extracts text from various document formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}

    def __init__(self):
        self.documents: list[dict] = []

    def load(self, path: str | Path) -> list[dict]:
        """Load a single file or all supported files in a directory."""
        path = Path(path)
        if path.is_file():
            return [self._load_file(path)]
        elif path.is_dir():
            docs = []
            for ext in self.SUPPORTED_EXTENSIONS:
                for f in path.rglob(f"*{ext}"):
                    docs.append(self._load_file(f))
            return docs
        else:
            raise FileNotFoundError(f"Path not found: {path}")

    def _load_file(self, path: Path) -> dict:
        ext = path.suffix.lower()

        if ext == ".pdf":
            text = self._load_pdf(path)
        elif ext in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="replace")
        elif ext in (".html", ".htm"):
            text = self._load_html(path)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")

        doc = {
            "source": str(path),
            "filename": path.name,
            "content": text,
            "size": len(text),
        }
        self.documents.append(doc)
        return doc

    def _load_pdf(self, path: Path) -> str:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text
        except ImportError:
            return f"[PDF loading requires PyPDF2: pip install PyPDF2]\nFile: {path}"

    def _load_html(self, path: Path) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            return path.read_text(encoding="utf-8", errors="replace")[:5000]
