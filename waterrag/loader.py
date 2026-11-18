from pathlib import Path

from pypdf import PdfReader


def load_documents(folder: Path) -> list[dict]:
    """Return [{source, page, text}] for every .pdf/.txt/.md under folder."""
    docs = []
    for path in sorted(Path(folder).rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append({"source": path.name, "page": i, "text": text})
        elif suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append({"source": path.name, "page": 1, "text": text})
    return docs
