"""Step 1-2 of the pipeline: load the handbook PDF and split it into chunks."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings


def _line_key(line: str) -> str:
    """Normalise a line so 'Page 3' and 'Page 4' count as the same repeated footer."""
    return re.sub(r"\d+", "#", line.strip().lower())


def remove_boilerplate(pages: list[Document], min_share: float = 0.6) -> list[Document]:
    """Drop header/footer lines that repeat on most pages (college name, page numbers, etc.).

    Repeated lines add noise to every chunk and make retrieval less precise.
    """
    if len(pages) < 3:
        return pages
    counts = Counter()
    for p in pages:
        counts.update({_line_key(l) for l in p.page_content.splitlines() if l.strip()})
    repeated = {k for k, c in counts.items() if c / len(pages) >= min_share}
    for p in pages:
        kept = [l for l in p.page_content.splitlines() if l.strip() and _line_key(l) not in repeated]
        p.page_content = "\n".join(kept)
    return pages


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Load a PDF as one Document per page, with a 1-based page number and file name."""
    pdf_path = Path(pdf_path)
    pages = PyPDFLoader(str(pdf_path)).load()
    for p in pages:
        p.metadata["page_number"] = int(p.metadata.get("page", 0)) + 1
        p.metadata["source_file"] = pdf_path.name
    pages = remove_boilerplate(pages)
    return [p for p in pages if p.page_content.strip()]


def split_documents(pages: list[Document], chunk_size: int | None = None,
                    chunk_overlap: int | None = None) -> list[Document]:
    """Split pages into overlapping chunks. Metadata (page number) is copied to every chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks
