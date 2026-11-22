"""Step 3 of the pipeline: embed chunks and store them in a FAISS index (cached on disk)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from langchain_community.vectorstores import FAISS

from .config import settings
from .ingest import load_pdf, split_documents


@dataclass
class IndexInfo:
    store: FAISS
    file_name: str
    n_pages: int
    n_chunks: int
    from_cache: bool


def get_embeddings():
    """Embedding model: FastEmbed runs locally for free; Gemini uses the Google API."""
    if settings.embed_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=settings.embed_model)
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    return FastEmbedEmbeddings(model_name=settings.embed_model)


def index_key(pdf_path: str | Path) -> str:
    """Cache key = PDF content + embedding model + chunk settings, so any change rebuilds the index."""
    h = hashlib.md5(Path(pdf_path).read_bytes())
    h.update(f"{settings.embed_provider}|{settings.embed_model}|{settings.chunk_size}|{settings.chunk_overlap}".encode())
    return h.hexdigest()[:16]


def build_or_load(pdf_path: str | Path, embeddings=None, index_dir: Path | None = None,
                  use_cache: bool = True) -> IndexInfo:
    pdf_path = Path(pdf_path)
    embeddings = embeddings or get_embeddings()
    cache = (index_dir or settings.index_dir) / index_key(pdf_path)
    meta_file = cache / "meta.json"

    if use_cache and (cache / "index.faiss").exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text())
        store = FAISS.load_local(str(cache), embeddings, allow_dangerous_deserialization=True)
        return IndexInfo(store, meta["file_name"], meta["n_pages"], meta["n_chunks"], True)

    pages = load_pdf(pdf_path)
    if not pages:
        raise ValueError("No text found in this PDF. If it is a scanned document, run OCR on it first.")
    chunks = split_documents(pages)
    store = FAISS.from_documents(chunks, embeddings)
    info = IndexInfo(store, pdf_path.name, len(pages), len(chunks), False)

    if use_cache:
        cache.mkdir(parents=True, exist_ok=True)
        store.save_local(str(cache))
        meta_file.write_text(json.dumps(
            {"file_name": info.file_name, "n_pages": info.n_pages, "n_chunks": info.n_chunks}))
    return info
