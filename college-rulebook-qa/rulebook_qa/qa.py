"""Steps 4-5 of the pipeline: retrieve relevant chunks and generate a grounded answer."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .config import settings
from .prompts import ANSWER_PROMPT, CONDENSE_PROMPT, NOT_FOUND


@dataclass
class Source:
    page: int
    text: str
    score: float
    source_file: str = ""
    cited: bool = False


@dataclass
class Answer:
    question: str
    standalone_question: str
    answer: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    cited_pages: list[int] = field(default_factory=list)
    retrieved_pages: list[int] = field(default_factory=list)
    latency_s: float = 0.0


def get_llm():
    """Chat model: any OpenAI-compatible API (default) or Google Gemini."""
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=settings.chat_model, temperature=settings.temperature)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=settings.chat_model, temperature=settings.temperature,
                      base_url=settings.openai_base_url or None, timeout=90, max_retries=2)


def message_text(msg) -> str:
    """Return plain text from an LLM reply (some Gemini versions return a list of parts)."""
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content).strip()


def is_not_found(text: str) -> bool:
    norm = lambda s: re.sub(r"[^a-z ]", "", s.lower())
    return norm(NOT_FOUND) in norm(text)


def parse_cited_pages(text: str) -> list[int]:
    """Extract page numbers from '(Source: Page 2)' or '(Source: Pages 2, 3)'."""
    m = re.search(r"source[s]?\s*:\s*pages?\s*([\d,\s and&-]+)", text, flags=re.I)
    return sorted({int(n) for n in re.findall(r"\d+", m.group(1))}) if m else []


def format_context(hits: list[tuple[Document, float]]) -> str:
    return "\n\n".join(f"[Page {d.metadata['page_number']}]\n{d.page_content}" for d, _ in hits)


class RulebookQA:
    """Question-answering engine over one handbook's FAISS index."""

    def __init__(self, store, llm=None, top_k: int | None = None):
        self.store = store
        self.llm = llm or get_llm()
        self.top_k = top_k or settings.top_k

    def retrieve(self, question: str, k: int | None = None) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_score(question, k=k or self.top_k)

    def condense(self, question: str, history: list[dict] | None) -> str:
        """Turn a follow-up like 'what about practicals?' into a standalone question."""
        if not history:
            return question
        turns = history[-2 * settings.history_turns:]
        convo = "\n".join(f"{m['role'].title()}: {m['content']}" for m in turns)
        rewritten = message_text(self.llm.invoke(CONDENSE_PROMPT.format(history=convo, question=question)))
        return rewritten or question

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        start = time.perf_counter()
        standalone = self.condense(question, history)
        hits = self.retrieve(standalone)
        prompt = ANSWER_PROMPT.format(not_found=NOT_FOUND, context=format_context(hits), question=standalone)
        text = message_text(self.llm.invoke(prompt))

        found = not is_not_found(text)
        cited = parse_cited_pages(text) if found else []
        sources = []
        if found:
            sources = [Source(page=d.metadata["page_number"], text=d.page_content, score=float(s),
                              source_file=d.metadata.get("source_file", ""),
                              cited=d.metadata["page_number"] in cited) for d, s in hits]
            sources.sort(key=lambda s: (not s.cited, s.score))  # cited pages first
        return Answer(question=question, standalone_question=standalone, answer=text, found=found,
                      sources=sources, cited_pages=cited,
                      retrieved_pages=sorted({d.metadata["page_number"] for d, _ in hits}),
                      latency_s=time.perf_counter() - start)
