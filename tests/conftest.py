"""Test fixtures. The tests run fully offline: no API key or internet needed.

- KeywordEmbeddings: a tiny bag-of-words embedding, enough to test retrieval logic.
- FakeLLM: returns predictable answers so the QA logic can be tested without Gemini.
"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rulebook_qa.prompts import NOT_FOUND  # noqa: E402

STOP = set("the a an of to in is and for or what my i can be if do how are with on at by it this that me".split())


class KeywordEmbeddings(Embeddings):
    dim = 1024

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if w not in STOP:
                v[int(hashlib.md5(w[:6].encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeReply:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Answers with the first context page; says NOT_FOUND for out-of-scope topics."""
    OUT_OF_SCOPE = ("placement", "bus", "canteen menu")

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if "Standalone question:" in prompt:  # condense step
            latest = prompt.split("Latest message:")[1].split("Standalone question:")[0].strip()
            return FakeReply(f"{latest} (rewritten)")
        question = prompt.split("Student question:")[1].lower()
        if any(k in question for k in self.OUT_OF_SCOPE):
            return FakeReply(NOT_FOUND)
        page = re.search(r"\[Page (\d+)\]", prompt).group(1)
        # Gemini sometimes returns a list of parts: test that format too
        return FakeReply([{"type": "text", "text": f"Here is the rule. (Source: Page {page})"}])


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    return ROOT / "data" / "sample_college_handbook.pdf"


@pytest.fixture(scope="session")
def embeddings():
    return KeywordEmbeddings()


@pytest.fixture(scope="session")
def index(sample_pdf, embeddings, tmp_path_factory):
    from rulebook_qa.vectorstore import build_or_load
    return build_or_load(sample_pdf, embeddings=embeddings, index_dir=tmp_path_factory.mktemp("idx"))


@pytest.fixture
def fake_llm():
    return FakeLLM()
