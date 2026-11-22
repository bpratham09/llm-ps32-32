import json

import pytest

from rulebook_qa.config import settings
from rulebook_qa.vectorstore import build_or_load

QUESTIONS = [q for q in json.loads(settings.eval_questions.read_text()) if q["expected_page"]]


def test_index_counts(index):
    assert index.n_pages == 6
    assert index.n_chunks >= 6


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q["question"][:40])
def test_expected_page_is_retrieved(index, q):
    hits = index.store.similarity_search_with_score(q["question"], k=settings.top_k)
    pages = {d.metadata["page_number"] for d, _ in hits}
    assert q["expected_page"] in pages


def test_index_is_cached(sample_pdf, embeddings, tmp_path):
    first = build_or_load(sample_pdf, embeddings=embeddings, index_dir=tmp_path)
    second = build_or_load(sample_pdf, embeddings=embeddings, index_dir=tmp_path)
    assert not first.from_cache and second.from_cache
    assert second.n_chunks == first.n_chunks
