from rulebook_qa.ingest import load_pdf, split_documents


def test_pdf_loads_with_page_numbers(sample_pdf):
    pages = load_pdf(sample_pdf)
    assert [p.metadata["page_number"] for p in pages] == [1, 2, 3, 4, 5, 6]
    assert all(p.metadata["source_file"] == sample_pdf.name for p in pages)


def test_repeated_footer_is_removed(sample_pdf):
    pages = load_pdf(sample_pdf)
    for p in pages:
        assert "sample document for demo use" not in p.page_content
    assert "75%" in pages[1].page_content  # real content is kept


def test_chunks_respect_size_and_keep_metadata(sample_pdf):
    chunks = split_documents(load_pdf(sample_pdf), chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 6
    assert all(len(c.page_content) <= 500 for c in chunks)
    assert all("page_number" in c.metadata for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
