"""Unit tests for chunking strategies and metadata preservation."""

import pytest
from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.recursive_chunker import RecursiveCharacterChunker
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.registry import get_chunker
from src.ingestion.models import Document, DocumentMetadata


@pytest.fixture
def sample_document():
    text = (
        "Retrieval-Augmented Generation (RAG) optimizes LLM outputs with external knowledge. "
        "Chunking is the foundational stage where long documents are partitioned into segments.\n\n"
        "Fixed-size chunkers split strictly by character counts. "
        "Recursive chunkers respect natural semantic boundaries like paragraphs and sentences.\n\n"
        "Semantic chunkers cluster sentences based on embedding similarity drops."
    )
    metadata = DocumentMetadata(
        source="rag_intro.md",
        doc_id="doc_rag_001",
        file_type=".md",
        page_number=1
    )
    return Document(id="doc_rag_001", text=text, metadata=metadata)


def test_fixed_chunker(sample_document):
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.doc_id == sample_document.id
        assert c.chunk_index == i
        assert c.metadata.source == "rag_intro.md"
        assert c.metadata.page_number == 1
        assert len(c.text) <= 100
        assert c.end_char > c.start_char


def test_recursive_chunker(sample_document):
    chunker = RecursiveCharacterChunker(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 2
    # Check that paragraphs or sentences were preserved cleanly
    for c in chunks:
        assert c.text.strip() != ""
        assert c.metadata.doc_id == sample_document.id


def test_semantic_chunker(sample_document):
    chunker = SemanticChunker(chunk_size=150, chunk_overlap=20)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 1
    assert chunks[0].doc_id == sample_document.id


def test_chunker_registry():
    c1 = get_chunker("fixed", chunk_size=256, chunk_overlap=32)
    assert isinstance(c1, FixedSizeChunker)

    c2 = get_chunker("recursive", chunk_size=512, chunk_overlap=64)
    assert isinstance(c2, RecursiveCharacterChunker)

    c3 = get_chunker("semantic", chunk_size=512, chunk_overlap=64)
    assert isinstance(c3, SemanticChunker)

    with pytest.raises(ValueError):
        get_chunker("unsupported_strategy")


def test_invalid_chunk_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, chunk_overlap=120)

