"""Unit tests for generation, grounding, and abstention."""

from src.generation.llm_generator import LLMGenerator
from src.ingestion.models import DocumentChunk, DocumentMetadata, GroundedCitation, RetrievalResult


def test_grounded_generation_with_citation():
    generator = LLMGenerator(provider="mock")
    meta = DocumentMetadata(source="raft.pdf", doc_id="d1", file_type=".pdf", page_number=3)
    chunk = DocumentChunk(
        chunk_id="c1",
        doc_id="d1",
        text="The Raft algorithm guarantees leader election and state machine safety across terms.",
        chunk_index=0,
        start_char=0,
        end_char=84,
        metadata=meta
    )
    citation = GroundedCitation(
        source="raft.pdf",
        chunk_id="c1",
        doc_id="d1",
        page_number=3,
        snippet=chunk.text[:50]
    )
    result = RetrievalResult(chunk=chunk, score=0.95, rank=1)

    response = generator.generate(
        query="What guarantees does the Raft algorithm provide?",
        context=chunk.text,
        citations=[citation],
        retrieved_chunks=[result]
    )

    assert response.is_abstained is False
    assert "Raft algorithm guarantees" in response.answer
    assert "[Source 1: raft.pdf, Page 3]" in response.answer
    assert len(response.citations) == 1


def test_abstention_on_unrelated_query():
    generator = LLMGenerator(provider="mock")
    meta = DocumentMetadata(source="raft.pdf", doc_id="d1", file_type=".pdf", page_number=1)
    chunk = DocumentChunk(
        chunk_id="c1",
        doc_id="d1",
        text="Raft handles consensus through leader election and heartbeat timers.",
        chunk_index=0,
        start_char=0,
        end_char=69,
        metadata=meta
    )
    result = RetrievalResult(chunk=chunk, score=0.1, rank=1)

    # Query about completely unrelated topic
    response = generator.generate(
        query="What is the chemical formula for photosynthesis?",
        context=chunk.text,
        citations=[],
        retrieved_chunks=[result]
    )

    assert response.is_abstained is True
    assert "I do not have sufficient information" in response.answer


def test_abstention_on_empty_context():
    generator = LLMGenerator(provider="mock")
    response = generator.generate(
        query="Any query",
        context="",
        citations=[],
        retrieved_chunks=[]
    )
    assert response.is_abstained is True
    assert "I do not have sufficient information" in response.answer

