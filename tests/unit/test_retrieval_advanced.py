"""Unit tests for advanced retrieval components: HyDE, RAG Fusion, and Smart Query Router."""

import pytest
from src.embeddings.fast_embedder import FastDeterministicEmbedder
from src.generation.llm_generator import LLMGenerator
from src.ingestion.models import DocumentChunk, DocumentMetadata, RetrievalResult
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.hyde_retriever import HyDERetriever
from src.retrieval.query_router import SmartQueryRouter
from src.retrieval.rag_fusion import RAGFusionRetriever
from src.retrieval.sparse_retriever import SparseRetriever


@pytest.fixture
def test_chunks():
    meta1 = DocumentMetadata(source="proctoriq_id.md", doc_id="d1", file_type=".md", page_number=1)
    meta2 = DocumentMetadata(source="proctoriq_audio.md", doc_id="d2", file_type=".md", page_number=1)
    meta3 = DocumentMetadata(source="proctoriq_lockdown.md", doc_id="d3", file_type=".md", page_number=2)

    return [
        DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            text="Section 1.0: Facial similarity threshold is 95% under FERPA standards.",
            chunk_index=0,
            start_char=0,
            end_char=70,
            metadata=meta1,
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            text="Section 2.0: Background acoustic noise exceeding 65 dB triggers an ambient audio violation.",
            chunk_index=0,
            start_char=0,
            end_char=91,
            metadata=meta2,
        ),
        DocumentChunk(
            chunk_id="c3",
            doc_id="d3",
            text="Section 3.0: Browser lockdown terminates secondary virtual displays and unauthorized keystrokes.",
            chunk_index=0,
            start_char=0,
            end_char=97,
            metadata=meta3,
        ),
    ]


@pytest.fixture
def dense_retriever(test_chunks):
    embedder = FastDeterministicEmbedder(dimension=64, normalize=True)
    retriever = DenseRetriever(embedder=embedder)
    retriever.index_chunks(test_chunks)
    return retriever


@pytest.fixture
def sparse_retriever(test_chunks):
    retriever = SparseRetriever(k1=1.5, b=0.75)
    retriever.index_chunks(test_chunks)
    return retriever


@pytest.fixture
def hybrid_retriever(dense_retriever, sparse_retriever):
    return HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion_method="rrf",
        dense_weight=0.5,
        sparse_weight=0.5,
        rrf_k=60,
    )


def test_hyde_retriever_without_generator(dense_retriever):
    embedder = FastDeterministicEmbedder(dimension=64, normalize=True)
    hyde = HyDERetriever(
        dense_retriever=dense_retriever,
        embedder=embedder,
        generator=None,
        blend_with_query=True,
        query_weight=0.3,
        hypothetical_weight=0.7,
    )

    query = "How does facial similarity matching operate?"
    hypo_doc = hyde.generate_hypothetical_document(query)
    assert len(hypo_doc) > len(query)
    assert query in hypo_doc

    results = hyde.retrieve(query, top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert results[0].score >= results[1].score


def test_hyde_retriever_with_generator(dense_retriever):
    embedder = FastDeterministicEmbedder(dimension=64, normalize=True)
    mock_gen = LLMGenerator(provider="mock")
    hyde = HyDERetriever(
        dense_retriever=dense_retriever,
        embedder=embedder,
        generator=mock_gen,
        blend_with_query=False,
    )

    query = "ID verification mechanism"
    hypo_doc = hyde.generate_hypothetical_document(query)
    assert len(hypo_doc) > 0

    results = hyde.retrieve(query, top_k=1)
    assert len(results) == 1
    assert results[0].chunk.chunk_id in {"c1", "c2", "c3"}


def test_rag_fusion_retriever(hybrid_retriever):
    rag_fusion = RAGFusionRetriever(
        base_retriever=hybrid_retriever,
        generator=None,
        num_queries=3,
        rrf_k=60,
    )

    query = "audio decibel limits and microphone sampling"
    variations = rag_fusion.generate_query_variations(query)
    assert len(variations) >= 3

    results = rag_fusion.retrieve(query, top_k=3)
    assert len(results) <= 3
    assert len(results) > 0
    # Ensure chunk IDs are deduplicated
    chunk_ids = [r.chunk.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_smart_query_router_intent_classification(dense_retriever, sparse_retriever, hybrid_retriever):
    embedder = FastDeterministicEmbedder(dimension=64, normalize=True)
    hyde = HyDERetriever(dense_retriever=dense_retriever, embedder=embedder)
    rag_fusion = RAGFusionRetriever(base_retriever=hybrid_retriever)

    router = SmartQueryRouter(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        hyde_retriever=hyde,
        rag_fusion_retriever=rag_fusion,
    )

    # 1. Exact identifier pattern -> sparse_bm25
    strat1, r1 = router.classify_intent('What does "Section 1.0" specify for FERPA?')
    assert strat1 == "sparse_bm25"
    assert r1 == sparse_retriever

    # 2. Multi-faceted / comparative pattern -> rag_fusion
    strat2, r2 = router.classify_intent("Compare audio monitoring vs browser lockdown rules")
    assert strat2 == "rag_fusion"
    assert r2 == rag_fusion

    # 3. Conceptual open-ended pattern -> hyde
    strat3, r3 = router.classify_intent("How does the identity verification pipeline validate photos?")
    assert strat3 == "hyde"
    assert r3 == hyde

    # 4. Standard default pattern -> hybrid_rrf
    strat4, r4 = router.classify_intent("candidate proctoring system configuration")
    assert strat4 == "hybrid_rrf"
    assert r4 == hybrid_retriever


def test_smart_query_router_retrieve(dense_retriever, sparse_retriever, hybrid_retriever):
    embedder = FastDeterministicEmbedder(dimension=64, normalize=True)
    hyde = HyDERetriever(dense_retriever=dense_retriever, embedder=embedder)
    rag_fusion = RAGFusionRetriever(base_retriever=hybrid_retriever)

    router = SmartQueryRouter(
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        hybrid_retriever=hybrid_retriever,
        hyde_retriever=hyde,
        rag_fusion_retriever=rag_fusion,
    )

    results = router.retrieve("What is Section 1.0 threshold?", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.chunk_id == "c1"
