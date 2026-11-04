"""Unit tests for dense, sparse, hybrid retrievers, reranking, and context optimization."""

import numpy as np
import pytest
from src.embeddings.fast_embedder import FastDeterministicEmbedder
from src.ingestion.models import Document, DocumentChunk, DocumentMetadata
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.context_optimizer import ContextOptimizer
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_enhancement import QueryEnhancer
from src.retrieval.sparse_retriever import SparseRetriever


@pytest.fixture
def test_chunks():
    meta1 = DocumentMetadata(source="raft.md", doc_id="d1", file_type=".md", page_number=1)
    meta2 = DocumentMetadata(source="cap.md", doc_id="d2", file_type=".md", page_number=1)
    meta3 = DocumentMetadata(source="vector.md", doc_id="d3", file_type=".md", page_number=2)

    return [
        DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            text="Raft consensus algorithm uses leader election and log replication.",
            chunk_index=0,
            start_char=0,
            end_char=68,
            metadata=meta1
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            text="CAP theorem states that a distributed system cannot achieve consistency and availability under partition.",
            chunk_index=0,
            start_char=0,
            end_char=105,
            metadata=meta2
        ),
        DocumentChunk(
            chunk_id="c3",
            doc_id="d3",
            text="HNSW graphs enable approximate nearest neighbor search in logarithmic time.",
            chunk_index=0,
            start_char=0,
            end_char=75,
            metadata=meta3
        ),
    ]


def test_embedder_normalization_and_dimension():
    embedder = FastDeterministicEmbedder(dimension=64, normalize=True)
    vecs = embedder.embed_texts(["hello world", "vector search"])
    assert vecs.shape == (2, 64)
    # Check L2 normalization: norm should equal 1.0
    for v in vecs:
        norm = np.linalg.norm(v)
        assert pytest.approx(norm, 0.001) == 1.0


def test_dense_retriever(test_chunks):
    embedder = FastDeterministicEmbedder(dimension=64)
    retriever = DenseRetriever(embedder=embedder)
    retriever.index_chunks(test_chunks)

    results = retriever.retrieve("Raft leader election", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].retriever_type == "dense"
    assert results[0].score >= results[1].score


def test_dense_retriever_metadata_filtering(test_chunks):
    embedder = FastDeterministicEmbedder(dimension=64)
    retriever = DenseRetriever(embedder=embedder)
    retriever.index_chunks(test_chunks)

    # Filter to only page 2 (which is c3)
    results = retriever.retrieve("distributed system", top_k=5, filters={"page_number": 2})
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c3"


def test_sparse_bm25_retriever(test_chunks):
    retriever = SparseRetriever()
    retriever.index_chunks(test_chunks)

    results = retriever.retrieve("CAP partition consistency", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c2"
    assert results[0].retriever_type == "sparse"


def test_hybrid_retriever_rrf(test_chunks):
    embedder = FastDeterministicEmbedder(dimension=64)
    dense = DenseRetriever(embedder=embedder)
    sparse = SparseRetriever()

    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, fusion_method="rrf")
    hybrid.index_chunks(test_chunks)

    results = hybrid.retrieve("HNSW nearest neighbor", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c3"
    assert results[0].retriever_type == "hybrid_rrf"


def test_empty_retrieval():
    embedder = FastDeterministicEmbedder(dimension=64)
    dense = DenseRetriever(embedder=embedder)
    sparse = SparseRetriever()
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    assert dense.retrieve("query", top_k=5) == []
    assert sparse.retrieve("query", top_k=5) == []
    assert hybrid.retrieve("query", top_k=5) == []
    assert dense.retrieve("", top_k=5) == []


def test_index_persistence(tmp_path, test_chunks):
    embedder = FastDeterministicEmbedder(dimension=64)
    dense = DenseRetriever(embedder=embedder)
    dense.index_chunks(test_chunks)

    save_dir = tmp_path / "indices"
    dense.save(save_dir)

    dense_loaded = DenseRetriever(embedder=embedder)
    dense_loaded.load(save_dir)

    assert len(dense_loaded.chunks) == 3
    assert dense_loaded.embeddings.shape == (3, 64)
    res = dense_loaded.retrieve("Raft", top_k=1)
    assert res[0].chunk.chunk_id == "c1"


def test_query_enhancement():
    q = "What is RAG and hybrid search?"
    norm = QueryEnhancer.normalize_query(q)
    assert "  " not in norm

    expanded, terms = QueryEnhancer.expand_query("RAG hybrid")
    assert len(terms) > 0
    assert "retrieval" in expanded.lower()

    multis = QueryEnhancer.generate_multi_queries("Raft vs Paxos consensus")
    assert len(multis) >= 2


def test_context_optimizer(test_chunks):
    optimizer = ContextOptimizer(max_context_chars=1000)
    from src.ingestion.models import RetrievalResult
    results = [
        RetrievalResult(chunk=test_chunks[0], score=0.9, rank=1),
        RetrievalResult(chunk=test_chunks[1], score=0.8, rank=2),
    ]
    context_str, citations, accepted = optimizer.optimize_context(results)
    assert len(citations) == 2
    assert "[Source 1: raft.md" in context_str
    assert "[Source 2: cap.md" in context_str

