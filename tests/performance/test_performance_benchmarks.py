"""Repeatable performance regression benchmark tests.

Enforces strict, multi-strategy latency SLAs as genuine pytest CI gates
with warm-up passes and multi-repetition averaging.
"""

import gc
import json
import time
from pathlib import Path
import numpy as np
import pytest

from src.chunking.registry import get_chunker
from src.embeddings.registry import get_embedder
from src.ingestion.models import Document, DocumentMetadata
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_retriever import SparseRetriever


@pytest.fixture(scope="module")
def benchmark_pipeline():
    """Module-scoped pipeline fixture to avoid duplicate indexing across performance tests."""
    benchmark_path = Path("data/benchmarks/rag_benchmark.json")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus_raw = data["corpus"]
    queries = [q["query"] for q in data["benchmark_queries"]]

    documents = [
        Document(
            id=item["id"],
            text=item["text"],
            metadata=DocumentMetadata(source=item["source"], doc_id=item["id"], file_type=".txt")
        )
        for item in corpus_raw
    ]
    chunker = get_chunker(strategy="recursive", chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_documents(documents)

    embedder = get_embedder(model_name="all-MiniLM-L6-v2")
    dense = DenseRetriever(embedder=embedder)
    dense.index_chunks(chunks)

    sparse = SparseRetriever()
    sparse.index_chunks(chunks)

    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, fusion_method="rrf", rrf_k=60)
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3, strict_mode=True)

    return {
        "queries": queries,
        "dense": dense,
        "sparse": sparse,
        "hybrid": hybrid,
        "reranker": reranker
    }


def _measure_latencies(fn, queries, warmup_count=2, repetitions=2):
    # Warm-up phase
    for q in queries[:warmup_count]:
        _ = fn(q)

    # Measurement phase
    latencies = []
    for _ in range(repetitions):
        for q in queries:
            t0 = time.perf_counter()
            _ = fn(q)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies)), float(np.percentile(latencies, 95))


@pytest.mark.performance
def test_sparse_bm25_latency_sla(benchmark_pipeline):
    """Enforce Sparse Okapi BM25 latency SLA (mean < 5.0ms, p95 < 15.0ms on CPU)."""
    sparse = benchmark_pipeline["sparse"]
    queries = benchmark_pipeline["queries"]
    mean_lat, p95_lat = _measure_latencies(lambda q: sparse.retrieve(q, top_k=5), queries)

    assert mean_lat <= 5.0, f"Sparse BM25 mean latency {mean_lat:.2f}ms breached SLA of 5.0ms"
    assert p95_lat <= 15.0, f"Sparse BM25 p95 latency {p95_lat:.2f}ms breached SLA of 15.0ms"


@pytest.mark.performance
def test_dense_vector_latency_sla(benchmark_pipeline):
    """Enforce Dense Vector latency SLA (mean < 25.0ms, p95 < 50.0ms on CPU)."""
    dense = benchmark_pipeline["dense"]
    queries = benchmark_pipeline["queries"]
    mean_lat, p95_lat = _measure_latencies(lambda q: dense.retrieve(q, top_k=5), queries)

    assert mean_lat <= 25.0, f"Dense vector mean latency {mean_lat:.2f}ms breached SLA of 25.0ms"
    assert p95_lat <= 50.0, f"Dense vector p95 latency {p95_lat:.2f}ms breached SLA of 50.0ms"


@pytest.mark.performance
def test_hybrid_rrf_latency_sla(benchmark_pipeline):
    """Enforce Hybrid RRF latency SLA (mean < 10.0ms, p95 < 25.0ms on CPU)."""
    hybrid = benchmark_pipeline["hybrid"]
    queries = benchmark_pipeline["queries"]
    mean_lat, p95_lat = _measure_latencies(lambda q: hybrid.retrieve(q, top_k=5), queries)

    assert mean_lat <= 10.0, f"Hybrid RRF mean latency {mean_lat:.2f}ms breached SLA of 10.0ms"
    assert p95_lat <= 25.0, f"Hybrid RRF p95 latency {p95_lat:.2f}ms breached SLA of 25.0ms"


@pytest.mark.performance
def test_hybrid_reranked_cpu_latency_sla(benchmark_pipeline):
    """Enforce Cross-Encoder Reranking SLA (mean < 350.0ms, p95 < 600.0ms on CPU)."""
    hybrid = benchmark_pipeline["hybrid"]
    reranker = benchmark_pipeline["reranker"]
    queries = benchmark_pipeline["queries"]

    def _rerank_search(q):
        candidates = hybrid.retrieve(q, top_k=5)
        return reranker.rerank(q, candidates)

    mean_lat, p95_lat = _measure_latencies(_rerank_search, queries, warmup_count=2, repetitions=2)

    assert mean_lat <= 350.0, f"Cross-Encoder reranking mean latency {mean_lat:.2f}ms breached SLA of 350.0ms"
    assert p95_lat <= 600.0, f"Cross-Encoder reranking p95 latency {p95_lat:.2f}ms breached SLA of 600.0ms"
