"""Repeatable performance regression benchmark tests.

Isolated from standard fast unit/integration CI matrix to prevent hardware-flakiness
in resource-constrained shared virtual machines.
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
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_retriever import SparseRetriever


@pytest.mark.performance
def test_hybrid_rrf_latency_sla():
    """Verify Hybrid RRF retrieval satisfies latency SLA (< 100ms per query on CPU)."""
    benchmark_path = Path("data/benchmarks/rag_benchmark.json")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus_raw = data["corpus"]
    queries = [q["query"] for q in data["benchmark_queries"]]

    # Ingest and Chunk
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

    # 1. Warm-up Phase: 2 queries
    for q in queries[:2]:
        _ = hybrid.retrieve(q, top_k=5)

    # 2. Measurement Phase: 2 repetitions
    latencies = []
    for _ in range(2):
        for q in queries:
            t0 = time.perf_counter()
            _ = hybrid.retrieve(q, top_k=5)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    # Assert reasonable CPU SLA
    assert mean_latency < 100.0, f"Performance regression: Hybrid RRF mean latency {mean_latency:.1f}ms exceeds 100ms SLA!"
    assert p95_latency < 200.0, f"Performance regression: Hybrid RRF p95 latency {p95_latency:.1f}ms exceeds 200ms SLA!"

