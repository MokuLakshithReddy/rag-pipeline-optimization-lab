"""Controlled experiment runner for systematic RAG pipeline benchmarking with real resource profiling."""

from __future__ import annotations
import datetime
import gc
import json
import logging
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path when invoked directly
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
import tracemalloc
from typing import Any, Dict, List, Optional
import psutil

from src.chunking.registry import get_chunker
from src.embeddings.registry import get_embedder
from src.evaluation.evaluator import RAGEvaluator
from src.experiments.tracker import ExperimentTracker
from src.generation.llm_generator import LLMGenerator
from src.ingestion.models import Document, DocumentMetadata
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.context_optimizer import ContextOptimizer
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_retriever import SparseRetriever


def load_benchmark_data(benchmark_path: str | Path = "data/benchmarks/rag_benchmark.json") -> Dict[str, Any]:
    """Load benchmark documents and queries."""
    path = Path(benchmark_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment_suite(
    benchmark_path: str | Path = "data/benchmarks/rag_benchmark.json",
    reports_dir: str | Path = "reports",
    embedding_model: str = "all-MiniLM-L6-v2",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Execute the full suite of controlled optimization experiments."""
    data = load_benchmark_data(benchmark_path)
    corpus_raw = data["corpus"]
    queries = data["benchmark_queries"]

    documents = [
        Document(
            id=item["id"],
            text=item["text"],
            metadata=DocumentMetadata(
                source=item["source"],
                doc_id=item["id"],
                file_type=f".{item['source'].split('.')[-1]}"
            )
        )
        for item in corpus_raw
    ]

    tracker = ExperimentTracker(reports_dir=reports_dir)
    tracker.clean_csv()
    results_summary: List[Dict[str, Any]] = []

    # Controlled experiment matrix
    experiments = [
        {
            "id": "EXP-01-Baseline-Dense-Fixed256",
            "chunk_strategy": "fixed",
            "chunk_size": 256,
            "chunk_overlap": 32,
            "retrieval_mode": "dense",
            "top_k": 5,
            "rerank": False,
            "query_enhancement": "none"
        },
        {
            "id": "EXP-02-Chunking-Recursive256",
            "chunk_strategy": "recursive",
            "chunk_size": 256,
            "chunk_overlap": 32,
            "retrieval_mode": "dense",
            "top_k": 5,
            "rerank": False,
            "query_enhancement": "none"
        },
        {
            "id": "EXP-03-Chunking-Recursive512",
            "chunk_strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "retrieval_mode": "dense",
            "top_k": 5,
            "rerank": False,
            "query_enhancement": "none"
        },
        {
            "id": "EXP-04-Sparse-BM25",
            "chunk_strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "retrieval_mode": "sparse",
            "top_k": 5,
            "rerank": False,
            "query_enhancement": "none"
        },
        {
            "id": "EXP-05-Hybrid-RRF",
            "chunk_strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "retrieval_mode": "hybrid",
            "top_k": 5,
            "rerank": False,
            "query_enhancement": "none"
        },
        {
            "id": "EXP-06-Hybrid-Reranked",
            "chunk_strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "retrieval_mode": "hybrid",
            "top_k": 5,
            "rerank": True,
            "query_enhancement": "none"
        },
    ]

    embedder = get_embedder(model_name=embedding_model)

    # Configurable LLM generator: "mock" for fast offline CI or "local"/"gemini" for real model execution
    resolved_provider = (
        llm_provider
        or os.environ.get("EXPERIMENT_LLM_PROVIDER")
        or os.environ.get("LLM_PROVIDER")
        or "mock"
    )
    resolved_model = (
        llm_model
        or os.environ.get("EXPERIMENT_LLM_MODEL")
        or os.environ.get("LLM_MODEL_NAME")
    )
    generator = LLMGenerator(provider=resolved_provider, model_name=resolved_model)

    # Strict mode guarantees neural cross-encoder runs and raises an error instead of silently falling back
    reranker = CrossEncoderReranker(model_name=reranker_model, top_n=3, strict_mode=True)

    process = psutil.Process()

    for exp in experiments:
        gc.collect()
        torch_cuda_peak_mb = 0.0
        if "torch" in sys.modules:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        tracemalloc.start()
        tracemalloc.reset_peak()
        mem_rss_before = process.memory_info().rss / (1024 * 1024)

        # 1. Chunking
        chunker = get_chunker(
            strategy=exp["chunk_strategy"],
            chunk_size=exp["chunk_size"],
            chunk_overlap=exp["chunk_overlap"]
        )
        chunks = chunker.chunk_documents(documents)

        # 2. Retriever setup
        dense_retriever = DenseRetriever(embedder=embedder)
        sparse_retriever = SparseRetriever()

        if exp["retrieval_mode"] == "dense":
            dense_retriever.index_chunks(chunks)
            active_retriever = dense_retriever
        elif exp["retrieval_mode"] == "sparse":
            sparse_retriever.index_chunks(chunks)
            active_retriever = sparse_retriever
        else:
            hybrid = HybridRetriever(
                dense_retriever=dense_retriever,
                sparse_retriever=sparse_retriever,
                fusion_method="rrf"
            )
            hybrid.index_chunks(chunks)
            active_retriever = hybrid

        active_reranker = reranker if exp["rerank"] else None

        # 3. Evaluate with full latency separation
        evaluator = RAGEvaluator(
            retriever=active_retriever,
            generator=generator,
            reranker=active_reranker,
            top_k=exp["top_k"]
        )
        metrics = evaluator.evaluate_dataset(queries)

        # 4. Strict reranker status validation
        if exp["rerank"]:
            status = metrics.get("reranker_status", "none")
            if "neural_cross_encoder" not in status:
                raise RuntimeError(
                    f"Reranker strict mode check failed for {exp['id']}: expected 'neural_cross_encoder', got '{status}'"
                )

        _, peak_traced_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mem_rss_after = process.memory_info().rss / (1024 * 1024)
        delta_rss_mb = max(0.0, mem_rss_after - mem_rss_before)
        peak_heap_mb = peak_traced_bytes / (1024 * 1024)

        if "torch" in sys.modules:
            import torch
            if torch.cuda.is_available():
                torch_cuda_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        record = {
            "experiment_id": exp["id"],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset_version": "rag_benchmark_v2_diverse",
            "chunking_strategy": exp["chunk_strategy"],
            "chunk_size": exp["chunk_size"],
            "chunk_overlap": exp["chunk_overlap"],
            "embedding_model": embedding_model,
            "retrieval_strategy": exp["retrieval_mode"],
            "top_k": exp["top_k"],
            "reranker_enabled": exp["rerank"],
            "reranker_model": reranker_model if exp["rerank"] else "none",
            "reranker_status": metrics.get("reranker_status", "none"),
            "query_enhancement": exp["query_enhancement"],
            "generator_provider": generator.provider,
            "generator_model": generator.model_name,
            "precision_at_k": metrics["precision_at_k"],
            "recall_at_k": metrics["recall_at_k"],
            "mrr": metrics["mrr"],
            "ndcg_at_k": metrics["ndcg_at_k"],
            "hit_at_k": metrics["hit_at_k"],
            "faithfulness": metrics["faithfulness"],
            "answer_f1": metrics["answer_f1"],
            "retrieval_latency_ms": metrics["retrieval_latency_ms"],
            "rerank_latency_ms": metrics["rerank_latency_ms"],
            "generation_latency_ms": metrics["generation_latency_ms"],
            "total_latency_mean_ms": metrics["total_latency_mean_ms"],
            "latency_p95_ms": metrics["latency_p95_ms"],
            "delta_rss_mb": round(delta_rss_mb, 2),
            "memory_rss_mb": round(mem_rss_after, 2),
            "peak_heap_mb": round(peak_heap_mb, 2),
            "torch_cuda_peak_mb": round(torch_cuda_peak_mb, 2),
        }

        tracker.record_run(record)
        results_summary.append(record)

    # Generate final reports
    tracker.generate_markdown_summary()
    tracker.generate_retrieval_analysis()

    return results_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run controlled RAG optimization benchmarks.")
    parser.add_argument("--benchmark-path", default="data/benchmarks/rag_benchmark.json")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--llm-provider", default=None, help="LLM provider: 'mock', 'local', or 'gemini'")
    parser.add_argument("--llm-model", default=None, help="Model name for generator")
    args = parser.parse_args()

    run_experiment_suite(
        benchmark_path=args.benchmark_path,
        reports_dir=args.reports_dir,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model
    )
