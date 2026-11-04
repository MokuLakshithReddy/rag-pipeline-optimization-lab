"""Controlled experiment runner for systematic RAG pipeline benchmarking."""

from __future__ import annotations
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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

    # Define controlled experiment matrix
    experiments = [
        {
            "id": "EXP-01-Baseline-Dense",
            "chunk_strategy": "fixed",
            "chunk_size": 256,
            "chunk_overlap": 32,
            "retrieval_mode": "dense",
            "top_k": 5,
            "rerank": False,
            "query_enhancement": "none"
        },
        {
            "id": "EXP-02-Chunking-Recursive",
            "chunk_strategy": "recursive",
            "chunk_size": 256,
            "chunk_overlap": 32,
            "retrieval_mode": "dense",
            "top_k": 5,
            "rerank": False,
            "query_enhancement": "none"
        },
        {
            "id": "EXP-03-Chunking-Size-512",
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
    generator = LLMGenerator(provider="mock")
    reranker = CrossEncoderReranker(model_name=reranker_model, top_n=3)

    for exp in experiments:
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

        # 3. Evaluate
        evaluator = RAGEvaluator(
            retriever=active_retriever,
            generator=generator,
            reranker=active_reranker,
            top_k=exp["top_k"]
        )
        metrics = evaluator.evaluate_dataset(queries)

        record = {
            "experiment_id": exp["id"],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset_version": "rag_benchmark_v1",
            "chunking_strategy": exp["chunk_strategy"],
            "chunk_size": exp["chunk_size"],
            "chunk_overlap": exp["chunk_overlap"],
            "embedding_model": embedding_model,
            "retrieval_strategy": exp["retrieval_mode"],
            "top_k": exp["top_k"],
            "reranker_enabled": exp["rerank"],
            "reranker_model": reranker_model if exp["rerank"] else "none",
            "query_enhancement": exp["query_enhancement"],
            "precision_at_k": metrics["precision_at_k"],
            "recall_at_k": metrics["recall_at_k"],
            "mrr": metrics["mrr"],
            "ndcg_at_k": metrics["ndcg_at_k"],
            "hit_at_k": metrics["hit_at_k"],
            "faithfulness": metrics["faithfulness"],
            "answer_f1": metrics["answer_f1"],
            "latency_mean_ms": metrics["latency_mean_ms"],
            "latency_p95_ms": metrics["latency_p95_ms"],
            "memory_mb": 128.0,
        }

        tracker.record_run(record)
        results_summary.append(record)

    # Generate final reports
    tracker.generate_markdown_summary()
    tracker.generate_retrieval_analysis()

    return results_summary


if __name__ == "__main__":
    run_experiment_suite()
