"""Evaluation harness for running quantitative benchmarks across RAG pipelines."""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.evaluation.metrics import (
    compute_faithfulness,
    compute_hit_at_k,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_token_f1,
)
from src.generation.base import BaseGenerator
from src.retrieval.base import BaseRetriever
from src.retrieval.context_optimizer import ContextOptimizer
from src.reranking.base import BaseReranker


class RAGEvaluator:
    """Evaluates retrieval accuracy, answer quality, and latency across test datasets."""

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: Optional[BaseGenerator] = None,
        reranker: Optional[BaseReranker] = None,
        context_optimizer: Optional[ContextOptimizer] = None,
        top_k: int = 5
    ):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.context_optimizer = context_optimizer or ContextOptimizer()
        self.top_k = top_k

    def evaluate_dataset(self, benchmark_items: List[Dict[str, Any]]) -> Dict[str, float]:
        """Run full evaluation suite over benchmark queries with ground truth."""
        precisions: List[float] = []
        recalls: List[float] = []
        reciprocal_ranks: List[float] = []
        ndcgs: List[float] = []
        hits: List[float] = []
        latencies: List[float] = []
        faithfulness_scores: List[float] = []
        f1_scores: List[float] = []

        for item in benchmark_items:
            query = item["query"]
            relevant_ids = set(item.get("relevant_doc_ids", []))
            ground_truth = item.get("ground_truth_answer", "")

            start_t = time.perf_counter()

            # 1. Retrieval
            retrieved = self.retriever.retrieve(query, top_k=self.top_k)

            # 2. Reranking (if enabled)
            if self.reranker is not None:
                retrieved = self.reranker.rerank(query, retrieved)

            latency_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(latency_ms)

            retrieved_chunk_doc_ids = [r.chunk.doc_id for r in retrieved]
            # Also allow matching by source name or chunk_id
            retrieved_all_identifiers = []
            for r in retrieved:
                retrieved_all_identifiers.append(r.chunk.doc_id)
                retrieved_all_identifiers.append(r.chunk.metadata.source)
                retrieved_all_identifiers.append(r.chunk.chunk_id)

            if relevant_ids:
                matched_ranks: List[str] = []
                for r in retrieved:
                    matched_id = None
                    for candidate in [r.chunk.doc_id, r.chunk.metadata.source, r.chunk.chunk_id]:
                        if candidate in relevant_ids:
                            matched_id = candidate
                            break
                    if matched_id:
                        matched_ranks.append(matched_id)
                    else:
                        matched_ranks.append("NON_RELEVANT")

                precisions.append(compute_precision_at_k(matched_ranks, relevant_ids, self.top_k))
                recalls.append(compute_recall_at_k(matched_ranks, relevant_ids, self.top_k))
                reciprocal_ranks.append(compute_reciprocal_rank(matched_ranks, relevant_ids))
                ndcgs.append(compute_ndcg_at_k(matched_ranks, relevant_ids, self.top_k))
                hits.append(compute_hit_at_k(matched_ranks, relevant_ids, self.top_k))

            # 3. Generation (if generator supplied)
            if self.generator is not None:
                context_str, citations, accepted_chunks = self.context_optimizer.optimize_context(retrieved)
                gen_resp = self.generator.generate(
                    query=query,
                    context=context_str,
                    citations=citations,
                    retrieved_chunks=accepted_chunks
                )
                if not gen_resp.is_abstained:
                    faithfulness_scores.append(compute_faithfulness(gen_resp.answer, context_str))
                    if ground_truth:
                        f1_scores.append(compute_token_f1(gen_resp.answer, ground_truth))

        results = {
            "total_queries": float(len(benchmark_items)),
            "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
            "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
            "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
            "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "hit_at_k": float(np.mean(hits)) if hits else 0.0,
            "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
            "faithfulness": float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
            "answer_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        }
        return results
