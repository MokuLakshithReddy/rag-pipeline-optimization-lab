"""Evaluation harness for running quantitative benchmarks across RAG pipelines."""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Set
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
    """Evaluates retrieval accuracy, answer quality, and granular component latencies."""

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: Optional[BaseGenerator] = None,
        reranker: Optional[BaseReranker] = None,
        context_optimizer: Optional[ContextOptimizer] = None,
        nli_evaluator: Optional[Any] = None,
        top_k: int = 5
    ):
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.context_optimizer = context_optimizer or ContextOptimizer()
        self.nli_evaluator = nli_evaluator
        self.top_k = top_k

    def evaluate_dataset(self, benchmark_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run full evaluation suite over benchmark queries with ground truth."""
        precisions: List[float] = []
        recalls: List[float] = []
        reciprocal_ranks: List[float] = []
        ndcgs: List[float] = []
        hits: List[float] = []
        
        retrieval_latencies: List[float] = []
        rerank_latencies: List[float] = []
        generation_latencies: List[float] = []
        total_latencies: List[float] = []
        
        faithfulness_scores: List[float] = []
        f1_scores: List[float] = []
        reranker_types: Set[str] = set()

        for item in benchmark_items:
            query = item["query"]
            relevant_ids = set(item.get("relevant_doc_ids", []))
            ground_truth = item.get("ground_truth_answer", "")

            t_total_start = time.perf_counter()

            # 1. Retrieval Phase Timing
            t_ret_start = time.perf_counter()
            retrieved = self.retriever.retrieve(query, top_k=self.top_k)
            retrieval_ms = (time.perf_counter() - t_ret_start) * 1000.0
            retrieval_latencies.append(retrieval_ms)

            # 2. Reranking Phase Timing (if active)
            rerank_ms = 0.0
            if self.reranker is not None and retrieved:
                t_rerank_start = time.perf_counter()
                retrieved = self.reranker.rerank(query, retrieved)
                rerank_ms = (time.perf_counter() - t_rerank_start) * 1000.0
                if retrieved:
                    reranker_types.add(retrieved[0].retriever_type)
            rerank_latencies.append(rerank_ms)

            # 3. Retrieval Metrics Calculation
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

                p = compute_precision_at_k(matched_ranks, relevant_ids, self.top_k)
                rec = compute_recall_at_k(matched_ranks, relevant_ids, self.top_k)
                rr = compute_reciprocal_rank(matched_ranks, relevant_ids)
                ndcg = compute_ndcg_at_k(matched_ranks, relevant_ids, self.top_k)
                hit = compute_hit_at_k(matched_ranks, relevant_ids, self.top_k)

                # Assert mathematical bounds
                assert 0.0 <= p <= 1.0, f"Precision@K {p} out of bounds"
                assert 0.0 <= rec <= 1.0, f"Recall@K {rec} out of bounds"
                assert 0.0 <= rr <= 1.0, f"RR {rr} out of bounds"
                assert 0.0 <= ndcg <= 1.0, f"NDCG@K {ndcg} out of bounds"
                assert 0.0 <= hit <= 1.0, f"Hit@K {hit} out of bounds"

                precisions.append(p)
                recalls.append(rec)
                reciprocal_ranks.append(rr)
                ndcgs.append(ndcg)
                hits.append(hit)

            # 4. Generation Phase Timing (if generator supplied)
            gen_ms = 0.0
            if self.generator is not None:
                context_str, citations, accepted_chunks = self.context_optimizer.optimize_context(retrieved)
                t_gen_start = time.perf_counter()
                gen_resp = self.generator.generate(
                    query=query,
                    context=context_str,
                    citations=citations,
                    retrieved_chunks=accepted_chunks
                )
                gen_ms = (time.perf_counter() - t_gen_start) * 1000.0
                generation_latencies.append(gen_ms)

                if not gen_resp.is_abstained:
                    faithfulness_scores.append(compute_faithfulness(gen_resp.answer, context_str, nli_evaluator=self.nli_evaluator))
                    if ground_truth:
                        f1_scores.append(compute_token_f1(gen_resp.answer, ground_truth))
            else:
                generation_latencies.append(0.0)

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            total_latencies.append(total_ms)

        reranker_status = "none"
        if self.reranker is not None:
            reranker_status = ", ".join(reranker_types) if reranker_types else "active"

        results = {
            "total_queries": float(len(benchmark_items)),
            "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
            "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
            "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
            "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "hit_at_k": float(np.mean(hits)) if hits else 0.0,
            "retrieval_latency_ms": float(np.mean(retrieval_latencies)) if retrieval_latencies else 0.0,
            "rerank_latency_ms": float(np.mean(rerank_latencies)) if rerank_latencies else 0.0,
            "generation_latency_ms": float(np.mean(generation_latencies)) if generation_latencies else 0.0,
            "total_latency_mean_ms": float(np.mean(total_latencies)) if total_latencies else 0.0,
            "latency_p50_ms": float(np.percentile(total_latencies, 50)) if total_latencies else 0.0,
            "latency_p95_ms": float(np.percentile(total_latencies, 95)) if total_latencies else 0.0,
            "faithfulness": float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
            "answer_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
            "reranker_status": reranker_status
        }
        return results
