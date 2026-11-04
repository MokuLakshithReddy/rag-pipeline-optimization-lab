"""Hybrid retriever combining dense semantic search and sparse BM25 with Rank Fusion."""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from src.ingestion.models import DocumentChunk, RetrievalResult
from src.retrieval.base import BaseRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever


class HybridRetriever(BaseRetriever):
    """Fuses dense vector retrieval and sparse keyword BM25 retrieval."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        fusion_method: Literal["rrf", "weighted"] = "rrf",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_method = fusion_method
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        self.dense_retriever.index_chunks(chunks)
        self.sparse_retriever.index_chunks(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        if not query.strip():
            return []

        # Retrieve a broader candidate pool from both branches
        candidate_k = max(top_k * 2, 20)
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k, filters=filters)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=candidate_k, filters=filters)

        if not dense_results and not sparse_results:
            return []
        if not dense_results:
            return sparse_results[:top_k]
        if not sparse_results:
            return dense_results[:top_k]

        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(dense_results, sparse_results, top_k)
        else:
            return self._weighted_score_fusion(dense_results, sparse_results, top_k)

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """Compute Reciprocal Rank Fusion (RRF) scores."""
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        # Dense contributions
        for rank, res in enumerate(dense_results, start=1):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            rrf_score = self.dense_weight / (self.rrf_k + rank)
            scores[cid] = scores.get(cid, 0.0) + rrf_score

        # Sparse contributions
        for rank, res in enumerate(sparse_results, start=1):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            rrf_score = self.sparse_weight / (self.rrf_k + rank)
            scores[cid] = scores.get(cid, 0.0) + rrf_score

        # Sort by fused score descending
        sorted_cids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

        fused_results: List[RetrievalResult] = []
        for rank, cid in enumerate(sorted_cids, start=1):
            fused_results.append(
                RetrievalResult(
                    chunk=chunk_map[cid],
                    score=float(scores[cid]),
                    rank=rank,
                    retriever_type="hybrid_rrf"
                )
            )

        return fused_results

    def _weighted_score_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """Min-max normalize dense and sparse scores, then calculate weighted sum."""
        chunk_map: Dict[str, DocumentChunk] = {}
        dense_scores: Dict[str, float] = {r.chunk.chunk_id: r.score for r in dense_results}
        sparse_scores: Dict[str, float] = {r.chunk.chunk_id: r.score for r in sparse_results}

        for r in dense_results:
            chunk_map[r.chunk.chunk_id] = r.chunk
        for r in sparse_results:
            chunk_map[r.chunk.chunk_id] = r.chunk

        # Normalize dense scores
        d_vals = list(dense_scores.values())
        min_d, max_d = min(d_vals), max(d_vals)
        range_d = (max_d - min_d) if (max_d - min_d) > 0 else 1.0

        # Normalize sparse scores
        s_vals = list(sparse_scores.values())
        min_s, max_s = min(s_vals), max(s_vals)
        range_s = (max_s - min_s) if (max_s - min_s) > 0 else 1.0

        all_cids = set(chunk_map.keys())
        fused_scores: Dict[str, float] = {}

        for cid in all_cids:
            d_norm = ((dense_scores.get(cid, min_d) - min_d) / range_d) if cid in dense_scores else 0.0
            s_norm = ((sparse_scores.get(cid, min_s) - min_s) / range_s) if cid in sparse_scores else 0.0
            fused_scores[cid] = (self.dense_weight * d_norm) + (self.sparse_weight * s_norm)

        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

        return [
            RetrievalResult(
                chunk=chunk_map[cid],
                score=float(fused_scores[cid]),
                rank=rank,
                retriever_type="hybrid_weighted"
            )
            for rank, cid in enumerate(sorted_cids, start=1)
        ]

    def clear(self) -> None:
        self.dense_retriever.clear()
        self.sparse_retriever.clear()

