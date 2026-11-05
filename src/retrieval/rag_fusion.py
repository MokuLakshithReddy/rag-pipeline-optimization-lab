"""RAG Fusion Retriever: Multi-Query Generation and Reciprocal Rank Fusion.

Implements RAG Fusion (Rackauckas, 2023):
1. Expands a single user query into multiple diverse query formulations/perspectives using an LLM or query enhancer.
2. Executes multi-query retrieval across the base retriever for every generated sub-query.
3. Fuses the resulting ranked candidate lists using Reciprocal Rank Fusion (RRF) to produce
   a comprehensive, high-recall ranking that resolves multi-faceted and complex information needs.
"""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

from src.generation.base import BaseGenerator
from src.ingestion.models import DocumentChunk, RetrievalResult
from src.retrieval.base import BaseRetriever
from src.retrieval.query_enhancement import QueryEnhancer

logger = logging.getLogger("rag_fusion")


class RAGFusionRetriever:
    """RAG Fusion Retriever combining multi-query expansion with Reciprocal Rank Fusion."""

    FUSION_PROMPT_TEMPLATE = (
        "Generate 3 diverse, distinct search queries from different angles or perspectives "
        "that help retrieve all necessary factual evidence to answer this question:\n"
        "User Question: {query}\n"
        "Provide 3 queries, one per line, without numbering or bullets:"
    )

    def __init__(
        self,
        base_retriever: BaseRetriever,
        generator: Optional[BaseGenerator] = None,
        num_queries: int = 3,
        rrf_k: int = 60
    ):
        self.base_retriever = base_retriever
        self.generator = generator
        self.num_queries = num_queries
        self.rrf_k = rrf_k

    def generate_query_variations(self, query: str) -> List[str]:
        """Generate diverse query formulations for multi-query retrieval."""
        variations: List[str] = [query.strip()]

        if self.generator is not None:
            try:
                prompt = self.FUSION_PROMPT_TEMPLATE.format(query=query)
                resp = self.generator.generate(
                    query=prompt,
                    context="Query expansion domain",
                    citations=[],
                    retrieved_chunks=[]
                )
                if resp and resp.answer:
                    lines = [line.strip() for line in resp.answer.splitlines() if line.strip()]
                    for line in lines:
                        clean_q = re.sub(r'^\d+[\.\)]\s*|-\s*', '', line).strip()
                        if len(clean_q) > 5 and clean_q not in variations:
                            variations.append(clean_q)
                        if len(variations) >= self.num_queries + 1:
                            break
            except Exception as e:
                logger.warning(f"RAG Fusion LLM expansion failed ({e}), falling back to deterministic enhancer.")

        # Fallback to QueryEnhancer if LLM expansion did not produce enough variations
        if len(variations) <= 1:
            decomposed = QueryEnhancer.generate_multi_queries(query)
            for q_dec in decomposed:
                if q_dec not in variations:
                    variations.append(q_dec)

            expanded_q, _ = QueryEnhancer.expand_query(query)
            if expanded_q not in variations:
                variations.append(expanded_q)

        return variations[: self.num_queries + 1]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Execute RAG Fusion retrieval: generate multi-queries -> retrieve per query -> RRF merge."""
        if not query.strip():
            return []

        # 1. Generate query variations
        query_variations = self.generate_query_variations(query)
        logger.debug(f"RAG Fusion expanded into {len(query_variations)} queries: {query_variations}")

        # 2. Retrieve candidates for each query variation
        ranked_lists: List[List[RetrievalResult]] = []
        for q_var in query_variations:
            candidates = self.base_retriever.retrieve(q_var, top_k=top_k * 2, filters=filters)
            ranked_lists.append(candidates)

        # 3. Apply Reciprocal Rank Fusion across all query result sets
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, DocumentChunk] = {}
        strategy_counts: Dict[str, int] = defaultdict(int)

        for rank_list in ranked_lists:
            for rank, item in enumerate(rank_list, start=1):
                chunk_id = getattr(item.chunk, "chunk_id", getattr(item.chunk, "id", None)) or f"c_{rank}"
                rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = item.chunk
                strategy_counts[chunk_id] += 1

        if not rrf_scores:
            return []

        # 4. Sort by fused score descending
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[RetrievalResult] = []
        for final_rank, (chunk_id, fused_score) in enumerate(sorted_chunks, start=1):
            results.append(
                RetrievalResult(
                    chunk=chunk_map[chunk_id],
                    score=float(fused_score),
                    rank=final_rank,
                    retrieval_strategy="rag_fusion"
                )
            )

        return results
