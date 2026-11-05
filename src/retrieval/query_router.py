"""Smart Query Router for adaptive multi-strategy retrieval dispatching.

Inspects incoming user queries and dynamically routes execution across:
1. `sparse_bm25`: Exact section numbers, acronyms, specific parameters, and quoted terms.
2. `hyde`: Abstract, open-ended conceptual inquiries ("how does ... work?", "explain the mechanism").
3. `rag_fusion`: Multi-faceted, comparative, and complex multi-part queries ("vs", "and", "difference between").
4. `hybrid_rrf`: Standard balanced informational queries combining dense and sparse search.

Followed by optional Cross-Encoder reranking and citation tracking.
"""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.ingestion.models import RetrievalResult
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.base import BaseRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.hyde_retriever import HyDERetriever
from src.retrieval.rag_fusion import RAGFusionRetriever
from src.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger("query_router")


class SmartQueryRouter:
    """Classifies queries by linguistic and structural intent to dispatch optimal retrieval strategy."""

    # Acronyms and specific technical identifier patterns
    EXACT_IDENTIFIER_PATTERN = re.compile(
        r'\b(?:section\s*\d+|rule\s*\d+|v\d+|\d+f\s*\+\s*\d+|rfc\s*\d+|iso\s*\d+|ferpa|gdpr)\b',
        re.IGNORECASE
    )
    ACRONYM_PATTERN = re.compile(r'\b[A-Z]{3,}\b')

    # Multi-faceted / comparative patterns
    MULTI_FACETED_PATTERN = re.compile(
        r'\b(?:vs|versus|difference\s+between|compare|and\s+also|both\s+.+\s+and)\b',
        re.IGNORECASE
    )

    # Conceptual / open-ended explanatory patterns
    CONCEPTUAL_PATTERN = re.compile(
        r'\b(?:how\s+does|explain\s+how|what\s+is\s+the\s+mechanism|why\s+does|rationale|understand)\b',
        re.IGNORECASE
    )

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        sparse_retriever: Optional[SparseRetriever] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        hyde_retriever: Optional[HyDERetriever] = None,
        rag_fusion_retriever: Optional[RAGFusionRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None
    ):
        self.hybrid_retriever = hybrid_retriever
        self.sparse_retriever = sparse_retriever or hybrid_retriever.sparse_retriever
        self.dense_retriever = dense_retriever or hybrid_retriever.dense_retriever
        self.hyde_retriever = hyde_retriever
        self.rag_fusion_retriever = rag_fusion_retriever
        self.reranker = reranker

    def get_retriever_for_strategy(self, strategy: str) -> BaseRetriever:
        """Map strategy string to retriever instance."""
        if strategy == "sparse_bm25" and self.sparse_retriever:
            return self.sparse_retriever
        elif strategy == "dense_vector" and self.dense_retriever:
            return self.dense_retriever
        elif strategy == "hyde" and self.hyde_retriever:
            return self.hyde_retriever
        elif strategy == "rag_fusion" and self.rag_fusion_retriever:
            return self.rag_fusion_retriever
        return self.hybrid_retriever

    def classify_intent(self, query: str) -> Tuple[str, BaseRetriever]:
        """Classify query intent and return both the strategy name and the retriever object."""
        strategy, _ = self.route_query(query)
        retriever = self.get_retriever_for_strategy(strategy)
        return strategy, retriever

    def route_query(self, query: str) -> Tuple[str, str]:
        """Classify query into optimal retrieval strategy and provide justification."""
        q = query.strip()

        # 1. Exact quoted strings or specific section/rule identifiers -> Sparse BM25
        if '"' in q or "'" in q or self.EXACT_IDENTIFIER_PATTERN.search(q) or self.ACRONYM_PATTERN.search(q):
            return "sparse_bm25", "Exact quotes, section codes, or technical identifiers detected"

        # 2. Multi-faceted or comparative question -> RAG Fusion
        if self.MULTI_FACETED_PATTERN.search(q) or len(re.split(r'\band\b', q, flags=re.IGNORECASE)) > 2:
            if self.rag_fusion_retriever is not None:
                return "rag_fusion", "Multi-faceted, comparative, or compound query detected"

        # 3. Conceptual / explanatory question -> HyDE
        if self.CONCEPTUAL_PATTERN.search(q) or len(q.split()) > 10:
            if self.hyde_retriever is not None:
                return "hyde", "Conceptual, open-ended explanatory inquiry detected"

        # 4. Standard default -> Hybrid RRF
        return "hybrid_rrf", "Standard informational query suited for balanced Hybrid RRF"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        apply_reranker: bool = True
    ) -> List[RetrievalResult]:
        """Dynamically route and execute retrieval based on query classification."""
        strategy, reason = self.route_query(query)
        logger.info(f"Smart Query Router selected '{strategy}' for query: '{query[:50]}...' ({reason})")

        # Select candidate retriever based on route
        retriever: BaseRetriever
        if strategy == "sparse_bm25" and self.sparse_retriever:
            retriever = self.sparse_retriever
        elif strategy == "hyde" and self.hyde_retriever:
            retriever = self.hyde_retriever
        elif strategy == "rag_fusion" and self.rag_fusion_retriever:
            retriever = self.rag_fusion_retriever
        else:
            strategy = "hybrid_rrf"
            retriever = self.hybrid_retriever

        # Retrieve initial candidate pool
        initial_k = top_k * 2 if (apply_reranker and self.reranker) else top_k
        candidates = retriever.retrieve(query, top_k=initial_k, filters=filters)

        # Annotate results with router metadata
        for c in candidates:
            c.retriever_type = f"{strategy}_routed"

        # Apply Cross-Encoder Reranker if configured
        if apply_reranker and self.reranker and candidates:
            reranked = self.reranker.rerank(query, candidates)
            return reranked[:top_k]

        return candidates[:top_k]
