"""Hypothetical Document Embeddings (HyDE) Retriever.

Implements HyDE (Gao et al., 2022):
1. Generates a hypothetical grounded answer passage for the user query using an LLM or generative template.
2. Embeds the hypothetical passage into the vector space via the dense embedder.
3. Retrieves nearest-neighbor chunks based on passage-to-passage vector similarity,
   bridging the lexical and conceptual gap between terse user queries and detailed knowledge base passages.
"""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional
import numpy as np

from src.embeddings.base import BaseEmbedder
from src.generation.base import BaseGenerator
from src.ingestion.models import DocumentChunk, RetrievalResult
from src.retrieval.dense_retriever import DenseRetriever

logger = logging.getLogger("hyde_retriever")


class HyDERetriever:
    """Hypothetical Document Embeddings (HyDE) Retriever."""

    HYDE_PROMPT_TEMPLATE = (
        "Write a detailed, factual reference passage answering the following technical question. "
        "Include domain terminology, concrete mechanisms, and operational invariants.\n"
        "Question: {query}\n"
        "Hypothetical Reference Passage:"
    )

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        embedder: BaseEmbedder,
        generator: Optional[BaseGenerator] = None,
        blend_with_query: bool = True,
        query_weight: float = 0.3,
        hypothetical_weight: float = 0.7
    ):
        self.dense_retriever = dense_retriever
        self.embedder = embedder
        self.generator = generator
        self.blend_with_query = blend_with_query
        self.query_weight = query_weight
        self.hypothetical_weight = hypothetical_weight

    def generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical answer passage for embedding."""
        if self.generator is not None:
            try:
                # Use active LLM generator to synthesize a plausible document
                prompt = self.HYDE_PROMPT_TEMPLATE.format(query=query)
                # Call generator with empty context to generate an ungrounded candidate
                resp = self.generator.generate(
                    query=prompt,
                    context=f"Technical domain context relating to: {query}",
                    citations=[],
                    retrieved_chunks=[]
                )
                if resp and resp.answer and len(resp.answer.strip()) > 15:
                    clean_ans = re.sub(r'\[Source[^\]]*\]', '', resp.answer).strip()
                    return clean_ans
            except Exception as e:
                logger.warning(f"HyDE LLM generation failed ({e}), using rule-based expansion.")

        # Fallback heuristic expansion when generator is offline
        words = re.findall(r'\b\w+\b', query)
        stopwords = {"what", "is", "the", "how", "to", "does", "in", "for", "a", "an", "of", "with"}
        keywords = [w for w in words if w.lower() not in stopwords]
        kw_str = " ".join(keywords) if keywords else query
        return (
            f"Technical documentation reference regarding {query}. "
            f"In production systems, {kw_str} is configured with standard protocol parameters, "
            f"error handling mechanisms, verification checks, and state monitoring invariants."
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Execute HyDE retrieval: query -> hypothetical doc -> vector embedding -> nearest neighbors."""
        if not query.strip() or len(self.dense_retriever.chunks) == 0:
            return []

        # 1. Generate hypothetical document
        hypothetical_doc = self.generate_hypothetical_document(query)
        logger.debug(f"Generated HyDE passage: {hypothetical_doc[:80]}...")

        # 2. Embed the hypothetical document
        hypo_vec = self.embedder.embed_query(hypothetical_doc)

        # 3. Optional vector blending with original query embedding
        if self.blend_with_query:
            query_vec = self.embedder.embed_query(query)
            search_vec = (self.query_weight * query_vec) + (self.hypothetical_weight * hypo_vec)
            norm = np.linalg.norm(search_vec)
            if norm > 0:
                search_vec = search_vec / norm
        else:
            search_vec = hypo_vec

        # 4. Search nearest neighbors in dense vector store
        similarities = np.dot(self.dense_retriever.embeddings, search_vec)

        # Apply metadata filters if provided
        valid_indices: List[int] = []
        for idx, chunk in enumerate(self.dense_retriever.chunks):
            if filters:
                match = True
                for k, v in filters.items():
                    chunk_meta = chunk.metadata.model_dump()
                    if chunk_meta.get(k) != v and chunk.metadata.extra.get(k) != v:
                        match = False
                        break
                if match:
                    valid_indices.append(idx)
            else:
                valid_indices.append(idx)

        if not valid_indices:
            return []

        filtered_scores = similarities[valid_indices]
        top_k_indices = np.argsort(filtered_scores)[::-1][:top_k]

        results: List[RetrievalResult] = []
        for rank, local_idx in enumerate(top_k_indices, start=1):
            original_idx = valid_indices[local_idx]
            score = float(similarities[original_idx])
            results.append(
                RetrievalResult(
                    chunk=self.dense_retriever.chunks[original_idx],
                    score=score,
                    rank=rank,
                    retrieval_strategy="hyde"
                )
            )

        return results

