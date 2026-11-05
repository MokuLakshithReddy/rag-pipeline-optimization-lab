"""Context optimization: deduplication, thresholding, ordering, and token budget management."""

from __future__ import annotations
import hashlib
from typing import List, Tuple
from src.ingestion.models import GroundedCitation, RetrievalResult


class ContextOptimizer:
    """Optimizes retrieved chunks into a clean, source-grounded context window."""

    def __init__(
        self,
        max_context_chars: int = 4000,
        similarity_threshold: float = 0.0,
        enable_deduplication: bool = True,
        lost_in_the_middle_ordering: bool = True
    ):
        self.max_context_chars = max_context_chars
        self.similarity_threshold = similarity_threshold
        self.enable_deduplication = enable_deduplication
        self.lost_in_the_middle_ordering = lost_in_the_middle_ordering

    def optimize_context(
        self,
        results: List[RetrievalResult]
    ) -> Tuple[str, List[GroundedCitation], List[RetrievalResult]]:
        """Filters, deduplicates, orders, and formats retrieved chunks within token budget."""
        if not results:
            return "", [], []

        # 1. Filter by minimum similarity threshold (sparse BM25 scores can be negative on small corpora)
        if self.similarity_threshold > 0.0:
            filtered = [r for r in results if r.retriever_type in {"sparse", "hybrid"} or r.score >= self.similarity_threshold]
        else:
            filtered = results

        # 2. Deduplicate by exact text hash
        deduped: List[RetrievalResult] = []
        seen_hashes = set()
        for r in filtered:
            h = hashlib.md5(r.chunk.text.strip().lower().encode("utf-8")).hexdigest()
            if self.enable_deduplication and h in seen_hashes:
                continue
            seen_hashes.add(h)
            deduped.append(r)

        if not deduped:
            return "", [], []

        # 3. Lost-in-the-middle context ordering:
        # Place most relevant chunks at the very beginning and very end of context.
        if self.lost_in_the_middle_ordering and len(deduped) > 2:
            ordered = self._reorder_lost_in_the_middle(deduped)
        else:
            ordered = deduped

        # 4. Token/character budget truncation and context string formatting
        formatted_blocks: List[str] = []
        citations: List[GroundedCitation] = []
        accepted_results: List[RetrievalResult] = []
        current_chars = 0

        for idx, item in enumerate(ordered, start=1):
            chunk = item.chunk
            page_info = f", Page {chunk.metadata.page_number}" if chunk.metadata.page_number else ""
            header = f"[Source {idx}: {chunk.metadata.source}{page_info} | Chunk ID: {chunk.chunk_id}]"
            block = f"{header}\n{chunk.text.strip()}\n"

            if current_chars + len(block) > self.max_context_chars and accepted_results:
                break

            formatted_blocks.append(block)
            current_chars += len(block)
            accepted_results.append(item)

            snippet = chunk.text.strip()[:160] + "..." if len(chunk.text.strip()) > 160 else chunk.text.strip()
            citations.append(
                GroundedCitation(
                    source=chunk.metadata.source,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    page_number=chunk.metadata.page_number,
                    snippet=snippet
                )
            )

        context_text = "\n---\n".join(formatted_blocks)
        return context_text, citations, accepted_results

    def _reorder_lost_in_the_middle(self, items: List[RetrievalResult]) -> List[RetrievalResult]:
        """Distribute top-ranked items between beginning and end."""
        reordered: List[RetrievalResult] = []
        left = True
        for item in items:
            if left:
                reordered.insert(0, item)
            else:
                reordered.append(item)
            left = not left
        return reordered

