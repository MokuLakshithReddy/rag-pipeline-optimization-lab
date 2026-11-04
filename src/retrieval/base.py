"""Abstract base interface for retrievers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from src.ingestion.models import DocumentChunk, RetrievalResult


class BaseRetriever(ABC):
    """Abstract interface for all retrieval implementations."""

    @abstractmethod
    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Index a collection of DocumentChunks."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve top_k chunks matching the query."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Reset and clear the index."""
        pass

