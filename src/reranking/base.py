"""Abstract base interface for rerankers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from src.ingestion.models import RetrievalResult


class BaseReranker(ABC):
    """Abstract interface for reranking candidate retrieval results."""

    def __init__(self, model_name: str, top_n: int = 3):
        self.model_name = model_name
        self.top_n = top_n

    @abstractmethod
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank candidate results given the user query."""
        pass

