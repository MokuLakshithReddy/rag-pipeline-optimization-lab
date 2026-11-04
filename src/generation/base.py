"""Abstract base interface for RAG generation."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from src.ingestion.models import GenerationResponse, GroundedCitation, RetrievalResult


class BaseGenerator(ABC):
    """Abstract interface for LLM answer generation."""

    def __init__(
        self,
        provider: str = "mock",
        temperature: float = 0.0,
        abstention_threshold: float = 0.15,
        abstention_message: str = "I do not have sufficient information in the provided context to answer this question."
    ):
        self.provider = provider
        self.temperature = temperature
        self.abstention_threshold = abstention_threshold
        self.abstention_message = abstention_message

    @abstractmethod
    def generate(
        self,
        query: str,
        context: str,
        citations: List[GroundedCitation],
        retrieved_chunks: List[RetrievalResult],
        retrieval_strategy: str = "dense"
    ) -> GenerationResponse:
        """Generate a grounded response or abstain if context is insufficient."""
        pass

