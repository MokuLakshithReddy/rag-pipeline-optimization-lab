"""Abstract base class for document chunkers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from src.ingestion.models import Document, DocumentChunk


class BaseChunker(ABC):
    """Abstract interface for all chunking strategies."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a single Document into a sequence of DocumentChunk instances."""
        pass

    def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """Split a batch of Documents into DocumentChunk instances."""
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

