"""Abstract base class for embedding models."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class BaseEmbedder(ABC):
    """Abstract interface for text embedding generation."""

    def __init__(
        self,
        model_name: str,
        dimension: int,
        normalize: bool = True,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.normalize = normalize
        self.batch_size = batch_size

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate dense vector embeddings for a list of text strings."""
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """Generate dense vector embedding for a single search query."""
        results = self.embed_texts([query])
        return results[0]

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Perform L2 vector normalization for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms

