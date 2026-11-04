"""Fast deterministic pseudo-semantic embedder for tests, CI, and lightweight environments."""

from __future__ import annotations
import hashlib
import re
from typing import List
import numpy as np

from src.embeddings.base import BaseEmbedder


class FastDeterministicEmbedder(BaseEmbedder):
    """Generates deterministic n-gram hash vector embeddings without requiring network access."""

    def __init__(
        self,
        model_name: str = "fast-hash-64",
        dimension: int = 64,
        normalize: bool = True,
        batch_size: int = 64
    ):
        super().__init__(
            model_name=model_name,
            dimension=dimension,
            normalize=normalize,
            batch_size=batch_size
        )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for row_idx, text in enumerate(texts):
            words = re.findall(r'\w+', text.lower())
            if not words:
                continue

            # Project words into dimension buckets using md5 hashing
            for word in words:
                h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
                bucket = h % self.dimension
                sign = 1.0 if ((h >> 4) & 1) == 0 else -1.0
                matrix[row_idx, bucket] += sign

            # Character bigrams to capture subword information
            for i in range(len(text) - 1):
                bg = text[i:i+2].lower()
                h_bg = int(hashlib.md5(bg.encode("utf-8")).hexdigest(), 16)
                bucket_bg = h_bg % self.dimension
                matrix[row_idx, bucket_bg] += 0.2

        if self.normalize:
            matrix = self._normalize_vectors(matrix)

        return matrix.astype(np.float32)

