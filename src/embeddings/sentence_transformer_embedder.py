"""SentenceTransformer implementation with batching, caching, and L2 normalization."""

from __future__ import annotations
import hashlib
from typing import Dict, List, Optional
import numpy as np

from src.embeddings.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder powered by HuggingFace sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
        batch_size: int = 32,
        device: str = "cpu",
        enable_cache: bool = True
    ):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name, device=device)
        if hasattr(self._model, "get_embedding_dimension"):
            dimension = self._model.get_embedding_dimension() or 384
        else:
            dimension = self._model.get_sentence_embedding_dimension() or 384
        super().__init__(
            model_name=model_name,
            dimension=dimension,
            normalize=normalize,
            batch_size=batch_size
        )
        self.device = device
        self.enable_cache = enable_cache
        self._cache: Dict[str, np.ndarray] = {}

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        indices_to_compute: List[int] = []
        texts_to_compute: List[str] = []
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)

        for i, text in enumerate(texts):
            h = self._hash_text(text)
            if self.enable_cache and h in self._cache:
                result[i] = self._cache[h]
            else:
                indices_to_compute.append(i)
                texts_to_compute.append(text)

        if texts_to_compute:
            computed = self._model.encode(
                texts_to_compute,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            ).astype(np.float32)

            for idx, comp_vec, text in zip(indices_to_compute, computed, texts_to_compute):
                result[idx] = comp_vec
                if self.enable_cache:
                    self._cache[self._hash_text(text)] = comp_vec

        return result
