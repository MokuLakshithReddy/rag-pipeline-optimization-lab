"""Dense vector retriever supporting cosine similarity, metadata filtering, and index persistence."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.embeddings.base import BaseEmbedder
from src.ingestion.models import DocumentChunk, DocumentMetadata, RetrievalResult
from src.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense vector retriever with vector normalization and cosine similarity search."""

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = np.empty((0, self.embedder.dimension), dtype=np.float32)

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return

        texts = [c.text for c in chunks]
        new_vecs = self.embedder.embed_texts(texts)

        if new_vecs.shape[1] != self.embedder.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedder.dimension}, got {new_vecs.shape[1]}"
            )

        self.chunks.extend(chunks)
        if self.embeddings.shape[0] == 0:
            self.embeddings = new_vecs
        else:
            self.embeddings = np.vstack([self.embeddings, new_vecs])

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        if not query.strip() or len(self.chunks) == 0:
            return []

        query_vec = self.embedder.embed_query(query)
        # Cosine similarity for normalized vectors is the dot product
        similarities = np.dot(self.embeddings, query_vec)

        # Apply metadata filters if provided
        valid_indices: List[int] = []
        for idx, chunk in enumerate(self.chunks):
            if filters:
                match = True
                for k, v in filters.items():
                    chunk_meta = chunk.metadata.model_dump()
                    if chunk_meta.get(k) != v and chunk.metadata.extra.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            valid_indices.append(idx)

        if not valid_indices:
            return []

        sub_sims = similarities[valid_indices]
        effective_k = min(top_k, len(valid_indices))
        # Top-K indices via argpartition
        top_sub_indices = np.argsort(sub_sims)[::-1][:effective_k]

        results: List[RetrievalResult] = []
        for rank, sub_idx in enumerate(top_sub_indices, start=1):
            original_idx = valid_indices[sub_idx]
            score = float(sub_sims[sub_idx])
            results.append(
                RetrievalResult(
                    chunk=self.chunks[original_idx],
                    score=score,
                    rank=rank,
                    retriever_type="dense"
                )
            )

        return results

    def clear(self) -> None:
        self.chunks = []
        self.embeddings = np.empty((0, self.embedder.dimension), dtype=np.float32)

    def save(self, directory: str | Path) -> None:
        """Persist index vectors and chunk metadata to disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        np.save(target_dir / "dense_embeddings.npy", self.embeddings)

        # Save metadata and chunks
        chunks_data = [c.model_dump() for c in self.chunks]
        meta = {
            "model_name": self.embedder.model_name,
            "dimension": self.embedder.dimension,
            "chunk_count": len(self.chunks)
        }
        with open(target_dir / "dense_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "chunks": chunks_data}, f, indent=2)

    def load(self, directory: str | Path) -> None:
        """Load persisted index vectors and chunk metadata from disk."""
        target_dir = Path(directory)
        if not (target_dir / "dense_embeddings.npy").exists():
            raise FileNotFoundError(f"Index not found in {directory}")

        self.embeddings = np.load(target_dir / "dense_embeddings.npy")
        with open(target_dir / "dense_metadata.json", "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.chunks = [DocumentChunk(**item) for item in payload["chunks"]]

