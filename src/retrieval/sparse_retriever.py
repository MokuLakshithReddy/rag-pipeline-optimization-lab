"""Sparse keyword retriever powered by BM25."""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi

from src.ingestion.models import DocumentChunk, RetrievalResult
from src.retrieval.base import BaseRetriever


class SparseRetriever(BaseRetriever):
    """Sparse retriever implementing Okapi BM25 ranking for exact keyword matches."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> List[str]:
        """Case-fold and extract alphanumeric word tokens."""
        return re.findall(r'\b\w+\b', text.lower())

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return

        self.chunks.extend(chunks)
        new_tokens = [self._tokenize(c.text) for c in chunks]
        self._tokenized_corpus.extend(new_tokens)
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        if not query.strip() or self._bm25 is None or len(self.chunks) == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Apply metadata filters
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

        # Sort valid indices by score descending
        sorted_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)
        effective_k = min(top_k, len(sorted_indices))
        top_indices = sorted_indices[:effective_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=score,
                    rank=rank,
                    retriever_type="sparse"
                )
            )

        return results

    def clear(self) -> None:
        self.chunks = []
        self._tokenized_corpus = []
        self._bm25 = None

    def save(self, directory: str | Path) -> None:
        """Persist BM25 index corpus and metadata to disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        chunks_data = [c.model_dump() for c in self.chunks]
        payload = {
            "params": {"k1": self.k1, "b": self.b},
            "tokenized_corpus": self._tokenized_corpus,
            "chunks": chunks_data
        }
        with open(target_dir / "sparse_index.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def load(self, directory: str | Path) -> None:
        """Load BM25 index from disk."""
        target_dir = Path(directory)
        index_file = target_dir / "sparse_index.json"
        if not index_file.exists():
            raise FileNotFoundError(f"Sparse index file not found in {directory}")

        with open(index_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.k1 = payload["params"]["k1"]
        self.b = payload["params"]["b"]
        self._tokenized_corpus = payload["tokenized_corpus"]
        self.chunks = [DocumentChunk(**c) for c in payload["chunks"]]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)

