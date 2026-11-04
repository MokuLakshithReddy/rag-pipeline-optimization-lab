"""Semantic chunker partitioning text at semantic shift boundaries using vector embeddings."""

from __future__ import annotations
import re
from typing import Any, Callable, List, Optional
import numpy as np

from src.chunking.base import BaseChunker
from src.ingestion.models import Document, DocumentChunk


class SemanticChunker(BaseChunker):
    """Splits text into sentences and clusters them based on semantic vector similarity shifts."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        similarity_threshold: float = 0.65,
        embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
        embedding_model_name: Optional[str] = "all-MiniLM-L6-v2"
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self._embed_fn = embed_fn
        self.embedding_model_name = embedding_model_name
        self._lazy_embedder = None

    def _get_embed_fn(self) -> Callable[[List[str]], np.ndarray]:
        """Obtain embedding callable, lazy-instantiating default sentence embedder if needed."""
        if self._embed_fn is not None:
            return self._embed_fn

        if self._lazy_embedder is None:
            try:
                from src.embeddings.registry import get_embedder
                embedder_instance = get_embedder(model_name=self.embedding_model_name or "all-MiniLM-L6-v2")
                self._lazy_embedder = embedder_instance.embed_texts
            except Exception:
                # Fallback to simple fast deterministic embedder
                from src.embeddings.fast_embedder import FastDeterministicEmbedder
                self._lazy_embedder = FastDeterministicEmbedder().embed_texts

        return self._lazy_embedder

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        text = document.text
        if not text:
            return []

        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [
                DocumentChunk(
                    chunk_id=f"{document.id}_sc0",
                    doc_id=document.id,
                    text=text,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata=document.metadata
                )
            ]

        breakpoints = self._find_breakpoints(sentences)

        chunks: List[DocumentChunk] = []
        current_sentences: List[str] = []
        current_len = 0
        chunk_idx = 0
        search_start = 0

        for i, sentence in enumerate(sentences):
            current_sentences.append(sentence)
            current_len += len(sentence) + 1

            is_breakpoint = (i in breakpoints)
            is_max_size = (current_len >= self.chunk_size)

            if is_breakpoint or is_max_size or (i == len(sentences) - 1):
                chunk_text = " ".join(current_sentences).strip()
                if chunk_text:
                    found_idx = text.find(chunk_text, search_start)
                    start_char = found_idx if found_idx != -1 else search_start
                    end_char = start_char + len(chunk_text)
                    search_start = max(search_start, start_char + 1)

                    chunk = DocumentChunk(
                        chunk_id=f"{document.id}_sc{chunk_idx}",
                        doc_id=document.id,
                        text=chunk_text,
                        chunk_index=chunk_idx,
                        start_char=start_char,
                        end_char=end_char,
                        metadata=document.metadata
                    )
                    chunks.append(chunk)
                    chunk_idx += 1

                current_sentences = []
                current_len = 0

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        raw = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw if s.strip()]

    def _find_breakpoints(self, sentences: List[str]) -> set:
        if len(sentences) <= 1:
            return set()

        embed_callable = self._get_embed_fn()
        try:
            embeddings = embed_callable(sentences)
            breakpoints = set()
            for i in range(len(embeddings) - 1):
                v1 = embeddings[i]
                v2 = embeddings[i + 1]
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                sim = (np.dot(v1, v2) / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 1.0
                if sim < self.similarity_threshold:
                    breakpoints.add(i)
            return breakpoints
        except Exception:
            pass

        # Lexical Jaccard fallback only if vector embedding fails
        breakpoints = set()
        for i in range(len(sentences) - 1):
            s1_tokens = set(re.findall(r'\b\w+\b', sentences[i].lower()))
            s2_tokens = set(re.findall(r'\b\w+\b', sentences[i + 1].lower()))
            if not s1_tokens or not s2_tokens:
                continue
            intersection = len(s1_tokens & s2_tokens)
            union = len(s1_tokens | s2_tokens)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard < (1.0 - self.similarity_threshold):
                breakpoints.add(i)

        return breakpoints
