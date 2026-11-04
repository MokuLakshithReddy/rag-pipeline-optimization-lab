"""Factory registry for instantiating document chunkers."""

from __future__ import annotations
from typing import Any, Dict, Optional
from src.chunking.base import BaseChunker
from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.recursive_chunker import RecursiveCharacterChunker
from src.chunking.semantic_chunker import SemanticChunker


def get_chunker(
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    **kwargs: Any
) -> BaseChunker:
    """Retrieve chunker instance configured by strategy name."""
    strategy_clean = strategy.lower().strip()
    if strategy_clean in {"fixed", "fixed_size"}:
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy_clean in {"recursive", "recursive_character"}:
        separators = kwargs.get("separators")
        return RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    elif strategy_clean in {"semantic"}:
        similarity_threshold = kwargs.get("similarity_threshold", 0.65)
        embed_fn = kwargs.get("embed_fn")
        return SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            similarity_threshold=similarity_threshold,
            embed_fn=embed_fn
        )
    else:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. Supported: 'fixed', 'recursive', 'semantic'"
        )

