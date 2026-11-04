"""Factory registry for instantiating embedding models."""

from __future__ import annotations
from typing import Any, Optional
from src.embeddings.base import BaseEmbedder
from src.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.embeddings.fast_embedder import FastDeterministicEmbedder


def get_embedder(
    model_name: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 32,
    device: str = "cpu",
    enable_cache: bool = True,
    **kwargs: Any
) -> BaseEmbedder:
    """Retrieve configured embedder instance."""
    clean_name = model_name.strip()
    if clean_name.startswith("fast-") or clean_name == "mock":
        dim = kwargs.get("dimension", 64)
        return FastDeterministicEmbedder(
            model_name=clean_name,
            dimension=dim,
            normalize=normalize,
            batch_size=batch_size
        )
    else:
        return SentenceTransformerEmbedder(
            model_name=clean_name,
            normalize=normalize,
            batch_size=batch_size,
            device=device,
            enable_cache=enable_cache
        )

