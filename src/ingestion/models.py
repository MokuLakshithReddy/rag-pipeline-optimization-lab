"""Core data models for the RAG optimization pipeline."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with an ingested document."""
    source: str
    doc_id: str
    file_type: str
    page_number: Optional[int] = None
    character_count: int = 0
    token_estimate: int = 0
    hash: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """An ingested textual document."""
    id: str
    text: str
    metadata: DocumentMetadata


class DocumentChunk(BaseModel):
    """A chunked segment of an ingested document."""
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: DocumentMetadata


class RetrievalResult(BaseModel):
    """Result of a retrieval or reranking operation."""
    chunk: DocumentChunk
    score: float
    rank: int
    retriever_type: str = "dense"


class GroundedCitation(BaseModel):
    """Citation reference pointing to source document and span."""
    source: str
    chunk_id: str
    doc_id: str
    page_number: Optional[int] = None
    snippet: str


class GenerationResponse(BaseModel):
    """End-to-end RAG generation response."""
    query: str
    answer: str
    citations: List[GroundedCitation] = Field(default_factory=list)
    retrieved_chunks: List[RetrievalResult] = Field(default_factory=list)
    is_abstained: bool = False
    retrieval_strategy: str = "dense"
    latency_ms: float = 0.0
    model_name: Optional[str] = None

