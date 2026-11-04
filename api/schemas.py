"""Pydantic request and response validation schemas for the REST API."""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from src.ingestion.models import GenerationResponse, GroundedCitation, RetrievalResult


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    environment: str = "development"
    total_documents: int = 0
    total_indexed_chunks: int = 0


class DocumentItem(BaseModel):
    id: str
    source: str
    file_type: str
    character_count: int
    page_number: Optional[int] = None


class DocumentUploadResponse(BaseModel):
    status: str = "success"
    uploaded_files: List[str]
    total_documents_ingested: int
    message: str


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentItem]


class IndexRequest(BaseModel):
    chunking_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0, le=1024)
    embedding_model: str = "all-MiniLM-L6-v2"


class IndexResponse(BaseModel):
    status: str = "success"
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    total_chunks_indexed: int
    index_time_ms: float


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    retrieval_strategy: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=50)
    enable_reranking: bool = True
    enable_query_enhancement: bool = False
    filters: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[GroundedCitation]
    retrieved_chunks: List[RetrievalResult]
    is_abstained: bool
    retrieval_strategy: str
    latency_ms: float
    model_name: Optional[str] = None
    request_id: str


class ExperimentRunItem(BaseModel):
    experiment_id: str
    timestamp: str
    retrieval_strategy: str
    chunking_strategy: str
    mrr: float
    ndcg_at_k: float
    hit_at_k: float
    latency_p95_ms: float


class ExperimentListResponse(BaseModel):
    total: int
    experiments: List[Dict[str, Any]]

