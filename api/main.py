"""FastAPI backend application for the RAG optimization platform."""

from __future__ import annotations
import logging
import os
import time
import uuid
from typing import List
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd

from api.pipeline_state import pipeline_state
from api.schemas import (
    DocumentItem,
    DocumentListResponse,
    DocumentUploadResponse,
    ExperimentListResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
)
from src.ingestion.validation import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    DocumentValidationError,
    sanitize_filename,
    validate_file_metadata,
)

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api")

app = FastAPI(
    title="RAG Pipeline Optimization & Experimentation Lab API",
    version="0.1.0",
    description="Production-grade API for Retrieval-Augmented Generation benchmarking and optimization."
)

# Security: Restrictive CORS configuration
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_and_timing_headers(request: Request, call_next):
    """Enforce request IDs, measure latency, and add security headers."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(f"Unhandled exception for request {request_id}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "An internal server error occurred.", "request_id": request_id}
        )

    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check and state metrics."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        environment=os.getenv("APP_ENV", "development"),
        total_documents=len(pipeline_state.documents),
        total_indexed_chunks=len(pipeline_state.chunks)
    )


@app.post("/v1/documents", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and ingest one or more documents safely."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    uploaded_names: List[str] = []
    total_new_docs = 0

    for file in files:
        safe_name = sanitize_filename(file.filename or "upload.txt")
        content = await file.read()

        is_valid, err_msg = validate_file_metadata(safe_name, len(content), max_size_bytes=DEFAULT_MAX_FILE_SIZE_BYTES)
        if not is_valid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"File '{safe_name}': {err_msg}")

        try:
            new_docs = pipeline_state.ingest_bytes(content, filename=safe_name)
            total_new_docs += len(new_docs)
            uploaded_names.append(safe_name)
        except DocumentValidationError as dve:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(dve))

    return DocumentUploadResponse(
        status="success",
        uploaded_files=uploaded_names,
        total_documents_ingested=total_new_docs,
        message=f"Successfully ingested {total_new_docs} document segment(s) from {len(uploaded_names)} file(s)."
    )


@app.get("/v1/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """List all ingested documents and their metadata."""
    items = [
        DocumentItem(
            id=d.id,
            source=d.metadata.source,
            file_type=d.metadata.file_type,
            character_count=d.metadata.character_count,
            page_number=d.metadata.page_number
        )
        for d in pipeline_state.documents
    ]
    return DocumentListResponse(total=len(items), documents=items)


@app.post("/v1/index", response_model=IndexResponse, tags=["Index"])
async def build_index(req: IndexRequest):
    """Trigger chunking and vector/sparse index construction."""
    if not pipeline_state.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Knowledge base is empty. Upload documents before indexing."
        )

    start_time = time.perf_counter()
    chunk_count = pipeline_state.build_index(
        strategy=req.chunking_strategy,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        embedding_model=req.embedding_model
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return IndexResponse(
        status="success",
        chunking_strategy=req.chunking_strategy,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        embedding_model=req.embedding_model,
        total_chunks_indexed=chunk_count,
        index_time_ms=elapsed_ms
    )


@app.post("/v1/query", response_model=QueryResponse, tags=["Query"])
async def query_pipeline(req: QueryRequest, request: Request):
    """Query the RAG pipeline with selected retrieval and reranking configuration."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    if not pipeline_state.chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Index is empty. Ingest documents and trigger /v1/index first."
        )

    response = pipeline_state.query(
        raw_query=req.query,
        retrieval_strategy=req.retrieval_strategy,
        top_k=req.top_k,
        enable_reranking=req.enable_reranking,
        enable_query_enhancement=req.enable_query_enhancement,
        filters=req.filters
    )

    return QueryResponse(
        query=response.query,
        answer=response.answer,
        citations=response.citations,
        retrieved_chunks=response.retrieved_chunks,
        is_abstained=response.is_abstained,
        retrieval_strategy=response.retrieval_strategy,
        latency_ms=response.latency_ms,
        model_name=response.model_name,
        request_id=request_id
    )


@app.get("/v1/experiments", response_model=ExperimentListResponse, tags=["Experiments"])
async def get_experiments():
    """Retrieve logged benchmark experiment runs."""
    csv_file = "reports/benchmark_results.csv"
    if not os.path.exists(csv_file):
        return ExperimentListResponse(total=0, experiments=[])

    df = pd.read_csv(csv_file)
    records = df.to_dict(orient="records")
    return ExperimentListResponse(total=len(records), experiments=records)

