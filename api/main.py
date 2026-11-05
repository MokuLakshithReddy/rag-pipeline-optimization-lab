"""FastAPI backend application for the multi-tenant RAG optimization platform."""

from __future__ import annotations
from contextlib import asynccontextmanager
import logging
import os
import time
import uuid
from typing import List
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd

from api.auth import TenantContext, get_current_tenant
from api.pipeline_state import tenant_manager, pipeline_state
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Restores persisted state across all active tenants on startup."""
    logger.info("Initializing API application lifecycle...")
    restored_tenants = tenant_manager.restore_all_tenants()
    logger.info(f"Auto-restored storage for {restored_tenants} tenant workspace(s).")
    yield
    logger.info("Shutting down API application...")


app = FastAPI(
    title="RAG Pipeline Optimization & Experimentation Lab API",
    version="0.2.0",
    description="Multi-tenant production-grade API for Retrieval-Augmented Generation benchmarking and optimization.",
    lifespan=lifespan
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
async def security_and_tracing_middleware(request: Request, call_next):
    """Audit middleware adding request correlation IDs and security response headers."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
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
    """Unauthenticated system health check and state metrics."""
    default_state = tenant_manager.get_state("default")
    return HealthResponse(
        status="healthy",
        version="0.2.0",
        environment=os.getenv("APP_ENV", "development"),
        total_documents=len(default_state.documents),
        total_indexed_chunks=len(default_state.chunks)
    )


@app.post("/v1/documents", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_documents(
    files: List[UploadFile] = File(...),
    tenant: TenantContext = Depends(get_current_tenant)
):
    """Upload and ingest documents into tenant-isolated knowledge base."""
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")

    state = tenant_manager.get_state(tenant.tenant_id)
    uploaded_names: List[str] = []
    total_new_docs = 0

    for file in files:
        safe_name = sanitize_filename(file.filename or "upload.txt")
        
        # Stream read file in 64KB blocks, aborting if size exceeds ceiling before full memory read
        chunks_data: List[bytes] = []
        bytes_read = 0
        CHUNK_SIZE = 64 * 1024

        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > DEFAULT_MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"File '{safe_name}' exceeds maximum permitted size of {DEFAULT_MAX_FILE_SIZE_BYTES / (1024*1024):.1f}MB."
                )
            chunks_data.append(chunk)

        content = b"".join(chunks_data)

        is_valid, err_msg = validate_file_metadata(safe_name, len(content), max_size_bytes=DEFAULT_MAX_FILE_SIZE_BYTES)
        if not is_valid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"File '{safe_name}': {err_msg}")

        try:
            new_docs = state.ingest_bytes(content, filename=safe_name, persist=True)
            total_new_docs += len(new_docs)
            uploaded_names.append(safe_name)
        except DocumentValidationError as dve:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(dve))

    return DocumentUploadResponse(
        status="success",
        uploaded_files=uploaded_names,
        total_documents_ingested=total_new_docs,
        message=f"[{tenant.tenant_id}] Successfully ingested {total_new_docs} document segment(s) from {len(uploaded_names)} file(s)."
    )


@app.get("/v1/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents(tenant: TenantContext = Depends(get_current_tenant)):
    """List all ingested documents in the authenticated tenant's workspace."""
    state = tenant_manager.get_state(tenant.tenant_id)
    items = [
        DocumentItem(
            id=d.id,
            source=d.metadata.source,
            file_type=d.metadata.file_type,
            character_count=d.metadata.character_count,
            page_number=d.metadata.page_number
        )
        for d in state.documents
    ]
    return DocumentListResponse(total=len(items), documents=items)


@app.post("/v1/index", response_model=IndexResponse, tags=["Index"])
async def build_index(
    req: IndexRequest,
    tenant: TenantContext = Depends(get_current_tenant)
):
    """Trigger chunking, vector/sparse index construction for the tenant."""
    state = tenant_manager.get_state(tenant.tenant_id)
    if not state.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Knowledge base is empty for tenant '{tenant.tenant_id}'. Upload documents before indexing."
        )

    start_time = time.perf_counter()
    chunk_count = state.build_index(
        strategy=req.chunking_strategy,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return IndexResponse(
        status="success",
        chunking_strategy=req.chunking_strategy,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        embedding_model=state.embedding_model_name,
        total_chunks_indexed=chunk_count,
        index_time_ms=elapsed_ms
    )


@app.post("/v1/query", response_model=QueryResponse, tags=["Query"])
async def query_pipeline(
    req: QueryRequest,
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant)
):
    """Query the authenticated tenant's isolated knowledge base."""
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    state = tenant_manager.get_state(tenant.tenant_id)

    if not state.chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Index is empty for tenant '{tenant.tenant_id}'. Ingest documents and trigger /v1/index first."
        )

    response = state.query(
        question=req.query,
        retrieval_strategy=req.retrieval_strategy,
        top_k=req.top_k,
        use_reranker=req.enable_reranking,
        use_query_expansion=req.enable_query_enhancement
    )

    return QueryResponse(
        query=req.query,
        answer=response.answer,
        citations=response.citations,
        retrieved_chunks=response.retrieved_chunks,
        is_abstained=response.is_abstained,
        retrieval_strategy=req.retrieval_strategy,
        latency_ms=response.latency_ms,
        model_name=state.generator.model_name,
        request_id=request_id
    )


@app.get("/v1/experiments", response_model=ExperimentListResponse, tags=["Experiments"])
async def get_experiments(tenant: TenantContext = Depends(get_current_tenant)):
    """Retrieve logged benchmark experiment runs."""
    csv_file = "reports/benchmark_results.csv"
    if not os.path.exists(csv_file):
        return ExperimentListResponse(total=0, experiments=[])

    df = pd.read_csv(csv_file)
    records = df.to_dict(orient="records")
    return ExperimentListResponse(total=len(records), experiments=records)
