"""Integration tests for FastAPI REST API endpoints."""

import io
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_api_document_upload_and_listing():
    file_content = b"Knowledge Base Document: Docker containers package software and its dependencies."
    files = [("files", ("docker.txt", io.BytesIO(file_content), "text/plain"))]

    upload_resp = client.post("/v1/documents", files=files)
    assert upload_resp.status_code == 200
    assert upload_resp.json()["status"] == "success"

    list_resp = client.get("/v1/documents")
    assert list_resp.status_code == 200
    docs = list_resp.json()["documents"]
    assert any(d["source"] == "docker.txt" for d in docs)


def test_api_indexing_and_querying():
    # Ensure index request
    idx_resp = client.post(
        "/v1/index",
        json={
            "chunking_strategy": "recursive",
            "chunk_size": 256,
            "chunk_overlap": 32,
            "embedding_model": "all-MiniLM-L6-v2"
        }
    )
    assert idx_resp.status_code == 200
    assert idx_resp.json()["total_chunks_indexed"] > 0

    # Query
    q_resp = client.post(
        "/v1/query",
        json={
            "query": "What do Docker containers package?",
            "retrieval_strategy": "hybrid",
            "top_k": 3,
            "enable_reranking": True
        }
    )
    assert q_resp.status_code == 200
    data = q_resp.json()
    assert data["query"] == "What do Docker containers package?"
    assert "Docker" in data["answer"]
    assert len(data["retrieved_chunks"]) > 0
    assert data["is_abstained"] is False
    assert q_resp.headers["X-Request-ID"] == data["request_id"]


def test_api_experiments_endpoint():
    resp = client.get("/v1/experiments")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "experiments" in data

