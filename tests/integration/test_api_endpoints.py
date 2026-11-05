"""Integration tests for FastAPI REST API endpoints with authentication and multi-tenancy."""

import io
import os
import pytest
from fastapi.testclient import TestClient

# Configure explicit test credentials in environment before app imports
os.environ["API_KEYS"] = "test-key-default:default,test-key-alpha:tenant-alpha,test-key-beta:tenant-beta,test-key-admin:admin"
os.environ["API_KEY"] = "test-key-default"

from api.main import app

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": "test-key-default"}
AUTH_HEADERS_ALPHA = {"X-API-Key": "test-key-alpha"}
AUTH_HEADERS_BETA = {"X-API-Key": "test-key-beta"}
AUTH_HEADERS_ADMIN = {"X-API-Key": "test-key-admin"}


def test_api_health():
    """Health check endpoint is unauthenticated."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_api_authentication_enforcement():
    """Protected endpoints require valid authentication."""
    # 1. Missing API Key
    resp = client.get("/v1/documents")
    assert resp.status_code == 401
    assert "Authentication required" in resp.json()["detail"]

    # 2. Invalid API Key
    resp_invalid = client.get("/v1/documents", headers={"X-API-Key": "invalid-token-9999"})
    assert resp_invalid.status_code == 401
    assert "Invalid API key" in resp_invalid.json()["detail"]

    # 3. Valid API Key succeeds
    resp_valid = client.get("/v1/documents", headers=AUTH_HEADERS)
    assert resp_valid.status_code == 200


def test_cross_tenant_escalation_blocked():
    """Ordinary tenants attempting to switch tenant workspaces via X-Tenant-ID must receive 403 Forbidden."""
    # Tenant Alpha attempts to access Tenant Beta's workspace
    resp = client.get("/v1/documents", headers={"X-API-Key": "test-key-alpha", "X-Tenant-ID": "tenant-beta"})
    assert resp.status_code == 403
    assert "Ordinary tenants are not authorized to switch tenant workspaces" in resp.json()["detail"]


def test_admin_cross_tenant_access_permitted():
    """Administrative keys are explicitly permitted to switch tenant workspaces."""
    resp = client.get("/v1/documents", headers={"X-API-Key": "test-key-admin", "X-Tenant-ID": "tenant-alpha"})
    assert resp.status_code == 200


def test_fail_closed_when_unconfigured(monkeypatch):
    """When no API keys are configured, all protected requests must fail closed with 401."""
    monkeypatch.setenv("API_KEYS", "")
    monkeypatch.delenv("API_KEY", raising=False)
    resp = client.get("/v1/documents", headers={"X-API-Key": "any-token"})
    assert resp.status_code == 401
    assert "No valid API keys are registered" in resp.json()["detail"]


def test_invalid_tenant_id_rejected():
    """Invalid tenant identifiers in X-Tenant-ID must be rejected with 400 Bad Request."""
    invalid_tenants = ["../traversal", "tenant/evil", "space in name", "special@char", ""]
    for inv in invalid_tenants:
        resp = client.get("/v1/documents", headers={"X-API-Key": "test-key-admin", "X-Tenant-ID": inv})
        assert resp.status_code == 400
        assert "Invalid 'X-Tenant-ID' header" in resp.json()["detail"]


def test_api_document_upload_and_listing():
    file_content = b"Knowledge Base Document: Docker containers package software and its dependencies."
    files = [("files", ("docker.txt", io.BytesIO(file_content), "text/plain"))]

    upload_resp = client.post("/v1/documents", files=files, headers=AUTH_HEADERS)
    assert upload_resp.status_code == 200
    data = upload_resp.json()
    assert data["total_documents_ingested"] >= 1
    assert "docker.txt" in data["uploaded_files"]

    # List documents
    list_resp = client.get("/v1/documents", headers=AUTH_HEADERS)
    assert list_resp.status_code == 200
    assert any(doc["source"] == "docker.txt" for doc in list_resp.json()["documents"])


def test_api_indexing_and_querying():
    # Ingest document
    file_content = b"Knowledge Base Document: Docker containers package software and its dependencies."
    files = [("files", ("docker.txt", io.BytesIO(file_content), "text/plain"))]
    client.post("/v1/documents", files=files, headers=AUTH_HEADERS)

    # Ensure index request
    idx_resp = client.post(
        "/v1/index",
        json={
            "chunking_strategy": "recursive",
            "chunk_size": 256,
            "chunk_overlap": 32,
            "embedding_model": "all-MiniLM-L6-v2"
        },
        headers=AUTH_HEADERS
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
        },
        headers=AUTH_HEADERS
    )
    assert q_resp.status_code == 200
    data = q_resp.json()
    assert data["query"] == "What do Docker containers package?"
    assert any(term in data["answer"].lower() for term in ["docker", "software", "dependencies"])
    assert len(data["retrieved_chunks"]) > 0
    assert data["is_abstained"] is False
    assert q_resp.headers["X-Request-ID"] == data["request_id"]


def test_multi_tenant_isolation():
    """Verify separate tenants have completely independent, non-leaking document stores and indices."""
    # 1. Tenant Alpha uploads document on Quantum Computing
    content_a = b"Quantum annealing uses quantum fluctuations to find the global minimum of an objective function."
    files_a = [("files", ("quantum.txt", io.BytesIO(content_a), "text/plain"))]
    resp_a = client.post("/v1/documents", files=files_a, headers=AUTH_HEADERS_ALPHA)
    assert resp_a.status_code == 200

    # 2. Tenant Beta uploads document on Aeronautical Propulsion
    content_b = b"Turbofan engines utilize bypass air around the core to achieve high propulsive efficiency."
    files_b = [("files", ("propulsion.txt", io.BytesIO(content_b), "text/plain"))]
    resp_b = client.post("/v1/documents", files=files_b, headers=AUTH_HEADERS_BETA)
    assert resp_b.status_code == 200

    # 3. Assert document lists are isolated
    docs_a = client.get("/v1/documents", headers=AUTH_HEADERS_ALPHA).json()["documents"]
    docs_b = client.get("/v1/documents", headers=AUTH_HEADERS_BETA).json()["documents"]

    assert any(d["source"] == "quantum.txt" for d in docs_a)
    assert not any(d["source"] == "propulsion.txt" for d in docs_a)

    assert any(d["source"] == "propulsion.txt" for d in docs_b)
    assert not any(d["source"] == "quantum.txt" for d in docs_b)

    # 4. Index both tenants independently
    client.post("/v1/index", json={"chunking_strategy": "recursive", "chunk_size": 256, "chunk_overlap": 32}, headers=AUTH_HEADERS_ALPHA)
    client.post("/v1/index", json={"chunking_strategy": "recursive", "chunk_size": 256, "chunk_overlap": 32}, headers=AUTH_HEADERS_BETA)

    # 5. Query Tenant Alpha for propulsion -> should abstain or find zero propulsion chunks
    q_a = client.post("/v1/query", json={"query": "How do turbofan engines work?", "retrieval_strategy": "hybrid"}, headers=AUTH_HEADERS_ALPHA).json()
    assert not any("turbofan" in c["chunk"]["text"].lower() for c in q_a["retrieved_chunks"])

    # 6. Query Tenant Beta for propulsion -> should find turbofan chunk
    q_b = client.post("/v1/query", json={"query": "How do turbofan engines work?", "retrieval_strategy": "hybrid"}, headers=AUTH_HEADERS_BETA).json()
    assert any("turbofan" in c["chunk"]["text"].lower() for c in q_b["retrieved_chunks"])


def test_api_experiments_endpoint():
    resp = client.get("/v1/experiments", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "experiments" in data
