"""Integration tests for state persistence across restarts, streaming upload limits, and request IDs."""

import io
from pathlib import Path
from fastapi.testclient import TestClient
from api.auth import DEFAULT_DEV_API_KEY
from api.main import app
from api.pipeline_state import PipelineState
from src.generation.llm_generator import LLMGenerator

client = TestClient(app)
AUTH_HEADERS = {"X-API-Key": DEFAULT_DEV_API_KEY}


def test_persistent_state_recovery_across_restarts(tmp_path):
    upload_dir = tmp_path / "uploads"
    index_dir = tmp_path / "indices"

    # 1. First Lifecycle: Ingest and Index
    state_v1 = PipelineState(
        upload_dir=str(upload_dir),
        index_dir=str(index_dir),
        auto_restore=False
    )
    doc_bytes = b"PostgreSQL uses Write-Ahead Logging to guarantee ACID durability across crashes."
    state_v1.ingest_bytes(doc_bytes, filename="postgres_wal.txt", persist=True)
    assert len(state_v1.documents) == 1

    chunks_count = state_v1.build_index(strategy="recursive", chunk_size=256, chunk_overlap=32)
    assert chunks_count >= 1

    # Verify files were persisted to disk
    assert (upload_dir / "postgres_wal.txt").exists()
    assert (upload_dir / "documents_manifest.json").exists()
    assert (index_dir / "dense_embeddings.npy").exists()
    assert (index_dir / "dense_metadata.json").exists()

    # 2. Second Lifecycle: Brand new instance simulating app restart
    state_v2 = PipelineState(
        upload_dir=str(upload_dir),
        index_dir=str(index_dir),
        auto_restore=True
    )
    # Documents and chunks must be restored automatically
    assert len(state_v2.documents) == 1
    assert len(state_v2.chunks) == chunks_count

    # Query against restored indices
    res = state_v2.query("Write-Ahead Logging durability", retrieval_strategy="hybrid", top_k=1)
    assert res.is_abstained is False
    assert any(w in res.answer for w in ["PostgreSQL", "ACID", "durability", "Write-Ahead", "Logging"])


def test_streaming_upload_rejects_oversized_payload():
    # 26 MB of dummy data
    large_data = b"0" * (26 * 1024 * 1024)
    files = [("files", ("huge_file.txt", io.BytesIO(large_data), "text/plain"))]

    response = client.post("/v1/documents", files=files, headers=AUTH_HEADERS)
    assert response.status_code == 413
    assert "exceeds maximum permitted size" in response.json()["detail"]


def test_request_id_consistency():
    custom_id = "test-request-id-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id


def test_generator_provider_configuration():
    gen_mock = LLMGenerator(provider="mock")
    assert gen_mock.provider == "mock"

    gen_gemini = LLMGenerator(provider="gemini", model_name="gemini-1.5-flash")
    assert gen_gemini.provider == "gemini"
    assert gen_gemini.model_name == "gemini-1.5-flash"

    gen_local = LLMGenerator(provider="local")
    assert gen_local.provider in {"local", "mock"}

