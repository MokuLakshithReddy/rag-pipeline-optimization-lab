"""Unit tests for document ingestion and security validation."""

import pytest
from pathlib import Path
from src.ingestion.document_loader import UniversalDocumentLoader
from src.ingestion.validation import (
    DocumentValidationError,
    compute_content_hash,
    detect_and_decode_text,
    sanitize_filename,
    validate_file_metadata,
    validate_file_path,
)


def test_sanitize_filename():
    assert sanitize_filename("test.txt") == "test.txt"
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("my report (v2) [final].pdf") == "my_report__v2___final_.pdf"
    assert sanitize_filename("...") == "unnamed_document.txt"


def test_validate_file_metadata():
    valid, msg = validate_file_metadata("doc.pdf", 1024)
    assert valid is True

    valid, msg = validate_file_metadata("malicious.exe", 1024)
    assert valid is False
    assert "Unsupported file extension" in msg

    valid, msg = validate_file_metadata("empty.txt", 0)
    assert valid is False
    assert "empty" in msg

    valid, msg = validate_file_metadata("huge.txt", 30 * 1024 * 1024, max_size_bytes=25 * 1024 * 1024)
    assert valid is False
    assert "exceeds limit" in msg


def test_validate_file_path_traversal(tmp_path):
    safe_base = tmp_path / "uploads"
    safe_base.mkdir()
    safe_file = safe_base / "file.txt"
    safe_file.write_text("safe")

    # Inside directory should succeed
    resolved = validate_file_path(safe_file, safe_base)
    assert resolved == safe_file.resolve()

    # Traversal outside directory should raise
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("danger")
    with pytest.raises(DocumentValidationError):
        validate_file_path(outside_file, safe_base)


def test_detect_and_decode_text():
    text = "Hello world! This is a test."
    decoded = detect_and_decode_text(text.encode("utf-8"))
    assert decoded == text

    # Binary null bytes should raise
    with pytest.raises(DocumentValidationError):
        detect_and_decode_text(b"\x00\x01\x02\x03nottext")


def test_universal_loader_text_and_deduplication():
    loader = UniversalDocumentLoader(deduplicate=True)
    content = b"Document content for deduplication test."
    docs1 = loader.load_from_bytes(content, filename="test.txt")
    assert len(docs1) == 1
    assert docs1[0].text == "Document content for deduplication test."
    assert docs1[0].metadata.source == "test.txt"

    # Second load with same content should be skipped by deduplication
    docs2 = loader.load_from_bytes(content, filename="test_duplicate.txt")
    assert len(docs2) == 0


def test_universal_loader_json():
    loader = UniversalDocumentLoader(deduplicate=False)
    json_bytes = b'[{"id": "rec1", "text": "Record 1 details"}, {"id": "rec2", "text": "Record 2 details"}]'
    docs = loader.load_from_bytes(json_bytes, filename="data.json")
    assert len(docs) == 2
    assert docs[0].id == "rec1"
    assert docs[0].text == "Record 1 details"
    assert docs[1].id == "rec2"


def test_universal_loader_csv():
    loader = UniversalDocumentLoader(deduplicate=False)
    csv_bytes = b"title,content\nAlgorithm,Binary search O(log N)\nSorting,Quicksort average O(N log N)"
    docs = loader.load_from_bytes(csv_bytes, filename="data.csv")
    assert len(docs) == 2
    assert "Binary search" in docs[0].text
    assert "Quicksort" in docs[1].text

