"""File upload and document ingestion security validation."""

from __future__ import annotations
import hashlib
import os
import re
from pathlib import Path
from typing import Optional, Set, Tuple

DEFAULT_ALLOWED_EXTENSIONS: Set[str] = {".pdf", ".txt", ".md", ".json", ".csv"}
DEFAULT_MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB


class DocumentValidationError(Exception):
    """Raised when document content or upload fails validation."""
    pass


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and injection attacks."""
    basename = os.path.basename(filename)
    # Remove any directory separators and control characters
    sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', basename)
    # Ensure it's not empty and does not begin with a dot
    sanitized = sanitized.lstrip('.')
    if not sanitized:
        sanitized = "unnamed_document.txt"
    return sanitized


def validate_file_path(path: Path | str, base_dir: Path | str) -> Path:
    """Ensure that the resolved file path resides strictly within the intended base directory."""
    base_resolved = Path(base_dir).resolve()
    target_resolved = Path(path).resolve()
    try:
        target_resolved.relative_to(base_resolved)
    except ValueError:
        raise DocumentValidationError(f"Path traversal detected: {path} is outside {base_dir}")
    return target_resolved


def validate_file_metadata(
    filename: str,
    size_bytes: int,
    allowed_extensions: Optional[Set[str]] = None,
    max_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES
) -> Tuple[bool, str]:
    """Validate file extension and size constraints."""
    ext = os.path.splitext(filename)[1].lower()
    allowed = allowed_extensions or DEFAULT_ALLOWED_EXTENSIONS
    if ext not in allowed:
        return False, f"Unsupported file extension '{ext}'. Allowed extensions: {sorted(list(allowed))}"

    if size_bytes <= 0:
        return False, "File is empty (0 bytes)."

    if size_bytes > max_size_bytes:
        return False, f"File size {size_bytes / (1024 * 1024):.2f}MB exceeds limit {max_size_bytes / (1024 * 1024):.2f}MB."

    return True, "Valid"


def compute_content_hash(content: bytes) -> str:
    """Compute SHA-256 hash of document binary content for deduplication."""
    return hashlib.sha256(content).hexdigest()


def detect_and_decode_text(raw_bytes: bytes) -> str:
    """Safely decode raw bytes into UTF-8 text with encoding fallbacks."""
    if b'\x00' in raw_bytes[:1024]:
        # Likely a non-text binary file (unless handled by specialized parser like PDF)
        raise DocumentValidationError("Binary or null bytes detected in plain text input.")

    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    for enc in encodings:
        try:
            decoded = raw_bytes.decode(enc)
            # Normalize line endings
            return decoded.replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            continue

    raise DocumentValidationError("Failed to decode document bytes using standard text encodings.")

