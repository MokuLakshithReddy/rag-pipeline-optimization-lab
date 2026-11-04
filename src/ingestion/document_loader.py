"""Universal document loader supporting multiple formats with metadata preservation."""

from __future__ import annotations
import csv
import io
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set
import pypdf

from src.ingestion.models import Document, DocumentMetadata
from src.ingestion.validation import (
    DocumentValidationError,
    compute_content_hash,
    detect_and_decode_text,
    sanitize_filename,
    validate_file_metadata,
)


class UniversalDocumentLoader:
    """Robust document loader for PDFs, Markdown, Text, JSON, and CSV."""

    def __init__(self, deduplicate: bool = True):
        self.deduplicate = deduplicate
        self._seen_hashes: Set[str] = set()

    def load_from_file(self, file_path: str | Path, source_override: Optional[str] = None) -> List[Document]:
        """Load and parse a document from disk."""
        path = Path(file_path)
        if not path.is_file():
            raise DocumentValidationError(f"File not found: {file_path}")

        filename = source_override or path.name
        content_bytes = path.read_bytes()
        return self.load_from_bytes(content_bytes, filename=filename)

    def load_from_bytes(self, content_bytes: bytes, filename: str) -> List[Document]:
        """Load and parse documents from raw bytes."""
        is_valid, msg = validate_file_metadata(filename, len(content_bytes))
        if not is_valid:
            raise DocumentValidationError(msg)

        content_hash = compute_content_hash(content_bytes)
        if self.deduplicate and content_hash in self._seen_hashes:
            # Duplicate document encountered
            return []

        self._seen_hashes.add(content_hash)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            return self._parse_pdf(content_bytes, filename, content_hash)
        elif ext in {".txt", ".md"}:
            return self._parse_text(content_bytes, filename, content_hash, ext)
        elif ext == ".json":
            return self._parse_json(content_bytes, filename, content_hash)
        elif ext == ".csv":
            return self._parse_csv(content_bytes, filename, content_hash)
        else:
            raise DocumentValidationError(f"Unsupported file format: {ext}")

    def _parse_pdf(self, content_bytes: bytes, filename: str, content_hash: str) -> List[Document]:
        """Parse PDF document preserving page numbers."""
        docs: List[Document] = []
        try:
            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
        except Exception as e:
            raise DocumentValidationError(f"Failed to parse PDF {filename}: {str(e)}")

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise DocumentValidationError(f"PDF {filename} contains no pages.")

        base_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}_{content_hash}"))

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            cleaned_text = page_text.strip()
            if not cleaned_text:
                continue

            metadata = DocumentMetadata(
                source=filename,
                doc_id=f"{base_id}_p{page_idx}",
                file_type=".pdf",
                page_number=page_idx,
                character_count=len(cleaned_text),
                token_estimate=len(cleaned_text) // 4,
                hash=content_hash,
                extra={"total_pages": total_pages, "page": page_idx}
            )
            docs.append(Document(
                id=f"{base_id}_p{page_idx}",
                text=cleaned_text,
                metadata=metadata
            ))

        return docs

    def _parse_text(self, content_bytes: bytes, filename: str, content_hash: str, ext: str) -> List[Document]:
        """Parse plain text or markdown document."""
        text = detect_and_decode_text(content_bytes).strip()
        if not text:
            return []

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}_{content_hash}"))
        metadata = DocumentMetadata(
            source=filename,
            doc_id=doc_id,
            file_type=ext,
            character_count=len(text),
            token_estimate=len(text) // 4,
            hash=content_hash
        )
        return [Document(id=doc_id, text=text, metadata=metadata)]

    def _parse_json(self, content_bytes: bytes, filename: str, content_hash: str) -> List[Document]:
        """Parse JSON document (single object or list of records)."""
        raw_text = detect_and_decode_text(content_bytes)
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise DocumentValidationError(f"Invalid JSON in {filename}: {str(e)}")

        docs: List[Document] = []
        base_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}_{content_hash}"))

        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    # Look for text or content keys
                    text = item.get("text") or item.get("content") or item.get("body") or json.dumps(item)
                    text = str(text).strip()
                    if not text:
                        continue
                    item_id = str(item.get("id", f"{base_id}_{idx}"))
                    meta = DocumentMetadata(
                        source=filename,
                        doc_id=item_id,
                        file_type=".json",
                        character_count=len(text),
                        token_estimate=len(text) // 4,
                        hash=content_hash,
                        extra={"index": idx}
                    )
                    docs.append(Document(id=item_id, text=text, metadata=meta))
                else:
                    text = str(item).strip()
                    if text:
                        item_id = f"{base_id}_{idx}"
                        meta = DocumentMetadata(
                            source=filename,
                            doc_id=item_id,
                            file_type=".json",
                            character_count=len(text),
                            token_estimate=len(text) // 4,
                            hash=content_hash
                        )
                        docs.append(Document(id=item_id, text=text, metadata=meta))
        elif isinstance(data, dict):
            text = data.get("text") or data.get("content") or json.dumps(data)
            text = str(text).strip()
            if text:
                meta = DocumentMetadata(
                    source=filename,
                    doc_id=base_id,
                    file_type=".json",
                    character_count=len(text),
                    token_estimate=len(text) // 4,
                    hash=content_hash
                )
                docs.append(Document(id=base_id, text=text, metadata=meta))

        return docs

    def _parse_csv(self, content_bytes: bytes, filename: str, content_hash: str) -> List[Document]:
        """Parse CSV documents into rows with column headers preserved."""
        raw_text = detect_and_decode_text(content_bytes)
        reader = csv.DictReader(io.StringIO(raw_text))
        docs: List[Document] = []
        base_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}_{content_hash}"))

        for idx, row in enumerate(reader):
            # Combine columns into structured textual representation
            formatted_parts = [f"{k}: {v}" for k, v in row.items() if v]
            text = "\n".join(formatted_parts).strip()
            if not text:
                continue

            row_id = f"{base_id}_row_{idx}"
            metadata = DocumentMetadata(
                source=filename,
                doc_id=row_id,
                file_type=".csv",
                character_count=len(text),
                token_estimate=len(text) // 4,
                hash=content_hash,
                extra={"row_index": idx}
            )
            docs.append(Document(id=row_id, text=text, metadata=metadata))

        return docs

